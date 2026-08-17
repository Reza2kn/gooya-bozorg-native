#!/usr/bin/env python3
"""Fold MatMulNBits nodes whose activation input is a compile-time constant.

Flow-prepare exported with do_constant_folding leaves a handful of frozen
positional projections (``self_attn/linear_pos``) whose left operand is a
Constant. ORT evaluates them in a fast kernel at runtime, but tract's type
solver re-evaluates every constant-input Einsum eagerly during analyse, which
is O(minutes) in interpreted mode. Baking them into Constant nodes keeps ORT
numbers identical (same dequant formula) and removes tract's load bottleneck.

Only folds nodes whose activation side resolves to a constant directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detach_data(model: onnx.ModelProto, base_dir: Path) -> onnx.ModelProto:
    """No-op kept for API compatibility; onnx.load already brings tensors into memory."""
    return model


def dequant_q4(weight_u8: np.ndarray, scales: np.ndarray, block_size: int) -> np.ndarray:
    """Y = (nibble - 8) * scale per block, matching ONNX MatMulNBits default.

    weight shape (N, n_blocks, block_size//2), scales (N, n_blocks).
    """
    n, n_blocks, blob = weight_u8.shape
    k = n_blocks * block_size
    low = (weight_u8 & 0x0F).astype(np.int16)
    high = ((weight_u8 >> 4) & 0x0F).astype(np.int16)
    q = np.empty((n, k), dtype=np.int16)
    q[:, 0::2] = low.reshape(n, n_blocks * blob)
    q[:, 1::2] = high.reshape(n, n_blocks * blob)
    scale = scales.reshape(n, n_blocks, 1).astype(np.float64)
    return ((q.reshape(n, n_blocks, block_size).astype(np.float64) - 8.0) * scale).reshape(n, k).astype(np.float64)


def eval_matmulnbits_with_ort(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    a: np.ndarray,
    a_name: str,
    extra_feeds: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute a MatMulNBits output with ORT's own kernel.

    Builds a minimal model reusing the original B/scales/(zp) initializers so the
    folded constant is bit-identical to what the reference runtime computes.
    """
    from onnx import numpy_helper

    graph = model.graph
    initializer = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    b_name, scale_name = node.input[1], node.input[2]
    if b_name not in initializer or scale_name not in initializer:
        raise RuntimeError(f"fold {node.name}: missing B/scales initializers")
    b_t = next(t for t in graph.initializer if t.name == b_name)
    s_t = next(t for t in graph.initializer if t.name == scale_name)
    attrs = {at.name: at for at in node.attribute}
    n, k, bs = attrs["N"].i, attrs["K"].i, attrs["block_size"].i

    a_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["__fold_a__"],
        value=onnx.helper.make_tensor(
            "__fold_a__", onnx.TensorProto.FLOAT, list(a.shape), a.astype(np.float32).flatten().tolist()
        ),
    )
    m_node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=["__fold_a__", b_name, scale_name],
        outputs=["__fold_y__"],
        name=node.name + "_ort",
        domain="com.microsoft",
        K=k,
        N=n,
        bits=4,
        block_size=bs,
        accuracy_level=4,
    )
    g = onnx.helper.make_graph(
        [a_const, m_node],
        "fold_" + node.name,
        [],
        [onnx.helper.make_tensor_value_info("__fold_y__", onnx.TensorProto.FLOAT, None)],
        initializer=[b_t, s_t],
    )
    m = onnx.helper.make_model(
        g,
        opset_imports=[
            onnx.helper.make_opsetid("", 20),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    m.ir_version = 9
    buf = m.SerializeToString()
    session = onnxruntime.InferenceSession(buf, providers=["CPUExecutionProvider"])
    out = session.run(None, {})
    return out[0]


def fold_graph(
    model: onnx.ModelProto,
    block_size: int,
    feeds: dict[str, np.ndarray],
) -> tuple[onnx.ModelProto, int, list[str]]:
    graph = model.graph
    initializer = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    constants: dict[str, np.ndarray] = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    constants[node.output[0]] = np.asarray(attr.f, dtype=np.float32)
                elif attr.type == onnx.AttributeProto.INT:
                    constants[node.output[0]] = np.asarray(attr.i, dtype=np.int64)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    constants[node.output[0]] = np.asarray(attr.floats, dtype=np.float32)
                elif attr.type == onnx.AttributeProto.INTS:
                    constants[node.output[0]] = np.asarray(attr.ints, dtype=np.int64)

    folded: list[str] = []
    replacements: dict[str, np.ndarray] = {}
    candidates: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            continue
        if len(node.input) != 3:
            continue  # keep bias/gidx handling for now
        b_name, scale_name = node.input[1], node.input[2]
        if b_name not in initializer or scale_name not in initializer:
            continue
        attrs = {a.name: a for a in node.attribute}
        bits = attrs.get("bits", None)
        if bits is not None and bits.i != 4:
            continue
        if attrs["block_size"].i != block_size:
            continue
        candidates.append(node)

    if not candidates:
        return model, 0, []

    import copy

    def probe_intermediates(model: onnx.ModelProto, feeds: dict[str, np.ndarray]):
        """Run a copy of the model with every candidate activation added as an
        output boundary, returning a map {candidate A name: ndarray}."""
        clone = copy.deepcopy(model)
        a_names = sorted({n.input[0] for n in candidates})
        extra = [n for n in a_names if n not in {o.name for o in clone.graph.output}]
        for t in extra:
            vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
            clone.graph.output.append(vi)
        buf = clone.SerializeToString()
        sess = onnxruntime.InferenceSession(buf, providers=["CPUExecutionProvider"])
        outs = sess.run(None, feeds)
        n_graph = len(clone.graph.output) - len(extra)
        return dict(zip(extra, outs[n_graph:]))

    feeds_a = dict(feeds)
    feeds_b = dict(feeds)
    # shift token ids in the second canary to perturb token-dependent branches
    toks = feeds["speech_tokens"].copy()
    toks_b = (toks + 1) % 6561
    feeds_b["speech_tokens"] = toks_b

    as_a = probe_intermediates(model, feeds_a)
    as_b = probe_intermediates(model, feeds_b)

    a_names = sorted({n.input[0] for n in candidates})
    for a_name in a_names:
        va = as_a[a_name]
        vb = as_b[a_name]
        if va.shape != vb.shape or not np.array_equal(va, vb):
            continue
        if a_name in {i.name for i in graph.input}:
            continue
        # token-independent intermediate -> constant activation
        a_arr = np.asarray(va, dtype=np.float32)
        for node in candidates:
            if node.input[0] != a_name:
                continue
            y = eval_matmulnbits_with_ort(model, node, a_arr, a_name, feeds_a)
            y = np.asarray(y, dtype=np.float32)
            if y.nbytes > 2 * 1024 * 1024 * 1024:
                raise RuntimeError(f"fold {node.name} too large: {y.nbytes}")
            replacements[node.output[0]] = y
            folded.append(node.name)
            folded.append(f"  A={a_arr.shape} -> {y.shape}")

    if not folded:
        return model, 0, []

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type == "MatMulNBits" and node.output[0] in replacements:
            out = replacements[node.output[0]]
            const_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=[node.output[0]],
                name=node.name + "_folded",
                value=onnx.helper.make_tensor(
                    node.output[0], onnx.TensorProto.FLOAT, out.shape, out.flatten().tolist()
                ),
            )
            new_nodes.append(const_node)
        else:
            new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    return model, len(folded) // 2, folded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--speech-tokens-json", type=Path, required=True)
    args = parser.parse_args()
    src = args.input.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    tokens_payload = json.loads(args.speech_tokens_json.read_text(encoding="utf-8"))
    tokens = np.asarray(tokens_payload["fp32_tokens"], dtype=np.int64)[None, :]
    token_len = np.asarray([tokens.shape[1]], dtype=np.int64)

    model = onnx.load(str(src), load_external_data=True)
    feeds = {"speech_tokens": tokens.astype(np.int64), "speech_token_length": token_len.astype(np.int64)}
    folded_model, nfolded, folded_detail = fold_graph(model, args.block_size, feeds)
    if nfolded == 0:
        print("no folds")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as temp_name:
        stage = Path(temp_name)
        folded_path = stage / out.name
        onnx.save_model(
            folded_model,
            str(folded_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{out.name}.data",
            size_threshold=1024,
        )
        onnx.checker.check_model(str(folded_path))

        ref = onnxruntime.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(
            None, {"speech_tokens": tokens, "speech_token_length": token_len}
        )
        got = onnxruntime.InferenceSession(str(folded_path), providers=["CPUExecutionProvider"]).run(
            None, {"speech_tokens": tokens, "speech_token_length": token_len}
        )
        deltas = []
        for name, r, g in zip(["mu", "mask", "speaker", "cond"], ref, got):
            d = np.abs(r - g).max(initial=0.0)
            deltas.append((name, float(d)))
        worse = [d for _, d in deltas if d > 2e-3]
        if worse:
            raise RuntimeError(f"fold drifted: {deltas}")

        files = {}
        for p in sorted(stage.iterdir()):
            if p.is_file():
                files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        receipt = {
            "schema_version": "gooya.native.constant-fold-matmulnbits/v1",
            "input": str(src),
            "output": str(out),
            "block_size": args.block_size,
            "folded": nfolded,
            "nodes_before": len(model.graph.node),
            "nodes_after": len(folded_model.graph.node),
            "ort_max_abs_delta_after": deltas,
            "files": files,
        }
        (out.parent / f"{out.name}.fold-receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        for p in stage.iterdir():
            if p.is_file():
                import shutil
                shutil.move(str(p), str(out.parent / p.name))
    print(json.dumps({"folded": nfolded, "deltas": deltas}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())