#!/usr/bin/env python3
"""Quantize T3 with a configurable FP32-exclusion set and measure prefill logit error.

This is a fast single-forward sensitivity probe. It reuses the QOperator
MatMulNBits quantizer, excludes the requested MatMul nodes from quantization,
and reports the prefill logits max/mean abs error plus top-1 stability against
the FP32 reference. It is deliberately lighter than the full greedy loop so we
can sweep many exclusion recipes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat


LAYERS = 30
ALL_COMPONENTS = ("attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down")
NODE_NAMES = {
    "attn_q": "self_attn/q_proj",
    "attn_k": "self_attn/k_proj",
    "attn_v": "self_attn/v_proj",
    "attn_o": "self_attn/o_proj",
    "mlp_gate": "mlp/gate_proj",
    "mlp_up": "mlp/up_proj",
    "mlp_down": "mlp/down_proj",
}


def parse_exclusions(spec: str) -> list[str]:
    """Parse a comma list: 'speech_head', 'layer:N', 'comp:attn_q', 'layerN:comp'."""
    excluded: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "speech_head":
            excluded.append("/speech_head/MatMul")
            continue
        if token.startswith("layer:"):
            layer = int(token.split(":", 1)[1])
            for comp in ALL_COMPONENTS:
                excluded.append(f"/tfmr/layers.{layer}/{NODE_NAMES[comp]}/MatMul")
            continue
        if token.startswith("comp:"):
            comp = token.split(":", 1)[1]
            if comp not in NODE_NAMES:
                raise ValueError(f"unknown component {comp}")
            for layer in range(LAYERS):
                excluded.append(f"/tfmr/layers.{layer}/{NODE_NAMES[comp]}/MatMul")
            continue
        if token in NODE_NAMES:
            for layer in range(LAYERS):
                excluded.append(f"/tfmr/layers.{layer}/{NODE_NAMES[token]}/MatMul")
            continue
        if ":" in token:
            layer, comp = token.split(":", 1)
            if not layer.startswith("L") or not comp:
                raise ValueError(f"bad node token {token}")
            layer = int(layer[1:])
            if comp not in NODE_NAMES:
                raise ValueError(f"unknown component {comp}")
            excluded.append(f"/tfmr/layers.{layer}/{NODE_NAMES[comp]}/MatMul")
            continue
        raise ValueError(f"unrecognized exclusion token {token!r}")
    return excluded


def all_weight_matmuls(source: Path) -> list[str]:
    """Names of every MatMul node that reads an initializer weight."""
    import onnx

    model = onnx.load_model(str(source / "t3-prefill.onnx"), load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}
    return [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul" and node.input[1] in initializers
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", required=True, help="comma list, e.g. speech_head,L0:attn_q")
    parser.add_argument("--invert", action="store_true", help="quantize only the listed nodes")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--smoke-ids", type=str, default="1473,1490,1456,1491,1434,2,1467,1456,1490,1464,2,1548,1477,1459,1471,1493,1453,9")
    args = parser.parse_args()

    source = args.source.resolve()
    excluded = parse_exclusions(args.exclude)
    if args.invert:
        quantize_these = excluded
        excluded = [name for name in all_weight_matmuls(source) if name not in quantize_these]
        if not quantize_these:
            raise ValueError("--invert needs at least one listed node to quantize")
    smoke_ids = [int(x) for x in args.smoke_ids.split(",")]
    ref_path = source.parent / f"{source.name}-prefill-reference.json"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"missing FP32 reference {ref_path}; run make_prefill_reference first"
        )
    reference = json.loads(ref_path.read_text(encoding="utf-8"))

    config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory(prefix="t3-probe-") as temp_name:
        stage = Path(temp_name)
        quantizer = MatMulNBitsQuantizer(
            str(source / "t3-prefill.onnx"),
            algo_config=config,
            nodes_to_exclude=excluded or None,
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(stage / "t3-prefill.onnx"), use_external_data_format=True)
        onnx.checker.check_model(str(stage / "t3-prefill.onnx"))

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 8
        session = ort.InferenceSession(
            str(stage / "t3-prefill.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            None, {"text_token_ids": np.array([smoke_ids], dtype=np.int64)}
        )[0][0, 0]
        expected = np.asarray(reference["logits"])
        delta = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
        top_expected = int(np.argmax(expected))
        top_actual = int(np.argmax(actual))
        top1_match = top_expected == top_actual
        top_delta = float(abs(float(expected[top_expected]) - float(actual[top_expected])))

        receipt = {
            "schema_version": "gooya.native.t3-sensitivity-probe/v1",
            "source": str(source),
            "excluded": excluded,
            "block_size": args.block_size,
            "excluded_count": len(excluded),
            "quantized_count": 211 - len(excluded) if not args.invert else len(quantize_these),
            "max_abs_error": float(delta.max(initial=0.0)),
            "mean_abs_error": float(delta.mean()),
            "top1": {"expected": top_expected, "actual": top_actual, "match": top1_match},
            "top1_logit_shift": top_delta,
        }
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
