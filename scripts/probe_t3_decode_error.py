#!/usr/bin/env python3
"""Measure T3 decode-step logit divergence between FP32 and a Q4 variant.

Two measurements per candidate recipe:

- ``fixed_cache_step``: both models decode from the FP32 cache (identical input
  context). This isolates pure weight-quantization error in a single decode step.
- ``own_cache_greedy``: each model prefills with its own weights and then runs
  greedy CFG decode feeding its own growing cache. This measures the
  end-to-end autoregressive token divergence the full comparison uses.
"""

from __future__ import annotations

import argparse
import gc
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

from compare_t3_generation import generate, next_greedy
from export_t3_onnx import cache_names


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 8
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def run_prefill(directory: Path, text_ids: np.ndarray) -> list[np.ndarray]:
    sess = session(directory / "t3-prefill.onnx")
    outputs = sess.run(None, {"text_token_ids": text_ids})
    del sess
    return outputs


def run_decode_one(
    sess: ort.InferenceSession, token: int, position: int, past: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    past_length = past[0].shape[2]
    outputs = sess.run(
        None,
        {
            "next_token_id": np.array([[token]], dtype=np.int64),
            "speech_position": np.array([position], dtype=np.int64),
            "cache_position": np.array([past_length], dtype=np.int64),
            **dict(zip(cache_names("past"), past, strict=True)),
        },
    )
    return outputs[0], outputs[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    args = parser.parse_args()

    from probe_t3_sensitivity import parse_exclusions

    fp32_dir = args.fp32.resolve()
    excluded = parse_exclusions(args.exclude)
    smoke_ids = json.loads((fp32_dir / "export-receipt.json").read_text(encoding="utf-8"))[
        "smoke"
    ]["token_ids"]
    text_ids = np.array([smoke_ids], dtype=np.int64)

    config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=args.bits,
    )
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory(prefix="t3-decode-probe-") as temp_name:
        stage = Path(temp_name)
        for graph_name in ("t3-prefill.onnx", "t3-decode.onnx"):
            quantizer = MatMulNBitsQuantizer(
                str(fp32_dir / graph_name),
                algo_config=config,
                nodes_to_exclude=excluded or None,
            )
            quantizer.process()
            quantizer.model.save_model_to_file(
                str(stage / graph_name), use_external_data_format=True
            )
            onnx.checker.check_model(str(stage / graph_name))

        fp32_prefill = run_prefill(fp32_dir, text_ids)

        # fixed_cache: self-consistent zero-cache decode step as a sanity anchor.
        token0 = next_greedy(fp32_prefill[0], args.cfg_weight)
        fp32_decode = session(fp32_dir / "t3-decode.onnx")
        fp32_logits, _ = run_decode_one(fp32_decode, token0, 1, fp32_prefill[1:])
        del fp32_decode, fp32_prefill
        gc.collect()

        q4_decode = session(stage / "t3-decode.onnx")
        q4_prefill = run_prefill(stage, text_ids)
        q4_logits, _ = run_decode_one(q4_decode, token0, 1, q4_prefill[1:])
        del q4_decode, q4_prefill
        gc.collect()

        delta = np.abs(fp32_logits[0, 0].astype(np.float32) - q4_logits[0, 0].astype(np.float32))
        fixed_cache_step = {
            "max_abs_error": float(delta.max(initial=0.0)),
            "mean_abs_error": float(delta.mean()),
            "fp32_top": int(np.argmax(fp32_logits[0, 0])),
            "q4_top": int(np.argmax(q4_logits[0, 0])),
            "top_match": bool(
                int(np.argmax(fp32_logits[0, 0])) == int(np.argmax(q4_logits[0, 0]))
            ),
        }

        fp32_tokens = generate(fp32_dir, smoke_ids, args.steps, args.cfg_weight)
        q4_tokens = generate(stage, smoke_ids, args.steps, args.cfg_weight)
        prefix = 0
        for expected, actual in zip(fp32_tokens, q4_tokens, strict=False):
            if expected != actual:
                break
            prefix += 1
        own_cache_greedy = {
            "fp32_tokens": fp32_tokens,
            "q4_tokens": q4_tokens,
            "matching_prefix_tokens": prefix,
            "exact_match": fp32_tokens == q4_tokens,
        }

        receipt = {
            "schema_version": "gooya.native.t3-decode-probe/v1",
            "excluded": excluded,
            "excluded_count": len(excluded),
            "block_size": args.block_size,
            "bits": args.bits,
            "steps": args.steps,
            "cfg_weight": args.cfg_weight,
            "fixed_cache_step": fixed_cache_step,
            "own_cache_greedy": own_cache_greedy,
        }
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())