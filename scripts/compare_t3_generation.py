#!/usr/bin/env python3
"""Compare greedy CFG speech-token generation between FP32 and Q4 T3 graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from export_t3_onnx import cache_names


def next_greedy(logits: np.ndarray, cfg_weight: float) -> int:
    cond, uncond = logits[0, -1], logits[1, -1]
    combined = cond + cfg_weight * (cond - uncond)
    return int(np.argmax(combined))


def generate(directory: Path, text_ids: list[int], steps: int, cfg_weight: float) -> list[int]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 8
    providers = ["CPUExecutionProvider"]
    prefill = ort.InferenceSession(
        str(directory / "t3-prefill.onnx"), sess_options=options, providers=providers
    )
    outputs = prefill.run(None, {"text_token_ids": np.array([text_ids], dtype=np.int64)})
    del prefill
    tokens: list[int] = []
    decode = ort.InferenceSession(
        str(directory / "t3-decode.onnx"), sess_options=options, providers=providers
    )
    for position in range(steps):
        token = next_greedy(outputs[0], cfg_weight)
        tokens.append(token)
        if token == 6562:
            break
        past = outputs[1:]
        past_length = past[0].shape[2]
        inputs = {
            "next_token_id": np.array([[token]], dtype=np.int64),
            "speech_position": np.array([position + 1], dtype=np.int64),
            "cache_position": np.array([past_length], dtype=np.int64),
            **dict(zip(cache_names("past"), past, strict=True)),
        }
        outputs = decode.run(None, inputs)
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    args = parser.parse_args()
    smoke = json.loads(args.export_receipt.read_text(encoding="utf-8"))["smoke"]
    fp32_tokens = generate(args.fp32.resolve(), smoke["token_ids"], args.steps, args.cfg_weight)
    q4_tokens = generate(args.q4.resolve(), smoke["token_ids"], args.steps, args.cfg_weight)
    prefix = 0
    for expected, actual in zip(fp32_tokens, q4_tokens, strict=False):
        if expected != actual:
            break
        prefix += 1
    overlap = sum(left == right for left, right in zip(fp32_tokens, q4_tokens, strict=False))
    receipt = {
        "schema_version": "gooya.native.t3-generation-comparison/v1",
        "steps_requested": args.steps,
        "cfg_weight": args.cfg_weight,
        "fp32_tokens": fp32_tokens,
        "q4_tokens": q4_tokens,
        "matching_prefix_tokens": prefix,
        "same_position_tokens": overlap,
        "exact_match": fp32_tokens == q4_tokens,
        "status": "CHARACTERIZED_NOT_QUALITY_ACCEPTED",
    }
    output = args.q4.resolve() / "generation-comparison.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
