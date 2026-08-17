#!/usr/bin/env python3
"""Compare exported T3 ONNX prefill/decode graphs with the pinned PyTorch source."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file
import torch

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config

from export_t3_onnx import Decode, Prefill, cache_names


def compare(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | list[int]]:
    delta = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(delta.max(initial=0.0)),
        "mean_abs_error": float(delta.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixed-voice", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path)
    parser.add_argument("--max-logit-error", type=float, default=2e-3)
    parser.add_argument("--max-cache-error", type=float, default=2e-3)
    args = parser.parse_args()

    source = args.source.resolve()
    fixed_voice = args.fixed_voice.resolve()
    graphs = args.onnx.resolve()
    export_receipt_path = (
        args.export_receipt.resolve() if args.export_receipt else graphs / "export-receipt.json"
    )
    export_receipt = json.loads(export_receipt_path.read_text(encoding="utf-8"))
    smoke_ids = export_receipt["smoke"]["token_ids"]
    text_ids = torch.tensor([smoke_ids], dtype=torch.long)

    with tempfile.TemporaryDirectory(prefix="gooya-t3-validate-") as temp_dir:
        temp = Path(temp_dir)
        t3 = T3(T3Config.multilingual()).eval()
        t3.load_state_dict(load_file(str(source / "t3_fa.safetensors")))
        t3.tfmr.set_attn_implementation("eager")
        conditioning = load_file(str(fixed_voice / "canonical-t3-conditioning.safetensors"))[
            "conditioning"
        ]
        prefill = Prefill(t3, conditioning).eval()
        decode = Decode(t3).eval()
        with torch.inference_mode():
            expected_prefill = tuple(value.detach().cpu().numpy() for value in prefill(text_ids))
            next_id = np.argsort(expected_prefill[0][0, 0])[-1:].astype(np.int64).reshape(1, 1)
            past_length = expected_prefill[1].shape[2]
            expected_decode = tuple(
                value.detach().cpu().numpy()
                for value in decode(
                    torch.from_numpy(next_id),
                    torch.tensor([1], dtype=torch.long),
                    torch.tensor([past_length], dtype=torch.long),
                    *(torch.from_numpy(value) for value in expected_prefill[1:]),
                )
            )
        np.savez(temp / "expected-prefill.npz", *expected_prefill)
        np.savez(temp / "expected-decode.npz", *expected_decode)
        del prefill, decode, t3, conditioning, expected_prefill, expected_decode
        gc.collect()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(8, (ort.get_available_providers() and 8)))
        providers = ["CPUExecutionProvider"]

        prefill_session = ort.InferenceSession(
            str(graphs / "t3-prefill.onnx"), sess_options=options, providers=providers
        )
        actual_prefill = prefill_session.run(None, {"text_token_ids": text_ids.numpy()})
        with np.load(temp / "expected-prefill.npz") as archive:
            expected_prefill = [archive[f"arr_{index}"] for index in range(len(archive.files))]
            prefill_logits = compare(expected_prefill[0], actual_prefill[0])
            prefill_top = {
                "pytorch": int(np.argmax(expected_prefill[0][0, 0])),
                "onnx": int(np.argmax(actual_prefill[0][0, 0])),
            }
            prefill_caches = [
                compare(expected, actual)
                for expected, actual in zip(expected_prefill[1:], actual_prefill[1:], strict=True)
            ]
            next_id = np.argsort(expected_prefill[0][0, 0])[-1:].astype(np.int64).reshape(1, 1)
            past_length = expected_prefill[1].shape[2]
            decode_inputs = {
                "next_token_id": next_id,
                "speech_position": np.array([1], dtype=np.int64),
                "cache_position": np.array([past_length], dtype=np.int64),
                **dict(zip(cache_names("past"), expected_prefill[1:], strict=True)),
            }
        del prefill_session, actual_prefill, expected_prefill
        gc.collect()

        decode_session = ort.InferenceSession(
            str(graphs / "t3-decode.onnx"), sess_options=options, providers=providers
        )
        actual_decode = decode_session.run(None, decode_inputs)
        with np.load(temp / "expected-decode.npz") as archive:
            expected_decode = [archive[f"arr_{index}"] for index in range(len(archive.files))]
            decode_logits = compare(expected_decode[0], actual_decode[0])
            decode_top = {
                "pytorch": int(np.argmax(expected_decode[0][0, 0])),
                "onnx": int(np.argmax(actual_decode[0][0, 0])),
            }
            decode_caches = [
                compare(expected, actual)
                for expected, actual in zip(expected_decode[1:], actual_decode[1:], strict=True)
            ]

    receipt = {
        "schema_version": "gooya.native.t3-onnx-validation/v1",
        "onnxruntime": ort.__version__,
        "provider": providers[0],
        "prefill": {
            "logits": prefill_logits,
            "top_token_id": prefill_top,
            "cache_max_abs_error": max(item["max_abs_error"] for item in prefill_caches),
            "cache_mean_abs_error": float(np.mean([item["mean_abs_error"] for item in prefill_caches])),
        },
        "decode": {
            "logits": decode_logits,
            "top_token_id": decode_top,
            "cache_max_abs_error": max(item["max_abs_error"] for item in decode_caches),
            "cache_mean_abs_error": float(np.mean([item["mean_abs_error"] for item in decode_caches])),
        },
    }
    passed = (
        prefill_logits["max_abs_error"] <= args.max_logit_error
        and decode_logits["max_abs_error"] <= args.max_logit_error
        and receipt["prefill"]["cache_max_abs_error"] <= args.max_cache_error
        and receipt["decode"]["cache_max_abs_error"] <= args.max_cache_error
        and prefill_top["pytorch"] == prefill_top["onnx"]
        and decode_top["pytorch"] == decode_top["onnx"]
    )
    receipt["status"] = "PASS" if passed else "FAIL"
    output = graphs / "onnxruntime-validation.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
