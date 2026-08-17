#!/usr/bin/env python3
"""Inspect safetensors without materializing tensors and report W4 candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import struct


DTYPE_BYTES = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "F16": 2,
               "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "F64": 8, "I64": 8,
               "U64": 8}


def header(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def family(name: str) -> str:
    if name.startswith("tfmr."):
        return "t3.transformer"
    if name.startswith(("text_", "speech_")):
        return "t3.embeddings_heads"
    if name.startswith("cond_enc."):
        return "t3.conditioning_export_only"
    if name.startswith("tokenizer."):
        return "s3gen.reference_tokenizer_export_only"
    if name.startswith("speaker_encoder."):
        return "s3gen.reference_speaker_export_only"
    if name.startswith("flow."):
        return "s3gen.flow"
    if name.startswith("mel2wav."):
        return "s3gen.vocoder"
    return name.split(".", 1)[0]


def inspect(path: Path) -> dict[str, object]:
    raw = header(path)
    tensors = {name: meta for name, meta in raw.items() if name != "__metadata__"}
    dtype_counts: Counter[str] = Counter()
    families: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "parameters": 0,
                                                               "source_bytes": 0,
                                                               "w4_candidate_parameters": 0})
    parameters = 0
    source_bytes = 0
    w4_candidates = 0
    for name, meta in tensors.items():
        dtype = str(meta["dtype"])
        shape = [int(value) for value in meta["shape"]]
        count = math.prod(shape)
        nbytes = count * DTYPE_BYTES[dtype]
        group = families[family(name)]
        group["tensors"] += 1
        group["parameters"] += count
        group["source_bytes"] += nbytes
        parameters += count
        source_bytes += nbytes
        dtype_counts[dtype] += 1
        # W4 is useful only on matrix/conv-like float weights with enough values.
        is_candidate = dtype in {"F32", "F16", "BF16"} and name.endswith(".weight") and len(shape) >= 2 and count >= 4096
        if is_candidate:
            group["w4_candidate_parameters"] += count
            w4_candidates += count
    return {
        "file": path.name,
        "tensor_count": len(tensors),
        "parameters": parameters,
        "source_tensor_bytes": source_bytes,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "w4_candidate_parameters": w4_candidates,
        "estimated_w4_payload_bytes_excluding_scales": (w4_candidates + 1) // 2,
        "families": dict(sorted(families.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [inspect(args.source / name) for name in
               ("t3_fa.safetensors", "s3gen.safetensors", "ve.safetensors")]
    report = {
        "schema_version": "gooya.native.source-inspection/v1",
        "source": "Reza2kn/Gooya-Bozorg-v1.5@8844b3a6ebbefa0e1ac4baef494d2e8d7eda9d9c",
        "weights": reports,
        "totals": {
            "parameters": sum(item["parameters"] for item in reports),
            "source_tensor_bytes": sum(item["source_tensor_bytes"] for item in reports),
            "w4_candidate_parameters": sum(item["w4_candidate_parameters"] for item in reports),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
