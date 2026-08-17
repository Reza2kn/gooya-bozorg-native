#!/usr/bin/env python3
"""Write the FP32 T3 prefill logits reference used by the sensitivity probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--smoke-ids", type=str, default="1473,1490,1456,1491,1434,2,1467,1456,1490,1464,2,1548,1477,1459,1471,1493,1453,9")
    args = parser.parse_args()
    source = args.source.resolve()
    smoke_ids = [int(x) for x in args.smoke_ids.split(",")]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(
        str(source / "t3-prefill.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    logits = session.run(None, {"text_token_ids": np.array([smoke_ids], dtype=np.int64)})[0]
    reference = {
        "schema_version": "gooya.native.t3-prefill-reference/v1",
        "source": str(source),
        "smoke_ids": smoke_ids,
        "logits": logits[0, 0].tolist(),
    }
    output = source.parent / f"{source.name}-prefill-reference.json"
    output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
