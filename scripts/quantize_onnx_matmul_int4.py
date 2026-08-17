#!/usr/bin/env python3
"""Selectively quantize ONNX MatMul weights to symmetric block INT4."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import tempfile

import onnx
import onnxruntime
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", nargs="+", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        stage = Path(temp_name)
        graphs: dict[str, object] = {}
        for graph_name in args.graphs:
            source_path = source / graph_name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            destination = stage / graph_name
            quantizer = MatMulNBitsQuantizer(str(source_path), algo_config=config)
            quantizer.process()
            quantizer.model.save_model_to_file(str(destination), use_external_data_format=True)
            onnx.checker.check_model(str(destination))
            model = onnx.load_model(str(destination), load_external_data=False)
            counts = Counter(node.op_type for node in model.graph.node)
            graphs[graph_name] = {
                "operators": dict(sorted(counts.items())),
                "quantized_matmuls": counts["MatMulNBits"],
                "remaining_matmuls": counts["MatMul"],
            }
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        receipt = {
            "schema_version": "gooya.native.selective-matmul-q4/v1",
            "source": str(source),
            "onnxruntime": onnxruntime.__version__,
            "quantization": {
                "bits": 4,
                "block_size": args.block_size,
                "symmetric": True,
                "format": "QOperator MatMulNBits",
                "scope": "MatMul weights only; convolution and all other weights remain float",
            },
            "graphs": graphs,
            "files": files,
            "validation_status": "UNVALIDATED_CANDIDATE",
        }
        (stage / "quantization-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        Path(temp_name).replace(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
