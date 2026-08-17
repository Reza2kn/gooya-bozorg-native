#!/usr/bin/env python3
"""Quantize T3 MatMul weights to tract-compatible symmetric Q4 MatMulNBits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import tempfile

import onnx
from onnx import TensorProto
import onnxruntime
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    HQQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat


GRAPH_NAMES = ("t3-prefill.onnx", "t3-decode.onnx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize(
    source: Path,
    destination: Path,
    block_size: int,
    algorithm: str,
    keep_speech_head_fp32: bool,
    keep_edge_layers_fp32: int,
    channel_wise: bool,
) -> dict[str, int]:
    if algorithm == "hqq":
        config = HQQWeightOnlyQuantConfig(
            block_size=block_size,
            bits=4,
            axis=1,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
    else:
        config = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
            bits=4,
            channel_wised_quantize=channel_wise,
        )
    excluded = ["/speech_head/MatMul"] if keep_speech_head_fp32 else []
    for layer in list(range(keep_edge_layers_fp32)) + list(
        range(30 - keep_edge_layers_fp32, 30)
    ):
        prefix = f"/tfmr/layers.{layer}/"
        excluded.extend(
            [
                f"{prefix}self_attn/{projection}/MatMul"
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            ]
            + [
                f"{prefix}mlp/{projection}/MatMul"
                for projection in ("gate_proj", "up_proj", "down_proj")
            ]
        )
    quantizer = MatMulNBitsQuantizer(
        str(source),
        algo_config=config,
        nodes_to_exclude=excluded or None,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(destination), use_external_data_format=True)
    onnx.checker.check_model(str(destination))
    model = onnx.load_model(str(destination), load_external_data=False)
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def merge_shared_weights(directory: Path, sidecar_name: str) -> dict[str, int]:
    """Repoint identical external initializers in both graphs at one sidecar."""
    shared_path = directory / sidecar_name
    locations_to_remove: set[Path] = set()
    blobs: dict[str, tuple[int, int]] = {}
    stats = {"unique_blobs": 0, "reused_blobs": 0, "bytes": 0}
    with shared_path.open("wb") as shared:
        for graph_name in GRAPH_NAMES:
            graph_path = directory / graph_name
            metadata = onnx.load_model(str(graph_path), load_external_data=False)
            populated = onnx.load_model(str(graph_path), load_external_data=True)
            for target, source in zip(
                metadata.graph.initializer, populated.graph.initializer, strict=True
            ):
                if target.data_location != TensorProto.EXTERNAL:
                    continue
                fields = {item.key: item.value for item in target.external_data}
                locations_to_remove.add(directory / fields["location"])
                raw = bytes(source.raw_data)
                digest = hashlib.sha256(raw).hexdigest()
                if digest in blobs:
                    offset, length = blobs[digest]
                    stats["reused_blobs"] += 1
                else:
                    offset, length = shared.tell(), len(raw)
                    shared.write(raw)
                    blobs[digest] = (offset, length)
                    stats["unique_blobs"] += 1
                    stats["bytes"] += length
                target.ClearField("raw_data")
                target.data_location = TensorProto.EXTERNAL
                del target.external_data[:]
                for key, value in (
                    ("location", sidecar_name),
                    ("offset", str(offset)),
                    ("length", str(length)),
                ):
                    item = target.external_data.add()
                    item.key = key
                    item.value = value
            graph_path.write_bytes(metadata.SerializeToString())
            onnx.checker.check_model(str(graph_path))
    for path in locations_to_remove:
        if path != shared_path and path.exists():
            path.unlink()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--algorithm", choices=("symmetric-rtn", "hqq"), default="symmetric-rtn")
    parser.add_argument("--keep-speech-head-fp32", action="store_true")
    parser.add_argument("--keep-edge-layers-fp32", type=int, default=0, choices=range(0, 6))
    parser.add_argument(
        "--channel-wise",
        action="store_true",
        help="Use one symmetric scale per output channel instead of block-32 scales.",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for graph_name in GRAPH_NAMES:
        if not (source / graph_name).is_file():
            raise FileNotFoundError(source / graph_name)

    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        stage = Path(temp_name)
        operators: dict[str, dict[str, int]] = {}
        for graph_name in GRAPH_NAMES:
            print(f"quantizing {graph_name}", flush=True)
            operators[graph_name] = quantize(
                source / graph_name,
                stage / graph_name,
                args.block_size,
                args.algorithm,
                args.keep_speech_head_fp32,
                args.keep_edge_layers_fp32,
                args.channel_wise,
            )
        deduplication = merge_shared_weights(stage, "t3-q4-shared.data")
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        receipt = {
            "schema_version": "gooya.native.t3-q4/v1",
            "source_export": str(source),
            "onnxruntime": onnxruntime.__version__,
            "quantization": {
                "bits": 4,
                "block_size": args.block_size,
                "symmetric": args.algorithm == "symmetric-rtn",
                "accuracy_level": 4 if args.algorithm == "symmetric-rtn" else None,
                "format": "QOperator MatMulNBits",
                "algorithm": args.algorithm,
                "speech_head": "fp32" if args.keep_speech_head_fp32 else "int4",
                "fp32_edge_layers_per_side": args.keep_edge_layers_fp32,
                "channel_wise": args.channel_wise,
            },
            "operator_counts": operators,
            "deduplication": deduplication,
            "validation_status": "UNVALIDATED_CANDIDATE",
            "files": files,
        }
        (stage / "quantization-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        Path(temp_name).replace(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
