#!/usr/bin/env python3
"""Convert the fixed-voice T3 prefill graph to Core ML W4A16."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)
import numpy as np
from safetensors.torch import load_file
import torch

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config

from export_t3_onnx import Prefill


class CoreMLPrefill(Prefill):
    def forward(self, text_token_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return super().forward(text_token_ids.to(torch.long))


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixed-voice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    args = parser.parse_args()
    source = args.source.resolve()
    fixed_voice = args.fixed_voice.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    t3 = T3(T3Config.multilingual()).eval()
    t3.load_state_dict(load_file(str(source / "t3_fa.safetensors")))
    t3.tfmr.set_attn_implementation("eager")
    conditioning = load_file(str(fixed_voice / "canonical-t3-conditioning.safetensors"))[
        "conditioning"
    ]
    model = CoreMLPrefill(t3, conditioning).eval()
    smoke_ids = json.loads(
        (fixed_voice / "fixed-voice-receipt.json").read_text(encoding="utf-8")
    )["tokenizer_smoke"]["token_ids"]
    example = torch.tensor([smoke_ids], dtype=torch.int32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False, check_trace=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        fp16_path = temp / "T3PrefillFP16.mlpackage"
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            inputs=[
                ct.TensorType(
                    name="text_token_ids",
                    shape=(1, ct.RangeDim(lower_bound=1, upper_bound=args.max_text_tokens)),
                    dtype=np.int32,
                )
            ],
        )
        mlmodel.save(str(fp16_path))
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int4",
                granularity="per_block",
                block_size=32,
                weight_threshold=2048,
            )
        )
        quantized = linear_quantize_weights(mlmodel, config=config)
        quantized.save(str(output))

    receipt = {
        "schema_version": "gooya.native.t3-coreml-prefill/v1",
        "coremltools": ct.__version__,
        "minimum_deployment_target": "iOS18/macOS15",
        "compute_precision": "W4A16",
        "quantization": "linear_symmetric per_block block_size=32",
        "text_token_range": [1, args.max_text_tokens],
        "package_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "tree_sha256": tree_hash(output),
        "validation_status": "CONVERTED_NOT_PREDICTION_VALIDATED",
    }
    receipt_path = output.parent / f"{output.name}-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
