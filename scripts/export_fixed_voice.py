#!/usr/bin/env python3
"""Bake the canonical voice conditioning and remove reference encoders at runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from chatterbox.models.s3gen import S3Gen
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm


SOURCE_REVISION = "8844b3a6ebbefa0e1ac4baef494d2e8d7eda9d9c"
REFERENCE_SHA256 = "2fb8f2b236fdc958bed28d497973abc5593976b164761f3316bed1e57f5edac5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_contiguous(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference = source / "samples" / "canonical.wav"
    if sha256(reference) != REFERENCE_SHA256:
        raise RuntimeError("canonical voice reference hash mismatch")

    device = torch.device(args.device)
    voice_encoder = VoiceEncoder()
    voice_encoder.load_state_dict(load_file(str(source / "ve.safetensors")))
    voice_encoder.to(device).eval()

    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_file(str(source / "t3_fa.safetensors")))
    t3.tfmr.set_attn_implementation("eager")
    t3.to(device).eval()

    s3gen = S3Gen()
    missing, unexpected = s3gen.load_state_dict(
        load_file(str(source / "s3gen.safetensors")), strict=False
    )
    if unexpected or missing not in ([], ["tokenizer.window"]):
        raise RuntimeError(f"S3Gen state mismatch: missing={missing}, unexpected={unexpected}")
    s3gen.to(device).eval()

    tokenizer = MTLTokenizer(str(source / "grapheme_mtl_merged_expanded_v1.json"))
    engine = ChatterboxMultilingualTTS(
        t3=t3,
        s3gen=s3gen,
        ve=voice_encoder,
        tokenizer=tokenizer,
        device=str(device),
    )
    engine.prepare_conditionals(str(reference), exaggeration=0.5)
    assert engine.conds is not None

    # Materialize the embedding lookup and Perceiver once. The native T3 hot
    # path begins at this fixed conditioning tensor.
    with torch.inference_mode():
        t3_conditioning = t3.prepare_conditioning(engine.conds.t3)

    t3_path = output / "canonical-t3-conditioning.safetensors"
    save_file({"conditioning": cpu_contiguous(t3_conditioning)}, str(t3_path))

    s3_tensors = {
        key: cpu_contiguous(value)
        for key, value in engine.conds.gen.items()
        if torch.is_tensor(value)
    }
    s3_path = output / "canonical-s3gen-conditioning.safetensors"
    save_file(s3_tensors, str(s3_path))

    # Freeze tokenizer behavior on the canonical smoke text. The runtime can
    # compare this vector before attempting multi-gigabyte model loading.
    smoke_text = "سلام، حالت چطوره؟"
    normalized = punc_norm(smoke_text)
    smoke_tokens = tokenizer.text_to_tokens(normalized, language_id=None)
    smoke_token_ids = [int(value) for value in smoke_tokens.reshape(-1).tolist()]

    receipt = {
        "schema_version": "gooya.native.fixed-voice/v1",
        "source_revision": SOURCE_REVISION,
        "reference": {
            "file": "samples/canonical.wav",
            "sha256": REFERENCE_SHA256,
        },
        "settings": {"exaggeration": 0.5, "cfg_weight": 0.5},
        "t3": {
            "file": t3_path.name,
            "sha256": sha256(t3_path),
            "size": t3_path.stat().st_size,
            "shape": list(t3_conditioning.shape),
            "dtype": str(t3_conditioning.dtype),
        },
        "s3gen": {
            "file": s3_path.name,
            "sha256": sha256(s3_path),
            "size": s3_path.stat().st_size,
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(s3_tensors.items())
            },
            "none_keys": sorted(key for key, value in engine.conds.gen.items() if value is None),
        },
        "tokenizer_smoke": {
            "text": smoke_text,
            "normalized": normalized,
            "token_ids": smoke_token_ids,
        },
    }
    receipt_path = output / "fixed-voice-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
