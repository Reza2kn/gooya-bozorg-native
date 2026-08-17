#!/usr/bin/env python3
"""Export Gooya Bozorg T3 prefill/decode graphs with an explicit KV cache."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from safetensors.torch import load_file
import torch
from torch import nn
from transformers import DynamicCache

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config


SOURCE_REVISION = "8844b3a6ebbefa0e1ac4baef494d2e8d7eda9d9c"
LAYERS = 30
HEADS = 16
HEAD_DIM = 64
HIDDEN = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_cache(cache: DynamicCache) -> tuple[torch.Tensor, ...]:
    return tuple(value for layer in cache.layers for value in (layer.keys, layer.values))


class Prefill(nn.Module):
    def __init__(self, t3: T3, conditioning: torch.Tensor) -> None:
        super().__init__()
        self.t3 = t3
        self.register_buffer("conditioning", conditioning)

    def forward(self, text_token_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        start = torch.full(
            (text_token_ids.shape[0], 1),
            self.t3.hp.start_text_token,
            dtype=torch.long,
            device=text_token_ids.device,
        )
        stop = torch.full(
            (text_token_ids.shape[0], 1),
            self.t3.hp.stop_text_token,
            dtype=torch.long,
            device=text_token_ids.device,
        )
        text_token_ids = torch.cat((start, text_token_ids.long(), stop), dim=1)
        # The source uses two rows for classifier-free guidance. The second row
        # deliberately has zero text embeddings while retaining voice/emotion.
        text_token_embeddings = self.t3.text_emb(text_token_ids)
        text_positions = self.t3.text_pos_emb(text_token_ids)
        # Source CFG zeroes only the unconditional token embeddings and then
        # adds learned positional embeddings to both rows.
        text = torch.cat(
            (
                text_token_embeddings + text_positions,
                torch.zeros_like(text_token_embeddings) + text_positions,
            ),
            dim=0,
        )
        cond = self.conditioning.expand(2, -1, -1)
        bos = torch.full(
            (1, 1), self.t3.hp.start_speech_token, dtype=torch.long, device=text_token_ids.device
        )
        bos_embed = self.t3.speech_emb(bos) + self.t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = bos_embed.expand(2, -1, -1)
        inputs = torch.cat((cond, text, bos_embed), dim=1)
        output = self.t3.tfmr(inputs_embeds=inputs, use_cache=True, return_dict=True)
        logits = self.t3.speech_head(output.last_hidden_state[:, -1:, :])
        return (logits, *flatten_cache(output.past_key_values))


class Decode(nn.Module):
    def __init__(self, t3: T3) -> None:
        super().__init__()
        self.t3 = t3

    def forward(
        self,
        next_token_id: torch.Tensor,
        speech_position: torch.Tensor,
        cache_position: torch.Tensor,
        *past: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache_pairs = tuple((past[index], past[index + 1]) for index in range(0, len(past), 2))
        cache = DynamicCache(cache_pairs, config=self.t3.tfmr.config)
        token = next_token_id.expand(2, -1)
        inputs = self.t3.speech_emb(token)
        inputs = inputs + self.t3.speech_pos_emb.get_fixed_embedding(speech_position).expand(2, -1, -1)
        output = self.t3.tfmr(
            inputs_embeds=inputs,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        logits = self.t3.speech_head(output.last_hidden_state[:, -1:, :])
        return (logits, *flatten_cache(output.past_key_values))


def source_semantics_prefill(
    t3: T3, conditioning: torch.Tensor, raw_text_ids: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Mirror T3.inference/prepare_input_embeds without calling the export wrapper."""
    start = torch.full_like(raw_text_ids[:, :1], t3.hp.start_text_token)
    stop = torch.full_like(raw_text_ids[:, :1], t3.hp.stop_text_token)
    bounded = torch.cat((start, raw_text_ids, stop), dim=1).expand(2, -1)
    text = t3.text_emb(bounded)
    text[1].zero_()
    text = text + t3.text_pos_emb(bounded)
    cond = conditioning.expand(2, -1, -1)
    bos = torch.full((2, 1), t3.hp.start_speech_token, dtype=torch.long)
    speech = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    inputs = torch.cat((cond, text, speech), dim=1)
    output = t3.tfmr(inputs_embeds=inputs, use_cache=True, return_dict=True)
    logits = t3.speech_head(output.last_hidden_state[:, -1:, :])
    return (logits, *flatten_cache(output.past_key_values))


def cache_names(prefix: str) -> list[str]:
    return [name for layer in range(LAYERS) for name in
            (f"{prefix}_key_{layer}", f"{prefix}_value_{layer}")]


def graph_summary(path: Path) -> dict[str, object]:
    model = onnx.load_model(str(path), load_external_data=False)
    external = sorted(
        {
            item.value
            for tensor in model.graph.initializer
            for item in tensor.external_data
            if item.key == "location"
        }
    )
    return {
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "initializers": len(model.graph.initializer),
        "external_data": external,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixed-voice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-text-tokens", type=int, default=18)
    args = parser.parse_args()
    source = args.source.resolve()
    fixed_voice = args.fixed_voice.resolve()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)

    t3 = T3(T3Config.multilingual()).eval()
    t3.load_state_dict(load_file(str(source / "t3_fa.safetensors")))
    t3.tfmr.set_attn_implementation("eager")
    conditioning = load_file(str(fixed_voice / "canonical-t3-conditioning.safetensors"))[
        "conditioning"
    ]
    prefill = Prefill(t3, conditioning).eval()
    decode = Decode(t3).eval()

    receipt = json.loads((fixed_voice / "fixed-voice-receipt.json").read_text(encoding="utf-8"))
    smoke_ids = receipt["tokenizer_smoke"]["token_ids"]
    if len(smoke_ids) != args.smoke_text_tokens:
        raise RuntimeError(
            f"tokenizer smoke length drifted: expected {args.smoke_text_tokens}, got {len(smoke_ids)}"
        )
    text_ids = torch.tensor([smoke_ids], dtype=torch.long)

    with torch.inference_mode():
        reference = prefill(text_ids)
        source_reference = source_semantics_prefill(t3, conditioning, text_ids)
        reference_logits = reference[0].detach().cpu().numpy()
    contract_max_error = max(
        float((actual - expected).abs().max())
        for actual, expected in zip(reference, source_reference, strict=True)
    )
    if contract_max_error > 1e-6:
        raise RuntimeError(f"prefill wrapper drifted from source semantics: {contract_max_error}")
    prefill_top = np.argsort(reference_logits[0, 0])[-8:][::-1].tolist()
    prefill_top_logits = [float(reference_logits[0, 0, index]) for index in prefill_top]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as tmp:
        stage = Path(tmp)
        prefill_path = stage / "t3-prefill.onnx"
        decode_path = stage / "t3-decode.onnx"
        present_names = cache_names("present")
        past_names = cache_names("past")

        torch.onnx.export(
            prefill,
            (text_ids,),
            str(prefill_path),
            input_names=["text_token_ids"],
            output_names=["logits", *present_names],
            dynamic_axes={
                "text_token_ids": {1: "text_sequence"},
                **{name: {2: "context_sequence"} for name in present_names},
            },
            opset_version=20,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(prefill_path))

        past_length = int(reference[1].shape[2])
        next_id = torch.tensor([[prefill_top[0]]], dtype=torch.long)
        speech_position = torch.tensor([1], dtype=torch.long)
        cache_position = torch.tensor([past_length], dtype=torch.long)
        past = tuple(value.detach() for value in reference[1:])
        torch.onnx.export(
            decode,
            (next_id, speech_position, cache_position, *past),
            str(decode_path),
            input_names=["next_token_id", "speech_position", "cache_position", *past_names],
            output_names=["logits", *present_names],
            dynamic_axes={
                **{name: {2: "past_sequence"} for name in past_names},
                **{name: {2: "present_sequence"} for name in present_names},
            },
            opset_version=20,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(decode_path))

        files = {
            path.name: {"size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        export_receipt = {
            "schema_version": "gooya.native.t3-onnx-export/v1",
            "source_revision": SOURCE_REVISION,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "opset": 20,
            "cache": {"layers": LAYERS, "heads": HEADS, "head_dim": HEAD_DIM},
            "text_contract": {
                "input": "raw tokenizer token ids without boundaries",
                "graph_inserts": {
                    "start_text_token": t3.hp.start_text_token,
                    "stop_text_token": t3.hp.stop_text_token,
                },
                "source_semantics_max_abs_error": contract_max_error,
            },
            "smoke": {
                "text": receipt["tokenizer_smoke"]["text"],
                "token_ids": smoke_ids,
                "prefill_top_token_ids": prefill_top,
                "prefill_top_logits": prefill_top_logits,
                "context_length": past_length,
            },
            "graphs": {
                "t3-prefill.onnx": graph_summary(prefill_path),
                "t3-decode.onnx": graph_summary(decode_path),
            },
            "files": files,
            "validation_status": "PYTORCH_REFERENCE_ONLY",
        }
        (stage / "export-receipt.json").write_text(
            json.dumps(export_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(tmp).replace(destination)
    print(json.dumps(export_receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
