#!/usr/bin/env python3
"""Export fixed-bucket S3Gen flow preparation and one Euler estimator step."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

from chatterbox.models.s3gen import S3Gen


PROMPT_MELS = 244
FLOW_STEPS = 10
SEED = 20260816


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FlowPrepare(nn.Module):
    def __init__(self, s3gen: S3Gen, conditioning: dict[str, torch.Tensor], bucket: int) -> None:
        super().__init__()
        self.flow = s3gen.flow
        self.bucket = bucket
        self.register_buffer("prompt_token", conditioning["prompt_token"].long())
        self.register_buffer("prompt_token_len", conditioning["prompt_token_len"].long())
        self.register_buffer("prompt_feat", conditioning["prompt_feat"].float())
        self.register_buffer("speaker_embedding", conditioning["embedding"].float())

    def forward(
        self, speech_tokens: torch.Tensor, speech_token_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = speech_tokens.shape[0]
        speaker = F.normalize(self.speaker_embedding, dim=1)
        speaker = self.flow.spk_embed_affine_layer(speaker)
        prompt = self.prompt_token.expand(batch, -1)
        tokens = torch.cat((prompt, speech_tokens.long()), dim=1)
        token_length = self.prompt_token_len.expand(batch) + speech_token_length.long()
        token_positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        token_mask = (token_positions < token_length[:, None]).unsqueeze(-1).to(speaker)
        token_embeddings = self.flow.input_embedding(tokens) * token_mask
        hidden, hidden_mask = self.flow.encoder(token_embeddings, token_length)
        mu = self.flow.encoder_proj(hidden).transpose(1, 2).contiguous()
        hidden_length = hidden_mask.sum(dim=-1).squeeze(dim=-1).long()
        generated_mels = self.bucket * self.flow.token_mel_ratio
        prompt_feat = self.prompt_feat.expand(batch, -1, -1)
        cond = torch.cat(
            (
                prompt_feat,
                torch.zeros(
                    (batch, generated_mels, self.flow.output_size),
                    dtype=prompt_feat.dtype,
                    device=prompt_feat.device,
                ),
            ),
            dim=1,
        ).transpose(1, 2).contiguous()
        mel_positions = torch.arange(mu.shape[2], device=mu.device)[None, :]
        mask = (mel_positions < hidden_length[:, None]).unsqueeze(1).to(mu)
        return mu, mask, speaker, cond


class FlowStep(nn.Module):
    def __init__(self, s3gen: S3Gen) -> None:
        super().__init__()
        self.estimator = s3gen.flow.decoder.estimator
        self.cfg = float(s3gen.flow.decoder.inference_cfg_rate)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker: torch.Tensor,
        cond: torch.Tensor,
        time: torch.Tensor,
        next_time: torch.Tensor,
    ) -> torch.Tensor:
        x_in = torch.cat((x, x), dim=0)
        mask_in = torch.cat((mask, mask), dim=0)
        mu_in = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        time_in = torch.cat((time, time), dim=0)
        speaker_in = torch.cat((speaker, torch.zeros_like(speaker)), dim=0)
        cond_in = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        derivative = self.estimator(
            x=x_in,
            mask=mask_in,
            mu=mu_in,
            t=time_in,
            spks=speaker_in,
            cond=cond_in,
            r=None,
        )
        conditioned, unconditioned = derivative.chunk(2, dim=0)
        derivative = (1.0 + self.cfg) * conditioned - self.cfg * unconditioned
        return x + (next_time - time).reshape(1, 1, 1) * derivative


def cosine_schedule(steps: int = FLOW_STEPS) -> torch.Tensor:
    linear = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    return 1 - torch.cos(linear * 0.5 * torch.pi)


def manual_flow(
    prepare: FlowPrepare,
    step: FlowStep,
    tokens: torch.Tensor,
    token_length: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    mu, mask, speaker, cond = prepare(tokens, token_length)
    state = noise
    schedule = cosine_schedule()
    with torch.inference_mode():
        for left, right in zip(schedule[:-1], schedule[1:], strict=True):
            state = step(state, mu, mask, speaker, cond, left[None], right[None])
    return state[:, :, PROMPT_MELS:], (mu, mask, speaker, cond)


def compare(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    delta = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(delta.max(initial=0.0)),
        "mean_abs_error": float(delta.mean()),
    }


def graph_summary(path: Path) -> dict[str, object]:
    model = onnx.load_model(str(path), load_external_data=False)
    return {
        "nodes": len(model.graph.node),
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "initializers": len(model.graph.initializer),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixed-voice", type=Path, required=True)
    parser.add_argument("--speech-tokens-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", type=int, default=32)
    args = parser.parse_args()
    source = args.source.resolve()
    fixed_voice = args.fixed_voice.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    tokens_payload = json.loads(args.speech_tokens_json.read_text(encoding="utf-8"))
    token_values = tokens_payload["fp32_tokens"][: args.bucket]
    if len(token_values) != args.bucket or max(token_values) >= 6561:
        raise RuntimeError("speech-token canary does not fill the selected valid bucket")
    tokens = torch.tensor([token_values], dtype=torch.long)
    token_length = torch.tensor([args.bucket], dtype=torch.long)

    s3gen = S3Gen().eval()
    missing, unexpected = s3gen.load_state_dict(
        load_file(str(source / "s3gen.safetensors")), strict=False
    )
    if unexpected or missing not in ([], ["tokenizer.window"]):
        raise RuntimeError(f"S3Gen state mismatch: missing={missing}, unexpected={unexpected}")
    conditioning = load_file(str(fixed_voice / "canonical-s3gen-conditioning.safetensors"))
    prepare = FlowPrepare(s3gen, conditioning, args.bucket).eval()
    step = FlowStep(s3gen).eval()

    with torch.inference_mode():
        prepared = prepare(tokens, token_length)
        torch.manual_seed(SEED)
        source_mel, _ = s3gen.flow.inference(
            token=tokens,
            token_len=torch.tensor([args.bucket], dtype=torch.long),
            prompt_token=conditioning["prompt_token"],
            prompt_token_len=conditioning["prompt_token_len"],
            prompt_feat=conditioning["prompt_feat"],
            prompt_feat_len=None,
            embedding=conditioning["embedding"],
            finalize=True,
            n_timesteps=FLOW_STEPS,
            noised_mels=None,
            meanflow=False,
        )
        torch.manual_seed(SEED)
        noise = torch.randn_like(prepared[0])
        manual_mel, _ = manual_flow(prepare, step, tokens, token_length, noise)
    source_delta = compare(source_mel.numpy(), manual_mel.numpy())
    if source_delta["max_abs_error"] > 1e-4:
        raise RuntimeError(f"manual flow drifted from source: {source_delta}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        stage = Path(temp_name)
        prepare_path = stage / f"s3-flow-prepare-b{args.bucket}.onnx"
        step_path = stage / f"s3-flow-step-b{args.bucket}.onnx"
        torch.onnx.export(
            prepare,
            (tokens, token_length),
            str(prepare_path),
            input_names=["speech_tokens", "speech_token_length"],
            output_names=["mu", "mask", "speaker", "cond"],
            opset_version=20,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(prepare_path))
        time = torch.tensor([0.0], dtype=torch.float32)
        next_time = cosine_schedule()[1:2]
        torch.onnx.export(
            step,
            (noise, *prepared, time, next_time),
            str(step_path),
            input_names=["x", "mu", "mask", "speaker", "cond", "time", "next_time"],
            output_names=["next_x"],
            opset_version=20,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(step_path))

        providers = ["CPUExecutionProvider"]
        prepare_session = ort.InferenceSession(str(prepare_path), providers=providers)
        ort_prepared = prepare_session.run(
            None,
            {
                "speech_tokens": tokens.numpy(),
                "speech_token_length": token_length.numpy(),
            },
        )
        step_session = ort.InferenceSession(str(step_path), providers=providers)
        state = noise.numpy()
        schedule = cosine_schedule().numpy()
        for index in range(FLOW_STEPS):
            state = step_session.run(
                None,
                {
                    "x": state,
                    "mu": ort_prepared[0],
                    "mask": ort_prepared[1],
                    "speaker": ort_prepared[2],
                    "cond": ort_prepared[3],
                    "time": schedule[index : index + 1],
                    "next_time": schedule[index + 1 : index + 2],
                },
            )[0]
        ort_mel = state[:, :, PROMPT_MELS:]
        ort_delta = compare(manual_mel.numpy(), ort_mel)
        if ort_delta["max_abs_error"] > 2e-3:
            raise RuntimeError(f"ONNX flow drifted from PyTorch: {ort_delta}")
        np.save(stage / "canary-mel.npy", ort_mel, allow_pickle=False)

        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        receipt = {
            "schema_version": "gooya.native.s3gen-flow-onnx/v1",
            "source_revision": "8844b3a6ebbefa0e1ac4baef494d2e8d7eda9d9c",
            "bucket_speech_tokens": args.bucket,
            "prompt_mels": PROMPT_MELS,
            "generated_mels": args.bucket * 2,
            "variable_length_contract": {
                "speech_tokens": f"right-padded to bucket {args.bucket}",
                "speech_token_length": "number of valid unpadded speech tokens",
                "waveform_trim_samples": "speech_token_length * 2 * 480",
            },
            "flow_steps": FLOW_STEPS,
            "seed": SEED,
            "source_vs_manual": source_delta,
            "manual_vs_onnxruntime": ort_delta,
            "graphs": {
                prepare_path.name: graph_summary(prepare_path),
                step_path.name: graph_summary(step_path),
            },
            "files": files,
            "validation_status": "PASS",
        }
        (stage / "flow-export-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        Path(temp_name).replace(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
