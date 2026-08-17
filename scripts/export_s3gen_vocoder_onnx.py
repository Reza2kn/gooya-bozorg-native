#!/usr/bin/env python3
"""Export deterministic S3Gen HiFT neural vocoder up to native ISTFT."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import wave

import numpy as np
import onnx
import onnxruntime as ort
from safetensors.torch import load_file
import torch
from torch import nn
from torch.distributions import Uniform
from torch.nn import functional as F

from chatterbox.models.s3gen import S3Gen


SEED = 20260816
PROMPT_MELS = 244


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    delta = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(delta.max(initial=0.0)),
        "mean_abs_error": float(delta.mean()),
    }


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int = 24000) -> None:
    pcm = (np.clip(waveform.reshape(-1), -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class F0Source(nn.Module):
    """F0 predictor and explicit harmonic source; native code performs STFT."""

    def __init__(self, s3gen: S3Gen) -> None:
        super().__init__()
        self.vocoder = s3gen.mel2wav
        harmonics = torch.arange(
            1, self.vocoder.nb_harmonics + 2, dtype=torch.float32
        ).reshape(1, -1, 1)
        self.register_buffer("harmonics", harmonics)

    def forward(
        self,
        mel: torch.Tensor,
        harmonic_phase: torch.Tensor,
        harmonic_noise: torch.Tensor,
    ) -> torch.Tensor:
        f0 = self.vocoder.f0_predictor(mel)
        f0_upsampled = self.vocoder.f0_upsamp(f0[:, None])
        frequency = f0_upsampled * self.harmonics / float(self.vocoder.sampling_rate)
        theta = 2 * torch.pi * torch.remainder(torch.cumsum(frequency, dim=-1), 1.0)
        sine = self.vocoder.m_source.sine_amp * torch.sin(theta + harmonic_phase)
        uv = (f0_upsampled > self.vocoder.m_source.l_sin_gen.voiced_threshold).to(mel.dtype)
        noise_amplitude = (
            uv * self.vocoder.m_source.l_sin_gen.noise_std
            + (1 - uv) * self.vocoder.m_source.sine_amp / 3
        )
        sine = sine * uv + noise_amplitude * harmonic_noise
        source = self.vocoder.m_source.l_tanh(
            self.vocoder.m_source.l_linear(sine.transpose(1, 2))
        ).transpose(1, 2)
        return source


class VocoderSpectral(nn.Module):
    """HiFT convolution stack; native code provides STFT and consumes ISTFT."""

    def __init__(self, s3gen: S3Gen) -> None:
        super().__init__()
        self.vocoder = s3gen.mel2wav

    def forward(
        self, mel: torch.Tensor, source_stft: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.vocoder.conv_pre(mel)
        for index in range(self.vocoder.num_upsamples):
            hidden = F.leaky_relu(hidden, self.vocoder.lrelu_slope)
            hidden = self.vocoder.ups[index](hidden)
            if index == self.vocoder.num_upsamples - 1:
                hidden = self.vocoder.reflection_pad(hidden)
            source_stage = self.vocoder.source_downs[index](source_stft)
            source_stage = self.vocoder.source_resblocks[index](source_stage)
            hidden = hidden + source_stage
            residual = self.vocoder.resblocks[index * self.vocoder.num_kernels](hidden)
            for kernel in range(1, self.vocoder.num_kernels):
                residual = residual + self.vocoder.resblocks[
                    index * self.vocoder.num_kernels + kernel
                ](hidden)
            hidden = residual / self.vocoder.num_kernels
        hidden = F.leaky_relu(hidden)
        hidden = self.vocoder.conv_post(hidden)
        bins = self.vocoder.istft_params["n_fft"] // 2 + 1
        magnitude = torch.exp(hidden[:, :bins, :]).clamp(max=1e2)
        phase = torch.sin(hidden[:, bins:, :])
        return magnitude, phase


def load_flow_canary(flow_dir: Path, tokens_json: Path) -> tuple[np.ndarray, int]:
    receipt = json.loads((flow_dir / "flow-export-receipt.json").read_text(encoding="utf-8"))
    bucket = int(receipt["bucket_speech_tokens"])
    cached = flow_dir / "canary-mel.npy"
    if cached.is_file():
        return np.load(cached, allow_pickle=False), bucket
    tokens = np.array(
        [json.loads(tokens_json.read_text(encoding="utf-8"))["fp32_tokens"][:bucket]],
        dtype=np.int64,
    )
    prepare = ort.InferenceSession(
        str(flow_dir / f"s3-flow-prepare-b{bucket}.onnx"), providers=["CPUExecutionProvider"]
    )
    prepare_inputs = {"speech_tokens": tokens}
    if any(value.name == "speech_token_length" for value in prepare.get_inputs()):
        prepare_inputs["speech_token_length"] = np.array([bucket], dtype=np.int64)
    prepared = prepare.run(None, prepare_inputs)
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    state = torch.randn(prepared[0].shape, generator=rng).numpy()
    schedule = (1 - torch.cos(torch.linspace(0, 1, 11) * 0.5 * torch.pi)).numpy()
    step = ort.InferenceSession(
        str(flow_dir / f"s3-flow-step-b{bucket}.onnx"), providers=["CPUExecutionProvider"]
    )
    for index in range(10):
        state = step.run(
            None,
            {
                "x": state,
                "mu": prepared[0],
                "mask": prepared[1],
                "speaker": prepared[2],
                "cond": prepared[3],
                "time": schedule[index : index + 1],
                "next_time": schedule[index + 1 : index + 2],
            },
        )[0]
    return state[:, :, PROMPT_MELS:], bucket


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--speech-tokens-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    flow_dir = args.flow.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    mel_numpy, bucket = load_flow_canary(flow_dir, args.speech_tokens_json.resolve())
    mel = torch.from_numpy(mel_numpy).float()
    s3gen = S3Gen().eval()
    missing, unexpected = s3gen.load_state_dict(
        load_file(str(source / "s3gen.safetensors")), strict=False
    )
    if unexpected or missing not in ([], ["tokenizer.window"]):
        raise RuntimeError(f"S3Gen state mismatch: missing={missing}, unexpected={unexpected}")
    source_wrapper = F0Source(s3gen).eval()
    spectral_wrapper = VocoderSpectral(s3gen).eval()

    source_length = mel.shape[2] * 480
    torch.manual_seed(SEED)
    with torch.inference_mode():
        source_wav, _ = s3gen.mel2wav.inference(mel)
    torch.manual_seed(SEED)
    harmonic_phase = Uniform(-torch.pi, torch.pi).sample(
        (1, s3gen.mel2wav.nb_harmonics + 1, 1)
    )
    harmonic_phase[:, 0, :] = 0
    harmonic_noise = torch.randn(
        (1, s3gen.mel2wav.nb_harmonics + 1, source_length), dtype=torch.float32
    )
    with torch.inference_mode():
        source = source_wrapper(mel, harmonic_phase, harmonic_noise)
        source_real, source_imag = s3gen.mel2wav._stft(source.squeeze(1))
        source_stft = torch.cat((source_real, source_imag), dim=1)
        magnitude, phase = spectral_wrapper(mel, source_stft)
        manual_wav = s3gen.mel2wav._istft(magnitude, phase).clamp(
            -s3gen.mel2wav.audio_limit, s3gen.mel2wav.audio_limit
        )
    # Tracing with a tensor born in inference_mode makes autograd-backed
    # parametrized convolutions reject it. Materialize a normal detached copy.
    source_stft = source_stft.detach().clone()
    source_delta = compare(source_wav.numpy(), manual_wav.numpy())
    if source_delta["max_abs_error"] > 2e-4:
        raise RuntimeError(f"explicit vocoder drifted from source: {source_delta}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        stage = Path(temp_name)
        source_path = stage / f"s3-vocoder-source-b{bucket}.onnx"
        graph_path = stage / f"s3-vocoder-spectral-b{bucket}.onnx"
        torch.onnx.export(
            source_wrapper,
            (mel, harmonic_phase, harmonic_noise),
            str(source_path),
            input_names=["mel", "harmonic_phase", "harmonic_noise"],
            output_names=["source"],
            opset_version=20,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(source_path))
        torch.onnx.export(
            spectral_wrapper,
            (mel, source_stft),
            str(graph_path),
            input_names=["mel", "source_stft"],
            output_names=["magnitude", "phase"],
            opset_version=20,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(graph_path))
        source_session = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
        ort_source = source_session.run(
            None,
            {
                "mel": mel.numpy(),
                "harmonic_phase": harmonic_phase.numpy(),
                "harmonic_noise": harmonic_noise.numpy(),
            },
        )[0]
        ort_source_real, ort_source_imag = s3gen.mel2wav._stft(
            torch.from_numpy(ort_source).squeeze(1)
        )
        ort_source_stft = torch.cat((ort_source_real, ort_source_imag), dim=1).numpy()
        session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"mel": mel.numpy(), "source_stft": ort_source_stft})
        spectral_validation = {
            "magnitude": compare(magnitude.numpy(), actual[0]),
            "phase": compare(phase.numpy(), actual[1]),
            "source": compare(source.numpy(), ort_source),
        }
        if max(item["max_abs_error"] for item in spectral_validation.values()) > 3e-3:
            raise RuntimeError(f"ONNX vocoder drifted: {spectral_validation}")
        ort_wav = s3gen.mel2wav._istft(
            torch.from_numpy(actual[0]), torch.from_numpy(actual[1])
        ).clamp(-s3gen.mel2wav.audio_limit, s3gen.mel2wav.audio_limit)
        waveform_validation = compare(manual_wav.numpy(), ort_wav.numpy())
        np.save(stage / "canary-waveform.npy", ort_wav.numpy(), allow_pickle=False)
        write_wav(stage / "canary-waveform.wav", ort_wav.numpy())
        models = {
            source_path.name: onnx.load_model(str(source_path), load_external_data=False),
            graph_path.name: onnx.load_model(str(graph_path), load_external_data=False),
        }
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        receipt = {
            "schema_version": "gooya.native.s3gen-vocoder-onnx/v1",
            "source_revision": "8844b3a6ebbefa0e1ac4baef494d2e8d7eda9d9c",
            "bucket_speech_tokens": bucket,
            "mel_frames": int(mel.shape[2]),
            "sample_rate": 24000,
            "native_dsp": {
                "operations": ["STFT(source)", "ISTFT(magnitude, phase)"],
                "n_fft": 16,
                "hop_length": 4
            },
            "source_vs_explicit": source_delta,
            "pytorch_vs_onnxruntime": spectral_validation,
            "waveform_after_native_istft": waveform_validation,
            "graphs": {
                name: {
                    "nodes": len(model.graph.node),
                    "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
                    "initializers": len(model.graph.initializer),
                }
                for name, model in models.items()
            },
            "files": files,
            "validation_status": "PASS",
        }
        (stage / "vocoder-export-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        Path(temp_name).replace(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
