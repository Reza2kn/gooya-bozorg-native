#!/usr/bin/env python3
"""Run the complete fixed-voice T3 -> S3Gen -> HiFT ONNX pipeline."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import wave

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

from export_t3_onnx import cache_names


SAMPLE_RATE = 24_000
PROMPT_MELS = 244
SPEECH_EOS = 6562


def normalize_text(text: str) -> str:
    text = " ".join(text.split())
    for old, new in (
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "),
        (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","),
        ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    ):
        text = text.replace(old, new)
    text = text.rstrip()
    if text and text[-1] not in ".!?-,":
        text += "."
    return text


def tokenize(tokenizer_path: Path, text: str) -> tuple[str, list[int]]:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("text must not be empty")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = tokenizer.encode(normalized.replace(" ", "[SPACE]")).ids
    return normalized, ids


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = min(8, max(1, __import__("os").cpu_count() or 1))
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def generate_tokens(t3_dir: Path, text_ids: list[int], bucket: int, cfg: float) -> tuple[list[int], float]:
    started = time.perf_counter()
    prefill = session(t3_dir / "t3-prefill.onnx")
    outputs = prefill.run(None, {"text_token_ids": np.array([text_ids], dtype=np.int64)})
    del prefill
    gc.collect()
    decode = session(t3_dir / "t3-decode.onnx")
    tokens: list[int] = []
    for position in range(bucket):
        conditional, unconditional = outputs[0][0, -1], outputs[0][1, -1]
        token = int(np.argmax(conditional + cfg * (conditional - unconditional)))
        if token == SPEECH_EOS:
            break
        tokens.append(token)
        past = outputs[1:]
        past_length = past[0].shape[2]
        outputs = decode.run(
            None,
            {
                "next_token_id": np.array([[token]], dtype=np.int64),
                "speech_position": np.array([position + 1], dtype=np.int64),
                "cache_position": np.array([past_length], dtype=np.int64),
                **dict(zip(cache_names("past"), past, strict=True)),
            },
        )
    del decode, outputs
    gc.collect()
    if not tokens:
        raise RuntimeError("T3 emitted EOS before any speech token")
    return tokens, time.perf_counter() - started


def run_flow(flow_dir: Path, tokens: list[int], bucket: int, seed: int) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    padded = np.zeros((1, bucket), dtype=np.int64)
    padded[0, : len(tokens)] = tokens
    prepare = session(flow_dir / f"s3-flow-prepare-b{bucket}.onnx")
    prepared = prepare.run(
        None,
        {
            "speech_tokens": padded,
            "speech_token_length": np.array([len(tokens)], dtype=np.int64),
        },
    )
    del prepare
    gc.collect()
    state = np.random.default_rng(seed).standard_normal(prepared[0].shape, dtype=np.float32)
    schedule = (1 - np.cos(np.linspace(0, 1, 11, dtype=np.float32) * 0.5 * np.pi)).astype(np.float32)
    step = session(flow_dir / f"s3-flow-step-b{bucket}.onnx")
    for index in range(10):
        state = step.run(
            None,
            {
                "x": state,
                "mu": prepared[0], "mask": prepared[1],
                "speaker": prepared[2], "cond": prepared[3],
                "time": schedule[index:index + 1],
                "next_time": schedule[index + 1:index + 2],
            },
        )[0]
    mel = state[:, :, PROMPT_MELS:PROMPT_MELS + bucket * 2]
    del step, prepared, state
    gc.collect()
    return mel, time.perf_counter() - started


def run_vocoder(vocoder_dir: Path, mel: np.ndarray, bucket: int, seed: int) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, (1, 9, 1)).astype(np.float32)
    phase[:, 0, :] = 0
    noise = rng.standard_normal((1, 9, mel.shape[2] * 480), dtype=np.float32)
    source_graph = session(vocoder_dir / f"s3-vocoder-source-b{bucket}.onnx")
    source = source_graph.run(None, {"mel": mel, "harmonic_phase": phase, "harmonic_noise": noise})[0]
    del source_graph
    window = torch.hann_window(16, periodic=True)
    source_stft = torch.stft(
        torch.from_numpy(source).squeeze(1), 16, 4, 16, window=window, return_complex=True
    )
    packed_stft = torch.cat((source_stft.real, source_stft.imag), dim=1).numpy()
    spectral_graph = session(vocoder_dir / f"s3-vocoder-spectral-b{bucket}.onnx")
    magnitude, output_phase = spectral_graph.run(None, {"mel": mel, "source_stft": packed_stft})
    del spectral_graph
    complex_spectrum = torch.polar(torch.from_numpy(magnitude).clamp(max=1e2), torch.from_numpy(output_phase))
    waveform = torch.istft(complex_spectrum, 16, 4, 16, window=window).clamp(-0.99, 0.99).numpy()
    gc.collect()
    return waveform, time.perf_counter() - started


def write_wav(path: Path, waveform: np.ndarray, samples: int) -> None:
    pcm = (waveform.reshape(-1)[:samples] * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--t3", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default="سلام، حالت چطوره؟")
    parser.add_argument("--bucket", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    normalized, text_ids = tokenize(
        args.source.resolve() / "grapheme_mtl_merged_expanded_v1.json", args.text
    )
    tokens, t3_seconds = generate_tokens(args.t3.resolve(), text_ids, args.bucket, args.cfg_weight)
    mel, flow_seconds = run_flow(args.flow.resolve(), tokens, args.bucket, args.seed)
    waveform, vocoder_seconds = run_vocoder(args.vocoder.resolve(), mel, args.bucket, args.seed)
    valid_samples = len(tokens) * 2 * 480
    wav_path = args.output / "pipeline.wav"
    write_wav(wav_path, waveform, valid_samples)
    peak = float(np.max(np.abs(waveform[:, :valid_samples])))
    rms = float(np.sqrt(np.mean(np.square(waveform[:, :valid_samples], dtype=np.float64))))
    receipt = {
        "schema_version": "gooya.native.onnx-pipeline/v1",
        "status": "PASS" if peak > 1e-4 and rms > 1e-5 else "FAIL",
        "text": args.text,
        "normalized_text": normalized,
        "text_token_ids": text_ids,
        "speech_token_ids": tokens,
        "speech_token_length": len(tokens),
        "bucket": args.bucket,
        "seed": args.seed,
        "generation": "greedy_cfg",
        "timing_seconds": {"t3": t3_seconds, "flow": flow_seconds, "vocoder": vocoder_seconds},
        "audio": {
            "path": str(wav_path), "sample_rate": SAMPLE_RATE,
            "samples": valid_samples, "duration_seconds": valid_samples / SAMPLE_RATE,
            "peak": peak, "rms": rms,
        },
        "provider": "CPUExecutionProvider",
    }
    (args.output / "pipeline-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
