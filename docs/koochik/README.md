# Gooya Koochik v2.0-exp in native Rust

The integration accepts raw Persian text, runs Negara v7.1, generates audio codes with OmniVoice, and decodes them to 24 kHz WAV. Apple Silicon uses tract Metal by default; other platforms use tract CPU. Koochik does not use ONNX Runtime or Python for inference.

**Release validation is in progress.** The current candidate is 503,670,910 bytes: grouped 6-bit transformer weights, unchanged text/audio embeddings and output heads, and FP32 arithmetic. Earlier candidates and their failures are retained in `quantization-status.json`. Nothing is promoted until the full Shenava gate and the actual native consumer checks pass.

## Load and test

Once the private derivative is published, authenticate with Hugging Face and use:

```sh
cargo build --release --no-default-features --manifest-path desktop/Cargo.toml --bin gooya_koochik
./desktop/target/release/gooya_koochik --download ./koochik
./desktop/target/release/gooya_koochik ./koochik output.wav 'سلام، حالت چطوره؟'
```

The downloader reads `HF_TOKEN` or the standard Hugging Face token cache, resolves an immutable commit, verifies every asset checksum, and expands the weight cache. The app provides a Koochik download button and model picker. `GOOYA_KOOCHIK_MODEL_DIR` points to an existing bundle.

## Runtime

Weights are quantized for download storage and dequantized to FP32 arithmetic. This is not low-bit activation inference or a sub-gigabyte RAM claim. The compact vocabulary keeps 3,190 text embedding rows. Unsupported text-token IDs are rejected.

`GOOYA_KOOCHIK_DEVICE=cpu` explicitly selects CPU; `metal` explicitly selects native Metal on macOS. Backend and generation timing are recorded in the output report. The Metal transform preserves FP32 precision and reports its actual Metal operator count. The frontend and codec remain on CPU.

The phonemizer compiles its ByT5 decoder once per phrase. A tract 0.23.4 `FoldUniformMask` bug incorrectly removed the five-beam broadcast dimension, so that one optimization pass is disabled for the dynamic decoder. All twelve sealed frontend, duration and conditioning comparisons pass exactly (`dynamic-g2p-results.json`).

The app retains its last speech and codec plans between requests, checks asset metadata to invalidate changed bundles, and shows preparation and per-pass progress. Changed input lengths specialize a fresh plan. Only the last shape is retained to bound memory.

## Acceptance

The user defined parity as **Shenava transcription parity**, with optional listening review. See `PARITY_CONTRACT.json` and the frozen `parity-suite.json`.

- Compare source and candidate recordings using the same Shenava process, model, token files, decoder and normalization.
- Send neither expected text nor hotwords to the recognizer.
- Require `1 - corpus paired-transcript WER > 0.98` across all twelve phrases, with nonempty transcripts.
- Retain every difference, including cases where the candidate matches the requested text better.
- Also report each side's errors against the requested text, audio duration, and actual native runtime.
- Internal audio-code equality and waveform cosine are diagnostics, not this acceptance gate.

Source recordings use the merged continuation-200 checkpoint loaded in PyTorch FP32, pinned at `537bb48320fd657415eef5b4fb6796b6cebab195`. The original lossless Rust CPU implementation independently matched every final code on all twelve cases (`baseline-results.json`).

Sampling experiments at 16 or 24 passes are compared to the same 32-pass original recordings. Their changed step count is reported; the transcript threshold is unchanged. GPU screening with `screen_mps.py` is never sufficient for acceptance: candidates must pass through actual native tract.

This is a finite development suite, not a guarantee of equivalent perceived quality or unseen-text accuracy. Default-voice short phrases are supported; custom reference input and long-form internal chunking are not yet implemented. Phrases estimated above ten seconds must be split. Native SplitMix64 seed 43 is not PyTorch seed 43.

## Reproduction tools

- `capture_reference.py`, `capture_suite.py`: source audio and explicit sampling noise.
- `export_step.py`, `export_frontend.py`, `export_codec.py`: portable graphs.
- `prune_vocabulary.py`: compact text embeddings without remapping valid token identities.
- `compress_weights.py`, `package_grouped.py`: grouped weight storage and checked bundles.
- `koochik_parity`, `run_native_suite.py`: full native generation with captured noise.
- `shenava_parity.py`, `watch_shenava_suite.py`: normalized paired transcripts and persistent receipts.
- `koochik_benchmark`: fixed-input timing and output hashes, separate from speech acceptance.
- `write_model_card.py`: verifies full-suite success, evaluated asset identities and sampler settings before promotion.

The rejected FP16 linear experiment remains opt-in for reproducibility (`GOOYA_EXPERIMENTAL_FP16_LINEAR=1`). `GOOYA_EXPERIMENTAL_STEPS` overrides the manifest's pass count for research; ordinary loading uses the manifest. `GOOYA_KOOCHIK_THREADS` controls CPU thread experiments. No experimental setting is promoted on speed alone.
