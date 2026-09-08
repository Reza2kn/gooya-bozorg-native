# Gooya Koochik v2.0-exp in native Rust

The integration accepts raw Persian text, runs Negara v7.1, generates audio codes with OmniVoice, and decodes them to 24 kHz WAV. Apple Silicon uses tract Metal by default; other platforms use tract CPU. Koochik does not use ONNX Runtime or Python for inference.

**Native validation passed; private publication is pending destination approval.** The selected bundle is 503,670,910 bytes: grouped 6-bit transformer weights, unchanged text/audio embeddings and output heads, FP32 arithmetic, and 24 passes. The complete native Metal suite achieved **98.507% Shenava parity** (one word difference in 67 words across twelve clips). Earlier failures remain in `quantization-status.json`.

The actual app generated the correct greeting in 31.3 seconds cold. A separate repeated-request check measured **31.2 seconds cold and 22.4 seconds cached** for 2.03 seconds of audio on Apple M2 with 24 GiB RAM; the cached plan produced byte-identical WAV output. This remains slower than real time. Peak process RSS was 7.95 GB during the two-request check; that is not a separate VRAM measurement. See `consumer-repeat.json` and `consumer-app-cold.json`. Fresh private-repository download verification remains pending publication.

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

## Kernel profiling

`latency-profile.json` records the fixed-input investigation. The speech graph has one device-to-host synchronization per pass. Host dispatch took roughly 0.02 seconds; the final synchronization waited about 0.75 seconds for queued GPU work. This does not attribute that time to the transfer itself or measure individual GPU kernels.

For the same FP32 inputs, the default MLX kernel produced the same output SHA before and after experimentation. Warm pass timings were 0.69–0.77 seconds initially and 0.96–1.18 seconds in the later control, showing run-to-run variability. MFA took 1.32–3.45 seconds and GGML 1.46–2.48 seconds. Neither improved speed; both changed numerical outputs, so neither was adopted or passed onward to the full speech gate. Production behavior and the accepted sampler remain unchanged.

`GOOYA_PROFILE_OPS=/path/to/ops.json` records host dispatch/wait times for the last graph run. `GOOYA_EXPERIMENTAL_METAL_GEMM=mfa|ggml|mlx` selects a research kernel. Leave both unset for ordinary use. These switches do not establish quality acceptance. Further speed work needs a materially different compute path or a separately validated lower-precision/fewer-pass model; this profiling pass found no safe kernel-switch speedup.

## Core ML quality work

Core ML is an opt-in Apple speech backend through direct Objective-C API calls from Rust. The sampler remains Rust and the codec remains tract CPU. The existing tract `MetalMlxGemm` name refers to a matrix kernel, not the MLX framework.

The fixed-capacity 419 export reconstructs the accepted grouped weights exactly. Short inputs preserve their original positions and attention block; appended keys are masked from real queries. An FP32 padding control retained 100% logit argmax agreement (`coreml-padding-control.json`). This export supports only phrases that fit its capacity and is not a general-input release.

Two full 12-clip Python screens passed the strict >98% Shenava gate:

| Core ML policy | Word differences | Corpus parity |
| --- | --- | --- |
| FP32, 24 passes | 1 / 67 | 98.5075% |
| 24 passes: first 8 FP32, middle 8 FP16, final 8 FP32 | 1 / 67 | 98.5075% |

The mixed policy repairs errors found with all-FP16 inference. These screens use Core ML prediction, the source Python sampler, and ORT CPU codec; see `coreml-python-mixed24.json` and `coreml-python-fp32-24.json`. The suite was used to select the policy, so this is development-set fidelity, not an unseen-text or perceptual-quality guarantee. The separate full native Rust run also passed at **98.5075% (1/67)**: `coreml-native-parity.json`. This uses direct Core ML calls, Rust sampling, and the tract CPU codec. `coreml-native-model-identity.json` and `coreml-native-runtime.json` pin the compiled models and executable. The earlier recognizer service disappeared and its on-disk executable had changed; an isolated evaluator retranscribed both sides with unchanged model/token hashes. All twelve source transcripts matched the earlier screen. See `coreml-native-recognizer.json`; no historical and fresh hypotheses were mixed.

Rejected experiments remain in `coreml-rejected-policies.json`: 32-pass prefix-only precision schedules accumulated at least two differences; enumerated shapes were slow and mispronounced the greeting; norm-preserving conversion did not finish model loading within seven minutes. More steps alone did not fix quality. Earlier all-FP16 and protected-output-head canaries both produced `هت` for `حالت`. Selective tract FP16 also failed (`precision-trunk-screen.json`).

Core ML Tools 9.0 requires the tested NumPy 2.2.6 environment here; NumPy 2.5.3 triggered a scalar-cast conversion error. Torch 2.11 is outside Core ML Tools' advertised tested range. The compile cache keys package contents, tool version, OS version and compute units, and uses independent APFS copy-on-write copies where available.

For local research, set `GOOYA_KOOCHIK_DEVICE=coreml`, `GOOYA_KOOCHIK_COREML_FP32` to the compiled FP32 `.mlmodelc` directory, `GOOYA_KOOCHIK_COREML_FP16` to its FP16 counterpart, and `GOOYA_KOOCHIK_COREML_MIXED=1`. Omit the mixed flag to use FP32 throughout. `koochik_native_suite BUNDLE CASES SUITE.json OUTPUT` retains the models while running the frozen suite. `koochik_consumer_check BUNDLE OUTPUT TEXT` checks raw-text synthesis and cache reuse.

The accepted 504 MB tract bundle remains the default download. The two Core ML research packages total about 2.77 GB before compression and are not a size-compliant replacement. No Core ML artifact is published. Compute-plan device placement is anticipated placement, not a hardware trace; compact packaging and broader input coverage remain required before promotion.

### Native consumer verification

`coreml-native-consumer.json` records raw Persian `سلام، حالت چطوره؟` through the same synthesis function used by the app. Shenava returned the correct greeting. Cold and warm WAVs were byte-identical and the speech graph was reused. On the M2, cold end-to-end time was 66.99 seconds, cached end-to-end time 14.04 seconds, and the output lasted 2.05 seconds. Speech generation itself took 10.82 / 11.96 seconds. Peak process memory footprint was 4.03 GB; this is not a separate VRAM measurement. No concurrent local build ran during this consumer timing.

The release suite and consumer binaries built successfully; `cargo check --offline --locked --manifest-path webview/Cargo.toml` passed. Existing Objective-C macro warnings and the transitive block crate future-compatibility warning remain. The GUI was not rebuilt or clicked for this change. Core ML remains opt-in until compact packaging, longer-input coverage and listening approval are complete; the old default download is unchanged.

### Tract latency baseline and prewarm

The production tract Metal graph measures about 0.59–0.64 seconds per warm pass on the M2, or roughly 14–16 seconds for 24 passes. The complete cold app check was 31.3 seconds: 16.35 seconds generation plus frontend, graph specialization and codec setup. The webview now prewarms the tract speech and codec graphs in a background thread when the bundle is available, so the first click does not pay graph-specialization cost. It performs no diffusion and synthesizes no audio during startup.

On Stallion, the same graph now runs through Shenava's native tract CUDA fork (`Reza2kn/tract`, `shenava` branch) on the RTX 5080. Warm graph passes were 0.17–0.23 seconds; the raw-text consumer generated 2.03 seconds of audio in 4.35 seconds warm and 5.03 seconds end-to-end. Cold setup was 16.11 seconds. See `stallion-tract-cuda.json`. This is the correct Linux/NVIDIA backend; the generic registry tract crates do not include CUDA. The CUDA provider needs system cuBLAS before the existing Gooya cuDNN directory in `LD_LIBRARY_PATH`. CUDA output still needs the paired Shenava suite before promotion.
