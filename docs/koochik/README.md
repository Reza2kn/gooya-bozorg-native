# Gooya Koochik v2.0-exp in Rust

A separate OmniVoice engine alongside Gooya Bozorg 1.5. Raw Persian text passes
through Negara v7.1, 32-step speech generation and the codec decoder, all in
Rust with tract 0.23.4. No Python or ONNX Runtime participates in Koochik inference.

Source: the merged listener-selected continuation-200 release,
`Reza2kn/Gooya-Koochik-v2.0-exp` at
`537bb48320fd657415eef5b4fb6796b6cebab195`.

## Run

```sh
cargo build --release --no-default-features --manifest-path desktop/Cargo.toml --bin gooya_koochik
./desktop/target/release/gooya_koochik --download ./koochik
./desktop/target/release/gooya_koochik ./koochik output.wav 'سلام، حالت چطوره؟'
```

The private HF repository is `Reza2kn/Gooya-Koochik-v2.0-exp-tract`.
The downloader uses `HF_TOKEN` or the standard Hugging Face token cache,
resolves an immutable commit, and verifies asset checksums. Release availability
is contingent on completing the Shenava gate described below.

The desktop app has a Koochik download button and model picker. Existing bundles
can be selected with `GOOYA_KOOCHIK_MODEL_DIR`, or installed at `koochik-v2` in
the application data directory. Bozorg remains available independently.

## Size and runtime

The compact bundle is 391,072,386 bytes before the small card and evaluation
receipts. Speech matrix weights use native tract Q4_0 in an NNEF graph archived
with Zstandard. Text embeddings are restricted to 3,190 supported rows, with
retained values unchanged. Unsupported text token IDs fail explicitly.
Frontend and codec weights remain FP32.

The default CPU path unpacks the Q4 matrix weights to FP32 in memory to use
faster whole-sequence matrix kernels. This preserves the small download but
requires several GB of RAM; it is not four-bit activation arithmetic. The
packed Q4 path also runs, but was slower on the tested Mac. Packed and unpacked
Q4 paths produced identical final audio codes on the initial canary.

The bundle plus decompressed NNEF cache takes about 0.7 GB disk. The superseded
1.16 GB lossless FP32 bundle is retained only as a local reference.

## Acceptance metric

The user clarified that **>98% parity means transcription parity using Shenava**.
`PARITY_CONTRACT.json` records this contract. The primary score is
`1 - corpus WER` between Shenava transcripts of paired source and compressed
outputs. Both sides use the same recognizer, decoder, normalization and explicit
sampling noise. Expected text and hotwords are never supplied to the recognizer.
Every per-phrase transcript difference and error against the requested text is
retained. Empty transcripts fail the gate.

`parity-suite.json` freezes twelve Persian phrases and their hashes. This is a
finite development test, not a guarantee of 98% perceived quality or accuracy
on unseen text. Subjective pronunciation, voice and prosody review complements
this transcript metric.

Exact internal-code agreement and samplewise waveform cosine are diagnostics,
not the release gate. Earlier INT8/Q4 numerical-screen failures therefore do
not establish poor speech fidelity. See `quantization-status.json` for history.

The FP32 native baseline independently passed all twelve source comparisons
with 100% final audio-code agreement and raw-waveform cosine above 0.99999999.
A fresh lossless load matched the processed source waveform above 0.99999996.
All twelve native frontend/duration/conditioning tensor checks matched exactly.
See `baseline-results.json`.

## Reproduction

Python scripts under `scripts/koochik` are development/export/evaluation tools,
not runtime dependencies. Export used PyTorch 2.11.0, Transformers 5.16.1,
ONNX and the original OmniVoice implementation at commit
`08be0b4ccbac3e13e374e86fbfead4b4cac343e2`.

- `prune_vocabulary.py`: retain supported text embeddings and an ID map.
- `koochik_q4_probe`: native Q4 export and serialized/reloaded evaluation.
  `GOOYA_Q4_DYNAMIC=1 GOOYA_Q4_EXPORT_ONLY=1` exports a general-length graph.
- `package_q4.py`: assemble the one-model native bundle.
- `run_native_suite.py`: replay all 32 generation steps with source noise.
- `watch_shenava_suite.py`: transcribe each source/candidate pair as it completes.
- `write_model_card.py`: require the complete Shenava gate before writing release metadata.

## Current scope

Default synthetic voice, 24 kHz mono, 32 steps and speed 0.85. Sentence boundaries
split generation and insert 350 ms gaps. Phrases estimated above ten seconds
are rejected and should be split. Custom reference recording/voice cloning and
long-form internal model chunking are not implemented.

The app's seed 43 uses portable SplitMix64 noise; it is not equivalent to
PyTorch seed 43. Controlled evaluation replays identical explicit noise tensors.

### Runtime optimization experiments

The production frontend now compiles the ByT5 decoder once per phrase, specializing the source length while retaining a dynamic target length. The tract 0.23.4 `FoldUniformMask` pass is disabled for that graph because it incorrectly removes the five-beam broadcast dimension. All 12 sealed frontend/conditioning comparisons pass exactly; see `dynamic-g2p-results.json`.

The app retains the last speech and codec plans between requests and reports preparation and diffusion progress. Q4 NNEF loading specializes symbolic input dimensions before optimization.

`GOOYA_EXPERIMENTAL_FP16_LINEAR=1` is a rejected experiment: its first paired transcript differed at one of three words. It is never the default. `GOOYA_EXPERIMENTAL_STEPS=16` or `8` evaluates fewer diffusion passes against the original recordings, separately from quantization-only parity. `GOOYA_KOOCHIK_THREADS` controls the thread-count timing probe. No experimental setting is promoted based on speed alone.
