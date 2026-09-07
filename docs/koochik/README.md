# Gooya Koochik v2.0-exp in Rust

This adds a separate OmniVoice engine alongside Gooya Bozorg 1.5. The frontend,
32-step speech generation and codec decoder run on CPU with tract 0.23.4. No
Python, ONNX Runtime or network service participates in Koochik inference.

The source is the merged, listener-selected continuation-200 release:
`Reza2kn/Gooya-Koochik-v2.0-exp` at
`537bb48320fd657415eef5b4fb6796b6cebab195`.

## Run

Build the CLI:

```sh
cargo build --release --no-default-features --manifest-path desktop/Cargo.toml --bin gooya_koochik
./desktop/target/release/gooya_koochik /path/to/bundle output.wav 'سلام، حالت چطوره؟'
```

For the desktop app, set `GOOYA_KOOCHIK_MODEL_DIR` to the bundle directory and
choose **Gooya Koochik v2.0-exp** in the model picker. It is also discovered at
`koochik-v2` within the existing application data directory. Bozorg remains
available when its assets are installed.

## Compression and limitations

The candidate stores the exact FP32 inference weights with lossless Zstandard
compression. This is **download/disk compression, not low-bit inference**.
The speech weight archive is 1,026,546,802 bytes versus 1,225,189,552 bytes for
the source BF16 safetensors. The complete bundle is approximately 1.16 GB.
The first load expands 2,450,309,184 bytes into `bundle/cache`; subsequent loads
verify and reuse that file. Allow at least 4 GB of free disk space for the bundle
and cache. Runtime memory is not reduced to INT8.

Only the default synthetic development voice is included. Custom reference
recording/voice cloning is not implemented in the native engine. Text is split
at sentence punctuation; phrases estimated above ten seconds are rejected with
an instruction to split them. Long-form model chunking is not implemented.
Generation uses 32 steps, speed 0.85 and a portable SplitMix64 seed of 43. A seed
number does not imply identical random noise to PyTorch's generator.

## Validation

`PARITY_CONTRACT.json` freezes the strict per-utterance >98% gate.
`parity-suite.json` contains the twelve development texts and hashes. This is a
finite reproducibility test, not a claim of >98% perceived quality on all texts.
Reference comparisons run the merged source weights as FP32 on CPU. Explicit
source position-noise tensors are replayed in Rust for full 32-step comparisons.

The grouped INT8 candidate failed: only 39.2157% of final code tokens agreed on
the first complete utterance. High logit cosine similarity hid this failure.
It is excluded from the bundle. The lossless candidate must pass all gates
before being described as validated.

Reproduction tools under `scripts/koochik/` export the three neural stages,
capture source outputs and noise, assemble the bundle, and score waveforms.
`koochik_bundle_check` verifies frontend, duration and complete input tensors.
`koochik_parity` accepts a bundle directory (or a diagnostic ONNX file), replays
the captured generation and writes raw/processed WAVs and exact code agreement.
All waveform comparisons use equal sample counts without alignment or time warp.

The exporter requires the pinned original OmniVoice implementation
`k2-fsa/OmniVoice` commit `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`,
PyTorch 2.11.0, Transformers 5.16.1, ONNX and NumPy. These are development tools;
users of the Rust application do not need them.

## Recorded results

See [baseline-results.json](baseline-results.json): all twelve source/tract FP32 generations matched every final audio code, and raw waveform cosine exceeded 0.99999999 on every case. The fresh lossless roundtrip also matched the processed source waveform above 0.99999996 with identical length. The desktop shared engine produced a WAV from raw Persian text, and the UI passed cargo check.

The user rejected the 1.16 GB download. The compact-vocabulary native Q4 canary is 301,326,336 bytes (259,107,996 compressed), projecting a 391,071,418-byte complete download. However, its full-generation code agreement was only 10.7843%, so it is not promoted. This is an exact internal-code score, not an audible-quality percentage. The canary has a concrete input shape; it is not yet a general-input release. The native Q4 path was also slower than FP32 in the observed runs.
