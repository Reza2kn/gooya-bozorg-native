# 🗣️ Gooya Bozorg 1.5 — Native Runtime Bundle

Fully on-device Persian (Farsi) TTS runtime assets for the
[**Gooya Bozorg 1.5**](https://huggingface.co/Reza2kn/Gooya-Bozorg-v1.5) voice.

| | |
| --- | --- |
| 🧠 Runtime | ONNX Runtime (T3 + flow-step) · tract (vocoder) |
| ⚖️ Weights | Q4 int4 (`MatMulNBits`, block 32), ~430 MB |
| 🪣 Bucket | b168 (≤168 speech tokens) |
| 🍎 Platform | macOS / Linux / Windows (CPU + per-OS GPU) |
| 🎯 Latency | ~12 s canonical prompt · Apple Silicon CPU |

---

## 📦 Contents

```
tract-bundle-b168/
├── t3-prefill.onnx / t3-decode.onnx        # speech-token transformer (Q4)
├── s3-flow-prepare-b168.onnx                # flow conditioning (folded)
├── s3-flow-step-b168.onnx                   # flow matching denoiser (Q4)
├── s3-vocoder-source-b168.onnx              # HiFT excitation (Q4)
├── s3-vocoder-spectral-b168.onnx           # spectral head (Q4)
└── *.data                                  # external weights
grapheme_mtl_merged_expanded_v1.json          # grapheme BPE tokenizer
```

## 🚀 Use it

```bash
# pure Rust asset fetcher (no Python needed)
cargo run --release --manifest-path desktop/Cargo.toml --bin gooya-fetch-assets
cargo build --release --manifest-path webview/Cargo.toml
./webview/target/release/gooya-native-webview

# or build the installable macOS app
./scripts/bundle_app.sh
```

Full docs: [github.com/Reza2kn/gooya-bozorg-native](https://github.com/Reza2kn/gooya-bozorg-native)

## 📜 License

- Model: Reza2kn / Gooya Bozorg 1.5 — **CC-BY-NC-4.0**
- Runtime bundles: **CC-BY-NC-4.0**

Non-commercial research/creative use only. 🎨