# 🗣️ Gooya Bozorg 1.5 — Native Desktop

Fast, fully-on-device Persian (Farsi) text-to-speech for macOS, Linux, and
Windows. No cloud, no telemetry, no text left your machine. ⚡

**گویا بزرگ — خوانش آفلاین فارسی روی همین دستگاه.**

---

## ✨ Features

- 🗣️ **Real Persian TTS** for the Gooya Bozorg v1.5 voice, straight from the
  grapheme tokenizer (no G2P needed).
- 📱 **Native webview UI** — real RTL text input with full Persian shaping,
  bidi, copy/paste, undo/redo, and native save dialogs.
- ⚡ **Fast local inference** — ONNX Runtime for T3 + flow-step, tract for the
  vocoder graphs. ~12 s for the canonical prompt on Apple Silicon CPU.
- 🧠 **Q4 int4 weights** bundled as a small ~430 MB package.
- ✂️ **Long-text chunking** — automatic word-boundary splits with 160 ms
  pauses, so long paragraphs never overflow the 168-token flow bucket.
- 🔌 **Cross-platform CPU + GPU** — CoreML (macOS), CUDA (Linux), DirectML
  (Windows), CPU everywhere.
- 🖥️ **macOS/Linux/Windows** via `wry` webview shell and `tao` windowing.

---

## 📦 Project layout

| Path | Purpose |
|---|---|
| `webview/` | 🖥️ **Primary desktop app** (wry webview shell, RTL UI, Rust/ORT backend) |
| `desktop/` | Legacy egui app + shared pipeline library (`gooya-native-desktop`) |
| `scripts/` | Export, quantize, fold, and gate tooling |
| `docs/` | Runtime matrix, platform GPU measurements, receipts |
| `model/` | Model card + evarible size notes |

---

## 🚀 Quick start (macOS)

```bash
# 1. Get the model weights (≈430 MB) from Hugging Face Hub
python scripts/download_assets.py

# 2. Build the webview app
cargo build --release --manifest-path webview/Cargo.toml

# 3. Run
./webview/target/release/gooya-native-webview
```

### Linux (CUDA GPU) / Windows (DirectML)

The build target triple picks the GPU execution provider automatically:

```bash
# macOS  → CoreML (Metal)
cargo build --release --manifest-path webview/Cargo.toml

# Linux → CUDA  /  Windows → DirectML
#   GOOYA_DEVICE=gpu selects the provider at runtime.
```

Runtime knob:

```bash
GOOYA_DEVICE=gpu   ./webview/target/release/gooya-native-webview   # GPU (per-OS provider)
GOOYA_DEVICE=auto   # provider, falls back to CPU
./webview/target/release/gooya-native-webview                      # default CPU
```

Additional knobs:

| Env | Meaning |
| --- | --- |
| `GOOYA_DEVICE` | `cpu` (default), `gpu`, `auto` |
| `GOOYA_COREML_UNITS` | CoreML compute units: `1` = CPU+GPU, `2` = ALL/ANE (default) |
| `GOOYA_RUNTIME=tract` | Force tract for T3+flow (tract fallback path) |
| `GOOYA_ORT_THREADS` | ONNX Runtime intra-op threads for flow-step (default 4) |
| `GOOYA_THREADS` | tract ray-on thread cap (default min(8, cores)) |
| `GOOYA_MODEL_DIR` | override the data directory |

---

## 🎹 Keyboard shortcuts

In the input field:

| Keys | Action |
| --- | --- |
| `Enter` | 🗣️ Speak (`بگو`) |
| `Cmd/Ctrl + A` | Select all |
| `Cmd/Ctrl + C` / `X` | Copy / Cut |
| `Cmd/Ctrl + V` | Paste |
| `Cmd/Ctrl + Z` / `Ctrl+Y` | Undo / Redo |

After audio is generated you get `دوباره پخش کن` (replay) and `ذخیره` (save
as WAV via a native dialog) buttons.

---

## 📏 Long-text handling

The b168 flow path can emit at most 168 speech tokens. Probed on the Persian
canonical prompt: a 46-char segment emitted 92 tokens; 92 chars saturated the
bucket. The UI therefore:

- splits normalized text at **word boundaries**,
- keeps intermediate chunks free of forced punctuation,
- synthesizes each chunk separately, and
- stitches them with **160 ms pauses**.

Oversized individual words fall back to token-level splits so a long URL or
run-on word can never overflow the flow bucket.

---

## 📊 Performance

Recorded macOS (Apple Silicon, b168 Q4 bundle, canonical prompt) — raw
synthesis (no UI):

| Path | t3 | flow | vocoder | total |
| --- | --- | --- | --- | --- |
| **CPU Q4 (default)** | ~2.1 s | ~8.4 s | ~1.2 s | **~12 s** |

GPU (CoreML EP) and FP32 variants were benchmarked and are **slower** on this
host; full numbers are in [`docs/RUNTIME_MATRIX.md`](docs/RUNTIME_MATRIX.md).
CUDA/DirectML GPU numbers are pending validation on their target hosts.

---

## 📜 License & credits

- **Model**: Gooya Bozorg 1.5 (Reza2kn) — `CC-BY-NC-4.0`
- **App**: `CC-BY-NC-4.0`
- **Runtime**: ONNX Runtime (`MIT`) · tract (`MIT/Apache-2.0`) · egui (`MIT/Apache-2.0`)
- **Font**: Vazirmatn font (embedded, `SIL OFL`)

This project is non-commercial research/creative code. 🎨

---

## 🙏 Thanks

- [Reza2kn/gooya-bozorg-v1.5](https://huggingface.co/Reza2kn/Gooya-Bozorg-v1.5) — the TTS model
- ONNX Runtime / tract / egui / wry — the runtimes
- You, for reading this far 💛