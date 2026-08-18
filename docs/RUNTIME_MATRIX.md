# Runtime matrix

| Target | UI | Model runtime | Weight path | Minimum target | Release gate |
|---|---|---|---|---|---|
| macOS | SwiftUI | Core ML | W4A16, block 32 | macOS 15 | physical Apple Silicon prediction + audio |
| iPhone | SwiftUI | Core ML | W4A16, block 32 | iOS 18 | physical iPhone prediction + audio |
| iPad | SwiftUI | Core ML | W4A16, block 32 | iPadOS 18 | physical iPad prediction + audio |
| Windows CPU | egui | tract | ONNX MatMulNBits Q4_0 | Windows 10 x64/arm64 | clean-machine synthesis |
| Windows GPU | egui | Burn/WGPU candidate | selective W4/FP16 | Windows 10 | named GPU synthesis + no-fallback profile receipt |
| Linux CPU | egui | tract | ONNX MatMulNBits Q4_0 | glibc 2.31 | clean-machine synthesis |
| Linux GPU | egui | Burn/WGPU candidate | selective W4/FP16 | Vulkan-capable host | named GPU synthesis + no-fallback profile receipt |
| Android | Compose | LiteRT CompiledModel | weight-only INT4 / FP activations | API 23 | physical arm64 synthesis |

## Precision contract

`INT4` means compressed model weights. Activations remain floating point for
Core ML and LiteRT weight-only builds. The desktop tract build requires ONNX
`MatMulNBits` to lower to tract's fused Q4_0 matmul; an ONNX graph containing
explicit dequantize-to-FP32 nodes does not satisfy the CPU Q4 performance gate.

S3Gen is selectively compressed. Large Linear/Conv weights use W4 where the
runtime has a supported kernel and the content gate passes. Normalization,
small tensors, recurrent state, and numerically sensitive flow/vocoder layers
may stay FP16. The manifest records every exception; the app must never label a
mixed artifact as full-int4.

## Profiles

- `efficiency`: deterministic seed, reduced flow steps only after quality
  acceptance, CPU-friendly kernels, bounded speech-token length.
- `balanced`: default release profile and source-equivalent flow step count.
- `gpu`: runtime-selected GPU backend with CPU fallback disabled during the
  benchmark gate so fallback cannot masquerade as GPU success.

## Performance receipt

Each hardware run writes JSON containing source/artifact hashes, OS, CPU/GPU,
runtime version, thread count, input text, seed, generated samples, cold load,
time to first audio, total latency, real-time factor, and peak resident memory.
The receipt also records whether each graph actually ran on the requested
compute unit.

The GPU rows are a target contract, not a claim that Burn supports every T3 or
S3Gen operator today. Unsupported operators or forced CPU fallback block that
profile rather than silently changing its label.

## Quality gate

Conversion is accepted only when source and quantized outputs are compared on
the fixed Persian canonical, conversational, punctuation, numeral, homograph,
and mixed-script prompts. Validation includes waveform sanity, duration, token
completion, and short-window Persian ASR anchor coverage. The final word is a
mandatory anchor.

## Desktop receipt (macOS, tract, b168 Q4)

Recorded from `tract_pipeline` (release) on an Apple Silicon host with the
`gate-fullq4-b168-v1` bundle, canonical prompt "سلام، حالت چطوره؟" (17 token
ids incl. trailing `9`), audio 1.48 s / 35520 samples.

- Artifact: `gate-fullq4-b168-v1` (T3 prefill/decode, flow prepare, flow step,
  vocoder source/spectral, all Q4_0 except numerically sensitive layers).
- Wall time: 301.95 s (user 298.43) for 1.48 s audio → RTF ≈ 204.
- Peak resident set: 1.14 GB.
- Output: peak 0.9900, rms 0.1729; tokens 37 (T3 Q4 near-tie drift vs 36-token
  ORT reference; first 8 tokens match reference exactly; audio ASR PASS).
- Input-tokenization and spectral/vocoder numerics verified against torch/ORT
  (istft max diff 1.4e-5, mag/phase vs ORT within 1.1e-3).

## Desktop receipt (macOS, tract, b168 Q4) — after constant-folding

The `flow.load.prepare` hotspot (224.6 s) is closed by baking the 10
constant-activation `MatMulNBits` (6 `encoder/encoders.*/self_attn/linear_pos`
+ 4 `up_encoders.*/linear_pos`) into f32 constants with
`scripts/fold_constant_matmuls.py` (ORT-kernel exact, max abs delta 0.0 on
mu/mask/speaker/cond; two-canary detection for token-independent activations).
`pipeline.rs` prefers `s3-flow-prepare-b{bucket}.folded.onnx` when present.

- `flow.load.prepare`: 224.6 s → **~0.3-0.5 s**
- `flow.load.step`: ~1.8-2.0 s (unchanged)
- `t3`: ~2.1 s, `flow.step` 10 × ~0.8 s = **~8.4 s**, `vocoder`: ~1.2 s
- Total ≈ **12 s** with the default hybrid runtime; canonical output has peak
  0.9900, rms 0.1784, tokens 36, and ASR PASS.

The desktop runtime uses ONNX Runtime 1.28 for T3 and flow-step, with 8 and 4
intra-op threads respectively. `GOOYA_ORT_THREADS` overrides the flow setting.
The prepare and vocoder graphs remain on tract because they are already faster
locally. Set `GOOYA_RUNTIME=tract` and `GOOYA_FLOW_RUNTIME=tract` for the full
tract fallback; it uses tract's Rayon `multithread-mm` executor, capped at 8
threads by default, with `GOOYA_THREADS` as its override.

## Desktop platforms

The default `cargo build --release` is the verified CPU build and is source-
portable across macOS, Linux, and Windows. The UI is a webview shell (egui
legacy under `apps/gooya-native/desktop`, active shell under
`apps/gooya-native/webview`). Inference uses ONNX Runtime plus tract for the
prepare/vocoder graphs. The ONNX Runtime execution provider is chosen by the
build target triple and requested at runtime:

- macOS: CoreML (Metal/ANE) via `GOOYA_DEVICE=gpu`; `GOOYA_COREML_UNITS`
  selects compute units (1 = CPU+GPU, 2 = ALL incl. ANE, default 2).
- Linux: CUDA via `GOOYA_DEVICE=gpu`.
- Windows: DirectML via `GOOYA_DEVICE=gpu`.
- `GOOYA_DEVICE=auto` attempts the compiled platform provider and otherwise
  falls back to CPU; the default is `cpu`.

### Measured macOS timings (canonical, b168, Apple Silicon)

| Path | t3 | flow (10 steps) | total |
|---|---|---|---|
| ORT CPU + Q4 weights (default) | ~2.1 s | ~8.4 s | ~12 s |
| ORT CoreML EP + Q4 weights | ~9.2 s | ~47.6 s | ~58 s |
| ORT CoreML EP + FP32 weights, CPU+GPU | ~8.0 s | ~12.3 s | ~22 s |
| ORT CoreML EP + FP32 weights, ALL/ANE | ~4.7 s | ~13.1 s | ~19 s |
| ORT CPU + FP32 weights | ~5.7 s | ~13.4 s | ~20 s |

The Q4 `MatMulNBits` ops are unsupported by the CoreML execution provider and
fall back to CPU; Core ML therefore cannot beat the CPU Q4 path on this
Apple Silicon host. Direct `.mlpackage` compilation is currently blocked by
coremltools 9 dropping ONNX as a conversion source (would require an
ONNX→PyTorch hop). GPU numbers for CUDA (Linux) and DirectML (Windows) still
require validation on their target hosts.

Linux/Windows builds and physical GPU execution still require validation on
their target hosts; the current verified hardware receipt is macOS CPU.

### Measured Linux CUDA timings (RTX 5080, Blackwell sm_120, b168)

The stock `ort` binaries (ort.pyke.io) ship **no sm_120 kernels**, so CUDA
silently fell back to CPU on RTX 50-series. Using the official
`onnxruntime-gpu` wheel (which has Blackwell kernels) via
`scripts/cuda_setup.sh`:

| Path | t3 | flow (10 steps) | vocoder | total |
|---|---|---|---|---|
| ORT CPU + Q4 | ~1.88 s | ~7.6 s | ~1.96 s | ~11.5 s |
| ORT CUDA + Q4, tract vocoder | ~1.49 s | ~1.34 s | ~2.04 s | ~4.9 s |
| **ORT CUDA + Q4, vocoder on CUDA** | **~1.54 s** | **~1.34 s** | **~0.31 s** | **~3.2 s** |

CUDA is **~3.6× faster overall**: flow ~5.7× and the vocoder ~6.6×
(convs → cuDNN, matmuls → cuBLAS on the 5080). T3 emits byte-identical
tokens; mel drift vs CPU is ≤0.018 (fp32 rounding) and ASR confirms identical
output.

TensorRT EP was also wired (TRT → CUDA → CPU automatic fallback) and tested.
On this model it produced no speedup and no TRT engines — the dynamic-shape
graphs and Q4 `MatMulNBits` ops do not partition onto TensorRT, so the EP
steps down to CUDA. CUDA remains the fastest path (~3.2 s).

### Cross-platform GUI status

The Linux (webkit2gtk/tao) and Windows (WebView2) webview shells compile and
the full pipeline runs on CUDA (verified headless on the RTX 5080). Visual
launch of the GUI must be validated on the physical desktop sessions (an SSH
shell has no access to the X/Wayland display). Build the installable packages
per OS:

- macOS: `scripts/bundle_app.sh` → `.app`/`.dmg`
- Linux: `scripts/package_linux.sh` → `.tar.xz`
- Windows: `powershell -File scripts/package_windows.ps1` → `.zip`

`scripts/cuda_setup.sh` wires the official ONNX Runtime (Blackwell sm_120
kernels) into Linux CUDA builds. GitHub Actions
(`.github/workflows/release.yml`) builds all three on a `v*` tag and attaches
them to the release.

## Long-text inference

The b168 flow path can emit at most 168 speech tokens. Probing showed a
46-character Persian word-boundary sample emitted 92 speech tokens, while a
92-character version saturated the bucket. The desktop UI therefore chunks
normalized text at word boundaries using a 64-token grapheme-BPE budget,
preferring existing sentence punctuation and never adding punctuation to an
intermediate chunk. Chunks are synthesized independently and joined with a
160 ms silence. Oversized individual words fall back to tokenizer-token splits
instead of overflowing the flow bucket.

This is the CPU Q4 baseline. The tract flow-step latency (~5 min/prompt) is the
open perf item vs the ORT (~14 s) reference; UI runs the job on a background
thread so the window stays responsive.
