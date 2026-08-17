#!/usr/bin/env bash
# CUDA setup for the Gooya Bozorg native runtime on NVIDIA Blackwell (sm_120)
# GPUs such as the RTX 50-series (RTX 5080 / "stallion").
#
# Why this exists: the ONNX Runtime binaries that `ort` downloads from
# ort.pyke.io do not ship sm_120 CUDA kernels, so CUDA silently falls back to
# CPU on Blackwell. The official `onnxruntime-gpu` pip wheel DOES include
# Blackwell kernels; this script wires that library into the Rust build via
# ORT_LIB_PATH + dynamic linking, plus the CUDA 13 runtime libs and cuDNN 9.
#
# Run on the Linux CUDA host as the unprivileged user (no sudo required):
#   bash scripts/cuda_setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ":: checking GPU arch"
if ! nvidia-smi >/dev/null 2>&1; then echo "no nvidia-smi"; exit 1; fi
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
echo "   compute capability: $arch"
if [[ "$arch" != "120" ]]; then
  echo "   not sm_120; the stock ort build may already work. Set GOOYA_DEVICE=gpu to test."
fi

echo ":: assembling CUDA 13 runtime libs (no root)"
CUDA_LIBS="$HOME/cuda13-libs"
mkdir -p "$CUDA_LIBS"
(cd "$CUDA_LIBS"
  for pkg in libcublas-13-1 cuda-cudart-13-1; do
    if ! ls ./*.deb >/dev/null 2>&1; then :; fi
    apt-get download "$pkg" >/dev/null 2>&1 || true
  done
  for deb in ./*.deb; do dpkg-deb -x "$deb" extracted 2>/dev/null || true; done
)
CUDALIBDIR=$(find "$CUDA_LIBS/extracted" -path "*targets/x86_64-linux/lib" 2>/dev/null | head -1)
echo "   cuda libs: $CUDALIBDIR"

echo ":: installing official ORT GPU + cuDNN via pip"
pip3 install --user --break-system-packages onnxruntime-gpu nvidia-cudnn-cu12 onnx >/dev/null 2>&1

echo ":: wiring official ORT into ort-sys"
CAPI=$(python3 -c "import onnxruntime,os; print(os.path.join(os.path.dirname(onnxruntime.__file__),'capi'))")
ORTLIB="$HOME/ortlib"
rm -rf "$ORTLIB"; mkdir -p "$ORTLIB"; cd "$ORTLIB"
VER=$(ls "$CAPI"/libonnxruntime.so.* | head -1)
ln -s "$VER" "$(basename "$VER")"
ln -s "$(basename "$VER")" libonnxruntime.so.1
ln -s "$(basename "$VER")" libonnxruntime.so
for f in libonnxruntime_providers_cuda.so libonnxruntime_providers_shared.so; do
  ln -s "$CAPI/$f" "$f"
done

echo ":: rebuilding with official ORT (dynamic)"
cd "$ROOT"
cargo clean --manifest-path desktop/Cargo.toml >/dev/null 2>&1 || true
ORT_LIB_PATH="$ORTLIB" ORT_SKIP_DOWNLOAD=1 ORT_PREFER_DYNAMIC_LINK=1 \
  cargo build --release --no-default-features --manifest-path desktop/Cargo.toml --bin tract_pipeline

SP=$(python3 -c "import nvidia.cudnn,os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__),'..'))" 2>/dev/null || echo /home/rezo/.local/lib/python3.14/site-packages/nvidia)
cat <<EOF

:: done. Run your benchmarks with:
export LD_LIBRARY_PATH=$SP/cudnn/lib:$SP/cublas/lib:$ORTLIB:$CUDALIBDIR:\$PWD/desktop/target/release
GOOYA_DEVICE=gpu GOOYA_PROFILE=1 ./desktop/target/release/tract_pipeline desktop/data/tract-bundle-b168 /tmp/out.wav 1473 1490 1456 1491 1434 2 1467 1456 1490 1464 2 1548 1477 1459 1471 1493 1453 9
EOF