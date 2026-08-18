#!/usr/bin/env bash
# Build a distributable Linux package for Gooya Bozorg:
#   dist/Gooya-Linux-x86_64.tar.xz
# containing a self-contained Gooya/ folder with the binary, the model data,
# the font, and an install.sh that adds a desktop launcher.
#
# System deps (Debian/Ubuntu): build-essential libssl-dev cmake
#   libgtk-3-dev libwebkit2gtk-4.1-dev libsoup-3.0-dev libx11-dev
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
PKG="$DIST/Gooya"

echo ":: fetching model assets (if missing)"
test -f "$ROOT/desktop/data/tract-bundle-b168/t3-prefill.onnx" || \
  cargo run --release --no-default-features --manifest-path "$ROOT/desktop/Cargo.toml" --bin gooya-fetch-assets

echo ":: building webview shell"
cargo build --release --manifest-path "$ROOT/webview/Cargo.toml"

echo ":: assembling package"
rm -rf "$PKG"
mkdir -p "$PKG/bin" "$PKG/lib" "$PKG/data" "$PKG/share"

cp "$ROOT/webview/target/release/gooya-native-webview" "$PKG/bin/Gooya"
# ship the ONNX Runtime providers alongside the binary (when dynamically linked)
cp "$ROOT/webview/target/release/"libonnxruntime*.so* "$PKG/lib/" 2>/dev/null || true
cp -R "$ROOT/desktop/data/." "$PKG/data/"

cat > "$PKG/bin/Gooya.sh" <<SH
#!/usr/bin/env bash
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
export GOOYA_MODEL_DIR="\$HERE/data"
export LD_LIBRARY_PATH="\$HERE/lib:\${LD_LIBRARY_PATH:-}"
"\$HERE/bin/Gooya" "\$@"
SH
chmod +x "$PKG/bin/Gooya.sh" "$PKG/bin/Gooya"

cat > "$PKG/install.sh" <<SH
#!/usr/bin/env bash
set -e
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "\$HOME/.local/bin" "\$HOME/.local/share/applications" "\$HOME/.local/share/icons"
ln -sf "\$HERE/bin/Gooya.sh" "\$HOME/.local/bin/gooya"
cat > "\$HOME/.local/share/applications/gooya.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=گویا (Gooya Bozorg)
Comment=Offline Persian TTS
Exec=\$HERE/bin/Gooya.sh
Icon=\$HOME/.local/share/icons/gooya
Categories=AudioVideo;
Terminal=false
EOF
echo "installed. run 'gooya' or launch from your app menu."
SH
chmod +x "$PKG/install.sh"

echo ":: packaging"
TARBALL="$DIST/Gooya-Linux-x86_64.tar.xz"
rm -f "$TARBALL"
(cd "$DIST" && tar -cJf "$TARBALL" Gooya)
echo ":: done"
du -sh "$PKG" "$TARBALL"
echo "tarball: $TARBALL"