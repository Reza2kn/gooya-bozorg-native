#!/usr/bin/env bash
# Build an installable macOS .app (+ optional .dmg) for Gooya Bozorg.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Gooya"
IDENTIFIER="app.gooya.native"
DIST="$ROOT/dist"
APP="$DIST/$APP_NAME.app"

echo ":: building webview shell (release)"
cargo build --release --manifest-path "$ROOT/webview/Cargo.toml"

echo ":: assembling $APP_NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$ROOT/webview/target/release/gooya-native-webview" "$APP/Contents/MacOS/$APP_NAME"
# Icon + app resources; the model weights are fetched on first launch, so the
# app stays small.
cp "$ROOT/assets/icon/Gooya.icns" "$APP/Contents/Resources/$APP_NAME.icns"
chmod +x "$APP/Contents/MacOS/$APP_NAME"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>گویا</string>
  <key>CFBundleIdentifier</key><string>$IDENTIFIER</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundleIconFile</key><string>$APP_NAME</string>
  <key>CFBundleVersion</key><string>1.5.0</string>
  <key>CFBundleShortVersionString</key><string>1.5.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.music</string>
  <key>NSHumanReadableCopyright</key><string>CC-BY-NC-4.0 — Gooya Bozorg 1.5</string>
</dict>
</plist>
PLIST

echo "== ad-hoc signing"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true

echo "== dmg"
DMG="$DIST/Gooya-1.5-macOS.dmg"
rm -f "$DMG"
hdiutil create -volname "Gooya 1.5" -srcfolder "$APP" -ov -format UDZO "$DMG" >/dev/null

echo "== done"
du -sh "$APP" "$DMG"
echo "app: $APP"
echo "dmg: $DMG"