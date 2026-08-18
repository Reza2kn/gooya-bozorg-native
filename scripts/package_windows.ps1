# PowerShell: build a distributable Windows package for Gooya Bozorg.
#   dist\Gooya-Windows-x86_64.zip
# containing a self-contained Gooya/ folder with the .exe, model data, font,
# and a launcher. Requires: Rust toolchain, and ONNX Runtime DirectML support
# (the app's Windows build links ort with the directml feature automatically).
#
# Note: build on a Windows machine; WebView2 is preinstalled on Win10/11.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dist = Join-Path $Root "dist"
$Pkg = Join-Path $Dist "Gooya"

Write-Host ":: building webview shell"
cargo build --release --manifest-path (Join-Path $Root "webview\Cargo.toml")

Write-Host ":: assembling package"
if (Test-Path $Pkg) { Remove-Item -Recurse -Force $Pkg }
New-Item -ItemType Directory -Force -Path (Join-Path $Pkg "bin") | Out-Null

Copy-Item (Join-Path $Root "webview\target\release\gooya-native-webview.exe") (Join-Path $Pkg "bin\Gooya.exe")
Copy-Item (Join-Path $Root "webview\target\release\*.dll") (Join-Path $Pkg "bin\") -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Root "assets\icon\Gooya.ico") (Join-Path $Pkg "bin\Gooya.ico")

Write-Host ":: packaging"
$Zip = Join-Path $Dist "Gooya-Windows-x86_64.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path $Pkg -DestinationPath $Zip
Write-Host "zip: $Zip"