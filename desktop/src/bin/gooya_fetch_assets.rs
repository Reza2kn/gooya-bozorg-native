//! Pure-Rust asset fetcher: downloads the Gooya Bozorg 1.5 native bundle from
//! Hugging Face Hub into `desktop/data/`. Uses the system `curl` binary, so no
//! Python or extra HTTP dependencies are needed.

use anyhow::{bail, Result};
use std::path::PathBuf;
use std::process::Command;

const HF_REPO: &str = "https://huggingface.co/Reza2kn/gooya-bozorg-v1.5-native/resolve/main";

const FILES: &[(&str, &str)] = &[
    (
        "grapheme_mtl_merged_expanded_v1.json",
        "grapheme_mtl_merged_expanded_v1.json",
    ),
    ("tract-bundle-b168/t3-prefill.onnx", "tract-bundle-b168/t3-prefill.onnx"),
    ("tract-bundle-b168/t3-decode.onnx", "tract-bundle-b168/t3-decode.onnx"),
    ("tract-bundle-b168/t3-q4-shared.data", "tract-bundle-b168/t3-q4-shared.data"),
    (
        "tract-bundle-b168/s3-flow-prepare-b168.onnx",
        "tract-bundle-b168/s3-flow-prepare-b168.onnx",
    ),
    (
        "tract-bundle-b168/s3-flow-prepare-b168.onnx.data",
        "tract-bundle-b168/s3-flow-prepare-b168.onnx.data",
    ),
    (
        "tract-bundle-b168/s3-flow-prepare-b168.folded.onnx",
        "tract-bundle-b168/s3-flow-prepare-b168.folded.onnx",
    ),
    (
        "tract-bundle-b168/s3-flow-prepare-b168.folded.onnx.data",
        "tract-bundle-b168/s3-flow-prepare-b168.folded.onnx.data",
    ),
    (
        "tract-bundle-b168/s3-flow-step-b168.onnx",
        "tract-bundle-b168/s3-flow-step-b168.onnx",
    ),
    (
        "tract-bundle-b168/s3-flow-step-b168.onnx.data",
        "tract-bundle-b168/s3-flow-step-b168.onnx.data",
    ),
    (
        "tract-bundle-b168/s3-vocoder-source-b168.onnx",
        "tract-bundle-b168/s3-vocoder-source-b168.onnx",
    ),
    (
        "tract-bundle-b168/s3-vocoder-source-b168.onnx.data",
        "tract-bundle-b168/s3-vocoder-source-b168.onnx.data",
    ),
    (
        "tract-bundle-b168/s3-vocoder-spectral-b168.onnx",
        "tract-bundle-b168/s3-vocoder-spectral-b168.onnx",
    ),
    (
        "tract-bundle-b168/s3-vocoder-spectral-b168.onnx.data",
        "tract-bundle-b168/s3-vocoder-spectral-b168.onnx.data",
    ),
];

fn main() -> Result<()> {
    let dest = std::env::var("GOOYA_MODEL_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")))
        .expect("data dir");

    for (remote, local) in FILES {
        let out = dest.join(local);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let url = format!("{HF_REPO}/{remote}");
        println!(":: {} ...", local);
        let status = Command::new("curl")
            .args(["-fsSL", "--retry", "3", "-o"])
            .arg(&out)
            .arg(&url)
            .status()?;
        if !status.success() {
            bail!("failed to download {remote}");
        }
    }

    let bundle = dest.join("tract-bundle-b168");
    if !bundle.join("t3-prefill.onnx").exists() {
        bail!("incomplete download: missing t3-prefill.onnx");
    }
    let bytes = duplicate_bytes(&dest)?;
    println!("done: {bytes:.1} MB of assets in {}", dest.display());
    Ok(())
}

fn duplicate_bytes(dir: &PathBuf) -> Result<f64> {
    let mut total = 0u64;
    for entry in walkdir(dir) {
        if entry.is_file() {
            total += entry.metadata()?.len();
        }
    }
    Ok(total as f64 / (1024.0 * 1024.0))
}

fn walkdir(dir: &PathBuf) -> Vec<PathBuf> {
    fn walk(dir: &PathBuf, acc: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, acc);
                } else {
                    acc.push(p);
                }
            }
        }
    }
    let mut acc = Vec::new();
    walk(dir, &mut acc);
    acc
}
