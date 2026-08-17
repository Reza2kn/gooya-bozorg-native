//! CLI wrapper around the shared tract pipeline.
//!
//!   tract_pipeline <model_dir> <output.wav> <text tokens...>
//!
//! model_dir must contain t3-prefill.onnx, t3-decode.onnx, the flow graphs
//! (s3-flow-prepare-b168.onnx, s3-flow-step-b168.onnx) and vocoder graphs
//! (s3-vocoder-source-b168.onnx, s3-vocoder-spectral-b168.onnx).

use std::path::PathBuf;

use anyhow::{bail, Result};
use gooya_native_desktop::pipeline;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: tract_pipeline <model_dir> <output.wav> <text tokens...>");
    }
    let dir = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let text_ids: Vec<i64> = args[3..].iter().map(|s| s.parse::<i64>()).collect::<Result<_, _>>()?;

    let report = pipeline::synthesize(&dir, &out, &text_ids)?;
    println!(
        "PIPELINE_OK tokens={} peak={:.4} rms={:.4} dur={:.2}s wrote={}",
        report.tokens.len(),
        report.peak,
        report.rms,
        report.duration_seconds,
        out.display(),
    );
    println!("T3_TOKENS {:?}", &report.tokens[..report.tokens.len().min(8)]);
    Ok(())
}
