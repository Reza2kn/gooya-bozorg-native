//! Fixed-input timing only. This is not a speech-quality or promotion test.
use anyhow::{Result, ensure};
use gooya_native_desktop::{koochik, koochik_bundle};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};
fn main() -> Result<()> {
    let _ = env_logger::try_init();
    let a: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(a.len() == 2, "usage: koochik_benchmark BUNDLE INPUTS.json");
    let inputs = koochik::read_inputs(&a[1])?;
    let start = Instant::now();
    let path = koochik_bundle::prepare(&a[0])?;
    let graph = koochik::Graph::load(&path, &inputs)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let mut runs = vec![];
    let mut hashes = vec![];
    for _ in 0..4 {
        let start = Instant::now();
        let output = graph.run(&inputs)?;
        ensure!(output.len() == 1, "unexpected output count");
        runs.push(start.elapsed().as_secs_f64());
        let mut hash = Sha256::new();
        for &value in output[0].to_plain_array_view::<f32>()?.iter() {
            hash.update(value.to_le_bytes());
        }
        hashes.push(format!("{:x}", hash.finalize()));
    }
    println!(
        "{}",
        serde_json::json!({"backend":graph.backend(),"output_sha256":hashes,"load_seconds":load_seconds,"run_seconds":runs,"threads":std::env::var("GOOYA_KOOCHIK_THREADS").unwrap_or("6".into()),"fp16_linear":std::env::var_os("GOOYA_EXPERIMENTAL_FP16_LINEAR").is_some()})
    );
    Ok(())
}
