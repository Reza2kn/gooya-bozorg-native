//! Raw-text cold/warm consumer check, including real WAV identity and cache reuse.
use anyhow::{ensure, Result};
use gooya_native_desktop::koochik_bundle;
use std::{fs, path::PathBuf};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(a.len() == 3, "usage: koochik_consumer_check BUNDLE OUT_DIR TEXT");
    let root=PathBuf::from(&a[0]);let out=PathBuf::from(&a[1]);fs::create_dir_all(&out)?;
    let cold=koochik_bundle::synthesize(&root,&out.join("cold.wav"),&a[2],43)?;
    let warm=koochik_bundle::synthesize(&root,&out.join("warm.wav"),&a[2],43)?;
    ensure!(cold.loaded_speech_graph && !warm.loaded_speech_graph, "speech graph was not reused");
    ensure!(fs::read(&cold.wav_path)? == fs::read(&warm.wav_path)?, "cached synthesis changed the output");
    let report=serde_json::json!({"cold":cold,"warm":warm,"identical_wav":true,"cache_reused":true});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");Ok(())
}
