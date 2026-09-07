use anyhow::{Result, bail};
use gooya_native_desktop::koochik_bundle;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 3 {
        bail!("usage: gooya_koochik BUNDLE_DIR OUTPUT.wav 'Persian text'");
    }
    let report = koochik_bundle::synthesize(Path::new(&a[0]), Path::new(&a[1]), &a[2], 43)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
