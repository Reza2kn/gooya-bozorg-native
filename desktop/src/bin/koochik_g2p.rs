use anyhow::{Context, Result};
use gooya_native_desktop::koochik_frontend::{Frontend, paced};
use std::path::PathBuf;
fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let path = PathBuf::from(a.next().context("frontend directory required")?);
    let text = a.next().context("Persian text required")?;
    let m = Frontend::load(&path)?;
    let phones = m.phones(&text)?;
    println!(
        "{}",
        serde_json::json!({"text":text,"phonemes":phones,"paced":paced(&text,&phones)})
    );
    Ok(())
}
