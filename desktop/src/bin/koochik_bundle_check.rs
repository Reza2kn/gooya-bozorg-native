//! Compare the raw-text native frontend and conditioning against captured source inputs.
use anyhow::{Result, ensure};
use gooya_native_desktop::{koochik, koochik_bundle, koochik_frontend};
use std::{fs, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    ensure!(a.len() == 2, "usage: koochik_bundle_check BUNDLE CASE_DIR");
    let root = Path::new(&a[0]);
    let case = Path::new(&a[1]);
    let task: serde_json::Value = serde_json::from_slice(&fs::read(case.join("task.json"))?)?;
    let text = task["text"].as_str().unwrap();
    let front = koochik_frontend::Frontend::load(&root.join("frontend"))?;
    let paced = koochik_frontend::paced(text, &front.phones(text)?);
    ensure!(
        paced == task["paced_text"].as_str().unwrap(),
        "frontend mismatch: {paced}"
    );
    let (actual, t) = koochik_bundle::inputs(root, &paced)?;
    ensure!(
        t == task["target_tokens"].as_u64().unwrap() as usize,
        "duration mismatch {t}"
    );
    let expected = koochik::read_inputs(&case.join("step-00/inputs.json"))?;
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        ensure!(a == b, "conditioning mismatch at input {i}");
    }
    println!(
        "Exact native frontend, duration and conditioning: {}",
        case.display()
    );
    Ok(())
}
