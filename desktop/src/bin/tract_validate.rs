use std::path::PathBuf;

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

const SMOKE_IDS: [i64; 18] = [
    1473, 1490, 1456, 1491, 1434, 2, 1467, 1456, 1490, 1464, 2, 1548, 1477, 1459, 1471,
    1493, 1453, 9,
];

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: tract_validate /absolute/path/to/t3-prefill.onnx")?;
    let mut model = tract_onnx::onnx()
        .model_for_path(&path)
        .with_context(|| format!("parse {}", path.display()))?;
    model.set_input_fact(
        0,
        InferenceFact::dt_shape(i64::datum_type(), tvec!(1, SMOKE_IDS.len() as i64)),
    )?;
    let model = model
        .into_optimized()
        .context("optimize graph")?
        .into_runnable()
        .context("build runnable graph")?;
    let input = Tensor::from_shape(&[1, SMOKE_IDS.len()], &SMOKE_IDS)?;
    let outputs = model.run(tvec!(input.into_tvalue()))?;
    let logits = outputs[0].to_plain_array_view::<f32>()?;
    let (top_token, top_logit) = logits
        .iter()
        .take(8194)
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .context("empty logits")?;
    println!(
        "TRACT_Q4_PREFILL_PASS outputs={} top_token={} top_logit={:.6}",
        outputs.len(),
        top_token,
        top_logit
    );
    Ok(())
}
