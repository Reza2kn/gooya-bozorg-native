use anyhow::{Result, bail};
use gooya_native_desktop::koochik::{self, DecodeFixture, Graph};
use std::{fs, path::PathBuf};
use tract_onnx::prelude::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if a.len() != 4 {
        bail!("usage: koochik_parity STEP.onnx DECODER.onnx CASE_DIR OUTPUT_DIR");
    }
    fs::create_dir_all(&a[3])?;
    let inputs = koochik::read_inputs(&a[2].join("step-00/inputs.json"))?;
    let fixture: DecodeFixture = serde_json::from_slice(&fs::read(a[2].join("decode.json"))?)?;
    let step = if a[0].is_dir() {
        gooya_native_desktop::koochik_bundle::prepare(&a[0])?
    } else {
        a[0].clone()
    };
    let graph = Graph::load(&step, &inputs)?;
    let codes = koochik::decode(&graph, inputs, &fixture)?;
    drop(graph);
    let agreement = codes
        .iter()
        .zip(&fixture.source_codes)
        .filter(|(a, b)| a == b)
        .count() as f64
        / codes.len() as f64;
    let inputs = vec![Tensor::from_shape(&[1, 8, fixture.target_tokens], &codes)?];
    let codec = Graph::load(&a[1], &inputs)?;
    let audio = codec.run(&inputs)?;
    let audio = audio[0].to_plain_array_view::<f32>()?;
    koochik::write_wav(&a[3].join("native.wav"), audio.as_slice().unwrap(), 24000)?;
    if a[0].is_dir() {
        let v = gooya_native_desktop::koochik_bundle::voice(&a[0])?;
        let processed =
            gooya_native_desktop::koochik_bundle::postprocess(audio.as_slice().unwrap(), v.rms);
        koochik::write_wav(&a[3].join("processed.wav"), &processed, 24000)?;
    }
    fs::write(a[3].join("codes.json"), serde_json::to_vec(&codes)?)?;
    let report = serde_json::json!({"runtime":"tract-0.23.4","full_32_step_code_agreement":agreement,"passes_code_gate":agreement>0.98,"samples":audio.len(),"promotion":false});
    fs::write(
        a[3].join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
