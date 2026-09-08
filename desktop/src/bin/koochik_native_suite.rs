//! Run a frozen suite while retaining the native speech model between cases.
use anyhow::{Context, Result, ensure};
use gooya_native_desktop::{
    koochik::{self, DecodeFixture, Graph},
    koochik_bundle,
};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};
use tract_onnx::prelude::*;
fn main() -> Result<()> {
    let a: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(
        a.len() == 4,
        "usage: koochik_native_suite BUNDLE CASES SUITE.json OUTPUT"
    );
    fs::create_dir_all(&a[3])?;
    let suite: serde_json::Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let cases = suite["texts"].as_array().context("missing cases")?;
    let manifest: koochik_bundle::Manifest =
        serde_json::from_slice(&fs::read(a[0].join("manifest.json"))?)?;
    let step = koochik_bundle::prepare(&a[0])?;
    let initial = koochik::read_inputs(
        &a[1]
            .join(cases[0]["id"].as_str().context("case id")?)
            .join("step-00/inputs.json"),
    )?;
    let start = Instant::now();
    let graph = Graph::load(&step, &initial)?;
    ensure!(
        matches!(graph.backend(), "coreml" | "tract-cuda"),
        "this suite runner requires Core ML or tract-cuda"
    );
    let load_seconds = start.elapsed().as_secs_f64();
    let voice = koochik_bundle::voice(&a[0])?;
    let binary_hash = format!("{:x}", Sha256::digest(fs::read(std::env::current_exe()?)?));
    for item in cases {
        let id = item["id"].as_str().context("case id")?;
        let case = a[1].join(id);
        let out = a[3].join(id);
        fs::create_dir_all(&out)?;
        let inputs = koochik::read_inputs(&case.join("step-00/inputs.json"))?;
        let mut fixture: DecodeFixture =
            serde_json::from_slice(&fs::read(case.join("decode.json"))?)?;
        fixture.num_steps = manifest.inference_steps;
        let start = Instant::now();
        let codes = koochik::decode(&graph, inputs, &fixture)?;
        let generation_seconds = start.elapsed().as_secs_f64();
        let inputs = vec![Tensor::from_shape(&[1, 8, fixture.target_tokens], &codes)?];
        let codec = Graph::load(&a[0].join("decoder.onnx"), &inputs)?;
        let audio = codec.run(&inputs)?;
        let audio = audio[0].to_plain_array_view::<f32>()?;
        let raw = audio.as_slice().context("noncontiguous audio")?;
        koochik::write_wav(&out.join("native.wav"), raw, 24000)?;
        let processed = koochik_bundle::postprocess(raw, voice.rms);
        koochik::write_wav(&out.join("processed.wav"), &processed, 24000)?;
        fs::write(out.join("codes.json"), serde_json::to_vec(&codes)?)?;
        let report = serde_json::json!({"id":id,"backend":graph.backend(),"runtime":"native Rust speech, Rust sampler, tract CPU codec","steps":fixture.num_steps,"coreml_mixed":std::env::var_os("GOOYA_KOOCHIK_COREML_MIXED").is_some(),"shared_model_load_seconds":load_seconds,"generation_seconds":generation_seconds,"audio_seconds":processed.len() as f64/24000.,"binary_sha256":binary_hash,"coreml_fp32":std::env::var("GOOYA_KOOCHIK_COREML_FP32").ok(),"coreml_fp16":std::env::var("GOOYA_KOOCHIK_COREML_FP16").ok(),"promotion":false});
        fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
    }
    Ok(())
}
