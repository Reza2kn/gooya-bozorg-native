//! Tract-only OmniVoice diffusion decoding. Artifact promotion is a separate parity gate.
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::{fs, path::Path, time::Instant};
use tract_onnx::prelude::*;

#[derive(Deserialize)]
pub struct Input {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: serde_json::Value,
}
pub fn read_inputs(path: &Path) -> Result<Vec<Tensor>> {
    let inputs: Vec<Input> = serde_json::from_slice(&fs::read(path)?)?;
    inputs
        .into_iter()
        .map(|x| -> Result<Tensor> {
            Ok(match x.dtype.as_str() {
                "int64" => {
                    Tensor::from_shape(&x.shape, &serde_json::from_value::<Vec<i64>>(x.data)?)?
                }
                "float32" => {
                    Tensor::from_shape(&x.shape, &serde_json::from_value::<Vec<f32>>(x.data)?)?
                }
                "bool" => {
                    Tensor::from_shape(&x.shape, &serde_json::from_value::<Vec<bool>>(x.data)?)?
                }
                d => bail!("unsupported input type {d}"),
            })
        })
        .collect()
}
pub struct Graph {
    plan: std::sync::Arc<TypedRunnableModel>,
}
impl Graph {
    pub fn from_plan(plan: std::sync::Arc<TypedRunnableModel>) -> Self {
        Self { plan }
    }
    pub fn load(path: &Path, inputs: &[Tensor]) -> Result<Self> {
        let mut m = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("load {}", path.display()))?;
        for (i, t) in inputs.iter().enumerate() {
            m.set_input_fact(i, InferenceFact::dt_shape(t.datum_type(), t.shape()))?;
        }
        let options = tract_core::runtime::RunOptions {
            executor: Some(tract_linalg::multithread::Executor::multithread(6)),
            ..Default::default()
        };
        Ok(Self {
            plan: m.into_optimized()?.into_runnable_with_options(&options)?,
        })
    }
    pub fn load_dynamic(path: &Path) -> Result<Self> {
        let m = tract_onnx::onnx().model_for_path(path)?;
        Ok(Self {
            plan: m.into_optimized()?.into_runnable()?,
        })
    }
    pub fn run(&self, inputs: &[Tensor]) -> Result<TVec<TValue>> {
        self.plan
            .run(inputs.iter().cloned().map(|x| x.into_tvalue()).collect())
    }
}
#[derive(Deserialize)]
pub struct DecodeFixture {
    pub target_tokens: usize,
    pub noise: Vec<Vec<f32>>,
    pub source_codes: Vec<i64>,
}
pub fn log_normalizer(x: &[f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + x.iter().map(|v| (*v - max).exp()).sum::<f32>().ln()
}
/// Uses the source implementation's explicit per-step position noise for reproducible comparisons.
pub fn decode(graph: &Graph, mut inputs: Vec<Tensor>, fixture: &DecodeFixture) -> Result<Vec<i64>> {
    let t = fixture.target_tokens;
    let s = inputs[0].shape()[2];
    ensure!(t > 0 && t <= s, "invalid target length");
    ensure!(fixture.noise.len() == 32, "expected 32 noise steps");
    let n = 8 * t;
    let mut tokens = vec![1024i64; n];
    let mut remaining = n;
    for step in 0..32 {
        let start = Instant::now();
        let outputs = graph.run(&inputs)?;
        let raw = outputs[0].to_plain_array_view::<f32>()?;
        ensure!(
            raw.shape() == [2, 8, s, 1025],
            "unexpected speech logit shape"
        );
        let logits = raw.as_slice().context("noncontiguous logits")?;
        let mut predicted = vec![0i64; n];
        let mut scores = vec![f32::NEG_INFINITY; n];
        ensure!(fixture.noise[step].len() == n, "noise size mismatch");
        for c in 0..8 {
            for j in 0..t {
                let idx = c * t + j;
                if tokens[idx] != 1024 {
                    continue;
                }
                let ci = (c * s + s - t + j) * 1025;
                let ui = ((8 + c) * s + j) * 1025;
                let cond = &logits[ci..ci + 1025];
                let uncond = &logits[ui..ui + 1025];
                let cn = log_normalizer(cond);
                let un = log_normalizer(uncond);
                let mixed: Vec<f32> = cond
                    .iter()
                    .zip(uncond)
                    .map(|(c, u)| {
                        let cp = *c - cn;
                        cp + 2.0 * (cp - (*u - un))
                    })
                    .collect();
                let z = log_normalizer(&mixed);
                let (token, value) = mixed[..1024]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .context("empty vocabulary")?;
                predicted[idx] = token as i64;
                scores[idx] = (*value - z - c as f32 * 5.0) / 5.0 + fixture.noise[step][idx];
            }
        }
        let timestep = |i: usize| {
            let x = i as f32 / 32.;
            0.1 * x / (1. + (0.1 - 1.) * x)
        };
        let k = if step == 31 {
            remaining
        } else {
            ((n as f64 * (timestep(step + 1) as f64 - timestep(step) as f64)).ceil() as usize)
                .min(remaining)
        };
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|a, b| scores[*b].total_cmp(&scores[*a]));
        for &i in indices.iter().take(k) {
            ensure!(
                tokens[i] == 1024,
                "schedule selected an already filled token"
            );
            tokens[i] = predicted[i];
        }
        remaining -= k;
        let mut view = inputs[0].to_plain_array_view_mut::<i64>()?;
        let ids = view.as_slice_mut().context("noncontiguous input IDs")?;
        for c in 0..8 {
            for j in 0..t {
                ids[c * s + s - t + j] = tokens[c * t + j];
                ids[(8 + c) * s + j] = tokens[c * t + j];
            }
        }
        eprintln!(
            "Koochik step {}/32 ({:.2}s)",
            step + 1,
            start.elapsed().as_secs_f32()
        );
    }
    ensure!(
        remaining == 0 && !tokens.contains(&1024),
        "incomplete diffusion output"
    );
    Ok(tokens)
}
pub fn write_wav(path: &Path, audio: &[f32], rate: u32) -> Result<()> {
    use std::io::Write;
    ensure!(
        !audio.is_empty() && audio.iter().all(|x| x.is_finite()),
        "invalid waveform"
    );
    let bytes = (audio.len() * 2) as u32;
    let mut f = fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + bytes).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&bytes.to_le_bytes())?;
    for &x in audio {
        f.write_all(&((x.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes())?;
    }
    Ok(())
}
