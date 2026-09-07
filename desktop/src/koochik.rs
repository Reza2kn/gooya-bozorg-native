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
fn wants_metal() -> bool {
    match std::env::var("GOOYA_KOOCHIK_DEVICE").as_deref() {
        Ok("metal") => true,
        Ok("cpu") => false,
        _ => cfg!(all(target_os = "macos", target_arch = "aarch64")),
    }
}
fn optimize_step(mut model: TypedModel, allow_fp16: bool) -> Result<TypedModel> {
    if allow_fp16 {
        if let Ok(device) = std::env::var("GOOYA_KOOCHIK_DEVICE") {
            ensure!(["cpu", "metal", "auto"].contains(&device.as_str()), "device must be cpu, metal, or auto");
        }
        #[cfg(not(target_os = "macos"))]
        ensure!(!wants_metal(), "Metal is only supported on macOS");
    }
    if allow_fp16 && std::env::var_os("GOOYA_EXPERIMENTAL_FP16_LINEAR").is_some() {
        // Cast only constant-weight matrix products. Keep graph boundaries,
        // attention scores, normalization and softmax in FP32.
        for id in model.eval_order()? {
            let node = model.node(id);
            if let Some(op) = node.op_as::<tract_core::ops::einsum::EinSum>() {
                if op.operating_dt != DatumType::F32
                    || !node
                        .inputs
                        .iter()
                        .any(|i| model.outlet_fact(*i).is_ok_and(|f| f.konst.is_some()))
                {
                    continue;
                }
                let mut patch = TypedModelPatch::default();
                let mut inputs = tvec![];
                for (ix, input) in node.inputs.iter().enumerate() {
                    let tap = patch.tap_model(&model, *input)?;
                    inputs.push(
                        patch.wire_node(
                            format!("{}.fp16-in-{ix}", node.name),
                            tract_core::ops::cast::cast(DatumType::F16),
                            &[tap],
                        )?[0],
                    );
                }
                let mut op = op.clone();
                op.operating_dt = DatumType::F16;
                let product = patch.wire_node(format!("{}.fp16", node.name), op, &inputs)?;
                let output = patch.wire_node(
                    format!("{}.fp32-out", node.name),
                    tract_core::ops::cast::cast(DatumType::F32),
                    &product,
                )?;
                patch.shunt_outside(&model, node.id.into(), output[0])?;
                patch.apply(&mut model)?;
            }
        }
    }
    #[cfg(target_os = "macos")]
    if allow_fp16 && wants_metal() {
        use tract_core::transform::ModelTransform;
        tract_metal::MetalTransform::default().transform(&mut model)?;
        let count = model.nodes().iter().filter(|n| n.op.name().starts_with("Metal")).count();
        ensure!(count > 0, "Metal requested but no Metal operators were created");
        eprintln!("Koochik backend: tract-metal, {count} Metal operators; FP32 arithmetic");
    }
    Ok(model.into_optimized()?)
}
impl Graph {
    pub fn from_plan(plan: std::sync::Arc<TypedRunnableModel>) -> Self {
        Self { plan }
    }
    pub fn load(path: &Path, inputs: &[Tensor]) -> Result<Self> {
        if path.to_string_lossy().ends_with(".nnef.tar") {
            return Self::load_nnef_shaped(path, true, inputs);
        }
        let mut m = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("load {}", path.display()))?;
        for (i, t) in inputs.iter().enumerate() {
            m.set_input_fact(i, InferenceFact::dt_shape(t.datum_type(), t.shape()))?;
        }
        let options = tract_core::runtime::RunOptions {
            skip_order_opt_ram: wants_metal(),
            executor: Some(tract_linalg::multithread::Executor::multithread(
                std::env::var("GOOYA_KOOCHIK_THREADS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(6)
                    .clamp(1, 8),
            )),
            ..Default::default()
        };
        Ok(Self {
            plan: optimize_step(
                m.into_typed()?.into_decluttered()?,
                path.file_name().is_some_and(|n| n == "step.onnx"),
            )?
            .into_runnable_with_options(&options)?,
        })
    }
    /// Q4 is the download format. Unpacking uses faster dense CPU matrix kernels
    /// for OmniVoice's whole-sequence diffusion, at the cost of runtime RAM.
    pub fn load_nnef(path: &Path, unpack: bool) -> Result<Self> {
        Self::load_nnef_shaped(path, unpack, &[])
    }
    pub fn load_nnef_shaped(path: &Path, unpack: bool, inputs: &[Tensor]) -> Result<Self> {
        let start = Instant::now();
        let mut m = tract_nnef::nnef().model_for_path(path)?;
        if !inputs.is_empty() {
            let mut symbols = std::collections::HashMap::new();
            for (i, input) in inputs.iter().enumerate() {
                let fact = m.input_fact(i)?;
                ensure!(fact.shape.len() == input.rank(), "input rank mismatch");
                for (dim, &actual) in fact.shape.iter().zip(input.shape()) {
                    if let TDim::Sym(symbol) = dim {
                        if let Some(old) = symbols.insert(symbol.clone(), actual.to_dim()) {
                            ensure!(old == actual.to_dim(), "inconsistent input dimensions");
                        }
                    }
                }
            }
            m = m.set_symbols(&symbols)?;
            for (i, input) in inputs.iter().enumerate() {
                ensure!(
                    m.input_fact(i)?.shape.as_concrete() == Some(input.shape()),
                    "unresolved input shape"
                );
            }
        }
        if unpack {
            for id in m.eval_order()? {
                let node = m.node(id);
                if let Some(op) = node.op_as::<tract_core::ops::konst::Const>() {
                    if let Ok(storage) = op
                        .val()
                        .try_storage_as::<tract_linalg::block_quant::BlockQuantStorage>()
                    {
                        let tensor = storage
                            .format()
                            .dequant_f32(storage.value())?
                            .into_shape(op.val().shape())?;
                        let op = tract_core::ops::konst::Const::new(std::sync::Arc::new(tensor))?;
                        TypedModelPatch::replace_single_op(&m, node, &[], op)?.apply(&mut m)?;
                    }
                }
            }
        }
        let options = tract_core::runtime::RunOptions {
            skip_order_opt_ram: wants_metal(),
            executor: Some(tract_linalg::multithread::Executor::multithread(
                std::env::var("GOOYA_KOOCHIK_THREADS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(6)
                    .clamp(1, 8),
            )),
            ..Default::default()
        };
        Ok(Self {
            plan: {
                let plan = optimize_step(m.into_decluttered()?, true)?
                    .into_runnable_with_options(&options)?;
                eprintln!(
                    "Koochik graph preparation: {:.2}s",
                    start.elapsed().as_secs_f32()
                );
                plan
            },
        })
    }
    pub fn load_g2p_decoder(path: &Path, source: usize) -> Result<Self> {
        let mut m = tract_onnx::onnx().model_for_path(path)?;
        let target = m.symbols.sym("target");
        m.set_input_fact(
            0,
            InferenceFact::dt_shape(DatumType::I64, tvec![5.to_dim(), target.to_dim()]),
        )?;
        m.set_input_fact(1, InferenceFact::dt_shape(DatumType::F32, [5, source, 512]))?;
        m.set_input_fact(2, InferenceFact::dt_shape(DatumType::I64, [5, source]))?;
        let mut model = m.into_typed()?;
        // tract 0.23.4 FoldUniformMask incorrectly removes the broadcast beam
        // dimension from this decoder's dynamic causal mask (5 -> 1).
        let mut declutter = tract_core::optim::Optimizer::declutter();
        declutter
            .passes
            .retain(|pass| !format!("{pass:?}").starts_with("FoldUniformMask"));
        declutter.optimize(&mut model)?;
        model.optimize()?;
        Ok(Self {
            plan: model.into_runnable()?,
        })
    }
    pub fn load_dynamic(path: &Path) -> Result<Self> {
        let m = tract_onnx::onnx().model_for_path(path)?;
        Ok(Self {
            plan: m.into_optimized()?.into_runnable()?,
        })
    }
    pub fn backend(&self) -> &'static str {
        if self.plan.model().nodes().iter().any(|n| n.op.name().starts_with("Metal")) { "tract-metal" } else { "tract-cpu" }
    }
    pub fn run(&self, inputs: &[Tensor]) -> Result<TVec<TValue>> {
        let run = || self.plan.run(inputs.iter().cloned().map(|x| x.into_tvalue()).collect());
        // Rust worker threads have no Cocoa event-loop pool. Metal command
        // buffers otherwise retain temporary resources across diffusion passes.
        #[cfg(target_os = "macos")]
        { objc::rc::autoreleasepool(run) }
        #[cfg(not(target_os = "macos"))]
        { run() }
    }
}
pub fn default_steps() -> usize {
    32
}
#[derive(Deserialize)]
pub struct DecodeFixture {
    #[serde(default = "default_steps")]
    pub num_steps: usize,
    pub target_tokens: usize,
    pub noise: Vec<Vec<f32>>,
    pub source_codes: Vec<i64>,
}
pub fn log_normalizer(x: &[f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + x.iter().map(|v| (*v - max).exp()).sum::<f32>().ln()
}
/// Uses the source implementation's explicit per-step position noise for reproducible comparisons.
pub fn decode(graph: &Graph, inputs: Vec<Tensor>, fixture: &DecodeFixture) -> Result<Vec<i64>> {
    decode_with_progress(graph, inputs, fixture, &mut |_, _| {})
}
pub fn decode_with_progress(
    graph: &Graph,
    mut inputs: Vec<Tensor>,
    fixture: &DecodeFixture,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<i64>> {
    let t = fixture.target_tokens;
    let s = inputs[0].shape()[2];
    ensure!(t > 0 && t <= s, "invalid target length");
    ensure!(fixture.noise.len() == 32, "expected 32 noise steps");
    let steps = std::env::var("GOOYA_EXPERIMENTAL_STEPS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(fixture.num_steps);
    ensure!(
        [8, 16, 24, 32].contains(&steps),
        "supported step counts: 8, 16, 24, 32"
    );
    let n = 8 * t;
    let mut tokens = vec![1024i64; n];
    let mut remaining = n;
    for step in 0..steps {
        progress(step, steps);
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
            let x = i as f32 / steps as f32;
            0.1 * x / (1. + (0.1 - 1.) * x)
        };
        let k = if step == steps - 1 {
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
            "Koochik step {}/{steps} ({:.2}s)",
            step + 1,
            start.elapsed().as_secs_f32()
        );
    }
    progress(steps, steps);
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
