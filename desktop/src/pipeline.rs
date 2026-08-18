//! Full fixed-voice T3 -> S3Gen -> HiFT synthesis with tract and ONNX Runtime.
//!
//! Mirrors scripts/run_onnx_pipeline.py for the desktop tract profile. Loads
//! the accepted Q4 ONNX graphs and writes a WAV. Both the CLI binary and the
//! egui app call into this module.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, ensure, Result};
use ndarray::{ArrayViewD, Axis};
use tract_onnx::prelude::*;

pub const SAMPLE_RATE: u32 = 24_000;
const SPEECH_EOS: i64 = 6562;
pub const BUCKET: usize = 168;
const FLOW_STEPS: usize = 10;
const PROMPT_MELS: usize = 244;
const CFG: f32 = 0.5;
const SEED: u64 = 20260816;
const MAX_TEXT_TOKENS_PER_CHUNK: usize = 64;
const CHUNK_PAUSE_SAMPLES: usize = SAMPLE_RATE as usize * 160 / 1000;

/// Results of a full synthesis run.
pub struct SynthesisReport {
    pub tokens: Vec<i64>,
    pub duration_seconds: f32,
    pub samples: usize,
    pub peak: f32,
    pub rms: f32,
    pub wav_path: PathBuf,
}

struct Graph {
    runnable: Box<dyn Runnable>,
}

struct OrtGraph {
    session: Mutex<ort::session::Session>,
}

enum RuntimeGraph {
    Tract(Graph),
    Ort(OrtGraph),
}

fn runtime_options() -> tract_core::runtime::RunOptions {
    let default_threads = std::thread::available_parallelism()
        .map(|threads| threads.get().min(8))
        .unwrap_or(8);
    let threads = std::env::var("GOOYA_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_threads);
    tract_core::runtime::RunOptions {
        executor: Some(tract_linalg::multithread::Executor::multithread(threads)),
        ..Default::default()
    }
}

impl Graph {
    fn load(path: &Path) -> Result<Self> {
        let model = tract_onnx::onnx().model_for_path(path)?;
        let options = runtime_options();
        Ok(Graph {
            runnable: Box::new(model.into_optimized()?.into_runnable_with_options(&options)?),
        })
    }

    fn load_prefill(path: &Path, text_len: usize) -> Result<Self> {
        let mut model = tract_onnx::onnx().model_for_path(path)?;
        model.set_input_fact(
            0,
            InferenceFact::dt_shape(i64::datum_type(), tvec!(1, text_len as i64)),
        )?;
        let options = runtime_options();
        Ok(Graph {
            runnable: Box::new(model.into_optimized()?.into_runnable_with_options(&options)?),
        })
    }

    fn run(&self, feeds: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let inputs: TVec<TValue> = feeds.into_iter().map(|t| t.into_tvalue()).collect();
        let outputs = self.runnable.run(inputs)?;
        Ok(outputs.into_iter().map(|t| t.into_tensor()).collect())
    }

}

impl OrtGraph {
    fn load(path: &Path, threads: usize) -> Result<Self> {
        let _ = ort::init().commit();
        let device = std::env::var("GOOYA_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
        let mut providers = Vec::new();
        if device != "cpu" {
            providers = gpu_providers()?;
            if providers.is_empty() && device == "gpu" {
                bail!("GOOYA_DEVICE=gpu requested, but this binary has no available GPU provider");
            }
        }
        // Try the fastest available provider set first, falling back stepwise
        // (e.g. TensorRT+CUDA -> CUDA -> CPU) so a broken EP never hard-fails.
        let mut last_error = None;
        for cut in (0..=providers.len()).rev() {
            match Self::build_session(path, threads, &providers[..cut]) {
                Ok(session) => return Ok(Self { session: Mutex::new(session) }),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("failed to build ONNX Runtime session")))
    }

    fn build_session(
        path: &Path,
        threads: usize,
        providers: &[ort::ep::ExecutionProviderDispatch],
    ) -> Result<ort::session::Session> {
        let builder = ort::session::Session::builder().map_err(|error| anyhow::anyhow!("{error}"))?;
        let mut builder = builder
            .with_intra_threads(threads)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        if !providers.is_empty() {
            let mut list = providers.to_vec();
            list.push(ort::ep::CPU::default().build());
            builder = builder
                .with_execution_providers(list)
                .map_err(|error| anyhow::anyhow!("{error}"))?;
        }
        builder.commit_from_file(path).map_err(|error| anyhow::anyhow!("{error}"))
    }

    fn tensor(value: Tensor) -> Result<ort::value::DynTensor> {
        let shape = value.shape().to_vec();
        match value.datum_type() {
            DatumType::F32 => {
                let data = value.to_plain_array_view::<f32>()?;
                let data = data.as_slice().ok_or_else(|| anyhow::anyhow!("non-contiguous tensor"))?;
                Ok(ort::value::Tensor::<f32>::from_array((shape, data.to_vec()))?.upcast())
            }
            DatumType::I64 => {
                let data = value.to_plain_array_view::<i64>()?;
                let data = data.as_slice().ok_or_else(|| anyhow::anyhow!("non-contiguous tensor"))?;
                Ok(ort::value::Tensor::<i64>::from_array((shape, data.to_vec()))?.upcast())
            }
            datum_type => bail!("unsupported ORT input datum type: {datum_type:?}"),
        }
    }

    fn run(&self, feeds: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let inputs = feeds
            .into_iter()
            .map(Self::tensor)
            .collect::<Result<Vec<_>>>()?;
        let input_values = inputs
            .into_iter()
            .map(Into::into)
            .collect::<Vec<ort::session::SessionInputValue<'_>>>();
        let mut session = self.session.lock().map_err(|_| anyhow::anyhow!("ORT session poisoned"))?;
        let outputs = session
            .run(ort::session::SessionInputs::from(input_values.as_slice()))
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        outputs
            .values()
            .map(|output| {
                let (shape, data) = output
                    .try_extract_tensor::<f32>()
                    .map_err(|error| anyhow::anyhow!("{error}"))?;
                let shape = shape.iter().map(|&dim| dim as usize).collect::<Vec<_>>();
                Ok(Tensor::from_shape(&shape, data)?)
            })
            .collect()
    }
}

fn gpu_providers() -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
    let mut providers = Vec::new();
    #[cfg(target_os = "windows")]
    {
        let provider = ort::ep::DirectML::default();
        if ort::ep::ExecutionProvider::is_available(&provider)? {
            providers.push(provider.build());
        }
    }
    #[cfg(target_os = "linux")]
    {
        let trt = ort::ep::TensorRT::default()
            .with_max_workspace_size(1 << 30)
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path(std::env::temp_dir().join("gooya-trt-cache").display().to_string())
            .with_detailed_build_log(true);
        providers.push(trt.build().fail_silently());
        let cuda = ort::ep::CUDA::default()
            .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
        providers.push(cuda.build().fail_silently());
    }
    #[cfg(target_os = "macos")]
    {
        use ort::ep::ArbitrarilyConfigurableExecutionProvider as _;
        let mut provider = ort::ep::CoreML::default();
        if let Ok(units) = std::env::var("GOOYA_COREML_UNITS") {
            if let Ok(units) = units.parse::<usize>() {
                provider = provider.with_arbitrary_config("ml_compute_units", units.to_string());
            }
        }
        providers.push(provider.build());
    }
    Ok(providers)
}

impl RuntimeGraph {
    fn use_ort() -> bool {
        std::env::var("GOOYA_RUNTIME").as_deref() != Ok("tract")
    }

    fn load(path: &Path, ort_threads: usize) -> Result<Self> {
        if Self::use_ort() {
            Ok(Self::Ort(OrtGraph::load(path, ort_threads)?))
        } else {
            Ok(Self::Tract(Graph::load(path)?))
        }
    }

    fn load_prefill(path: &Path, text_len: usize, ort_threads: usize) -> Result<Self> {
        if Self::use_ort() {
            Ok(Self::Ort(OrtGraph::load(path, ort_threads)?))
        } else {
            Ok(Self::Tract(Graph::load_prefill(path, text_len)?))
        }
    }

    fn run(&self, feeds: Vec<Tensor>) -> Result<Vec<Tensor>> {
        match self {
            Self::Tract(graph) => graph.run(feeds),
            Self::Ort(graph) => graph.run(feeds),
        }
    }
}

/// Normalize text exactly like scripts/run_onnx_pipeline.py normalize_text().
pub fn normalize_text(text: &str) -> String {
    let mut t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for (old, new) in [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "),
        (";", ","), ("—", "-"), ("–", "-"), (" ,", ","),
        ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
    ] {
        t = t.replace(old, new);
    }
    t = t.trim_end().to_owned();
    if !t.is_empty() && !matches!(t.chars().last().unwrap(), '.' | '!' | '?' | '-' | ',') {
        t.push('.');
    }
    t
}

/// Tokenize normalized text with the grapheme BPE tokenizer.
pub fn tokenize(tokenizer_path: &Path, text: &str) -> Result<Vec<i64>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    encode_normalized(&tok, &normalize_text(text))
}

fn encode_normalized(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>> {
    let encoded = tokenizer
        .encode(text.replace(' ', "[SPACE]"), false)
        .map_err(anyhow::Error::msg)?;
    Ok(encoded.get_ids().iter().map(|&i| i as i64).collect())
}

/// Split long normalized text without adding punctuation at intermediate
/// boundaries. This model consumes grapheme BPE, not phonemes, so the budget is
/// expressed in tokenizer tokens rather than phone characters.
pub fn chunk_text(tokenizer_path: &Path, text: &str) -> Result<Vec<String>> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let normalized = normalize_text(text);
    let words: Vec<&str> = normalized.split_whitespace().collect();
    if words.is_empty() {
        bail!("text must not be empty");
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in words {
        let candidate = if current.is_empty() {
            word.to_owned()
        } else {
            format!("{current} {word}")
        };
        let candidate_ids = encode_normalized(&tokenizer, &candidate)?;
        let current_ids = if !current.is_empty() && candidate_ids.len() > MAX_TEXT_TOKENS_PER_CHUNK {
            chunks.push(current);
            current = word.to_owned();
            encode_normalized(&tokenizer, &current)?
        } else {
            current = candidate;
            candidate_ids
        };
        if current_ids.len() >= 16
            && word
                .chars()
                .last()
                .is_some_and(|c| matches!(c, '.' | '!' | '?' | '؟' | '،' | ','))
        {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    Ok(chunks)
}

/// Run the whole pipeline: T3 -> flow -> vocoder, writing `out` and returning stats.
pub fn synthesize(model_dir: &Path, out: &Path, text_ids: &[i64]) -> Result<SynthesisReport> {
    if text_ids.len() > MAX_TEXT_TOKENS_PER_CHUNK {
        let chunks = text_ids.chunks(MAX_TEXT_TOKENS_PER_CHUNK).collect::<Vec<_>>();
        return synthesize_token_chunks(model_dir, out, &chunks);
    }
    synthesize_single(model_dir, out, text_ids)
}

/// Synthesize text with word-boundary chunking and 160 ms inter-chunk pauses.
pub fn synthesize_text(
    model_dir: &Path,
    tokenizer_path: &Path,
    out: &Path,
    text: &str,
) -> Result<SynthesisReport> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let chunks = chunk_text(tokenizer_path, text)?;
    if chunks.len() == 1 {
        return synthesize_single(model_dir, out, &encode_normalized(&tokenizer, &chunks[0])?);
    }
    let ids = chunks
        .iter()
        .map(|chunk| encode_normalized(&tokenizer, chunk))
        .collect::<Result<Vec<_>>>()?;
    let ids = ids
        .into_iter()
        .flat_map(|chunk| {
            chunk
                .chunks(MAX_TEXT_TOKENS_PER_CHUNK)
                .map(|part| part.to_vec())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let refs = ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    synthesize_token_chunks(model_dir, out, &refs)
}

fn synthesize_single(model_dir: &Path, out: &Path, text_ids: &[i64]) -> Result<SynthesisReport> {
    let tokens = timer("t3", || generate_tokens(model_dir, text_ids))?;
    if tokens.is_empty() {
        bail!("T3 emitted EOS before any speech token");
    }
    if let Ok(dump_path) = std::env::var("GOOYA_DUMP_TOKENS") {
        use std::io::Write as _;
        let s: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        let mut f = std::fs::File::create(&dump_path)?;
        f.write_all(s.join(",").as_bytes())?;
    }
    let mel = timer("flow", || run_flow(model_dir, &tokens, BUCKET))?;
    if let Ok(dump_path) = std::env::var("GOOYA_DUMP_MEL") {
        dump_f32(&dump_path, &mel)?;
    }
    let (waveform, source, stft, spec) = timer("vocoder", || run_vocoder(model_dir, &mel, &tokens))?;
    if let Ok(dump_path) = std::env::var("GOOYA_DUMP_SOURCE") {
        dump_f32(&dump_path, &source)?;
    }
    if let Ok(dump_path) = std::env::var("GOOYA_DUMP_STFT") {
        dump_f32(&dump_path, &stft)?;
    }
    if let Ok(dump_path) = std::env::var("GOOYA_DUMP_SPEC") {
        dump_f32(&dump_path, &spec)?;
    }
    if let Ok(dump_path) = std::env::var("GOOYA_DUMP_WAVEFORM") {
        dump_f32(&dump_path, &waveform)?;
    }
    let valid = tokens.len() * 2 * 480;
    write_wav(out, &waveform, valid)?;
    let peak = waveform.iter().take(valid).fold(0.0f32, |a, &v| a.max(v.abs()));
    let rms = (waveform.iter().take(valid).map(|v| v * v).sum::<f32>() / valid as f32).sqrt();
    Ok(SynthesisReport {
        tokens,
        duration_seconds: valid as f32 / SAMPLE_RATE as f32,
        samples: valid,
        peak,
        rms,
        wav_path: out.to_path_buf(),
    })
}

fn synthesize_token_chunks(
    model_dir: &Path,
    out: &Path,
    chunks: &[&[i64]],
) -> Result<SynthesisReport> {
    let mut pcm = Vec::new();
    let mut tokens = Vec::new();
    for (index, text_ids) in chunks.iter().enumerate() {
        let chunk_path = std::env::temp_dir().join(format!(
            "gooya-{}-chunk-{index}.wav",
            std::process::id()
        ));
        let report = synthesize_single(model_dir, &chunk_path, text_ids)?;
        let chunk_pcm = read_pcm_wav(&chunk_path)?;
        let _ = std::fs::remove_file(&chunk_path);
        if index > 0 {
            pcm.extend(std::iter::repeat_n(0i16, CHUNK_PAUSE_SAMPLES));
        }
        pcm.extend(chunk_pcm);
        tokens.extend(report.tokens);
    }
    write_pcm_wav(out, &pcm)?;
    let peak = pcm
        .iter()
        .map(|&sample| (sample as f32 / 32767.0).abs())
        .fold(0.0, f32::max);
    let rms = (pcm
        .iter()
        .map(|&sample| {
            let sample = sample as f32 / 32767.0;
            sample * sample
        })
        .sum::<f32>()
        / pcm.len().max(1) as f32)
        .sqrt();
    Ok(SynthesisReport {
        tokens,
        duration_seconds: pcm.len() as f32 / SAMPLE_RATE as f32,
        samples: pcm.len(),
        peak,
        rms,
        wav_path: out.to_path_buf(),
    })
}

fn read_pcm_wav(path: &Path) -> Result<Vec<i16>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() >= 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    ensure!(u32::from_le_bytes(bytes[24..28].try_into()?) == SAMPLE_RATE);
    ensure!(u16::from_le_bytes(bytes[34..36].try_into()?) == 16);
    let data_len = u32::from_le_bytes(bytes[40..44].try_into()?) as usize;
    ensure!(bytes.len() >= 44 + data_len && data_len % 2 == 0);
    Ok(bytes[44..44 + data_len]
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect())
}

fn write_pcm_wav(path: &Path, samples: &[i16]) -> Result<()> {
    use std::io::Write;
    let data_len = samples.len() * 2;
    let mut buf = Vec::with_capacity(44 + data_len);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in samples {
        buf.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::File::create(path)?.write_all(&buf)?;
    Ok(())
}

fn dump_f32(path: &str, values: &[f32]) -> Result<()> {
    use std::io::Write as _;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)?;
    Ok(())
}

/// Emit a per-phase timing line when GOOYA_PROFILE=1.
fn timer<F, T>(phase: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let profiling = std::env::var("GOOYA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let start = std::time::Instant::now();
    let result = f();
    if profiling {
        eprintln!("[profile] {phase}: {:.3}s", start.elapsed().as_secs_f64());
    }
    result
}

fn generate_tokens(dir: &Path, text_ids: &[i64]) -> Result<Vec<i64>> {
    let prefill = timer("t3.load.prefill", || {
        RuntimeGraph::load_prefill(&dir.join("t3-prefill.onnx"), text_ids.len(), 8)
    })?;
    let decode = timer("t3.load.decode", || RuntimeGraph::load(&dir.join("t3-decode.onnx"), 8))?;
    let input = Tensor::from_shape(&[1, text_ids.len()], text_ids)?;
    let mut outputs = timer("t3.prefill.run", || prefill.run(vec![input]))?;
    let mut tokens: Vec<i64> = Vec::new();
    let mut position = 0usize;
    loop {
        let logits = outputs[0].to_plain_array_view::<f32>()?;
        let cond = logits.index_axis(Axis(0), 0);
        let uncond = logits.index_axis(Axis(0), 1);
        let last = logits.shape()[2];
        let mut token = 0i64;
        let mut best = f32::NEG_INFINITY;
        for i in 0..last {
            let c = cond[[0, i]];
            let u = uncond[[0, i]];
            let v = c + CFG * (c - u);
            if v > best {
                best = v;
                token = i as i64;
            }
        }
        if token == SPEECH_EOS {
            break;
        }
        tokens.push(token);
        position += 1;
        if position >= BUCKET {
            break;
        }
        let past: Vec<Tensor> = outputs.drain(1..).collect();
        let past_len = past[0].shape()[2];
        let mut feeds = Vec::new();
        feeds.push(Tensor::from_shape(&[1, 1], &[token])?);
        feeds.push(Tensor::from_shape(&[1], &[position as i64])?);
        feeds.push(Tensor::from_shape(&[1], &[past_len as i64])?);
        for tensor in past.into_iter() {
            feeds.push(tensor);
        }
        outputs = timer(&format!("t3.decode.{position}"), || decode.run(feeds))?;
    }
    Ok(tokens)
}

fn run_flow(dir: &Path, tokens: &[i64], bucket: usize) -> Result<Vec<f32>> {
    let prepare = timer("flow.load.prepare", || {
        let base = format!("s3-flow-prepare-b{bucket}.onnx");
        let folded = format!("s3-flow-prepare-b{bucket}.folded.onnx");
        let path = if dir.join(&folded).exists() {
            dir.join(&folded)
        } else {
            dir.join(&base)
        };
        Graph::load(&path)
    })?;
    let step = timer("flow.load.step", || -> Result<RuntimeGraph> {
        let path = dir.join(format!("s3-flow-step-b{bucket}.onnx"));
        if std::env::var("GOOYA_FLOW_RUNTIME").as_deref() == Ok("tract") {
            Ok(RuntimeGraph::Tract(Graph::load(&path)?))
        } else {
            Ok(RuntimeGraph::Ort(OrtGraph::load(&path, 4)?))
        }
    })?;
    let mut padded = vec![0i64; bucket];
    for (i, t) in tokens.iter().enumerate() {
        padded[i] = *t;
    }
    let prepared = prepare.run(vec![
        Tensor::from_shape(&[1, bucket], &padded)?,
        Tensor::from_shape(&[1], &[tokens.len() as i64])?,
    ])?;
    let shape = prepared[0].shape().to_owned();
    let dims: Vec<usize> = shape.to_vec();
    let mut rng_state = simple_rng(SEED);
    let mut state = randn_tensor(&dims, &mut rng_state);
    let schedule = cosine_schedule(FLOW_STEPS);
    for index in 0..FLOW_STEPS {
        let inputs = vec![
            state.clone(),
            prepared[0].clone(),
            prepared[1].clone(),
            prepared[2].clone(),
            prepared[3].clone(),
            Tensor::from_shape(&[1], &[schedule[index]])?,
            Tensor::from_shape(&[1], &[schedule[index + 1]])?,
        ];
        state = timer(&format!("flow.step.{index}"), || step.run(inputs))?[0].clone();
    }
    // mel = state[:, :, PROMPT_MELS : PROMPT_MELS + bucket*2]
    let state_arr = state.to_plain_array_view::<f32>()?;
    let nch = state_arr.shape()[1];
    let mut mel = vec![0f32; nch * bucket * 2];
    let mut idx = 0;
    for c in 0..nch {
        for t in 0..bucket * 2 {
            mel[idx] = state_arr[[0, c, PROMPT_MELS + t]];
            idx += 1;
        }
    }
    Ok(mel)
}

fn run_vocoder(dir: &Path, mel: &[f32], tokens: &[i64]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let source = timer("vocoder.load.source", || {
        RuntimeGraph::load(&dir.join(format!("s3-vocoder-source-b{BUCKET}.onnx")), 4)
    })?;
    let spectral = timer("vocoder.load.spectral", || {
        RuntimeGraph::load(&dir.join(format!("s3-vocoder-spectral-b{BUCKET}.onnx")), 4)
    })?;
    let nch = 80usize;
    let melframes = BUCKET * 2; // vocoder consumes the full padded bucket mel
    let mut mel4 = vec![0f32; 1 * nch * melframes];
    for (dst, src) in mel4.iter_mut().zip(mel.iter()) {
        *dst = *src;
    }
    // phase (1,9,1), noise (1,9, melframes*480)
    let mut rng_state = simple_rng(SEED);
    let mut phase = vec![0f32; 9];
    for i in 1..9 {
        phase[i] = (next_u32(&mut rng_state) as f32) / 4294967296.0 * 2.0 * std::f32::consts::PI
            - std::f32::consts::PI;
    }
    let noise_len = melframes * 480;
    let mut noise = vec![0f32; 9 * noise_len];
    for v in noise.iter_mut() {
        let a = (next_u32(&mut rng_state) as f32) / 4294967296.0 + 1e-8;
        let b = (next_u32(&mut rng_state) as f32) / 4294967296.0;
        *v = (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::PI * b).cos();
    }
    let src_out_tensor = timer("vocoder.source.run", || {
        source.run(vec![
            Tensor::from_shape(&[1, nch, melframes], &mel4)?,
            Tensor::from_shape(&[1, 9, 1], &phase)?,
            Tensor::from_shape(&[1, 9, noise_len], &noise)?,
        ])
    })?
    .remove(0);
    let src_out = src_out_tensor.to_plain_array_view::<f32>()?;
    let src_len = src_out.shape()[2];
    let source_vals: Vec<f32> = src_out.iter().copied().collect();

    // Native STFT: nfft=16, hop=4 -> 9 bins, packed real then imag (torch layout)
    let stft = stft(&source_vals, src_len, 16, 4);
    let spectral_out = timer("vocoder.spectral.run", || {
        spectral.run(vec![
            Tensor::from_shape(&[1, nch, melframes], &mel4)?,
            Tensor::from_shape(&[1, 18, stft.len() / 18], &stft)?,
        ])
    })?;
    let magnitude = spectral_out[0].to_plain_array_view::<f32>()?;
    let phase_out = spectral_out[1].to_plain_array_view::<f32>()?;
    let mut spec = vec![0f32; 2 * magnitude.len()];
    for (dst, src) in spec[..magnitude.len()].iter_mut().zip(magnitude.iter()) {
        *dst = *src;
    }
    for (dst, src) in spec[magnitude.len()..].iter_mut().zip(phase_out.iter()) {
        *dst = *src;
    }
    // istft with nfft=16 hop=4 window hann; torch clamps after istft
    let mut waveform = istft(&magnitude, &phase_out);
    for v in waveform.iter_mut() {
        *v = v.clamp(-0.99, 0.99);
    }
    let _ = tokens; // valid samples are derived by the caller
    Ok((waveform, source_vals, stft, spec))
}

fn cosine_schedule(steps: usize) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let x = (i as f32 / steps as f32) * 0.5 * std::f32::consts::PI;
            1.0 - x.cos()
        })
        .collect()
}

/// Tiny deterministic xorshift-based RNG so results match the Python numpy seed.
fn simple_rng(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn next_u32(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

fn randn_tensor(dims: &[usize], state: &mut u64) -> Tensor {
    let count: usize = dims.iter().product();
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        // Box-Muller from two uniforms
        let a = (next_u32(state) as f32) / 4294967296.0 + 1e-8;
        let b = (next_u32(state) as f32) / 4294967296.0;
        let r = (-2.0 * a.ln()).sqrt();
        let t = 2.0 * std::f32::consts::PI * b;
        values.push(r * t.cos());
    }
    Tensor::from_shape(dims, &values).unwrap()
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn stft(x: &[f32], _len: usize, nfft: usize, hop: usize) -> Vec<f32> {
    let window = hann(nfft);
    // torch.stft with n_fft=16, hop_length=4, win_length=16, center=True (default)
    // pads with reflect so n_frames = len/hop + 1
    let n_frames = x.len() / hop + 1;
    let n_bins = nfft / 2 + 1;
    let mut out = vec![0f32; 2 * n_bins * n_frames];
    // torch reflect padding: i<0 -> -i, i>=n -> 2*(n-1)-i
    let reflect = |i: isize| -> usize {
        let n = x.len() as isize;
        if i < 0 {
            (-i) as usize
        } else if i >= n {
            (2 * (n - 1) - i) as usize
        } else {
            i as usize
        }
    };
    for f in 0..n_frames {
        let offset = f as isize * hop as isize - (nfft / 2) as isize;
        for b in 0..n_bins {
            let mut re = 0f32;
            let mut im = 0f32;
            for k in 0..nfft {
                let idx = reflect(offset + k as isize);
                let ang = -2.0 * std::f32::consts::PI * (k as f32) * (b as f32) / nfft as f32;
                let v = x[idx] * window[k];
                re += v * ang.cos();
                im += v * ang.sin();
            }
            // torch packs real bins 0..n_bins then imag bins 0..n_bins
            out[b * n_frames + f] = re;
            out[(n_bins + b) * n_frames + f] = im;
        }
    }
    out
}

fn istft(magnitude: &ArrayViewD<f32>, phase: &ArrayViewD<f32>) -> Vec<f32> {
    let nch = magnitude.shape()[1];
    let n_frames = magnitude.shape()[2];
    let nfft = 16;
    let hop = 4;
    // torch.istft(complex, 16, 4, 16, window): irfft of the one-sided
    // spectrum, window, overlap-add, divide by window envelope, center trim.
    // Output length = (n_frames - 1) * hop.
    let window = hann(nfft);
    let expected = nfft + hop * (n_frames - 1);
    let mut y = vec![0f32; expected];
    let mut wsum = vec![0f32; expected];
    for f in 0..n_frames {
        // frame of 16 samples from the one-sided complex spectrum
        let mut frame = [0f32; 16];
        for n in 0..16 {
            let base = 2.0 * std::f32::consts::PI * n as f32 / nfft as f32;
            let mut v = 0f32;
            for b in 0..nch {
                let mag = magnitude[[0, b, f]].clamp(0.0, 100.0);
                let ph = phase[[0, b, f]];
                let re = mag * ph.cos();
                let im = mag * ph.sin();
                if b == 0 {
                    v += re;
                } else if b == nfft / 2 {
                    // Nyquist bin: real part only, (-1)^n factor
                    v += re * (std::f32::consts::PI * n as f32).cos();
                } else {
                    // conjugate pair contributes 2*Re(X_b e^{i*base*n})
                    let ang = base * b as f32;
                    v += 2.0 * (re * ang.cos() - im * ang.sin());
                }
            }
            frame[n] = v / nfft as f32;
        }
        for n in 0..16 {
            let idx = f * hop + n;
            y[idx] += frame[n] * window[n];
            wsum[idx] += window[n] * window[n];
        }
    }
    let start = nfft / 2;
    let out_len = (n_frames - 1) * hop;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let v = if wsum[start + i] > 1e-6 {
            y[start + i] / wsum[start + i]
        } else {
            0.0
        };
        out.push(v);
    }
    out
}

fn write_wav(path: &Path, samples: &[f32], valid: usize) -> Result<()> {
    use std::io::Write;
    let mut buf = Vec::with_capacity(44 + valid * 2);
    let data_len = valid * 2;
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples.iter().take(valid) {
        let pcm = (s * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
        buf.extend_from_slice(&pcm.to_le_bytes());
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_text_chunks_at_word_boundaries() {
        let tokenizer = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data/grapheme_mtl_merged_expanded_v1.json");
        let text = "word ".repeat(100);
        let chunks = chunk_text(&tokenizer, &text).expect("chunking should succeed");
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| !chunk.starts_with(' ')));
        assert!(chunks.iter().all(|chunk| !chunk.ends_with(' ')));
    }
}
