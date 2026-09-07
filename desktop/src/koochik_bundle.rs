//! Standalone Koochik bundle: verified lossless weights, native frontend and synthesis.
use crate::{
    koochik::{self, DecodeFixture, Graph},
    koochik_frontend::{Frontend, paced},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tract_onnx::prelude::*;

#[derive(Deserialize)]
pub struct Manifest {
    pub source_revision: String,
    #[serde(default = "koochik::default_steps")]
    pub inference_steps: usize,
    pub files: Vec<Asset>,
    #[serde(alias = "expanded_weights")]
    pub expanded_model: Asset,
    #[serde(default)]
    pub graph_format: String,
}
#[derive(Deserialize)]
pub struct Asset {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Deserialize)]
pub struct Voice {
    pub paced_text: String,
    pub frames: usize,
    pub codes: Vec<i64>,
    pub rms: f32,
}
#[derive(Serialize)]
pub struct Report {
    pub model: &'static str,
    pub samples: usize,
    pub duration_seconds: f64,
    pub wav_path: PathBuf,
    pub phrases: Vec<String>,
    pub wall_seconds: f64,
    pub generation_seconds: f64,
    pub loaded_speech_graph: bool,
}
fn hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub(crate) fn verify(path: &Path, asset: &Asset) -> Result<()> {
    ensure!(
        fs::metadata(path)?.len() == asset.bytes,
        "wrong size: {}",
        path.display()
    );
    ensure!(
        hash(path)? == asset.sha256,
        "checksum mismatch: {}",
        path.display()
    );
    Ok(())
}
/// Verify the release and expand losslessly into a local cache. No Python is used.
pub fn prepare(root: &Path) -> Result<PathBuf> {
    let m: Manifest = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    ensure!(
        m.source_revision == "537bb48320fd657415eef5b4fb6796b6cebab195",
        "unsupported source revision"
    );
    for a in &m.files {
        ensure!(
            Path::new(&a.path)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe asset path"
        );
        verify(&root.join(&a.path), a)?;
    }
    let cache = root.join("cache");
    fs::create_dir_all(&cache)?;
    ensure!(
        Path::new(&m.expanded_model.path).components().count() == 1
            && Path::new(&m.expanded_model.path)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "unsafe expanded model path"
    );
    let weights = cache.join(&m.expanded_model.path);
    if !weights.exists() || verify(&weights, &m.expanded_model).is_err() {
        let tmp = cache.join(format!(
            "{}.{}.partial",
            m.expanded_model.path,
            std::process::id()
        ));
        let result = (|| -> Result<()> {
            let input = fs::File::open(root.join(format!("{}.zst", m.expanded_model.path)))?;
            let mut decoder = zstd::stream::read::Decoder::new(input)?;
            let mut out = fs::File::create(&tmp)?;
            let n = std::io::copy(
                &mut decoder.by_ref().take(m.expanded_model.bytes + 1),
                &mut out,
            )?;
            ensure!(n == m.expanded_model.bytes, "expanded weight size mismatch");
            out.flush()?;
            out.sync_all()?;
            verify(&tmp, &m.expanded_model)?;
            fs::rename(&tmp, &weights)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(tmp);
        }
        result?;
    }
    if m.graph_format == "nnef-q4" {
        return Ok(weights);
    }
    ensure!(
        m.graph_format.is_empty() || m.graph_format == "onnx-f32",
        "unsupported graph format"
    );
    fs::copy(root.join("step.onnx"), cache.join("step.onnx"))?;
    Ok(cache.join("step.onnx"))
}
pub fn voice(root: &Path) -> Result<Voice> {
    let v: Voice = serde_json::from_slice(&fs::read(root.join("voice.json"))?)?;
    ensure!(
        v.frames > 0
            && v.codes.len() == 8 * v.frames
            && v.codes.iter().all(|x| (0..1024).contains(x)),
        "invalid reference codes"
    );
    Ok(v)
}
fn weight(s: &str) -> f64 {
    s.chars()
        .map(|c| {
            if c == ' ' {
                0.2
            } else if c.is_ascii_digit() {
                3.5
            } else if c.is_ascii_alphabetic() {
                1.0
            } else {
                0.5
            }
        })
        .sum()
}
pub fn inputs(root: &Path, text: &str) -> Result<(Vec<Tensor>, usize)> {
    ensure!(text.is_ascii(), "expected formatted ASCII phonemes");
    let v = voice(root)?;
    let raw = weight(text) / (weight(&v.paced_text) / v.frames as f64);
    let target = (if raw < 50. {
        50. * (raw / 50.).powf(1. / 3.)
    } else {
        raw
    }) / 0.85;
    let t = (target as usize).max(1);
    ensure!(
        t <= 250,
        "phrase exceeds 10 seconds; split into shorter sentences"
    );
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let encode = |s: &str| -> Result<Vec<i64>> {
        Ok(tokenizer
            .encode(s, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .iter()
            .map(|&x| x as i64)
            .collect())
    };
    let mut prefix =
        encode("<|denoise|><|lang_start|>fa<|lang_end|><|instruct_start|>None<|instruct_end|>")?;
    prefix.extend(encode(&format!(
        "<|text_start|>{} {}<|text_end|>",
        v.paced_text.trim(),
        text.trim()
    ))?);
    let allowed_path = root.join("allowed_text_ids.json");
    if allowed_path.exists() {
        let allowed: std::collections::HashSet<i64> =
            serde_json::from_slice(&fs::read(allowed_path)?)?;
        ensure!(
            prefix.iter().all(|id| allowed.contains(id)),
            "text contains a token outside this compact model's phoneme vocabulary"
        );
    }
    let p = prefix.len();
    let s = p + v.frames + t;
    let mut ids = vec![1024i64; 16 * s];
    for c in 0..8 {
        ids[c * s..c * s + p].copy_from_slice(&prefix);
        ids[c * s + p..c * s + p + v.frames]
            .copy_from_slice(&v.codes[c * v.frames..(c + 1) * v.frames]);
    }
    let mut am = vec![false; 2 * s];
    am[p..s].fill(true);
    am[s..s + t].fill(true);
    let mut att = vec![f32::MIN; 2 * s * s];
    att[..s * s].fill(0.);
    for i in 0..s {
        if i < t {
            att[s * s + i * s..s * s + i * s + t].fill(0.);
        } else {
            att[s * s + i * s + i] = 0.;
        }
    }
    let pos: Vec<i64> = (0..2).flat_map(|_| (0..s).map(|x| x as i64)).collect();
    Ok((
        vec![
            Tensor::from_shape(&[2, 8, s], &ids)?,
            Tensor::from_shape(&[2, s], &am)?,
            Tensor::from_shape(&[2, 1, s, s], &att)?,
            Tensor::from_shape(&[2, s], &pos)?,
        ],
        t,
    ))
}
/// Portable SplitMix64 position noise. Seed numbers are not PyTorch RNG equivalents.
pub fn noise(seed: u64, t: usize) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..32)
        .map(|_| {
            (0..8 * t)
                .map(|_| {
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                    z ^= z >> 31;
                    let u = (((z >> 40) as f64 + 0.5) / 16777216.) as f32;
                    -(-u.ln()).ln()
                })
                .collect()
        })
        .collect()
}
fn split_phrases(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut s = String::new();
    for c in text.chars() {
        if c == '\n' {
            if !s.trim().is_empty() {
                out.push(s.trim().to_owned());
            }
            s.clear();
        } else {
            if !matches!(c, '.' | '!' | '?' | '؟') && s.ends_with(['.', '!', '?', '؟']) {
                if !s.trim().is_empty() {
                    out.push(s.trim().to_owned());
                }
                s.clear();
            }
            s.push(c);
        }
    }
    if !s.trim().is_empty() {
        out.push(s.trim().to_owned());
    }
    out
}
// Match the source pydub PCM16 silence detection at 24 kHz, then fade and pad.
pub fn postprocess(raw: &[f32], rms: f32) -> Vec<f32> {
    let pcm: Vec<i16> = raw
        .iter()
        .map(|x| (x * 32768.).clamp(-32768., 32767.) as i16)
        .collect();
    let ms = (pcm.len() as f64 / 24.).round_ties_even() as usize;
    let energy = |a: usize, b: usize| -> f64 {
        let x = &pcm[(a * 24).min(pcm.len())..(b * 24).min(pcm.len())];
        if x.is_empty() {
            0.
        } else {
            (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64)
                .sqrt()
                .floor()
        }
    };
    let threshold = 32768. * 10f64.powf(-50. / 20.);
    let mut silent = Vec::new();
    if ms >= 500 {
        let mut starts: Vec<usize> = (0..=ms - 500).step_by(10).collect();
        if starts.last() != Some(&(ms - 500)) {
            starts.push(ms - 500);
        }
        for a in starts {
            if energy(a, a + 500) <= threshold {
                silent.push(a);
            }
        }
    }
    let mut ranges = Vec::new();
    if silent.is_empty() {
        ranges.push((0, ms));
    } else {
        let mut sr = Vec::new();
        let mut a = silent[0];
        let mut prev = a;
        for &i in &silent[1..] {
            if i > prev + 10 && i > prev + 500 {
                sr.push((a, prev + 500));
                a = i;
            }
            prev = i;
        }
        sr.push((a, prev + 500));
        let mut end = 0;
        for (a, b) in sr {
            if a > end {
                ranges.push((end, a));
            }
            end = b;
        }
        if end < ms {
            ranges.push((end, ms));
        }
    }
    let mut expanded: Vec<(usize, usize)> = ranges
        .iter()
        .map(|&(a, b)| (a.saturating_sub(500), (b + 500).min(ms)))
        .collect();
    for i in 1..expanded.len() {
        if expanded[i].0 < expanded[i - 1].1 {
            let middle = (expanded[i].0 + expanded[i - 1].1) / 2;
            expanded[i - 1].1 = middle;
            expanded[i].0 = middle;
        }
    }
    let mut kept = Vec::new();
    for (a, b) in expanded {
        kept.extend_from_slice(&pcm[(a * 24).min(pcm.len())..(b * 24).min(pcm.len())]);
    }
    let trim = |x: &[i16]| -> usize {
        let mut n = 0;
        while n * 24 < x.len() {
            let e = &x[n * 24..((n + 10) * 24).min(x.len())];
            let r = (e.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / e.len() as f64)
                .sqrt()
                .floor();
            if r >= threshold {
                break;
            }
            n += 10;
        }
        n.saturating_sub(100) * 24
    };
    let lead = trim(&kept).min(kept.len());
    kept.drain(..lead);
    kept.reverse();
    let tail = trim(&kept).min(kept.len());
    kept.drain(..tail);
    kept.reverse();
    let mut audio: Vec<f32> = kept
        .iter()
        .map(|&x| {
            let v = x as f32 / 32768.;
            if rms < 0.1 { v * rms / 0.1 } else { v }
        })
        .collect();
    let n = audio.len();
    let k = 2400.min(n / 2);
    if k > 1 {
        for i in 0..k {
            audio[i] *= i as f32 / (k - 1) as f32;
            audio[n - k + i] *= (k - 1 - i) as f32 / (k - 1) as f32;
        }
    }
    if n == 0 {
        return audio;
    }
    let mut out = vec![0.; 2400];
    out.extend(audio);
    out.extend(vec![0.; 2400]);
    out
}
/// A single cached engine avoids repeated asset hashing and retains the last
/// shape-specialized speech and codec plans. Switching bundles drops the cache.
struct Engine {
    root: PathBuf,
    manifest: Vec<u8>,
    step: PathBuf,
    frontend: Frontend,
    voice: Voice,
    steps: usize,
    asset_stamps: Vec<(PathBuf, u64, std::time::SystemTime)>,
    graph: Option<(Vec<usize>, Graph)>,
    codec: Option<(usize, Graph)>,
}
impl Engine {
    fn load(root: &Path, manifest: Vec<u8>) -> Result<Self> {
        let settings: Manifest = serde_json::from_slice(&manifest)?;
        ensure!(
            [8, 16, 24, 32].contains(&settings.inference_steps),
            "unsupported generation steps"
        );
        let step = prepare(root)?;
        let mut paths: Vec<PathBuf> = settings.files.iter().map(|a| root.join(&a.path)).collect();
        paths.push(step.clone());
        paths.push(root.join("cache").join(&settings.expanded_model.path));
        let asset_stamps = paths
            .into_iter()
            .map(|path| -> Result<_> {
                let metadata = fs::metadata(&path)?;
                Ok((path, metadata.len(), metadata.modified()?))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            asset_stamps,
            steps: settings.inference_steps,
            root: root.into(),
            manifest,
            step,
            frontend: Frontend::load(&root.join("frontend"))?,
            voice: voice(root)?,
            graph: None,
            codec: None,
        })
    }
    fn assets_unchanged(&self) -> bool {
        self.asset_stamps.iter().all(|(path, size, modified)| {
            fs::metadata(path)
                .is_ok_and(|m| m.len() == *size && m.modified().is_ok_and(|t| t == *modified))
        })
    }
    fn synthesize(
        &mut self,
        out: &Path,
        text: &str,
        seed: u64,
        progress: &mut dyn FnMut(&str),
    ) -> Result<Report> {
        ensure!(text.len() <= 4000, "text exceeds 4000 UTF-8 bytes");
        let phrases = split_phrases(text);
        ensure!(!phrases.is_empty(), "empty text");
        let mut generation_seconds = 0.;
        let mut loaded_speech_graph = false;
        let mut audio = Vec::new();
        let mut phones = Vec::new();
        for (index, phrase) in phrases.iter().enumerate() {
            progress(&format!(
                "آماده‌سازی تلفظ · بخش {}/{}",
                index + 1,
                phrases.len()
            ));
            let p = paced(phrase, &self.frontend.phones(phrase)?);
            let (i, t) = inputs(&self.root, &p)?;
            let fixture = DecodeFixture {
                num_steps: self.steps,
                target_tokens: t,
                noise: noise(seed, t),
                source_codes: Vec::new(),
            };
            let shape = i[0].shape().to_vec();
            if self.graph.as_ref().is_none_or(|(s, _)| *s != shape) {
                progress("آماده‌سازی مدل صدا…");
                loaded_speech_graph = true;
                self.graph = None;
                self.graph = Some((shape, Graph::load(&self.step, &i)?));
            }
            let generation_start = std::time::Instant::now();
            let codes = koochik::decode_with_progress(
                &self.graph.as_ref().unwrap().1,
                i,
                &fixture,
                &mut |done, total| {
                    progress(&format!(
                        "ساخت صدا · بخش {}/{} · {done}/{total}",
                        index + 1,
                        phrases.len()
                    ))
                },
            )?;
            generation_seconds += generation_start.elapsed().as_secs_f64();
            progress("آماده‌سازی فایل صوتی…");
            let ci = vec![Tensor::from_shape(&[1, 8, t], &codes)?];
            if self.codec.as_ref().is_none_or(|(length, _)| *length != t) {
                self.codec = None;
                self.codec = Some((t, Graph::load(&self.root.join("decoder.onnx"), &ci)?));
            }
            let result = self.codec.as_ref().unwrap().1.run(&ci)?;
            let raw = result[0].to_plain_array_view::<f32>()?;
            let processed = postprocess(raw.as_slice().context("codec output")?, self.voice.rms);
            ensure!(!processed.is_empty(), "model produced silence");
            if !audio.is_empty() {
                audio.extend(vec![0.; 8400]);
            }
            audio.extend(processed);
            phones.push(p);
        }
        koochik::write_wav(out, &audio, 24000)?;
        Ok(Report {
            wall_seconds: 0.,
            generation_seconds,
            loaded_speech_graph,
            model: "Gooya Koochik v2.0-exp",
            samples: audio.len(),
            duration_seconds: audio.len() as f64 / 24000.,
            wav_path: out.into(),
            phrases: phones,
        })
    }
}
pub fn synthesize_with_progress(
    root: &Path,
    out: &Path,
    text: &str,
    seed: u64,
    progress: &mut dyn FnMut(&str),
) -> Result<Report> {
    static ENGINE: std::sync::OnceLock<std::sync::Mutex<Option<Engine>>> =
        std::sync::OnceLock::new();
    let start = std::time::Instant::now();
    let root = root.canonicalize()?;
    let manifest = fs::read(root.join("manifest.json"))?;
    let mut cached = ENGINE
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .map_err(|_| anyhow::anyhow!("speech engine lock poisoned"))?;
    if cached
        .as_ref()
        .is_none_or(|e| e.root != root || e.manifest != manifest || !e.assets_unchanged())
    {
        progress("بارگذاری و بررسی مدل…");
        *cached = None;
        *cached = Some(Engine::load(&root, manifest)?);
    }
    let mut report = cached.as_mut().unwrap().synthesize(out, text, seed, progress)?;
    report.wall_seconds = start.elapsed().as_secs_f64();
    Ok(report)
}
pub fn synthesize(root: &Path, out: &Path, text: &str, seed: u64) -> Result<Report> {
    synthesize_with_progress(root, out, text, seed, &mut |message| eprintln!("{message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sentence_boundaries_keep_punctuation() {
        assert_eq!(
            split_phrases("سلام سلام! حدس بزنید کی اومده؟\nآفرین"),
            vec!["سلام سلام!", "حدس بزنید کی اومده؟", "آفرین"]
        );
        assert_eq!(split_phrases("سلام!!! خوبی؟"), vec!["سلام!!!", "خوبی؟"]);
    }
    #[test]
    fn portable_noise_is_finite_and_reproducible() {
        let a = noise(43, 51);
        assert_eq!(a, noise(43, 51));
        assert_ne!(a, noise(44, 51));
        assert!(a.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(a.len(), 32);
        assert_eq!(a[0].len(), 408);
    }
    #[test]
    fn silence_is_not_fabricated_speech() {
        assert!(postprocess(&vec![0.; 24000], 0.05).is_empty());
    }
}
