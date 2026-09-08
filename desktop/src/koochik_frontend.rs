//! Negara v7.1 ByT5 beam search and the release's phone formatter, in Rust.
use crate::koochik::{Graph, log_normalizer};
use anyhow::{Context, Result, ensure};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use tract_onnx::prelude::*;
use unicode_categories::UnicodeCategories;
pub struct Frontend {
    encoder: PathBuf,
    decoder: PathBuf,
    rules: HashMap<(String, String), String>,
}
fn words(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c.is_mark() || c == '\u{200c}' {
            word.push(c)
        } else if !word.is_empty() {
            result.push(std::mem::take(&mut word));
        }
    }
    if !word.is_empty() {
        result.push(word);
    }
    result
}
impl Frontend {
    pub fn load(path: &Path) -> Result<Self> {
        let data: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("overlay.json"))?)?;
        let mut rules = HashMap::new();
        for x in data["rules"].as_array().context("missing overlay rules")? {
            rules.insert(
                (
                    x["surface"].as_str().unwrap().into(),
                    x["raw"].as_str().unwrap().into(),
                ),
                x["target"].as_str().unwrap().into(),
            );
        }
        Ok(Self {
            encoder: path.join("encoder.onnx"),
            decoder: path.join("decoder.onnx"),
            rules,
        })
    }
    pub fn raw_phones(&self, text: &str) -> Result<String> {
        ensure!(
            !text.is_empty() && text.len() <= 160,
            "G2P input must have 1–160 UTF-8 bytes"
        );
        let ids: Vec<i64> = text.bytes().map(|b| b as i64 + 3).collect();
        let s = ids.len();
        let mask = vec![1i64; s];
        let ei = vec![
            Tensor::from_shape(&[1, s], &ids)?,
            Tensor::from_shape(&[1, s], &mask)?,
        ];
        let enc = Graph::load(&self.encoder, &ei)?.run(&ei)?;
        let h = enc[0].to_plain_array_view::<f32>()?;
        let mut hidden = Vec::new();
        for _ in 0..5 {
            hidden.extend(h.iter().copied());
        }
        let hidden = Tensor::from_shape(&[5, s, 512], &hidden)?;
        let mask = Tensor::from_shape(&[5, s], &vec![1i64; 5 * s])?;
        let decoder = Graph::load_g2p_decoder(&self.decoder, s)?;
        let mut beams = vec![vec![0i64]; 5];
        let mut scores = vec![-1e9f32; 5];
        scores[0] = 0.;
        let mut finished: Vec<(f32, Vec<i64>)> = Vec::new();
        for _step in 0..512 {
            let length = beams[0].len();
            let ids: Vec<i64> = beams.iter().flatten().copied().collect();
            let di = vec![
                Tensor::from_shape(&[5, length], &ids)?,
                hidden.clone(),
                mask.clone(),
            ];
            let out = decoder.run(&di)?;
            let arr = out[0].to_plain_array_view::<f32>()?;
            ensure!(arr.shape() == [5, 384], "unexpected G2P logits");
            let logits = arr.as_slice().unwrap();
            let mut candidates = Vec::with_capacity(5 * 384);
            for b in 0..5 {
                let row = &logits[b * 384..(b + 1) * 384];
                let z = log_normalizer(row);
                for (token, &value) in row.iter().enumerate() {
                    candidates.push((scores[b] + value - z, b, token as i64));
                }
            }
            candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let mut next = Vec::new();
            let mut next_scores = Vec::new();
            for (rank, (score, b, token)) in candidates.into_iter().take(10).enumerate() {
                if token == 1 {
                    if rank < 5 {
                        finished.push((score / length as f32, beams[b].clone()));
                    }
                    continue;
                }
                let mut seq = beams[b].clone();
                seq.push(token);
                next.push(seq);
                next_scores.push(score);
                if next.len() == 5 {
                    break;
                }
            }
            ensure!(next.len() == 5, "insufficient live beams");
            beams = next;
            scores = next_scores;
            if finished.len() >= 5 {
                break;
            }
        }
        ensure!(
            !finished.is_empty(),
            "G2P reached the length limit without EOS"
        );
        finished.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let bytes: Vec<u8> = finished[0]
            .1
            .iter()
            .filter_map(|&x| {
                if (3..259).contains(&x) {
                    Some((x - 3) as u8)
                } else {
                    None
                }
            })
            .collect();
        Ok(String::from_utf8(bytes)?.trim().to_string())
    }
    pub fn phones(&self, text: &str) -> Result<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        for word in text.split_whitespace() {
            ensure!(word.len() <= 160, "text token exceeds frontend limit");
            if !current.is_empty() && current.len() + 1 + word.len() > 160 {
                chunks.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        ensure!(!chunks.is_empty(), "empty text");
        let mut all = Vec::new();
        for chunk in chunks {
            let raw = self.raw_phones(&chunk)?;
            let mut ps: Vec<String> = raw.split_whitespace().map(String::from).collect();
            let ws = words(&chunk);
            if ps.len() == ws.len() {
                for (w, p) in ws.iter().zip(ps.iter_mut()) {
                    if let Some(v) = self.rules.get(&(w.clone(), p.clone())) {
                        *p = v.clone();
                    }
                    if w == "حدستون" {
                        *p = "hadsetun".into();
                    }
                }
            }
            all.extend(ps);
        }
        Ok(all.join(" "))
    }
}
pub fn paced(text: &str, phones: &str) -> String {
    let map = |c: char| match c {
        '،' => Some(','),
        '؛' => Some(';'),
        '؟' => Some('?'),
        ',' | ';' | ':' | '.' | '!' | '?' => Some(c),
        _ => None,
    };
    let mut marks = Vec::new();
    let mut count = 0;
    let mut in_word = false;
    for c in text.chars() {
        if let Some(p) = map(c) {
            if in_word {
                count += 1;
                in_word = false;
            }
            marks.push((count, p));
        } else if c.is_whitespace() {
            if in_word {
                count += 1;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }
    if in_word {
        count += 1;
    }
    let ps: Vec<_> = phones.split_whitespace().collect();
    if ps.is_empty() {
        return String::new();
    }
    let mut projected: HashMap<usize, Vec<char>> = HashMap::new();
    for (n, c) in marks {
        let at = if n == 0 {
            0
        } else if count == 0 || n >= count {
            ps.len()
        } else {
            ((n as f64 * ps.len() as f64 / count as f64).round_ties_even() as usize)
                .max(1)
                .min(ps.len().saturating_sub(1).max(1))
        };
        projected.entry(at).or_default().push(c);
    }
    let mut tokens = Vec::new();
    if let Some(cs) = projected.get(&0) {
        tokens.extend(cs.iter().map(char::to_string));
    }
    for (i, p) in ps.iter().enumerate() {
        tokens.extend(
            p.chars()
                .map(|c| if c == '?' { 'Q' } else { c })
                .map(|c| c.to_string()),
        );
        if let Some(cs) = projected.get(&(i + 1)) {
            tokens.extend(cs.iter().map(char::to_string));
        }
        if i + 1 < ps.len() {
            tokens.push("/".into());
        }
    }
    tokens.join(" ")
}
