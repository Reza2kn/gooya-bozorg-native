"""Write the user-facing model card only after the complete Shenava gate passes."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--native-cases',type=Path,required=True);p.add_argument('--candidate',required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--recognizer',type=Path,required=True);a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['passes'] and r['cases']==12 and r['word_parity']>.98,'Full Shenava gate has not passed';m=json.loads((a.bundle/'manifest.json').read_text());size=sum(x['bytes']for x in m['files']);steps=m.get('inference_steps',32);compression=m['compression'];
for case in r['results']:
 receipt=json.loads((a.native_cases/case['id']/a.candidate/'report.json').read_text())
 assert receipt['bundle_manifest']['files']==m['files'], 'Evaluated assets differ from release assets'
 assert int(receipt.get('steps',32))==steps, 'Evaluated sampler differs from release sampler'
 assert not receipt.get('fp16_linear',False), 'Rejected FP16 runtime cannot be promoted'
m['validated']=True;m['validation']={'metric':r['metric'],'word_parity':r['word_parity'],'word_errors':r['word_errors'],'word_total':r['word_total'],'cases':r['cases'],'scope':'finite twelve-phrase development suite'};(a.bundle/'manifest.json').write_text(json.dumps(m,indent=2));shutil.copy2(a.evaluation,a.bundle/'shenava-parity.json');shutil.copy2(a.recognizer,a.bundle/'shenava-recognizer.json')
card=f'''---
model_name: Gooya Koochik v2.0-exp — Native Rust
language:
- fa
pipeline_tag: text-to-speech
library_name: tract
base_model: Reza2kn/Gooya-Koochik-v2.0-exp
base_model_relation: quantized
tags:
- rust
- tract
- onnx
- quantized
- persian
- experimental
---

# Gooya Koochik v2.0-exp — Native Rust

**Persian text in, WAV out. One {size/1e6:.1f} MB native bundle.** Includes Negara v7.1, the merged listener-selected continuation-200 speech model, the codec decoder and the default synthetic voice. No adapters or Python inference runtime are needed.

## Load and test

Use the Koochik integration in [gooya-bozorg-native](https://github.com/Reza2kn/gooya-bozorg-native/pull/2):

```sh
cargo build --release --no-default-features --manifest-path desktop/Cargo.toml --bin gooya_koochik
./desktop/target/release/gooya_koochik --download ./koochik
./desktop/target/release/gooya_koochik ./koochik output.wav 'سلام، حالت چطوره؟'
```

The repository is private. The downloader uses `HF_TOKEN` or your existing Hugging Face token cache. It resolves a commit, downloads that immutable revision and checks all asset SHA-256 hashes. In the desktop app, use the Koochik download button and select **Gooya Koochik v2.0-exp**.

## What is compressed

The speech weights use **{compression}**, with a compact text embedding table (3,190 retained rows). Retained rows are selected without changing their original values before quantization. The native frontend accepts spaced ASCII phonemes and rejects unsupported text-token IDs. The weight payload is archived with Zstandard. Codec and frontend weights remain FP32.

Integer weights are dequantized to **FP32 arithmetic in memory**. This is a download-size reduction, not a claim of low-bit activations or a sub-gigabyte RAM footprint. Allow several GB of working RAM. The app retains its last speech and codec plans between requests and displays preparation and generation progress. The frontend compiles its decoder once per phrase.

## Measured transcription parity

Using the same running Shenava recognizer for both source and compressed audio:

- **{100*r['word_parity']:.3f}% paired transcription agreement**, measured as `1 − corpus WER`.
- {r['word_errors']} differing words out of {r['word_total']} source-transcript words across {r['cases']} frozen Persian phrases.
- Source and candidate use the same reference voice and phoneme text. The original recordings use 32 passes; this candidate uses {steps}. Explicit captured noise is replayed for evaluation. When fewer than 32 passes are used, this measures compression plus the faster sampler against the original recordings.
- Recognizer: Shenava 0.1.1, greedy decoder, sentencepiece-v2; no expected text or hotwords supplied. Exact binary/model/token hashes and response metadata are in `shenava-recognizer.json` and `shenava-parity.json`.

This passes the requested **strictly greater than 98% transcription-parity gate on this finite development suite**. It is not a claim of 98% perceived quality, speaker similarity or accuracy on unseen text. Raw and normalized transcripts and per-phrase errors against the requested text are retained. Internal codec-token identity and samplewise waveform cosine are diagnostics, not the acceptance metric. Subjective pronunciation, pacing, timbre and prosody review remains useful.

The original FP32 Rust implementation independently matched every final audio code in all twelve source comparisons. Source inference for these comparisons used PyTorch FP32 CPU, not CUDA BF16 arithmetic.

## Scope

Experimental default-voice Persian TTS, 24 kHz mono, {steps} generation steps, speed 0.85. Sentence punctuation controls splitting; the runtime inserts a 350 ms gap between sentences. Phrases estimated above ten seconds are rejected and should be split. Custom reference recording/voice cloning and long-form internal chunking are not implemented in the native engine. Seed 43 in the native app uses SplitMix64 and is not equivalent to PyTorch's seed 43.

## Provenance

Source: `Reza2kn/Gooya-Koochik-v2.0-exp` at `537bb48320fd657415eef5b4fb6796b6cebab195`, the merged continuation-200 checkpoint. Quantization does not retrain or rewrite its labels. Original OmniVoice implementation: `k2-fsa/OmniVoice` commit `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`. Original model and dataset terms continue to apply; this release does not grant additional rights over training data or voices.
'''
(a.bundle/'README.md').write_text(card);print('Wrote model card and validation metadata')
