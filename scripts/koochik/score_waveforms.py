"""Decode source codes once and score raw/processed native audio without alignment."""
import argparse,json
from pathlib import Path
import numpy as np,soundfile as sf,torch
from transformers import AutoModel
p=argparse.ArgumentParser();p.add_argument('--codec',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--candidate',default='native-f32');a=p.parse_args();torch.set_num_threads(2);model=AutoModel.from_pretrained(a.codec,trust_remote_code=True).eval()
for case in sorted(a.cases.glob('case-*')):
 candidate=case/a.candidate
 if not (candidate/'native.wav').exists():continue
 if not (case/'source-raw.wav').exists():
  with torch.inference_mode():raw=model.decode(torch.from_numpy(np.load(case/'source_codes.npy')).unsqueeze(0),return_dict=False)[0].numpy().reshape(-1)
  sf.write(case/'source-raw.wav',raw,24000,subtype='FLOAT')
 results={}
 for name,source,target in [('raw','source-raw.wav','native.wav'),('processed','source.wav','processed.wav')]:
  if not (candidate/target).exists():continue
  x,sr=sf.read(case/source);y,yr=sf.read(candidate/target);same=len(x)==len(y) and sr==yr;cos=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))) if same else None
  results[name]=dict(cosine=cos,same_length=same,source_samples=len(x),native_samples=len(y),finite=bool(np.isfinite(y).all()),passes=bool(same and cos>.98 and np.isfinite(y).all()))
 (candidate/'waveform-parity.json').write_text(json.dumps(results,indent=2));print(case.name,results,flush=True)
