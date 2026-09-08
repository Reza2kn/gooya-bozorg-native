"""Assemble a lossless tract candidate with immutable source provenance."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,soundfile as sf
p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
for src,dst in [('step-f32/step.onnx','step.onnx'),('step-f32/step.weights.zst','step.weights.zst'),('codec-f32/decoder.onnx','decoder.onnx'),('frontend-f32/encoder.onnx','frontend/encoder.onnx'),('frontend-f32/decoder.onnx','frontend/decoder.onnx'),('frontend-f32/overlay.json','frontend/overlay.json')]:
 d=a.out/dst;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.artifacts/src,d)
shutil.copy2(a.source/'tokenizer.json',a.out/'tokenizer.json')
f=json.loads((a.artifacts/'parity/case-00/step-00/inputs.json').read_text());ids=np.array(f[0]['data']).reshape(f[0]['shape']);mask=np.array(f[1]['data']).reshape(f[1]['shape']);t=json.loads((a.artifacts/'parity/case-00/task.json').read_text())['target_tokens'];start=int(np.where(mask[0])[0][0]);codes=ids[0,:,start:-t];voice=json.loads((a.source/'default_voice.json').read_text());wav,sr=sf.read(a.source/'default_voice.wav',dtype='float32');voice.update(frames=codes.shape[1],codes=codes.ravel().tolist(),rms=float(np.sqrt(np.mean(wav**2))));(a.out/'voice.json').write_text(json.dumps(voice,ensure_ascii=False,indent=2))
def asset(p,name):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=name,bytes=p.stat().st_size,sha256=h.hexdigest())
files=[asset(p,str(p.relative_to(a.out))) for p in sorted(a.out.rglob('*')) if p.is_file() and p.name!='manifest.json' and 'cache' not in p.parts]
m=dict(schema='gooya.koochik.tract/v1',source='Reza2kn/Gooya-Koochik-v2.0-exp',source_revision='537bb48320fd657415eef5b4fb6796b6cebab195',compression='lossless zstd FP32 storage; not low-bit arithmetic',runtime='tract 0.23.4 FP32 CPU',files=files,expanded_weights=asset(a.artifacts/'step-f32/step.weights','step.weights'),validated=False)
(a.out/'manifest.json').write_text(json.dumps(m,indent=2));print('Bundle bytes',sum(x['bytes'] for x in files),flush=True)
