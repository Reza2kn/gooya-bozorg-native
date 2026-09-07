"""Package grouped integer storage; the native graph computes in FP32."""
import argparse, hashlib, json, shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--reference-bundle',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--allowed-ids',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
for name in ['decoder.onnx','frontend/encoder.onnx','frontend/decoder.onnx','frontend/overlay.json','tokenizer.json','voice.json']:
 dst=a.out/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(a.reference_bundle/name,dst)
for name in ['step.onnx','step.weights.zst']:
 shutil.copy2(a.weights/name,a.out/name)
shutil.copy2(a.allowed_ids,a.out/'allowed_text_ids.json')
def asset(path,name):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=name,bytes=path.stat().st_size,sha256=h.hexdigest())
files=[asset(f,str(f.relative_to(a.out))) for f in sorted(a.out.rglob('*')) if f.is_file() and 'cache' not in f.relative_to(a.out).parts and f.name!='manifest.json']
m=dict(schema='gooya.koochik.tract/v2',source='Reza2kn/Gooya-Koochik-v2.0-exp',source_revision='537bb48320fd657415eef5b4fb6796b6cebab195',graph_format='onnx-f32',compression=json.loads((a.weights/'compression.json').read_text())['method'],runtime='tract CPU; integer storage dequantized to FP32',files=files,expanded_model=asset(a.weights/'step.weights','step.weights'),validated=False)
(a.out/'manifest.json').write_text(json.dumps(m,indent=2)); print('Bundle bytes',sum(f['bytes'] for f in files))
