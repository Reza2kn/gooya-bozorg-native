"""Build the compact native bundle from a dynamic Q4 NNEF export."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference-bundle',type=Path,required=True);p.add_argument('--q4',type=Path,required=True);p.add_argument('--allowed-ids',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
for f in a.reference_bundle.rglob('*'):
 if not f.is_file() or 'cache' in f.relative_to(a.reference_bundle).parts or f.name in ['step.onnx','step.weights.zst','manifest.json']:continue
 dst=a.out/f.relative_to(a.reference_bundle);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
shutil.copy2(a.q4.with_name(a.q4.name+'.zst'),a.out/(a.q4.name+'.zst'));shutil.copy2(a.allowed_ids,a.out/'allowed_text_ids.json')
def asset(path,name):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=name,bytes=path.stat().st_size,sha256=h.hexdigest())
files=[asset(f,str(f.relative_to(a.out)))for f in sorted(a.out.rglob('*'))if f.is_file() and 'cache'not in f.relative_to(a.out).parts and f.name!='manifest.json']
m=dict(schema='gooya.koochik.tract/v2',source='Reza2kn/Gooya-Koochik-v2.0-exp',source_revision='537bb48320fd657415eef5b4fb6796b6cebab195',graph_format='nnef-q4',compression='Q4_0 matrix weights; compact text vocabulary; Zstandard archive',runtime='tract CPU; Q4 weights unpacked to FP32 for whole-sequence matrix kernels',files=files,expanded_model=asset(a.q4,a.q4.name),validated=False)
(a.out/'manifest.json').write_text(json.dumps(m,indent=2));print('Bundle bytes',sum(f['bytes']for f in files))
