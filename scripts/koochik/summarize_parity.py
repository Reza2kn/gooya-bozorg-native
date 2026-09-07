"""Fail closed on missing or failing per-case evidence; never align waveforms."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--candidate',default='native-lossless');p.add_argument('--out',type=Path,required=True);a=p.parse_args();suite=json.loads(a.suite.read_text());results=[]
for x in suite['texts']:
 case=a.cases/x['id'];r={'id':x['id'],'text':x['text'],'missing':[]};assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256']
 for key,path in [('generation',case/a.candidate/'report.json'),('waveform',case/a.candidate/'waveform-parity.json'),('frontend',case/'frontend-parity.json')]:
  if path.exists():r[key]=json.loads(path.read_text())
  else:r['missing'].append(str(path))
 r['passes']=bool(not r['missing'] and r['generation']['full_32_step_code_agreement']>.98 and r['frontend']['exact'] and r['waveform'].get('raw',{}).get('passes',False) and r['waveform'].get('processed',{}).get('passes',False));results.append(r)
out={'source': 'Reza2kn/Gooya-Koochik-v2.0-exp','source_revision':suite['source_revision'],'source_runtime':'PyTorch 2.11.0 FP32 CPU','candidate_runtime':'tract 0.23.4 FP32 CPU; zstd lossless storage','threshold_strictly_greater_than':.98,'scope':suite['scope'],'cases':results,'all_neural_and_audio_gates_pass':all(r['passes'] for r in results),'consumer_smoke_required_separately':True};a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2));print([(r['id'],r['passes'])for r in results]);raise SystemExit(0 if out['all_neural_and_audio_gates_pass'] else 1)
