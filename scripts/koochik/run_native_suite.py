"""Run every sealed case through Rust and retain individual failure evidence."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--name',default='native-lossless');a=p.parse_args()
for x in json.loads(a.suite.read_text())['texts']:
 case=a.cases/x['id'];out=case/a.name;out.mkdir(exist_ok=True)
 if (out/'report.json').exists():continue
 if not (case/'decode.json').exists():raise RuntimeError('Missing source capture: '+x['id'])
 binary_sha=hashlib.sha256(a.binary.read_bytes()).hexdigest()
 start=time.monotonic()
 with (out/'run.log').open('w') as log:subprocess.run([str(a.binary),str(a.bundle),str(a.bundle/'decoder.onnx'),str(case),str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
 r=json.loads((out/'report.json').read_text());r['binary_sha256']=binary_sha;r['wall_seconds']=time.monotonic()-start;r['bundle_manifest']=json.loads((a.bundle/'manifest.json').read_text());(out/'report.json').write_text(json.dumps(r,indent=2));print(x['id'],r['full_32_step_code_agreement'],r['wall_seconds'],flush=True)
