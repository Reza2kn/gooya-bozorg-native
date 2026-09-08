import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
for x in json.loads(a.suite.read_text())['texts']:
 out=a.out/x['id'];out.mkdir(parents=True,exist_ok=True)
 if (out/'decode.json').exists():continue
 with (out/'capture.log').open('w') as log:subprocess.run([sys.executable,str(Path(__file__).with_name('capture_reference.py')),'--model',str(a.model),'--out',str(out),'--text',x['text']],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Captured',x['id'],flush=True)
