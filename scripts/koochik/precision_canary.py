"""Sequential native precision screening with fresh paired Shenava transcripts."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
from shenava_parity import transcribe, normalize, distance
p=argparse.ArgumentParser()
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--cases',type=Path,required=True)
p.add_argument('--suite',type=Path,required=True)
p.add_argument('--scope',choices=['mlp','attention','trunk','all'],required=True)
p.add_argument('--endpoint',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
texts=json.loads(a.suite.read_text())['texts'];order=[0,9,10]+[i for i in range(len(texts)) if i not in [0,9,10]]
env=dict(os.environ,GOOYA_EXPERIMENTAL_FP16_SCOPE=a.scope)
identity=None;results=[];binary_sha=hashlib.sha256(a.binary.read_bytes()).hexdigest()
for index in order:
 item=texts[index];case=a.cases/item['id'];out=a.out/item['id'];out.mkdir(exist_ok=True)
 start=time.monotonic()
 with (out/'run.log').open('w') as log:
  subprocess.run([str(a.binary),str(a.bundle),str(a.bundle/'decoder.onnx'),str(case),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 row={'id':item['id'],'requested_text':item['text'],'wall_seconds':time.monotonic()-start,'scope':a.scope,'binary_sha256':binary_sha,'native_report':json.loads((out/'report.json').read_text())}
 for side,path in [('source',case/'source.wav'),('candidate',out/'processed.wav')]:
  row[side]=transcribe(path,a.endpoint)
  observed={k:row[side]['response'].get(k)for k in ['backend','decoder','version','decoder_revision']}
  if identity is None:identity=observed
  if identity!=observed:raise RuntimeError('Recognizer changed')
 ref=row['source']['normalized_transcript'];hyp=row['candidate']['normalized_transcript'];expected=normalize(item['text'])
 row.update(word_errors=distance(ref.split(),hyp.split()),word_total=len(ref.split()),source_vs_requested_word_errors=distance(expected.split(),ref.split()),candidate_vs_requested_word_errors=distance(expected.split(),hyp.split()),nonempty=bool(ref and hyp))
 (out/'shenava.json').write_text(json.dumps(row,ensure_ascii=False,indent=2));results.append(row)
 errors=sum(r['word_errors']for r in results)
 summary={'scope':a.scope,'cases':len(results),'full_suite_cases':len(texts),'word_errors':errors,'word_total':sum(r['word_total']for r in results),'passes':False,'results':results,'decision':'in progress'}
 if errors>=2 or not row['nonempty']:summary['decision']='reject: already exceeds full-suite allowance of one difference in 67 words'
 elif len(results)==len(texts):
  score=1-errors/max(1,summary['word_total']);passed=summary['word_total']==67 and score>.98 and all(r['nonempty'] for r in results)
  summary.update(passes=passed,decision='full suite passed' if passed else 'full suite rejected or source denominator changed',word_parity=score)
 (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print(item['id'],ref,'|',hyp,'cumulative errors',errors,flush=True)
 if errors>=2 or not row['nonempty']:break
