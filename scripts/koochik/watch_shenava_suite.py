"""Score each native output when ready, with bounded waiting and persistent receipts."""
import argparse,json,time
from pathlib import Path
from shenava_parity import transcribe,normalize,distance
p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--candidate',default='native-q4');p.add_argument('--endpoint',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);pairs=json.loads(a.suite.read_text())['texts'];deadline=time.monotonic()+7200;identity=None
for pair in pairs:
 case=a.cases/pair['id'];out=a.out/(pair['id']+'.json');candidate=case/a.candidate/'processed.wav'
 while not (candidate.exists() and (case/a.candidate/'report.json').exists()):
  if time.monotonic()>deadline:raise TimeoutError('Native generation did not complete: '+pair['id'])
  time.sleep(5)
 if out.exists():continue
 row={'id':pair['id'],'requested_text':pair['text']}
 for side,path in [('source',case/'source.wav'),('candidate',candidate)]:
  row[side]=transcribe(path,a.endpoint);observed={k:row[side]['response'].get(k)for k in ['backend','decoder','version','decoder_revision']}
  if identity is None:identity=observed
  if identity!=observed:raise RuntimeError('Shenava settings changed')
 source=row['source']['normalized_transcript'];hyp=row['candidate']['normalized_transcript'];expected=normalize(pair['text']);we=distance(source.split(),hyp.split());ce=distance(source.replace(' ',''),hyp.replace(' ',''));row.update(word_errors=we,word_total=len(source.split()),char_errors=ce,char_total=len(source.replace(' ','')),word_parity=1-we/max(1,len(source.split())),source_vs_requested_word_errors=distance(expected.split(),source.split()),candidate_vs_requested_word_errors=distance(expected.split(),hyp.split()),nonempty=bool(source and hyp));out.write_text(json.dumps(row,ensure_ascii=False,indent=2));print(pair['id'],source,'|',hyp,'word errors',we,flush=True)
results=[json.loads((a.out/(p['id']+'.json')).read_text())for p in pairs]
identities={tuple(r[side]['response'].get(k)for k in ['backend','decoder','version','decoder_revision'])for r in results for side in ['source','candidate']}
if len(identities)!=1:raise RuntimeError('Shenava settings differ across persisted results')
total=sum(r['word_total']for r in results);errors=sum(r['word_errors']for r in results);score=1-errors/max(1,total);summary={'metric':'1 - corpus WER between paired Shenava transcripts','cases':len(results),'word_errors':errors,'word_total':total,'word_parity':score,'passes':bool(score>.98 and all(r['nonempty']for r in results)),'results':results};(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('Full suite:',score,'passes',summary['passes'],flush=True)
