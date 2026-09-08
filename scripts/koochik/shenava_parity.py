"""Paired Shenava transcript parity. Never send the expected text to the recognizer."""
import argparse,hashlib,json,re,time,unicodedata,urllib.request,uuid
from pathlib import Path

def normalize(s):
 s=unicodedata.normalize('NFKC',s or '').replace('\u200c',' ').replace('\u200b','').replace('ي','ی').replace('ك','ک')
 s=''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s)
 return re.sub(r'\s+',' ',s).strip()
def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]
def transcribe(path,endpoint):
 raw=path.read_bytes();assert raw[:4]==b'RIFF' and raw[8:12]==b'WAVE'
 boundary='gooya-parity-'+uuid.uuid4().hex
 body=(f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\noffline\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+raw+f'\r\n--{boundary}--\r\n'.encode()
 req=urllib.request.Request(endpoint,data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary});start=time.monotonic()
 with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
 return dict(audio_sha256=hashlib.sha256(raw).hexdigest(),response=result,raw_transcript=result.get('text',''),normalized_transcript=normalize(result.get('text','')),elapsed_seconds=time.monotonic()-start)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--endpoint',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);pairs=json.loads(a.manifest.read_text());results=[];identity=None
 for pair in pairs:
  output=a.out/(pair['id']+'.json');row={'id':pair['id'],'requested_text':pair['text'],'endpoint':a.endpoint}
  for side in ['source','candidate']:
   path=Path(pair[side]);row[side]=transcribe(path,a.endpoint)
   observed={k:row[side]['response'].get(k) for k in ['backend','decoder','version','decoder_revision']}
   if identity is None:identity=observed
   if observed!=identity:raise RuntimeError('Shenava identity/settings changed during paired evaluation')
  ref=row['source']['normalized_transcript'];hyp=row['candidate']['normalized_transcript'];expected=normalize(pair['text']);words=ref.split();chars=ref.replace(' ','');we=distance(words,hyp.split());ce=distance(chars,hyp.replace(' ',''));row.update(word_errors=we,word_total=len(words),char_errors=ce,char_total=len(chars),word_parity=1-we/max(1,len(words)),char_parity=1-ce/max(1,len(chars)),source_vs_requested_word_errors=distance(expected.split(),words),candidate_vs_requested_word_errors=distance(expected.split(),hyp.split()),nonempty=bool(ref and hyp));output.write_text(json.dumps(row,ensure_ascii=False,indent=2));results.append(row);print(pair['id'],ref,'|',hyp,'word parity',row['word_parity'],flush=True)
 total=sum(r['word_total']for r in results);errors=sum(r['word_errors']for r in results);score=1-errors/max(total,1);summary=dict(recognizer_response_identity=identity,metric='1 - corpus WER(candidate Shenava transcript, source Shenava transcript)',normalization='NFKC; ZWNJ to space; remove ZWSP; Arabic yeh/kaf to Persian; Unicode punctuation to spaces; collapse whitespace',word_errors=errors,word_total=total,word_parity=score,threshold_strictly_greater_than=.98,passes=bool(total and score>.98 and all(r['nonempty']for r in results)),cases=len(results),scope='finite paired suite; not a perceptual-quality or unseen-text guarantee',results=results);(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('Summary',score,flush=True)
if __name__=='__main__':main()
