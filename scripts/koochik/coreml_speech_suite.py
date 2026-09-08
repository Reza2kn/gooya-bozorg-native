"""Full Core ML speech screening with frozen noise and paired Shenava receipts."""
import argparse,json,time,math,types,hashlib
from pathlib import Path
import numpy as np,torch,coremltools as ct,onnxruntime as ort,soundfile as sf
from omnivoice.models.omnivoice import OmniVoice,OmniVoiceGenerationConfig
from coreml_cache import load_model
from shenava_parity import transcribe,normalize,distance
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--endpoint',required=True);p.add_argument('--steps',type=int,choices=[24,32],default=32);p.add_argument('--pad-to',type=int);p.add_argument('--fp32-package',type=Path);p.add_argument('--fp32-prefix',type=int,default=0);p.add_argument('--first',type=int,nargs='+',default=[0,9,10]);p.add_argument('--fp32-suffix',type=int,default=0);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
start=time.monotonic();model,reused=load_model(a.package,ct.ComputeUnit.ALL);fp32=load_model(a.fp32_package,ct.ComputeUnit.ALL)[0] if a.fp32_package else None;assert 0<=a.fp32_prefix+a.fp32_suffix<=a.steps and (a.fp32_prefix+a.fp32_suffix==0 or fp32 is not None);load=time.monotonic()-start;print('Core ML loaded',load,flush=True)
options=ort.SessionOptions();options.intra_op_num_threads=2;codec=ort.InferenceSession(str(a.bundle/'decoder.onnx'),sess_options=options,providers=['CPUExecutionProvider']);rms=json.loads((a.bundle/'voice.json').read_text())['rms'];cfg=OmniVoiceGenerationConfig(num_step=a.steps);dummy=types.SimpleNamespace(config=types.SimpleNamespace(audio_mask_id=1024),sampling_rate=24000)
items=json.loads(a.suite.read_text())['texts'];order=a.first+[i for i in range(len(items))if i not in a.first];assert sorted(order)==list(range(len(items)));results=[];identity=None
for i in order:
 item=items[i];case=a.cases/item['id'];out=a.out/item['id'];out.mkdir(exist_ok=True);f=json.loads((case/'decode.json').read_text());t=f['target_tokens'];n=8*t;steps=a.steps
 rows=json.loads((case/'step-00/inputs.json').read_text());inputs={name:np.array(r['data'],dtype=np.float32 if r['dtype']=='float32' else np.int32).reshape(r['shape'])for name,r in zip(['ids','mask','attention','pos'],rows)};s=inputs['ids'].shape[-1]
 if a.pad_to:
  length=a.pad_to;assert length>=s
  old={k:v.copy()for k,v in inputs.items()}
  inputs['ids']=np.pad(inputs['ids'],((0,0),(0,0),(0,length-s)),constant_values=1024)
  inputs['mask']=np.pad(inputs['mask'],((0,0),(0,length-s)),constant_values=1)
  inputs['pos']=np.pad(inputs['pos'],((0,0),(0,length-s)),constant_values=0)
  att=np.full((2,1,length,length),-np.inf,dtype=np.float32);att[:,:,:s,:s]=old['attention']
  for j in range(s,length):att[:,:,j,j]=0
  inputs['attention']=att
  assert np.array_equal(inputs['ids'][:,:,:s],old['ids']) and np.array_equal(att[:,:,:s,:s],old['attention'])
 tokens=torch.full((8,t),1024,dtype=torch.int64);remaining=n;timings=[];start=time.monotonic();ts=np.linspace(0,1,steps+1,dtype=np.float32);ts=np.float32(.1)*ts/(np.float32(1)+(np.float32(.1)-np.float32(1))*ts)
 for step in range(steps):
  tic=time.monotonic();raw=next(iter((fp32 if step<a.fp32_prefix or step>=steps-a.fp32_suffix else model).predict(inputs).values()));logits=torch.from_numpy(raw.astype(np.float32));assert list(logits.shape)==[2,8,a.pad_to or s,1025]
  pred,scores=OmniVoice._predict_tokens_with_scoring(dummy,logits[0:1,:,s-t:s],logits[1:2,:,:t],cfg);scores=(scores-torch.arange(8).reshape(1,8,1)*5)/5+torch.tensor(f['noise'][step],dtype=torch.float32).reshape(1,8,t);scores.masked_fill_(tokens.unsqueeze(0)!=1024,-float('inf'));k=remaining if step==steps-1 else min(remaining,math.ceil(n*(float(ts[step+1])-float(ts[step]))));selected=torch.topk(scores.flatten(),k).indices;tokens.flatten()[selected]=pred.flatten()[selected];remaining-=k;inputs['ids'][0,:,s-t:s]=tokens.numpy();inputs['ids'][1,:,:t]=tokens.numpy();timings.append(time.monotonic()-tic)
  if step%8==7:print(item['id'],'step',step+1,round(timings[-1],3),flush=True)
 elapsed=time.monotonic()-start;assert remaining==0 and not (tokens==1024).any();np.save(out/'codes.npy',tokens.numpy());audio=codec.run(None,{codec.get_inputs()[0].name:tokens.numpy()[None]})[0].reshape(1,-1);audio=OmniVoice._post_process_audio(dummy,audio,rms,cfg);sf.write(out/'processed.wav',audio.reshape(-1),24000)
 r={'id':item['id'],'requested_text':item['text'],'steps':steps,'fp32_suffix':a.fp32_suffix,'fp32_prefix':a.fp32_prefix,'original_sequence_length':s,'padded_sequence_length':a.pad_to,'generation_seconds':elapsed,'pass_seconds':timings,'audio_seconds':audio.size/24000}
 for side,path in [('source',case/'source.wav'),('candidate',out/'processed.wav')]:
  r[side]=transcribe(path,a.endpoint);observed={k:r[side]['response'].get(k)for k in ['backend','decoder','version','decoder_revision']}
  if identity is None:identity=observed
  if observed!=identity:raise RuntimeError('Shenava identity changed')
 ref=r['source']['normalized_transcript'];hyp=r['candidate']['normalized_transcript'];expected=normalize(item['text']);r.update(word_errors=distance(ref.split(),hyp.split()),word_total=len(ref.split()),source_vs_requested_word_errors=distance(expected.split(),ref.split()),candidate_vs_requested_word_errors=distance(expected.split(),hyp.split()),nonempty=bool(ref and hyp));(out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));results.append(r);errors=sum(x['word_errors']for x in results);total=sum(x['word_total']for x in results)
 summary={'runtime':'Core ML prediction; source Python sampler; ORT CPU codec','screening_only':True,'compiled_cache_reused':reused,'fp32_suffix':a.fp32_suffix,'fp32_prefix':a.fp32_prefix,'cases':len(results),'full_suite_cases':len(items),'steps':steps,'load_seconds':load,'word_errors':errors,'word_total':total,'word_parity':1-errors/max(1,total),'passes':len(results)==len(items) and total==67 and errors<=1 and all(x['nonempty']for x in results),'results':results};(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(item['id'],ref,'|',hyp,'cumulative errors',errors,flush=True)
 if errors>=2 or not r['nonempty']:print('Rejected: exceeds frozen full-suite allowance',flush=True);break
