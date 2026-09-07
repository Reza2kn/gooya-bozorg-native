"""Core ML speech screening with source sampler and CPU codec, not Rust acceptance."""
import argparse,json,time,math,types
from pathlib import Path
import numpy as np, torch, coremltools as ct, onnxruntime as ort,soundfile as sf
from omnivoice.models.omnivoice import OmniVoice,OmniVoiceGenerationConfig
from shenava_parity import transcribe,normalize,distance
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--endpoint',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
f=json.loads((a.case/'decode.json').read_text());t=f['target_tokens'];n=8*t;steps=24;cfg=OmniVoiceGenerationConfig(num_step=steps);dummy=types.SimpleNamespace(config=types.SimpleNamespace(audio_mask_id=1024),sampling_rate=24000)
rows=json.loads((a.case/'step-00/inputs.json').read_text());inputs={name:np.array(r['data'],dtype=np.float32 if r['dtype']=='float32' else np.int32).reshape(r['shape'])for name,r in zip(['ids','mask','attention','pos'],rows)};s=inputs['ids'].shape[-1]
start=time.monotonic();model=ct.models.MLModel(str(a.package),compute_units=ct.ComputeUnit.ALL);load=time.monotonic()-start;tokens=torch.full((8,t),1024,dtype=torch.int64);remaining=n;timings=[];start=time.monotonic()
ts=np.linspace(0,1,steps+1,dtype=np.float32);ts=np.float32(.1)*ts/(np.float32(1)+(np.float32(.1)-np.float32(1))*ts)
for step in range(steps):
 tic=time.monotonic();raw=next(iter(model.predict(inputs).values()));logits=torch.from_numpy(raw.astype(np.float32));pred,scores=OmniVoice._predict_tokens_with_scoring(dummy,logits[0:1,:,s-t:],logits[1:2,:,:t],cfg)
 scores=(scores-torch.arange(8).reshape(1,8,1)*5)/5+torch.tensor(f['noise'][step],dtype=torch.float32).reshape(1,8,t);scores.masked_fill_(tokens.unsqueeze(0)!=1024,-float('inf'));k=remaining if step==steps-1 else min(remaining,math.ceil(n*(float(ts[step+1])-float(ts[step]))))
 selected=torch.topk(scores.flatten(),k).indices;tokens.flatten()[selected]=pred.flatten()[selected];remaining-=k;inputs['ids'][0,:,s-t:]=tokens.numpy();inputs['ids'][1,:,:t]=tokens.numpy();timings.append(time.monotonic()-tic);print(step+1,round(timings[-1],3),flush=True)
elapsed=time.monotonic()-start;assert remaining==0 and not (tokens==1024).any();np.save(a.out/'codes.npy',tokens.numpy())
options=ort.SessionOptions();options.intra_op_num_threads=2;codec=ort.InferenceSession(str(a.bundle/'decoder.onnx'),sess_options=options,providers=['CPUExecutionProvider']);audio=codec.run(None,{codec.get_inputs()[0].name:tokens.numpy()[None]})[0].reshape(1,-1);rms=json.loads((a.bundle/'voice.json').read_text())['rms'];audio=OmniVoice._post_process_audio(dummy,audio,rms,cfg);sf.write(a.out/'processed.wav',audio.reshape(-1),24000)
r={'runtime':'Core ML speech graph; Python source sampler; ORT CPU codec','screening_only':True,'steps':steps,'load_seconds':load,'generation_seconds':elapsed,'pass_seconds':timings,'audio_seconds':audio.size/24000}
for side,path in [('source',a.case/'source.wav'),('candidate',a.out/'processed.wav')]:r[side]=transcribe(path,a.endpoint)
r['word_errors']=distance(r['source']['normalized_transcript'].split(),r['candidate']['normalized_transcript'].split());r['word_total']=len(r['source']['normalized_transcript'].split());(a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False),flush=True)
