"""GPU screening only. Every selected candidate must still pass native tract evaluation."""
import argparse,json,math,time
from pathlib import Path
import numpy as np,torch,soundfile as sf,onnxruntime as ort
from omnivoice.models.omnivoice import OmniVoice,OmniVoiceGenerationConfig
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bits',type=int,default=6);p.add_argument('--steps',type=int,nargs='+',default=[32,24,16]);p.add_argument('--ids',nargs='+',default=['case-00','case-03','case-06','case-09','case-10']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
assert torch.backends.mps.is_available();torch.set_num_threads(2)
model=OmniVoice.from_pretrained(a.model,train=True,dtype=torch.float32,device_map='cpu').eval();model.llm.set_attn_implementation('eager');model.llm.config.use_cache=False;model.sampling_rate=24000
with torch.no_grad():
 for name,param in model.named_parameters():
  if param.ndim!=2 or any(s in name for s in ['embed_tokens','audio_embeddings','audio_heads']):continue
  x=param.numpy().reshape(param.shape[0],-1,32);limit=(1<<(a.bits-1))-1;scale=np.maximum(np.max(np.abs(x),axis=-1,keepdims=True)/limit,np.finfo(np.float32).tiny).astype(np.float32);x[:]=np.clip(np.rint(x/scale),-limit,limit)*scale
model.to('mps');torch.mps.synchronize();print('MPS model ready',flush=True)
options=ort.SessionOptions();options.intra_op_num_threads=2;codec=ort.InferenceSession(str(a.bundle/'decoder.onnx'),sess_options=options,providers=['CPUExecutionProvider']);rms=json.loads((a.bundle/'voice.json').read_text())['rms']
with torch.inference_mode():
 for steps in a.steps:
  cfg=OmniVoiceGenerationConfig(num_step=steps)
  for case_id in a.ids:
   case=a.cases/case_id;dest=a.out/f's{steps}'/case_id;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'report.json').exists():continue
   fixture=json.loads((case/'decode.json').read_text());t=fixture['target_tokens'];n=8*t
   raw=json.loads((case/'step-00/inputs.json').read_text());inputs=[torch.from_numpy(np.asarray(x['data'],dtype=x['dtype']).reshape(x['shape'])).to('mps') for x in raw];ids,mask,attention,pos=inputs;s=ids.shape[-1]
   tokens=torch.full((8,t),1024,dtype=torch.int64,device='mps');remaining=n;start=time.monotonic()
   ts=np.linspace(0,1,steps+1,dtype=np.float32);ts=np.float32(.1)*ts/(np.float32(1)+(np.float32(.1)-np.float32(1))*ts)
   for step in range(steps):
    logits=model(input_ids=ids,audio_mask=mask,attention_mask=attention,position_ids=pos).logits.float()
    pred,scores=model._predict_tokens_with_scoring(logits[0:1,:,s-t:],logits[1:2,:,:t],cfg)
    scores=(scores-torch.arange(8,device='mps').reshape(1,8,1)*5)/5+torch.tensor(fixture['noise'][step],dtype=torch.float32,device='mps').reshape(1,8,t)
    scores.masked_fill_(tokens.unsqueeze(0)!=1024,-float('inf'));k=remaining if step==steps-1 else min(remaining,math.ceil(n*(float(ts[step+1])-float(ts[step]))))
    selected=torch.topk(scores.flatten(),k).indices;tokens.flatten()[selected]=pred.flatten()[selected];remaining-=k;ids[0,:,s-t:]=tokens;ids[1,:,:t]=tokens
   torch.mps.synchronize();elapsed=time.monotonic()-start;codes=tokens.cpu().numpy();assert remaining==0 and not (codes==1024).any();np.save(dest/'codes.npy',codes)
   audio=codec.run(None,{codec.get_inputs()[0].name:codes[None]})[0].reshape(1,-1);audio=model._post_process_audio(audio,rms,cfg);sf.write(dest/'processed.wav',audio.reshape(-1),24000)
   report=dict(screening_only=True,runtime='PyTorch MPS + ORT CPU codec; not native acceptance',bits=a.bits,preserved_io=True,steps=steps,seconds=elapsed,source_code_agreement=float((codes.reshape(-1)==np.asarray(fixture['source_codes'])).mean()))
   (dest/'report.json').write_text(json.dumps(report,indent=2));print(steps,case_id,round(elapsed,2),flush=True)
