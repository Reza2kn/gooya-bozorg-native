"""Capture complete source generation, exact per-step noise, and native inputs."""
import argparse,json,sys,types
from pathlib import Path
import numpy as np,torch,soundfile as sf
import omnivoice.models.omnivoice as om

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--text',default='سلام، حالت چطوره؟');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.model.resolve()));from gooya import Gooya
 torch.set_num_threads(6);g=Gooya.from_pretrained(a.model,'cpu');model=g.model;original=model.forward;noise=[];steps=[0];tokens=[]
 def forward(*args,**kwargs):
  result=original(*args,**kwargs)
  if steps[0] in [0,15,31]:
   s=a.out/f'step-{steps[0]:02d}';s.mkdir(exist_ok=True);ids=kwargs['input_ids'];mask=kwargs['attention_mask'];mask=torch.where(mask,0.,torch.finfo(torch.float32).min);pos=torch.arange(ids.shape[-1]).repeat(ids.shape[0],1)
   inputs=[]
   for val in [ids,kwargs['audio_mask'],mask,pos]:
    v=val.cpu().numpy();inputs.append({'shape':list(v.shape),'dtype':str(v.dtype),'data':v.ravel().tolist()})
   (s/'inputs.json').write_text(json.dumps(inputs));np.save(s/'logits.npy',result.logits.cpu().numpy())
  steps[0]+=1;return result
 model.forward=forward
 def gumbel(logits,temp):
  u=torch.rand_like(logits);n=-torch.log(-torch.log(u+1e-10)+1e-10);noise.append(n.cpu().numpy());return logits/temp+n
 om._gumbel_sample=gumbel
 iterative=model._generate_iterative
 def capture(task,config):
  result=iterative(task,config);tokens.extend(result)
  (a.out/'task.json').write_text(json.dumps({'text':a.text,'paced_text':task.texts[0],'target_tokens':task.target_lens[0],'num_step':config.num_step,'guidance_scale':config.guidance_scale,'t_shift':config.t_shift,'position_temperature':config.position_temperature,'layer_penalty_factor':config.layer_penalty_factor,'seed':43,'source_model':'merged continuation-200 FP32 runtime'},ensure_ascii=False,indent=2));return result
 model._generate_iterative=capture
 g.speak(a.text,str(a.out/'source.wav'))
 np.save(a.out/'noise.npy',np.stack(noise));np.save(a.out/'source_codes.npy',tokens[0].cpu().numpy())
 # JSON keeps the Rust harness free of a NumPy runtime dependency.
 (a.out/'decode.json').write_text(json.dumps({'target_tokens':tokens[0].shape[-1],'noise':np.stack(noise).reshape(32,-1).tolist(),'source_codes':tokens[0].cpu().reshape(-1).tolist()}))
 print('Captured full generation',steps[0],flush=True)
if __name__=='__main__':main()
