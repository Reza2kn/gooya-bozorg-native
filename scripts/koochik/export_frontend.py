import argparse,json
from pathlib import Path
import torch,numpy as np,onnx
from transformers import AutoModelForSeq2SeqLM
class Encoder(torch.nn.Module):
 def __init__(self,m):super().__init__();self.encoder=m.encoder
 def forward(self,ids,mask):return self.encoder(input_ids=ids,attention_mask=mask,return_dict=False)[0]
class Decoder(torch.nn.Module):
 def __init__(self,m):super().__init__();self.model=m
 def forward(self,ids,hidden,mask):return self.model(decoder_input_ids=ids,encoder_outputs=(hidden,),attention_mask=mask,use_cache=False,return_dict=False)[0][:,-1,:]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6);m=AutoModelForSeq2SeqLM.from_pretrained(a.model).eval();m.set_attn_implementation('eager')
 ids=torch.tensor([[b+3 for b in 'سلام'.encode()]],dtype=torch.int64);mask=torch.ones_like(ids);enc=Encoder(m)
 with torch.no_grad():hidden=enc(ids,mask)
 for name,wrapped,inputs,names,axes in [
  ('encoder',enc,(ids,mask),['ids','mask'],{'ids':{0:'batch',1:'source'},'mask':{0:'batch',1:'source'},'output':{0:'batch',1:'source'}}),
  ('decoder',Decoder(m),(torch.zeros((5,3),dtype=torch.int64),hidden.repeat(5,1,1),mask.repeat(5,1)),['ids','hidden','mask'],{'ids':{1:'target'},'hidden':{1:'source'},'mask':{1:'source'}})]:
  with torch.no_grad():ref=wrapped(*inputs).numpy()
  np.save(a.out/(name+'-reference.npy'),ref);(a.out/(name+'-inputs.json')).write_text(json.dumps([{'shape':list(x.shape),'dtype':str(x.numpy().dtype),'data':x.numpy().ravel().tolist()} for x in inputs]))
  torch.onnx.export(wrapped,inputs,str(a.out/(name+'.onnx')),input_names=names,output_names=['output'],opset_version=17,dynamo=False,dynamic_axes=axes);onnx.checker.check_model(str(a.out/(name+'.onnx')));print(name,'exported',ref.shape,flush=True)
if __name__=='__main__':main()
