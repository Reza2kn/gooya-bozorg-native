import argparse,json
from pathlib import Path
import numpy as np,torch,onnx
from transformers import HiggsAudioV2TokenizerModel
class Decoder(torch.nn.Module):
 def __init__(self,m):super().__init__();self.model=m
 def forward(self,codes):return self.model.decode(codes,return_dict=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6)
 m=HiggsAudioV2TokenizerModel.from_pretrained(a.model).eval();wrapped=Decoder(m);torch.manual_seed(771);codes=torch.randint(0,1024,(1,8,16))
 with torch.no_grad():ref=wrapped(codes).numpy()
 np.save(a.out/'reference.npy',ref);(a.out/'inputs.json').write_text(json.dumps([{'shape':list(codes.shape),'dtype':'int64','data':codes.flatten().tolist()}]))
 torch.onnx.export(wrapped,(codes,),str(a.out/'decoder.onnx'),input_names=['codes'],output_names=['audio'],opset_version=17,dynamo=False,dynamic_axes={'codes':{2:'frames'},'audio':{2:'samples'}})
 onnx.checker.check_model(str(a.out/'decoder.onnx'));print('Decoder exported',ref.shape,flush=True)
if __name__=='__main__':main()
