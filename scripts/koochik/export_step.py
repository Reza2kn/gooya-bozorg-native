"""Export OmniVoice's bidirectional speech step with external weights for tract."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch,onnx
from onnx import helper,TensorProto
from omnivoice.models.omnivoice import OmniVoice

class Step(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,input_ids,audio_mask,attention_mask,position_ids):
        return self.model(input_ids=input_ids,audio_mask=audio_mask,attention_mask=attention_mask,position_ids=position_ids).logits

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6)
 model=OmniVoice.from_pretrained(str(a.model),train=True,dtype=torch.float32,device_map='cpu').eval();model.llm.set_attn_implementation('eager');model.llm.config.use_cache=False
 wrapped=Step(model).eval();length=32;ids=torch.full((2,8,length),1024,dtype=torch.int64);ids[:,:,0:8]=torch.arange(8).reshape(1,1,8)+151650
 am=torch.ones((2,length),dtype=torch.bool);am[:,:8]=False;attention=torch.zeros((2,1,length,length));pos=torch.arange(length).repeat(2,1)
 args=(ids,am,attention,pos)
 with torch.no_grad():reference=wrapped(*args).numpy()
 for name,value in zip(['input_ids','audio_mask','attention_mask','position_ids'],args):np.save(a.out/(name+'.npy'),value.numpy())
 np.save(a.out/'reference_logits.npy',reference)
 torch.onnx.export(wrapped,args,str(a.out/'step.graph.onnx'),input_names=['input_ids','audio_mask','attention_mask','position_ids'],output_names=['logits'],export_params=False,do_constant_folding=False,opset_version=17,dynamo=False,dynamic_axes={'input_ids':{2:'sequence'},'audio_mask':{1:'sequence'},'attention_mask':{2:'sequence',3:'sequence'},'position_ids':{1:'sequence'},'logits':{2:'sequence'}})
 graph=onnx.load(a.out/'step.graph.onnx');state=wrapped.state_dict();weights=[];offset=0
 with (a.out/'step.weights').open('wb') as f:
  for inp in list(graph.graph.input)[4:]:
   if inp.name not in state:raise KeyError(f'Unmapped graph weight: {inp.name}')
   arr=state[inp.name].detach().cpu().contiguous().numpy();raw=arr.tobytes();f.write(raw)
   t=TensorProto();t.name=inp.name;t.data_type=helper.np_dtype_to_tensor_dtype(arr.dtype);t.dims.extend(arr.shape);t.data_location=TensorProto.EXTERNAL
   for key,value in [('location','step.weights'),('offset',str(offset)),('length',str(len(raw)))]:e=t.external_data.add();e.key=key;e.value=value
   graph.graph.initializer.append(t);graph.graph.input.remove(inp);weights.append({'name':inp.name,'shape':list(arr.shape),'bytes':len(raw)});offset+=len(raw)
 onnx.save(graph,a.out/'step.onnx');onnx.checker.check_model(str(a.out/'step.onnx'))
 (a.out/'export.json').write_text(json.dumps({'weights_bytes':offset,'weights':weights,'inputs':[x.name for x in graph.graph.input],'ops':sorted({n.op_type for n in graph.graph.node}),'baseline':'merged BF16 weights loaded as FP32; no weight quantization'},indent=2));print('Export complete',offset,flush=True)
if __name__=='__main__':main()
