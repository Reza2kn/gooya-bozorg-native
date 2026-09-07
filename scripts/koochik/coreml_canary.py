"""Fixed-shape Core ML conversion canary using the accepted grouped weights.
This is a runtime experiment, not a general-input release or quality gate.
"""
import argparse,json,time,gc
from pathlib import Path
import numpy as np, torch, onnx
from onnx import numpy_helper
from omnivoice.models.omnivoice import OmniVoice
import coremltools as ct
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--precision',choices=['fp16','fp32','protected'],default='fp16');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
class MappedEmbedding(torch.nn.Module):
 def __init__(self,w,m):
  super().__init__();self.embedding=torch.nn.Embedding.from_pretrained(w);self.register_buffer('lookup',m)
 def forward(self,x):return self.embedding(self.lookup[x])
class Step(torch.nn.Module):
 def __init__(self,m):super().__init__();self.model=m
 def forward(self,ids,mask,attention,pos):
  return self.model(input_ids=ids.long(),audio_mask=mask.bool(),attention_mask=attention,position_ids=pos.long()).logits
package=a.out/'step.mlpackage'
if not package.exists():
 model=OmniVoice.from_pretrained(str(a.source),train=True,dtype=torch.float32,device_map='cpu').eval();model.llm.set_attn_implementation('eager');model.llm.config.use_cache=False
 graph=onnx.load(a.bundle/'cache/step.onnx',load_external_data=False) if (a.bundle/'cache/step.onnx').exists() else onnx.load(a.bundle/'step.onnx',load_external_data=False)
 root=a.bundle/'cache'
 arrays={}
 for t in graph.graph.initializer:
  ext={e.key:e.value for e in t.external_data}
  arrays[t.name]=np.memmap(root/ext['location'],mode='r',offset=int(ext['offset']),shape=tuple(t.dims),dtype=onnx.helper.tensor_dtype_to_np_dtype(t.data_type)) if ext else numpy_helper.to_array(t)
 wrapped=Step(model).eval()
 with torch.no_grad():
  for name,param in wrapped.named_parameters():
   if name=='model.llm.embed_tokens.weight':continue
   if name in arrays:arr=arrays[name]
   elif name+'.int8' in arrays:arr=(arrays[name+'.int8'].astype(np.float32)*arrays[name+'.scale']).reshape(param.shape)
   else:raise KeyError(name)
   param.copy_(torch.from_numpy(np.array(arr)))
  model.llm.embed_tokens=MappedEmbedding(torch.from_numpy(np.array(arrays['model.llm.embed_tokens.weight'])),torch.from_numpy(np.array(arrays['native_text_id_map'])))
 del arrays,graph;gc.collect()
 rows=json.loads(a.inputs.read_text());vals=[np.array(r['data'],dtype=np.float32 if r['dtype']=='float32' else np.int32).reshape(r['shape']) for r in rows];args=tuple(torch.from_numpy(x)for x in vals)
 with torch.no_grad():
  np.save(a.out/'torch_logits.npy',wrapped(*args).numpy());traced=torch.jit.trace(wrapped,args,check_trace=False).eval()
 del wrapped,model;gc.collect()
 print('Tracing complete; converting Core ML',flush=True)
 inputs=[ct.TensorType(name=n,shape=v.shape,dtype=v.dtype)for n,v in zip(['ids','mask','attention','pos'],vals)]
 precision=ct.precision.FLOAT32
 if a.precision=='fp16':precision=ct.precision.FLOAT16
 elif a.precision=='protected':
  def selector(op):
   # Keep codec-vocabulary output heads and their downstream logits in FP32.
   # Transformer hidden sizes are 1024/3072; output vocabulary is 1025 x 8.
   return not any(any(d in (1025,8200) for d in getattr(v,'shape',())) for v in op.outputs)
  precision=ct.transform.FP16ComputePrecision(op_selector=selector)
 converted=ct.convert(traced,inputs=inputs,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,compute_precision=precision,skip_model_load=True)
 converted.save(str(package));print('Saved package',flush=True)
 del traced,converted;gc.collect()
print('Core ML package ready',flush=True)
