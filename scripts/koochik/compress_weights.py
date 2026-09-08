"""Portable per-row INT8 weight storage; float arithmetic after dequantization.
This reduces artifact size, not a claim of int8 execution or reduced runtime RAM.
"""
import argparse,json
from pathlib import Path
import numpy as np,onnx
from onnx import helper,TensorProto

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preserve-io',action='store_true');p.add_argument('--bits',type=int,choices=[6,7,8],default=8);p.add_argument('--group-size',type=int,default=0);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 m=onnx.load(a.source,load_external_data=False);initializers=[];nodes=[];receipts=[];offset=0
 def emit(f,name,arr):
  nonlocal offset
  raw=arr.tobytes();f.write(raw);t=TensorProto();t.name=name;t.data_type=helper.np_dtype_to_tensor_dtype(arr.dtype);t.dims.extend(arr.shape);t.data_location=TensorProto.EXTERNAL
  for k,v in [('location','step.weights'),('offset',str(offset)),('length',str(len(raw)))]:e=t.external_data.add();e.key=k;e.value=v
  initializers.append(t);offset+=len(raw)
 with (a.out/'step.weights').open('wb') as dest:
  for t in m.graph.initializer:
   ext={e.key:e.value for e in t.external_data};dtype=helper.tensor_dtype_to_np_dtype(t.data_type);arr=np.memmap(a.source.parent/ext['location'],mode='r',offset=int(ext['offset']),shape=tuple(t.dims),dtype=dtype)
   if t.data_type==TensorProto.FLOAT and len(t.dims)==2 and arr.size>=4096 and not (a.preserve_io and any(x in t.name for x in ['embed_tokens','audio_embeddings','audio_heads'])):
    limit=(1 << (a.bits-1))-1
    view=arr.reshape(arr.shape[0],-1,a.group_size) if a.group_size else arr
    scale=np.maximum(np.max(np.abs(view),axis=-1,keepdims=True)/limit,np.finfo(np.float32).tiny).astype(np.float32);q=np.clip(np.rint(view/scale),-limit,limit).astype(np.int8)
    emit(dest,t.name+'.int8',q);emit(dest,t.name+'.scale',scale)
    nodes.extend([helper.make_node('Cast',[t.name+'.int8'],[t.name+'.float'],to=TensorProto.FLOAT,name=t.name+'.dequant_cast'),helper.make_node('Mul',[t.name+'.float',t.name+'.scale'],[t.name+'.grouped' if a.group_size else t.name],name=t.name+'.dequant_scale')])
    if a.group_size:
     emit(dest,t.name+'.shape',np.asarray(arr.shape,dtype=np.int64));nodes.append(helper.make_node('Reshape',[t.name+'.grouped',t.name+'.shape'],[t.name]))
    receipts.append({'name':t.name,'source_bytes':arr.nbytes,'stored_bytes':q.nbytes+scale.nbytes,'bits':a.bits})
   else:emit(dest,t.name,np.asarray(arr))
 del m.graph.initializer[:];m.graph.initializer.extend(initializers);old=list(m.graph.node);del m.graph.node[:];m.graph.node.extend(nodes+old);onnx.save(m,a.out/'step.onnx');onnx.checker.check_model(str(a.out/'step.onnx'))
 (a.out/'compression.json').write_text(json.dumps({'method':f'symmetric grouped W{a.bits}A32, int8 container with zstd, portable dequantization','preserve_io':a.preserve_io,'group_size':a.group_size,'runtime_memory_reduction_claimed':False,'stored_weights_bytes':offset,'quantized_tensors':receipts},indent=2));print('Stored weights bytes',offset)
if __name__=='__main__':main()
