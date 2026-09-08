"""Restrict text embeddings to the native spaced-ASCII-phone input contract.
Retained rows are bit-identical. Unsupported text token IDs must fail in Rust.
"""
import argparse,json
from pathlib import Path
import numpy as np,onnx
from onnx import TensorProto,helper
from tokenizers import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
tok=Tokenizer.from_file(str(a.tokenizer));keep={i for i in range(tok.get_vocab_size()) if (s:=tok.decode([i],skip_special_tokens=False)).isascii() and len(s)<=2};keep.update(tok.encode('<|denoise|><|lang_start|>fa<|lang_end|><|instruct_start|>None<|instruct_end|><|text_start|><|text_end|>').ids);keep.update([0,1024]);keep=sorted(keep)
m=onnx.load(a.source,load_external_data=False);new=[];offset=0;embedding='model.llm.embed_tokens.weight';lookup=np.zeros(tok.get_vocab_size(),dtype=np.int64);lookup[keep]=np.arange(len(keep))
with (a.out/'step.weights').open('wb') as f:
 def emit(name,arr):
  global offset
  raw=arr.tobytes();f.write(raw);t=TensorProto(name=name,data_type=helper.np_dtype_to_tensor_dtype(arr.dtype),dims=arr.shape,data_location=TensorProto.EXTERNAL)
  for k,v in [('location','step.weights'),('offset',str(offset)),('length',str(len(raw)))]:e=t.external_data.add();e.key=k;e.value=v
  new.append(t);offset+=len(raw)
 for t in m.graph.initializer:
  ext={e.key:e.value for e in t.external_data};arr=np.memmap(a.source.parent/ext['location'],mode='r',offset=int(ext['offset']),shape=tuple(t.dims),dtype=helper.tensor_dtype_to_np_dtype(t.data_type));emit(t.name,np.asarray(arr[keep] if t.name==embedding else arr))
 emit('native_text_id_map',lookup)
nodes=[]
for n in m.graph.node:
 if embedding in n.input:
  original=n.input[1];n.input[1]=original+'.compact';nodes.append(helper.make_node('Gather',['native_text_id_map',original],[n.input[1]],name='NativeTextVocabularyMap',axis=0))
 nodes.append(n)
del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.initializer[:];m.graph.initializer.extend(new);onnx.save(m,a.out/'step.onnx');onnx.checker.check_model(str(a.out/'step.onnx'))
(a.out/'allowed_text_ids.json').write_text(json.dumps(keep));(a.out/'pruning.json').write_text(json.dumps(dict(original_rows=tok.get_vocab_size(),retained_rows=len(keep),stored_bytes=offset,retained_values='bit-identical',input_contract='spaced ASCII phonemes plus fixed fa/default style; reject unsupported text IDs'),indent=2));print('Rows',len(keep),'stored bytes',offset)
