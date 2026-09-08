"""Rebuild a FP32 Core ML canary with an explicit selective precision policy."""
import argparse,json,collections
from pathlib import Path
import coremltools as ct
from coremltools.converters.mil.frontend.milproto.load import load
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--policy',choices=['norms','stable','linear','mlp'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
m=ct.models.MLModel(str(a.source),skip_model_load=True);spec=m.get_spec();prog=load(spec,spec.specificationVersion,file_weights_dir=m.weights_dir);counts=collections.Counter();selected=[]
def selector(op):
 heads=any(any(d in (1025,8200)for d in getattr(v,'shape',()))for v in op.outputs)
 keep={'pow','reduce_mean','rsqrt','softmax','cos','sin','reduce_sum'}
 if a.policy=='stable':keep|={'add','mul','gather'}
 yes=not heads and op.op_type not in keep
 if a.policy in ['linear','mlp']:
  yes=op.op_type=='linear' and not heads
  if a.policy=='mlp':yes=yes and any(part in op.weight.name for part in ['gate_proj','up_proj','down_proj'])
 counts[('fp16:' if yes else 'fp32:')+op.op_type]+=1
 if yes:selected.append(op.name)
 return yes
converted=ct.convert(prog,source='milinternal',convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,compute_precision=ct.transform.FP16ComputePrecision(op_selector=selector),skip_model_load=True)
converted.save(str(a.out/'step.mlpackage'));(a.out/'precision-policy.json').write_text(json.dumps({'policy':a.policy,'operation_counts':dict(counts),'selected_ops':selected},indent=2));print(dict(counts),flush=True)
