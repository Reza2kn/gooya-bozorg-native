"""Measure a real Core ML package and preserve planned device placement."""
import argparse,json,time,hashlib,collections
from pathlib import Path
import numpy as np,coremltools as ct
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--units',choices=['ALL','CPU_AND_GPU','CPU_AND_NE','CPU_ONLY'],default='ALL');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
units=getattr(ct.ComputeUnit,a.units);start=time.monotonic();m=ct.models.MLModel(str(a.package),compute_units=units);load=time.monotonic()-start
print('Loaded Core ML',load,flush=True)
rows=json.loads(a.inputs.read_text());inputs={n:np.array(r['data'],dtype=np.float32 if r['dtype']=='float32' else np.int32).reshape(r['shape'])for n,r in zip(['ids','mask','attention','pos'],rows)}
times=[];hashes=[]
for i in range(4):
 start=time.monotonic();out=m.predict(inputs);times.append(time.monotonic()-start);v=next(iter(out.values()));hashes.append(hashlib.sha256(v.tobytes()).hexdigest());print('run',i,times[-1],flush=True)
np.save(a.out/'logits.npy',v)
r={'runtime':'Core ML','coremltools':ct.__version__,'requested_compute_units':a.units,'load_seconds':load,'run_seconds':times,'output_sha256':hashes,'finite':bool(np.isfinite(v).all()),'package_bytes':sum(x.stat().st_size for x in a.package.rglob('*')if x.is_file()),'device_plan_note':'Anticipated placement reported by Core ML, not hardware trace.'}
(a.out/'benchmark.json').write_text(json.dumps(r,indent=2))
try:
 from coremltools.models.compute_plan import MLComputePlan
 plan=MLComputePlan.load_from_path(m.get_compiled_model_path(),compute_units=units);program=plan.model_structure.program;counts=collections.Counter()
 for fn in program.functions.values():
  for op in fn.block.operations:
   usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
   counts[type(usage.preferred_compute_device).__name__ if usage else 'unknown']+=1
 r['planned_operation_devices']=dict(counts)
except Exception as e:r['device_plan_error']=str(e)
(a.out/'benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
