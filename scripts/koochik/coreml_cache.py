"""Local research cache for compiled Core ML packages; invalidate changed inputs."""
import hashlib,json,shutil,platform,ctypes,os
from pathlib import Path
import coremltools as ct

def clone_copy(source,dest):
 lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
 lib.clonefile.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];lib.clonefile.restype=ctypes.c_int
 if lib.clonefile(os.fsencode(source),os.fsencode(dest),0)!=0:shutil.copy2(source,dest)
 return dest

def load_model(package,units):
 package=Path(package).resolve()
 stamps=[]
 for p in sorted(package.rglob('*')):
  if p.is_file():
   digest=hashlib.sha256()
   with p.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
   stamps.append((str(p.relative_to(package)),p.stat().st_size,digest.hexdigest()))
 key=hashlib.sha256(json.dumps([stamps,ct.__version__,platform.mac_ver(),units.name]).encode()).hexdigest()[:20]
 cache=package.parent/('compiled-'+key+'.mlmodelc')
 if cache.exists():return ct.models.CompiledMLModel(str(cache),compute_units=units),True
 model=ct.models.MLModel(str(package),compute_units=units)
 partial=cache.with_name(cache.name+'.partial')
 if partial.exists():raise RuntimeError('Incomplete compilation cache exists: '+str(partial))
 shutil.copytree(model.get_compiled_model_path(),partial,copy_function=clone_copy);partial.rename(cache)
 return model,False
