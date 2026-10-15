"""Measure GPU precision and NUTS cost at saved points; never sample a posterior.

Run separate processes for float32 and float64 so model data are imported at
the requested precision. Equal keys across precisions need not give equal
random numbers or particle genealogies. Compare replicated likelihoods, not
individual paired values. Kernel timings exclude the first compilation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--precision',required=True,choices=['float32','float64'])
    p.add_argument('--particles',type=int,default=5000)
    p.add_argument('--force-particles',type=int,default=1000)
    p.add_argument('--force-precision',choices=['float32'],default=None,
                   help='Optional reduced precision for the proposal force only')
    p.add_argument('--reps',type=int,default=16)
    p.add_argument('--kernel-reps',type=int,default=3,
                   help='Postcompile updates at a fixed state; zero skips NUTS timing')
    p.add_argument('--batch-size',type=int,default=1)
    a=p.parse_args()
    os.environ['JAX_ENABLE_X64']=str(a.precision=='float64').lower()
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    from model import setup
    from kernels import make_kernel
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pypomp
    assert jax.config.jax_enable_x64==(a.precision=='float64')
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(a.inputs,a.output/'inputs.npz')
    (a.output/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f,a.output/'source'/f.name)
    data=np.load(a.inputs);dtype=jnp.float64 if a.precision=='float64' else jnp.float32
    zs=jnp.asarray(data['z'],dtype=dtype);chols=jnp.asarray(data['chol'],dtype=dtype)
    eps=jnp.asarray(data['step'],dtype=dtype)
    assert zs.ndim==2 and len(zs)>=a.batch_size
    m=setup(a.particles,.95,1.,force_particles=a.force_particles,
            force_precision=a.force_precision)
    repo=Path(pypomp.__file__).resolve().parents[1]
    result={'utc':datetime.now(timezone.utc).isoformat(),'precision':a.precision,
            'devices':[str(x) for x in jax.devices()],'jax':jax.__version__,
            'pypomp_revision':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'inputs_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
            'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.output/'source').glob('*.py')},
            'likelihoods':[],'kernel_timings':[],'complete':False}
    def save():
        temp=a.output/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.output/'result.json')
    save();print(json.dumps({'phase':'loaded','precision':a.precision,'devices':result['devices']}),flush=True)
    for i,z in enumerate(zs):
        rows=[]
        for rep in range(a.reps):
            key=jax.random.key(2026103100+100*i+rep);start=time.monotonic()
            value=float(m['loglik'](z,key));seconds=time.monotonic()-start
            if not np.isfinite(value):raise ValueError('Nonfinite likelihood')
            rows.append({'seed':2026103100+100*i+rep,'loglik':value,'seconds':seconds,
                         'compile_included':i==0 and rep==0})
        result['likelihoods'].append(rows);save()
        print(json.dumps({'phase':'likelihoods','point':i,'mean':float(np.mean([r['loglik'] for r in rows]))}),flush=True)
    if a.kernel_reps:
        init,kernel=make_kernel(m['loglik'],m['logprior'],lambda z,u:m['value_force'](z,u)[1],'nuts',4,4)
        if a.batch_size==1:
            state=init(zs[0],jax.random.key(2026103200));chol=chols[0];step=eps[0]
            get_key=lambda i:jax.random.key(2026103300+i)
        else:
            keys=jax.random.split(jax.random.key(2026103200),a.batch_size)
            state=jax.jit(jax.vmap(init))(zs[:a.batch_size],keys)
            kernel=jax.jit(jax.vmap(kernel));chol=chols[:a.batch_size];step=eps[:a.batch_size]
            get_key=lambda i:jax.random.split(jax.random.key(2026103300+i),a.batch_size)
        jax.block_until_ready(state)
        for i in range(a.kernel_reps+1):
            start=time.monotonic();new,info=kernel(state,get_key(i),step,chol)
            jax.block_until_ready((new,info));seconds=time.monotonic()-start
            info=np.asarray(info);info=info[None] if info.ndim==1 else info
            row={'seconds':seconds,'compile_included':i==0,'batch_size':a.batch_size,
                 'integration_steps':info[:,7].tolist(),'divergences':info[:,6].tolist(),
                 'acceptance':info[:,2].tolist(),'finite':bool(np.isfinite(np.asarray(new.z)).all() and np.isfinite(np.asarray(new.loglik)).all())}
            result['kernel_timings'].append(row);save();print(json.dumps(row),flush=True)
    result['complete']=True;save()


if __name__=='__main__':
    main()
