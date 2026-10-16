"""Continue a completed chain exactly, keeping its frozen tuning and RNG state.

Write a new directory containing the complete old trace plus new draws. The
parent remains immutable. No optimization, reinitialization or adaptation.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--draws',required=True,type=int)
    p.add_argument('--checkpoint',type=int,default=25)
    a=p.parse_args()
    if a.draws<1 or a.checkpoint<1:
        p.error('Positive counts required')
    if not (a.parent/'completion.json').exists():
        p.error('Parent must be completed, with adaptation frozen')
    cfg=json.loads((a.parent/'configuration.json').read_text())
    source=a.parent/'source'
    for name,digest in cfg['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Changed parent source: {name}')
    # All model/kernel imports come from the parent source snapshot.
    sys.path.insert(0,str(source.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pypomp
    from model import setup
    from kernels import State,make_kernel
    from run import write_json
    repo=Path(pypomp.__file__).resolve().parents[1]
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if revision!=cfg['pypomp_revision'] or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True):
        raise ValueError('Pypomp source differs from parent')
    if [str(x) for x in jax.devices()]!=cfg['devices']:
        raise ValueError('Continue on the same backend as the parent')
    cp=np.load(a.parent/'checkpoint.npz')
    old=len(cp['trace']);total=old+a.draws
    if old<=cfg['warmup']:
        raise ValueError('No retained sampling in parent')
    state=State(jnp.asarray(cp['z']),jax.random.wrap_key_data(jnp.asarray(cp['u'])),
                jnp.asarray(cp['loglik']),jnp.asarray(cp['force']))
    key=jax.random.wrap_key_data(jnp.asarray(cp['key']))
    chol=jnp.asarray(cp['chol']);eps=float(cp['next_step_size'])
    traces=list(cp['trace']);infos=list(cp['info']);eps_trace=list(cp['step_size'])
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source,a.output/'source')
    cfg.update(output=str(a.output),draws=total-cfg['warmup'],parent=str(a.parent.resolve()),
               parent_checkpoint_sha256=hashlib.sha256((a.parent/'checkpoint.npz').read_bytes()).hexdigest(),
               continuation_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
               continuation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    shutil.copy2(Path(__file__),a.output/'continuation_driver.py')
    write_json(a.output/'configuration.json',cfg)
    m=setup(cfg['particles'],cfg['alpha'],cfg['prior_scale'],sort_particles=cfg.get('sort_particles',False))
    _,kernel=make_kernel(m['loglik'],m['logprior'],lambda z,u:m['value_force'](z,u)[1],cfg['method'],cfg['steps'])
    start=time.monotonic()
    parent_seconds=json.loads((a.parent/'status.json').read_text())['seconds']
    for i in range(old,total):
        key,k=jax.random.split(key)
        state,info=kernel(state,k,eps,chol)
        infos.append(np.asarray(info))
        traces.append(np.r_[float(state.loglik),float(m['logprior'](state.z)),np.asarray(state.z)])
        eps_trace.append(eps)
        if (i+1)%a.checkpoint==0 or i+1==total:
            tmp=a.output/'checkpoint.tmp.npz'
            np.savez_compressed(tmp,trace=np.array(traces),info=np.array(infos),step_size=np.array(eps_trace),
                z=np.asarray(state.z),u=np.asarray(jax.random.key_data(state.u)),force=np.asarray(state.force),
                loglik=np.asarray(state.loglik),key=np.asarray(jax.random.key_data(key)),
                chol=np.asarray(chol),next_step_size=eps)
            tmp.replace(a.output/'checkpoint.npz')
            recent=np.asarray(infos)[-a.checkpoint:]
            status=dict(iteration=i+1,warmup=cfg['warmup'],total=total,complete=i+1==total,
                        phase='sampling',seconds=parent_seconds+time.monotonic()-start,
                        loglik=float(state.loglik),step_size=eps,
                        parameter_acceptance_recent=float(recent[:,0].mean()),
                        randomness_acceptance_recent=float(recent[:,1].mean()))
            write_json(a.output/'status.json',status)
            print(json.dumps(status),flush=True)
    write_json(a.output/'completion.json',status)

if __name__=='__main__':
    main()
