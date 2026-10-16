"""Compare MOP forces with local fixed-seed PF finite differences.

Resampling makes this target discontinuous. Agreement at shrinking steps is
only a local check; it neither proves global smoothness nor sampler mixing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from model import setup
import jax
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--particles',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026102900)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=a.output/'source';source.mkdir()
    for name in ['model.py','conditional_force_check.py']:
        shutil.copy2(Path(__file__).parent/name,source/name)
    z=np.load(a.initial);np.save(a.output/'initial.npy',z)
    key=jax.random.key(a.seed)
    rng=np.random.default_rng(a.seed)
    directions=rng.normal(size=(2,len(z)))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    np.save(a.output/'directions.npy',directions)
    m=setup(a.particles,0.)
    target=lambda z:float(m['loglik'](z,key)+m['logprior'](z))
    forces=[]
    for alpha in [0.,.95]:
        other=setup(a.particles,alpha)
        value,force=other['value_force'](z,key)
        force=np.asarray(force)
        forces.append(dict(alpha=alpha,value=float(value),finite=bool(np.isfinite(force).all()),
                           directional_derivatives=(directions@force).tolist()))
    rows=[]
    for i,direction in enumerate(directions):
        for h in [1e-5,1e-6,1e-7,1e-8,1e-9]:
            delta=(target(z+h*direction)-target(z-h*direction))/(2*h)
            row=dict(direction=i,step=h,finite_difference=delta)
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),particles=a.particles,
        seed=a.seed,affinity=sorted(os.sched_getaffinity(0)),x64=jax.config.jax_enable_x64,
        target_value=target(z),forces=forces,finite_differences=rows,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},
        interpretation='Local fixed-seed check, not a claim of global differentiability or posterior convergence')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
