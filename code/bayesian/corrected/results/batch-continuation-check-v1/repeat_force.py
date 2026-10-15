"""Measure repeatability of the same frozen GPU force at identical inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
cfg=json.loads((a.parent/'chain0/configuration.json').read_text())
sys.path.insert(0,str((a.parent/'chain0/source').resolve()))
from model import setup
import jax
import jax.numpy as jnp

arrays=[np.load(a.parent/f'chain{c}/checkpoint.npz') for c in range(cfg['batch_size'])]
z=jnp.asarray(np.stack([cp['z'] for cp in arrays]))
u=jax.random.wrap_key_data(jnp.asarray(np.stack([cp['u'] for cp in arrays])))
m=setup(cfg['particles'],cfg['alpha'],cfg['prior_scale'],
        force_particles=cfg['force_particles'],force_precision=cfg['force_precision'])
force=jax.jit(jax.vmap(m['value_force']))
ll=jax.jit(jax.vmap(m['loglik']))
forces=[]; likelihoods=[]; timings=[]
for i in range(5):
    start=time.monotonic();_,g=force(z,u);forces.append(np.asarray(g));timings.append(time.monotonic()-start)
    likelihoods.append(np.asarray(ll(z,u)))
    print(json.dumps({'evaluation':i+1,'force_seconds':timings[-1]}),flush=True)
forces=np.stack(forces);likelihoods=np.stack(likelihoods)
result=dict(devices=[str(d) for d in jax.devices()],repetitions=5,
            xla_flags=os.environ.get('XLA_FLAGS',''),force_seconds=timings,
            maximum_force_range_by_chain=np.ptp(forces,axis=0).max(axis=1).tolist(),
            maximum_likelihood_range_by_chain=np.ptp(likelihoods,axis=0).tolist(),
            force_bitwise_repeatable=bool(np.all(forces==forces[0])),
            likelihood_bitwise_repeatable=bool(np.all(likelihoods==likelihoods[0])))
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
