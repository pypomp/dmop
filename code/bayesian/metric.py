"""Regularized proposal preconditioning from replicated MOP scores.

The score second moment is a tuning heuristic, not a posterior covariance or
an asserted Fisher information estimate. It never enters the target density.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
import argparse
import json
from pathlib import Path
import time
from model import setup
import jax
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--initial',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--particles',type=int,default=1000)
p.add_argument('--reps',type=int,default=64)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
m=setup(a.particles)
z=np.load(a.initial)
forces=[]
for i in range(a.reps):
 t=time.monotonic();value,g=m['value_force'](z,jax.random.key(2026102100+i))
 g=np.asarray(g)
 if not np.isfinite(g).all():
  raise RuntimeError('Nonfinite pilot force')
 forces.append(g)
 print(dict(rep=i,seconds=time.monotonic()-t,objective=float(value),force_norm=float(np.linalg.norm(g))),flush=True)
 np.save(a.output/'forces.npy',np.array(forces))
G=np.array(forces)
second=G.T@G/len(G)
# Shrink the second moment slightly toward its diagonal and add a ridge.
precision=.9*second+.1*np.diag(np.diag(second))+np.eye(len(z))
eig,vec=np.linalg.eigh(precision)
cov=(vec/eig)@vec.T
chol=np.linalg.cholesky(cov)
np.save(a.output/'chol.npy',chol)
(a.output/'summary.json').write_text(json.dumps(dict(particles=a.particles,reps=a.reps,
 initial=str(a.initial),eigenvalues=eig.tolist(),
 proposal_marginal_sd=np.sqrt(np.diag(cov)).tolist(),
 interpretation='proposal preconditioning heuristic, not posterior covariance'),indent=2)+'\n')
