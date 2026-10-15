"""Pilot proposal covariance from the regularized MOP Hessian.

This matrix only preconditions proposals. It never changes the prior or
acceptance target, and is not reported as a posterior covariance estimate.
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
p.add_argument('--reps',type=int,default=4)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
m=setup(a.particles)
z=np.load(a.initial)
hessian=jax.jit(jax.jacfwd(lambda z,k:m['value_force'](z,k)[1],argnums=0))
Hs=[]
for i in range(a.reps):
 t=time.monotonic();H=np.asarray(hessian(z,jax.random.key(2026102100+i)))
 if not np.isfinite(H).all():
  raise RuntimeError('Nonfinite pilot Hessian')
 Hs.append(H)
 print(dict(rep=i,seconds=time.monotonic()-t,eigenvalues=np.linalg.eigvalsh(-(H+H.T)/2).tolist()),flush=True)
 np.save(a.output/'raw_hessians.npy',np.array(Hs))
H=np.mean(Hs,axis=0)
eig,vec=np.linalg.eigh(-(H+H.T)/2)
# Prior curvature on standardized coordinates is one. Preserve that floor
# where the approximate likelihood Hessian is indefinite or weak.
regularized=np.maximum(eig,1.)
cov=(vec/regularized)@vec.T
chol=np.linalg.cholesky(cov)
np.save(a.output/'chol.npy',chol)
(a.output/'summary.json').write_text(json.dumps(dict(particles=a.particles,reps=a.reps,
 initial=str(a.initial),eigenvalues=eig.tolist(),regularized=regularized.tolist(),
 proposal_marginal_sd=np.sqrt(np.diag(cov)).tolist()),indent=2)+'\n')
