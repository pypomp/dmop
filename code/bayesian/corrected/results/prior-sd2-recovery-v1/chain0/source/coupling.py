"""Measure PF log-ratio variability under paired nearby parameters."""
import argparse
import json
from pathlib import Path
import time
import jax
import numpy as np
from model import setup

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--particles',type=int,default=1000)
p.add_argument('--reps',type=int,default=24)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for sorted_ in [False,True]:
 m=setup(a.particles,sort_particles=sorted_)
 z=m['initial']; rng=np.random.default_rng(83156)
 dz=rng.normal(size=len(z))*.005
 for i in range(a.reps):
  k=jax.random.key(2026101900+i);t=time.monotonic()
  ll0=float(m['loglik'](z,k));ll1=float(m['loglik'](z+dz,k))
  rows.append(dict(sorted=sorted_,rep=i,ll0=ll0,ll1=ll1,ratio=ll1-ll0,seconds=time.monotonic()-t))
  print(rows[-1],flush=True)
  (a.output/'pairs.json').write_text(json.dumps(rows,indent=2)+'\n')
 summary={str(s):dict(loglik_variance=float(np.var([r['ll0'] for r in rows if r['sorted']==s],ddof=1)),
                     logratio_variance=float(np.var([r['ratio'] for r in rows if r['sorted']==s],ddof=1)))
          for s in [False,True] if any(r['sorted']==s for r in rows)}
 (a.output/'summary.json').write_text(json.dumps(dict(particles=a.particles,reps=a.reps,
          displacement=dz.tolist(),summary=summary),indent=2)+'\n')
print(summary,flush=True)
