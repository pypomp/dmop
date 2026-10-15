"""Reproduce the local float32/float64 likelihood comparison."""
from pathlib import Path
import hashlib
import json
import numpy as np

base=Path(__file__).resolve().parent
files=[base/f'{precision}-reps128/result.json' for precision in ['float32','float64']]
a,b=[json.loads(f.read_text()) for f in files]
assert a['complete'] and b['complete'] and a['inputs_sha256']==b['inputs_sha256']
rng=np.random.default_rng(2026103400)
rows=[]
for i,(aa,bb) in enumerate(zip(a['likelihoods'],b['likelihoods'])):
    assert [r['seed'] for r in aa]==[r['seed'] for r in bb]
    x=np.array([r['loglik'] for r in aa]);y=np.array([r['loglik'] for r in bb])
    assert len(x)==len(y)==128
    shift=max(x.max(),y.max());wx=np.exp(x-shift);wy=np.exp(y-shift)
    delta=float(np.log(wx.mean())-np.log(wy.mean()))
    indices=rng.integers(0,len(x),size=(4000,len(x)))
    bootstrap=np.log(wx[indices].mean(axis=1))-np.log(wy[indices].mean(axis=1))
    rows.append(dict(point=i,replicates=128,
        mean_loglik_float32=float(x.mean()),mean_loglik_float64=float(y.mean()),
        loglik_sd_float32=float(x.std(ddof=1)),loglik_sd_float64=float(y.std(ddof=1)),
        log_mean_likelihood_difference=delta,
        bootstrap_95_interval=np.quantile(bootstrap,[.025,.975]).tolist()))
result=dict(input_sha256=a['inputs_sha256'],
    result_hashes={str(f.relative_to(base)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    points=rows,
    interpretation='Float32 minus float64 log average likelihood estimates. Paired bootstrap resamples seed-index pairs; it does not presume equal random variates or genealogies across precision. Four local points and Monte Carlo intervals do not establish global posterior equivalence.')
(base/'precision-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows,indent=2))
