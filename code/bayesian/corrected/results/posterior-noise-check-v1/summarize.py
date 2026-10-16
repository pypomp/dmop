"""Summarize the expanded plateau diagnostic; never pool its initial32 twice."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parent
result=json.loads((root/'expanded/result.json').read_text())
if not result['complete']:raise ValueError('Expanded check is still running')
with np.load(root/'expanded/j5000-point0.npz') as a,np.load(root/'expanded/j10000-point0.npz') as b:
    x,y=a['loglik'].copy(),b['loglik'].copy()
    np.testing.assert_array_equal(a['seeds'],b['seeds'])
    seconds=[float(np.median(a['batch_seconds'][1:])),float(np.median(b['batch_seconds'][1:]))]
assert len(x)==len(y)==128 and np.isfinite(x).all() and np.isfinite(y).all()
with np.load(root/'plateau-input.npz') as d:cached=float(d['cached_loglik'][0])
def logmean(v):
    peak=v.max(axis=-1)
    return peak+np.log(np.exp(v-peak[...,None]).mean(axis=-1))
rng=np.random.default_rng(2026110500)
index=rng.integers(0,len(x),size=(4000,len(x)))
bx,by=x[index],y[index]
vx,vy=bx.var(axis=1,ddof=1),by.var(axis=1,ddof=1)
def item(estimate,bootstrap):
    return dict(estimate=float(estimate),bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist())
summary=dict(replicates=128,initial32_included_not_pooled=True,
             interpretation='Local diagnostic with percentile bootstrap intervals; does not certify global particle adequacy or chain convergence',
             loglik_variance_5000=item(x.var(ddof=1),vx),
             loglik_variance_10000=item(y.var(ddof=1),vy),
             variance_ratio_5000_over_10000=item(x.var(ddof=1)/y.var(ddof=1),vx/vy),
             log_mean_likelihood_difference_5000_minus_10000=item(logmean(x)-logmean(y),logmean(bx)-logmean(by)),
             recorded_5000_minus_log_mean=item(cached-logmean(x),cached-logmean(bx)),
             fixed_parameter_5000_refresh_acceptance=item(np.exp(np.minimum(0.,x-cached)).mean(),
                                                        np.exp(np.minimum(0.,bx-cached)).mean(axis=1)),
             observed_seconds_per_16_replicates=dict(j5000=seconds[0],j10000=seconds[1]),
             timing_limit='Shared GPU load; timings exclude compilation but are not a controlled method comparison')
(root/'expanded-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
