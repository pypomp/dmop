"""Measure particle likelihood noise at saved sampling states.

Independent filter replicates are evaluated in batches. No parameter proposals
or posterior draws are generated. The retained likelihood is compared with
fresh estimates only at its original particle count.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from model import setup
import jax
import jax.numpy as jnp
import numpy as np
import pypomp


def summarize(values, cached=None):
    values = np.asarray(values)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('Invalid likelihood estimate')
    finite = np.isfinite(values)
    peak = values.max()
    weights = np.exp(values-peak) if finite.any() else np.zeros_like(values)
    logmean = float(peak+np.log(weights.mean())) if finite.any() else None
    row = dict(replicates=len(values), zero_likelihood_estimates=int((~finite).sum()),
               loglik_variance=float(values.var(ddof=1)) if finite.all() else None,
               log_mean_likelihood=logmean,
               likelihood_relative_variance=float(weights.var(ddof=1)/weights.mean()**2) if finite.any() else None)
    if cached is not None:
        row.update(recorded_loglik=float(cached),
                   recorded_minus_log_mean=float(cached-logmean) if logmean is not None else None,
                   estimated_fixed_parameter_refresh_acceptance=float(np.exp(np.minimum(0., values-cached)).mean()))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--particles', type=int, nargs='+', default=[5000, 10000])
    p.add_argument('--reps', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    if min(a.particles) < 1 or a.reps < 2 or a.batch_size < 1 or a.reps % a.batch_size:
        p.error('Positive counts required; replicates must be a multiple of batch size')
    if not jax.config.jax_enable_x64:
        p.error('Use float64 for this target check')
    with np.load(a.inputs) as data:
        zs = data['z'].copy(); cached = data['cached_loglik'].copy()
        recorded_particles = int(data['recorded_particles'])
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.inputs, a.output/'inputs.npz')
    (a.output/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f, a.output/'source'/f.name)
    repo = Path(pypomp.__file__).resolve().parents[1]
    result = dict(utc=datetime.now(timezone.utc).isoformat(), devices=[str(d) for d in jax.devices()],
                  xla_flags=os.environ.get('XLA_FLAGS', ''), jax=jax.__version__, x64=True,
                  pypomp_revision=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  inputs_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
                  source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.output/'source').glob('*.py')},
                  points=[], complete=False)
    def save():
        tmp=a.output/'result.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(a.output/'result.json')
    save()
    for particles in a.particles:
        m = setup(particles)
        evaluate = jax.jit(jax.vmap(m['loglik'], in_axes=(None, 0)))
        for point, z in enumerate(zs):
            seeds = np.arange(a.reps, dtype=np.uint32) + 2026110400 + 1000*point
            keys = jax.vmap(jax.random.key)(jnp.asarray(seeds))
            chunks=[]; seconds=[]
            for start in range(0, a.reps, a.batch_size):
                t=time.monotonic(); values=evaluate(jnp.asarray(z), keys[start:start+a.batch_size])
                chunks.append(np.asarray(values)); seconds.append(time.monotonic()-t)
            values=np.concatenate(chunks)
            name=f'j{particles}-point{point}.npz'
            np.savez_compressed(a.output/name, loglik=values, seeds=seeds, batch_seconds=seconds)
            row=dict(point=point, particles=particles, evaluations=name,
                     first_batch_includes_compile=point==0,
                     **summarize(values, cached[point] if particles==recorded_particles else None))
            result['points'].append(row);save();print(json.dumps(row),flush=True)
    result['complete']=True;save()


if __name__ == '__main__':
    main()
