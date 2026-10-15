"""Compare MOP force noise and cost at one fixed point on the same CPU core.

This is a proposal-tuning diagnostic, not a posterior or mixing assessment.
All settings retain the same acceptance likelihood and prior.
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
import numpy as np
import pypomp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--metric', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--counts', type=int, nargs='+', default=[1000, 2500, 5000])
    p.add_argument('--reps', type=int, default=16)
    p.add_argument('--alpha', type=float, default=.95)
    p.add_argument('--seed', type=int, default=2026102500)
    a = p.parse_args()
    if a.reps < 2 or min(a.counts) < 1:
        p.error('At least two replicates and positive particle counts required')
    a.output.mkdir(parents=True, exist_ok=False)
    z = np.load(a.initial)
    chol = np.load(a.metric)
    np.save(a.output/'initial.npy', z)
    np.save(a.output/'chol.npy', chol)
    source = a.output/'source'
    source.mkdir()
    for name in ['model.py', 'force_check.py']:
        shutil.copy2(Path(__file__).parent/name, source/name)
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(),
        counts=a.counts, reps=a.reps, alpha=a.alpha, seed=a.seed,
        affinity=sorted(os.sched_getaffinity(0)), devices=[str(d) for d in jax.devices()],
        jax=jax.__version__, x64=jax.config.jax_enable_x64,
        pypomp_revision=subprocess.check_output(['git','-C',str(Path(pypomp.__file__).resolve().parents[1]),
                                                  'rev-parse','HEAD'],text=True).strip(),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},
        interpretation='Single-point force variability and serial timing; not evidence of posterior convergence')
    (a.output/'configuration.json').write_text(json.dumps(manifest, indent=2)+'\n')
    summary = []
    for count in a.counts:
        m = setup(5000, a.alpha, force_particles=count)
        start = time.monotonic()
        np.asarray(m['value_force'](z, jax.random.key(a.seed-1))[1])
        compile_seconds = time.monotonic()-start
        forces, seconds = [], []
        for i in range(a.reps):
            start = time.monotonic()
            force = np.asarray(m['value_force'](z, jax.random.key(a.seed+i))[1])
            seconds.append(time.monotonic()-start)
            forces.append(force)
        forces = np.asarray(forces)
        if not np.isfinite(forces).all():
            raise ValueError(f'Nonfinite raw force for {count} particles')
        # B^T G is the force in the sampler's whitened momentum coordinates.
        projected = forces @ chol
        row = dict(force_particles=count, first_call_seconds=compile_seconds,
            median_seconds=float(np.median(seconds)),
            projected_mean_squared_norm=float(np.sum(projected.mean(axis=0)**2)),
            projected_noise_trace=float(np.var(projected, axis=0, ddof=1).sum()))
        np.savez_compressed(a.output/f'force-{count}.npz', forces=forces, seconds=seconds)
        summary.append(row)
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    (a.output/'completion.json').write_text(json.dumps({'complete':True})+'\n')


if __name__ == '__main__':
    main()
