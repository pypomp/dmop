"""Run independent NUTS or PMMH chains together, saving ordinary chain files.

Batching changes execution, not the per-chain kernel or adaptation rule. Supply
one initial vector, proposal factor and seed per chain. Initialization fitting
is separate and must be accounted for when reporting total computational cost.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np
import pypomp
from model import setup, NAMES, CENTER, SCALE
from kernels import make_kernel
from extend import adapt_tuning
from run import write_json


def batched_transition(kernel):
    """Advance each chain's own key exactly as the scalar runner does."""
    @jax.jit
    def advance(states, keys, eps, chols):
        split = jax.vmap(jax.random.split)(keys)
        new, info = jax.vmap(kernel)(states, split[:, 1], eps, chols)
        return new, split[:, 0], info
    return advance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initials', type=Path, nargs='+', required=True)
    p.add_argument('--metrics', type=Path, nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--method', choices=['nuts', 'pmrw'], required=True)
    p.add_argument('--particles', type=int, default=5000)
    p.add_argument('--force-particles', type=int, default=1000)
    p.add_argument('--force-precision', choices=['float32'], default=None)
    p.add_argument('--alpha', type=float, default=.95)
    p.add_argument('--prior-scale', type=float, default=1.)
    p.add_argument('--max-depth', type=int, default=4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--step-size', type=float, required=True)
    p.add_argument('--max-step-size', type=float, default=2.)
    p.add_argument('--checkpoint', type=int, default=5)
    a = p.parse_args()
    n = len(a.seeds)
    if n != len(a.initials) or n != len(a.metrics) or len(set(a.seeds)) != n:
        p.error('Supply equal numbers of initial vectors, factors and distinct seeds')
    if min(a.particles, a.force_particles, a.max_depth, a.draws,
           a.checkpoint, a.step_size, a.prior_scale) <= 0 or a.warmup < 0:
        p.error('Counts/scales must be positive; warmup may be zero')
    if not jax.config.jax_enable_x64:
        p.error('The acceptance target must use float64')
    zs = np.stack([np.load(f) for f in a.initials])
    chols = np.stack([np.load(f) for f in a.metrics])
    if zs.shape != (n, len(NAMES)) or not np.isfinite(zs).all():
        p.error('Invalid initial vectors')
    if (chols.shape != (n, len(NAMES), len(NAMES)) or not np.isfinite(chols).all()
            or np.any(np.diagonal(chols, axis1=1, axis2=2) <= 0)):
        p.error('Invalid proposal factors')
    np.testing.assert_allclose(chols, np.tril(chols), atol=1e-12)
    a.output.mkdir(parents=True, exist_ok=False)
    repo = Path(pypomp.__file__).resolve().parents[1]
    meta = dict(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                jax=jax.__version__, x64=True, devices=[str(x) for x in jax.devices()],
                affinity=sorted(os.sched_getaffinity(0)), batch_size=n,
                timing_convention='iteration_seconds is wall time for the whole batch; do not sum across chains',
                pypomp_revision=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                pypomp_dirty=bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)),
                names=NAMES, prior_center=CENTER.tolist(), prior_sd=(SCALE*a.prior_scale).tolist())
    if a.method == 'nuts':
        meta['nuts_dependencies'] = {name: importlib.metadata.version(name)
                                    for name in ['blackjax', 'jaxopt', 'fastprogress']}
        if meta['nuts_dependencies']['blackjax'] != '1.3':
            raise ValueError('BlackJAX 1.3 required')
        meta['info_columns'] = ['position_changed', 'randomness_accepted', 'trajectory_acceptance',
                                'energy', 'randomness_log_acceptance', 'force_norm', 'divergent',
                                'integration_steps', 'tree_expansions', 'stopped_by_turning']
    folders = [a.output/f'chain{c}' for c in range(n)]
    for c, folder in enumerate(folders):
        source = folder/'source'
        source.mkdir(parents=True)
        for f in Path(__file__).parent.glob('*.py'):
            shutil.copy2(f, source/f.name)
        cfg = {k: v for k, v in vars(a).items() if k not in ['output', 'initials', 'metrics', 'seeds']}
        cfg.update(meta, output=str(folder), seed=a.seeds[c], initial=str(a.initials[c]),
                   metric=str(a.metrics[c]), optimize=0, optimize_particles=a.particles,
                   sort_particles=False, steps=4, batch_chain_index=c,
                   source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')})
        write_json(folder/'configuration.json', cfg)
        np.save(folder/'initial.npy', zs[c]); np.save(folder/'warm_start.npy', zs[c])
        np.save(folder/'initial_chol.npy', chols[c]); np.save(folder/'optimization.npy', np.array([]))
    m = setup(a.particles, a.alpha, a.prior_scale, force_particles=a.force_particles,
              force_precision=a.force_precision)
    init, kernel = make_kernel(m['loglik'], m['logprior'], lambda z, u: m['value_force'](z, u)[1],
                               a.method, 4, a.max_depth)
    # Match run.py's two initial splits, including its unused jitter key.
    keys = jax.vmap(jax.random.key)(jnp.asarray(a.seeds))
    keys = jax.vmap(jax.random.split)(keys)[:, 0]
    splits = jax.vmap(jax.random.split)(keys)
    keys, us = splits[:, 0], splits[:, 1]
    start = time.monotonic()
    states = jax.jit(jax.vmap(init))(jnp.asarray(zs), us)
    jax.block_until_ready(states)
    if not np.isfinite(states.loglik).all():
        raise ValueError('Initial particle filter collapsed')
    initialize_seconds = time.monotonic()-start
    advance = batched_transition(kernel)
    eps = np.full(n, a.step_size)
    metric_traces = np.sum(chols**2, axis=(1, 2))
    traces, infos, steps, times = [], [], [], []
    metrics = [[] for _ in range(n)]
    for i in range(a.warmup+a.draws):
        t = time.monotonic()
        states, keys, info = advance(states, keys, jnp.asarray(eps), jnp.asarray(chols))
        jax.block_until_ready((states, info))
        times.append(time.monotonic()-t)
        prior = np.asarray(jax.vmap(m['logprior'])(states.z))
        traces.append(np.column_stack([np.asarray(states.loglik), prior, np.asarray(states.z)]))
        infos.append(np.asarray(info)); steps.append(eps.copy())
        trace_array = np.asarray(traces)
        for c in range(n):
            chol, eps[c], record = adapt_tuning(i, infos[-1][c], trace_array[:, c], chols[c],
                                               eps[c], metric_traces[c], a.warmup, a.max_step_size, a.method)
            chols[c] = np.asarray(chol)
            if record is not None:
                metrics[c].append(record)
        if (i+1) % a.checkpoint == 0 or i+1 == a.warmup+a.draws:
            for c, folder in enumerate(folders):
                tmp = folder/'checkpoint.tmp.npz'
                np.savez_compressed(tmp, trace=trace_array[:, c], info=np.asarray(infos)[:, c],
                                    step_size=np.asarray(steps)[:, c], iteration_seconds=np.asarray(times),
                                    z=np.asarray(states.z[c]), u=np.asarray(jax.random.key_data(states.u[c])),
                                    force=np.asarray(states.force[c]), loglik=np.asarray(states.loglik[c]),
                                    key=np.asarray(jax.random.key_data(keys[c])), chol=chols[c], next_step_size=eps[c])
                tmp.replace(folder/'checkpoint.npz')
                recent = np.asarray(infos)[-a.checkpoint:, c]
                status = dict(iteration=i+1, warmup=a.warmup, total=a.warmup+a.draws,
                              complete=i+1 == a.warmup+a.draws,
                              phase='warmup' if i < a.warmup else 'sampling',
                              seconds=time.monotonic()-start, batch_initialize_seconds=initialize_seconds,
                              loglik=float(states.loglik[c]), step_size=float(eps[c]),
                              parameter_acceptance_recent=float(recent[:, 0].mean()),
                              randomness_acceptance_recent=float(recent[:, 1].mean()))
                write_json(folder/'status.json', status); write_json(folder/'metrics.json', metrics[c])
                if status['complete']:
                    write_json(folder/'completion.json', status)
            write_json(a.output/'status.json', dict(iteration=i+1, complete=status['complete'],
                                                   seconds=time.monotonic()-start, batch_size=n))
            print(json.dumps({'iteration': i+1, 'batch_seconds': times[-1], 'step_sizes': eps.tolist()}), flush=True)


if __name__ == '__main__':
    main()
