"""Continue a completed or interrupted batch using its frozen implementation.

The new directory contains each complete parent trace and its continuation.
The original batch size, lane order, cached forces, random states and tuning
are retained. Incomplete per-chain writes may leave different trace lengths;
each lane then resumes its own iteration count, without dropping any draws.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def load_parent(parent, recover, draws):
    import numpy as np
    first = json.loads((parent/'chain0/configuration.json').read_text())
    n = first['batch_size']
    folders = [parent/f'chain{c}' for c in range(n)]
    configs, arrays, hashes = [], [], []
    if recover:
        audit = json.loads((parent/'interruption.json').read_text())
        if not audit.get('verified_terminal'):
            raise ValueError('Recovery requires a verified terminal process')
        if draws is not None:
            raise ValueError('Recovery retains the original run length')
    elif draws is None or draws <= 0:
        raise ValueError('Specify positive additional draws for a completed batch')
    for c, folder in enumerate(folders):
        cfg = json.loads((folder/'configuration.json').read_text())
        if cfg['batch_chain_index'] != c:
            raise ValueError('Changed batch lane order')
        for field in ['batch_size', 'method', 'particles', 'force_particles', 'force_precision',
                      'alpha', 'prior_scale', 'warmup', 'draws', 'max_depth', 'max_step_size',
                      'fixed_metric', 'source_sha256', 'jax', 'x64', 'devices', 'pypomp_revision',
                      'names', 'prior_center', 'prior_sd', 'sort_particles', 'steps', 'nuts_dependencies']:
            if cfg.get(field) != first.get(field):
                raise ValueError(f'Inconsistent batch configuration: {field}')
        for name, digest in cfg['source_sha256'].items():
            if hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Changed frozen source: {folder/name}')
        payload = (folder/'checkpoint.npz').read_bytes()
        with np.load(io.BytesIO(payload)) as saved:
            cp = {k: saved[k].copy() for k in saved.files}
        count = len(cp['trace'])
        original_total = cfg['warmup'] + cfg['draws']
        if not 0 < count <= original_total:
            raise ValueError('Invalid checkpoint length')
        if not recover and (count != original_total or not (folder/'completion.json').exists()):
            raise ValueError('Parent batch is incomplete; use explicit recovery after verifying termination')
        if any(len(cp[k]) != count for k in ['info', 'step_size', 'iteration_seconds']):
            raise ValueError('Inconsistent checkpoint histories')
        np.testing.assert_array_equal(cp['z'], cp['trace'][-1, 2:])
        np.testing.assert_array_equal(cp['loglik'], cp['trace'][-1, 0])
        if not all(np.isfinite(cp[k]).all() for k in ['trace', 'force', 'chol', 'next_step_size']):
            raise ValueError('Nonfinite saved state')
        configs.append(cfg); arrays.append(cp); hashes.append(hashlib.sha256(payload).hexdigest())
    if len({cfg['seed'] for cfg in configs}) != n:
        raise ValueError('Repeated chain seeds')
    if recover and all(len(cp['trace']) == original_total for cp in arrays):
        raise ValueError('Batch already completed its requested iterations')
    total = original_total if recover else original_total + draws
    return folders, configs, arrays, hashes, total


def check_state(model, states, configs):
    """Check target and cached forces without replacing the saved values.

    A fused integrator and a standalone force evaluation can round differently
    in float32. The force check allows 16 single-precision ulps at the largest
    force component; the float64 target still uses the strict absolute check.
    Exact continuation is tested by replay, not inferred from this tolerance.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from kernels import finite_force
    ll = np.asarray(jax.jit(jax.vmap(model['loglik']))(states.z, states.u))
    np.testing.assert_allclose(ll, np.asarray(states.loglik), rtol=0, atol=1e-9)
    if configs[0]['method'] == 'pmrw':
        force = np.zeros_like(states.force)
    else:
        force = np.asarray(jax.jit(jax.vmap(
            lambda z, u: finite_force(model['value_force'](z, u)[1])))(states.z, states.u))
    cached = np.asarray(states.force)
    delta = np.max(np.abs(force-cached), axis=1)
    tolerance = (16*np.finfo(np.float32).eps*np.maximum(1., np.max(np.abs(cached), axis=1))
                 if configs[0].get('force_precision') == 'float32'
                 else 1e-8 + 1e-10*np.max(np.abs(cached), axis=1))
    if np.any(delta > tolerance):
        raise ValueError(f'Cached force differs beyond precision tolerance: {delta} > {tolerance}')
    if configs[0]['method'] == 'pmrw':
        np.testing.assert_array_equal(cached, np.zeros_like(cached))
    return dict(likelihood_differences=(ll-np.asarray(states.loglik)).tolist(),
                maximum_force_differences=delta.tolist(), force_tolerances=tolerance.tolist(),
                cached_values_preserved=True)


def advance_active(advance, states, keys, eps, chols, active):
    """Retain finished lanes while keeping the original execution shape."""
    import jax
    import jax.numpy as jnp
    new, next_keys, info = advance(states, keys, eps, chols)
    n = len(active)
    selected = jax.tree.map(lambda old, value: jnp.where(
        jnp.asarray(active).reshape((n,)+(1,)*(value.ndim-1)), value, old), states, new)
    return selected, jnp.where(jnp.asarray(active), next_keys, keys), info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int)
    p.add_argument('--recover-interrupted', action='store_true')
    p.add_argument('--checkpoint', type=int, default=25)
    a = p.parse_args()
    if a.checkpoint <= 0:
        p.error('Checkpoint interval must be positive')
    folders, configs, arrays, hashes, total = load_parent(a.parent, a.recover_interrupted, a.draws)
    cfg = configs[0]; n = len(configs)
    # Import the entire execution path from the frozen parent, not current code.
    sys.path.insert(0, str((folders[0]/'source').resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pypomp
    import importlib.metadata
    from model import setup
    from kernels import State, make_kernel
    from batch import batched_transition
    from extend import adapt_tuning
    from run import write_json
    repo = Path(pypomp.__file__).resolve().parents[1]
    if (subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != cfg['pypomp_revision']
            or subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)):
        raise ValueError('Pypomp source differs from parent')
    if jax.__version__ != cfg['jax'] or bool(jax.config.jax_enable_x64) != cfg['x64']:
        raise ValueError('JAX version or precision differs from parent')
    if [str(x) for x in jax.devices()] != cfg['devices']:
        raise ValueError('Use the same device backend as parent')
    for name, version in cfg.get('nuts_dependencies', {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Changed dependency: {name}')
    m = setup(cfg['particles'], cfg['alpha'], cfg['prior_scale'],
              force_particles=cfg['force_particles'], force_precision=cfg.get('force_precision'))
    states = State(jnp.asarray(np.stack([cp['z'] for cp in arrays])),
                   jax.random.wrap_key_data(jnp.asarray(np.stack([cp['u'] for cp in arrays]))),
                   jnp.asarray(np.stack([cp['loglik'] for cp in arrays])),
                   jnp.asarray(np.stack([cp['force'] for cp in arrays])))
    keys = jax.random.wrap_key_data(jnp.asarray(np.stack([cp['key'] for cp in arrays])))
    start = time.monotonic()
    verification = check_state(m, states, configs)
    _, kernel = make_kernel(m['loglik'], m['logprior'], lambda z, u: m['value_force'](z, u)[1],
                            cfg['method'], cfg['steps'], cfg['max_depth'])
    advance = batched_transition(kernel)
    traces = [list(cp['trace']) for cp in arrays]; infos = [list(cp['info']) for cp in arrays]
    steps = [list(cp['step_size']) for cp in arrays]; times = [list(cp['iteration_seconds']) for cp in arrays]
    chols = np.stack([cp['chol'] for cp in arrays]); eps = np.array([float(cp['next_step_size']) for cp in arrays])
    metric_traces = [np.sum(np.load(f/'initial_chol.npy')**2) for f in folders]
    metrics = [[x for x in json.loads((f/'metrics.json').read_text()) if x['iteration'] <= len(cp['trace'])]
               for f, cp in zip(folders, arrays)]
    parent_seconds = [json.loads((f/'status.json').read_text())['seconds'] for f in folders]
    a.output.mkdir(parents=True, exist_ok=False)
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shutil.copy2(__file__, a.output/'continuation_driver.py')
    outputs = [a.output/f'chain{c}' for c in range(n)]
    for c, (parent, out) in enumerate(zip(folders, outputs)):
        shutil.copytree(parent/'source', out/'source')
        for name in ['initial.npy', 'initial_chol.npy', 'warm_start.npy', 'optimization.npy']:
            shutil.copy2(parent/name, out/name)
        config = dict(configs[c], output=str(out), draws=total-cfg['warmup'], pid=os.getpid(),
                      parent=str(parent.resolve()), parent_checkpoint_sha256=hashes[c],
                      continuation_driver_sha256=driver_hash, recovery=a.recover_interrupted,
                      continuation_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'configuration.json', config)
    write_json(a.output/'state_check.json', verification)
    tick = 0
    while any(len(t) < total for t in traces):
        active = np.array([len(t) < total for t in traces])
        t = time.monotonic()
        # Finished lanes remain present to preserve batch shape, but neither
        # their proposed states nor their random keys are committed.
        states, keys, info = advance_active(advance, states, keys, jnp.asarray(eps), jnp.asarray(chols), active)
        jax.block_until_ready((states, info)); elapsed = time.monotonic()-t
        prior = np.asarray(jax.vmap(m['logprior'])(states.z)); info = np.asarray(info)
        for c in range(n):
            if not active[c]:
                continue
            i = len(traces[c])
            traces[c].append(np.r_[float(states.loglik[c]), prior[c], np.asarray(states.z[c])])
            infos[c].append(info[c]); steps[c].append(eps[c]); times[c].append(elapsed)
            extra = {'adapt_metric': False} if cfg.get('fixed_metric', False) else {}
            chol, eps[c], record = adapt_tuning(i, info[c], traces[c], chols[c], eps[c],
                                               metric_traces[c], cfg['warmup'], cfg['max_step_size'],
                                               cfg['method'], **extra)
            chols[c] = np.asarray(chol)
            if record is not None:
                metrics[c].append(record)
        tick += 1
        done = all(len(t) == total for t in traces)
        if tick % a.checkpoint == 0 or done:
            for c, out in enumerate(outputs):
                tmp = out/'checkpoint.tmp.npz'
                np.savez_compressed(tmp, trace=np.asarray(traces[c]), info=np.asarray(infos[c]),
                                    step_size=np.asarray(steps[c]), iteration_seconds=np.asarray(times[c]),
                                    z=np.asarray(states.z[c]), u=np.asarray(jax.random.key_data(states.u[c])),
                                    force=np.asarray(states.force[c]), loglik=np.asarray(states.loglik[c]),
                                    key=np.asarray(jax.random.key_data(keys[c])), chol=chols[c], next_step_size=eps[c])
                tmp.replace(out/'checkpoint.npz')
                count = len(traces[c]); recent = np.asarray(infos[c])[-a.checkpoint:]
                status = dict(iteration=count, warmup=cfg['warmup'], total=total, complete=count == total,
                              phase='warmup' if count < cfg['warmup'] else 'sampling',
                              seconds=parent_seconds[c]+time.monotonic()-start, loglik=float(states.loglik[c]),
                              step_size=float(eps[c]), parameter_acceptance_recent=float(recent[:, 0].mean()),
                              randomness_acceptance_recent=float(recent[:, 1].mean()))
                write_json(out/'status.json', status); write_json(out/'metrics.json', metrics[c])
                if count == total:
                    write_json(out/'completion.json', status)
            write_json(a.output/'status.json', dict(iterations=[len(t) for t in traces], complete=done,
                                                   seconds=max(parent_seconds)+time.monotonic()-start, batch_size=n))
            print(json.dumps({'iterations': [len(t) for t in traces], 'batch_seconds': elapsed}), flush=True)


if __name__ == '__main__':
    main()
