"""Recompute Fig. 1A's likelihood curves; plot.py handles inexpensive edits."""

import argparse
import json
import os
from pathlib import Path

# The float32 particle genealogy is sensitive to GPU reduction order.
# Apply these before importing JAX; preserve any additional user XLA flags.
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Preserve the notebook's float32 arithmetic and pre-partitionable Threefry
# stream; modern JAX otherwise changes the random draws even at the same seed.
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_threefry_partitionable', False)
jax.config.update('jax_default_prng_impl', 'threefry2x32')

from legacy_filters import pfilter_debug, mop_debug
from legacy_model import transform_thetas

HERE = Path(__file__).resolve().parent


def theta(gamma):
    return transform_thetas(
        gamma, 0.06, 0, 19.1, np.exp(-4.5), jnp.array(1), -0.00498,
        3.13, 0.23, jnp.array([0.747, 6.38, -3.44, 4.23, 3.33, 4.55]),
        jnp.log(jnp.array([0.184, 0.0786, 0.0584, 0.00917, 0.000208, 0.0124])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'curves.csv')
    parser.add_argument('--particles', type=int, default=10000)
    args = parser.parse_args()
    ys = jnp.array(pd.read_csv(HERE / 'data/dacca.csv', index_col=0)['cholera.deaths'].values)
    covars = pd.read_csv(HERE / 'data/covars.csv', index_col=0).reset_index(drop=True)
    covars.index = pd.read_csv(HERE / 'data/covart.csv', index_col=0).reset_index(drop=True).squeeze()
    covars = jnp.array(covars.reindex(np.array([1891 + i / 240 for i in range(12037)])).interpolate().values)
    # The notebook rounds 101 grid points to one decimal place. Evaluate its
    # 11 distinct values once: the filter resets the seed on every call.
    grid = np.unique(np.linspace(17.5, 18.5, 101).round(1))
    pf = jax.jit(pfilter_debug, static_argnums=2)
    _, _, meas_phi = pf(theta(18.0), ys, args.particles, covars)
    rows = []
    for gamma in grid:
        current = theta(gamma)
        pf_value, _, _ = pf(current, ys, args.particles, covars)
        row = {'gamma': gamma, 'particle_filter': -float(pf_value)}
        for alpha, name in [(1.0, 'alpha_1'), (0.97, 'alpha_097'), (0.0, 'alpha_0')]:
            value, _ = mop_debug(current, ys, args.particles, meas_phi, covars, alpha=alpha)
            row[name] = -float(value)
        rows.append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    args.output.with_suffix('.json').write_text(json.dumps({
        'source_repository': 'https://github.com/hetankevin/diffPomp',
        'source_revision': 'ab31c911ae223aa91eb4e216d7d33710103050c3',
        'source_notebook': 'cholera_mop.ipynb', 'source_cells_zero_based': [2, 3, 4],
        'particles': args.particles, 'baseline_gamma': 18.0, 'seed_each_month': 0,
        'jax': jax.__version__, 'jax_enable_x64': jax.config.jax_enable_x64,
        'jax_default_prng_impl': str(jax.config.jax_default_prng_impl),
        'jax_threefry_partitionable': jax.config.jax_threefry_partitionable,
        'xla_flags': os.environ.get('XLA_FLAGS', ''),
        'devices': [str(device) for device in jax.devices()],
        'numpy': np.__version__, 'pandas': pd.__version__,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
