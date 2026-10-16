"""Preserve lane order, completed lanes, source and cached random states."""
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from batch import batched_transition
from batch_extend import advance_active, check_state, load_parent
from kernels import make_kernel


def parent_fixture(path):
    for c, count in enumerate([6, 5]):
        folder = path/f'chain{c}'; source = folder/'source'; source.mkdir(parents=True)
        (source/'model.py').write_text('# frozen fixture\n')
        cfg = dict(batch_size=2, batch_chain_index=c, seed=70+c, warmup=3, draws=4,
                   source_sha256={'model.py': hashlib.sha256((source/'model.py').read_bytes()).hexdigest()})
        (folder/'configuration.json').write_text(json.dumps(cfg))
        trace = np.arange(count*4, dtype=float).reshape(count, 4)
        np.savez(folder/'checkpoint.npz', trace=trace, info=np.zeros((count, 6)),
                 step_size=np.ones(count), iteration_seconds=np.ones(count),
                 z=trace[-1, 2:], loglik=trace[-1, 0], force=np.zeros(2),
                 u=np.array([0, 7], dtype=np.uint32), key=np.array([0, 8], dtype=np.uint32),
                 chol=np.eye(2), next_step_size=1.)
    (path/'interruption.json').write_text(json.dumps({'verified_terminal': True}))


def test_partial_batch_checkpoint_is_retained(tmp_path):
    parent_fixture(tmp_path)
    _, cfg, arrays, hashes, total = load_parent(tmp_path, True, None)
    assert [len(x['trace']) for x in arrays] == [6, 5]
    assert total == 7 and [x['seed'] for x in cfg] == [70, 71]
    assert len(set(hashes)) == 2
    with pytest.raises(ValueError, match='incomplete'):
        load_parent(tmp_path, False, 2)


@pytest.mark.parametrize('corruption', ['source', 'lane', 'state', 'terminal'])
def test_reject_changed_parent(tmp_path, corruption):
    parent_fixture(tmp_path)
    if corruption == 'source':
        (tmp_path/'chain1/source/model.py').write_text('changed')
    elif corruption == 'lane':
        f = tmp_path/'chain1/configuration.json'; d = json.loads(f.read_text())
        d['batch_chain_index'] = 0; f.write_text(json.dumps(d))
    elif corruption == 'state':
        f = tmp_path/'chain1/checkpoint.npz'
        with np.load(f) as cp:
            d = {k: cp[k].copy() for k in cp.files}
        d['z'][0] += 1.; np.savez(f, **d)
    else:
        (tmp_path/'interruption.json').write_text(json.dumps({'verified_terminal': False}))
    with pytest.raises((ValueError, AssertionError)):
        load_parent(tmp_path, True, None)


@pytest.mark.parametrize('method', ['nuts', 'pmrw'])
def test_unequal_lane_replay_preserves_random_states(method):
    ll = lambda z, u: -.5*z@z + .1*z[0]*jax.random.normal(u) - .005*z[0]**2
    prior = lambda z: -.5*z@z
    force = lambda z, u: (-1.7*z.astype(jnp.float32)).astype(jnp.float64)
    init, kernel = make_kernel(ll, prior, force, method, max_depth=3)
    states = jax.vmap(init)(jnp.array([[.1, .2], [-.2, .3]]), jax.random.split(jax.random.key(8), 2))
    keys = jax.random.split(jax.random.key(9), 2)
    eps = jnp.array([.2, .3]); chols = jnp.tile(jnp.eye(2)[None], (2, 1, 1))
    advance = batched_transition(kernel)
    snapshots = []
    for i in range(10):
        states, keys, _ = advance(states, keys, eps, chols)
        snapshots.append((states, keys))
    # Restore lane0 after7 steps and lane1 after6, as after a partial save.
    restored = jax.tree.map(lambda a, b: jnp.stack([a[0], b[1]]), snapshots[6][0], snapshots[5][0])
    restored_keys = jnp.stack([snapshots[6][1][0], snapshots[5][1][1]])
    counts = np.array([7, 6])
    while np.any(counts < 10):
        active = counts < 10
        restored, restored_keys, _ = advance_active(advance, restored, restored_keys, eps, chols, active)
        counts += active
    for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(states)):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(jax.random.key_data(restored_keys), jax.random.key_data(keys))
    model = {'loglik': ll, 'value_force': lambda z, u: (ll(z, u), force(z, u))}
    check_state(model, restored, [{'method': method, 'force_precision': 'float32'}])
    with pytest.raises(AssertionError):
        check_state(model, restored._replace(loglik=restored.loglik+1.), [{'method': method}])
    with pytest.raises((ValueError, AssertionError)):
        check_state(model, restored._replace(force=restored.force+1.), [{'method': method}])
