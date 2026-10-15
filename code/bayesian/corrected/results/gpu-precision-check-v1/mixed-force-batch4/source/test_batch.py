"""Batched execution must retain the independent scalar transition laws."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from batch import batched_transition
from kernels import make_kernel


@pytest.mark.parametrize('method', ['nuts', 'pmrw'])
def test_scalar_and_batch_transitions_agree(method):
    ll = lambda z, u: -.5*z@z + .2*z[0]*jax.random.normal(u) - .02*z[0]**2
    prior = lambda z: -.5*z@z
    # A rounded surrogate force exercises the mixed-precision boundary.
    force = lambda z, u: (-1.7*z.astype(jnp.float32)).astype(jnp.float64)
    init, kernel = make_kernel(ll, prior, force, method, max_depth=4)
    zs = jnp.array([[.1, -.2], [1., -.5], [-.3, .7], [2., 1.]])
    us = jax.random.split(jax.random.key(52), 4)
    keys = jax.random.split(jax.random.key(53), 4)
    states = jax.vmap(init)(zs, us)
    scalar = [jax.tree.map(lambda x: x[c], states) for c in range(4)]
    scalar_keys = [keys[c] for c in range(4)]
    eps = jnp.array([.05, .2, .4, .8])
    chols = jnp.tile(jnp.array([[1., 0.], [.2, .7]])[None], (4, 1, 1))
    advance = batched_transition(kernel)
    for _ in range(8):
        states, keys, infos = advance(states, keys, eps, chols)
        for c in range(4):
            scalar_keys[c], k = jax.random.split(scalar_keys[c])
            scalar[c], info = kernel(scalar[c], k, eps[c], chols[c])
            np.testing.assert_array_equal(jax.random.key_data(keys[c]), jax.random.key_data(scalar_keys[c]))
            np.testing.assert_array_equal(jax.random.key_data(states.u[c]), jax.random.key_data(scalar[c].u))
            for field in ['z', 'loglik', 'force']:
                np.testing.assert_allclose(getattr(states, field)[c], getattr(scalar[c], field), atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(infos[c], info, atol=1e-10, rtol=1e-10)
