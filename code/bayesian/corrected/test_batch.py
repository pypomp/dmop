"""Batched execution must retain the independent scalar transition laws."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from batch import batched_transition
from kernels import make_kernel
from extend import adapt_tuning


def test_supplied_metric_stays_fixed_during_step_adaptation():
    chol = np.array([[1., 0.], [.2, .7]])
    info = np.array([1., 1., .8])
    # No covariance data are needed when a shared preparation metric is fixed.
    new, eps, record = adapt_tuning(399, info, [], chol, .03,
                                   np.sum(chol**2), 500, 2., 'nuts', adapt_metric=False)
    np.testing.assert_array_equal(new, chol)
    assert record is None and eps > .03
    new, eps2, record = adapt_tuning(500, info, [], chol, eps,
                                    np.sum(chol**2), 500, 2., 'nuts', adapt_metric=False)
    assert eps2 == eps and record is None


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
