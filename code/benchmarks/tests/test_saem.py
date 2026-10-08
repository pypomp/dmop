import jax
import jax.numpy as jnp
import numpy as np

import models
from saem import make_saem
from smoothing import make_smoother


def test_oscillator_statistics_match_complete_path_likelihood():
    model, y = models.oscillator(points=120)
    z = jnp.log(jnp.array([3., .8, .6]))
    sample, complete, _ = make_smoother("oscillator", model, y, 100)
    path, _ = sample(z, jax.random.key(197))
    sufficient, objective, maximize = make_saem("oscillator", model, y,
                                               np.log([[.1, 10.], [.05, 5.], [.05, 2.]]))
    statistics = sufficient(path)
    for x in [z, z+jnp.array([.1, -.2, .2])]:
        np.testing.assert_allclose(objective(x, statistics), complete(x, path), atol=1e-7)
        np.testing.assert_allclose(jax.grad(objective)(x, statistics), jax.grad(complete)(x, path),
                                   rtol=1e-6, atol=1e-6)
    candidate, valid = maximize(z, statistics)
    assert valid and objective(candidate, statistics) >= objective(z, statistics)-1e-7


def test_linear_mstep_maximizes_the_same_complete_likelihood():
    model, y = models.linear_gaussian(points=80)
    z = jnp.array([.2, .8])
    sample, complete, _ = make_smoother("linear", model, y, 100)
    path, _ = sample(z, jax.random.key(198))
    sufficient, objective, maximize = make_saem("linear", model, y, np.array([[-.98, .98]]*2))
    statistics = sufficient(path)
    np.testing.assert_allclose(jax.grad(objective)(z, statistics), jax.grad(complete)(z, path), atol=1e-8)
    candidate, valid = maximize(z, statistics)
    assert valid and complete(candidate, path) >= complete(z, path)
    np.testing.assert_allclose(jax.grad(complete)(candidate, path), 0., atol=1e-8)
