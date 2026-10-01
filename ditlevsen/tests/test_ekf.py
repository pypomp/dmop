import jax.numpy as jnp
import numpy as np

from ditlevsen.data import load_dacca_data
from ditlevsen.ekf import ekf_loglik, fit_ekf
from ditlevsen.model import default_unconstrained_parameters


def test_short_ekf_likelihood_and_fit_are_finite():
    data = load_dacca_data(5, max_observations=5)
    start = default_unconstrained_parameters()
    value = ekf_loglik(
        jnp.asarray(start),
        jnp.asarray(data.step_covariates),
        jnp.asarray(data.observation_covariates[:, 2]),
        jnp.asarray(data.observations),
        order=2,
    )
    assert np.isfinite(float(value))
    result = fit_ekf(start, data, order=2, iterations=1, learning_rate=0.001)
    assert result.converged
    assert len(result.objective_trace) == 2
    assert result.parameter_trace.shape == (2, start.size)
    assert result.elapsed_trace.shape == (2,)
    assert np.all(np.diff(result.elapsed_trace) >= 0.0)
    assert np.all(np.isfinite(result.objective_trace))
