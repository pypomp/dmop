import jax.numpy as jnp
import numpy as np

from ditlevsen.data import load_dacca_data
from ditlevsen.model import (
    DEFAULT_PARAMETERS,
    decode_parameters,
    default_unconstrained_parameters,
    drift,
    initial_state,
)


def test_data_grid_matches_monthly_observations():
    data = load_dacca_data(20, max_observations=3)
    assert data.step_covariates.shape == (3, 20, 9)
    assert data.observation_covariates.shape == (3, 9)
    interval_starts = np.concatenate(([1891.0], data.observation_times[:-1]))
    np.testing.assert_allclose(data.step_times[:, 0], interval_starts)
    np.testing.assert_allclose(
        data.step_times[:, -1],
        interval_starts
        + (data.observation_times - interval_starts) * (19.0 / 20.0),
    )
    assert data.observations.tolist() == [2641.0, 939.0, 905.0]


def test_parameter_round_trip_and_initial_state():
    unconstrained = default_unconstrained_parameters()
    theta = decode_parameters(jnp.asarray(unconstrained))
    assert np.isclose(float(theta["gamma"]), DEFAULT_PARAMETERS["gamma"])
    assert np.isclose(float(theta["beta_trend"]), DEFAULT_PARAMETERS["beta_trend"])
    state = np.asarray(initial_state(jnp.asarray(unconstrained)))
    assert state.shape == (6,)
    assert np.isclose(state[:5].sum(), 1.0)
    assert state[-1] == 0.0


def test_drift_is_finite():
    data = load_dacca_data(20, max_observations=1)
    unconstrained = jnp.asarray(default_unconstrained_parameters())
    value = drift(
        initial_state(unconstrained), unconstrained, jnp.asarray(data.initial_covariates)
    )
    assert np.all(np.isfinite(np.asarray(value)))
