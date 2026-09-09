import jax
import jax.numpy as jnp
import numpy as np

from ditlevsen.data import load_dacca_data
from ditlevsen.model import (
    default_unconstrained_parameters,
    diffusion,
    drift,
    initial_state,
)
from ditlevsen.transition import (
    completed_covariance,
    ditlevsen_block_transition,
    ditlevsen_mean,
    ditlevsen_covariance,
    lie_bracket_diagnostics,
    stratonovich_drift,
    transition_diagnostics,
)


def test_first_bracket_ditlevsen_covariance_has_rank_at_most_two():
    data = load_dacca_data(20, max_observations=1)
    unconstrained = default_unconstrained_parameters()
    state = np.asarray(initial_state(jnp.asarray(unconstrained)))
    diagnostics = transition_diagnostics(
        state,
        unconstrained,
        data.initial_covariates,
        1 / 240,
        order=2,
    )
    assert diagnostics["rank"] == 2
    covariance = np.asarray(
        ditlevsen_covariance(
            jnp.asarray(state),
            jnp.asarray(unconstrained),
            jnp.asarray(data.initial_covariates),
            1 / 240,
        )
    )
    assert np.linalg.matrix_rank(covariance) <= 2


def test_completed_covariance_is_symmetric_positive_with_floor():
    data = load_dacca_data(20, max_observations=1)
    unconstrained = jnp.asarray(default_unconstrained_parameters())
    state = initial_state(unconstrained)
    covariance = np.asarray(
        completed_covariance(
            state,
            unconstrained,
            jnp.asarray(data.initial_covariates),
            1 / 240,
            relative_floor=1e-10,
        )
    )
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_monthly_block_composes_unregularized_order_1p5_moments():
    data = load_dacca_data(20, max_observations=1)
    unconstrained = jnp.asarray(default_unconstrained_parameters())
    state = initial_state(unconstrained)
    mean, covariance = ditlevsen_block_transition(
        state,
        unconstrained,
        jnp.asarray(data.step_covariates[0]),
        1 / 240,
        endpoint_relative_floor=0.0,
    )
    covariance = np.asarray(covariance)
    assert np.all(np.isfinite(np.asarray(mean)))
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() >= -1e-14 * eigenvalues.max()

    _, stabilized = ditlevsen_block_transition(
        state,
        unconstrained,
        jnp.asarray(data.step_covariates[0]),
        1 / 240,
        endpoint_relative_floor=1e-12,
    )
    assert np.isfinite(np.linalg.cholesky(np.asarray(stabilized))).all()


def test_ditlevsen_mean_and_bracket_depth_are_finite():
    data = load_dacca_data(20, max_observations=1)
    unconstrained = default_unconstrained_parameters()
    state = np.asarray(initial_state(jnp.asarray(unconstrained)))
    mean = ditlevsen_mean(
        jnp.asarray(state),
        jnp.asarray(unconstrained),
        jnp.asarray(data.initial_covariates),
        1 / 12,
    )
    assert np.all(np.isfinite(np.asarray(mean)))

    diagnostics = lie_bracket_diagnostics(
        state,
        unconstrained,
        data.initial_covariates,
        maximum_depth=5,
    )
    assert [int(row["rank"]) for row in diagnostics] == [1, 2, 3, 4, 5, 6]


def test_hormander_diagnostic_uses_stratonovich_drift():
    data = load_dacca_data(20, max_observations=1)
    unconstrained = jnp.asarray(default_unconstrained_parameters())
    state = initial_state(unconstrained)
    covariates = jnp.asarray(data.initial_covariates)
    noise = diffusion(state, unconstrained, covariates)
    expected = drift(state, unconstrained, covariates) - 0.5 * (
        jax.jacfwd(diffusion, argnums=0)(state, unconstrained, covariates)
        @ noise
    )
    np.testing.assert_allclose(
        np.asarray(stratonovich_drift(state, unconstrained, covariates)),
        np.asarray(expected),
    )
