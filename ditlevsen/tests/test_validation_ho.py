import numpy as np

from ditlevsen.validation_ho import (
    HOParameters,
    conditional_particle_filter,
    exact_transition,
    fit_ho_smc_score,
    simulate_ho,
    strong_15_transition,
)


def test_strong_15_transition_has_hypoelliptic_variance_scaling():
    parameters = HOParameters()
    _, covariance = strong_15_transition(parameters, 0.02)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    expected_v_variance = parameters.diffusion**2 * 0.02**3 / 3.0
    assert np.isclose(covariance[0, 0], expected_v_variance)


def test_strong_15_moments_are_close_to_exact_moments():
    exact_mean, exact_covariance = exact_transition(HOParameters(), 0.02)
    approximate_mean, approximate_covariance = strong_15_transition(
        HOParameters(), 0.02
    )
    np.testing.assert_allclose(approximate_mean, exact_mean, atol=3e-5)
    # Equation (34) retains a different subset of O(dt^3) terms than the exact
    # covariance expansion, so the rough-coordinate error is itself O(dt^3).
    np.testing.assert_allclose(approximate_covariance, exact_covariance, atol=3e-6)


def test_paper_particle_filter_matches_analytic_filter():
    trajectory = simulate_ho(points=300, seed=9)
    result = conditional_particle_filter(
        trajectory[:, 0], particles=500, seed=10
    )
    difference = result.filtered_mean - result.analytic_mean
    assert np.sqrt(np.mean(difference**2)) < 0.08
    assert np.isfinite(result.loglik)
    assert np.all((result.ess >= 1.0) & (result.ess <= 500.0 + 1e-8))
    assert result.sampled_path.shape == (300,)
    assert 1 <= result.unique_initial_ancestors <= 500


def test_short_ho_score_fit_is_finite_and_moves():
    trajectory = simulate_ho(points=100, seed=11)
    result = fit_ho_smc_score(
        trajectory[:, 0], particles=50, iterations=2, learning_rate=0.001, seed=12
    )
    assert result.parameter_trace.shape == (3, 3)
    assert np.all(np.isfinite(result.marginal_loglik_trace))
    assert not np.allclose(result.parameter_trace[0], result.parameter_trace[-1])
