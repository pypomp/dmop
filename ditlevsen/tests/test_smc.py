import numpy as np

from ditlevsen.data import load_dacca_data
from ditlevsen.model import default_unconstrained_parameters
from ditlevsen.smc import (
    bootstrap_filter,
    complete_path_loglik,
    fit_pseudo_score,
    sample_smoothing_path,
)


def test_short_bootstrap_filter_is_finite_and_reports_diagnostics():
    data = load_dacca_data(5, max_observations=3)
    result = bootstrap_filter(
        default_unconstrained_parameters(),
        data,
        particles=16,
        seed=2,
        order=2,
        relative_floor=1e-7,
    )
    assert np.isfinite(result.loglik)
    assert result.loglik_increments.shape == (3,)
    assert result.ess.shape == (3,)
    assert np.all(result.ess >= 1.0)
    assert np.all(result.ess <= 16.0 + 1e-8)
    assert np.all((result.invalid_fraction >= 0.0) & (result.invalid_fraction <= 1.0))


def test_path_smc_returns_a_valid_shape_and_complete_density():
    data = load_dacca_data(5, max_observations=3)
    start = default_unconstrained_parameters()
    result = sample_smoothing_path(
        start, data, particles=24, seed=7, order=2, relative_floor=1e-7
    )
    assert result.path.shape == (3 * 5 + 1, 6)
    assert result.valid_path
    assert np.isfinite(
        complete_path_loglik(
            start, result.path, data, order=2, relative_floor=1e-7
        )
    )
    assert 1 <= result.unique_initial_ancestors <= 24


def test_short_smc_score_fit_moves_finitely():
    data = load_dacca_data(5, max_observations=3)
    start = default_unconstrained_parameters()
    result = fit_pseudo_score(
        start,
        data,
        particles=24,
        iterations=1,
        learning_rate=1e-4,
        seed=11,
        order=2,
        relative_floor=1e-7,
    )
    assert result.completed_updates == 1
    assert result.parameter_trace.shape == (2, start.size)
    assert np.all(np.isfinite(result.marginal_loglik_trace))
    assert np.all(np.isfinite(result.complete_loglik_trace))
    assert not np.allclose(result.unconstrained, start)
    assert result.termination_reason == "completed"
    assert result.accepted_step_size_trace.shape == (2,)
    assert result.backtrack_count_trace.shape == (2,)
    assert np.isfinite(result.accepted_step_size_trace[0])
    assert np.isnan(result.accepted_step_size_trace[-1])


def test_short_smc_score_fit_accepts_bridge_rejuvenation():
    data = load_dacca_data(5, max_observations=3)
    result = fit_pseudo_score(
        default_unconstrained_parameters(),
        data,
        particles=16,
        iterations=1,
        learning_rate=1e-4,
        seed=21,
        order=2,
        relative_floor=1e-7,
        bridge_particles=6,
    )
    assert np.all(np.isfinite(result.bridge_update_fraction_trace))
    assert np.all((result.bridge_update_fraction_trace >= 0.0))
    assert np.all((result.bridge_update_fraction_trace <= 1.0))
