import numpy as np

from ditlevsen.block_smc import (
    _parameter_step_size,
    block_bootstrap_filter,
    complete_block_path_loglik,
    fit_block_pseudo_score,
    sample_block_smoothing_path,
)
from ditlevsen.data import load_dacca_data
from ditlevsen.model import default_unconstrained_parameters


def test_short_block_filter_and_complete_path_are_finite():
    data = load_dacca_data(20, max_observations=3)
    start = default_unconstrained_parameters()
    filtered = block_bootstrap_filter(
        start,
        data,
        particles=16,
        seed=2,
        endpoint_relative_floor=1e-12,
    )
    assert np.isfinite(filtered.loglik)
    assert filtered.loglik_increments.shape == (3,)
    assert np.all((filtered.invalid_fraction >= 0.0))
    assert np.all((filtered.invalid_fraction <= 1.0))

    path = sample_block_smoothing_path(
        start,
        data,
        particles=16,
        seed=3,
        endpoint_relative_floor=1e-12,
    )
    assert path.path.shape == (4, 6)
    assert path.valid_path
    assert np.isfinite(
        complete_block_path_loglik(
            start,
            path.path,
            data,
            endpoint_relative_floor=1e-12,
        )
    )


def test_short_block_score_fit_moves_finitely():
    data = load_dacca_data(20, max_observations=3)
    start = default_unconstrained_parameters()
    result = fit_block_pseudo_score(
        start,
        data,
        particles=16,
        iterations=1,
        learning_rate=1e-4,
        seed=5,
        endpoint_relative_floor=1e-12,
    )
    assert result.completed_updates == 1
    assert result.termination_reason == "completed"
    assert np.all(np.isfinite(result.marginal_loglik_trace))
    assert np.all(np.isfinite(result.complete_loglik_trace))
    assert not np.allclose(result.unconstrained, start)


def test_parameter_step_is_constant_initially_then_decays():
    assert _parameter_step_size(0, 0.05, 30, 0.6) == 0.05
    assert _parameter_step_size(29, 0.05, 30, 0.6) == 0.05
    assert _parameter_step_size(30, 0.05, 30, 0.6) == 0.05
    np.testing.assert_allclose(
        _parameter_step_size(31, 0.05, 30, 0.6),
        0.05 * 2.0**-0.6,
    )
    assert _parameter_step_size(100, 0.05, 30, 0.6) < 0.005
