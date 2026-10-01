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
from ditlevsen.smc_benchmark import (
    _checkpoint_targets,
    _combine_loglik_replicates,
)


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

    guided = block_bootstrap_filter(
        start,
        data,
        particles=16,
        seed=2,
        endpoint_relative_floor=1e-12,
        proposal="guided",
    )
    assert np.isfinite(guided.loglik)
    assert guided.loglik_increments.shape == (3,)

    path = sample_block_smoothing_path(
        start,
        data,
        particles=16,
        seed=3,
        endpoint_relative_floor=1e-12,
        proposal="guided",
    )
    assert path.path.shape == (4, 6)
    assert path.valid_path
    assert 0 <= path.backward_fallbacks <= 2
    assert np.isfinite(
        complete_block_path_loglik(
            start,
            path.path,
            data,
            endpoint_relative_floor=1e-12,
        )
    )


def test_unknown_block_proposal_is_rejected():
    data = load_dacca_data(20, max_observations=1)
    start = default_unconstrained_parameters()
    with np.testing.assert_raises_regex(ValueError, "proposal"):
        block_bootstrap_filter(start, data, particles=4, proposal="unknown")


def test_block_score_fit_rejects_nonpositive_guard_interval():
    data = load_dacca_data(20, max_observations=1)
    start = default_unconstrained_parameters()
    with np.testing.assert_raises_regex(ValueError, "guard_interval"):
        fit_block_pseudo_score(
            start,
            data,
            iterations=0,
            likelihood_guard_interval=0,
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
        likelihood_guard_particles=0,
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


def test_checkpoint_targets_use_nearest_iterates_and_include_final():
    elapsed = np.array([2.0, 48.0, 103.0, 149.0, 203.0, 247.0])
    assert _checkpoint_targets(elapsed, 100.0) == [
        (0, 0),
        (100, 2),
        (200, 4),
        (247, 5),
    ]


def test_combine_loglik_replicates_averages_likelihood_scale():
    combined, standard_error = _combine_loglik_replicates(
        np.log(np.array([1.0, 3.0]))
    )
    np.testing.assert_allclose(combined, np.log(2.0))
    np.testing.assert_allclose(standard_error, 0.5)
