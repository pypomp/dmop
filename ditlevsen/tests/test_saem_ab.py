from types import SimpleNamespace

import numpy as np

from ditlevsen.saem_ab import score_settings, selected_score
from ditlevsen.saem_ab_report import paired_mcse


def test_selection_uses_last_finite_estimate_before_deadline():
    result = SimpleNamespace(elapsed_trace=np.array([1., 2., 3., 4.]),
        parameter_trace=np.array([[1.], [2.], [np.nan], [4.]]),
        marginal_loglik_trace=np.array([-2., -3., -1., 0.]))
    theta, index = selected_score(result, np.array([0.]), 3.5)
    np.testing.assert_array_equal(theta, [2.])
    assert index == 1  # Not the highest likelihood, invalid third row, or overtime fourth row.
    theta, index = selected_score(result, np.array([0.]), .5)
    np.testing.assert_array_equal(theta, [0.])
    assert index == -1


def test_score_arm_keeps_archived_optimization_settings():
    cold, warm = [score_settings(regime, 1000, 850.) for regime in ('cold', 'warm')]
    assert cold['learning_rate'] == .1 and warm['learning_rate'] == .0005
    assert cold['project_to_bounds'] and not warm['project_to_bounds']
    for settings in (cold, warm):
        assert settings['burnin'] == 30 and settings['gain_exponent'] == .9
        assert settings['proposal'] == 'guided' and settings['endpoint_relative_floor'] == 1e-12
        assert settings['likelihood_guard_particles'] == 100
        assert settings['maximum_elapsed_seconds'] == 850.


def test_paired_mc_error_accounts_for_shared_draws():
    a = np.array([-10., -11., -9., -12.])
    # Multiplying every likelihood draw by a constant changes the log ratio
    # deterministically. Treating the two evaluations as independent would
    # incorrectly give a positive MC error for this difference.
    np.testing.assert_allclose(paired_mcse(a, a+7.), 0., atol=1e-14)
    assert paired_mcse(a, a[::-1]) > 0.
