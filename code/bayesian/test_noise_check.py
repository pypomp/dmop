"""Likelihood means include zero estimates and use the original target scale."""
import numpy as np
import pytest
from noise_check import summarize


def test_mean_likelihood_and_refresh_probability():
    result=summarize(np.log([1.,2.,4.]),cached=np.log(4.))
    assert np.exp(result['log_mean_likelihood']) == pytest.approx(7/3)
    assert result['estimated_fixed_parameter_refresh_acceptance'] == pytest.approx(7/12)
    assert result['zero_likelihood_estimates'] == 0


def test_zero_estimates_are_not_discarded():
    result=summarize([np.log(2.),-np.inf],cached=np.log(2.))
    assert result['log_mean_likelihood'] == pytest.approx(0.)
    assert result['estimated_fixed_parameter_refresh_acceptance'] == .5
    assert result['loglik_variance'] is None
    assert result['likelihood_relative_variance'] == pytest.approx(2.)
    assert result['zero_likelihood_estimates'] == 1
    with pytest.raises(ValueError):
        summarize([1.,np.nan])
