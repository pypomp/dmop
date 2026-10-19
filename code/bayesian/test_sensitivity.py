"""Reject comparisons confounding a prior or particle sensitivity analysis."""
from copy import deepcopy

import pytest

from sensitivity import validate_pair


def report():
    return dict(method='pmrw', model_sha256='model',
                target=dict(particles=10000, prior_scale=1., prior_center=[0., 1.],
                            prior_sd=[2., 3.], names=['a', 'b'], pypomp_revision='revision'))


def test_prior_only():
    left = report(); right = deepcopy(left)
    right['target'].update(prior_scale=2., prior_sd=[4., 6.])
    validate_pair(left, right, 'prior')
    right['target']['particles'] = 5000
    with pytest.raises(ValueError):
        validate_pair(left, right, 'prior')


def test_particle_only():
    left = report(); right = deepcopy(left)
    right['target']['particles'] = 5000
    validate_pair(left, right, 'particles')
    right['target']['prior_center'] = [0., 2.]
    with pytest.raises(ValueError):
        validate_pair(left, right, 'particles')


def test_reject_wrong_scaling_and_identical_reports():
    left = report(); right = deepcopy(left)
    for kind in ['prior', 'particles']:
        with pytest.raises(ValueError):
            validate_pair(left, right, kind)
    right['target'].update(prior_scale=2., prior_sd=[4., 5.])
    with pytest.raises(AssertionError):
        validate_pair(left, right, 'prior')
    right['model_sha256'] = 'different'
    with pytest.raises(ValueError):
        validate_pair(left, right, 'prior')
