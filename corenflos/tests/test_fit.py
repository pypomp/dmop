from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from corenflos.dpf import DPFResult
from corenflos.fit import fit_dpf
from ditlevsen.model import default_unconstrained_parameters


def test_fit_records_initial_and_every_completed_update() -> None:
    def evaluator(parameters, key):
        del key
        result = DPFResult(
            loglik=jnp.sum(parameters),
            increments=jnp.asarray([jnp.sum(parameters)]),
            ess=jnp.asarray([10.0]),
            resampled=jnp.asarray([False]),
            invalid_fraction=jnp.asarray([0.0]),
        )
        return result, jnp.ones_like(parameters)

    result = fit_dpf(
        default_unconstrained_parameters(),
        evaluator,
        seed=1,
        optimizer="adam",
        learning_rate=0.001,
        maximum_updates=2,
        maximum_elapsed_seconds=60.0,
    )
    assert result.completed_updates == 2
    assert result.parameter_trace.shape == (3, 23)
    assert result.pseudo_loglik_trace.shape == (3,)
    assert np.all(np.diff(result.elapsed_trace) >= 0.0)
    assert result.pseudo_loglik_trace[-1] >= result.pseudo_loglik_trace[0]


def test_fit_rolls_back_an_invalid_candidate() -> None:
    calls = 0

    def evaluator(parameters, key):
        nonlocal calls
        del key
        invalid = 0.9 if calls == 1 else 0.0
        calls += 1
        result = DPFResult(
            loglik=jnp.sum(parameters),
            increments=jnp.asarray([jnp.sum(parameters)]),
            ess=jnp.asarray([10.0]),
            resampled=jnp.asarray([False]),
            invalid_fraction=jnp.asarray([invalid]),
        )
        return result, jnp.ones_like(parameters)

    initial = default_unconstrained_parameters()
    result = fit_dpf(
        initial,
        evaluator,
        seed=1,
        optimizer="adam",
        learning_rate=0.001,
        maximum_acceptable_invalid_fraction=0.5,
        maximum_updates=2,
        maximum_elapsed_seconds=60.0,
    )
    assert result.completed_updates == 2
    assert result.restart_count_trace.tolist() == [0, 0, 1]
    np.testing.assert_allclose(result.parameter_trace[0], result.parameter_trace[2])


def test_fit_rejects_an_all_invalid_initial_filter() -> None:
    def evaluator(parameters, key):
        del key
        result = DPFResult(
            loglik=jnp.sum(parameters),
            increments=jnp.asarray([jnp.sum(parameters)]),
            ess=jnp.asarray([1.0]),
            resampled=jnp.asarray([False]),
            invalid_fraction=jnp.asarray([1.0]),
        )
        return result, jnp.ones_like(parameters)

    result = fit_dpf(
        default_unconstrained_parameters(),
        evaluator,
        seed=1,
        maximum_updates=2,
        maximum_elapsed_seconds=60.0,
    )
    assert result.completed_updates == 0
    assert result.termination_reason == "no-valid-filter-evaluation"
