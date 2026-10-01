from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np

from corenflos.dpf import (
    DPFConfig,
    DaccaArrays,
    _euler_step,
    _initial_particles,
    _measurement_log_density,
    dacca_dpf,
    dpf_value_and_grad,
)
from ditlevsen.data import load_dacca_data
from ditlevsen.model import (
    default_unconstrained_parameters,
    physical_parameter_dict,
)

jax.config.update("jax_enable_x64", True)


def test_short_dacca_filter_and_gradient_are_finite() -> None:
    data = DaccaArrays.from_data(load_dacca_data(20, max_observations=3))
    parameters = jnp.asarray(default_unconstrained_parameters())
    config = DPFConfig(particles=12, ess_threshold=0.5)
    result, gradient = dpf_value_and_grad(parameters, jax.random.key(123), data, config)

    assert result.increments.shape == (3,)
    assert result.ess.shape == (3,)
    assert np.isfinite(float(result.loglik))
    assert bool(jnp.all(jnp.isfinite(result.increments)))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert gradient.shape == parameters.shape


def test_filter_is_deterministic_for_a_fixed_key() -> None:
    data = DaccaArrays.from_data(load_dacca_data(5, max_observations=2))
    parameters = jnp.asarray(default_unconstrained_parameters())
    config = DPFConfig(particles=8)
    first = dacca_dpf(parameters, jax.random.key(9), data, config)
    second = dacca_dpf(parameters, jax.random.key(9), data, config)
    np.testing.assert_array_equal(np.asarray(first.loglik), np.asarray(second.loglik))
    np.testing.assert_array_equal(np.asarray(first.ess), np.asarray(second.ess))


def test_one_euler_step_matches_pypomp_dacca() -> None:
    module = importlib.import_module("pypomp.models.dacca")
    loaded = load_dacca_data(20, max_observations=1)
    parameters = jnp.asarray(default_unconstrained_parameters())
    covariates = jnp.asarray(loaded.step_covariates[0, 0])
    particles, invalid = _initial_particles(
        parameters, jnp.asarray(loaded.initial_covariates), 4
    )
    key = jax.random.key(741)
    normals = jax.random.normal(key, (4,), dtype=jnp.float64)
    actual, actual_invalid = _euler_step(
        particles, invalid, normals, covariates, parameters, 1.0 / 240.0
    )

    names = (
        "trend",
        "dpopdt",
        "pop",
        "seas1",
        "seas2",
        "seas3",
        "seas4",
        "seas5",
        "seas6",
    )
    pypomp_state = {
        "S": particles[:, 0],
        "I": particles[:, 1],
        "Y": jnp.zeros((4,), dtype=jnp.float64),
        "Mn": particles[:, 2],
        "R1": particles[:, 3],
        "R2": particles[:, 4],
        "R3": particles[:, 5],
        "count": jnp.zeros((4,), dtype=jnp.float64),
    }
    expected = module._rproc(
        pypomp_state,
        physical_parameter_dict(parameters),
        key,
        dict(zip(names, covariates, strict=True)),
        1891.0,
        1.0 / 240.0,
    )
    expected_particles = jnp.stack(
        (
            expected["S"],
            expected["I"],
            expected["Mn"],
            expected["R1"],
            expected["R2"],
            expected["R3"],
        ),
        axis=1,
    )
    np.testing.assert_allclose(actual, expected_particles, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_array_equal(actual_invalid, expected["count"] > 0)


def test_measurement_density_matches_pypomp_dacca() -> None:
    module = importlib.import_module("pypomp.models.dacca")
    parameters = jnp.asarray(default_unconstrained_parameters())
    state = jnp.asarray([[10.0, 2.0, 51.0, 3.0, 4.0, 5.0]])
    observation = jnp.asarray(49.0)
    actual = _measurement_log_density(
        observation, state, jnp.asarray([False]), parameters
    )[0]
    expected = module._dmeas(
        {"deaths": observation},
        {"Mn": state[0, 2], "count": jnp.asarray(0.0)},
        physical_parameter_dict(parameters),
        {},
        0.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_no_resampling_gradient_matches_finite_difference() -> None:
    data = DaccaArrays.from_data(load_dacca_data(20, max_observations=2))
    parameters = jnp.asarray(default_unconstrained_parameters())
    config = DPFConfig(particles=24, ess_threshold=0.0)
    key = jax.random.key(991)
    _, gradient = dpf_value_and_grad(parameters, key, data, config)
    step = 1.0e-5
    for index in (0, 5):
        direction = jnp.zeros_like(parameters).at[index].set(step)
        upper = dacca_dpf(parameters + direction, key, data, config).loglik
        lower = dacca_dpf(parameters - direction, key, data, config).loglik
        finite_difference = (upper - lower) / (2.0 * step)
        np.testing.assert_allclose(
            gradient[index], finite_difference, rtol=2.0e-5, atol=2.0e-5
        )
