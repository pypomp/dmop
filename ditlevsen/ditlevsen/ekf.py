"""Deterministic Gaussian quasi-likelihood benchmark for the Dacca model.

This is deliberately a favorable diagnostic for the Ditlevsen transition idea:
it maximizes the marginal likelihood of an extended-Kalman approximation instead
of adding Monte Carlo error from SAEM.  It is not presented as the literal
Ditlevsen--Samson SAEM algorithm; the distinction is central to the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .data import DaccaData
from .model import decode_parameters, initial_state, project_parameters
from .transition import ditlevsen_mean, gaussian_transition

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FitResult:
    unconstrained: np.ndarray
    parameter_trace: np.ndarray
    objective_trace: np.ndarray
    gradient_norm_trace: np.ndarray
    elapsed_trace: np.ndarray
    elapsed_seconds: float
    converged: bool


def _measurement_update(
    mean: jax.Array,
    covariance: jax.Array,
    observation: jax.Array,
    population: jax.Array,
    tau: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """EKF measurement update followed by the monthly Mn reset."""

    predicted_deaths = population * mean[-1]
    # This is the manuscript measurement model: sd(Y | M) = tau * M.
    measurement_sd = tau * jnp.maximum(jnp.abs(predicted_deaths), 1e-6)
    projected_variance = population**2 * covariance[-1, -1]
    innovation_variance = jnp.maximum(
        projected_variance + measurement_sd**2, 1e-12
    )
    innovation = observation - predicted_deaths
    covariance_with_observation = population * covariance[:, -1]
    gain = covariance_with_observation / innovation_variance
    updated_mean = mean + gain * innovation
    updated_covariance = covariance - jnp.outer(
        covariance_with_observation, covariance_with_observation
    ) / innovation_variance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)

    loglik = -0.5 * (
        jnp.log(2.0 * jnp.pi * innovation_variance)
        + innovation**2 / innovation_variance
    )

    # Pypomp resets the accumulated-death state immediately after dmeasure.
    updated_mean = updated_mean.at[-1].set(0.0)
    updated_covariance = updated_covariance.at[-1, :].set(0.0)
    updated_covariance = updated_covariance.at[:, -1].set(0.0)
    return updated_mean, updated_covariance, loglik


def ekf_loglik(
    unconstrained: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    *,
    order: int,
    relative_floor: float = 1e-10,
) -> jax.Array:
    """Ditlevsen-Gaussian extended-Kalman marginal log likelihood."""

    nstep = step_covariates.shape[1]
    dt = 1.0 / (12.0 * nstep)
    theta = decode_parameters(unconstrained)
    mean0 = initial_state(unconstrained)
    covariance0 = jnp.zeros((mean0.shape[0], mean0.shape[0]), dtype=jnp.float64)

    def observation_interval(carry, inputs):
        mean, covariance, total_loglik = carry
        interval_covariates, observation_population, observation = inputs

        def process_step(process_carry, covariates):
            process_mean, process_covariance = process_carry
            predicted_mean, process_noise = gaussian_transition(
                process_mean,
                unconstrained,
                covariates,
                dt,
                order=order,
                relative_floor=relative_floor,
            )
            transition_jacobian = jax.jacfwd(ditlevsen_mean, argnums=0)(
                process_mean, unconstrained, covariates, dt
            )
            predicted_covariance = (
                transition_jacobian
                @ process_covariance
                @ transition_jacobian.T
                + process_noise
            )
            predicted_covariance = 0.5 * (
                predicted_covariance + predicted_covariance.T
            )
            return (predicted_mean, predicted_covariance), None

        (predicted_mean, predicted_covariance), _ = jax.lax.scan(
            process_step, (mean, covariance), interval_covariates
        )
        updated_mean, updated_covariance, increment = _measurement_update(
            predicted_mean,
            predicted_covariance,
            observation,
            observation_population,
            theta["tau"],
        )

        # A smooth barrier prevents invalid Gaussian means from silently looking
        # competitive while retaining finite derivatives for optimization.
        negative = jax.nn.relu(-updated_mean[:5])
        excessive = jax.nn.relu(updated_mean[:5] - 1.2)
        barrier = 1e5 * jnp.sum(negative**2 + excessive**2)
        increment = increment - barrier
        return (
            updated_mean,
            updated_covariance,
            total_loglik + increment,
        ), increment

    (_, _, total), _ = jax.lax.scan(
        observation_interval,
        (mean0, covariance0, jnp.asarray(0.0, dtype=jnp.float64)),
        (step_covariates, observation_populations, observations),
    )
    return total


def filtered_means(
    unconstrained: jax.Array,
    data: DaccaData,
    *,
    order: int,
    relative_floor: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return month-end filtered means and log-likelihood increments."""

    covariate_array = jnp.asarray(data.step_covariates)
    observation_populations = jnp.asarray(data.observation_covariates[:, 2])
    observation_array = jnp.asarray(data.observations)
    nstep = data.nstep
    dt = 1.0 / (12.0 * nstep)
    theta = decode_parameters(unconstrained)
    mean0 = initial_state(unconstrained)
    covariance0 = jnp.zeros((6, 6), dtype=jnp.float64)

    def interval(carry, inputs):
        mean, covariance = carry
        interval_covariates, observation_population, observation = inputs

        def step(process_carry, covariates):
            process_mean, process_covariance = process_carry
            predicted_mean, process_noise = gaussian_transition(
                process_mean,
                unconstrained,
                covariates,
                dt,
                order=order,
                relative_floor=relative_floor,
            )
            transition_jacobian = jax.jacfwd(ditlevsen_mean, argnums=0)(
                process_mean, unconstrained, covariates, dt
            )
            predicted_covariance = (
                transition_jacobian
                @ process_covariance
                @ transition_jacobian.T
                + process_noise
            )
            return (predicted_mean, predicted_covariance), None

        (mean, covariance), _ = jax.lax.scan(
            step, (mean, covariance), interval_covariates
        )
        mean, covariance, increment = _measurement_update(
            mean,
            covariance,
            observation,
            observation_population,
            theta["tau"],
        )
        return (mean, covariance), (mean, increment)

    (_, _), (means, increments) = jax.lax.scan(
        interval,
        (mean0, covariance0),
        (covariate_array, observation_populations, observation_array),
    )
    return np.asarray(means), np.asarray(increments)


def fit_ekf(
    start: np.ndarray,
    data: DaccaData,
    *,
    order: int,
    iterations: int = 100,
    learning_rate: float = 0.01,
    relative_floor: float = 1e-10,
    gradient_clip: float = 25.0,
) -> FitResult:
    """Maximize the Gaussian quasi-likelihood with projected Adam."""

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    parameters = project_parameters(jnp.asarray(start, dtype=jnp.float64))
    first_moment = jnp.zeros_like(parameters)
    second_moment = jnp.zeros_like(parameters)
    covariates = jnp.asarray(data.step_covariates)
    observation_populations = jnp.asarray(data.observation_covariates[:, 2])
    observations = jnp.asarray(data.observations)

    def objective(value):
        return ekf_loglik(
            value,
            covariates,
            observation_populations,
            observations,
            order=order,
            relative_floor=relative_floor,
        )

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    parameter_trace: list[np.ndarray] = []
    objective_trace: list[float] = []
    gradient_norm_trace: list[float] = []
    elapsed_trace: list[float] = []
    started = perf_counter()

    for iteration in range(iterations + 1):
        parameter_trace.append(np.asarray(parameters))
        value, gradient = value_and_gradient(parameters)
        value_float = float(value)
        gradient_array = np.asarray(gradient)
        gradient_norm = float(np.linalg.norm(gradient_array))
        objective_trace.append(value_float)
        gradient_norm_trace.append(gradient_norm)
        elapsed_trace.append(perf_counter() - started)
        if iteration == iterations:
            break
        if not np.isfinite(value_float) or not np.all(np.isfinite(gradient_array)):
            break

        scale = jnp.minimum(1.0, gradient_clip / jnp.maximum(gradient_norm, 1e-12))
        gradient = gradient * scale
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        first_unbiased = first_moment / (1.0 - 0.9 ** (iteration + 1))
        second_unbiased = second_moment / (1.0 - 0.999 ** (iteration + 1))
        parameters = parameters + learning_rate * first_unbiased / (
            jnp.sqrt(second_unbiased) + 1e-8
        )
        parameters = project_parameters(parameters)

    elapsed = perf_counter() - started
    finite = np.all(np.isfinite(objective_trace)) and np.all(
        np.isfinite(gradient_norm_trace)
    )
    converged = finite and len(objective_trace) == iterations + 1
    return FitResult(
        unconstrained=np.asarray(parameters),
        parameter_trace=np.asarray(parameter_trace),
        objective_trace=np.asarray(objective_trace),
        gradient_norm_trace=np.asarray(gradient_norm_trace),
        elapsed_trace=np.asarray(elapsed_trace),
        elapsed_seconds=elapsed,
        converged=converged,
    )
