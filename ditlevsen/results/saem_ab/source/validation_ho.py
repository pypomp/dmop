"""Paper-faithful SMC validation on the harmonic oscillator.

Ditlevsen and Samson (2019), Sections 2.5.1, 5.1, and 6.1, use the
hypoelliptic oscillator

    dV = U dt,
    dU = (-D V - gamma U) dt + sigma dB,

with exactly observed ``V`` and hidden ``U``.  This module implements their
conditional Gaussian proposal and multinomial-resampling particle filter.
The linear-Gaussian model also admits an analytic filter, which gives us a
useful implementation oracle before applying any extension to Dacca.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class HOParameters:
    damping: float = 0.5
    stiffness: float = 4.0
    diffusion: float = 0.5


@dataclass(frozen=True)
class HOFilterResult:
    filtered_mean: np.ndarray
    analytic_mean: np.ndarray
    ess: np.ndarray
    loglik: float
    sampled_path: np.ndarray
    unique_initial_ancestors: int


@dataclass(frozen=True)
class HOScoreFitResult:
    parameters: HOParameters
    parameter_trace: np.ndarray
    marginal_loglik_trace: np.ndarray
    complete_loglik_trace: np.ndarray
    score_norm_trace: np.ndarray


def _system_matrix(parameters: HOParameters) -> np.ndarray:
    return np.asarray(
        [[0.0, 1.0], [-parameters.stiffness, -parameters.damping]], dtype=float
    )


def exact_transition(
    parameters: HOParameters, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact OU transition matrix and covariance from paper equation (16)."""

    matrix = _system_matrix(parameters)
    noise = np.asarray([[0.0], [parameters.diffusion]], dtype=float)
    transition = expm(matrix * dt)
    stationary = solve_continuous_lyapunov(matrix, -(noise @ noise.T))
    covariance = stationary - transition @ stationary @ transition.T
    covariance = 0.5 * (covariance + covariance.T)
    return transition, covariance


def strong_15_transition(
    parameters: HOParameters, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Order-1.5 Gaussian transition in paper equations (19) and (21)."""

    matrix = _system_matrix(parameters)
    transition = np.eye(2) + dt * matrix + 0.5 * dt**2 * matrix @ matrix
    damping = parameters.damping
    stiffness = parameters.stiffness
    diffusion_variance = parameters.diffusion**2
    # This is Sigma_HO in equation (34), i.e. the covariance of the
    # order-1.5 *scheme*.  It intentionally differs in its O(dt^3) terms from
    # the Taylor expansion of the exact covariance in equation (21).
    covariance = diffusion_variance * np.asarray(
        [
            [dt**3 / 3.0, dt**2 / 2.0 - damping * dt**3 / 3.0],
            [
                dt**2 / 2.0 - damping * dt**3 / 3.0,
                dt - damping * dt**2 + damping**2 * dt**3 / 3.0,
            ],
        ],
        dtype=float,
    )
    covariance = 0.5 * (covariance + covariance.T)
    return transition, covariance


def simulate_ho(
    *,
    parameters: HOParameters = HOParameters(),
    dt: float = 0.02,
    points: int = 1000,
    seed: int = 419,
) -> np.ndarray:
    """Simulate from the exact stationary oscillator, as in Section 6.1."""

    if points < 2:
        raise ValueError("points must be at least two")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    rng = np.random.default_rng(seed)
    transition, covariance = exact_transition(parameters, dt)
    matrix = _system_matrix(parameters)
    noise = np.asarray([[0.0], [parameters.diffusion]], dtype=float)
    stationary = solve_continuous_lyapunov(matrix, -(noise @ noise.T))
    trajectory = np.empty((points, 2), dtype=float)
    trajectory[0] = rng.multivariate_normal(np.zeros(2), stationary)
    for index in range(1, points):
        trajectory[index] = rng.multivariate_normal(
            transition @ trajectory[index - 1], covariance
        )
    return trajectory


def _normal_logpdf(value: np.ndarray, mean: np.ndarray, variance: float) -> np.ndarray:
    return -0.5 * (
        np.log(2.0 * np.pi * variance) + (value - mean) ** 2 / variance
    )


def analytic_conditional_filter(
    observed: np.ndarray,
    *,
    parameters: HOParameters = HOParameters(),
    dt: float = 0.02,
) -> np.ndarray:
    """Analytic filter under the same order-1.5 Gaussian approximation."""

    transition, covariance = strong_15_transition(parameters, dt)
    hidden_variance = parameters.diffusion**2 / (2.0 * parameters.damping)
    mean = np.asarray([observed[0], 0.0], dtype=float)
    state_covariance = np.asarray([[0.0, 0.0], [0.0, hidden_variance]])
    result = np.empty(len(observed), dtype=float)
    result[0] = mean[1]
    for index in range(1, len(observed)):
        predicted_mean = transition @ mean
        predicted_covariance = (
            transition @ state_covariance @ transition.T + covariance
        )
        innovation_variance = predicted_covariance[0, 0]
        gain = predicted_covariance[:, 0] / innovation_variance
        mean = predicted_mean + gain * (observed[index] - predicted_mean[0])
        state_covariance = predicted_covariance - np.outer(
            predicted_covariance[:, 0], predicted_covariance[0, :]
        ) / innovation_variance
        mean[0] = observed[index]
        state_covariance[0, :] = 0.0
        state_covariance[:, 0] = 0.0
        result[index] = mean[1]
    return result


def analytic_observed_loglik(
    observed: np.ndarray,
    *,
    parameters: HOParameters = HOParameters(),
    dt: float = 0.02,
) -> float:
    """Observed-coordinate likelihood under the order-1.5 Gaussian model."""

    observed = np.asarray(observed, dtype=float)
    transition, covariance = strong_15_transition(parameters, dt)
    hidden_variance = parameters.diffusion**2 / (2.0 * parameters.damping)
    mean = np.asarray([observed[0], 0.0], dtype=float)
    state_covariance = np.asarray([[0.0, 0.0], [0.0, hidden_variance]])
    total = 0.0
    for index in range(1, len(observed)):
        predicted_mean = transition @ mean
        predicted_covariance = (
            transition @ state_covariance @ transition.T + covariance
        )
        innovation_variance = predicted_covariance[0, 0]
        total += float(
            _normal_logpdf(
                np.asarray(observed[index]),
                np.asarray(predicted_mean[0]),
                innovation_variance,
            )
        )
        gain = predicted_covariance[:, 0] / innovation_variance
        mean = predicted_mean + gain * (observed[index] - predicted_mean[0])
        state_covariance = predicted_covariance - np.outer(
            predicted_covariance[:, 0], predicted_covariance[0, :]
        ) / innovation_variance
        mean[0] = observed[index]
        state_covariance[0, :] = 0.0
        state_covariance[:, 0] = 0.0
    return total


@jax.jit
def _complete_loglik(
    log_parameters: jax.Array,
    observed: jax.Array,
    hidden: jax.Array,
    dt: jax.Array,
) -> jax.Array:
    """Complete order-1.5 oscillator pseudo-log-likelihood."""

    stiffness, damping, diffusion = jnp.exp(log_parameters)
    matrix = jnp.asarray([[0.0, 1.0], [-stiffness, -damping]])
    transition = jnp.eye(2) + dt * matrix + 0.5 * dt**2 * matrix @ matrix
    covariance = diffusion**2 * jnp.asarray(
        [
            [dt**3 / 3.0, dt**2 / 2.0 - damping * dt**3 / 3.0],
            [
                dt**2 / 2.0 - damping * dt**3 / 3.0,
                dt - damping * dt**2 + damping**2 * dt**3 / 3.0,
            ],
        ]
    )
    cholesky = jnp.linalg.cholesky(covariance)
    states = jnp.column_stack((observed, hidden))
    residuals = states[1:] - states[:-1] @ transition.T
    standardized = jax.scipy.linalg.solve_triangular(
        cholesky, residuals.T, lower=True
    ).T
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    transitions = -0.5 * jnp.sum(
        2.0 * jnp.log(2.0 * jnp.pi)
        + logdet
        + jnp.sum(standardized**2, axis=1)
    )
    initial_variance = diffusion**2 / (2.0 * damping)
    initial = -0.5 * (
        jnp.log(2.0 * jnp.pi * initial_variance)
        + hidden[0] ** 2 / initial_variance
    )
    return initial + transitions


def fit_ho_smc_score(
    observed: np.ndarray,
    *,
    start: HOParameters = HOParameters(damping=1.0, stiffness=3.0, diffusion=0.8),
    dt: float = 0.02,
    particles: int = 100,
    iterations: int = 80,
    learning_rate: float = 0.005,
    burnin: int = 30,
    gain_exponent: float = 0.9,
    seed: int = 631409,
) -> HOScoreFitResult:
    """Validate the Dacca stochastic-score construction on a known-truth model."""

    observed = np.asarray(observed, dtype=float)
    raw = jnp.log(
        jnp.asarray([start.stiffness, start.damping, start.diffusion], dtype=jnp.float64)
    )
    first_moment = jnp.zeros(3, dtype=jnp.float64)
    second_moment = jnp.zeros(3, dtype=jnp.float64)
    averaged_score = jnp.zeros(3, dtype=jnp.float64)
    value_and_score = jax.jit(jax.value_and_grad(_complete_loglik))

    parameter_trace = []
    marginal_trace = []
    complete_trace = []
    score_trace = []
    for iteration in range(iterations + 1):
        physical = np.exp(np.asarray(raw))
        current = HOParameters(
            stiffness=float(physical[0]),
            damping=float(physical[1]),
            diffusion=float(physical[2]),
        )
        filtered = conditional_particle_filter(
            observed,
            parameters=current,
            dt=dt,
            particles=particles,
            seed=seed + 104729 * iteration,
        )
        complete, score = value_and_score(
            raw,
            jnp.asarray(observed, dtype=jnp.float64),
            jnp.asarray(filtered.sampled_path, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
        )
        parameter_trace.append(physical)
        marginal_trace.append(
            analytic_observed_loglik(observed, parameters=current, dt=dt)
        )
        complete_trace.append(float(complete))
        score_trace.append(float(jnp.linalg.norm(score)))
        if iteration == iterations:
            break

        gain = (
            1.0
            if iteration < burnin
            else float(iteration - burnin + 1) ** (-gain_exponent)
        )
        averaged_score = (1.0 - gain) * averaged_score + gain * score
        averaged_score = averaged_score * jnp.minimum(
            1.0, 100.0 / jnp.maximum(jnp.linalg.norm(averaged_score), 1e-12)
        )
        first_moment = 0.9 * first_moment + 0.1 * averaged_score
        second_moment = 0.999 * second_moment + 0.001 * averaged_score**2
        first_unbiased = first_moment / (1.0 - 0.9 ** (iteration + 1))
        second_unbiased = second_moment / (1.0 - 0.999 ** (iteration + 1))
        raw = raw + learning_rate * first_unbiased / (
            jnp.sqrt(second_unbiased) + 1e-8
        )
        raw = jnp.clip(
            raw,
            jnp.log(jnp.asarray([0.1, 0.05, 0.05])),
            jnp.log(jnp.asarray([10.0, 5.0, 2.0])),
        )

    final = np.exp(np.asarray(raw))
    return HOScoreFitResult(
        parameters=HOParameters(
            stiffness=float(final[0]),
            damping=float(final[1]),
            diffusion=float(final[2]),
        ),
        parameter_trace=np.asarray(parameter_trace),
        marginal_loglik_trace=np.asarray(marginal_trace),
        complete_loglik_trace=np.asarray(complete_trace),
        score_norm_trace=np.asarray(score_trace),
    )


def conditional_particle_filter(
    observed: np.ndarray,
    *,
    parameters: HOParameters = HOParameters(),
    dt: float = 0.02,
    particles: int = 100,
    seed: int = 631409,
) -> HOFilterResult:
    """Algorithm 1 with the paper's conditional transition proposal.

    The proposal is ``p(U_i | V_i, V_{i-1}, U_{i-1})``.  Consequently the
    incremental importance weight is the marginal density
    ``p(V_i | V_{i-1}, U_{i-1})``.  Complete particle histories are retained
    so this also exercises the genealogy used to draw an imputed SAEM path.
    """

    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or len(observed) < 2:
        raise ValueError("observed must be a one-dimensional series")
    if particles < 2:
        raise ValueError("particles must be at least two")

    rng = np.random.default_rng(seed)
    transition, covariance = strong_15_transition(parameters, dt)
    q_vv = covariance[0, 0]
    conditional_coefficient = covariance[1, 0] / q_vv
    conditional_variance = covariance[1, 1] - covariance[1, 0] ** 2 / q_vv
    initial_hidden_sd = parameters.diffusion / np.sqrt(2.0 * parameters.damping)

    histories = np.empty((particles, len(observed)), dtype=float)
    histories[:, 0] = rng.normal(0.0, initial_hidden_sd, size=particles)
    weights = np.full(particles, 1.0 / particles)
    lineages = np.arange(particles)
    filtered_mean = np.empty(len(observed), dtype=float)
    filtered_mean[0] = np.sum(weights * histories[:, 0])
    ess = np.empty(len(observed) - 1, dtype=float)
    increments = np.empty(len(observed) - 1, dtype=float)

    for index in range(1, len(observed)):
        ancestors = rng.choice(particles, size=particles, replace=True, p=weights)
        histories = histories[ancestors].copy()
        lineages = lineages[ancestors]
        previous_hidden = histories[:, index - 1]
        previous_states = np.column_stack(
            (np.full(particles, observed[index - 1]), previous_hidden)
        )
        means = previous_states @ transition.T

        conditional_mean = means[:, 1] + conditional_coefficient * (
            observed[index] - means[:, 0]
        )
        histories[:, index] = rng.normal(
            conditional_mean, np.sqrt(conditional_variance)
        )

        log_weights = _normal_logpdf(observed[index], means[:, 0], q_vv)
        normalizer = logsumexp(log_weights)
        weights = np.exp(log_weights - normalizer)
        increments[index - 1] = normalizer - np.log(float(particles))
        ess[index - 1] = 1.0 / np.sum(weights**2)
        filtered_mean[index] = np.sum(weights * histories[:, index])

    selected = rng.choice(particles, p=weights)
    return HOFilterResult(
        filtered_mean=filtered_mean,
        analytic_mean=analytic_conditional_filter(
            observed, parameters=parameters, dt=dt
        ),
        ess=ess,
        loglik=float(np.sum(increments)),
        sampled_path=histories[selected].copy(),
        unique_initial_ancestors=int(len(np.unique(lineages))),
    )
