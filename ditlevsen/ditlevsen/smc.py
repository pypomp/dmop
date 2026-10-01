"""Particle methods for the Dacca transition-density extension.

This module is intentionally separate from the earlier EKF diagnostic.  It
uses particles during inference.  Dacca is not in Ditlevsen--Samson's exact
observation/model class, so the bootstrap filter below is a generalization to
noisy monthly observations of a fully latent state, not their conditional
proposal for an exactly observed smooth coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .data import DaccaData
from .model import (
    decode_parameters,
    initial_state,
    normalize_parameters,
    project_parameters,
)
from .transition import gaussian_transition

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SMCFilterResult:
    loglik: float
    loglik_increments: np.ndarray
    ess: np.ndarray
    invalid_fraction: np.ndarray


@dataclass(frozen=True)
class SMCPathResult:
    """One smoothing-path draw and diagnostics from path-space SMC."""

    path: np.ndarray
    loglik: float
    loglik_increments: np.ndarray
    ess: np.ndarray
    invalid_fraction: np.ndarray
    unique_initial_ancestors: int
    valid_path: bool
    backward_fallbacks: int = 0


@dataclass(frozen=True)
class SMCScoreFitResult:
    """Trace from Fisher-identity stochastic-approximation score ascent."""

    unconstrained: np.ndarray
    parameter_trace: np.ndarray
    marginal_loglik_trace: np.ndarray
    complete_loglik_trace: np.ndarray
    score_norm_trace: np.ndarray
    median_ess_trace: np.ndarray
    minimum_ess_trace: np.ndarray
    maximum_invalid_fraction_trace: np.ndarray
    unique_initial_ancestors_trace: np.ndarray
    backward_fallback_trace: np.ndarray
    bridge_update_fraction_trace: np.ndarray
    bridge_median_ess_trace: np.ndarray
    accepted_step_size_trace: np.ndarray
    backtrack_count_trace: np.ndarray
    elapsed_trace: np.ndarray
    elapsed_seconds: float
    completed_updates: int
    termination_reason: str


def _normal_logpdf(value: jax.Array, mean: jax.Array, scale: jax.Array) -> jax.Array:
    return -0.5 * (
        jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.log(scale)
        + ((value - mean) / scale) ** 2
    )


@partial(jax.jit, static_argnames=("particles", "order"))
def _bootstrap_filter_kernel(
    unconstrained: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    key: jax.Array,
    *,
    particles: int,
    order: int,
    relative_floor: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Jitted bootstrap particle filter over monthly observation blocks."""

    nstep = step_covariates.shape[1]
    dt = 1.0 / (12.0 * nstep)
    theta = decode_parameters(unconstrained)
    state0 = initial_state(unconstrained)
    states0 = jnp.repeat(state0[None, :], particles, axis=0)
    valid0 = jnp.ones((particles,), dtype=bool)
    log_particles = jnp.log(jnp.asarray(float(particles), dtype=jnp.float64))
    log_tolerance = jnp.log(jnp.asarray(1e-18, dtype=jnp.float64))

    def observation_interval(carry, inputs):
        states, valid, random_key = carry
        covariates, observation_population, observation = inputs

        def process_step(process_carry, forcing):
            process_states, process_valid, process_key = process_carry
            process_key, noise_key = jax.random.split(process_key)

            def moments(state):
                return gaussian_transition(
                    state,
                    unconstrained,
                    forcing,
                    dt,
                    order=order,
                    relative_floor=relative_floor,
                )

            means, covariances = jax.vmap(moments)(process_states)
            cholesky = jnp.linalg.cholesky(covariances)
            innovations = jax.random.normal(
                noise_key, (particles, state0.shape[0]), dtype=jnp.float64
            )
            proposed = means + jnp.einsum("nij,nj->ni", cholesky, innovations)
            finite = jnp.all(jnp.isfinite(proposed), axis=1)
            nonnegative = jnp.all(proposed >= 0.0, axis=1)
            process_valid = process_valid & finite & nonnegative

            # Invalid proposals retain zero likelihood.  Safe replacements
            # prevent NaNs from contaminating later vectorized drift calls.
            safe = jnp.nan_to_num(proposed, nan=0.0, posinf=1.0, neginf=0.0)
            safe = jnp.maximum(safe, 0.0)
            return (safe, process_valid, process_key), None

        (states, valid, random_key), _ = jax.lax.scan(
            process_step, (states, valid, random_key), covariates
        )
        deaths = observation_population * states[:, 5]
        measurement_scale = theta["tau"] * deaths + 1e-18
        finite_scale = jnp.isfinite(measurement_scale) & (measurement_scale > 0.0)
        gaussian = _normal_logpdf(
            observation,
            deaths,
            jnp.where(finite_scale, measurement_scale, 1.0),
        )
        raw_log_weights = jnp.where(
            valid & finite_scale,
            jnp.logaddexp(gaussian, log_tolerance),
            -jnp.inf,
        )
        survived = jnp.any(jnp.isfinite(raw_log_weights))
        finite_normalizer = jax.scipy.special.logsumexp(raw_log_weights)
        normalizer = jnp.where(survived, finite_normalizer, -jnp.inf)
        normalized_log_weights = jnp.where(
            survived,
            raw_log_weights - finite_normalizer,
            -log_particles,
        )
        weights = jnp.exp(normalized_log_weights)
        ess = 1.0 / jnp.sum(weights**2)
        increment = normalizer - log_particles
        invalid_fraction = 1.0 - jnp.mean(valid.astype(jnp.float64))

        random_key, resample_key = jax.random.split(random_key)
        ancestors = jax.random.categorical(
            resample_key, normalized_log_weights, shape=(particles,)
        )
        states = states[ancestors]
        valid = valid[ancestors]
        states = states.at[:, 5].set(0.0)
        return (states, valid, random_key), (increment, ess, invalid_fraction)

    (_, _, _), diagnostics = jax.lax.scan(
        observation_interval,
        (states0, valid0, key),
        (step_covariates, observation_populations, observations),
    )
    return diagnostics


def bootstrap_filter(
    unconstrained: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 100,
    seed: int = 631409,
    order: int = 2,
    relative_floor: float = 1e-8,
) -> SMCFilterResult:
    """Run an SMC likelihood smoke test for the DS transition extension."""

    if particles < 2:
        raise ValueError("particles must be at least two")
    if order < 2:
        raise ValueError("order must be at least two")
    increments, ess, invalid_fraction = _bootstrap_filter_kernel(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
        jnp.asarray(data.observations, dtype=jnp.float64),
        jax.random.key(seed),
        particles=particles,
        order=order,
        relative_floor=relative_floor,
    )
    increments = np.asarray(increments)
    return SMCFilterResult(
        loglik=float(np.sum(increments)),
        loglik_increments=increments,
        ess=np.asarray(ess),
        invalid_fraction=np.asarray(invalid_fraction),
    )


@partial(jax.jit, static_argnames=("particles", "order"))
def _path_filter_kernel(
    unconstrained: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    key: jax.Array,
    *,
    particles: int,
    order: int,
    relative_floor: float,
):
    """Path-space bootstrap SMC, retaining segments and monthly ancestry."""

    nstep = step_covariates.shape[1]
    dt = 1.0 / (12.0 * nstep)
    theta = decode_parameters(unconstrained)
    state0 = initial_state(unconstrained)
    states0 = jnp.repeat(state0[None, :], particles, axis=0)
    valid0 = jnp.ones((particles,), dtype=bool)
    uniform_log_weights = jnp.full(
        (particles,), -jnp.log(jnp.asarray(float(particles), dtype=jnp.float64))
    )
    log_particles = jnp.log(jnp.asarray(float(particles), dtype=jnp.float64))
    log_tolerance = jnp.log(jnp.asarray(1e-18, dtype=jnp.float64))
    identity_ancestors = jnp.arange(particles)

    def observation_interval(carry, inputs):
        states, valid, normalized_log_weights, random_key = carry
        month, covariates, observation_population, observation = inputs
        random_key, resample_key = jax.random.split(random_key)
        sampled_ancestors = jax.random.categorical(
            resample_key, normalized_log_weights, shape=(particles,)
        )
        ancestors = jax.lax.cond(
            month == 0,
            lambda _: identity_ancestors,
            lambda _: sampled_ancestors,
            operand=None,
        )
        states = states[ancestors]
        valid = valid[ancestors]
        states = states.at[:, 5].set(0.0)

        def process_step(process_carry, forcing):
            process_states, process_valid, process_key = process_carry
            process_key, noise_key = jax.random.split(process_key)

            def moments(state):
                return gaussian_transition(
                    state,
                    unconstrained,
                    forcing,
                    dt,
                    order=order,
                    relative_floor=relative_floor,
                )

            means, covariances = jax.vmap(moments)(process_states)
            cholesky = jnp.linalg.cholesky(covariances)
            innovations = jax.random.normal(
                noise_key, (particles, state0.shape[0]), dtype=jnp.float64
            )
            proposed = means + jnp.einsum("nij,nj->ni", cholesky, innovations)
            finite = jnp.all(jnp.isfinite(proposed), axis=1)
            nonnegative = jnp.all(proposed >= 0.0, axis=1)
            process_valid = process_valid & finite & nonnegative
            safe = jnp.nan_to_num(proposed, nan=0.0, posinf=1.0, neginf=0.0)
            safe = jnp.maximum(safe, 0.0)
            return (safe, process_valid, process_key), safe

        (states, valid, random_key), segment = jax.lax.scan(
            process_step, (states, valid, random_key), covariates
        )
        deaths = observation_population * states[:, 5]
        measurement_scale = theta["tau"] * deaths + 1e-18
        finite_scale = jnp.isfinite(measurement_scale) & (measurement_scale > 0.0)
        gaussian = _normal_logpdf(
            observation,
            deaths,
            jnp.where(finite_scale, measurement_scale, 1.0),
        )
        raw_log_weights = jnp.where(
            valid & finite_scale,
            jnp.logaddexp(gaussian, log_tolerance),
            -jnp.inf,
        )
        survived = jnp.any(jnp.isfinite(raw_log_weights))
        finite_normalizer = jax.scipy.special.logsumexp(raw_log_weights)
        normalizer = jnp.where(survived, finite_normalizer, -jnp.inf)
        normalized_log_weights = jnp.where(
            survived,
            raw_log_weights - finite_normalizer,
            -log_particles,
        )
        weights = jnp.exp(normalized_log_weights)
        increment = normalizer - log_particles
        ess = 1.0 / jnp.sum(weights**2)
        invalid_fraction = 1.0 - jnp.mean(valid.astype(jnp.float64))
        return (
            states,
            valid,
            normalized_log_weights,
            random_key,
        ), (
            segment,
            ancestors,
            increment,
            ess,
            invalid_fraction,
        )

    months = jnp.arange(step_covariates.shape[0])
    (final_states, final_valid, final_log_weights, final_key), outputs = jax.lax.scan(
        observation_interval,
        (states0, valid0, uniform_log_weights, key),
        (months, step_covariates, observation_populations, observations),
    )
    del final_states
    final_key, selection_key = jax.random.split(final_key)
    selected = jax.random.categorical(selection_key, final_log_weights)
    segments, ancestors, increments, ess, invalid_fraction = outputs
    return (
        segments,
        ancestors,
        selected,
        final_valid[selected],
        increments,
        ess,
        invalid_fraction,
    )


def sample_smoothing_path(
    unconstrained: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 100,
    seed: int = 631409,
    order: int = 2,
    relative_floor: float = 1e-8,
) -> SMCPathResult:
    """Draw one complete latent path with the bootstrap particle smoother.

    This is the naive forward-genealogy smoother used by DS Algorithm 1,
    generalized to Dacca's noisy indirect observations.  It is intentionally
    kept as the baseline against which bridge rejuvenation can be assessed.
    """

    if particles < 2:
        raise ValueError("particles must be at least two")
    outputs = _path_filter_kernel(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
        jnp.asarray(data.observations, dtype=jnp.float64),
        jax.random.key(seed),
        particles=particles,
        order=order,
        relative_floor=relative_floor,
    )
    (
        segment_array,
        ancestor_array,
        selected,
        selected_valid,
        increments,
        ess,
        invalid_fraction,
    ) = (np.asarray(value) for value in outputs)

    particle_index = int(selected)
    selected_segments: list[np.ndarray] = []
    for month in range(segment_array.shape[0] - 1, -1, -1):
        selected_segments.append(segment_array[month, :, particle_index, :])
        particle_index = int(ancestor_array[month, particle_index])
    selected_segments.reverse()
    path = np.concatenate(
        (
            np.asarray(initial_state(jnp.asarray(unconstrained)))[None, :],
            np.concatenate(selected_segments, axis=0),
        ),
        axis=0,
    )

    lineages = np.arange(particles)
    for month in range(1, ancestor_array.shape[0]):
        lineages = lineages[ancestor_array[month]]
    return SMCPathResult(
        path=path,
        loglik=float(np.sum(increments)),
        loglik_increments=np.asarray(increments),
        ess=np.asarray(ess),
        invalid_fraction=np.asarray(invalid_fraction),
        unique_initial_ancestors=int(len(np.unique(lineages))),
        valid_path=bool(selected_valid),
    )


@partial(jax.jit, static_argnames=("order",))
def _complete_path_loglik(
    unconstrained: jax.Array,
    path_targets: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    *,
    order: int,
    relative_floor: float,
) -> jax.Array:
    """Gaussian complete pseudo-log-likelihood for a fixed imputed path."""

    nstep = step_covariates.shape[1]
    dt = 1.0 / (12.0 * nstep)
    theta = decode_parameters(unconstrained)
    initial = initial_state(unconstrained)
    targets = path_targets.reshape(step_covariates.shape[0], nstep, 6)
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64))
    log_tolerance = jnp.log(jnp.asarray(1e-18, dtype=jnp.float64))

    def interval(carry, inputs):
        previous, total = carry
        covariates, month_targets, population, observation = inputs
        previous = previous.at[5].set(0.0)

        def transition_step(step_carry, step_inputs):
            step_previous, step_total = step_carry
            forcing, target = step_inputs
            mean, covariance = gaussian_transition(
                step_previous,
                unconstrained,
                forcing,
                dt,
                order=order,
                relative_floor=relative_floor,
            )
            cholesky = jnp.linalg.cholesky(covariance)
            standardized = jax.scipy.linalg.solve_triangular(
                cholesky, target - mean, lower=True
            )
            contribution = -0.5 * (
                6.0 * log_two_pi
                + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
                + jnp.dot(standardized, standardized)
            )
            return (target, step_total + contribution), None

        (previous, total), _ = jax.lax.scan(
            transition_step, (previous, total), (covariates, month_targets)
        )
        deaths = population * previous[5]
        measurement_scale = theta["tau"] * deaths + 1e-18
        measurement_loglik = _normal_logpdf(
            observation, deaths, measurement_scale
        )
        total = total + jnp.logaddexp(measurement_loglik, log_tolerance)
        return (previous, total), None

    (_, total), _ = jax.lax.scan(
        interval,
        (initial, jnp.asarray(0.0, dtype=jnp.float64)),
        (step_covariates, targets, observation_populations, observations),
    )
    return total


def complete_path_loglik(
    unconstrained: np.ndarray,
    path: np.ndarray,
    data: DaccaData,
    *,
    order: int = 2,
    relative_floor: float = 1e-8,
) -> float:
    """Public wrapper for the fixed-path pseudo-log-likelihood."""

    expected = data.observations.size * data.nstep + 1
    if np.asarray(path).shape != (expected, 6):
        raise ValueError(f"path must have shape {(expected, 6)}")
    return float(
        _complete_path_loglik(
            jnp.asarray(unconstrained, dtype=jnp.float64),
            jnp.asarray(path[1:], dtype=jnp.float64),
            jnp.asarray(data.step_covariates, dtype=jnp.float64),
            jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
            jnp.asarray(data.observations, dtype=jnp.float64),
            order=order,
            relative_floor=relative_floor,
        )
    )


def fit_pseudo_score(
    start: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 100,
    iterations: int = 10,
    learning_rate: float = 0.001,
    burnin: int = 30,
    gain_exponent: float = 0.9,
    seed: int = 631409,
    order: int = 2,
    relative_floor: float = 1e-8,
    gradient_clip: float = 100.0,
    bridge_particles: int = 0,
    maximum_backtracks: int = 6,
    maximum_consecutive_rejections: int = 10,
    maximum_path_retries: int = 3,
    backtrack_factor: float = 0.5,
    maximum_acceptable_invalid_fraction: float = 0.5,
    maximum_elapsed_seconds: float | None = None,
    project_to_bounds: bool = True,
) -> SMCScoreFitResult:
    """Stochastic pseudo-score ascent driven by an SMC path.

    SMC supplies one approximate smoothing-path draw.  Holding that draw fixed
    while differentiating the complete pseudo-log-likelihood gives a Monte
    Carlo estimate of the observed-data score of the *regularized Gaussian
    pseudo-model* by Fisher's identity.  It is not the score of the original
    Dacca SDE or Euler process, whose transition density is unavailable.  The
    code does not differentiate through sampling or resampling.

    DS instead average parameter-independent sufficient statistics and perform
    a closed-form SAEM M-step for their curved exponential-family examples.
    Dacca has not been shown to possess that structure, so this routine has no
    SAEM M-step: it averages score estimates with the DS gain schedule and
    takes a projected Adam step.
    """

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if maximum_backtracks < 0:
        raise ValueError("maximum_backtracks must be nonnegative")
    if maximum_consecutive_rejections < 1:
        raise ValueError("maximum_consecutive_rejections must be positive")
    if maximum_path_retries < 0:
        raise ValueError("maximum_path_retries must be nonnegative")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must be strictly between zero and one")
    if not 0.0 <= maximum_acceptable_invalid_fraction < 1.0:
        raise ValueError(
            "maximum_acceptable_invalid_fraction must be in [0, 1)"
        )
    if maximum_elapsed_seconds is not None and maximum_elapsed_seconds <= 0.0:
        raise ValueError("maximum_elapsed_seconds must be positive")
    constrain = project_parameters if project_to_bounds else normalize_parameters
    parameters = constrain(jnp.asarray(start, dtype=jnp.float64))
    first_moment = jnp.zeros_like(parameters)
    second_moment = jnp.zeros_like(parameters)
    averaged_score = jnp.zeros_like(parameters)
    covariates = jnp.asarray(data.step_covariates, dtype=jnp.float64)
    populations = jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64)
    observations = jnp.asarray(data.observations, dtype=jnp.float64)
    value_and_score = jax.jit(
        jax.value_and_grad(
            lambda candidate, targets: _complete_path_loglik(
                candidate,
                targets,
                covariates,
                populations,
                observations,
                order=order,
                relative_floor=relative_floor,
            )
        )
    )

    parameter_trace: list[np.ndarray] = []
    marginal_trace: list[float] = []
    complete_trace: list[float] = []
    score_norm_trace: list[float] = []
    median_ess_trace: list[float] = []
    minimum_ess_trace: list[float] = []
    invalid_trace: list[float] = []
    ancestor_trace: list[int] = []
    backward_fallback_trace: list[int] = []
    bridge_update_trace: list[float] = []
    bridge_ess_trace: list[float] = []
    accepted_step_trace: list[float] = []
    backtrack_trace: list[int] = []
    elapsed_trace: list[float] = []
    started = perf_counter()
    completed_updates = 0
    termination_reason = "completed"
    pending_path_result: SMCPathResult | None = None
    working_learning_rate = learning_rate
    consecutive_rejections = 0

    for iteration in range(iterations + 1):
        parameter_trace.append(np.asarray(parameters))
        if pending_path_result is None:
            for retry in range(maximum_path_retries + 1):
                path_result = sample_smoothing_path(
                    np.asarray(parameters),
                    data,
                    particles=particles,
                    seed=seed + 104729 * iteration + 15485863 * retry,
                    order=order,
                    relative_floor=relative_floor,
                )
                if path_result.valid_path and np.isfinite(path_result.loglik):
                    break
        else:
            path_result = pending_path_result
            pending_path_result = None
        score_path = path_result.path
        if bridge_particles:
            from .kbridge import bridge_cpf_rejuvenate

            bridge_result = bridge_cpf_rejuvenate(
                np.asarray(parameters),
                score_path,
                data,
                particles=bridge_particles,
                seed=seed + 104729 * iteration + 48611,
                order=order,
                relative_floor=relative_floor,
            )
            score_path = bridge_result.path
            bridge_update_trace.append(bridge_result.update_fraction)
            bridge_ess_trace.append(bridge_result.median_ess)
        else:
            bridge_update_trace.append(np.nan)
            bridge_ess_trace.append(np.nan)
        targets = jnp.asarray(score_path[1:], dtype=jnp.float64)
        complete_value, score = value_and_score(parameters, targets)
        score_array = np.asarray(score)
        score_norm = float(np.linalg.norm(score_array))

        marginal_trace.append(path_result.loglik)
        complete_trace.append(float(complete_value))
        score_norm_trace.append(score_norm)
        median_ess_trace.append(float(np.median(path_result.ess)))
        minimum_ess_trace.append(float(np.min(path_result.ess)))
        invalid_trace.append(float(np.max(path_result.invalid_fraction)))
        ancestor_trace.append(path_result.unique_initial_ancestors)
        backward_fallback_trace.append(path_result.backward_fallbacks)
        accepted_step_trace.append(np.nan)
        backtrack_trace.append(0)
        elapsed_trace.append(perf_counter() - started)

        finite = (
            path_result.valid_path
            and np.isfinite(path_result.loglik)
            and np.isfinite(float(complete_value))
            and np.all(np.isfinite(score_array))
        )
        if not finite:
            termination_reason = "non-finite-path-or-score"
            break
        if (
            maximum_elapsed_seconds is not None
            and elapsed_trace[-1] >= maximum_elapsed_seconds
        ):
            termination_reason = "time-budget"
            break
        if iteration == iterations:
            break

        gain = (
            1.0
            if iteration < burnin
            else float(iteration - burnin + 1) ** (-gain_exponent)
        )
        averaged_score = (1.0 - gain) * averaged_score + gain * score
        averaged_norm = jnp.linalg.norm(averaged_score)
        averaged_score = averaged_score * jnp.minimum(
            1.0, gradient_clip / jnp.maximum(averaged_norm, 1e-12)
        )
        first_moment = 0.9 * first_moment + 0.1 * averaged_score
        second_moment = 0.999 * second_moment + 0.001 * averaged_score**2
        first_unbiased = first_moment / (1.0 - 0.9 ** (iteration + 1))
        second_unbiased = second_moment / (1.0 - 0.999 ** (iteration + 1))
        direction = first_unbiased / (
            jnp.sqrt(second_unbiased) + 1e-8
        )
        # A wide-box stochastic update can occasionally move into a region in
        # which every forward particle is invalid.  Test the next iteration's
        # path before committing and halve only such unusable steps.  The
        # accepted draw is cached, so a successful trial adds no extra SMC run.
        for backtrack in range(maximum_backtracks + 1):
            trial_step = working_learning_rate * backtrack_factor**backtrack
            candidate = constrain(parameters + trial_step * direction)
            candidate_path = sample_smoothing_path(
                np.asarray(candidate),
                data,
                particles=particles,
                seed=seed + 104729 * (iteration + 1),
                order=order,
                relative_floor=relative_floor,
            )
            acceptable_invalidity = (
                float(np.max(candidate_path.invalid_fraction))
                <= maximum_acceptable_invalid_fraction
            )
            if (
                candidate_path.valid_path
                and np.isfinite(candidate_path.loglik)
                and acceptable_invalidity
            ):
                parameters = candidate
                pending_path_result = candidate_path
                accepted_step_trace[-1] = trial_step
                backtrack_trace[-1] = backtrack
                working_learning_rate = trial_step
                completed_updates += 1
                consecutive_rejections = 0
                break
        else:
            accepted_step_trace[-1] = 0.0
            backtrack_trace[-1] = maximum_backtracks + 1
            consecutive_rejections += 1
            if consecutive_rejections >= maximum_consecutive_rejections:
                termination_reason = "consecutive-rejections"
                break

    return SMCScoreFitResult(
        unconstrained=np.asarray(parameters),
        parameter_trace=np.asarray(parameter_trace),
        marginal_loglik_trace=np.asarray(marginal_trace),
        complete_loglik_trace=np.asarray(complete_trace),
        score_norm_trace=np.asarray(score_norm_trace),
        median_ess_trace=np.asarray(median_ess_trace),
        minimum_ess_trace=np.asarray(minimum_ess_trace),
        maximum_invalid_fraction_trace=np.asarray(invalid_trace),
        unique_initial_ancestors_trace=np.asarray(ancestor_trace),
        backward_fallback_trace=np.asarray(backward_fallback_trace),
        bridge_update_fraction_trace=np.asarray(bridge_update_trace),
        bridge_median_ess_trace=np.asarray(bridge_ess_trace),
        accepted_step_size_trace=np.asarray(accepted_step_trace),
        backtrack_count_trace=np.asarray(backtrack_trace),
        elapsed_trace=np.asarray(elapsed_trace),
        elapsed_seconds=perf_counter() - started,
        completed_updates=completed_updates,
        termination_reason=termination_reason,
    )


# Backward-compatible names retained for validation scripts written before the
# surrogate/target distinction was made explicit.
fit_fisher_score = fit_pseudo_score
fit_smc_score = fit_pseudo_score
