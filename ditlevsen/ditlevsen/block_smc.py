"""Monthly SMC based on composed Ditlevsen--Samson order-1.5 moments."""

from __future__ import annotations

from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .data import DaccaData
from .model import decode_parameters, initial_state, project_parameters
from .smc import SMCFilterResult, SMCPathResult, SMCScoreFitResult, _normal_logpdf
from .transition import ditlevsen_block_transition

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("particles",))
def _block_filter_kernel(
    unconstrained: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    key: jax.Array,
    *,
    particles: int,
    endpoint_relative_floor: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Bootstrap filter with one composed Gaussian draw per month."""

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
        states = states.at[:, 5].set(0.0)
        random_key, noise_key = jax.random.split(random_key)

        def moments(state):
            return ditlevsen_block_transition(
                state,
                unconstrained,
                covariates,
                dt,
                endpoint_relative_floor=endpoint_relative_floor,
            )

        means, covariances = jax.vmap(moments)(states)
        cholesky = jnp.linalg.cholesky(covariances)
        innovations = jax.random.normal(
            noise_key, (particles, state0.shape[0]), dtype=jnp.float64
        )
        proposed = means + jnp.einsum("nij,nj->ni", cholesky, innovations)
        finite = jnp.all(jnp.isfinite(proposed), axis=1)
        nonnegative = jnp.all(proposed >= 0.0, axis=1)
        valid = valid & finite & nonnegative
        states = jnp.nan_to_num(proposed, nan=0.0, posinf=1.0, neginf=0.0)
        states = jnp.maximum(states, 0.0)

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

        random_key, resample_key = jax.random.split(random_key)
        ancestors = jax.random.categorical(
            resample_key, normalized_log_weights, shape=(particles,)
        )
        return (
            states[ancestors],
            valid[ancestors],
            random_key,
        ), (increment, ess, invalid_fraction)

    (_, _, _), diagnostics = jax.lax.scan(
        observation_interval,
        (states0, valid0, key),
        (step_covariates, observation_populations, observations),
    )
    return diagnostics


def block_bootstrap_filter(
    unconstrained: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 100,
    seed: int = 631409,
    endpoint_relative_floor: float = 1e-12,
) -> SMCFilterResult:
    """Run the monthly DS order-1.5 block Gaussian particle filter."""

    if particles < 2:
        raise ValueError("particles must be at least two")
    increments, ess, invalid_fraction = _block_filter_kernel(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
        jnp.asarray(data.observations, dtype=jnp.float64),
        jax.random.key(seed),
        particles=particles,
        endpoint_relative_floor=endpoint_relative_floor,
    )
    increments = np.asarray(increments)
    return SMCFilterResult(
        loglik=float(np.sum(increments)),
        loglik_increments=increments,
        ess=np.asarray(ess),
        invalid_fraction=np.asarray(invalid_fraction),
    )


@partial(jax.jit, static_argnames=("particles",))
def _block_path_filter_kernel(
    unconstrained: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    key: jax.Array,
    *,
    particles: int,
    endpoint_relative_floor: float,
):
    """Monthly path-space filter retaining endpoints and ancestry."""

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
        states = states[ancestors].at[:, 5].set(0.0)
        valid = valid[ancestors]
        random_key, noise_key = jax.random.split(random_key)

        def moments(state):
            return ditlevsen_block_transition(
                state,
                unconstrained,
                covariates,
                dt,
                endpoint_relative_floor=endpoint_relative_floor,
            )

        means, covariances = jax.vmap(moments)(states)
        cholesky = jnp.linalg.cholesky(covariances)
        innovations = jax.random.normal(
            noise_key, (particles, state0.shape[0]), dtype=jnp.float64
        )
        proposed = means + jnp.einsum("nij,nj->ni", cholesky, innovations)
        finite = jnp.all(jnp.isfinite(proposed), axis=1)
        nonnegative = jnp.all(proposed >= 0.0, axis=1)
        valid = valid & finite & nonnegative
        states = jnp.nan_to_num(proposed, nan=0.0, posinf=1.0, neginf=0.0)
        states = jnp.maximum(states, 0.0)

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
        ), (states, ancestors, increment, ess, invalid_fraction)

    months = jnp.arange(step_covariates.shape[0])
    (_, final_valid, final_log_weights, final_key), outputs = jax.lax.scan(
        observation_interval,
        (states0, valid0, uniform_log_weights, key),
        (months, step_covariates, observation_populations, observations),
    )
    final_key, selection_key = jax.random.split(final_key)
    selected = jax.random.categorical(selection_key, final_log_weights)
    endpoints, ancestors, increments, ess, invalid_fraction = outputs
    return (
        endpoints,
        ancestors,
        selected,
        final_valid[selected],
        increments,
        ess,
        invalid_fraction,
    )


def sample_block_smoothing_path(
    unconstrained: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 100,
    seed: int = 631409,
    endpoint_relative_floor: float = 1e-12,
) -> SMCPathResult:
    """Draw one monthly endpoint path from the forward genealogy."""

    if particles < 2:
        raise ValueError("particles must be at least two")
    outputs = _block_path_filter_kernel(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
        jnp.asarray(data.observations, dtype=jnp.float64),
        jax.random.key(seed),
        particles=particles,
        endpoint_relative_floor=endpoint_relative_floor,
    )
    (
        endpoint_array,
        ancestor_array,
        selected,
        selected_valid,
        increments,
        ess,
        invalid_fraction,
    ) = (np.asarray(value) for value in outputs)

    particle_index = int(selected)
    selected_endpoints: list[np.ndarray] = []
    for month in range(endpoint_array.shape[0] - 1, -1, -1):
        selected_endpoints.append(endpoint_array[month, particle_index, :])
        particle_index = int(ancestor_array[month, particle_index])
    selected_endpoints.reverse()
    path = np.concatenate(
        (
            np.asarray(initial_state(jnp.asarray(unconstrained)))[None, :],
            np.asarray(selected_endpoints),
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


@jax.jit
def _complete_block_path_loglik(
    unconstrained: jax.Array,
    path_targets: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    endpoint_relative_floor: float,
) -> jax.Array:
    """Complete pseudo-log-likelihood for monthly endpoint states."""

    nstep = step_covariates.shape[1]
    dt = 1.0 / (12.0 * nstep)
    theta = decode_parameters(unconstrained)
    initial = initial_state(unconstrained)
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64))
    log_tolerance = jnp.log(jnp.asarray(1e-18, dtype=jnp.float64))

    def interval(carry, inputs):
        previous, total = carry
        covariates, target, population, observation = inputs
        previous = previous.at[5].set(0.0)
        mean, covariance = ditlevsen_block_transition(
            previous,
            unconstrained,
            covariates,
            dt,
            endpoint_relative_floor=endpoint_relative_floor,
        )
        cholesky = jnp.linalg.cholesky(covariance)
        standardized = jax.scipy.linalg.solve_triangular(
            cholesky, target - mean, lower=True
        )
        transition_loglik = -0.5 * (
            6.0 * log_two_pi
            + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
            + jnp.dot(standardized, standardized)
        )
        deaths = population * target[5]
        measurement_scale = theta["tau"] * deaths + 1e-18
        measurement_loglik = _normal_logpdf(
            observation, deaths, measurement_scale
        )
        total = (
            total
            + transition_loglik
            + jnp.logaddexp(measurement_loglik, log_tolerance)
        )
        return (target, total), None

    (_, total), _ = jax.lax.scan(
        interval,
        (initial, jnp.asarray(0.0, dtype=jnp.float64)),
        (
            step_covariates,
            path_targets,
            observation_populations,
            observations,
        ),
    )
    return total


def complete_block_path_loglik(
    unconstrained: np.ndarray,
    path: np.ndarray,
    data: DaccaData,
    *,
    endpoint_relative_floor: float = 1e-12,
) -> float:
    """Evaluate the monthly block complete pseudo-log-likelihood."""

    expected = data.observations.size + 1
    if np.asarray(path).shape != (expected, 6):
        raise ValueError(f"path must have shape {(expected, 6)}")
    return float(
        _complete_block_path_loglik(
            jnp.asarray(unconstrained, dtype=jnp.float64),
            jnp.asarray(path[1:], dtype=jnp.float64),
            jnp.asarray(data.step_covariates, dtype=jnp.float64),
            jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
            jnp.asarray(data.observations, dtype=jnp.float64),
            jnp.asarray(endpoint_relative_floor, dtype=jnp.float64),
        )
    )


def _parameter_step_size(
    iteration: int,
    learning_rate: float,
    decay_start: int,
    decay_exponent: float,
) -> float:
    """Adam step size, constant initially and polynomially decaying afterward."""

    if iteration < decay_start:
        return learning_rate
    age = iteration - decay_start + 1
    return learning_rate * float(age) ** (-decay_exponent)


def fit_block_pseudo_score(
    start: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 100,
    iterations: int = 10,
    learning_rate: float = 0.001,
    learning_rate_decay_start: int = 30,
    learning_rate_decay_exponent: float = 0.0,
    burnin: int = 30,
    gain_exponent: float = 0.9,
    seed: int = 631409,
    endpoint_relative_floor: float = 1e-12,
    gradient_clip: float = 100.0,
    maximum_backtracks: int = 6,
    maximum_consecutive_rejections: int = 10,
    maximum_path_retries: int = 3,
    backtrack_factor: float = 0.5,
    maximum_acceptable_invalid_fraction: float = 0.5,
    maximum_elapsed_seconds: float | None = None,
) -> SMCScoreFitResult:
    """Fit the monthly block pseudo-model by SMC Fisher-score ascent."""

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if learning_rate_decay_start < 0:
        raise ValueError("learning_rate_decay_start must be nonnegative")
    if learning_rate_decay_exponent < 0.0:
        raise ValueError("learning_rate_decay_exponent must be nonnegative")
    if gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive")
    if endpoint_relative_floor < 0.0:
        raise ValueError("endpoint_relative_floor must be nonnegative")
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

    parameters = project_parameters(jnp.asarray(start, dtype=jnp.float64))
    first_moment = jnp.zeros_like(parameters)
    second_moment = jnp.zeros_like(parameters)
    averaged_score = jnp.zeros_like(parameters)
    covariates = jnp.asarray(data.step_covariates, dtype=jnp.float64)
    populations = jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64)
    observations = jnp.asarray(data.observations, dtype=jnp.float64)
    floor = jnp.asarray(endpoint_relative_floor, dtype=jnp.float64)
    value_and_score = jax.jit(
        jax.value_and_grad(
            lambda candidate, targets: _complete_block_path_loglik(
                candidate,
                targets,
                covariates,
                populations,
                observations,
                floor,
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
    accepted_step_trace: list[float] = []
    backtrack_trace: list[int] = []
    elapsed_trace: list[float] = []
    started = perf_counter()
    completed_updates = 0
    termination_reason = "completed"
    pending_path_result: SMCPathResult | None = None
    backtracking_scale = 1.0
    consecutive_rejections = 0

    for iteration in range(iterations + 1):
        parameter_trace.append(np.asarray(parameters))
        if pending_path_result is None:
            for retry in range(maximum_path_retries + 1):
                path_result = sample_block_smoothing_path(
                    np.asarray(parameters),
                    data,
                    particles=particles,
                    seed=seed + 104729 * iteration + 15485863 * retry,
                    endpoint_relative_floor=endpoint_relative_floor,
                )
                if path_result.valid_path and np.isfinite(path_result.loglik):
                    break
        else:
            path_result = pending_path_result
            pending_path_result = None

        targets = jnp.asarray(path_result.path[1:], dtype=jnp.float64)
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
        direction = first_unbiased / (jnp.sqrt(second_unbiased) + 1e-8)

        scheduled_step = _parameter_step_size(
            iteration,
            learning_rate,
            learning_rate_decay_start,
            learning_rate_decay_exponent,
        )
        for backtrack in range(maximum_backtracks + 1):
            trial_step = (
                scheduled_step
                * backtracking_scale
                * backtrack_factor**backtrack
            )
            candidate = project_parameters(parameters + trial_step * direction)
            candidate_path = sample_block_smoothing_path(
                np.asarray(candidate),
                data,
                particles=particles,
                seed=seed + 104729 * (iteration + 1),
                endpoint_relative_floor=endpoint_relative_floor,
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
                backtracking_scale *= backtrack_factor**backtrack
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

    nan_trace = np.full(len(parameter_trace), np.nan)
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
        bridge_update_fraction_trace=nan_trace,
        bridge_median_ess_trace=nan_trace,
        accepted_step_size_trace=np.asarray(accepted_step_trace),
        backtrack_count_trace=np.asarray(backtrack_trace),
        elapsed_trace=np.asarray(elapsed_trace),
        elapsed_seconds=perf_counter() - started,
        completed_updates=completed_updates,
        termination_reason=termination_reason,
    )
