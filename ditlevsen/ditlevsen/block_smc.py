"""Monthly SMC based on composed Ditlevsen--Samson order-1.5 moments."""

from __future__ import annotations

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
from .smc import SMCFilterResult, SMCPathResult, SMCScoreFitResult, _normal_logpdf
from .transition import ditlevsen_block_transition

jax.config.update("jax_enable_x64", True)


def _batched_mvn_logpdf(
    values: jax.Array,
    means: jax.Array,
    cholesky: jax.Array,
) -> jax.Array:
    """Multivariate normal log densities for a particle batch."""

    differences = values - means
    standardized = jax.vmap(
        lambda factor, difference: jax.scipy.linalg.solve_triangular(
            factor, difference, lower=True
        )
    )(cholesky, differences)
    dimension = means.shape[1]
    return -0.5 * (
        dimension * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64))
        + 2.0
        * jnp.sum(jnp.log(jnp.diagonal(cholesky, axis1=1, axis2=2)), axis=1)
        + jnp.sum(standardized**2, axis=1)
    )


def _draw_block_proposal(
    means: jax.Array,
    covariances: jax.Array,
    innovations: jax.Array,
    observation: jax.Array,
    observation_population: jax.Array,
    tau: jax.Array,
    *,
    guided: bool,
) -> tuple[jax.Array, jax.Array]:
    """Draw from the bootstrap or observation-guided Gaussian proposal."""

    transition_cholesky = jnp.linalg.cholesky(covariances)
    if not guided:
        proposed = means + jnp.einsum(
            "nij,nj->ni", transition_cholesky, innovations
        )
        return proposed, jnp.zeros((means.shape[0],), dtype=jnp.float64)

    predicted_deaths = observation_population * means[:, 5]
    surrogate_scale = tau * jnp.maximum(jnp.abs(predicted_deaths), 1e-6)
    cross_covariance = observation_population * covariances[:, :, 5]
    innovation_variance = (
        observation_population**2 * covariances[:, 5, 5]
        + surrogate_scale**2
    )
    proposal_means = means + cross_covariance * (
        (observation - predicted_deaths) / innovation_variance
    )[:, None]
    proposal_covariances = covariances - (
        cross_covariance[:, :, None] * cross_covariance[:, None, :]
        / innovation_variance[:, None, None]
    )
    proposal_covariances = 0.5 * (
        proposal_covariances
        + jnp.swapaxes(proposal_covariances, 1, 2)
    )
    proposal_scale = jnp.maximum(
        jnp.max(jnp.diagonal(proposal_covariances, axis1=1, axis2=2), axis=1),
        1e-20,
    )
    proposal_covariances = proposal_covariances + (
        1e-12
        * proposal_scale[:, None, None]
        * jnp.eye(means.shape[1], dtype=jnp.float64)[None, :, :]
    )
    proposal_cholesky = jnp.linalg.cholesky(proposal_covariances)
    proposed = proposal_means + jnp.einsum(
        "nij,nj->ni", proposal_cholesky, innovations
    )
    prior_logpdf = _batched_mvn_logpdf(
        proposed, means, transition_cholesky
    )
    proposal_logpdf = _batched_mvn_logpdf(
        proposed, proposal_means, proposal_cholesky
    )
    return proposed, prior_logpdf - proposal_logpdf


def _is_guided(proposal: str) -> bool:
    if proposal not in ("bootstrap", "guided"):
        raise ValueError("proposal must be 'bootstrap' or 'guided'")
    return proposal == "guided"


@partial(jax.jit, static_argnames=("particles", "guided"))
def _block_filter_kernel(
    unconstrained: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    key: jax.Array,
    *,
    particles: int,
    endpoint_relative_floor: float,
    guided: bool,
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
        innovations = jax.random.normal(
            noise_key, (particles, state0.shape[0]), dtype=jnp.float64
        )
        proposed, proposal_correction = _draw_block_proposal(
            means,
            covariances,
            innovations,
            observation,
            observation_population,
            theta["tau"],
            guided=guided,
        )
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
            proposal_correction
            + jnp.logaddexp(gaussian, log_tolerance),
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
    proposal: str = "bootstrap",
) -> SMCFilterResult:
    """Run the monthly DS order-1.5 block Gaussian particle filter."""

    if particles < 2:
        raise ValueError("particles must be at least two")
    guided = _is_guided(proposal)
    increments, ess, invalid_fraction = _block_filter_kernel(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
        jnp.asarray(data.observations, dtype=jnp.float64),
        jax.random.key(seed),
        particles=particles,
        endpoint_relative_floor=endpoint_relative_floor,
        guided=guided,
    )
    increments = np.asarray(increments)
    return SMCFilterResult(
        loglik=float(np.sum(increments)),
        loglik_increments=increments,
        ess=np.asarray(ess),
        invalid_fraction=np.asarray(invalid_fraction),
    )


@partial(jax.jit, static_argnames=("particles", "guided"))
def _block_path_filter_kernel(
    unconstrained: jax.Array,
    step_covariates: jax.Array,
    observation_populations: jax.Array,
    observations: jax.Array,
    key: jax.Array,
    *,
    particles: int,
    endpoint_relative_floor: float,
    guided: bool,
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

        # The FFBSi backward pass needs p(x[m+1] | x[m]) for every filtering
        # particle x[m].  Compute those moments before resampling and retain
        # them.  The forward proposal then gathers the same moments by its
        # sampled ancestry, avoiding a second 20-step propagation in FFBSi.
        transition_starts = states.at[:, 5].set(0.0)

        def moments(state):
            return ditlevsen_block_transition(
                state,
                unconstrained,
                covariates,
                dt,
                endpoint_relative_floor=endpoint_relative_floor,
            )

        transition_means, transition_covariances = jax.vmap(moments)(
            transition_starts
        )
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
        states = transition_starts[ancestors]
        valid = valid[ancestors]
        means = transition_means[ancestors]
        covariances = transition_covariances[ancestors]
        random_key, noise_key = jax.random.split(random_key)
        innovations = jax.random.normal(
            noise_key, (particles, state0.shape[0]), dtype=jnp.float64
        )
        proposed, proposal_correction = _draw_block_proposal(
            means,
            covariances,
            innovations,
            observation,
            observation_population,
            theta["tau"],
            guided=guided,
        )
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
            proposal_correction
            + jnp.logaddexp(gaussian, log_tolerance),
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
            states,
            ancestors,
            normalized_log_weights,
            increment,
            ess,
            invalid_fraction,
            transition_means,
            transition_covariances,
        )

    months = jnp.arange(step_covariates.shape[0])
    (_, final_valid, final_log_weights, final_key), outputs = jax.lax.scan(
        observation_interval,
        (states0, valid0, uniform_log_weights, key),
        (months, step_covariates, observation_populations, observations),
    )
    final_key, selection_key = jax.random.split(final_key)
    selected = jax.random.categorical(selection_key, final_log_weights)
    (
        endpoints,
        ancestors,
        normalized_log_weights,
        increments,
        ess,
        invalid_fraction,
        transition_means,
        transition_covariances,
    ) = outputs
    lineages = jax.lax.fori_loop(
        1,
        ancestors.shape[0],
        lambda month, current: current[ancestors[month]],
        jnp.arange(particles),
    )
    unique_initial_ancestors = jnp.sum(
        jnp.bincount(lineages, length=particles) > 0
    )
    return (
        endpoints,
        ancestors,
        normalized_log_weights,
        selected,
        final_valid[selected],
        increments,
        ess,
        invalid_fraction,
        unique_initial_ancestors,
        transition_means,
        transition_covariances,
    )


@jax.jit
def _block_backward_sample_kernel(
    endpoints: jax.Array,
    ancestors: jax.Array,
    normalized_log_weights: jax.Array,
    transition_means: jax.Array,
    transition_covariances: jax.Array,
    selected_final: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Draw endpoints by FFBSi using moments retained by the forward pass."""

    particles = endpoints.shape[1]
    dimension = endpoints.shape[2]
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64))
    uniform_log_weight = -jnp.log(
        jnp.asarray(float(particles), dtype=jnp.float64)
    )
    final_state = endpoints[-1, selected_final]

    def backward_step(carry, inputs):
        next_state, next_index, fallback_count, random_key = carry
        (
            states,
            filter_log_weights,
            means,
            covariances,
            forward_ancestors,
        ) = inputs
        cholesky = jnp.linalg.cholesky(covariances)
        differences = next_state[None, :] - means
        standardized = jax.vmap(
            lambda factor, difference: jax.scipy.linalg.solve_triangular(
                factor, difference, lower=True
            )
        )(cholesky, differences)
        transition_loglik = -0.5 * (
            dimension * log_two_pi
            + 2.0 * jnp.sum(
                jnp.log(jnp.diagonal(cholesky, axis1=1, axis2=2)), axis=1
            )
            + jnp.sum(standardized**2, axis=1)
        )
        backward_log_weights = filter_log_weights + transition_loglik
        finite = jnp.isfinite(backward_log_weights)
        survived = jnp.any(finite)
        sampling_log_weights = jnp.where(
            survived,
            backward_log_weights,
            uniform_log_weight,
        )
        random_key, selection_key = jax.random.split(random_key)
        backward_selected = jax.random.categorical(
            selection_key, sampling_log_weights
        )
        selected = jnp.where(
            survived,
            backward_selected,
            forward_ancestors[next_index],
        )
        selected_state = states[selected]
        return (
            selected_state,
            selected,
            fallback_count + (~survived).astype(jnp.int32),
            random_key,
        ), selected_state

    (_, _, fallback_count, _), reverse_states = jax.lax.scan(
        backward_step,
        (final_state, selected_final, jnp.asarray(0, dtype=jnp.int32), key),
        (
            endpoints[:-1][::-1],
            normalized_log_weights[:-1][::-1],
            transition_means[1:][::-1],
            transition_covariances[1:][::-1],
            ancestors[1:][::-1],
        ),
    )
    selected_endpoints = jnp.concatenate(
        (reverse_states[::-1], final_state[None, :]), axis=0
    )
    return selected_endpoints, fallback_count


def sample_block_smoothing_path(
    unconstrained: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 100,
    seed: int = 631409,
    endpoint_relative_floor: float = 1e-12,
    proposal: str = "bootstrap",
) -> SMCPathResult:
    """Draw one monthly endpoint path with an FFBSi particle smoother."""

    if particles < 2:
        raise ValueError("particles must be at least two")
    guided = _is_guided(proposal)
    outputs = _block_path_filter_kernel(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        jnp.asarray(data.observation_covariates[:, 2], dtype=jnp.float64),
        jnp.asarray(data.observations, dtype=jnp.float64),
        jax.random.key(seed),
        particles=particles,
        endpoint_relative_floor=endpoint_relative_floor,
        guided=guided,
    )
    (
        endpoints,
        ancestors,
        normalized_log_weights,
        selected,
        selected_valid,
        increments,
        ess,
        invalid_fraction,
        unique_initial_ancestors,
        transition_means,
        transition_covariances,
    ) = outputs
    selected_endpoints, backward_fallbacks = _block_backward_sample_kernel(
        endpoints,
        ancestors,
        normalized_log_weights,
        transition_means,
        transition_covariances,
        selected,
        jax.random.key(seed + 32452843),
    )
    selected_endpoints = np.asarray(selected_endpoints)
    increments = np.asarray(increments)
    ess = np.asarray(ess)
    invalid_fraction = np.asarray(invalid_fraction)
    path = np.concatenate(
        (
            np.asarray(initial_state(jnp.asarray(unconstrained)))[None, :],
            selected_endpoints,
        ),
        axis=0,
    )

    return SMCPathResult(
        path=path,
        loglik=float(np.sum(increments)),
        loglik_increments=increments,
        ess=ess,
        invalid_fraction=invalid_fraction,
        unique_initial_ancestors=int(np.asarray(unique_initial_ancestors)),
        valid_path=bool(np.asarray(selected_valid))
        and bool(np.isfinite(selected_endpoints).all()),
        backward_fallbacks=int(np.asarray(backward_fallbacks)),
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
    proposal: str = "bootstrap",
    gradient_clip: float = 100.0,
    maximum_backtracks: int = 6,
    maximum_consecutive_rejections: int = 10,
    maximum_path_retries: int = 3,
    backtrack_factor: float = 0.5,
    maximum_acceptable_invalid_fraction: float = 0.5,
    likelihood_guard_particles: int = 50,
    likelihood_guard_interval: int = 10,
    maximum_guard_loglik_drop: float = 5.0,
    maximum_elapsed_seconds: float | None = None,
    project_to_bounds: bool = True,
) -> SMCScoreFitResult:
    """Fit the monthly block pseudo-model by SMC Fisher-score ascent."""

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    _is_guided(proposal)
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
    if likelihood_guard_particles != 0 and likelihood_guard_particles < 2:
        raise ValueError("likelihood_guard_particles must be zero or at least two")
    if likelihood_guard_interval < 1:
        raise ValueError("likelihood_guard_interval must be positive")
    if maximum_guard_loglik_drop < 0.0:
        raise ValueError("maximum_guard_loglik_drop must be nonnegative")
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
    backward_fallback_trace: list[int] = []
    accepted_step_trace: list[float] = []
    backtrack_trace: list[int] = []
    elapsed_trace: list[float] = []
    started = perf_counter()
    completed_updates = 0
    termination_reason = "completed"
    retained_parameters = parameters
    guard_checks = 0
    consecutive_rejections = 0

    for iteration in range(iterations + 1):
        parameter_trace.append(np.asarray(parameters))
        for retry in range(maximum_path_retries + 1):
            path_result = sample_block_smoothing_path(
                np.asarray(parameters),
                data,
                particles=particles,
                seed=seed + 104729 * iteration + 15485863 * retry,
                endpoint_relative_floor=endpoint_relative_floor,
                proposal=proposal,
            )
            if path_result.valid_path and np.isfinite(path_result.loglik):
                break

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
        direction = first_unbiased / (jnp.sqrt(second_unbiased) + 1e-8)

        scheduled_step = _parameter_step_size(
            iteration,
            learning_rate,
            learning_rate_decay_start,
            learning_rate_decay_exponent,
        )
        guard_iteration = bool(likelihood_guard_particles) and (
            (iteration + 1) % likelihood_guard_interval == 0
        )
        if guard_iteration:
            guard_checks += 1
            guard_seed = seed + 49979687 + 104729 * iteration
            guard_reference = block_bootstrap_filter(
                np.asarray(retained_parameters),
                data,
                particles=likelihood_guard_particles,
                seed=guard_seed,
                endpoint_relative_floor=endpoint_relative_floor,
                proposal=proposal,
            )
        else:
            guard_reference = None
        for backtrack in range(maximum_backtracks + 1):
            trial_step = scheduled_step * backtrack_factor**backtrack
            candidate = constrain(parameters + trial_step * direction)
            if guard_iteration:
                guard_candidate = block_bootstrap_filter(
                    np.asarray(candidate),
                    data,
                    particles=likelihood_guard_particles,
                    seed=guard_seed,
                    endpoint_relative_floor=endpoint_relative_floor,
                    proposal=proposal,
                )
                candidate_finite = np.isfinite(guard_candidate.loglik)
                acceptable_invalidity = (
                    float(np.max(guard_candidate.invalid_fraction))
                    <= maximum_acceptable_invalid_fraction
                )
                acceptable_likelihood = (
                    candidate_finite
                    and (
                        guard_reference is None
                        or not np.isfinite(guard_reference.loglik)
                        or guard_candidate.loglik
                        >= guard_reference.loglik - maximum_guard_loglik_drop
                    )
                )
            else:
                candidate_finite = bool(
                    np.all(np.isfinite(np.asarray(candidate)))
                )
                acceptable_invalidity = True
                acceptable_likelihood = True
            if (
                candidate_finite
                and acceptable_invalidity
                and acceptable_likelihood
            ):
                parameters = candidate
                if not likelihood_guard_particles:
                    retained_parameters = candidate
                elif guard_iteration and (
                    guard_reference is None
                    or not np.isfinite(guard_reference.loglik)
                    or guard_candidate.loglik >= guard_reference.loglik
                ):
                    retained_parameters = candidate
                accepted_step_trace[-1] = trial_step
                backtrack_trace[-1] = backtrack
                completed_updates += 1
                consecutive_rejections = 0
                break
        else:
            accepted_step_trace[-1] = 0.0
            backtrack_trace[-1] = maximum_backtracks + 1
            if guard_iteration:
                parameters = retained_parameters
                averaged_score = jnp.zeros_like(parameters)
                first_moment = jnp.zeros_like(parameters)
                second_moment = jnp.zeros_like(parameters)
            consecutive_rejections += 1
            if consecutive_rejections >= maximum_consecutive_rejections:
                termination_reason = "consecutive-rejections"
                break

    nan_trace = np.full(len(parameter_trace), np.nan)
    return SMCScoreFitResult(
        # The guard rejects numerically unsafe proposals during fitting.  It
        # must not silently replace the terminal iterate with a much older
        # block-surrogate optimum: on Dacca the surrogate and Euler-20 targets
        # are not aligned closely enough for that to be a valid stopping rule.
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
        bridge_update_fraction_trace=nan_trace,
        bridge_median_ess_trace=nan_trace,
        accepted_step_size_trace=np.asarray(accepted_step_trace),
        backtrack_count_trace=np.asarray(backtrack_trace),
        elapsed_trace=np.asarray(elapsed_trace),
        elapsed_seconds=perf_counter() - started,
        completed_updates=completed_updates,
        termination_reason=termination_reason,
    )
