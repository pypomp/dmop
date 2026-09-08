"""Karppinen-style bridge-CPF rejuvenation for within-month Dacca states.

This is a blocked-interior specialization derived from Algorithm 8: both the
lower and upper block endpoints are fixed.  The proposal is a time-varying
linear-Gaussian approximation, so its multi-step bridge laws are available in
closed form.  Potentials contain the nonlinear DS-Gaussian transition divided
by the auxiliary transition, as required by the Feynman--Kac correction.

It is not the operative bridge-backward-sampling update in Algorithm 8 and it
is not the whole CPF-BBS Algorithm 7.  In particular, monthly block boundaries
come from one independent forward-SMC path and cannot reconnect to a different
lower-boundary particle.  Consequently its update fraction is only an
interior-movement diagnostic, not KSV's probability of lower-boundary update
(PLU) and not evidence of grid-stable global smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .data import DaccaData
from .transition import ditlevsen_mean, gaussian_transition

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class BridgeCPFResult:
    path: np.ndarray
    updated_months: np.ndarray
    update_fraction: float
    interior_rms_change: float
    minimum_ess: float
    median_ess: float


@partial(jax.jit, static_argnames=("order",))
def _linear_gaussian_guide(
    unconstrained: jax.Array,
    starts: jax.Array,
    endpoints: jax.Array,
    step_covariates: jax.Array,
    *,
    order: int,
    relative_floor: float,
):
    """Linearize each auxiliary step along endpoint interpolation."""

    nstep = step_covariates.shape[1]
    dt = 1.0 / (12.0 * nstep)
    fractions = jnp.arange(nstep, dtype=jnp.float64) / float(nstep)
    reference_previous = starts[:, None, :] + fractions[None, :, None] * (
        endpoints[:, None, :] - starts[:, None, :]
    )

    def one_moment(state, forcing):
        mean, covariance = gaussian_transition(
            state,
            unconstrained,
            forcing,
            dt,
            order=order,
            relative_floor=relative_floor,
        )
        matrix = jax.jacfwd(ditlevsen_mean, argnums=0)(
            state, unconstrained, forcing, dt
        )
        offset = mean - matrix @ state
        return matrix, offset, covariance

    return jax.vmap(jax.vmap(one_moment))(reference_previous, step_covariates)


def _conditional_guide(
    matrices: np.ndarray,
    offsets: np.ndarray,
    covariances: np.ndarray,
    endpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute M(x_k | x_{k-1}, x_u) for every block and step."""

    months, nstep, dimension, _ = matrices.shape
    conditional_matrices = np.empty((months, nstep - 1, dimension, dimension))
    conditional_offsets = np.empty((months, nstep - 1, dimension))
    conditional_covariances = np.empty(
        (months, nstep - 1, dimension, dimension)
    )
    identity = np.eye(dimension)

    for month in range(months):
        suffix_matrices = np.empty((nstep + 1, dimension, dimension))
        suffix_offsets = np.empty((nstep + 1, dimension))
        suffix_covariances = np.empty((nstep + 1, dimension, dimension))
        suffix_matrices[nstep] = identity
        suffix_offsets[nstep] = 0.0
        suffix_covariances[nstep] = 0.0
        for step in range(nstep - 1, -1, -1):
            remainder_matrix = suffix_matrices[step + 1]
            suffix_matrices[step] = remainder_matrix @ matrices[month, step]
            suffix_offsets[step] = (
                remainder_matrix @ offsets[month, step]
                + suffix_offsets[step + 1]
            )
            suffix_covariances[step] = (
                remainder_matrix
                @ covariances[month, step]
                @ remainder_matrix.T
                + suffix_covariances[step + 1]
            )

        for step in range(nstep - 1):
            remainder_matrix = suffix_matrices[step + 1]
            remainder_offset = suffix_offsets[step + 1]
            remainder_covariance = suffix_covariances[step + 1]
            step_covariance = covariances[month, step]
            endpoint_covariance = (
                remainder_matrix @ step_covariance @ remainder_matrix.T
                + remainder_covariance
            )
            endpoint_covariance = 0.5 * (
                endpoint_covariance + endpoint_covariance.T
            )
            cross = step_covariance @ remainder_matrix.T
            gain = np.linalg.solve(endpoint_covariance, cross.T).T
            residual_matrix = identity - gain @ remainder_matrix
            conditional_matrices[month, step] = (
                residual_matrix @ matrices[month, step]
            )
            conditional_offsets[month, step] = (
                residual_matrix @ offsets[month, step]
                + gain @ (endpoints[month] - remainder_offset)
            )
            conditional_covariance = step_covariance - gain @ cross.T
            conditional_covariance = 0.5 * (
                conditional_covariance + conditional_covariance.T
            )
            # Do not regularize this Schur complement independently.  The
            # p/M correction below is exact only when samples come from the
            # bridge law of the same one-step guide M used in its denominator.
            # The guide covariance itself already contains the disclosed
            # relative floor supplied to ``gaussian_transition``.
            conditional_covariances[month, step] = conditional_covariance
    return conditional_matrices, conditional_offsets, conditional_covariances


def _gaussian_logpdf_batch(
    targets: jax.Array, means: jax.Array, covariances: jax.Array
) -> jax.Array:
    cholesky = jnp.linalg.cholesky(covariances)
    residuals = targets - means
    standardized = jax.vmap(
        lambda factor, residual: jax.scipy.linalg.solve_triangular(
            factor, residual, lower=True
        )
    )(cholesky.reshape(-1, 6, 6), residuals.reshape(-1, 6))
    logdet = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1
    )
    quadratic = jnp.sum(standardized.reshape(residuals.shape) ** 2, axis=-1)
    return -0.5 * (6.0 * jnp.log(2.0 * jnp.pi) + logdet + quadratic)


@partial(jax.jit, static_argnames=("particles", "order"))
def _bridge_cpf_kernel(
    unconstrained: jax.Array,
    starts: jax.Array,
    endpoints: jax.Array,
    reference_interiors: jax.Array,
    step_covariates: jax.Array,
    guide_matrices: jax.Array,
    guide_offsets: jax.Array,
    guide_covariances: jax.Array,
    conditional_matrices: jax.Array,
    conditional_offsets: jax.Array,
    conditional_cholesky: jax.Array,
    key: jax.Array,
    *,
    particles: int,
    order: int,
    relative_floor: float,
):
    """Vectorized fixed-endpoint specialization of KSV Algorithm 8."""

    months = starts.shape[0]
    nstep = step_covariates.shape[1]
    dt = 1.0 / (12.0 * nstep)
    states = jnp.repeat(starts[:, None, :], particles, axis=1)
    log_weights = jnp.zeros((months, particles), dtype=jnp.float64)

    def transition_log_ratio(previous, target, forcing, matrix, offset, covariance):
        flat_previous = previous.reshape(-1, 6)
        flat_target = target.reshape(-1, 6)
        flat_forcing = jnp.repeat(forcing[:, None, :], particles, axis=1).reshape(
            -1, forcing.shape[-1]
        )

        def target_moment(state, covariate):
            return gaussian_transition(
                state,
                unconstrained,
                covariate,
                dt,
                order=order,
                relative_floor=relative_floor,
            )

        target_means, target_covariances = jax.vmap(target_moment)(
            flat_previous, flat_forcing
        )
        target_logpdf = _gaussian_logpdf_batch(
            flat_target, target_means, target_covariances
        ).reshape(months, particles)
        guide_means = jnp.einsum("mij,mnj->mni", matrix, previous) + offset[:, None, :]
        repeated_covariances = jnp.repeat(
            covariance[:, None, :, :], particles, axis=1
        )
        guide_logpdf = _gaussian_logpdf_batch(
            target, guide_means, repeated_covariances
        )
        valid = jnp.all(jnp.isfinite(target), axis=-1) & jnp.all(target >= 0.0, axis=-1)
        return jnp.where(valid, target_logpdf - guide_logpdf, -jnp.inf)

    def interior_step(carry, step):
        previous_states, previous_log_weights, random_key = carry
        random_key, resampling_key, noise_key = jax.random.split(random_key, 3)
        resampling_keys = jax.random.split(resampling_key, months)
        ancestors = jax.vmap(
            lambda one_key, weights: jax.random.categorical(
                one_key, weights, shape=(particles,)
            )
        )(resampling_keys, previous_log_weights)
        ancestors = ancestors.at[:, 0].set(0)
        parents = jnp.take_along_axis(
            previous_states, ancestors[:, :, None], axis=1
        )
        means = (
            jnp.einsum(
                "mij,mnj->mni", conditional_matrices[:, step], parents
            )
            + conditional_offsets[:, step, None, :]
        )
        innovations = jax.random.normal(
            noise_key, (months, particles, 6), dtype=jnp.float64
        )
        proposed = means + jnp.einsum(
            "mij,mnj->mni", conditional_cholesky[:, step], innovations
        )
        proposed = proposed.at[:, 0, :].set(reference_interiors[:, step, :])
        new_log_weights = transition_log_ratio(
            parents,
            proposed,
            step_covariates[:, step],
            guide_matrices[:, step],
            guide_offsets[:, step],
            guide_covariances[:, step],
        )
        normalized = new_log_weights - jax.scipy.special.logsumexp(
            new_log_weights, axis=1, keepdims=True
        )
        ess = 1.0 / jnp.sum(jnp.exp(normalized) ** 2, axis=1)
        return (proposed, new_log_weights, random_key), (proposed, ancestors, ess)

    (states, log_weights, key), (particle_states, ancestors, ess) = jax.lax.scan(
        interior_step,
        (states, log_weights, key),
        jnp.arange(nstep - 1),
    )
    repeated_endpoints = jnp.repeat(endpoints[:, None, :], particles, axis=1)
    final_ratio = transition_log_ratio(
        states,
        repeated_endpoints,
        step_covariates[:, -1],
        guide_matrices[:, -1],
        guide_offsets[:, -1],
        guide_covariances[:, -1],
    )
    selection_weights = log_weights + final_ratio
    selection_weights = selection_weights - jax.scipy.special.logsumexp(
        selection_weights, axis=1, keepdims=True
    )
    selection_keys = jax.random.split(key, months)
    selected = jax.vmap(jax.random.categorical)(selection_keys, selection_weights)
    final_ess = 1.0 / jnp.sum(jnp.exp(selection_weights) ** 2, axis=1)
    return particle_states, ancestors, selected, ess, final_ess


def bridge_cpf_rejuvenate(
    unconstrained: np.ndarray,
    path: np.ndarray,
    data: DaccaData,
    *,
    particles: int = 16,
    seed: int = 864209,
    order: int = 2,
    relative_floor: float = 1e-7,
) -> BridgeCPFResult:
    """Update interiors with fixed-endpoint conditional SMC bridge blocks.

    This kernel borrows the bridge construction in KSV Algorithm 8 but does
    not reconnect a fixed upper endpoint to a forward particle pool.  It must
    therefore not be reported as full CPF-BBS or diagnosed using KSV's PLU.
    """

    if particles < 2:
        raise ValueError("particles must be at least two")
    expected_shape = (data.observations.size * data.nstep + 1, 6)
    path = np.asarray(path, dtype=float)
    if path.shape != expected_shape:
        raise ValueError(f"path must have shape {expected_shape}")
    if data.nstep == 1:
        return BridgeCPFResult(
            path=path.copy(),
            updated_months=np.zeros(data.observations.size, dtype=bool),
            update_fraction=0.0,
            interior_rms_change=0.0,
            minimum_ess=float("nan"),
            median_ess=float("nan"),
        )

    targets = path[1:].reshape(data.observations.size, data.nstep, 6).copy()
    endpoints = targets[:, -1, :].copy()
    starts = path[:: data.nstep][:-1].copy()
    starts[:, 5] = 0.0
    reference_interiors = targets[:, :-1, :].copy()
    guide = _linear_gaussian_guide(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(starts, dtype=jnp.float64),
        jnp.asarray(endpoints, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        order=order,
        relative_floor=relative_floor,
    )
    guide_matrices, guide_offsets, guide_covariances = (
        np.asarray(value) for value in guide
    )
    conditional_matrices, conditional_offsets, conditional_covariances = (
        _conditional_guide(
            guide_matrices, guide_offsets, guide_covariances, endpoints
        )
    )
    conditional_cholesky = np.linalg.cholesky(conditional_covariances)
    outputs = _bridge_cpf_kernel(
        jnp.asarray(unconstrained, dtype=jnp.float64),
        jnp.asarray(starts, dtype=jnp.float64),
        jnp.asarray(endpoints, dtype=jnp.float64),
        jnp.asarray(reference_interiors, dtype=jnp.float64),
        jnp.asarray(data.step_covariates, dtype=jnp.float64),
        jnp.asarray(guide_matrices, dtype=jnp.float64),
        jnp.asarray(guide_offsets, dtype=jnp.float64),
        jnp.asarray(guide_covariances, dtype=jnp.float64),
        jnp.asarray(conditional_matrices, dtype=jnp.float64),
        jnp.asarray(conditional_offsets, dtype=jnp.float64),
        jnp.asarray(conditional_cholesky, dtype=jnp.float64),
        jax.random.key(seed),
        particles=particles,
        order=order,
        relative_floor=relative_floor,
    )
    particle_states, ancestor_array, selected, ess, final_ess = (
        np.asarray(value) for value in outputs
    )
    # Scan outputs are (interior_step, month, particle, coordinate).
    new_interiors = np.empty_like(reference_interiors)
    for month in range(data.observations.size):
        particle_index = int(selected[month])
        for step in range(data.nstep - 2, -1, -1):
            new_interiors[month, step] = particle_states[
                step, month, particle_index
            ]
            particle_index = int(ancestor_array[step, month, particle_index])

    targets[:, :-1, :] = new_interiors
    new_path = np.concatenate((path[:1], targets.reshape(-1, 6)), axis=0)
    differences = new_interiors - reference_interiors
    updated = np.any(np.abs(differences) > 1e-12, axis=(1, 2))
    all_ess = np.concatenate((ess.reshape(-1), final_ess.reshape(-1)))
    return BridgeCPFResult(
        path=new_path,
        updated_months=updated,
        update_fraction=float(np.mean(updated)),
        interior_rms_change=float(np.sqrt(np.mean(differences**2))),
        minimum_ess=float(np.min(all_ess)),
        median_ess=float(np.median(all_ess)),
    )
