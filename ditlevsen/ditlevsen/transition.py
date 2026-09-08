"""Ditlevsen--Samson moment approximation and Dacca diagnostics."""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np

from .model import diffusion, drift

jax.config.update("jax_enable_x64", True)


def rk4_mean(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float | jax.Array,
) -> jax.Array:
    """Fourth-order deterministic mean, with local covariates held fixed."""

    k1 = drift(state, unconstrained, covariates)
    k2 = drift(state + 0.5 * dt * k1, unconstrained, covariates)
    k3 = drift(state + 0.5 * dt * k2, unconstrained, covariates)
    k4 = drift(state + dt * k3, unconstrained, covariates)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ditlevsen_mean(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float | jax.Array,
) -> jax.Array:
    """Second-order conditional mean used by Ditlevsen--Samson.

    Their equation (10) is ``x + dt*b + dt**2*L b/2``, where ``L`` is the
    Ito generator.  Dacca has one Brownian vector field, so
    ``L b = J_b b + Hess(b)[g,g]/2``.  Covariates are held fixed over the
    local transition, consistently with the manuscript's numerical model.
    """

    drift_value = drift(state, unconstrained, covariates)
    drift_jacobian = jax.jacfwd(drift, argnums=0)(
        state, unconstrained, covariates
    )
    diffusion_value = diffusion(state, unconstrained, covariates)
    drift_hessian = jax.jacfwd(jax.jacfwd(drift, argnums=0), argnums=0)(
        state, unconstrained, covariates
    )
    diffusion_curvature = jnp.einsum(
        "i,kij,j->k", diffusion_value, drift_hessian, diffusion_value
    )
    generator_drift = (
        drift_jacobian @ drift_value + 0.5 * diffusion_curvature
    )
    return state + dt * drift_value + 0.5 * dt**2 * generator_drift


def stratonovich_drift(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
) -> jax.Array:
    """Drift vector field used by the weak Hörmander bracket condition."""

    diffusion_value = diffusion(state, unconstrained, covariates)
    diffusion_jacobian = jax.jacfwd(diffusion, argnums=0)(
        state, unconstrained, covariates
    )
    return (
        drift(state, unconstrained, covariates)
        - 0.5 * diffusion_jacobian @ diffusion_value
    )


def _krylov_factor(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float | jax.Array,
    order: int,
) -> jax.Array:
    """Scaled controllability columns ``J^k g dt^(k+1/2) / k!``."""

    jacobian = jax.jacfwd(drift, argnums=0)(state, unconstrained, covariates)
    vector = diffusion(state, unconstrained, covariates)
    columns = []
    for power in range(order):
        columns.append(vector * (dt ** (power + 0.5)) / math.factorial(power))
        vector = jacobian @ vector
    return jnp.stack(columns, axis=1)


def _integrated_polynomial_covariance(order: int) -> jax.Array:
    indices = jnp.arange(order, dtype=jnp.float64)
    return 1.0 / (indices[:, None] + indices[None, :] + 1.0)


def local_gaussian_covariance(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float | jax.Array,
    *,
    order: int,
    relative_floor: float = 1e-12,
) -> jax.Array:
    """Covariance from a truncated stochastic controllability expansion.

    ``order=2`` is the leading Ditlevsen--Samson order-1.5 covariance after
    locally freezing the state-dependent diffusion,
    ``dt gg' + dt^2 (Jg g' + g (Jg)')/2 + dt^3 Jg(Jg)'/3``.
    Higher orders complete the covariance for Dacca's longer Hörmander chain.
    A tiny numerical floor is added only after constructing the covariance.
    """

    factor = _krylov_factor(state, unconstrained, covariates, dt, order)
    hilbert = _integrated_polynomial_covariance(order)
    covariance = factor @ hilbert @ factor.T
    covariance = 0.5 * (covariance + covariance.T)
    scale = jnp.maximum(jnp.max(jnp.diag(covariance)), 1e-20)
    return covariance + relative_floor * scale * jnp.eye(state.shape[0])


def ditlevsen_covariance(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float | jax.Array,
    *,
    relative_floor: float = 0.0,
) -> jax.Array:
    """First-bracket DS leading-covariance generalization to Dacca.

    Dacca is outside the paper's one-smooth/full-rank-rough model class, so
    calling this a literal application of their scheme would be inaccurate.
    """

    return local_gaussian_covariance(
        state,
        unconstrained,
        covariates,
        dt,
        order=2,
        relative_floor=relative_floor,
    )


def completed_covariance(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float | jax.Array,
    *,
    relative_floor: float = 1e-12,
) -> jax.Array:
    """Six-direction local Gaussian extension used in the runnable benchmark."""

    return local_gaussian_covariance(
        state,
        unconstrained,
        covariates,
        dt,
        order=state.shape[0],
        relative_floor=relative_floor,
    )


def gaussian_transition(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float | jax.Array,
    *,
    order: int = 6,
    relative_floor: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """Return mean and covariance of the local hypoelliptic Gaussian."""

    mean = ditlevsen_mean(state, unconstrained, covariates, dt)
    covariance = local_gaussian_covariance(
        state,
        unconstrained,
        covariates,
        dt,
        order=order,
        relative_floor=relative_floor,
    )
    return mean, covariance


def lie_bracket_diagnostics(
    state: np.ndarray,
    unconstrained: np.ndarray,
    covariates: np.ndarray,
    *,
    maximum_depth: int = 5,
) -> list[dict[str, float]]:
    """Ranks of ``g, [b,g], [b,[b,g]], ...`` after column normalization.

    This distinguishes a rank-deficient instantaneous diffusion (the defining
    hypoelliptic feature) from the number of drift brackets required for the
    finite-time transition to acquire all state-space directions.
    """

    if maximum_depth < 0:
        raise ValueError("maximum_depth must be nonnegative")
    theta = jnp.asarray(unconstrained)
    forcing = jnp.asarray(covariates)

    def drift_field(value):
        return stratonovich_drift(value, theta, forcing)

    def diffusion_field(value):
        return diffusion(value, theta, forcing)

    def bracket(left, right):
        def field(value):
            return (
                jax.jacfwd(right)(value) @ left(value)
                - jax.jacfwd(left)(value) @ right(value)
            )

        return field

    fields = [diffusion_field]
    for _ in range(maximum_depth):
        fields.append(bracket(drift_field, fields[-1]))

    point = jnp.asarray(state)
    columns: list[np.ndarray] = []
    rows: list[dict[str, float]] = []
    for depth, field in enumerate(fields):
        column = np.asarray(field(point), dtype=float)
        columns.append(column)
        matrix = np.stack(columns, axis=1)
        norms = np.linalg.norm(matrix, axis=0)
        normalized = matrix / np.maximum(norms, np.finfo(float).tiny)
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        tolerance = (
            singular_values[0]
            * max(normalized.shape)
            * np.finfo(float).eps
        )
        rows.append(
            {
                "bracket_depth": float(depth),
                "directions": float(depth + 1),
                "rank": float(np.sum(singular_values > tolerance)),
                "smallest_normalized_singular_value": float(
                    singular_values[-1]
                ),
            }
        )
    return rows


def transition_diagnostics(
    state: np.ndarray,
    unconstrained: np.ndarray,
    covariates: np.ndarray,
    dt: float,
    *,
    order: int,
    rank_tolerance: float | None = None,
) -> dict[str, float]:
    """Compute numerical rank and spectral conditioning without a nugget."""

    covariance = np.asarray(
        local_gaussian_covariance(
            jnp.asarray(state),
            jnp.asarray(unconstrained),
            jnp.asarray(covariates),
            dt,
            order=order,
            relative_floor=0.0,
        )
    )
    singular_values = np.linalg.svd(covariance, compute_uv=False)
    tolerance = (
        rank_tolerance
        if rank_tolerance is not None
        else singular_values[0] * max(covariance.shape) * np.finfo(float).eps
    )
    positive = singular_values[singular_values > tolerance]
    condition = float(positive[0] / positive[-1]) if len(positive) else np.inf
    return {
        "rank": float(np.sum(singular_values > tolerance)),
        "condition_nonzero": condition,
        "largest_eigenvalue": float(singular_values[0]),
        "smallest_eigenvalue": float(singular_values[-1]),
        "tolerance": float(tolerance),
    }
