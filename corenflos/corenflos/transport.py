"""Entropy-regularized optimal-transport resampling in JAX.

The implementation follows ``filterflow`` rather than using a generic OT
solver.  In particular, it retains epsilon annealing, averaged Sinkhorn
updates, the final differentiable fixed-point update (gradient stitching),
and the clipped transport adjoint used by Corenflos et al.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class TransportConfig:
    """Numerical controls matching the FilterFlow defaults/examples."""

    epsilon: float = 0.5
    scaling: float = 0.75
    threshold: float = 1.0e-3
    max_iterations: int = 100


def _cost(x: jax.Array, y: jax.Array) -> jax.Array:
    """Pairwise half squared Euclidean distances."""

    xx = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    yy = jnp.sum(jnp.square(y), axis=-1)[None, :]
    return jnp.maximum(xx - 2.0 * (x @ y.T) + yy, 0.0) / 2.0


def _softmin(epsilon: jax.Array, cost: jax.Array, values: jax.Array) -> jax.Array:
    return -epsilon * jax.scipy.special.logsumexp(
        values[None, :] - cost / epsilon, axis=1
    )


def _scaled_particles(particles: jax.Array) -> jax.Array:
    centered = particles - jax.lax.stop_gradient(jnp.mean(particles, axis=0))
    diameter = jnp.max(jnp.std(particles, axis=0))
    diameter = jnp.where(diameter > 0.0, diameter, 1.0)
    scale = jax.lax.stop_gradient(diameter * jnp.sqrt(particles.shape[-1]))
    return centered / scale


def _solve_potentials(
    log_weights: jax.Array,
    x: jax.Array,
    epsilon: float,
    scaling: float,
    threshold: float,
    max_iterations: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the annealed, averaged Sinkhorn fixed-point iterations.

    The converged values are intentionally detached below.  One additional
    update at the target epsilon supplies the implicit/fixed-point gradient,
    as in the original TensorFlow implementation.
    """

    count = x.shape[0]
    uniform = jnp.full_like(log_weights, -jnp.log(float(count)))
    x_left = x
    x_right = jax.lax.stop_gradient(x)
    cost_xy = _cost(x_left, x_right)
    cost_yx = _cost(x_left, jax.lax.stop_gradient(x_left))

    spread = jnp.max(x) - jnp.min(x)
    running_epsilon = jnp.maximum(jnp.square(spread), epsilon)
    a = _softmin(running_epsilon, cost_yx, log_weights)
    b = _softmin(running_epsilon, cost_xy, uniform)
    active = jnp.asarray(True)
    total = jnp.asarray(0, dtype=jnp.int32)

    def body(_, carry):
        a_old, b_old, eps_old, active_old, iterations = carry
        at = _softmin(eps_old, cost_yx, log_weights + b_old / eps_old)
        bt = _softmin(eps_old, cost_xy, uniform + a_old / eps_old)
        a_candidate = 0.5 * (a_old + at)
        b_candidate = 0.5 * (b_old + bt)
        update = jnp.maximum(
            jnp.max(jnp.abs(a_candidate - a_old)),
            jnp.max(jnp.abs(b_candidate - b_old)),
        )
        next_epsilon = jnp.maximum(eps_old * scaling**2, epsilon)
        continue_local = update > threshold
        continue_annealing = next_epsilon < eps_old
        next_active = active_old & (continue_local | continue_annealing)
        a_new = jnp.where(active_old, a_candidate, a_old)
        b_new = jnp.where(active_old, b_candidate, b_old)
        eps_new = jnp.where(active_old, next_epsilon, eps_old)
        iterations_new = iterations + active_old.astype(jnp.int32)
        return a_new, b_new, eps_new, next_active, iterations_new

    # Fixed iteration bounds compile much faster on the GPU.  Once converged,
    # the active mask freezes the potentials, so the numerical result is the
    # same as terminating the loop early.
    a, b, _, _, total = jax.lax.fori_loop(
        0,
        max(0, max_iterations - 1),
        body,
        (a, b, running_epsilon, active, total),
    )
    a = jax.lax.stop_gradient(a)
    b = jax.lax.stop_gradient(b)
    target = jnp.asarray(epsilon, dtype=x.dtype)
    # FilterFlow uses a full final update rather than averaging this one.
    final_a = _softmin(target, cost_yx, log_weights + b / target)
    final_b = _softmin(target, cost_xy, uniform + a / target)
    return final_a, final_b, total + 2


def _transport_matrix_impl(
    particles: jax.Array,
    log_weights: jax.Array,
    epsilon: float,
    scaling: float,
    threshold: float,
    max_iterations: int,
) -> jax.Array:
    count = particles.shape[0]
    scaled = _scaled_particles(particles)
    a, b, _ = _solve_potentials(
        log_weights,
        scaled,
        epsilon,
        scaling,
        threshold,
        max_iterations,
    )
    logits = (a[:, None] + b[None, :] - _cost(scaled, scaled)) / epsilon
    # Exact column normalization guarantees column sums J*w.  Sinkhorn makes
    # the row sums approximately one.
    logits = (
        logits
        - jax.scipy.special.logsumexp(logits, axis=0, keepdims=True)
        + jnp.log(float(count))
        + log_weights[None, :]
    )
    return jnp.exp(logits)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def transport_matrix(
    particles: jax.Array,
    log_weights: jax.Array,
    epsilon: float = 0.5,
    scaling: float = 0.75,
    threshold: float = 1.0e-3,
    max_iterations: int = 100,
) -> jax.Array:
    """Return the FilterFlow transport matrix.

    Rows index equally weighted output particles and columns index weighted
    input particles.  Therefore ``T @ particles`` is the resampled ensemble.
    """

    return _transport_matrix_impl(
        particles, log_weights, epsilon, scaling, threshold, max_iterations
    )


def _transport_fwd(
    particles: jax.Array,
    log_weights: jax.Array,
    epsilon: float,
    scaling: float,
    threshold: float,
    max_iterations: int,
):
    result = _transport_matrix_impl(
        particles, log_weights, epsilon, scaling, threshold, max_iterations
    )
    return result, (particles, log_weights)


def _transport_bwd(
    epsilon: float,
    scaling: float,
    threshold: float,
    max_iterations: int,
    residual,
    cotangent: jax.Array,
):
    particles, log_weights = residual
    _, pullback = jax.vjp(
        lambda x, w: _transport_matrix_impl(
            x, w, epsilon, scaling, threshold, max_iterations
        ),
        particles,
        log_weights,
    )
    return pullback(jnp.clip(cotangent, -1.0, 1.0))


transport_matrix.defvjp(_transport_fwd, _transport_bwd)


def ensemble_transform(
    particles: jax.Array,
    log_weights: jax.Array,
    config: TransportConfig = TransportConfig(),
) -> jax.Array:
    """Map a weighted ensemble to an approximately equally weighted one."""

    matrix = transport_matrix(
        particles,
        log_weights,
        config.epsilon,
        config.scaling,
        config.threshold,
        config.max_iterations,
    )
    return matrix @ particles


def marginal_errors(
    matrix: jax.Array, log_weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Maximum row- and column-marginal errors, for tests/diagnostics."""

    count = matrix.shape[0]
    row_error = jnp.max(jnp.abs(jnp.sum(matrix, axis=1) - 1.0))
    column_error = jnp.max(
        jnp.abs(jnp.sum(matrix, axis=0) - count * jnp.exp(log_weights))
    )
    return row_error, column_error
