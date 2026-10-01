from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from corenflos.transport import ensemble_transform, marginal_errors, transport_matrix

jax.config.update("jax_enable_x64", True)


def test_transport_has_expected_marginals_and_weighted_mean() -> None:
    particles = jnp.asarray(
        [[-1.0, 0.2], [-0.1, 0.0], [0.4, 1.2], [2.0, -0.5]],
        dtype=jnp.float64,
    )
    weights = jnp.asarray([0.05, 0.15, 0.30, 0.50], dtype=jnp.float64)
    log_weights = jnp.log(weights)
    matrix = transport_matrix(particles, log_weights)
    row_error, column_error = marginal_errors(matrix, log_weights)
    transformed = matrix @ particles

    assert float(row_error) < 8.0e-3
    assert float(column_error) < 1.0e-12
    np.testing.assert_allclose(
        np.asarray(jnp.mean(transformed, axis=0)),
        np.asarray(jnp.sum(weights[:, None] * particles, axis=0)),
        atol=1.0e-12,
    )


def test_uniform_weights_give_finite_smooth_transform() -> None:
    particles = jnp.asarray([[-1.0], [-0.4], [0.2], [1.4]], dtype=jnp.float64)
    log_weights = jnp.full((4,), -jnp.log(4.0), dtype=jnp.float64)
    transformed = ensemble_transform(particles, log_weights)
    assert transformed.shape == particles.shape
    assert bool(jnp.all(jnp.isfinite(transformed)))
    np.testing.assert_allclose(
        np.asarray(jnp.mean(transformed, axis=0)),
        np.asarray(jnp.mean(particles, axis=0)),
        atol=1.0e-12,
    )


def test_transport_custom_gradient_is_finite() -> None:
    particles = jnp.asarray(
        [[-0.8, 0.1], [-0.1, 0.4], [0.6, -0.2], [1.3, 0.7]],
        dtype=jnp.float64,
    )
    logits = jnp.asarray([-1.0, 0.2, 0.7, -0.3], dtype=jnp.float64)

    def loss(x, raw_weights):
        log_weights = jax.nn.log_softmax(raw_weights)
        result = ensemble_transform(x, log_weights)
        return jnp.mean(jnp.square(result))

    grad_particles, grad_logits = jax.grad(loss, argnums=(0, 1))(particles, logits)
    assert bool(jnp.all(jnp.isfinite(grad_particles)))
    assert bool(jnp.all(jnp.isfinite(grad_logits)))
    assert float(jnp.linalg.norm(grad_particles)) > 0.0
    assert float(jnp.linalg.norm(grad_logits)) > 0.0
