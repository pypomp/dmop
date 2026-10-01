"""Fine-grid backward-sampling and Karppinen block-lookahead diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .data import DaccaData
from .model import initial_state
from .transition import ditlevsen_mean, gaussian_transition

jax.config.update("jax_enable_x64", True)


def _linear_block_moments(
    state: jax.Array,
    unconstrained: jax.Array,
    covariates: jax.Array,
    dt: float,
    nstep: int,
    *,
    order: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One-step and fixed-physical-time local linear Gaussian moments."""

    one_mean, one_covariance = gaussian_transition(
        state,
        unconstrained,
        covariates,
        dt,
        order=order,
        relative_floor=1e-12,
    )
    transition = jax.jacfwd(ditlevsen_mean, argnums=0)(
        state, unconstrained, covariates, dt
    )
    block_transition = jnp.eye(state.shape[0], dtype=jnp.float64)
    block_covariance = jnp.zeros_like(one_covariance)
    block_mean = state
    for _ in range(nstep):
        block_mean = one_mean + transition @ (block_mean - state)
        block_covariance = (
            transition @ block_covariance @ transition.T + one_covariance
        )
        block_transition = transition @ block_transition
    return (
        one_mean,
        one_covariance,
        block_mean,
        block_transition,
        block_covariance,
    )


def _logpdf_batch(
    value: np.ndarray, means: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues[-1]) * 1e-12, 1e-18)
    eigenvalues = np.maximum(eigenvalues, floor)
    residual = value[None, :] - means
    whitened = (residual @ eigenvectors) / np.sqrt(eigenvalues)[None, :]
    return -0.5 * np.sum(whitened**2, axis=1)


def _ess(log_weights: np.ndarray) -> float:
    weights = np.exp(log_weights - np.max(log_weights))
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights**2))


def backward_ess_experiment(
    data: DaccaData,
    unconstrained: np.ndarray,
    *,
    particles: int = 128,
    replicates: int = 100,
    seed: int = 631409,
    order: int = 6,
    parent_spread: float = 0.05,
) -> dict[str, float]:
    """Compare unit-step BS with a one-month Karppinen-style lookahead weight.

    Parent candidates have a fixed fraction ``parent_spread`` of the one-month
    process standard deviation, held fixed as the numerical grid changes.  The ordinary
    backward kernel evaluates a one-substep density; the block kernel evaluates
    the fixed one-month endpoint density.  This is an ancestor-weight ESS proxy
    for the degeneracy mechanism that Karppinen et al. target, not a claim that
    the full CPF-BBS Markov kernel has been run.
    """

    rng = np.random.default_rng(seed)
    state = initial_state(jnp.asarray(unconstrained))
    covariates = jnp.asarray(data.initial_covariates)
    dt = 1.0 / (12.0 * data.nstep)
    one_mean, one_covariance, block_mean, block_transition, block_covariance = (
        _linear_block_moments(
            state,
            jnp.asarray(unconstrained),
            covariates,
            dt,
            data.nstep,
            order=order,
        )
    )
    state_np = np.asarray(state)
    one_mean_np = np.asarray(one_mean)
    one_covariance_np = np.asarray(one_covariance)
    transition_np = np.asarray(
        jax.jacfwd(ditlevsen_mean, argnums=0)(
            state, jnp.asarray(unconstrained), covariates, dt
        )
    )
    block_mean_np = np.asarray(block_mean)
    block_transition_np = np.asarray(block_transition)
    block_covariance_np = np.asarray(block_covariance)

    one_cholesky = np.linalg.cholesky(
        one_covariance_np + 1e-15 * np.eye(state_np.size)
    )
    block_cholesky = np.linalg.cholesky(
        block_covariance_np + 1e-15 * np.eye(state_np.size)
    )
    unit_ess = []
    bridge_ess = []
    unit_reference_probability = []
    bridge_reference_probability = []

    for _ in range(replicates):
        offsets = (
            parent_spread
            * rng.normal(size=(particles, state_np.size))
            @ block_cholesky.T
        )
        offsets[0] = 0.0
        parents = state_np[None, :] + offsets

        one_child = one_mean_np + one_cholesky @ rng.normal(size=state_np.size)
        one_means = one_mean_np[None, :] + offsets @ transition_np.T
        one_log_weights = _logpdf_batch(one_child, one_means, one_covariance_np)
        unit_ess.append(_ess(one_log_weights))
        one_weights = np.exp(one_log_weights - np.max(one_log_weights))
        one_weights /= one_weights.sum()
        unit_reference_probability.append(float(one_weights[0]))

        block_child = block_mean_np + block_cholesky @ rng.normal(
            size=state_np.size
        )
        block_means = block_mean_np[None, :] + offsets @ block_transition_np.T
        block_log_weights = _logpdf_batch(
            block_child, block_means, block_covariance_np
        )
        bridge_ess.append(_ess(block_log_weights))
        block_weights = np.exp(block_log_weights - np.max(block_log_weights))
        block_weights /= block_weights.sum()
        bridge_reference_probability.append(float(block_weights[0]))

    return {
        "nstep": float(data.nstep),
        "unit_backward_ess_mean": float(np.mean(unit_ess)),
        "unit_backward_ess_sd": float(np.std(unit_ess, ddof=1)),
        "bridge_backward_ess_mean": float(np.mean(bridge_ess)),
        "bridge_backward_ess_sd": float(np.std(bridge_ess, ddof=1)),
        "unit_reference_probability_mean": float(
            np.mean(unit_reference_probability)
        ),
        "bridge_reference_probability_mean": float(
            np.mean(bridge_reference_probability)
        ),
    }
