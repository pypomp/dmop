"""Linear-Gaussian block transitions and bridge conditionals.

These are the tractable multi-step objects required by Karppinen, Singh, and
Vihola (2024).  They apply to an auxiliary proposal

    X_k = F_k X_{k-1} + c_k + epsilon_k,  epsilon_k ~ N(0, Q_k),

not directly to the nonlinear Dacca transition.  A conditional particle
method must retain the nonlinear/auxiliary density ratio in its potentials,
as derived in ``method.tex``.
"""

from __future__ import annotations

import numpy as np


def _as_transition_arrays(
    transitions: np.ndarray,
    offsets: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transitions = np.asarray(transitions, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if transitions.ndim != 3 or transitions.shape[1] != transitions.shape[2]:
        raise ValueError("transitions must have shape (steps, state, state)")
    steps, state_dimension, _ = transitions.shape
    if offsets.shape != (steps, state_dimension):
        raise ValueError("offsets must have shape (steps, state)")
    if covariances.shape != (steps, state_dimension, state_dimension):
        raise ValueError("covariances must have shape (steps, state, state)")
    if steps < 1:
        raise ValueError("at least one transition is required")
    return transitions, offsets, covariances


def block_transition(
    transitions: np.ndarray,
    offsets: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(A, b, Q)`` with block endpoint law ``N(A x+b, Q)``."""

    transitions, offsets, covariances = _as_transition_arrays(
        transitions, offsets, covariances
    )
    state_dimension = transitions.shape[1]
    block_matrix = np.eye(state_dimension)
    block_offset = np.zeros(state_dimension)
    block_covariance = np.zeros((state_dimension, state_dimension))
    for transition, offset, covariance in zip(transitions, offsets, covariances):
        block_offset = transition @ block_offset + offset
        block_covariance = (
            transition @ block_covariance @ transition.T + covariance
        )
        block_matrix = transition @ block_matrix
    block_covariance = 0.5 * (block_covariance + block_covariance.T)
    return block_matrix, block_offset, block_covariance


def bridge_conditional(
    previous_state: np.ndarray,
    endpoint: np.ndarray,
    transitions: np.ndarray,
    offsets: np.ndarray,
    covariances: np.ndarray,
    *,
    relative_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the next-state law conditional on a fixed block endpoint.

    ``transitions[0]`` advances from ``previous_state`` to the state to be
    sampled.  The remaining transitions advance that state to ``endpoint``.
    """

    transitions, offsets, covariances = _as_transition_arrays(
        transitions, offsets, covariances
    )
    previous_state = np.asarray(previous_state, dtype=float)
    endpoint = np.asarray(endpoint, dtype=float)
    state_dimension = transitions.shape[1]
    if previous_state.shape != (state_dimension,) or endpoint.shape != (
        state_dimension,
    ):
        raise ValueError("states must have shape (state,)")

    next_mean = transitions[0] @ previous_state + offsets[0]
    next_covariance = covariances[0]
    if transitions.shape[0] == 1:
        return endpoint.copy(), np.zeros_like(next_covariance)

    remainder_matrix, remainder_offset, remainder_covariance = block_transition(
        transitions[1:], offsets[1:], covariances[1:]
    )
    endpoint_mean = remainder_matrix @ next_mean + remainder_offset
    cross_covariance = next_covariance @ remainder_matrix.T
    endpoint_covariance = (
        remainder_matrix @ next_covariance @ remainder_matrix.T
        + remainder_covariance
    )
    endpoint_covariance = 0.5 * (endpoint_covariance + endpoint_covariance.T)
    scale = max(float(np.max(np.diag(endpoint_covariance))), 1e-20)
    stabilized = endpoint_covariance + relative_floor * scale * np.eye(
        state_dimension
    )
    gain = np.linalg.solve(stabilized, cross_covariance.T).T
    conditional_mean = next_mean + gain @ (endpoint - endpoint_mean)
    conditional_covariance = next_covariance - gain @ cross_covariance.T
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    return conditional_mean, conditional_covariance
