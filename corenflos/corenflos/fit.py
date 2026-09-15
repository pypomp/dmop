"""Projected stochastic optimization of the Corenflos DPF likelihood."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ditlevsen.model import project_parameters

from .dpf import DPFConfig, DPFResult, DaccaArrays, dpf_value_and_grad


ValueAndGradient = Callable[[jax.Array, jax.Array], tuple[DPFResult, jax.Array]]


@dataclass(frozen=True)
class FitResult:
    unconstrained: np.ndarray
    parameter_trace: np.ndarray
    pseudo_loglik_trace: np.ndarray
    gradient_norm_trace: np.ndarray
    minimum_ess_trace: np.ndarray
    median_ess_trace: np.ndarray
    resampling_count_trace: np.ndarray
    maximum_invalid_fraction_trace: np.ndarray
    learning_rate_trace: np.ndarray
    incumbent_pseudo_loglik_trace: np.ndarray
    restart_count_trace: np.ndarray
    elapsed_trace: np.ndarray
    elapsed_seconds: float
    completed_updates: int
    termination_reason: str


def make_value_and_gradient(data: DaccaArrays, config: DPFConfig) -> ValueAndGradient:
    """Compile one full-series DPF value/gradient kernel."""

    return jax.jit(
        lambda parameters, key: dpf_value_and_grad(parameters, key, data, config)
    )


def _global_norm_clip(gradient: jax.Array, limit: float) -> jax.Array:
    norm = jnp.linalg.norm(gradient)
    multiplier = jnp.minimum(1.0, limit / jnp.maximum(norm, 1.0e-300))
    return gradient * multiplier


def fit_dpf(
    initial: np.ndarray,
    evaluator: ValueAndGradient,
    *,
    seed: int,
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    learning_rate_decay: float = 0.0,
    gradient_clip: float = 100.0,
    maximum_acceptable_invalid_fraction: float = 0.5,
    maximum_pseudo_loglik_drop: float = 30.0,
    rollback_patience: int = 10,
    maximum_restarts: int = 6,
    restart_factor: float = 0.5,
    maximum_updates: int = 5000,
    maximum_elapsed_seconds: float = 800.0,
    change_seed: bool = True,
) -> FitResult:
    """Maximize the biased DPF log-likelihood under the manuscript box.

    JIT compilation is expected to be warmed by the caller.  Elapsed time
    includes every value/gradient filter call, including the evaluation of the
    initial bounding-box draw, and excludes later Euler-20 evaluation.
    """

    if optimizer not in {"adam", "sgd"}:
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if maximum_updates < 0:
        raise ValueError("maximum_updates must be nonnegative")
    if maximum_elapsed_seconds <= 0.0:
        raise ValueError("maximum_elapsed_seconds must be positive")
    if not 0.0 <= maximum_acceptable_invalid_fraction <= 1.0:
        raise ValueError("maximum_acceptable_invalid_fraction must be in [0, 1]")
    if maximum_pseudo_loglik_drop <= 0.0:
        raise ValueError("maximum_pseudo_loglik_drop must be positive")
    if rollback_patience < 1:
        raise ValueError("rollback_patience must be positive")
    if maximum_restarts < 0:
        raise ValueError("maximum_restarts must be nonnegative")
    if not 0.0 < restart_factor < 1.0:
        raise ValueError("restart_factor must be in (0, 1)")

    parameters = project_parameters(jnp.asarray(initial, dtype=jnp.float64))
    first_key = jax.random.key(seed)
    moment = jnp.zeros_like(parameters)
    second_moment = jnp.zeros_like(parameters)
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1.0e-8

    parameter_trace: list[np.ndarray] = []
    pseudo_trace: list[float] = []
    gradient_norm_trace: list[float] = []
    minimum_ess_trace: list[float] = []
    median_ess_trace: list[float] = []
    resampling_count_trace: list[int] = []
    invalid_trace: list[float] = []
    rate_trace: list[float] = []
    incumbent_trace: list[float] = []
    restart_trace: list[int] = []
    elapsed_trace: list[float] = []
    termination = "maximum-updates"
    best_parameters: jax.Array | None = None
    best_value = -np.inf
    consecutive_large_drops = 0
    restarts = 0
    rate_multiplier = 1.0
    optimizer_step = 0
    started = time.perf_counter()

    for iteration in range(maximum_updates + 1):
        key = jax.random.fold_in(first_key, iteration) if change_seed else first_key
        result, gradient = evaluator(parameters, key)
        jax.block_until_ready(gradient)
        elapsed = time.perf_counter() - started
        value = float(result.loglik)
        gradient_norm = float(jnp.linalg.norm(gradient))
        parameter_trace.append(np.asarray(parameters))
        pseudo_trace.append(value)
        gradient_norm_trace.append(gradient_norm)
        minimum_ess_trace.append(float(jnp.min(result.ess)))
        median_ess_trace.append(float(jnp.median(result.ess)))
        resampling_count_trace.append(int(jnp.sum(result.resampled)))
        invalid_trace.append(float(jnp.max(result.invalid_fraction)))
        elapsed_trace.append(elapsed)

        acceptable = (
            np.isfinite(value)
            and np.isfinite(gradient_norm)
            and invalid_trace[-1] <= maximum_acceptable_invalid_fraction
        )
        if acceptable and value > best_value:
            best_value = value
            best_parameters = parameters
        incumbent_trace.append(best_value)
        restart_trace.append(restarts)
        if elapsed >= maximum_elapsed_seconds:
            rate_trace.append(0.0)
            termination = "time-budget"
            break
        if iteration == maximum_updates:
            rate_trace.append(0.0)
            break

        if not acceptable:
            rate_trace.append(0.0)
            if best_parameters is None:
                termination = "no-valid-filter-evaluation"
                break
            if restarts >= maximum_restarts:
                termination = "maximum-restarts"
                break
            parameters = best_parameters
            moment = jnp.zeros_like(parameters)
            second_moment = jnp.zeros_like(parameters)
            optimizer_step = 0
            rate_multiplier *= restart_factor
            restarts += 1
            consecutive_large_drops = 0
            continue

        if value < best_value - maximum_pseudo_loglik_drop:
            consecutive_large_drops += 1
        else:
            consecutive_large_drops = 0
        if consecutive_large_drops >= rollback_patience:
            rate_trace.append(0.0)
            if restarts >= maximum_restarts:
                termination = "maximum-restarts"
                break
            parameters = best_parameters
            moment = jnp.zeros_like(parameters)
            second_moment = jnp.zeros_like(parameters)
            optimizer_step = 0
            rate_multiplier *= restart_factor
            restarts += 1
            consecutive_large_drops = 0
            continue

        rate = learning_rate * rate_multiplier / (1.0 + learning_rate_decay * iteration)
        clipped = _global_norm_clip(gradient, gradient_clip)
        optimizer_step += 1
        if optimizer == "adam":
            moment = beta1 * moment + (1.0 - beta1) * clipped
            second_moment = beta2 * second_moment + (1.0 - beta2) * jnp.square(clipped)
            corrected_moment = moment / (1.0 - beta1**optimizer_step)
            corrected_second = second_moment / (1.0 - beta2**optimizer_step)
            increment = (
                rate * corrected_moment / (jnp.sqrt(corrected_second) + adam_epsilon)
            )
        else:
            increment = rate * clipped
        parameters = project_parameters(parameters + increment)
        rate_trace.append(rate)

    return FitResult(
        unconstrained=np.asarray(parameter_trace[-1]),
        parameter_trace=np.asarray(parameter_trace),
        pseudo_loglik_trace=np.asarray(pseudo_trace),
        gradient_norm_trace=np.asarray(gradient_norm_trace),
        minimum_ess_trace=np.asarray(minimum_ess_trace),
        median_ess_trace=np.asarray(median_ess_trace),
        resampling_count_trace=np.asarray(resampling_count_trace),
        maximum_invalid_fraction_trace=np.asarray(invalid_trace),
        learning_rate_trace=np.asarray(rate_trace),
        incumbent_pseudo_loglik_trace=np.asarray(incumbent_trace),
        restart_count_trace=np.asarray(restart_trace),
        elapsed_trace=np.asarray(elapsed_trace),
        elapsed_seconds=float(elapsed_trace[-1]),
        completed_updates=len(parameter_trace) - 1,
        termination_reason=termination,
    )
