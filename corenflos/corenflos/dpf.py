"""Corenflos differentiable particle filter for the Dacca model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ditlevsen.data import DaccaData
from ditlevsen.model import decode_parameters

from .transport import TransportConfig, transport_matrix

jax.config.update("jax_enable_x64", True)


class DaccaArrays(NamedTuple):
    """JAX-array version of the Dacca observations and forcings."""

    observations: jax.Array
    step_covariates: jax.Array
    initial_covariates: jax.Array
    nstep: int

    @classmethod
    def from_data(cls, data: DaccaData) -> "DaccaArrays":
        return cls(
            observations=jnp.asarray(data.observations, dtype=jnp.float64),
            step_covariates=jnp.asarray(data.step_covariates, dtype=jnp.float64),
            initial_covariates=jnp.asarray(data.initial_covariates, dtype=jnp.float64),
            nstep=int(data.nstep),
        )


@dataclass(frozen=True)
class DPFConfig:
    particles: int = 100
    ess_threshold: float = 0.5
    transport: TransportConfig = TransportConfig()


class DPFResult(NamedTuple):
    loglik: jax.Array
    increments: jax.Array
    ess: jax.Array
    resampled: jax.Array
    invalid_fraction: jax.Array


def _initial_particles(
    unconstrained: jax.Array, initial_covariates: jax.Array, count: int
) -> tuple[jax.Array, jax.Array]:
    theta = decode_parameters(unconstrained)
    population = initial_covariates[2]
    # Active order matches the update below: S, I, Mn, R1, R2, R3.
    state = population * jnp.asarray(
        [
            theta["ivp"][0],
            theta["ivp"][1],
            0.0,
            theta["ivp"][2],
            theta["ivp"][3],
            theta["ivp"][4],
        ],
        dtype=jnp.float64,
    )
    state = state.at[2].set(0.0)
    return jnp.broadcast_to(state, (count, 6)), jnp.zeros((count,), dtype=bool)


def _euler_step(
    state: jax.Array,
    invalid: jax.Array,
    normal: jax.Array,
    covariates: jax.Array,
    unconstrained: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Exact Gaussian Euler recurrence used by ``pypomp.models.dacca``."""

    theta = decode_parameters(unconstrained)
    s, i, deaths, r1, r2, r3 = (state[:, k] for k in range(6))
    trend, dpopdt, population = covariates[:3]
    seasonal = covariates[3:]

    waning_rate = 3.0 * theta["epsilon"]
    pass0 = theta["gamma"] * i
    pass1 = waning_rate * r1
    pass2 = waning_rate * r2
    pass3 = waning_rate * r3
    beta = jnp.exp(theta["beta_trend"] * trend + jnp.dot(seasonal, theta["bs"]))
    omega = jnp.exp(jnp.dot(seasonal, theta["omegas"]))
    dw = normal * jnp.sqrt(dt)
    births = dpopdt + theta["delta"] * population
    infections = (omega + (beta + theta["sigma"] * dw / dt) * (i / population)) * s
    disease = theta["m"] * i

    next_s = s + (births - infections - theta["delta"] * s + pass3) * dt
    next_i = i + (infections - disease - theta["delta"] * i - pass0) * dt
    next_r1 = r1 + (pass0 - pass1 - theta["delta"] * r1) * dt
    next_r2 = r2 + (pass1 - pass2 - theta["delta"] * r2) * dt
    next_r3 = r3 + (pass2 - pass3 - theta["delta"] * r3) * dt
    next_deaths = deaths + disease * dt
    next_state = jnp.stack(
        (next_s, next_i, next_deaths, next_r1, next_r2, next_r3), axis=1
    )
    newly_invalid = jnp.any(next_state < 0.0, axis=1) | ~jnp.all(
        jnp.isfinite(next_state), axis=1
    )
    return jnp.maximum(next_state, 0.0), invalid | newly_invalid


def _measurement_log_density(
    observation: jax.Array,
    state: jax.Array,
    invalid: jax.Array,
    unconstrained: jax.Array,
) -> jax.Array:
    theta = decode_parameters(unconstrained)
    mean = state[:, 2]
    scale = theta["tau"] * mean
    # Pypomp adds this tolerance to the Normal scale before applying the same
    # 1e-18 density floor.
    safe_scale = scale + 1.0e-18
    standardized = (observation - mean) / safe_scale
    normal_log_density = (
        -0.5 * jnp.square(standardized)
        - jnp.log(safe_scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    # This is algebraically the same finite floor as the Pypomp rmeasure.
    floored = jnp.logaddexp(normal_log_density, jnp.log(1.0e-18))
    return jnp.where(invalid, jnp.log(1.0e-18), floored)


def dacca_dpf(
    unconstrained: jax.Array,
    key: jax.Array,
    data: DaccaArrays,
    config: DPFConfig = DPFConfig(),
) -> DPFResult:
    """Run the biased differentiable particle filter.

    OT resampling occurs after a measurement when ESS/J is below the chosen
    threshold.  The invalid-state flag is deliberately not transported: it is
    an auxiliary diagnostic in Pypomp, and mixing it through the dense
    entropic map would mark every output invalid if one negligible input were
    invalid.  Resampling resets it, just as multinomial resampling eliminates
    unselected negligible invalid particles.
    """

    particles, invalid = _initial_particles(
        unconstrained, data.initial_covariates, config.particles
    )
    log_weights = jnp.full(
        (config.particles,), -jnp.log(float(config.particles)), dtype=jnp.float64
    )
    dt = 1.0 / (12.0 * data.nstep)
    keys = jax.random.split(key, data.observations.shape[0])

    def one_month(carry, inputs):
        current_particles, current_invalid, previous_log_weights = carry
        observation, month_covariates, month_key = inputs
        # Mn is an accumulator and Pypomp resets it at each observation time.
        current_particles = current_particles.at[:, 2].set(0.0)
        step_keys = jax.random.split(month_key, data.nstep)
        normals = jax.vmap(
            lambda step_key: jax.random.normal(
                step_key, (config.particles,), dtype=jnp.float64
            )
        )(step_keys)

        def process_step(process_carry, process_input):
            process_particles, process_invalid = process_carry
            covariates, normal = process_input
            return (
                _euler_step(
                    process_particles,
                    process_invalid,
                    normal,
                    covariates,
                    unconstrained,
                    dt,
                ),
                None,
            )

        (propagated, propagated_invalid), _ = jax.lax.scan(
            process_step,
            (current_particles, current_invalid),
            (month_covariates, normals),
        )
        measurement = _measurement_log_density(
            observation, propagated, propagated_invalid, unconstrained
        )
        unnormalized = previous_log_weights + measurement
        increment = jax.scipy.special.logsumexp(unnormalized)
        normalized = unnormalized - increment
        ess = jnp.exp(-jax.scipy.special.logsumexp(2.0 * normalized))
        do_resample = ess < config.ess_threshold * config.particles

        def resample(_):
            # FilterFlow constructs the coupling on the complete particle
            # state.  Here that is S, I, Mn, R1, R2, R3; only Pypomp's
            # diagnostic invalid-state flag is kept outside the transport.
            matrix = transport_matrix(
                propagated,
                normalized,
                config.transport.epsilon,
                config.transport.scaling,
                config.transport.threshold,
                config.transport.max_iterations,
            )
            transformed = matrix @ propagated
            return (
                transformed,
                jnp.zeros_like(propagated_invalid),
                jnp.full_like(normalized, -jnp.log(float(config.particles))),
            )

        def retain(_):
            return propagated, propagated_invalid, normalized

        next_carry = jax.lax.cond(do_resample, resample, retain, operand=None)
        diagnostics = (
            increment,
            ess,
            do_resample,
            jnp.mean(propagated_invalid.astype(jnp.float64)),
        )
        return next_carry, diagnostics

    # Rematerialization avoids retaining one JxJ OT graph per month.  The
    # backward pass recomputes monthly work instead of exhausting GPU memory.
    scan_step = jax.checkpoint(one_month)
    _, (increments, ess, resampled, invalid_fraction) = jax.lax.scan(
        scan_step,
        (particles, invalid, log_weights),
        (data.observations, data.step_covariates, keys),
    )
    return DPFResult(
        loglik=jnp.sum(increments),
        increments=increments,
        ess=ess,
        resampled=resampled,
        invalid_fraction=invalid_fraction,
    )


def dpf_value_and_grad(
    unconstrained: jax.Array,
    key: jax.Array,
    data: DaccaArrays,
    config: DPFConfig = DPFConfig(),
) -> tuple[DPFResult, jax.Array]:
    """Return filter output and the gradient of its pseudo-log-likelihood."""

    def objective(parameters):
        result = dacca_dpf(parameters, key, data, config)
        return result.loglik, result

    (_, result), gradient = jax.value_and_grad(objective, has_aux=True)(unconstrained)
    return result, gradient
