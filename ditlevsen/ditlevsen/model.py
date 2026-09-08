"""Reduced, active-coordinate representation of the DMOP Dacca model.

The paper fixes ``c=alpha=1``, ``rho=Y_0=0`` and ``delta=0.02`` in the
benchmark.  Consequently the inapparent-infection compartment Y remains zero.
We remove Y and the diagnostic ``count`` variable, retain the five dynamic
state coordinates and the within-month death accumulator, and work in
population fractions for numerical stability.  The epidemiological model is
used here as a computational-statistics benchmark.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ACTIVE_STATE_NAMES = ("S", "I", "R1", "R2", "R3", "Mn")
POSITIVE_PARAMETER_NAMES = ("gamma", "m", "epsilon", "sigma", "tau")
SPLINE_PARAMETER_NAMES = tuple(f"bs{i}" for i in range(1, 7))
RESERVOIR_PARAMETER_NAMES = tuple(f"omegas{i}" for i in range(1, 7))
INITIAL_PARAMETER_NAMES = ("S_0", "I_0", "R1_0", "R2_0", "R3_0")
ESTIMATED_PARAMETER_NAMES = (
    "gamma",
    "m",
    "epsilon",
    "beta_trend",
    "sigma",
    "tau",
    *SPLINE_PARAMETER_NAMES,
    *RESERVOIR_PARAMETER_NAMES,
    *INITIAL_PARAMETER_NAMES,
)

DEFAULT_PARAMETERS = {
    "gamma": 20.8,
    "m": 0.06,
    "epsilon": 19.1,
    "beta_trend": -0.00498,
    "sigma": 3.13,
    "tau": 0.23,
    **{
        f"bs{i + 1}": value
        for i, value in enumerate((0.747, 6.38, -3.44, 4.23, 3.33, 4.55))
    },
    **{
        f"omegas{i + 1}": value
        for i, value in enumerate(
            np.log((0.184, 0.0786, 0.0584, 0.00917, 0.000208, 0.0124))
        )
    },
    "S_0": 0.621,
    "I_0": 0.378,
    "R1_0": 0.000843,
    "R2_0": 0.000972,
    "R3_0": 1.16e-7,
}


def encode_parameters(parameters: Mapping[str, float]) -> np.ndarray:
    """Map physical parameters to the unconstrained optimization scale."""

    ivp = np.asarray([parameters[name] for name in INITIAL_PARAMETER_NAMES], dtype=float)
    ivp = ivp / ivp.sum()
    values = [
        math.log(parameters["gamma"]),
        math.log(parameters["m"]),
        math.log(parameters["epsilon"]),
        100.0 * parameters["beta_trend"],
        math.log(parameters["sigma"]),
        math.log(parameters["tau"]),
    ]
    values.extend(parameters[name] for name in SPLINE_PARAMETER_NAMES)
    values.extend(parameters[name] for name in RESERVOIR_PARAMETER_NAMES)
    values.extend(np.log(ivp))
    return np.asarray(values, dtype=float)


def default_unconstrained_parameters() -> np.ndarray:
    return encode_parameters(DEFAULT_PARAMETERS)


def decode_parameters(unconstrained: jax.Array) -> dict[str, jax.Array]:
    """Map the 23-vector used by the benchmark to physical parameters."""

    u = jnp.asarray(unconstrained, dtype=jnp.float64)
    if u.shape != (len(ESTIMATED_PARAMETER_NAMES),):
        raise ValueError(
            f"expected shape {(len(ESTIMATED_PARAMETER_NAMES),)}, got {u.shape}"
        )
    ivp = jax.nn.softmax(u[18:23])
    return {
        "gamma": jnp.exp(u[0]),
        "m": jnp.exp(u[1]),
        "epsilon": jnp.exp(u[2]),
        "beta_trend": u[3] / 100.0,
        "sigma": jnp.exp(u[4]),
        "tau": jnp.exp(u[5]),
        "bs": u[6:12],
        "omegas": u[12:18],
        "ivp": ivp,
        # Values fixed in the paper's global-search benchmark.
        "rho": jnp.asarray(0.0),
        "c": jnp.asarray(1.0),
        "alpha": jnp.asarray(1.0),
        "delta": jnp.asarray(0.02),
    }


# Bounds reproduce the global-search box in dmop/code/global_search/prep.py.
_LOWER = np.asarray(
    [
        np.log(10.0),
        np.log(0.03),
        np.log(0.20),
        -1.0,
        np.log(1.0),
        np.log(0.10),
        -4.0,
        0.0,
        -4.0,
        0.0,
        0.0,
        0.0,
        *([-10.0] * 6),
        *([-20.0] * 5),
    ],
    dtype=float,
)
_UPPER = np.asarray(
    [
        np.log(40.0),
        np.log(0.60),
        np.log(30.0),
        0.0,
        np.log(5.0),
        np.log(0.50),
        4.0,
        8.0,
        4.0,
        8.0,
        8.0,
        8.0,
        *([0.0] * 6),
        *([0.0] * 5),
    ],
    dtype=float,
)


def parameter_bounds() -> tuple[jax.Array, jax.Array]:
    return jnp.asarray(_LOWER), jnp.asarray(_UPPER)


def project_parameters(unconstrained: jax.Array) -> jax.Array:
    """Project to the paper's search box and remove the IVP softmax null shift."""

    lower, upper = parameter_bounds()
    projected = jnp.clip(unconstrained, lower, upper)
    ivp = projected[18:23]
    ivp = ivp - jnp.max(ivp)
    return projected.at[18:23].set(ivp)


def initial_state(unconstrained: jax.Array) -> jax.Array:
    """Initial active state in population-fraction coordinates."""

    theta = decode_parameters(unconstrained)
    return jnp.concatenate((theta["ivp"], jnp.zeros((1,), dtype=jnp.float64)))


def drift(
    state: jax.Array, unconstrained: jax.Array, covariates: jax.Array
) -> jax.Array:
    """Itô drift for ``(S,I,R1,R2,R3,Mn) / population``.

    Covariates are ``(trend, dpopdt, pop, seas1, ..., seas6)`` and are frozen
    over one local transition, matching the Euler convention used by Pypomp.
    """

    theta = decode_parameters(unconstrained)
    s, i, r1, r2, r3, mn = state
    trend, dpopdt, pop = covariates[:3]
    seas = covariates[3:]
    relative_population_growth = dpopdt / pop
    beta = jnp.exp(theta["beta_trend"] * trend + jnp.dot(seas, theta["bs"]))
    omega = jnp.exp(jnp.dot(seas, theta["omegas"]))
    infection = (omega + beta * i) * s
    waning_rate = 3.0 * theta["epsilon"]
    common_loss = theta["delta"] + relative_population_growth

    return jnp.asarray(
        [
            relative_population_growth
            + theta["delta"]
            - infection
            - theta["delta"] * s
            + waning_rate * r3
            - relative_population_growth * s,
            infection
            - (theta["m"] + theta["delta"] + theta["gamma"]) * i
            - relative_population_growth * i,
            theta["gamma"] * i - (waning_rate + common_loss) * r1,
            waning_rate * r1 - (waning_rate + common_loss) * r2,
            waning_rate * r2 - (waning_rate + common_loss) * r3,
            theta["m"] * i - relative_population_growth * mn,
        ],
        dtype=jnp.float64,
    )


def diffusion(
    state: jax.Array, unconstrained: jax.Array, covariates: jax.Array
) -> jax.Array:
    """The single Brownian vector field on the active fractional state."""

    del covariates
    theta = decode_parameters(unconstrained)
    s, i = state[:2]
    magnitude = theta["sigma"] * s * i
    return magnitude * jnp.asarray((-1.0, 1.0, 0.0, 0.0, 0.0, 0.0))


def physical_parameter_dict(unconstrained: np.ndarray | jax.Array) -> dict[str, float]:
    """Convert an optimizer vector to a Pypomp-compatible parameter dictionary."""

    decoded = decode_parameters(jnp.asarray(unconstrained))
    result = {
        "gamma": float(decoded["gamma"]),
        "m": float(decoded["m"]),
        "rho": 0.0,
        "epsilon": float(decoded["epsilon"]),
        "c": 1.0,
        "beta_trend": float(decoded["beta_trend"]),
        "sigma": float(decoded["sigma"]),
        "tau": float(decoded["tau"]),
        "alpha": 1.0,
        "delta": 0.02,
        "Y_0": 0.0,
    }
    result.update(
        {f"bs{i + 1}": float(decoded["bs"][i]) for i in range(6)}
    )
    result.update(
        {f"omegas{i + 1}": float(decoded["omegas"][i]) for i in range(6)}
    )
    result.update(
        {
            name: float(decoded["ivp"][index])
            for index, name in enumerate(INITIAL_PARAMETER_NAMES)
        }
    )
    return result
