"""Small benchmark models and analytic likelihoods; SPX comes from Pypomp."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypomp as pp

from ditlevsen.validation_ho import simulate_ho

jax.config.update("jax_enable_x64", True)


def ho_moments(theta, dt=0.02):
    D, gamma, sigma = theta["D"], theta["gamma"], theta["sigma"]
    A = jnp.array([[0., 1.], [-D, -gamma]])
    F = jnp.eye(2) + dt * A + 0.5 * dt**2 * A @ A
    Q = sigma**2 * jnp.array([
        [dt**3 / 3, dt**2 / 2 - gamma * dt**3 / 3],
        [dt**2 / 2 - gamma * dt**3 / 3,
         dt - gamma * dt**2 + gamma**2 * dt**3 / 3]])
    return F, Q


def _log_to(theta):
    return {k: jnp.log(v) for k, v in theta.items()}


def _log_from(theta):
    return {k: jnp.exp(v) for k, v in theta.items()}


@lru_cache(maxsize=8)
def oscillator(seed=631410, points=1000):
    """Exact simulated position; conditional strong-1.5 velocity proposal.

    The auxiliary state `log_weight` holds p(v_n | v_{n-1}, u_{n-1}).
    Its product is the correct importance weight for sampling
    u_n | v_n, v_{n-1}, u_{n-1}; no noisy observation is introduced.
    """
    observed = simulate_ho(points=points, seed=seed)[:, 0]
    v = jnp.asarray(observed)

    def rinit(theta_, key, covars, t0):
        return {"U": jax.random.normal(key) * theta_["sigma"]
                / jnp.sqrt(2 * theta_["gamma"]), "log_weight": jnp.array(0.)}

    def rproc(X_, theta_, key, covars, t, dt):
        i = jnp.rint(t).astype(int)
        F, Q = ho_moments(theta_)
        prediction = F @ jnp.array([v[i], X_["U"]])
        residual = v[i + 1] - prediction[0]
        mean = prediction[1] + Q[1, 0] / Q[0, 0] * residual
        variance = Q[1, 1] - Q[1, 0]**2 / Q[0, 0]
        return {"U": mean + jnp.sqrt(variance) * jax.random.normal(key),
                "log_weight": jax.scipy.stats.norm.logpdf(
                    v[i + 1], prediction[0], jnp.sqrt(Q[0, 0]))}

    def dmeas(Y_, X_, theta_, covars, t):
        return X_["log_weight"]

    model = pp.Pomp(
        ys=pd.DataFrame({"V": observed[1:]}, index=np.arange(1, points, dtype=float)),
        theta=pp.PompParameters({"D": 4., "gamma": .5, "sigma": .5}),
        t0=0., nstep=1, rinit=rinit, rproc=rproc, dmeas=dmeas,
        statenames=["U", "log_weight"],
        par_trans=pp.ParTrans(to_est=_log_to, from_est=_log_from))
    return model, observed


def oscillator_loglik(theta, observed):
    """Kalman likelihood of the fitted approximation, conditional on V_0."""
    F, Q = ho_moments(theta)
    m = jnp.array([observed[0], 0.])
    C = jnp.diag(jnp.array([0., theta["sigma"]**2 / (2 * theta["gamma"])]))

    def step(carry, y):
        mean, covariance = carry
        mean = F @ mean
        covariance = F @ covariance @ F.T + Q
        variance = covariance[0, 0]
        weight = jax.scipy.stats.norm.logpdf(y, mean[0], jnp.sqrt(variance))
        gain = covariance[:, 0] / variance
        mean = mean + gain * (y - mean[0])
        covariance = covariance - jnp.outer(gain, covariance[0])
        return (mean, covariance), weight

    _, ll = jax.lax.scan(step, (m, C), jnp.asarray(observed[1:]))
    return jnp.sum(ll)


@lru_cache(maxsize=8)
def linear_gaussian(seed=631420, points=150):
    """CTDD21 E.1 dynamics; the declared initial law is X_0 ~ N(0,I)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=2)
    observed = []
    for _ in range(points):
        x = .5 * x + np.sqrt(.5) * rng.normal(size=2)
        observed.append(x + np.sqrt(.1) * rng.normal(size=2))
    observed = np.array(observed)

    def rinit(theta_, key, covars, t0):
        x = jax.random.normal(key, (2,))
        return {"X1": x[0], "X2": x[1]}

    def rproc(X_, theta_, key, covars, t, dt):
        noise = jnp.sqrt(.5) * jax.random.normal(key, (2,))
        return {"X1": theta_["a1"] * X_["X1"] + noise[0],
                "X2": theta_["a2"] * X_["X2"] + noise[1]}

    def dmeas(Y_, X_, theta_, covars, t):
        return sum(jax.scipy.stats.norm.logpdf(Y_[f"Y{i}"], X_[f"X{i}"],
                                              jnp.sqrt(.1)) for i in (1, 2))

    model = pp.Pomp(
        ys=pd.DataFrame(observed, columns=["Y1", "Y2"],
                        index=np.arange(1, points + 1, dtype=float)),
        theta=pp.PompParameters({"a1": .5, "a2": .5}),
        t0=0., nstep=1, rinit=rinit, rproc=rproc, dmeas=dmeas,
        statenames=["X1", "X2"])
    return model, observed


def linear_loglik(theta, observed):
    a = jnp.array([theta["a1"], theta["a2"]])

    def step(carry, y):
        m, c = carry
        m, c = a * m, a**2 * c + .5
        variance = c + .1
        ll = jax.scipy.stats.norm.logpdf(y, m, jnp.sqrt(variance)).sum()
        gain = c / variance
        return (m + gain * (y - m), c * (1 - gain)), ll

    _, ll = jax.lax.scan(step, (jnp.zeros(2), jnp.ones(2)), jnp.asarray(observed))
    return ll.sum()


def spx():
    """Use the installed Pypomp source and data without copying its simulator."""
    return pp.models.spx()


def source_manifest(model):
    """Hash observations, covariates, and the actual SPX source/data files."""
    import importlib
    spx_module = importlib.import_module("pypomp.models.spx")
    paths = [Path(spx_module.__file__), Path(spx_module.data_file)]
    return {"files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "observations_sha256": hashlib.sha256(model.ys.to_csv().encode()).hexdigest(),
            "observations": len(model.ys),
            "covariates_sha256": hashlib.sha256(model.covars.to_csv().encode()).hexdigest()}
