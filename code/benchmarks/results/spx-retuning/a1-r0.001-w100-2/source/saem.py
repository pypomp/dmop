"""SAEM sufficient statistics and Gaussian-model maximization steps.

The oscillator uses the strong-1.5 Gaussian transition and conditional initial
velocity law used by the filtering code. Maximizing the averaged complete-data
log likelihood implements the SAEM M-step, rather than averaging its score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from models import ho_moments


def make_saem(name, model, observations, bounds):
    if name not in ("linear", "oscillator"):
        raise ValueError("Gaussian SAEM is implemented only for the two Gaussian examples")
    count = len(model.ys)
    indices = jnp.array([model.statenames.index(n) for n in
                         (["X1", "X2"] if name == "linear" else ["U"])])

    @jax.jit
    def sufficient(path):
        hidden = path[:, indices]
        x = hidden if name == "linear" else jnp.column_stack((jnp.asarray(observations), hidden[:, 0]))
        dx = x[1:]-x[:-1]
        # Increments reduce cancellation in the oscillator's small-noise coordinate.
        return jnp.concatenate(((x[:-1].T@x[:-1]).ravel(),
                                (dx.T@x[:-1]).ravel(), (dx.T@dx).ravel(),
                                jnp.array([x[0, -1]**2])))

    @jax.jit
    def objective(z, stats):
        Sxx, Sdx, Sdd = stats[:4].reshape(2, 2), stats[4:8].reshape(2, 2), stats[8:12].reshape(2, 2)
        if name == "oscillator":
            p = dict(zip(model.canonical_param_names, jnp.exp(z)))
            F, Q = ho_moments(p)
            initial_variance = p["sigma"]**2/(2*p["gamma"])
            initial = -.5*(jnp.log(2*jnp.pi*initial_variance)+stats[-1]/initial_variance)
        else:
            F, Q, initial = jnp.diag(z), .5*jnp.eye(2), 0.
        B = F-jnp.eye(2)
        residual = Sdd-B@Sdx.T-Sdx@B.T+B@Sxx@B.T
        return initial-.5*(count*(2*jnp.log(2*jnp.pi)+jnp.linalg.slogdet(Q)[1])
                            + jnp.trace(jnp.linalg.solve(Q, residual)))

    value_grad = jax.jit(jax.value_and_grad(lambda z, s: -objective(z, s)))

    def maximize(z, stats):
        if name == "linear":
            Sxx = np.asarray(stats[:4]).reshape(2, 2)
            Sdx = np.asarray(stats[4:8]).reshape(2, 2)
            candidate = 1+np.diag(Sdx)/np.diag(Sxx)
            return jnp.asarray(np.clip(candidate, bounds[:, 0], bounds[:, 1])), True
        result = minimize(lambda x: tuple(np.asarray(v) for v in value_grad(x, stats)),
                          np.asarray(z), jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 100, "ftol": 1e-10, "gtol": 1e-6})
        # A line-search warning can accompany a numerically stationary solution.
        # Accept only a finite, nondecreasing complete-data objective.
        before = float(objective(z, stats))
        after = float(objective(result.x, stats))
        valid = np.isfinite(result.x).all() and np.isfinite(after) and after >= before-1e-7
        return jnp.asarray(result.x if valid else z), bool(valid)

    return sufficient, objective, maximize
