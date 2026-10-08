"""SMC backward simulation and complete-data scores for the small models.

These are the numerical-score extensions used for the DS19 comparison, not
an implementation of the paper's model-specific SAEM maximization step.
"""

import jax
import jax.numpy as jnp

from pypomp.core.algorithms.helpers import _resample
from models import ho_moments


def densities(name, model, observed=None):
    """Return initial and joint transition/measurement log densities."""
    struct = model.to_struct()
    names = model.canonical_param_names

    def decode(z):
        raw = dict(zip(names, z))
        if name == "linear":
            return raw
        if name == "oscillator":
            return {k: jnp.exp(v) for k, v in raw.items()}
        from pypomp.models.spx import _from_est
        return _from_est(raw)

    if name == "oscillator":
        v = jnp.asarray(observed)
        u_index = model.statenames.index("U")

        def initial(z, x):
            p = decode(z)
            return jax.scipy.stats.norm.logpdf(x[u_index], 0.,
                                               p["sigma"] / jnp.sqrt(2 * p["gamma"]))

        def joint(z, previous, following, i):
            F, Q = ho_moments(decode(z))
            x = jnp.array([v[i], previous[u_index]])
            y = jnp.array([v[i + 1], following[u_index]])
            return jax.scipy.stats.multivariate_normal.logpdf(y, F @ x, Q)

    elif name == "linear":
        y = jnp.asarray(observed)

        def initial(z, x):
            return jax.scipy.stats.norm.logpdf(x, 0., 1.).sum()

        def joint(z, previous, following, i):
            return (jax.scipy.stats.norm.logpdf(following, z * previous, jnp.sqrt(.5)).sum()
                    + jax.scipy.stats.norm.logpdf(y[i], following, jnp.sqrt(.1)).sum())

    elif name == "spx":
        v_index = model.statenames.index("V")
        y = struct.ys[:, 0]
        y_previous = struct.covars_extended[:-1, 0]

        def initial(z, x):
            # V_0 is deterministic. Its score enters the first transition.
            return jnp.array(0.)

        def joint(z, previous, following, i):
            p = decode(z)
            v = jnp.where(i == 0, p["V_0"], previous[v_index])
            mean = (v + p["kappa"] * (p["theta"] - v)
                    + p["xi"] * p["rho"] * (y_previous[i] - p["mu"] + .5 * v))
            scale = p["xi"] * jnp.sqrt(v * (1 - p["rho"]**2))
            new_v = following[v_index]
            continuous = jax.scipy.stats.norm.logpdf(new_v, mean, scale)
            atom = jax.scipy.special.log_ndtr((1e-32 - mean) / scale)
            transition = jnp.where(new_v <= 1e-32, atom, continuous)
            return transition + jax.scipy.stats.norm.logpdf(
                y[i], p["mu"] - .5 * new_v, jnp.sqrt(new_v))
    else:
        raise ValueError(name)
    return initial, joint


def make_smoother(name, model, observed=None, particles=100):
    """One backward-simulated path from the particle smoothing distribution."""
    struct = model.to_struct()
    initial_density, joint = densities(name, model, observed)
    T = len(struct.ys)

    @jax.jit
    def sample(z, key):
        keys = jax.random.split(key, particles + 2)
        cov0 = None if struct.covars_extended is None else struct.covars_extended[0]
        x0 = struct.rinit_pf(z, keys[2:], cov0, struct.t0, True)
        uniform = jnp.full((particles,), -jnp.log(float(particles)))

        def step(carry, i):
            x, t, t_idx, key = carry
            key, process_key, resample_key = jax.random.split(key, 3)
            x, t_idx = struct.rproc_pf(x, z, process_key,
                struct.covars_extended, struct.dt_array_extended, t, t_idx,
                struct.nstep_array[i].astype(int), struct.accumvars, True)
            t = struct.times[i]
            cov = None if struct.covars_extended is None else struct.covars_extended[t_idx]
            lw = struct.dmeas_pf(struct.ys[i], x, z, cov, t, True)
            inc = jax.scipy.special.logsumexp(lw) - jnp.log(float(particles))
            lw = lw - jax.scipy.special.logsumexp(lw)
            indices = _resample(lw, resample_key)
            return (x[indices], t, t_idx, key), (x, lw, inc)

        _, (xs, weights, increments) = jax.lax.scan(step,
            (x0, jnp.asarray(struct.t0), jnp.array(0), keys[0]), jnp.arange(T))
        xs = jnp.concatenate((x0[None], xs))
        weights = jnp.concatenate((uniform[None], weights))
        backwards = jax.random.split(keys[1], T + 1)
        last = xs[-1, jax.random.categorical(backwards[-1], weights[-1])]

        def backward(following, i):
            scores = weights[i] + jax.vmap(lambda x: joint(z, x, following, i))(xs[i])
            x = xs[i, jax.random.categorical(backwards[i], scores)]
            return x, x

        _, path = jax.lax.scan(backward, last, jnp.arange(T), reverse=True)
        return jnp.concatenate((path, last[None])), increments.sum()

    @jax.jit
    def complete(z, path):
        transitions = jax.vmap(joint, in_axes=(None, 0, 0, 0))(
            z, path[:-1], path[1:], jnp.arange(T))
        return initial_density(z, path[0]) + transitions.sum()

    @jax.jit
    def score(z, key):
        path, ll = sample(z, key)
        value, gradient = jax.value_and_grad(complete)(z, jax.lax.stop_gradient(path))
        return ll, gradient

    return sample, complete, score
