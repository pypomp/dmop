"""Daphnia adapters for the CTDD21 and DS19 comparisons.

CTDD21 uses the existing S10 Euler simulator. The DS19 extension composes
strong-1.5 local Gaussian moments over an observation interval, as in the
Dhaka comparison. Its fitting model is approximate; final likelihoods must
be evaluated with the original S10 model.
"""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ctdd import make_filter
from corenflos.transport import TransportConfig
from pypomp.core.algorithms.contexts import PfilterContext
from pypomp.core.algorithms.pfilter import _pfilter_internal
from pypomp.core.algorithms.helpers import _resample

_path = Path(__file__).resolve().parents[1] / "daphnia" / "model.py"
_spec = importlib.util.spec_from_file_location("daphnia_reference", _path)
reference = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reference
_spec.loader.exec_module(reference)

STATES = reference.STATENAMES[:8]
SHARED = tuple(n for n in reference.SHARED_PARAMETERS if n not in reference.FIXED_PARAMETERS)
NAMES = SHARED + tuple(f"{n}_{u}" for n in reference.UNIT_PARAMETERS for u in reference.UNITS)
SCALES = jnp.array([3., 1., 3., 1., 1., 1., 16., 25.])
UPPER = jnp.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e20, 1e20])


def parameters(z, unit):
    p = {n: jnp.exp(z[i]) for i, n in enumerate(SHARED)}
    p.update({n: jnp.array(0.) for n in reference.FIXED_PARAMETERS})
    for k, n in enumerate(reference.UNIT_PARAMETERS):
        p[n] = jnp.exp(z[len(SHARED) + k * len(reference.UNITS) + unit])
    return p


def pack(payload):
    values = [payload["shared"].loc[n, "shared"] for n in SHARED]
    values += [payload["unit_specific"].loc[n, u]
               for n in reference.UNIT_PARAMETERS for u in reference.UNITS]
    return np.log(np.array(values, dtype=float))


def drift(x, p):
    sn, inn, jn, si, ii, ji, food, spores = x
    fn, fi, xi = p["f_Sn"], p["f_Si"], p["xi"]
    infection_n, infection_i = p["probn"] * fn * sn * spores, p["probi"] * fi * si * spores
    return jnp.array([
        .1*jn - (p["theta_Sn"] + .013)*sn - infection_n,
        infection_n - (p["theta_In"] + .013)*inn,
        p["rn"]*fn*food*sn - (p["theta_Jn"] + .113)*jn,
        .1*ji - (p["theta_Si"] + .013)*si - infection_i,
        infection_i - (p["theta_Ii"] + .013)*ii,
        p["ri"]*fi*food*si - (p["theta_Ji"] + .113)*ji,
        .37 - .013*food - fn*food*(sn + xi*inn + jn) - fi*food*(si + xi*ii + ji),
        30.*(p["theta_In"]*inn + p["theta_Ii"]*ii)
        - (p["theta_P"] + .013)*spores - fn*(sn + xi*inn)*spores - fi*(si + xi*ii)*spores])


def coefficients(p):
    return jnp.array([p[n] for n in ("sigSn", "sigIn", "sigJn", "sigSi",
                                     "sigIi", "sigJi", "sigF", "sigP")])


def local_moments(x, p, dt=.25):
    """Strong-1.5 moments for commuting diagonal multiplicative noise.

    The drift has zero diagonal second derivatives, so its Ito-generator
    correction is Db b. The diffusion fields commute and each depends only
    on its own coordinate. Consequently mixed diffusion derivatives vanish.
    """
    b = drift(x, p)
    B = jax.jacfwd(drift)(x, p)
    sig = coefficients(p)
    g = jnp.diag(sig * x)
    l1b, l0g = B @ g, jnp.diag(sig * b)
    eta = g + .5 * dt * (l1b + l0g)
    xi = l1b - l0g
    milstein, third = jnp.diag(sig**2*x), jnp.diag(sig**3*x)
    Q = (dt*eta@eta.T + dt**3/12*xi@xi.T
         + dt**2/2*milstein@milstein.T + dt**3/6*third@third.T)
    return x + dt*b + .5*dt**2*(B@b), Q


def block_moments(x, p, observation_index, relative_floor=1e-8):
    """Linearized moment composition; inoculation follows the S10 grid."""
    count = jnp.where(observation_index == 0, 24, 20)
    t0 = jnp.where(observation_index == 0, 1., 7. + 5.*(observation_index - 1))

    def step(carry, i):
        def advance(carry):
            m, C = carry
            F = jax.jacfwd(lambda x: local_moments(x, p)[0])(m)
            mean, Q = local_moments(m, p)
            t = t0 + .25*i
            mean = mean.at[7].add(jnp.where((t <= 4.) & (t + .25 > 4.), 25., 0.))
            C = F@C@F.T + Q
            return mean, .5*(C+C.T)
        return jax.lax.cond(i < count, advance, lambda c: c, carry), None

    (mean, covariance), _ = jax.lax.scan(step, (x, jnp.zeros((8, 8))), jnp.arange(24))
    # Regularize in fixed, declared physical scales, not units chosen by a fit.
    C = covariance / SCALES[:, None] / SCALES[None, :]
    floor = relative_floor * jnp.maximum(jnp.max(jnp.diag(C)), 1e-12)
    covariance = covariance + jnp.diag(floor*SCALES**2)
    return mean, covariance


def measurement(y, x, p):
    return sum(reference.nb_logpmf(y[k], x[index], p[name]) for k, index, name in
               ((0, 0, "k_Sn"), (1, 1, "k_In"), (2, 3, "k_Si"), (3, 4, "k_Ii")))


def make_euler_objectives(particles=100, eval_particles=2000):
    """Reuse S10 components for transport fitting and ordinary PF evaluation."""
    units = reference._unit_models(reference.DATA_PATH.resolve())
    functions, evaluators = [], []
    for i, unit in enumerate(reference.UNITS):
        model = units[unit]
        ct = make_filter(model, particles, TransportConfig(epsilon=.5),
                         active_names=STATES, reset_names=("error_count",))
        context = PfilterContext.from_struct(model.to_struct(), J=eval_particles, should_trans=True)
        functions.append(ct)
        evaluators.append(jax.jit(lambda z, key, c=context: -_pfilter_internal(z, key, c)["neg_loglik"]))
    canonical = units[reference.UNITS[0]].canonical_param_names

    def unit_z(z, i):
        p = parameters(z, i)
        return jnp.array([p[n] if n in reference.FIXED_PARAMETERS else jnp.log(p[n]) for n in canonical])

    @jax.jit
    def ct_objective(z, key):
        keys = jax.random.split(key, len(units))
        return sum(fn(unit_z(z, i), keys[i]) for i, fn in enumerate(functions))

    @jax.jit
    def pf_evaluate(z, key):
        keys = jax.random.split(key, len(units))
        return jnp.array([fn(unit_z(z, i), keys[i]) for i, fn in enumerate(evaluators)])

    return ct_objective, pf_evaluate


def make_ds_score(particles=100, relative_floor=1e-8):
    """Gaussian block proposals, exact NB weights, and backward path scores.

    Negative/out-of-range Gaussian proposals have zero weight in this
    approximation. This is not the Euler simulator's boundary mechanism;
    independent Euler evaluation is therefore essential.
    """
    units = reference._unit_models(reference.DATA_PATH.resolve())
    observations = jnp.array([units[u].ys.to_numpy() for u in reference.UNITS])
    x0 = jnp.array([2.333, 0., 0., .667, 0., 0., 16.667, 0.])
    observed_indices = jnp.array([0, 1, 3, 4])

    def log_normal(x, means, chol):
        residual = jax.vmap(lambda L, d: jax.scipy.linalg.solve_triangular(L, d, lower=True))(
            chol, x-means)
        return -.5*(8*jnp.log(2*jnp.pi) + 2*jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)).sum(1)
                    + (residual**2).sum(1))

    def sample_unit(z, unit, ys, key):
        p = parameters(z, unit)
        uniform = jnp.full((particles,), -jnp.log(float(particles)))
        keys = jax.random.split(key, 21)

        def step(carry, inputs):
            x, weights = carry
            i, y, key = inputs
            means, C = jax.vmap(lambda x: block_moments(x, p, i, relative_floor))(x)
            L = jnp.linalg.cholesky(C)
            ancestors_key, noise_key = jax.random.split(key)
            ancestors = _resample(weights, ancestors_key)
            m, covariance = means[ancestors], C[ancestors]
            predicted = jnp.maximum(m[:, observed_indices], 1e-8)
            sizes = jnp.array([p[n] for n in ("k_Sn", "k_In", "k_Si", "k_Ii")])
            R = jax.vmap(jnp.diag)(predicted + predicted**2/sizes)
            cross = covariance[:, :, observed_indices]
            innovation_C = covariance[:, observed_indices[:, None], observed_indices[None, :]] + R
            K = jnp.swapaxes(jnp.linalg.solve(innovation_C, jnp.swapaxes(cross, 1, 2)), 1, 2)
            proposed_mean = m + jnp.einsum('nij,nj->ni', K, y-m[:, observed_indices])
            proposed_C = covariance - K@jnp.swapaxes(cross, 1, 2)
            proposed_C = .5*(proposed_C+jnp.swapaxes(proposed_C, 1, 2))
            proposed_L = jnp.linalg.cholesky(proposed_C)
            proposed = proposed_mean + jnp.einsum('nij,nj->ni', proposed_L,
                                                jax.random.normal(noise_key, (particles, 8)))
            valid = jnp.all(jnp.isfinite(proposed) & (proposed >= 0.) & (proposed <= UPPER), axis=1)
            correction = (log_normal(proposed, m, L[ancestors])
                          - log_normal(proposed, proposed_mean, proposed_L))
            ll = jax.vmap(lambda x: measurement(y, jnp.maximum(x, 1e-10), p))(proposed)
            raw = jnp.where(valid, ll+correction, -jnp.inf)
            normalizer = jax.scipy.special.logsumexp(raw)
            new_weights = jnp.where(jnp.isfinite(normalizer), raw-normalizer, uniform)
            safe = jnp.clip(jnp.nan_to_num(proposed, nan=0., posinf=0., neginf=0.), 0., UPPER)
            return (safe, new_weights), (x, weights, means, L, safe, new_weights,
                                         normalizer-jnp.log(float(particles)))

        _, history = jax.lax.scan(step, (jnp.tile(x0, (particles, 1)), uniform),
                                 (jnp.arange(10), ys, keys[:10]))
        previous, previous_w, means, chol, following, weights, increments = history
        last = following[-1, jax.random.categorical(keys[-1], weights[-1])]

        def backward(x, i):
            logits = previous_w[i] + log_normal(x, means[i], chol[i])
            selected = previous[i, jax.random.categorical(keys[10+i], logits)]
            return selected, selected

        _, path = jax.lax.scan(backward, last, jnp.arange(10), reverse=True)
        return jnp.concatenate((path, last[None])), increments.sum()

    def complete_unit(z, unit, ys, path):
        p = parameters(z, unit)
        def term(i, old, new, y):
            m, C = block_moments(old, p, i, relative_floor)
            return jax.scipy.stats.multivariate_normal.logpdf(new, m, C) + measurement(y, new, p)
        return jax.vmap(term)(jnp.arange(10), path[:-1], path[1:], ys).sum()

    @jax.jit
    def sample(z, key):
        return jax.vmap(sample_unit, in_axes=(None, 0, 0, 0))(
            z, jnp.arange(8), observations, jax.random.split(key, 8))

    @jax.jit
    def complete(z, paths):
        return jax.vmap(complete_unit, in_axes=(None, 0, 0, 0))(
            z, jnp.arange(8), observations, paths).sum()

    @jax.jit
    def score(z, key):
        paths, ll = sample(z, key)
        gradient = jax.grad(complete)(z, jax.lax.stop_gradient(paths))
        return ll.sum(), gradient

    return score, sample, complete
