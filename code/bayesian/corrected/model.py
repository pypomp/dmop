"""Pypomp Dhaka model and common priors for the corrected Bayesian experiment.

No diffPomp code is imported. Sampling coordinates are standardized versions
of Pypomp's estimation coordinates; fixed parameters never enter the sampler.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pypomp as pp
import importlib
d = importlib.import_module("pypomp.models.dacca")
from pypomp.functional import mop, pfilter
from pypomp.types import ObservationDict, StateDict, ParamDict, CovarDict, TimeFloat, RNGKey, StepSizeFloat
from pypomp.core.model_mechanics import vectorized

NAMES = ('gamma', 'm', 'epsilon', 'beta_trend', 'sigma', 'tau',
         *(f'bs{i}' for i in range(1, 7)), *(f'omegas{i}' for i in range(1, 7)))
# The old search ranges specify a transparent, common reference prior,
# not a density fitted to IF2 output. Normal SD = half the range width.
BOUNDS = np.array([np.log([10, 40]), np.log([.03, .6]), np.log([.2, 30]),
                   [-1, 0], np.log([1, 5]), np.log([.1, .5]),
                   *[[-4, 4], [0, 8], [-4, 4], [0, 8], [0, 8], [0, 8]],
                   *[[-10, 0]] * 6])
CENTER = BOUNDS.mean(axis=1)
SCALE = np.diff(BOUNDS, axis=1).ravel()/2


def gaussian_measurement(Y_: ObservationDict, X_: StateDict, theta_: ParamDict,
                         covars: CovarDict, t: TimeFloat):
    """Normalized Gaussian observation density; invalid Euler paths are killed.

    An absorbing cemetery state emits a special failure symbol, outside the
    observed real-valued series. Thus invalid paths have zero data likelihood.
    This removes the library's additive density floor, without altering its
    simulator or its treatment of negative Euler states.
    """
    sd = theta_['tau'] * X_['Mn']
    valid = (X_['count'] == 0) & jnp.isfinite(sd) & (sd > 0)
    safe_sd = jnp.where(valid, sd, 1.)
    safe_mean = jnp.where(valid, X_['Mn'], 0.)
    value = jax.scipy.stats.norm.logpdf(Y_['deaths'], safe_mean, safe_sd)
    return jnp.where(valid, value, -jnp.inf)


@vectorized
def sorted_rproc(X_: StateDict, theta_: ParamDict, key: RNGKey,
                 covars: CovarDict, t: TimeFloat, dt: StepSizeFloat):
    """Library simulator, with particles ordered by deaths before resampling.

    Permuting exchangeable particles changes neither the transition law nor
    the PF likelihood expectation. It improves the common-random-number
    coupling between nearby parameter values. Sorting occurs only at monthly
    endpoints; the intermediate Euler steps remain unchanged.
    """
    states = d._rproc(X_, theta_, key, covars, t, dt)
    phase = (t+dt-1891.)*12
    def ordered(states):
        order = jnp.argsort(states['Mn'], stable=True)
        return {name: value[order] for name, value in states.items()}
    return jax.lax.cond(jnp.abs(phase-jnp.round(phase)) < 1e-6,
                        ordered, lambda x: x, states)


def setup(particles=1000, alpha=.95, prior_scale=1., measurement='gaussian', sort_particles=False, force_particles=None):
    force_particles = particles if force_particles is None else force_particles
    # Construct via the same library components as d.dhaka; change only the
    # explicitly documented measurement floor. Keep Gaussian process noise.
    model = pp.Pomp(rinit=d._rinit, rproc=sorted_rproc if sort_particles else d._rproc,
                    dmeas=gaussian_measurement if measurement == 'gaussian' else d._dmeas,
                    ys=d.ys, t0=1891., dt=None, nstep=20,
                    theta=pp.PompParameters(d.theta), covars=d.covars,
                    statenames=d.statenames, accumvars=d.accumvars,
                    par_trans=pp.ParTrans(to_est=d._to_est, from_est=d._from_est))
    struct = model.to_struct()
    # The stabilized library measurement supplies a finite approximate force.
    # The acceptance target uses the normalized measurement above.
    force_model = pp.Pomp(rinit=d._rinit, rproc=sorted_rproc if sort_particles else d._rproc,
                         dmeas=d._dmeas, ys=d.ys, t0=1891., dt=None, nstep=20,
                         theta=pp.PompParameters(d.theta), covars=d.covars,
                         statenames=d.statenames, accumvars=d.accumvars,
                         par_trans=pp.ParTrans(to_est=d._to_est, from_est=d._from_est))
    force_struct = force_model.to_struct()
    names = model.canonical_param_names
    index = jnp.array([names.index(n) for n in NAMES])
    baseline = jnp.array([d._to_est(d.theta)[n] for n in names])

    def full_est(z):
        return baseline.at[index].set(jnp.asarray(CENTER) + jnp.asarray(SCALE)*z)

    def full_nat(z):
        return struct.par_trans._transform_array(full_est(z)[None], names,
                                                  direction='from_est')[0]

    def logprior(z):
        # Density in z coordinates; both affine Jacobians are constant.
        # On natural parameters this is lognormal for the positive rates.
        return -.5*jnp.sum((z/prior_scale)**2) - len(NAMES)*jnp.log(prior_scale)

    def loglik(z, key):
        return pfilter(struct, full_nat(z)[None], particles,
                       key[None, None], thresh=0.)['logLik'][0, 0]

    def mop_target(z, key):
        return -mop(force_struct, full_est(z)[None], force_particles, alpha, key[None])[0] + logprior(z)

    return dict(model=model, struct=struct, full_est=full_est, full_nat=full_nat,
                logprior=jax.jit(logprior), loglik=jax.jit(loglik),
                value_force=jax.jit(jax.value_and_grad(mop_target)),
                initial=(np.array([d._to_est(d.theta)[n] for n in NAMES])-CENTER)/SCALE)
