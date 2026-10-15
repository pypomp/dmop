"""Metropolis-corrected MOP proposals on an explicit pseudo-marginal state.

All proposal settings are fixed during retained sampling. A finite arbitrary
force is allowed: MALA uses both proposal densities; fixed-length HMC uses a
reversible, volume-preserving map and the endpoint energy difference.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class State(NamedTuple):
    z: jax.Array
    u: jax.Array
    loglik: jax.Array
    force: jax.Array


def finite_force(force):
    # Deterministic force regularization changes proposals, not the MH target.
    force = jnp.nan_to_num(force, nan=0., posinf=0., neginf=0.)
    return force * jnp.minimum(1., 1e4/jnp.maximum(jnp.linalg.norm(force), 1.))


def leapfrog(z, p, u, force, get_force, chol, step_size, steps):
    """Momentum-flipped reversible map, valid for any deterministic force."""
    p = p + .5*step_size*(chol.T@force)
    def one(i, carry):
        z, p, force = carry
        z = z + step_size*(chol@p)
        force = finite_force(get_force(z, u))
        p = p + jnp.where(i == steps-1, .5, 1.)*step_size*(chol.T@force)
        return z, p, force
    z, p, force = jax.lax.fori_loop(0, steps, one, (z, p, force))
    return z, -p, force


def make_kernel(loglik, logprior, get_force, method='mala', steps=4):
    def init(z, u):
        return State(z, u, loglik(z, u), finite_force(get_force(z, u)))

    def step(state, key, step_size, chol):
        ku, ka, kp, kb = jax.random.split(key, 4)
        # Independent draw from the base law of the PF randomness. Keep the
        # current random state and likelihood on rejection (no noisy-MH reset).
        proposed_ll = loglik(state.z, ku)
        loga_u = proposed_ll-state.loglik
        accept_u = jnp.isfinite(proposed_ll) & (jnp.log(jax.random.uniform(ka)) < loga_u)
        state = jax.lax.cond(accept_u,
            lambda: State(state.z, ku, proposed_ll, finite_force(get_force(state.z, ku))),
            lambda: state)
        noise = jax.random.normal(kp, state.z.shape)
        if method == 'hmc':
            proposed_z, momentum, force = leapfrog(
                state.z, noise, state.u, state.force, get_force, chol, step_size, steps)
            logq_ratio = .5*(noise@noise - momentum@momentum)
        elif method == 'mala':
            drift = .5*step_size**2*(chol@(chol.T@state.force))
            proposed_z = state.z + drift + step_size*(chol@noise)
            force = finite_force(get_force(proposed_z, state.u))
            reverse_drift = .5*step_size**2*(chol@(chol.T@force))
            reverse_noise = jax.scipy.linalg.solve_triangular(
                chol, (state.z-proposed_z-reverse_drift)/step_size, lower=True)
            logq_ratio = .5*(noise@noise - reverse_noise@reverse_noise)
        else:
            raise ValueError(method)
        proposed_ll = loglik(proposed_z, state.u)
        loga = proposed_ll-state.loglik + logprior(proposed_z)-logprior(state.z) + logq_ratio
        accept = jnp.isfinite(loga) & (jnp.log(jax.random.uniform(kb)) < loga)
        new = jax.lax.cond(accept,
            lambda: State(proposed_z, state.u, proposed_ll, force), lambda: state)
        info = jnp.array([accept, accept_u, jnp.exp(jnp.minimum(0., jnp.nan_to_num(loga, nan=-jnp.inf))),
                          loga, loga_u, jnp.linalg.norm(new.force)])
        return new, info
    return jax.jit(init), jax.jit(step)
