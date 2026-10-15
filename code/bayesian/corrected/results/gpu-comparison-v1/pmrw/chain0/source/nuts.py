"""BlackJAX NUTS conditional on explicit PF randomness, plus an MH refresh.

The integrator uses a deterministic MOP force, whereas every trajectory
weight/energy uses the unscaled PF target. See METHOD.md for the conditions
behind using a reversible volume-preserving map with a surrogate force.
"""
import jax
import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.nuts import build_kernel
from kernels import State,finite_force


def force_integrator(get_force, u):
    def factory(logdensity_fn, kinetic_energy_fn):
        velocity=jax.grad(kinetic_energy_fn)
        def one(state, step_size):
            p=state.momentum+.5*step_size*state.logdensity_grad
            z=state.position+step_size*velocity(p)
            force=finite_force(get_force(z,u))
            logdensity=logdensity_fn(z)
            p=p+.5*step_size*force
            return IntegratorState(z,p,logdensity,force)
        return one
    return factory


def make_nuts_kernel(loglik,logprior,get_force,max_depth=5):
    def initialize(z,u):
        return State(z,u,loglik(z,u),finite_force(get_force(z,u)))

    def step(state,key,step_size,chol):
        ku,ka,kn=jax.random.split(key,3)
        ll=loglik(state.z,ku)
        loga_u=ll-state.loglik
        accept_u=jnp.isfinite(ll)&(jnp.log(jax.random.uniform(ka))<loga_u)
        state=jax.lax.cond(accept_u,
            lambda: State(state.z,ku,ll,finite_force(get_force(state.z,ku))),
            lambda: state)
        logdensity=lambda z: loglik(z,state.u)+logprior(z)
        kernel=build_kernel(integrator=force_integrator(get_force,state.u))
        hs=HMCState(state.z,state.loglik+logprior(state.z),state.force)
        hs,diagnostics=kernel(kn,hs,logdensity,step_size,chol@chol.T,max_depth)
        new=State(hs.position,state.u,hs.logdensity-logprior(hs.position),hs.logdensity_grad)
        info=jnp.array([jnp.any(new.z!=state.z),accept_u,diagnostics.acceptance_rate,
                        diagnostics.energy,loga_u,jnp.linalg.norm(new.force),
                        diagnostics.is_divergent,diagnostics.num_integration_steps,
                        diagnostics.num_trajectory_expansions,diagnostics.is_turning])
        return new,info
    return jax.jit(initialize),jax.jit(step)
