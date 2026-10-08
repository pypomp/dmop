"""CTDD21 resampling with the actual Pypomp model components.

    The model's compiled simulator, interpolation, parameter transformation and
    measurement density are reused. Only resampling changes. For SPX the
    unused stock-price coordinate does not enter the transport cost. For the
    conditional oscillator the auxiliary log weight is also excluded.
"""

from functools import partial

import jax
import jax.numpy as jnp

from corenflos.transport import TransportConfig, transport_matrix


def make_filter(model, particles=25, config=TransportConfig(),
                active_names=None, reset_names=()):
    struct = model.to_struct()
    active = jnp.array([model.statenames.index(n) for n in
                        (active_names or model.statenames)])
    reset = tuple(model.statenames.index(n) for n in reset_names)

    @jax.jit
    def loglik(z, key):
        keys = jax.random.split(key, particles + 1)
        cov0 = None if struct.covars_extended is None else struct.covars_extended[0]
        x = struct.rinit_pf(z, keys[1:], cov0, struct.t0, True)
        uniform = jnp.full((particles,), -jnp.log(float(particles)))

        def step(carry, i):
            x, weights, t, t_idx, key = carry
            key, process_key = jax.random.split(key)
            x, t_idx = struct.rproc_pf(
                x, z, process_key, struct.covars_extended,
                struct.dt_array_extended, t, t_idx,
                struct.nstep_array[i].astype(int), struct.accumvars, True)
            t = struct.times[i]
            cov = None if struct.covars_extended is None else struct.covars_extended[t_idx]
            unnormalized = weights + struct.dmeas_pf(struct.ys[i], x, z, cov, t, True)
            increment = jax.scipy.special.logsumexp(unnormalized)
            weights = unnormalized - increment
            ess = jnp.exp(-jax.scipy.special.logsumexp(2 * weights))

            def resample(inputs):
                x, weights = inputs
                T = transport_matrix(x[:, active], weights, config.epsilon,
                                     config.scaling, config.threshold, config.max_iterations)
                x = T @ x
                for index in reset:
                    x = x.at[:, index].set(0.)
                return x, uniform

            x, weights = jax.lax.cond(ess < particles / 2, resample,
                                      lambda inputs: inputs, (x, weights))
            return (x, weights, t, t_idx, key), increment

        _, increments = jax.lax.scan(jax.checkpoint(step),
            (x, uniform, jnp.asarray(struct.t0), jnp.array(0), keys[0]),
            jnp.arange(len(struct.ys)))
        return increments.sum()

    return loglik
