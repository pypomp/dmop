"""A cheaper MOP force must not change the acceptance target."""
from model import setup
import jax
import jax.numpy as jnp
import numpy as np


def test_force_particle_count_does_not_change_target():
    full=setup(1000,force_particles=1000)
    cheap=setup(1000,force_particles=100)
    z=full['initial'];key=jax.random.key(667)
    a=float(full['loglik'](z,key));b=float(cheap['loglik'](z,key))
    assert np.isfinite(a)
    assert a==b
    assert float(full['logprior'](z))==float(cheap['logprior'](z))
    _,g1=full['value_force'](z,key)
    _,g2=cheap['value_force'](z,key)
    assert np.isfinite(g1).all() and np.isfinite(g2).all()
    assert not np.allclose(g1,g2)


def test_mixed_force_preserves_float64_target_and_batches():
    full = setup(100, force_particles=20)
    mixed = setup(100, force_particles=20, force_precision='float32')
    zs = jnp.stack([jnp.asarray(full['initial']), jnp.asarray(full['initial']) + .001])
    keys = jax.random.split(jax.random.key(42), 2)
    values, forces = jax.jit(jax.vmap(mixed['value_force']))(zs, keys)
    assert forces.dtype == jnp.float64
    assert np.isfinite(forces).all() and np.isfinite(values).all()
    for i, (z, key) in enumerate(zip(zs, keys)):
        a, b = full['loglik'](z, key), mixed['loglik'](z, key)
        assert a.dtype == b.dtype == jnp.float64
        assert np.isfinite(a)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(full['logprior'](z), mixed['logprior'](z))
        v, g = mixed['value_force'](z, key)
        # Batched linear algebra may differ at float32 rounding precision.
        np.testing.assert_allclose(values[i], v, rtol=1e-5)
        np.testing.assert_allclose(forces[i], g, rtol=1e-4, atol=1e-4)
        _, repeated = mixed['value_force'](z, key)
        np.testing.assert_array_equal(g, repeated)
