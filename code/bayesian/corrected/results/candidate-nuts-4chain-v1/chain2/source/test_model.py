"""A cheaper MOP force must not change the acceptance target."""
from model import setup
import jax
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
