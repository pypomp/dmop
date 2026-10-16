"""Forward differentiation must preserve objective, stop-gradient rules and dtype."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from force_modes import select_value_force


@pytest.mark.parametrize('dtype', [jnp.float32,jnp.float64])
def test_forward_matches_value_and_gradient_with_stopped_terms(dtype):
    def objective(z, key):
        x=z.astype(dtype)
        return jnp.sum(jnp.sin(x)+x*x*jax.lax.stop_gradient(x))+jax.random.normal(key,dtype=dtype)*x[0]
    reference=jax.jit(jax.value_and_grad(objective))
    candidate=select_value_force(reference,'forward')
    assert select_value_force(reference) is reference
    zs=jnp.array([[.2,-.8,1.3],[1.,2.,-.3]],dtype=jnp.float64)
    keys=jax.random.split(jax.random.key(91),2)
    actual=jax.jit(jax.vmap(candidate))(zs,keys)
    expected=jax.jit(jax.vmap(reference))(zs,keys)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_allclose(actual[1],expected[1],rtol=2e-6,atol=2e-6)
    assert actual[1].dtype==expected[1].dtype
    for a,b in zip(actual,jax.jit(jax.vmap(candidate))(zs,keys)):
        np.testing.assert_array_equal(a,b)


def test_reject_unknown_mode():
    with pytest.raises(ValueError,match='Unknown'):
        select_value_force(lambda x,k: (x,x),'other')
