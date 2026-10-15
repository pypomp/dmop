"""Tests against reversible maps and an analytically known extended target."""
import jax
import jax.numpy as jnp
import numpy as np
from kernels import finite_force, leapfrog, make_kernel

jax.config.update('jax_enable_x64', True)


def test_reversibility_nonconservative_force():
    # Deliberately not the gradient of any potential: unequal cross partials.
    force = lambda z, u: jnp.array([-z[0]+.3*z[1], -2*z[1]-.2*z[0]])
    z, p = jnp.array([.4, -.7]), jnp.array([1.2, .3])
    chol = jnp.array([[.7, 0.], [.2, 1.1]])
    z1, p1, f1 = leapfrog(z, p, None, force(z, None), force, chol, .15, 7)
    z2, p2, _ = leapfrog(z1, p1, None, f1, force, chol, .15, 7)
    np.testing.assert_allclose(z2, z, atol=1e-12)
    np.testing.assert_allclose(p2, p, atol=1e-12)
    def mapping(x):
        zz, pp = x[:2], x[2:]
        zz, pp, _ = leapfrog(zz, pp, None, force(zz, None), force, chol, .15, 7)
        return jnp.concatenate([zz, pp])
    np.testing.assert_allclose(abs(np.linalg.det(jax.jacfwd(mapping)(jnp.r_[z,p]))), 1., atol=1e-12)


def test_extended_gaussian_target():
    # E_u Lhat(z,u) = exp(-z^2/2), with a proper N(0,1) prior.
    # Marginal z ~ N(0,1/2), despite a noisy likelihood and biased force.
    a = .8
    loglik = lambda z, u: -.5*z@z + a*z[0]*jax.random.normal(u) - .5*a*a*z@z
    prior = lambda z: -.5*z@z
    force = lambda z, u: -.6*z  # deliberately inaccurate, omits the noise term
    for method in ['mala', 'hmc', 'pmrw', 'pmmala']:
        init, step = make_kernel(loglik, prior, force, method=method, steps=5)
        state = init(jnp.array([0.]), jax.random.key(41))
        def one(s, k):
            s, info = step(s, k, .5, jnp.eye(1))
            return s, (s.z, jax.random.normal(s.u), info)
        state, (zs, us, infos) = jax.jit(lambda s, ks: jax.lax.scan(one,s,ks))(
            state, jax.random.split(jax.random.key(876), 240000))
        z = np.asarray(zs)[2000:, 0]
        u = np.asarray(us)[2000:]
        assert abs(z.mean()) < .035
        assert abs(z.var()-.5) < .035
        # E(u|z)=a*z, Var(u)=1+a^2/2: verifies updates to auxiliary state.
        assert abs(np.mean(z*u)-a*.5) < .04
        assert abs(u.var()-(1+a*a*.5)) < .05
        assert np.asarray(infos)[2000:, :2].mean(axis=0).min() > .1


def test_force_regularization_is_finite():
    f = finite_force(jnp.array([jnp.inf, jnp.nan, 1e10]))
    assert np.isfinite(f).all()
    assert np.linalg.norm(f) <= 10000.00001
