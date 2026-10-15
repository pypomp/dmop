"""NUTS checks with deliberately nonconservative force and noisy likelihood."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc.integrators import IntegratorState
from nuts import force_integrator,make_nuts_kernel


def test_nuts_integrator_reversible_volume():
    force=lambda z,u:jnp.array([-z[0]+.3*z[1],-2*z[1]-.2*z[0]])
    logdensity=lambda z:-.5*z@z
    kinetic=lambda p:.5*p@p
    one=force_integrator(force,None)(logdensity,kinetic)
    z,p=jnp.array([.4,-.7]),jnp.array([1.2,.3])
    s=IntegratorState(z,p,logdensity(z),force(z,None))
    f=one(s,.17)
    b=one(f,-.17)
    np.testing.assert_allclose(b.position,z,atol=1e-12)
    np.testing.assert_allclose(b.momentum,p,atol=1e-12)
    def mapping(x):
        s=IntegratorState(x[:2],x[2:],logdensity(x[:2]),force(x[:2],None))
        t=one(s,.17)
        return jnp.r_[t.position,t.momentum]
    np.testing.assert_allclose(np.linalg.det(jax.jacfwd(mapping)(jnp.r_[z,p])),1.,atol=1e-12)


def test_nuts_extended_target_nonconservative_force():
    a=.8
    ll=lambda z,u:-.5*z@z+a*z[0]*jax.random.normal(u)-.5*a*a*z[0]**2
    prior=lambda z:-.5*z@z
    # Nonconservative, biased, and omits u entirely. The target is still
    # z~N(0,I/2), Cov(z0,u)=a/2, Var(u)=1+a^2/2.
    force=lambda z,u:jnp.array([-z[0]+.3*z[1],-1.2*z[1]-.2*z[0]])
    init,step=make_nuts_kernel(ll,prior,force,max_depth=5)
    state=init(jnp.zeros(2),jax.random.key(10))
    def one(s,k):
        s,info=step(s,k,.4,jnp.array([[1.,0.],[.2,.8]]))
        return s,(s.z,jax.random.normal(s.u),info)
    state,(zs,us,infos)=jax.jit(lambda s,ks:jax.lax.scan(one,s,ks))(
        state,jax.random.split(jax.random.key(82491),120000))
    z=np.asarray(zs)[2000:];u=np.asarray(us)[2000:]
    np.testing.assert_allclose(z.mean(0),0.,atol=.03)
    np.testing.assert_allclose(np.cov(z.T),.5*np.eye(2),atol=.035)
    assert abs(np.mean(z[:,0]*u)-.4)<.04
    assert abs(u.var()-1.32)<.05
    assert np.asarray(infos)[2000:,6].mean()<.01
