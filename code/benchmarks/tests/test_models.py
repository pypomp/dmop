import importlib

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

import models
from ctdd import make_filter
from smoothing import densities, make_smoother
from ditlevsen.validation_ho import analytic_observed_loglik, HOParameters
from pypomp.core.algorithms.contexts import PfilterContext
from pypomp.core.algorithms.pfilter import _pfilter_internal


def test_oscillator_likelihood_agrees_with_original_validation():
    _, observations = models.oscillator(points=120)
    p = {"D": 3.2, "gamma": .7, "sigma": .6}
    expected = analytic_observed_loglik(observations,
        parameters=HOParameters(stiffness=p["D"], damping=p["gamma"], diffusion=p["sigma"]))
    np.testing.assert_allclose(models.oscillator_loglik(p, observations), expected, atol=1e-8)


def test_conditional_oscillator_particle_likelihood_targets_kalman():
    m, y = models.oscillator(points=80)
    z = jnp.log(jnp.array([4., .5, .5]))
    context = PfilterContext.from_struct(m.to_struct(), J=2000, should_trans=True)
    fn = jax.jit(jax.vmap(lambda key: -_pfilter_internal(z, key, context)["neg_loglik"]))
    ll = np.asarray(fn(jax.random.split(jax.random.key(70), 24)))
    exact = float(models.oscillator_loglik({"D": 4., "gamma": .5, "sigma": .5}, y))
    # A likelihood average, not an average of log likelihoods.
    assert abs(logsumexp(ll) - np.log(len(ll)) - exact) < .15


def test_linear_kalman_matches_dense_gaussian():
    _, y = models.linear_gaussian(points=12)
    a = [.3, .7]
    exact = 0.
    for k in range(2):
        covariance = np.empty((len(y), len(y)))
        variance = 1.
        for i in range(len(y)):
            variance = a[k]**2 * variance + .5
            for j in range(i, len(y)):
                covariance[i, j] = covariance[j, i] = variance * a[k]**(j - i)
        covariance += .1 * np.eye(len(y))
        exact += float(jax.scipy.stats.multivariate_normal.logpdf(y[:, k],
                         jnp.zeros(len(y)), jnp.asarray(covariance)))
    np.testing.assert_allclose(models.linear_loglik(dict(zip(["a1", "a2"], a)), y), exact)


def test_ctdd_uses_model_and_has_finite_gradients():
    for factory, z, active in ((models.linear_gaussian, jnp.array([.5, .5]), None),
                              (models.oscillator, jnp.log(jnp.array([4., .5, .5])), ["U"])):
        model, _ = factory(points=20)
        fn = make_filter(model, particles=10, active_names=active)
        value, gradient = jax.value_and_grad(fn)(z, jax.random.key(81))
        assert np.isfinite(value) and np.isfinite(gradient).all()


def test_backward_score_agrees_with_linear_kalman_in_expectation():
    m, y = models.linear_gaussian(points=12)
    z = jnp.array([.5, .5])
    sample, complete, score = make_smoother("linear", m, y, particles=500)
    estimates = np.asarray(jax.jit(jax.vmap(lambda key: score(z, key)[1]))(
        jax.random.split(jax.random.key(82), 256)))
    truth = jax.grad(lambda z: models.linear_loglik(dict(zip(["a1", "a2"], z)), y))(z)
    error = np.abs(estimates.mean(0) - truth)
    assert np.all(error < 4 * estimates.std(0) / np.sqrt(len(estimates)) + .2)


def test_spx_is_pypomp_and_clipping_mass_is_in_density():
    m = models.spx()
    reference = importlib.import_module("pypomp.models.spx")
    np.testing.assert_array_equal(m.ys, reference.spx().ys)
    p = reference.theta
    z = jnp.array([reference._to_est(p)[n] for n in m.canonical_param_names])
    _, joint = densities("spx", m)
    vi = m.statenames.index("V")
    previous = jnp.array([p[n + "_0"] if n == "V" else 1105. for n in m.statenames])
    following = previous.at[vi].set(1e-32)
    v = p["V_0"]
    yprev = m.to_struct().covars_extended[0, 0]
    mean = v + p["kappa"] * (p["theta"] - v) + p["xi"] * p["rho"] * (yprev - p["mu"] + .5*v)
    sd = p["xi"] * np.sqrt(v * (1 - p["rho"]**2))
    measurement = jax.scipy.stats.norm.logpdf(m.ys.iloc[0, 0], p["mu"] - .5e-32, 1e-16)
    expected = jax.scipy.special.log_ndtr((1e-32 - mean) / sd) + measurement
    np.testing.assert_allclose(joint(z, previous, following, 0), expected)
