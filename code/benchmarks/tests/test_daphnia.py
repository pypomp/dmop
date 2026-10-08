import jax
import jax.numpy as jnp
import numpy as np

import daphnia as d
from pypomp.core.algorithms.contexts import PfilterContext
from pypomp.core.algorithms.pfilter import _pfilter_internal


def test_drift_and_measurements_match_original_model():
    p = dict(d.reference.PARAMETERS)
    for n in p:
        if n.startswith("sig"):
            p[n] = 0.
    x = jnp.array([2.4, .1, .5, .8, .1, .3, 16., 25.])
    state = {n: jnp.array(0.) for n in d.reference.STATENAMES}
    state.update(dict(zip(d.STATES, x)))
    new = d.reference.euler_step(state, p, jax.random.key(1), None, 8., .25)
    np.testing.assert_allclose(jnp.array([new[n] for n in d.STATES]),
                               x + .25*d.drift(x, p), rtol=1e-12)
    y = jnp.array([3., 1., 2., 0.])
    obs = dict(zip(["dentadult", "dentinf", "lumadult", "luminf"], y))
    np.testing.assert_allclose(d.measurement(y, jnp.array([new[n] for n in d.STATES]), p),
                               d.reference.dmeas(obs, new, p, None, 8.25))


def test_block_moments_finite_and_covariance_positive():
    p = d.parameters(jnp.array(d.pack(d.reference.sample_starts(1, 0, jitter_sd=0.)[0])), 0)
    x = jnp.array([2.333, 0., 0., .667, 0., 0., 16.667, 0.])
    mean, covariance = jax.jit(d.block_moments)(x, p, 0)
    assert np.isfinite(mean).all() and np.isfinite(covariance).all()
    assert mean[7] > 0.  # First interval includes the day-4 inoculation.
    assert np.linalg.eigvalsh(covariance).min() > 0.


def test_independent_evaluation_reuses_original_unit_filters():
    payload = d.reference.sample_starts(1, 0, jitter_sd=0.)[0]
    z = jnp.array(d.pack(payload))
    _, evaluate = d.make_euler_objectives(particles=5, eval_particles=20)
    key = jax.random.key(451)
    got = np.asarray(evaluate(z, key))
    units = d.reference._unit_models(d.reference.DATA_PATH.resolve())
    expected = []
    for i, (unit, subkey) in enumerate(zip(d.reference.UNITS, jax.random.split(key, 8))):
        model = units[unit]
        p = d.parameters(z, i)
        transformed = d.reference.to_est(p)
        theta = jnp.array([transformed[n] for n in model.canonical_param_names])
        context = PfilterContext.from_struct(model.to_struct(), J=20, should_trans=True)
        expected.append(-_pfilter_internal(theta, subkey, context)["neg_loglik"])
    np.testing.assert_allclose(got, expected, rtol=1e-10)
