import jax
import jax.numpy as jnp
import numpy as np

from ditlevsen.block_saem import averaged_objective, maximize_objective, update_weights
from ditlevsen.block_smc import _complete_block_path_loglik, sample_block_smoothing_path
from ditlevsen.data import load_dacca_data
from ditlevsen.model import default_unconstrained_parameters


def test_averaged_objective_recomputes_old_paths_at_current_parameter():
    # Gaussian complete-data M-step has a known solution; averaging scores
    # evaluated at old parameters would give a different update.
    paths = jnp.array([[1.], [5.], [jnp.nan]])
    weights = update_weights(np.zeros(3), 0, 1.)
    weights = update_weights(weights, 1, .25)
    vg = averaged_objective(lambda theta, path: -.5*jnp.sum((theta-path)**2))
    value, grad = vg(jnp.array([4.]), paths, weights)
    np.testing.assert_allclose(value, -3.5)
    np.testing.assert_allclose(grad, [-2.])
    optimum, details = maximize_objective(lambda z: vg(z, paths, weights),
                                          np.array([4.]), maxiter=30)
    np.testing.assert_allclose(optimum, [2.], atol=1e-6)
    assert details['accepted'] and details['q_after'] > details['q_before']


def test_gain_one_discards_the_previous_objective():
    weights = update_weights(np.zeros(4), 0, 1.)
    weights = update_weights(weights, 1, .25)
    np.testing.assert_allclose(weights, [.75, .25, 0., 0.])
    np.testing.assert_allclose(update_weights(weights, 2, 1.), [0., 0., 1., 0.])


def test_dhaka_average_matches_separate_complete_path_evaluations():
    data = load_dacca_data(20, max_observations=3)
    theta = jnp.asarray(default_unconstrained_parameters())
    paths = jnp.asarray([sample_block_smoothing_path(theta, data, particles=16,
        seed=seed, proposal='guided').path[1:] for seed in (2, 3)])
    def ll(z, path):
        # Use a well-conditioned covariance to isolate objective averaging.
        # The unchanged 1e-12 floor in the Dhaka pilot is separately diagnosed.
        return _complete_block_path_loglik(z, path, jnp.asarray(data.step_covariates),
            jnp.asarray(data.observation_covariates[:, 2]), jnp.asarray(data.observations), 1e-8)
    vg = averaged_objective(ll)
    actual, grad = vg(theta, paths, jnp.array([.7, .3]))
    first, second = [jax.value_and_grad(ll)(theta, path) for path in paths]
    expected = .7*first[0]+.3*second[0]
    expected_grad = .7*first[1]+.3*second[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-7, atol=1e-6)
