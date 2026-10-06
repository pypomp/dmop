import numpy as np

from ditlevsen.data import load_dacca_data
from ditlevsen.kbridge import _conditional_guide, bridge_cpf_rejuvenate
from ditlevsen.model import default_unconstrained_parameters
from ditlevsen.smc import complete_path_loglik, sample_smoothing_path


def test_bridge_cpf_preserves_month_endpoints_and_finite_target():
    data = load_dacca_data(5, max_observations=3)
    parameters = default_unconstrained_parameters()
    initial = sample_smoothing_path(
        parameters,
        data,
        particles=24,
        seed=17,
        order=2,
        relative_floor=1e-7,
    )
    result = bridge_cpf_rejuvenate(
        parameters,
        initial.path,
        data,
        particles=8,
        seed=18,
        order=2,
        relative_floor=1e-7,
    )
    assert result.path.shape == initial.path.shape
    np.testing.assert_allclose(result.path[5::5], initial.path[5::5])
    assert result.updated_months.shape == (3,)
    assert 0.0 <= result.update_fraction <= 1.0
    assert np.isfinite(result.median_ess)
    assert np.isfinite(
        complete_path_loglik(
            parameters,
            result.path,
            data,
            order=2,
            relative_floor=1e-7,
        )
    )


def test_conditional_guide_matches_direct_gaussian_conditioning():
    rng = np.random.default_rng(381)
    matrices = np.stack(
        [np.eye(6) + 0.03 * rng.normal(size=(6, 6)) for _ in range(3)]
    )[None, ...]
    offsets = rng.normal(scale=0.1, size=(1, 3, 6))
    roots = rng.normal(scale=0.05, size=(1, 3, 6, 6))
    covariances = roots @ np.swapaxes(roots, -1, -2) + 0.02 * np.eye(6)
    endpoint = rng.normal(size=(1, 6))
    conditional_matrix, conditional_offset, conditional_covariance = (
        _conditional_guide(matrices, offsets, covariances, endpoint)
    )

    previous = rng.normal(size=6)
    step = 0
    remainder_matrix = matrices[0, 2] @ matrices[0, 1]
    remainder_offset = matrices[0, 2] @ offsets[0, 1] + offsets[0, 2]
    remainder_covariance = (
        matrices[0, 2] @ covariances[0, 1] @ matrices[0, 2].T
        + covariances[0, 2]
    )
    one_step_mean = matrices[0, step] @ previous + offsets[0, step]
    endpoint_covariance = (
        remainder_matrix
        @ covariances[0, step]
        @ remainder_matrix.T
        + remainder_covariance
    )
    gain = np.linalg.solve(
        endpoint_covariance,
        (covariances[0, step] @ remainder_matrix.T).T,
    ).T
    expected_mean = one_step_mean + gain @ (
        endpoint[0] - remainder_matrix @ one_step_mean - remainder_offset
    )
    expected_covariance = (
        covariances[0, step]
        - gain @ remainder_matrix @ covariances[0, step]
    )

    np.testing.assert_allclose(
        conditional_matrix[0, step] @ previous + conditional_offset[0, step],
        expected_mean,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        conditional_covariance[0, step],
        expected_covariance,
        rtol=1e-11,
        atol=1e-11,
    )
