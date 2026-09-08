import numpy as np

from ditlevsen.bridge import block_transition, bridge_conditional


def test_block_transition_matches_manual_two_step_moments():
    transitions = np.asarray(
        [[[0.9, 0.2], [0.0, 0.8]], [[1.1, 0.0], [-0.1, 0.7]]]
    )
    offsets = np.asarray([[0.1, -0.2], [0.3, 0.4]])
    covariances = np.asarray([np.diag([0.2, 0.1]), np.diag([0.3, 0.4])])

    matrix, offset, covariance = block_transition(
        transitions, offsets, covariances
    )

    np.testing.assert_allclose(matrix, transitions[1] @ transitions[0])
    np.testing.assert_allclose(offset, transitions[1] @ offsets[0] + offsets[1])
    np.testing.assert_allclose(
        covariance,
        transitions[1] @ covariances[0] @ transitions[1].T + covariances[1],
    )


def test_bridge_conditional_matches_joint_gaussian_formula():
    transitions = np.asarray(
        [
            [[0.9, 0.1], [0.0, 0.8]],
            [[1.0, 0.2], [-0.1, 0.9]],
            [[0.8, 0.0], [0.1, 1.1]],
        ]
    )
    offsets = np.asarray([[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1]])
    covariances = np.asarray(
        [np.diag([0.3, 0.2]), np.diag([0.4, 0.1]), np.diag([0.2, 0.5])]
    )
    previous = np.asarray([0.4, -0.7])
    endpoint = np.asarray([0.2, 0.9])

    mean, covariance = bridge_conditional(
        previous, endpoint, transitions, offsets, covariances, relative_floor=0.0
    )

    next_mean = transitions[0] @ previous + offsets[0]
    remainder_matrix, remainder_offset, remainder_covariance = block_transition(
        transitions[1:], offsets[1:], covariances[1:]
    )
    endpoint_mean = remainder_matrix @ next_mean + remainder_offset
    cross = covariances[0] @ remainder_matrix.T
    endpoint_covariance = (
        remainder_matrix @ covariances[0] @ remainder_matrix.T
        + remainder_covariance
    )
    expected_gain = cross @ np.linalg.inv(endpoint_covariance)
    expected_mean = next_mean + expected_gain @ (endpoint - endpoint_mean)
    expected_covariance = covariances[0] - expected_gain @ cross.T

    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(covariance, expected_covariance)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_one_step_bridge_is_endpoint_point_mass():
    mean, covariance = bridge_conditional(
        np.asarray([0.0]),
        np.asarray([1.5]),
        np.asarray([[[0.9]]]),
        np.asarray([[0.1]]),
        np.asarray([[[0.2]]]),
    )
    np.testing.assert_allclose(mean, [1.5])
    np.testing.assert_allclose(covariance, [[0.0]])
