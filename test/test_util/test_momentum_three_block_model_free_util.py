import numpy as np

from fi_nomad.util.momentum_three_block_model_free_util import (
    _compute_least_squares_solution,
    update_H,
    update_W,
)


def test_compute_least_squares_solution():
    # fmt: off
    X = np.asarray([
        [1.0, 2.0], 
        [3.0, 4.0], 
        [5.0, 6.0], 
        [7.0, 8.0]
    ])
    expected_result = np.asarray([
        [0.5, 0.7], 
        [-0.5, -1.2]
    ])
    # fmt: on
    actual_result = _compute_least_squares_solution(X, X @ expected_result)
    np.testing.assert_array_almost_equal(actual_result, expected_result)


def test_update_W():
    # fmt: off
    H = np.array([
        [1.0, 2.0], 
        [3.0, 4.0]
    ])
    Z = np.array([
        [5.0, 6.0], 
        [7.0, 8.0]
    ])

    # fmt: off
    expected_result = np.linalg.inv(H @ H.T) @ H @ Z.T
    expected_result = expected_result.T
    actual_result = update_W(H, Z)
    np.testing.assert_array_almost_equal(actual_result, expected_result)


def test_update_H():
    # fmt: off
    W = np.array([
        [1.0, 2.0], 
        [3.0, 4.0]
    ])
    Z = np.array([
        [5.0, 6.0], 
        [7.0, 8.0]
    ])

    # fmt: on

    # solve using direct computation of the inverse
    expected_result = np.linalg.inv(W.T @ W) @ W.T @ Z
    actual_result = update_H(W, Z)
    np.testing.assert_array_almost_equal(actual_result, expected_result)
