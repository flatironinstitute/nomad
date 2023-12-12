import numpy as np

from fi_nomad.util.base_model_free_util import (
    construct_utility,
)


def test_construct_utility() -> None:
    # fmt: off
    base_matrix = np.array([
        [ 1,  5,  0],
        [ 0,  0,  0]
    ])
    low_rank = np.array([
        [ 0, -1,  2],
        [-1,  1, -5]
    ])
    # Expect: non-zero values from base_matrix should be preserved
    # --> [ 1,  5,  0]
    #     [ 0,  0,  0]
    # Then negative values from low_rank should be passed to any
    # remaining 0s:
    # --> [ 1,  5,  0]
    #     [-1,  0, -5]
    expected_matrix = np.array([
        [ 1,  5,  0],
        [-1,  0, -5]
    ])
    # fmt: on
    result = construct_utility(low_rank, base_matrix)
    np.testing.assert_array_equal(expected_matrix, result)
