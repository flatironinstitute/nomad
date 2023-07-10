import numpy as np
import sys
from unittest.mock import Mock, patch
from pytest import raises
from typing import cast

from lzcompression.lzcomp import (
    InitializationStrategy,
    initialize_low_rank_candidate,
    construct_utility,
    find_low_rank,
    compress_sparse_matrix,
)


def test_initialize_returns_copy() -> None:
    # fmt: off
    base_matrix = np.array([
        [ 1,  5,  0],
        [ 0,  0,  0]
    ])
    # fmt: on
    res = initialize_low_rank_candidate(base_matrix, InitializationStrategy.COPY)
    np.testing.assert_allclose(base_matrix, res)


def test_initialize_returns_mean() -> None:
    # fmt: off
    base_matrix = np.array([
        [ 1,  5,  0],
        [ 0,  0,  0]
    ])
    # fmt: on
    res = initialize_low_rank_candidate(
        base_matrix, InitializationStrategy.BROADCAST_MEAN
    )
    expected = np.ones((2, 3))
    np.testing.assert_allclose(res, expected)


def test_initialize_throws_on_unknown_strategy() -> None:
    base_matrix = np.ones((2, 2))
    with raises(ValueError, match="Unsupported"):
        _ = initialize_low_rank_candidate(base_matrix, cast(InitializationStrategy, -5))


def test_find_low_rank_uses_preallocated_memory() -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    allocated_memory = np.zeros(arbitrary_matrix.shape)
    result = find_low_rank(arbitrary_matrix, 3, allocated_memory)
    assert result is allocated_memory


def test_find_low_rank_matches_shape_of_input_matrix() -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    allocated_memory = np.zeros(arbitrary_matrix.shape)
    result = find_low_rank(arbitrary_matrix, 3, allocated_memory)
    assert result.shape == arbitrary_matrix.shape


def test_find_low_rank_recovers_known_result() -> None:
    # fmt: off
    low_rank_input = np.array([
        [1,  4, -3],
        [2,  8, -6],
        [3, 12, -9]
    ])
    # fmt: on
    allocated_memory = np.zeros(low_rank_input.shape)
    result = find_low_rank(low_rank_input, 1, allocated_memory)
    np.testing.assert_array_almost_equal(low_rank_input, result)


def test_find_low_rank_recovers_full_rank_input_when_allowed_full_rank() -> None:
    # fmt: off
    base_matrix = np.array([
        [3, 2,  2],
        [2, 3, -2]
    ])
    # fmt: on
    allocated_memory = np.zeros(base_matrix.shape)
    returned_val = find_low_rank(base_matrix, sys.maxsize, allocated_memory)
    np.testing.assert_array_almost_equal(base_matrix, returned_val)


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


def test_compress_sparse_matrix_throws_if_input_has_negative_elements() -> None:
    bad_matrix = np.array([[3, 2, 2], [2, 3, -2]])
    with raises(ValueError, match="nonnegative"):
        _ = compress_sparse_matrix(bad_matrix, 4)


@patch("lzcompression.lzcomp.construct_utility")
@patch("lzcompression.lzcomp.find_low_rank")
def test_compress_sparse_matrix_obeys_max_iterations(
    mock_find_lr: Mock, mock_make_z: Mock
) -> None:
    target_rank = 5
    max_iterations = 100 * target_rank
    # fmt: off
    sparse_matrix = np.array([
        [3, 2, 2],
        [2, 3, 1]
    ])
    # fmt: on
    mock_find_lr.return_value = np.ones(sparse_matrix.shape)
    mock_make_z.return_value = np.zeros(sparse_matrix.shape)
    result = compress_sparse_matrix(sparse_matrix, target_rank)
    # Confirm it returned the L, not the Z
    np.testing.assert_allclose(result, np.ones(sparse_matrix.shape))
    assert mock_find_lr.call_count == max_iterations
    assert mock_make_z.call_count == max_iterations


@patch("numpy.linalg.norm")
@patch("lzcompression.lzcomp.construct_utility")
@patch("lzcompression.lzcomp.find_low_rank")
def test_compress_sparse_matrix_stops_when_error_within_tolerance(
    mock_find_lr: Mock, mock_make_z: Mock, mock_loss_norm: Mock
) -> None:
    tolerance = 5
    target_rank = 5
    # fmt: off
    sparse_matrix = np.array([
        [3, 2, 2],
        [2, 3, 1]
    ])
    # fmt: on
    mock_find_lr.return_value = np.zeros(sparse_matrix.shape)
    mock_make_z.return_value = np.ones(sparse_matrix.shape)
    mock_loss_norm.side_effect = [tolerance + 1, tolerance - 1]
    result = compress_sparse_matrix(sparse_matrix, target_rank, tolerance)
    np.testing.assert_allclose(result, mock_find_lr.return_value)
    # Called 2x: first result identified as above tolerance, second result below tolerance
    assert mock_find_lr.call_count == 2
    assert mock_make_z.call_count == 2


# TODO: Check compress_sparse_matrix gives right answer
