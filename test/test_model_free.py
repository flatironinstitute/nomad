import numpy as np
from unittest.mock import Mock, patch
from pytest import raises, LogCaptureFixture

from lzcompression.model_free import (
    construct_utility,
    compress_sparse_matrix,
)

from lzcompression.types import SVDStrategy


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


# TODO: Should use a test matrix...
@patch("lzcompression.model_free.construct_utility")
@patch("lzcompression.model_free.find_low_rank")
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
    result = compress_sparse_matrix(
        sparse_matrix, target_rank, strategy=SVDStrategy.FULL
    )
    # Confirm it returned the L, not the Z
    np.testing.assert_allclose(result, np.ones(sparse_matrix.shape))
    assert mock_find_lr.call_count == max_iterations
    assert mock_make_z.call_count == max_iterations


## NOTE: compute_loss, find_low_rank are not defined in model_free, but that's
# where we monkey-patch from. What gives? Because they're imported individually,
# we need to monkey-patch the ones actually in the module being tested,
# not the ones in the overall library. (There are more explicit dependency
# injection strategies for this, but that's probably overkill for us.)
@patch("lzcompression.model_free.compute_loss")
@patch("lzcompression.model_free.construct_utility")
@patch("lzcompression.model_free.find_low_rank")
def test_compress_sparse_matrix_stops_when_error_within_tolerance(
    mock_find_lr: Mock, mock_make_z: Mock, mock_loss: Mock
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
    mock_loss.side_effect = [tolerance + 1, tolerance - 1]
    result = compress_sparse_matrix(
        sparse_matrix, target_rank, tolerance=tolerance, strategy=SVDStrategy.FULL
    )
    np.testing.assert_allclose(result, mock_find_lr.return_value)
    # Called 2x: first result identified as above tolerance, second result below tolerance
    assert mock_find_lr.call_count == 2
    assert mock_make_z.call_count == 2


def test_compress_sparse_matrix_honors_verbosity(caplog: LogCaptureFixture) -> None:
    sparse_matrix = np.eye(3)
    _ = compress_sparse_matrix(sparse_matrix, 1, manual_max_iterations=0)
    assert "Initiating run" not in caplog.text
    _ = compress_sparse_matrix(sparse_matrix, 1, manual_max_iterations=0, verbose=True)
    assert "Initiating run" in caplog.text


# def test_compress_sparse_matrix_works() -> None:
#     target_rank = 10
#     n_dimension = 250
#     m_dimension = 300

#     generator = np.random.default_rng() # can pass optional seed

#     N = generator.normal(size=(n_dimension, target_rank))
#     M = generator.normal(size=(target_rank, m_dimension))

#     base_matrix = N @ M
#     sparse = np.copy(base_matrix)
#     sparse[sparse < 0] = 0

#     low_rank = compress_sparse_matrix(sparse, target_rank, tolerance=0.01)


#     raise NotImplementedError
