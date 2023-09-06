import numpy as np
from unittest.mock import Mock, patch
from pytest import raises
import itertools
from typing import Any

from lzcompression.types import InitializationStrategy, SVDStrategy, LossType
from lzcompression.gauss_model import (
    compress_sparse_matrix_probabilistic,
    construct_posterior_model_matrix_Z,
    estimate_new_model_variance,
)


def test_compress_sparse_matrix_probabilistic_throws_on_negative_elements() -> None:
    # fmt: off
    bad_matrix = np.array(
        [[3, 2,  2],
         [2, 3, -2],]
    )
    # fmt: on
    with raises(ValueError, match="nonnegative"):
        _ = compress_sparse_matrix_probabilistic(bad_matrix, 4)


@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.find_low_rank")
@patch("lzcompression.gauss_model.construct_posterior_model_matrix_Z")
def test_compress_sparse_matrix_probabilistic_obeys_max_iterations(
    mock_makeZ: Mock, mock_lr: Mock, mock_loss: Mock
) -> None:
    target_rank = 5
    max_iter = target_rank * 100
    tolerance = 0.01
    sparse = np.eye(3)

    mock_lr.return_value = np.zeros(sparse.shape)
    mock_makeZ.return_value = np.ones(sparse.shape)
    mock_loss.return_value = tolerance + 1

    (result, _) = compress_sparse_matrix_probabilistic(sparse, target_rank)
    np.testing.assert_allclose(result, mock_lr.return_value)
    assert mock_loss.call_count == max_iter
    assert mock_makeZ.call_count == max_iter
    assert mock_lr.call_count == max_iter


@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.find_low_rank")
@patch("lzcompression.gauss_model.construct_posterior_model_matrix_Z")
@patch("lzcompression.gauss_model.initialize_low_rank_candidate")
def test_compress_sparse_matrix_probabilistic_honors_strategy_choices(
    mock_init: Mock, mock_makeZ: Mock, mock_lr: Mock, mock_loss: Mock
) -> None:
    target_rank = 1
    sparse = np.eye(3)

    mock_init.return_value = np.ones(sparse.shape) * 0.33
    mock_lr.return_value = np.zeros(sparse.shape)
    mock_makeZ.return_value = np.ones(sparse.shape)
    mock_loss.return_value = 1

    _ = compress_sparse_matrix_probabilistic(sparse, target_rank)
    assert mock_init.call_args.args[-1] == InitializationStrategy.BROADCAST_MEAN
    assert mock_lr.call_args.args[-1] == SVDStrategy.RANDOM_TRUNCATED

    _ = compress_sparse_matrix_probabilistic(
        sparse,
        target_rank,
        svd_strategy=SVDStrategy.EXACT_TRUNCATED,
        initialization=InitializationStrategy.COPY,
    )
    assert mock_init.call_args.args[-1] == InitializationStrategy.COPY
    assert mock_lr.call_args.args[-1] == SVDStrategy.EXACT_TRUNCATED

    _ = compress_sparse_matrix_probabilistic(
        sparse, target_rank, svd_strategy=SVDStrategy.FULL
    )
    assert mock_lr.call_args.args[-1] == SVDStrategy.FULL


@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.estimate_new_model_variance")
@patch("lzcompression.gauss_model.construct_posterior_model_matrix_Z")
@patch("lzcompression.gauss_model.find_low_rank")
def test_compress_sparse_matrix_probabilistic_stops_when_error_within_tolerance(
    mock_lr: Mock, mock_makeZ: Mock, mock_var: Mock, mock_loss: Mock
) -> None:
    tolerance = 5
    target_rank = 5
    sparse_matrix = np.eye(3)

    mock_lr.return_value = np.ones(sparse_matrix.shape)
    mock_makeZ.return_value = np.zeros(sparse_matrix.shape)
    mock_var.return_value = 5
    mock_loss.side_effect = [tolerance + 1, tolerance - 1]

    (result, var) = compress_sparse_matrix_probabilistic(
        sparse_matrix, target_rank, tolerance=tolerance
    )
    np.testing.assert_allclose(result, mock_lr.return_value)
    assert var == mock_var.return_value
    # Called 2x: first result identified as above tolerance, second result below tolerance
    assert mock_lr.call_count == 2
    assert mock_makeZ.call_count == 2


### These two are super brittle and not terribly informative, should probably just cut.
@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.construct_posterior_model_matrix_Z")
@patch("lzcompression.gauss_model.find_low_rank")
@patch("builtins.print")
def test_compress_sparse_matrix_probabilistic_honors_running_loss_printout(
    mock_print: Mock, mock_lr: Mock, mock_makeZ: Mock, mock_loss: Mock
) -> None:
    tolerance = 5
    target_rank = 5
    sparse_matrix = np.eye(3)

    mock_lr.return_value = np.ones(sparse_matrix.shape)
    mock_makeZ.return_value = np.zeros(sparse_matrix.shape)
    mock_loss.side_effect = [tolerance + 1, tolerance - 1]
    _ = compress_sparse_matrix_probabilistic(
        sparse_matrix, target_rank, tolerance=tolerance, print_running_loss=True
    )
    loss_print = mock_print.call_args_list[1]  # First call is the parameters report
    assert loss_print[0] == (f"\t\tIteration 1: loss={tolerance + 1}",)


### These two are super brittle and not terribly informative, should probably just cut.
@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.low_rank_matrix_log_likelihood")
@patch("lzcompression.gauss_model.construct_posterior_model_matrix_Z")
@patch("lzcompression.gauss_model.find_low_rank")
@patch("builtins.print")
def test_compress_sparse_matrix_probabilistic_honors_running_likelihood_printout(
    mock_print: Mock,
    mock_lr: Mock,
    mock_makeZ: Mock,
    mock_likelihood: Mock,
    mock_loss: Mock,
) -> None:
    tolerance = 5
    target_rank = 5
    sparse_matrix = np.eye(3)

    mock_lr.return_value = np.ones(sparse_matrix.shape)
    mock_makeZ.return_value = np.zeros(sparse_matrix.shape)
    mock_likelihood.return_value = 12
    mock_loss.side_effect = [tolerance + 1, tolerance - 1]
    _ = compress_sparse_matrix_probabilistic(
        sparse_matrix, target_rank, tolerance=tolerance, print_running_likelihood=True
    )
    loss_print = mock_print.call_args_list[1]
    assert loss_print[0] == (
        f"\t\tIteration 1: likelihood={mock_likelihood.return_value}",
    )


# And this is still a little brittle--ought to use better support for regex-style checking of response.
@patch("lzcompression.gauss_model.low_rank_matrix_log_likelihood")
@patch("lzcompression.gauss_model.construct_posterior_model_matrix_Z")
@patch("lzcompression.gauss_model.find_low_rank")
@patch("builtins.print")
def test_compress_sparse_matrix_probabilistic_warns_on_nondecreasing_likelihood(
    mock_print: Mock, mock_lr: Mock, mock_makeZ: Mock, mock_likelihood: Mock
) -> None:
    target_rank = 5
    sparse_matrix = np.eye(3)

    vals = itertools.chain([10.0, 9.0], itertools.repeat(5.0))

    def se(*args: list[Any]) -> float:
        for i in vals:
            return i
        return -1.0

    mock_lr.return_value = np.ones(sparse_matrix.shape)
    mock_makeZ.return_value = np.zeros(sparse_matrix.shape)
    mock_likelihood.side_effect = se
    _ = compress_sparse_matrix_probabilistic(
        sparse_matrix, target_rank, print_running_likelihood=False
    )
    loss_print = mock_print.call_args_list[-1]
    assert loss_print[0] == (
        f"\tWARNING: Likelihood decreased in one or more iterations.",
    )


# def test_compress_sparse_matrix_probabilistic_works() -> None:
#     raise NotImplementedError


@patch("lzcompression.gauss_model.pdf_to_cdf_ratio_psi")
def test_construct_posterior_model_matrix_Z(mock_psi: Mock) -> None:
    mock_psi.side_effect = lambda x: x
    sparse = np.eye(3)
    low_rank = np.ones(sparse.shape) * 10
    stddev_norm = np.ones(sparse.shape) * 3
    sigma_sq = 16
    res = construct_posterior_model_matrix_Z(low_rank, sparse, stddev_norm, sigma_sq)
    expected = np.eye(3)
    expected[expected == 0] = 22.0
    np.testing.assert_allclose(res, expected)


def test_estimate_new_model_variance() -> None:
    utility = np.ones((3, 3)) * 7
    lr = np.ones((3, 3)) * 4
    var = np.ones((3, 3)) * 2
    # Should be taking the mean of a matrix, each of whose elements are
    # (7-4)^2 + 2 = 9 + 2 = 11
    expected = 11.0
    result = estimate_new_model_variance(utility, lr, var)
    assert result == expected
