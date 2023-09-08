import numpy as np
from unittest.mock import Mock, patch
from pytest import approx, raises, LogCaptureFixture
import itertools
from typing import Any

from lzcompression.types import InitializationStrategy, SVDStrategy, LossType
from lzcompression.gauss_model import (
    estimate_gaussian_model,
    get_posterior_means_Z,
    estimate_new_model_variance,
    get_stddev_normalized_matrix_gamma,
    get_elementwise_posterior_variance_dZbar,
    target_matrix_log_likelihood,
)


def test_estimate_gaussian_model_throws_on_negative_elements() -> None:
    # fmt: off
    bad_matrix = np.array(
        [[3, 2,  2],
         [2, 3, -2],]
    )
    # fmt: on
    with raises(ValueError, match="nonnegative"):
        _ = estimate_gaussian_model(bad_matrix, 4)


@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.find_low_rank")
@patch("lzcompression.gauss_model.get_posterior_means_Z")
def test_estimate_gaussian_model_obeys_max_iterations(
    mock_makeZ: Mock, mock_lr: Mock, mock_loss: Mock
) -> None:
    target_rank = 5
    max_iter = target_rank * 100
    tolerance = 0.01
    sparse = np.eye(3)

    mock_lr.return_value = np.zeros(sparse.shape)
    mock_makeZ.return_value = np.ones(sparse.shape)
    mock_loss.return_value = tolerance + 1

    (result, _) = estimate_gaussian_model(sparse, target_rank)
    np.testing.assert_allclose(result, mock_lr.return_value)
    assert mock_loss.call_count == max_iter
    assert mock_makeZ.call_count == max_iter
    assert mock_lr.call_count == max_iter


@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.find_low_rank")
@patch("lzcompression.gauss_model.get_posterior_means_Z")
@patch("lzcompression.gauss_model.initialize_low_rank_candidate")
def test_estimate_gaussian_model_honors_strategy_choices(
    mock_init: Mock, mock_makeZ: Mock, mock_lr: Mock, mock_loss: Mock
) -> None:
    target_rank = 1
    sparse = np.eye(3)

    mock_init.return_value = np.ones(sparse.shape) * 0.33
    mock_lr.return_value = np.zeros(sparse.shape)
    mock_makeZ.return_value = np.ones(sparse.shape)
    mock_loss.return_value = 1

    _ = estimate_gaussian_model(sparse, target_rank)
    assert mock_init.call_args.args[-1] == InitializationStrategy.BROADCAST_MEAN
    assert mock_lr.call_args.args[-1] == SVDStrategy.RANDOM_TRUNCATED

    _ = estimate_gaussian_model(
        sparse,
        target_rank,
        svd_strategy=SVDStrategy.EXACT_TRUNCATED,
        initialization=InitializationStrategy.COPY,
    )
    assert mock_init.call_args.args[-1] == InitializationStrategy.COPY
    assert mock_lr.call_args.args[-1] == SVDStrategy.EXACT_TRUNCATED

    _ = estimate_gaussian_model(sparse, target_rank, svd_strategy=SVDStrategy.FULL)
    assert mock_lr.call_args.args[-1] == SVDStrategy.FULL


@patch("lzcompression.gauss_model.compute_loss")
@patch("lzcompression.gauss_model.estimate_new_model_variance")
@patch("lzcompression.gauss_model.get_posterior_means_Z")
@patch("lzcompression.gauss_model.find_low_rank")
def test_estimate_gaussian_model_stops_when_error_within_tolerance(
    mock_lr: Mock, mock_makeZ: Mock, mock_var: Mock, mock_loss: Mock
) -> None:
    tolerance = 5
    target_rank = 5
    sparse_matrix = np.eye(3)

    mock_lr.return_value = np.ones(sparse_matrix.shape)
    mock_makeZ.return_value = np.zeros(sparse_matrix.shape)
    mock_var.return_value = 5
    mock_loss.side_effect = [tolerance + 1, tolerance - 1]

    (result, var) = estimate_gaussian_model(
        sparse_matrix, target_rank, tolerance=tolerance
    )
    np.testing.assert_allclose(result, mock_lr.return_value)
    assert var == mock_var.return_value
    # Called 2x: first result identified as above tolerance, second result below tolerance
    assert mock_lr.call_count == 2
    assert mock_makeZ.call_count == 2


@patch("lzcompression.gauss_model.low_rank_matrix_log_likelihood")
@patch("lzcompression.gauss_model.get_posterior_means_Z")
@patch("lzcompression.gauss_model.find_low_rank")
def test_estimate_gaussian_model_warns_on_nondecreasing_likelihood(
    mock_lr: Mock, mock_makeZ: Mock, mock_likelihood: Mock, caplog: LogCaptureFixture
) -> None:
    target_rank = 5
    sparse_matrix = np.eye(3)

    # python mocks don't have an easy way to say "give me these X values, then a default"
    # (instead they start throwing errors or something)
    # so we define a function that does that, and tell the mock to use that function.
    vals = itertools.chain([10.0, 9.0], itertools.repeat(5.0))

    def se(*args: list[Any]) -> float:
        for i in vals:
            return i
        return -1.0

    mock_lr.return_value = np.ones(sparse_matrix.shape)
    mock_makeZ.return_value = np.zeros(sparse_matrix.shape)
    mock_likelihood.side_effect = se
    _ = estimate_gaussian_model(sparse_matrix, target_rank)
    assert "likelihood decreased" in caplog.text


def test_estimate_gaussian_model_honors_verbosity(
    caplog: LogCaptureFixture,
) -> None:
    sparse_matrix = np.eye(3)
    _ = estimate_gaussian_model(sparse_matrix, 1, manual_max_iterations=0)
    assert "Initiating run" not in caplog.text
    _ = estimate_gaussian_model(sparse_matrix, 1, manual_max_iterations=0, verbose=True)
    assert "Initiating run" in caplog.text


def test_estimate_gaussian_model_works() -> None:
    sparse_matrix = np.array(
        [
            [5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0],
        ]
    )
    (l, _) = estimate_gaussian_model(sparse_matrix, 6)
    relu_l = np.copy(l)
    relu_l[relu_l < 0] = 0
    assert np.allclose(relu_l, sparse_matrix)


@patch("lzcompression.gauss_model.pdf_to_cdf_ratio_psi")
def test_construct_posterior_means_matrix_Z(mock_psi: Mock) -> None:
    mock_psi.side_effect = lambda x: x
    sparse = np.eye(3)
    low_rank = np.ones(sparse.shape) * 10
    stddev_norm = np.ones(sparse.shape) * 3
    sigma_sq = 16
    res = get_posterior_means_Z(low_rank, sparse, stddev_norm, sigma_sq)
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


def test_get_stddev_normalized_matrix_gamma() -> None:
    input = np.eye(3) * 16
    sigma_squared = 4
    expected_result = np.eye(3) * 8
    res = get_stddev_normalized_matrix_gamma(input, sigma_squared)
    np.testing.assert_allclose(expected_result, res)


def test_get_elementwise_posterior_variance_dZbar_uses_0_variance_for_known_values() -> (
    None
):
    sparse = np.eye(3)
    var = 1.0
    model = np.array(range(9)).reshape((3, 3)) * 0.1
    res = get_elementwise_posterior_variance_dZbar(sparse, var, model)
    np.testing.assert_allclose(res[sparse > 0], np.zeros((3, 3))[sparse > 0])
    a = res[sparse == 0]
    b = sparse[sparse == 0]
    c = model[sparse == 0]
    for i in range(len(a)):
        assert a[i] != b[i]
        assert a[i] != c[i]


def test_get_elementwise_posterior_variance_dZbar_computes_variances_where_needed() -> (
    None
):
    sparse = np.eye(3)
    var = 2.0
    stddev_normalized_model_matrix = np.array(range(9)).reshape((3, 3)) * 0.1
    # fmt: off
    expected_result = np.array([
        [0.72676046, 0.68430569, 0.6441387],
        [0.60622898, 0.57052543, 0.53696081],
        [0.50545572, 0.47592198, 0.4482657]
    ])
    # fmt: on
    result = get_elementwise_posterior_variance_dZbar(
        sparse, var, stddev_normalized_model_matrix
    )
    np.testing.assert_allclose(result[sparse == 0], expected_result[sparse == 0])
    np.testing.assert_allclose(result[sparse > 0], np.zeros((3, 3))[sparse > 0])


def test_target_matrix_log_likelihood() -> None:
    sparse = np.eye(3)
    low_rank = (
        np.ones((3, 3)) * 2
    )  # this is not actually low rank but that doesn't matter
    sigma_sq = 9
    # this will result in us taking logpdf(1, loc=2, scale=3) for the 1s in the
    # identity matrix. That's 3 values at ~ -2.073106 each.
    stddev_norm_lr = np.ones(
        (3, 3)
    )  # obviously not the actual std-dev-normalized values of low_rank
    # we just need something identifiable for sparse's zero-valued entries
    # normal.logcdf of -1 is ~ -1.841021645, and we'll have 6 of those
    expected = (3 * -2.073106) + (6 * -1.841021645)
    result = target_matrix_log_likelihood(sparse, low_rank, stddev_norm_lr, sigma_sq)
    assert approx(result) == expected
