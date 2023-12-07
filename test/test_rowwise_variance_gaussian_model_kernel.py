import numpy as np
from unittest.mock import Mock, patch
from pytest import LogCaptureFixture

from lzcompression.kernels.rowwise_variance_gauss_model import (
    RowwiseVarianceGaussianModelKernel,
)
from lzcompression.types import KernelInputType, SVDStrategy

def make_default_kernel_initialization() -> KernelInputType:
    default_sparse = np.eye(3) * 3
    default_candidate = np.ones(default_sparse.shape)
    default_target_rank = 2
    default_svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    default_tolerance = 3.0
    return KernelInputType(
        default_sparse,
        default_candidate,
        default_target_rank,
        default_svd_strategy,
        default_tolerance,
    )

def test_rowwise_variance_gauss_model_inits_correctly() -> None:
    indata = make_default_kernel_initialization()
    variance = np.var(indata.sparse_matrix_X, axis=1)
    shape = indata.low_rank_candidate_L.shape
    denom = np.repeat(np.sqrt(variance), shape[1]).reshape(shape)
    sparse_over_sqrt_variance = indata.low_rank_candidate_L / denom

    kernel = RowwiseVarianceGaussianModelKernel(indata)
    np.testing.assert_array_equal(kernel.model_variance_sigma_squared, variance)
    np.testing.assert_array_equal(sparse_over_sqrt_variance, kernel.gamma)


def test_rowwise_variance_gauss_model_running_report() -> None:
    indata = make_default_kernel_initialization()
    kernel = RowwiseVarianceGaussianModelKernel(indata)
    txt = kernel.running_report()
    assert "Likelihoods" in txt


def test_rowwise_variance_gauss_model_final_report() -> None:
    indata = make_default_kernel_initialization()
    kernel = RowwiseVarianceGaussianModelKernel(indata)
    result = kernel.report()
    assert "0 total iterations" in result.summary
    assert "final loss" in result.summary
    assert "likelihood" in result.summary
    np.testing.assert_array_equal(result.data[0], indata.low_rank_candidate_L)
    np.testing.assert_array_equal(result.data[1], np.var(indata.sparse_matrix_X, axis=1))


def test_rowwise_variance_gauss_model_warns_on_decreased_likelihood_full_iteration(caplog: LogCaptureFixture) -> None:
    indata = make_default_kernel_initialization()
    kernel = RowwiseVarianceGaussianModelKernel(indata)
    kernel.likelihood = float("inf")
    kernel.step()
    assert "Likelihood decreased," in caplog.text


@patch("lzcompression.kernels.rowwise_variance_gauss_model.target_matrix_log_likelihood")
def test_rowwise_variance_gauss_model_warns_on_decreased_likelihood_half_iteration(mock_likelihood: Mock, caplog: LogCaptureFixture) -> None:
    mock_likelihood.side_effect = [10, -10]
    indata = make_default_kernel_initialization()
    kernel = RowwiseVarianceGaussianModelKernel(indata)
    kernel.step()
    assert "Likelihood decreased between variance update and means" in caplog.text


# This is a pretty vague test: we just want to make sure that scaling is applied appropriately during
# the means-matrix update process, as there was a bug in development where we left that out
@patch("lzcompression.kernels.rowwise_variance_gauss_model.scale_by_rowwise_stddev")
@patch("lzcompression.kernels.rowwise_variance_gauss_model.find_low_rank")
@patch("lzcompression.kernels.rowwise_variance_gauss_model.get_stddev_normalized_matrix_gamma")
def test_rowwise_variance_means_update(mock_gamma: Mock, mock_lowrank: Mock, mock_scale: Mock) -> None:
    mock_gamma.return_value = np.eye(3)
    mock_lowrank.return_value = np.eye(3) * 2
    mock_scale.return_value = np.eye(3) * 3
    post_means = np.eye(3) * 4

    indata = make_default_kernel_initialization()
    kernel = RowwiseVarianceGaussianModelKernel(indata)
    mock_gamma.reset_mock()
    mock_gamma.return_value = np.eye(3)
    kernel.do_means_update(post_means)

    mock_gamma.assert_called_once_with(post_means, kernel.model_variance_sigma_squared)
    mock_lowrank.assert_called_once_with(mock_gamma.return_value, kernel.target_rank, kernel.model_means_L, kernel.svd_strategy)
    mock_scale.assert_called_once_with(mock_lowrank.return_value, kernel.model_variance_sigma_squared)


# Again, this is mostly a sanity check that we didn't somehow skip something.
# Ought to either delete or come up with more meaningful assertions.
@patch("lzcompression.kernels.rowwise_variance_gauss_model.compute_loss")
@patch("lzcompression.kernels.rowwise_variance_gauss_model.scale_by_rowwise_stddev")
@patch("lzcompression.kernels.rowwise_variance_gauss_model.find_low_rank")
@patch("lzcompression.kernels.rowwise_variance_gauss_model.target_matrix_log_likelihood")
@patch(
    "lzcompression.kernels.rowwise_variance_gauss_model.get_stddev_normalized_matrix_gamma"
)
@patch("lzcompression.kernels.rowwise_variance_gauss_model.estimate_new_model_variance")
@patch(
    "lzcompression.kernels.rowwise_variance_gauss_model.get_elementwise_posterior_variance_dZbar"
)
@patch("lzcompression.kernels.rowwise_variance_gauss_model.get_posterior_means_Z")
def test_rowwise_variance_step_calls(
    mock_get_postmeans: Mock,
    mock_get_postvar: Mock,
    mock_new_variance: Mock,
    mock_getgamma: Mock,
    mock_likelihood: Mock,
    mock_lowrank: Mock,
    mock_scale: Mock,
    mock_loss: Mock,
) -> None:
    mock_likelihood.return_value = 5.
    indata = make_default_kernel_initialization()
    kernel = RowwiseVarianceGaussianModelKernel(indata)
    mock_getgamma.reset_mock()
    kernel.step()

    # Should be doing the posterior means and variance checks 2x each (before each update)
    assert mock_get_postmeans.call_count == 2
    assert mock_get_postvar.call_count == 2
    # estimate new variance once (before updating)
    mock_new_variance.assert_called_once()
    # likelihood 2x each (after each update step)
    assert mock_likelihood.call_count == 2
    # get-gamma 3x (once after each update, and one more during means update)
    assert mock_getgamma.call_count == 3
    # one find-low-rank call and rescale call
    mock_lowrank.assert_called_once()
    mock_scale.assert_called_once()
    # one compute-loss call
    mock_loss.assert_called_once()
