import numpy as np
from unittest.mock import Mock, patch
from pytest import LogCaptureFixture

from lzcompression.kernels.single_variance_gauss_model import (
    SingleVarianceGaussianModelKernel,
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


def test_single_variance_gauss_model_inits_correctly() -> None:
    indata = make_default_kernel_initialization()

    variance = float(np.var(indata.sparse_matrix_X))  # s.b. 2.0 with eye(3) * 3
    # just a reminder, this is initialized based on the current estimate for the means
    # *not* on the input sparse matrix
    sparse_over_sqrt_variance = indata.low_rank_candidate_L / np.sqrt(variance)

    kernel = SingleVarianceGaussianModelKernel(indata)
    assert kernel.model_variance_sigma_squared == variance
    np.testing.assert_array_equal(sparse_over_sqrt_variance, kernel.gamma)


def test_single_variance_gauss_model_running_report() -> None:
    indata = make_default_kernel_initialization()
    kernel = SingleVarianceGaussianModelKernel(indata)
    txt = kernel.running_report()
    assert "Iteration 0" in txt


def test_single_variance_gauss_model_final_report() -> None:
    indata = make_default_kernel_initialization()
    kernel = SingleVarianceGaussianModelKernel(indata)
    result = kernel.report()
    assert "0 total iterations" in result.summary
    assert "final loss" in result.summary
    assert "likelihood" in result.summary
    np.testing.assert_array_equal(result.data[0], indata.low_rank_candidate_L)
    assert result.data[1] == float(np.var(indata.sparse_matrix_X))


# TODO: These are not very interesting assertions
@patch("lzcompression.kernels.single_variance_gauss_model.compute_loss")
@patch("lzcompression.kernels.single_variance_gauss_model.target_matrix_log_likelihood")
@patch(
    "lzcompression.kernels.single_variance_gauss_model.get_stddev_normalized_matrix_gamma"
)
@patch(
    "lzcompression.kernels.single_variance_gauss_model.estimate_new_model_global_variance"
)
@patch("lzcompression.kernels.single_variance_gauss_model.find_low_rank")
@patch(
    "lzcompression.kernels.single_variance_gauss_model.get_elementwise_posterior_variance_dZbar"
)
@patch("lzcompression.kernels.single_variance_gauss_model.get_posterior_means_Z")
def test_single_variance_gauss_model_step(
    mock_get_postmeans: Mock,
    mock_get_postvar: Mock,
    mock_lowrank: Mock,
    mock_new_variance: Mock,
    mock_getgamma: Mock,
    mock_likelihood: Mock,
    mock_loss: Mock,
) -> None:
    mock_get_postmeans.return_value = np.full((3, 3), 1)
    mock_get_postvar.return_value = np.full((3, 3), 2)
    mock_lowrank.return_value = np.full((3, 3), 3)
    mock_new_variance.return_value = 6.0
    mock_getgamma.return_value = np.full((3, 3), 4)
    mock_likelihood.return_value = 0.5
    mock_loss.return_value = 1.0

    indata = make_default_kernel_initialization()
    kernel = SingleVarianceGaussianModelKernel(indata)

    kernel.step()

    mock_get_postmeans.assert_called_once()
    mock_get_postvar.assert_called_once()
    mock_lowrank.assert_called_once()
    mock_new_variance.assert_called_once()
    mock_getgamma.assert_called()
    mock_likelihood.assert_called_once()
    mock_loss.assert_called_once()

    np.testing.assert_array_equal(kernel.model_means_L, mock_lowrank.return_value)
    np.testing.assert_array_equal(kernel.gamma, mock_getgamma.return_value)
    assert kernel.model_variance_sigma_squared == mock_new_variance.return_value
    assert kernel.likelihood == mock_likelihood.return_value
    assert kernel.loss == mock_loss.return_value


def test_single_variance_gauss_model_warns_on_decreased_likelihood(
    caplog: LogCaptureFixture,
) -> None:
    indata = make_default_kernel_initialization()
    kernel = SingleVarianceGaussianModelKernel(indata)
    kernel.likelihood = float("inf")
    kernel.step()
    assert "Likelihood decreased" in caplog.text
