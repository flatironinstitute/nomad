from typing import Tuple, cast
import numpy as np
from unittest.mock import Mock, patch
from pytest import LogCaptureFixture
import pytest

from fi_nomad.kernels import (
    RowwiseVarianceGaussianModelKernel,
)
from fi_nomad.types import (
    KernelInputType,
    RowwiseVarianceGaussianModelKernelReturnType,
    SVDStrategy,
)

Fixture = Tuple[KernelInputType, RowwiseVarianceGaussianModelKernel]
PKG = "fi_nomad.kernels.rowwise_variance_gauss_model"


@pytest.fixture
def fixture() -> Fixture:
    default_sparse = np.eye(3) * 3
    default_candidate = np.ones(default_sparse.shape)
    default_target_rank = 2
    default_svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    default_tolerance = 3.0
    indata = KernelInputType(
        default_sparse,
        default_candidate,
        default_target_rank,
        default_svd_strategy,
        default_tolerance,
    )
    kernel = RowwiseVarianceGaussianModelKernel(indata)
    return (indata, kernel)


def test_rowwise_variance_gauss_model_inits_correctly(fixture: Fixture) -> None:
    (indata, kernel) = fixture
    variance = np.var(indata.sparse_matrix_X, axis=1)
    shape = indata.low_rank_candidate_L.shape
    denom = np.repeat(np.sqrt(variance), shape[1]).reshape(shape)
    sparse_over_sqrt_variance = indata.low_rank_candidate_L / denom

    np.testing.assert_array_equal(kernel.model_variance_sigma_squared, variance)
    np.testing.assert_array_equal(sparse_over_sqrt_variance, kernel.gamma)


def test_rowwise_variance_gauss_model_running_report(fixture: Fixture) -> None:
    (_, kernel) = fixture
    txt = kernel.running_report()
    assert "Likelihoods" in txt


def test_rowwise_variance_gauss_model_final_report(fixture: Fixture) -> None:
    (indata, kernel) = fixture
    result = kernel.report()
    assert "0 total iterations" in result.summary
    assert "final loss" in result.summary
    assert "likelihood" in result.summary
    np.testing.assert_allclose(
        indata.low_rank_candidate_L,
        result.data.factors[0] @ result.data.factors[1],
    )
    result_data = cast(RowwiseVarianceGaussianModelKernelReturnType, result.data)
    np.testing.assert_array_equal(
        result_data.variance, np.var(indata.sparse_matrix_X, axis=1)
    )


def test_rowwise_variance_gauss_model_warns_on_decreased_likelihood_full_iteration(
    caplog: LogCaptureFixture, fixture: Fixture
) -> None:
    (_, kernel) = fixture
    kernel.likelihood = float("inf")
    kernel.step()
    assert "Likelihood decreased," in caplog.text


@patch(f"{PKG}.target_matrix_log_likelihood")
def test_rowwise_variance_gauss_model_warns_on_decreased_likelihood_half_iteration(
    mock_likelihood: Mock, caplog: LogCaptureFixture, fixture: Fixture
) -> None:
    (_, kernel) = fixture
    mock_likelihood.side_effect = [10, -10]
    kernel.step()
    assert "Likelihood decreased between variance update and means" in caplog.text


# This is a pretty vague test: we just want to make sure that scaling is applied appropriately during
# the means-matrix update process, as there was a bug in development where we left that out
@patch(f"{PKG}.scale_by_rowwise_stddev")
@patch(f"{PKG}.find_low_rank")
@patch(f"{PKG}.get_stddev_normalized_matrix_gamma")
def test_rowwise_variance_means_update(
    mock_gamma: Mock, mock_lowrank: Mock, mock_scale: Mock, fixture: Fixture
) -> None:
    mock_gamma.return_value = np.eye(3)
    mock_lowrank.return_value = np.eye(3) * 2
    mock_scale.return_value = np.eye(3) * 3
    post_means = np.eye(3) * 4

    (_, kernel) = fixture
    mock_gamma.reset_mock()
    mock_gamma.return_value = np.eye(3)
    kernel.make_updated_means_estimate(post_means)

    mock_gamma.assert_called_once_with(post_means, kernel.model_variance_sigma_squared)
    mock_lowrank.assert_called_once_with(
        mock_gamma.return_value,
        kernel.target_rank,
        kernel.model_means_L,
        kernel.svd_strategy,
    )
    mock_scale.assert_called_once_with(
        mock_lowrank.return_value, kernel.model_variance_sigma_squared
    )


# Again, this is mostly a sanity check that we didn't somehow skip something.
# Ought to either delete or come up with more meaningful assertions.
@patch(f"{PKG}.compute_loss")
@patch(f"{PKG}.scale_by_rowwise_stddev")
@patch(f"{PKG}.find_low_rank")
@patch(f"{PKG}.target_matrix_log_likelihood")
@patch(f"{PKG}.get_stddev_normalized_matrix_gamma")
@patch(f"{PKG}.estimate_new_model_variance")
@patch(f"{PKG}.get_elementwise_posterior_variance_dZbar")
@patch(f"{PKG}.get_posterior_means_Z")
def test_rowwise_variance_step_calls(
    mock_get_postmeans: Mock,
    mock_get_postvar: Mock,
    mock_new_variance: Mock,
    mock_getgamma: Mock,
    mock_likelihood: Mock,
    mock_lowrank: Mock,
    mock_scale: Mock,
    mock_loss: Mock,
    fixture: Fixture,
) -> None:
    mock_likelihood.return_value = 5.0
    (_, kernel) = fixture
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
