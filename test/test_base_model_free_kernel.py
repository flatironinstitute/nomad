from typing import cast
import numpy as np
from unittest.mock import Mock, patch
from lzcompression.kernels.base_model_free import BaseModelFree
from lzcompression.types import FloatArrayType, KernelInputType, LossType, SVDStrategy


@patch("lzcompression.kernels.base_model_free.compute_loss")
@patch("lzcompression.kernels.base_model_free.find_low_rank")
@patch("lzcompression.kernels.base_model_free.construct_utility")
def test_base_model_free_kernel_step(
    mock_construct: Mock, mock_find_low_rank: Mock, mock_compute_loss: Mock
) -> None:
    sparse = np.eye(3) * 3
    candidate = np.ones(sparse.shape)
    target_rank = 5
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = 3.0

    fours = np.full(sparse.shape, 4)
    fives = np.full(sparse.shape, 5)
    mock_construct.return_value = fours
    mock_find_low_rank.return_value = fives
    mock_compute_loss.return_value = 3.0

    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel = BaseModelFree(indata)
    kernel.step()
    mock_construct.assert_called_once_with(candidate, sparse)
    mock_find_low_rank.assert_called_once_with(
        fours, target_rank, candidate, SVDStrategy.RANDOM_TRUNCATED
    )
    mock_compute_loss.assert_called_once_with(fours, fives, LossType.FROBENIUS)
    assert kernel.loss == 3.0


@patch("lzcompression.kernels.base_model_free.compute_loss")
@patch("lzcompression.kernels.base_model_free.find_low_rank")
@patch("lzcompression.kernels.base_model_free.construct_utility")
def test_base_model_free_kernel_step_skips_setting_loss_if_unset_tolerance(
    mock_construct: Mock, mock_find_low_rank: Mock, mock_compute_loss: Mock
) -> None:
    sparse = np.eye(3) * 3
    candidate = np.ones(sparse.shape)
    target_rank = 5
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = None

    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel = BaseModelFree(indata)

    assert kernel.loss == float("inf")
    kernel.step()
    assert kernel.loss == float("inf")
    mock_construct.assert_called_once()
    mock_find_low_rank.assert_called_once()
    mock_compute_loss.assert_not_called()


def test_base_model_free_kernel_running_report() -> None:
    sparse = np.eye(3)
    candidate = np.ones(sparse.shape)
    target_rank = 5
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = None

    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel = BaseModelFree(indata)
    first_txt = kernel.running_report()
    # tolerance is unset, so we skip loss computation and return empty string
    assert first_txt == ""
    kernel.tolerance = 5.0
    second_txt = kernel.running_report()
    assert "iteration" in second_txt
    assert "loss" in second_txt


def test_base_model_free_kernel_final_report() -> None:
    sparse = np.eye(3) * 3
    candidate = np.ones(sparse.shape)
    target_rank = 5
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = None

    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel = BaseModelFree(indata)

    result_1 = kernel.report()
    assert "Not Tracked" in result_1.summary
    np.testing.assert_array_equal(
        candidate, cast(FloatArrayType, result_1.data.reconstruction)
    )

    kernel.loss = 3.0
    result_2 = kernel.report()
    assert "3.0" in result_2.summary
