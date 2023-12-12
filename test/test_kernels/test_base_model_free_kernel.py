from typing import Tuple, cast
import numpy as np
from unittest.mock import Mock, patch

import pytest
from fi_nomad.kernels import BaseModelFree
from fi_nomad.types import FloatArrayType, KernelInputType, LossType, SVDStrategy

Fixture = Tuple[KernelInputType, BaseModelFree]
PKG = "fi_nomad.kernels.base_model_free"


@pytest.fixture
def fixture_with_tol() -> Fixture:
    sparse = np.eye(3) * 3
    candidate = np.ones(sparse.shape)
    target_rank = 5
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = 3.0
    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel = BaseModelFree(indata)
    return (indata, kernel)


@pytest.fixture
def fixture_no_tol() -> Fixture:
    sparse = np.eye(3) * 3
    candidate = np.ones(sparse.shape)
    target_rank = 5
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = None

    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel = BaseModelFree(indata)
    return (indata, kernel)


@patch(f"{PKG}.compute_loss")
@patch(f"{PKG}.find_low_rank")
@patch(f"{PKG}.construct_utility")
def test_base_model_free_kernel_step(
    mock_construct: Mock,
    mock_find_low_rank: Mock,
    mock_compute_loss: Mock,
    fixture_with_tol: Fixture,
) -> None:
    (indata, kernel) = fixture_with_tol
    mock_construct.return_value = np.full(indata.sparse_matrix_X.shape, 4)
    mock_find_low_rank.return_value = np.full(indata.sparse_matrix_X.shape, 5)
    mock_compute_loss.return_value = 3.0

    kernel.step()
    mock_construct.assert_called_once_with(
        indata.low_rank_candidate_L, indata.sparse_matrix_X
    )
    mock_find_low_rank.assert_called_once_with(
        mock_construct.return_value,
        indata.target_rank,
        indata.low_rank_candidate_L,
        SVDStrategy.RANDOM_TRUNCATED,
    )
    mock_compute_loss.assert_called_once_with(
        mock_construct.return_value, mock_find_low_rank.return_value, LossType.FROBENIUS
    )
    assert kernel.loss == mock_compute_loss.return_value


@patch(f"{PKG}.compute_loss")
@patch(f"{PKG}.find_low_rank")
@patch(f"{PKG}.construct_utility")
def test_base_model_free_kernel_step_skips_setting_loss_if_unset_tolerance(
    mock_construct: Mock,
    mock_find_low_rank: Mock,
    mock_compute_loss: Mock,
    fixture_no_tol: Fixture,
) -> None:
    (_, kernel) = fixture_no_tol
    assert kernel.loss == float("inf")
    kernel.step()
    assert kernel.loss == float("inf")
    mock_construct.assert_called_once()
    mock_find_low_rank.assert_called_once()
    mock_compute_loss.assert_not_called()


def test_base_model_free_kernel_running_report(fixture_no_tol: Fixture) -> None:
    (_, kernel) = fixture_no_tol
    first_txt = kernel.running_report()
    # tolerance is unset, so we skip loss computation and return empty string
    assert first_txt == ""
    kernel.tolerance = 5.0
    second_txt = kernel.running_report()
    assert "iteration" in second_txt
    assert "loss" in second_txt


def test_base_model_free_kernel_final_report(fixture_no_tol: Fixture) -> None:
    (indata, kernel) = fixture_no_tol

    result_1 = kernel.report()
    assert "Not Tracked" in result_1.summary
    np.testing.assert_array_equal(
        indata.low_rank_candidate_L, cast(FloatArrayType, result_1.data.reconstruction)
    )

    kernel.loss = 3.0
    result_2 = kernel.report()
    assert "3.0" in result_2.summary
