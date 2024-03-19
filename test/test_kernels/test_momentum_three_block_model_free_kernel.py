from pathlib import Path
from typing import Tuple, cast
import numpy as np
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, fixture
from fi_nomad.kernels import Momentum3BlockModelFreeKernel
from fi_nomad.types import (
    FloatArrayType,
    KernelInputType,
    Momentum3BlockAdditionalParameters,
    LossType,
    SVDStrategy,
    DiagnosticLevel,
)

Fixture = Tuple[
    KernelInputType, Momentum3BlockAdditionalParameters, Momentum3BlockModelFreeKernel
]
PKG = "fi_nomad.kernels.momentum_three_block_model_free"


@fixture
def fixture_with_tol_W0H0_given() -> Fixture:
    sparse = np.eye(9) * 3.0
    (m, n) = sparse.shape
    target_rank = 5
    candidate_W = np.ones((m, target_rank))
    candidate_H = np.ones((target_rank, n))
    candidate = candidate_W @ candidate_H
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = 3.0
    momentum_beta = 0.7
    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel_params = Momentum3BlockAdditionalParameters(
        momentum_beta, candidate_W, candidate_H
    )
    kernel = Momentum3BlockModelFreeKernel(indata, kernel_params)
    return (indata, kernel_params, kernel)


@fixture
def fixture_no_tol_W0H0_not_given() -> Fixture:
    sparse = np.eye(9) * 3.0
    (m, n) = sparse.shape
    target_rank = 5
    candidate_W = np.ones((m, target_rank))
    candidate_H = np.ones((target_rank, n))
    candidate = candidate_W @ candidate_H
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = None
    momentum_beta = 0.7
    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel_params = Momentum3BlockAdditionalParameters(momentum_beta)
    kernel = Momentum3BlockModelFreeKernel(indata, kernel_params)
    return (indata, kernel_params, kernel)


@patch(f"{PKG}.compute_loss")
@patch(f"{PKG}.update_H")
@patch(f"{PKG}.update_W")
@patch(f"{PKG}.apply_momentum")
@patch(f"{PKG}.construct_utility")
def test_momentum_3block_model_free_kernel_step_W0H0_given_one_iter(
    mock_construct: Mock,
    mock_apply_momentum: Mock,
    mock_update_W: Mock,
    mock_update_H: Mock,
    mock_compute_loss: Mock,
    fixture_with_tol_W0H0_given: Fixture,
) -> None:
    (indata, kernel_params, kernel) = fixture_with_tol_W0H0_given
    shape_X = indata.sparse_matrix_X.shape
    shape_W0 = (shape_X[0], indata.target_rank)
    shape_H0 = (indata.target_rank, shape_X[1])
    mock_construct.return_value = np.full(shape_X, 4.0)
    mock_apply_momentum.return_value = np.full(shape_X, 5.0)
    mock_update_W.return_value = np.full(shape_W0, 1.0)
    mock_update_H.return_value = np.full(shape_H0, 1.0)
    mock_compute_loss.return_value = 3.0

    kernel.step()
    mock_construct.assert_called_once_with(
        indata.low_rank_candidate_L, indata.sparse_matrix_X
    )
    mock_apply_momentum.assert_called_once_with(
        mock_construct.return_value,
        indata.sparse_matrix_X,
        kernel.momentum_beta,
    )
    mock_update_W.assert_called_once_with(
        kernel_params.candidate_factor_H0, mock_apply_momentum.return_value
    )
    mock_update_H.assert_called_once_with(
        mock_update_W.return_value, mock_apply_momentum.return_value
    )

    mock_compute_loss.assert_called_once_with(
        mock_apply_momentum.return_value,
        kernel.low_rank_candidate_L,
        LossType.FROBENIUS,
    )


@patch(f"{PKG}.apply_momentum")
def test_momentum_3block_model_free_kernel_step_apply_momentum_call_count(
    mock_apply_momentum: Mock,
    fixture_no_tol_W0H0_not_given: Fixture,
) -> None:
    (indata, _, kernel) = fixture_no_tol_W0H0_not_given
    shape_X = indata.sparse_matrix_X.shape

    mock_apply_momentum.return_value = np.full(shape_X, 5.0)
    kernel.elapsed_iterations = 0
    kernel.step()
    kernel.increment_elapsed()

    # apply_momentum called one time in first iteration
    assert mock_apply_momentum.call_count == 1

    kernel.step()

    # apply_momentum called two times in second iteration
    assert mock_apply_momentum.call_count == 3


def test_momentum_3block_model_free_kernel_W0H0_initialized_if_not_given(
    fixture_no_tol_W0H0_not_given: Fixture,
) -> None:
    (indata, _, kernel) = fixture_no_tol_W0H0_not_given

    np.testing.assert_array_almost_equal(
        indata.low_rank_candidate_L,
        kernel.candidate_factor_W @ kernel.candidate_factor_H,
    )


def test_momentum_3block_model_free_kernel_running_report(
    fixture_no_tol_W0H0_not_given: Fixture,
) -> None:
    (_, _, kernel) = fixture_no_tol_W0H0_not_given
    first_txt = kernel.running_report()
    # tolerance is unset, so we skip loss computation and return empty string
    assert first_txt == ""
    kernel.tolerance = 5.0
    second_txt = kernel.running_report()
    assert "iteration" in second_txt
    assert "loss" in second_txt


def test_momentum_3block_model_free_kernel_final_report(
    fixture_no_tol_W0H0_not_given: Fixture,
) -> None:
    (indata, _, kernel) = fixture_no_tol_W0H0_not_given

    result_1 = kernel.report()
    assert "Not Tracked" in result_1.summary
    np.testing.assert_allclose(
        indata.low_rank_candidate_L,
        cast(
            FloatArrayType,
            result_1.data.factors[0] @ result_1.data.factors[1],
        ),
    )

    kernel.loss = 3.0
    result_2 = kernel.report()
    assert "3.0" in result_2.summary


def test_momentum_3block_model_free_kernel_per_iteration_diagnostic_calls_base(
    fixture_no_tol_W0H0_not_given: Fixture, caplog: LogCaptureFixture
) -> None:
    (_, _, kernel) = fixture_no_tol_W0H0_not_given
    test_path_str = "test-path"
    test_path = Path(test_path_str)
    kernel.out_dir = test_path
    kernel.diagnostic_level = DiagnosticLevel.MINIMAL
    kernel.per_iteration_diagnostic()
    assert "Per-iteration diagnostic" in caplog.text
    assert f"{test_path_str}" in caplog.text
