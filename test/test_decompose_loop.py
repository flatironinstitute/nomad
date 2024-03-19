import logging
from typing import Optional, Tuple, cast
import numpy as np
from unittest.mock import Mock, patch, call
from pytest import raises, LogCaptureFixture, fixture
from fi_nomad.kernels import KernelBase

from fi_nomad.entry import compute_max_iterations, decompose, do_final_report

from fi_nomad.types import (
    BaseModelFreeKernelReturnType,
    InitializationStrategy,
    KernelInputType,
    KernelReturnType,
    KernelStrategy,
    SVDStrategy,
    DiagnosticDataConfig,
    DiagnosticLevel,
)

from fi_nomad.util import two_part_factor

TEST_KERNEL_TOLERANCE_ITERATIONS: int = 5
PKG = "fi_nomad.entry"


class TestKernel(KernelBase):
    __test__ = False  # tell Pytest this isn't a class with tests

    def __init__(self, input: KernelInputType) -> None:
        super().__init__(input)

    def step(self) -> None:
        # elapsed_iterations + 1 gives the iteration we're currently in
        if self.elapsed_iterations + 1 >= TEST_KERNEL_TOLERANCE_ITERATIONS:
            if self.tolerance is not None:
                self.loss = self.tolerance - 1

    def running_report(self) -> str:
        return f"{self.elapsed_iterations}"

    def report(self) -> KernelReturnType:
        return KernelReturnType(
            "Complete",
            BaseModelFreeKernelReturnType(two_part_factor(self.sparse_matrix_X)),
        )


Fixture = Tuple[KernelInputType, TestKernel]


def make_test_kernel(tol: Optional[int] = None) -> Fixture:
    target_rank = 5
    # fmt: off
    sparse_matrix = np.array([
        [3, 2, 2],
        [2, 3, 1]
    ])
    # fmt: on

    mocked_input = KernelInputType(
        sparse_matrix, sparse_matrix, target_rank, SVDStrategy.FULL, tol
    )
    mock_kernel = TestKernel(mocked_input)
    return (mocked_input, mock_kernel)


@fixture
def test_kernel_fix() -> Fixture:
    return make_test_kernel()


@fixture
def test_kernel_fix_tol() -> Fixture:
    return make_test_kernel(5)


## Computing max iterations


def test_compute_max_iterations_honors_manual_count() -> None:
    manual_max = 17
    target_rank = 5
    rcvd = compute_max_iterations(manual_max, target_rank)
    assert manual_max == rcvd


def test_compute_max_iterations_defaults_to_rank() -> None:
    manual_max = None
    target_rank = 5
    rcvd = compute_max_iterations(manual_max, target_rank)
    assert rcvd == 100 * target_rank


## Final reporting


@patch(f"{PKG}.time")
def test_do_final_report_computes_time(
    mock_time: Mock, caplog: LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    mock_data_return = cast(BaseModelFreeKernelReturnType, Mock)
    mock_report = Mock()
    mock_report.summary = "Complete"
    mock_report.data = mock_data_return

    mock_kernel = Mock()
    mock_kernel.report = lambda: mock_report
    mock_kernel.elapsed_iterations = 8

    test_loop_start = 20.0
    test_run_start = 10.0
    mock_time.perf_counter = lambda: test_loop_start + (
        10.0 * mock_kernel.elapsed_iterations
    )

    res = do_final_report(
        test_loop_start, test_run_start, cast(KernelBase, mock_kernel)
    )
    assert res == mock_data_return
    assert "Complete" in caplog.text
    assert f"Initialization took {test_loop_start - test_run_start}" in caplog.text
    assert "loop took 80.0" in caplog.text
    assert "(10.0/ea)" in caplog.text


## Actual decompose loop


def test_decompose_throws_if_input_has_negative_elements() -> None:
    bad_matrix = np.array([[3, 2, 2], [2, 3, -2]])
    with raises(ValueError, match="nonnegative"):
        _ = decompose(bad_matrix, 4, kernel_strategy=KernelStrategy.BASE_MODEL_FREE)


@patch(f"{PKG}.instantiate_kernel")
def test_decompose_obeys_max_iterations(
    mock_get_kernel: Mock, test_kernel_fix: Fixture
) -> None:
    (indata, test_kernel) = test_kernel_fix

    mock_get_kernel.return_value = test_kernel

    max_iterations = 10
    result = decompose(
        indata.sparse_matrix_X,
        indata.target_rank,
        kernel_strategy=KernelStrategy.TEST,
        initialization=InitializationStrategy.COPY,
        svd_strategy=SVDStrategy.FULL,
        manual_max_iterations=max_iterations,
    )

    # Assert that we correctly simulated the object passed to the kernel creator
    # (This is just a check on the mocks)
    passed_input = mock_get_kernel.call_args.args[1]
    np.testing.assert_array_equal(
        test_kernel.sparse_matrix_X, passed_input.sparse_matrix_X
    )
    np.testing.assert_array_equal(
        test_kernel.low_rank_candidate_L, passed_input.low_rank_candidate_L
    )
    assert indata.target_rank == passed_input.target_rank
    assert indata.svd_strategy == passed_input.svd_strategy
    assert passed_input.tolerance is None

    assert test_kernel.elapsed_iterations == max_iterations
    np.testing.assert_allclose(
        test_kernel.sparse_matrix_X, result.factors[0] @ result.factors[1]
    )


@patch(f"{PKG}.instantiate_kernel")
def test_decompose_stops_when_error_within_tolerance(
    mock_get_kernel: Mock, test_kernel_fix_tol: Fixture
) -> None:
    (indata, kernel) = test_kernel_fix_tol
    assert indata.tolerance == 5
    mock_get_kernel.return_value = kernel

    _ = decompose(
        indata.sparse_matrix_X,
        indata.target_rank,
        kernel_strategy=KernelStrategy.TEST,
        tolerance=indata.tolerance,
    )
    assert kernel.elapsed_iterations == TEST_KERNEL_TOLERANCE_ITERATIONS


def test_decompose_honors_verbosity(caplog: LogCaptureFixture) -> None:
    sparse_matrix = np.eye(3)
    _ = decompose(
        sparse_matrix, 1, KernelStrategy.BASE_MODEL_FREE, manual_max_iterations=0
    )
    assert "Initiating run" not in caplog.text
    _ = decompose(
        sparse_matrix,
        1,
        KernelStrategy.BASE_MODEL_FREE,
        manual_max_iterations=0,
        verbose=True,
    )
    assert "Initiating run" in caplog.text


@patch(f"{PKG}.instantiate_kernel")
def test_decompose_calls_diagnostic_output_fn(
    mock_get_kernel: Mock, test_kernel_fix: Fixture
) -> None:
    (indata, kernel) = test_kernel_fix
    mock_per_iter_fn = Mock()
    kernel.per_iteration_diagnostic = mock_per_iter_fn  # type: ignore[method-assign]
    diag_config = DiagnosticDataConfig(DiagnosticLevel.EXTREME, ".", True)
    mock_get_kernel.return_value = kernel

    max_iterations = 10
    _ = decompose(
        indata.sparse_matrix_X,
        indata.target_rank,
        kernel_strategy=KernelStrategy.TEST,
        initialization=InitializationStrategy.COPY,
        svd_strategy=SVDStrategy.FULL,
        manual_max_iterations=max_iterations,
        diagnostic_config=diag_config,
    )

    assert mock_per_iter_fn.call_count == max_iterations


@patch(f"{PKG}.instantiate_kernel")
def test_kernel_step_called_before_increment_elapsed(
    mock_get_kernel: Mock, test_kernel_fix: Fixture
) -> None:
    (indata, _) = test_kernel_fix

    mock_kernel = Mock()
    mock_kernel.elapsed_iterations = 0
    mock_kernel.step = Mock()

    def mock_step() -> None:
        mock_kernel.elapsed_iterations += 1

    mock_kernel.increment_elapsed = Mock(side_effect=mock_step)
    mock_get_kernel.return_value = mock_kernel

    decompose(
        indata.sparse_matrix_X,
        indata.target_rank,
        kernel_strategy=KernelStrategy.TEST,
        tolerance=indata.tolerance,
        manual_max_iterations=1,
    )

    expected_calls = [call.step(), call.increment_elapsed()]
    mock_kernel.assert_has_calls(expected_calls, any_order=False)
