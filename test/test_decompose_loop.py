from typing import cast
import numpy as np
from unittest.mock import Mock, patch
from pytest import raises, LogCaptureFixture
from fi_nomad.kernels import KernelBase

from fi_nomad.decompose import (
    compute_max_iterations,
    decompose,
    initialize_candidate,
    instantiate_kernel,
)

from fi_nomad.types import (
    BaseModelFreeKernelReturnType,
    FloatArrayType,
    InitializationStrategy,
    KernelInputType,
    KernelReturnType,
    KernelStrategy,
    SVDStrategy,
)

TEST_KERNEL_TOLERANCE_ITERATIONS: int = 5
PKG = "fi_nomad.decompose"


class TestKernel(KernelBase):
    __test__ = False  # tell Pytest this isn't a class with tests

    def __init__(self, input: KernelInputType) -> None:
        super().__init__(input)

    def step(self) -> None:
        if self.elapsed_iterations >= TEST_KERNEL_TOLERANCE_ITERATIONS:
            if self.tolerance is not None:
                self.loss = self.tolerance - 1

    def running_report(self) -> str:
        return f"{self.elapsed_iterations}"

    def report(self) -> KernelReturnType:
        return KernelReturnType(
            "Complete", BaseModelFreeKernelReturnType(self.sparse_matrix_X)
        )


## Kernel Instantiation
def test_instantiate_kernel_throws_on_unsupported_kernel() -> None:
    mock_data_in = Mock()
    with raises(ValueError, match="Unsupported"):
        _ = instantiate_kernel(KernelStrategy.TEST, mock_data_in)


@patch(f"{PKG}.SingleVarianceGaussianModelKernel", autospec=True)
@patch(f"{PKG}.BaseModelFree", autospec=True)
def test_instantiate_kernel_honors_strategy_selection(
    mock_base: Mock, mock_svgauss: Mock
) -> None:
    mock_data_in = Mock()
    _ = instantiate_kernel(KernelStrategy.BASE_MODEL_FREE, mock_data_in)
    mock_base.assert_called_once()
    _ = instantiate_kernel(KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE, mock_data_in)
    mock_svgauss.assert_called_once()


## low-rank candidate initialization
@patch(f"{PKG}.initialize_low_rank_candidate")
def test_initialize_candidate_honors_initialization_strategy(mock_init: Mock) -> None:
    mock_x = np.array([[0, 2], [2, 0]])
    mock_checkpoint = np.eye(2)
    with raises(AssertionError):
        np.testing.assert_array_equal(mock_x, mock_checkpoint)
    k_strat = KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE

    _ = initialize_candidate(
        InitializationStrategy.COPY, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.BROADCAST_MEAN, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.BROADCAST_MEAN)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(
        mock_checkpoint, InitializationStrategy.KNOWN_MATRIX
    )


@patch(f"{PKG}.initialize_low_rank_candidate")
def test_initialize_candidate_forces_copy_strategy_for_base_model_free(
    mock_init: Mock,
) -> None:
    mock_x = np.array([[0, 2], [2, 0]])
    mock_checkpoint = Mock()
    k_strat = KernelStrategy.BASE_MODEL_FREE

    _ = initialize_candidate(
        InitializationStrategy.COPY, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.BROADCAST_MEAN, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)


@patch(f"{PKG}.initialize_low_rank_candidate")
def test_initialize_candidate_defaults_to_copy_on_null_checkpoint(
    mock_init: Mock,
) -> None:
    mock_x = np.array([[0, 2], [2, 0]])
    mock_checkpoint = None
    k_strat = KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE
    _ = initialize_candidate(
        InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.KNOWN_MATRIX)


def test_initialize_candidate_throws_on_checkpoint_size_mismatch() -> None:
    mock_x = np.eye(3)
    mock_checkpoint = np.eye(4)
    k_strat = KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE
    with raises(ValueError, match="shape"):
        _ = initialize_candidate(
            InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
        )


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


## Actual decompose loop


def test_decompose_throws_if_input_has_negative_elements() -> None:
    bad_matrix = np.array([[3, 2, 2], [2, 3, -2]])
    with raises(ValueError, match="nonnegative"):
        _ = decompose(bad_matrix, 4, kernel_strategy=KernelStrategy.BASE_MODEL_FREE)


@patch(f"{PKG}.instantiate_kernel")
def test_decompose_obeys_max_iterations(mock_get_kernel: Mock) -> None:
    target_rank = 5
    max_iterations = 10
    # fmt: off
    sparse_matrix = np.array([
        [3, 2, 2],
        [2, 3, 1]
    ])
    # fmt: on

    mocked_input = KernelInputType(
        sparse_matrix, sparse_matrix, target_rank, SVDStrategy.FULL, None
    )
    mock_kernel = TestKernel(mocked_input)
    mock_get_kernel.return_value = mock_kernel

    result = decompose(
        sparse_matrix,
        target_rank,
        kernel_strategy=KernelStrategy.TEST,
        initialization=InitializationStrategy.COPY,
        svd_strategy=SVDStrategy.FULL,
        manual_max_iterations=max_iterations,
    )

    # Assert that we correctly simulated the object passed to the kernel creator
    # (This is just a check on the mocks)
    passed_input = mock_get_kernel.call_args.args[1]
    np.testing.assert_array_equal(
        mocked_input.sparse_matrix_X, passed_input.sparse_matrix_X
    )
    np.testing.assert_array_equal(
        mocked_input.low_rank_candidate_L, passed_input.low_rank_candidate_L
    )
    assert mocked_input.target_rank == passed_input.target_rank
    assert mocked_input.svd_strategy == passed_input.svd_strategy
    assert passed_input.tolerance is None

    assert mock_kernel.elapsed_iterations == max_iterations
    np.testing.assert_array_equal(
        sparse_matrix, cast(FloatArrayType, result.reconstruction)
    )


@patch(f"{PKG}.instantiate_kernel")
def test_decompose_stops_when_error_within_tolerance(mock_get_kernel: Mock) -> None:
    tolerance = 5
    target_rank = 5
    # fmt: off
    sparse_matrix = np.array([
        [3, 2, 2],
        [2, 3, 1]
    ])
    # fmt: on

    mocked_input = KernelInputType(
        sparse_matrix, sparse_matrix, target_rank, SVDStrategy.FULL, tolerance
    )
    mock_kernel = TestKernel(mocked_input)
    mock_get_kernel.return_value = mock_kernel

    _ = decompose(
        sparse_matrix,
        target_rank,
        kernel_strategy=KernelStrategy.TEST,
        tolerance=tolerance,
    )
    assert mock_kernel.elapsed_iterations == TEST_KERNEL_TOLERANCE_ITERATIONS


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


# Could add some tests that the timing features are actually working
