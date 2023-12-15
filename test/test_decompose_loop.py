import numpy as np
from unittest.mock import Mock, patch
from pytest import raises, LogCaptureFixture
from fi_nomad.kernels import KernelBase

from fi_nomad.entry import (
    compute_max_iterations,
    decompose,
)

from fi_nomad.types import (
    BaseModelFreeKernelReturnType,
    InitializationStrategy,
    KernelInputType,
    KernelReturnType,
    KernelStrategy,
    SVDStrategy,
)

from fi_nomad.util import two_part_factor

TEST_KERNEL_TOLERANCE_ITERATIONS: int = 5
PKG = "fi_nomad.entry"


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
            "Complete",
            BaseModelFreeKernelReturnType(two_part_factor(self.sparse_matrix_X)),
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
    np.testing.assert_allclose(sparse_matrix, result.factors[0] @ result.factors[1])


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
