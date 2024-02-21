"""Defines the main decompose loop for all nonlinear matrix decomposition algorithms.

Functions:
    decompose: Main library entry point. Iteratively applies selected algorithm.

"""

from typing import Optional
import time
import logging
import numpy as np

from fi_nomad.types import (
    FloatArrayType,
    InitializationStrategy,
    KernelInputType,
    KernelSpecificParameters,
    KernelReturnDataType,
    KernelStrategy,
    SVDStrategy,
    DiagnosticDataConfig,
)
from fi_nomad.kernels import KernelBase
from fi_nomad.util import initialize_candidate
from fi_nomad.util.factory_util import instantiate_kernel

logger = logging.getLogger(__name__)


def validate_sparse_matrix(sparse_matrix_X: FloatArrayType) -> None:
    """Confirms that the input matrix is actually nonnegative.

    Sparsity is not yet checked, since we don't have a threshold for it.

    Args:
        sparse_matrix_X: Matrix to verify sparsity.

    Raises:
        ValueError: Raised if the input matrix contains negative elements.
    """
    if np.any(sparse_matrix_X < 0):
        raise ValueError("Sparse input matrix must be nonnegative.")


def compute_max_iterations(
    manual_max_iterations: Optional[int], target_rank: int
) -> int:
    """Helper function to compute the maximum number of iterations to run,
    if not supplied by user.

    Args:
        manual_max_iterations: If supplied by the user, this value will be used
        target_rank: The target rank for the decomposition

    Returns:
        A maximum number of iterations to run before giving up.
    """
    if (manual_max_iterations) is not None:
        return manual_max_iterations
    return 100 * target_rank


def do_final_report(
    loop_start_time: float, run_start_time: float, kernel: KernelBase
) -> KernelReturnDataType:
    """Calls kernel final report, computes timings and returns kernel data.

    Args:
        loop_start_time: Timestamp of main loop start
        run_start_time: Timestamp of initialization start
        kernel: The instantiated kernel object used during the main loop

    Returns:
        The data returned by the instantiated kernel.
    """
    end_time = time.perf_counter()
    init_e = loop_start_time - run_start_time
    loop_e = end_time - loop_start_time
    per_loop_e = loop_e / (
        kernel.elapsed_iterations if kernel.elapsed_iterations > 0 else 1
    )
    result = kernel.report()
    logger.info(result.summary)
    logger.info(
        f"\tInitialization took {init_e} loop took {loop_e} overall ({per_loop_e}/ea)"
    )
    return result.data


def decompose(
    sparse_matrix_X: FloatArrayType,
    target_rank: int,
    kernel_strategy: KernelStrategy,
    *,
    svd_strategy: SVDStrategy = SVDStrategy.RANDOM_TRUNCATED,
    initialization: InitializationStrategy = InitializationStrategy.BROADCAST_MEAN,
    initial_guess_matrix: Optional[FloatArrayType] = None,
    kernel_params: Optional[KernelSpecificParameters] = None,
    tolerance: Optional[float] = None,
    manual_max_iterations: Optional[int] = None,
    verbose: bool = False,
    diagnostic_config: Optional[DiagnosticDataConfig] = None,
) -> KernelReturnDataType:
    """Main matrix decomposition loop.

    This is the main entry point for the library. The function handles setup of input data
    and instantiation of the actual solution kernel, and executes the main solution loop.

    Args:
        sparse_matrix_X: The sparse nonnegative input matrix, for which a low-rank
            approximation is sought
        target_rank: The target rank of the low-rank approximation
        kernel_strategy: Choice of solution kernel method
        svd_strategy: Algorithm to use for SVD decompositions within the kernels. Defaults to
            SVDStrategy.RANDOM_TRUNCATED.
        initialization: Strategy for initializing the first low-rank approximation estimate in
            the kernels. Defaults to InitializationStrategy.BROADCAST_MEAN.
        initial_guess_matrix: The initial low-rank approximation estimate to use, if the
            "KNOWN_MATRIX" strategy is chosen (which allows the user to start from a checkpoint).
            Defaults to None.
        kernel_params: Optional kernel-specific hyperparameters. Defaults to None.
        tolerance: If set, the algorithm will terminate once the low-rank candidate's Frobenius-norm
            loss (compared to the sparse input matrix) drops below this value. Defaults to None.
        manual_max_iterations: If set, will restrict the algorithm to run no more than this many
            iterations. If set to None (default), max iterations are set to 100 * target_rank.
        verbose: If set, will log any per-iteration status updates output by the kernel. Defaults to
            False.
        diagnostic_level: How much per-iteration diagnostic information the kernel should
            print on each iteration (if supported by the kernel). Defaults to NONE.
        diagnostic_output_basedir: The base path where kernel diagnostic information, if any,
            should be output.
        use_exact_diagnostic_basepath: If True, and the diagnostic output is requested and
            supported, then the diagnostic base path will be used exactly. If False (the
            default), the system will create a timestamped subdirectory of the given base
            directory.

    Returns:
        An object with the kernel's final summary description and the low-rank approximation of the
        input matrix.
    """
    if verbose:
        logger.setLevel(logging.INFO)
    logger.info(f"\tInitiating run, target_rank: {target_rank}, tolerance: {tolerance}")

    validate_sparse_matrix(sparse_matrix_X)
    max_iterations = compute_max_iterations(manual_max_iterations, target_rank)
    low_rank_candidate_L = initialize_candidate(
        initialization, kernel_strategy, sparse_matrix_X, initial_guess_matrix
    )
    kernel_input = KernelInputType(
        sparse_matrix_X, low_rank_candidate_L, target_rank, svd_strategy, tolerance
    )

    run_start_time = time.perf_counter()
    kernel = instantiate_kernel(
        kernel_strategy,
        kernel_input,
        kernel_params=kernel_params,
        diagnostic_config=diagnostic_config,
    )

    loop_start_time = time.perf_counter()
    while kernel.elapsed_iterations < max_iterations:
        kernel.step()
        kernel.increment_elapsed()

        running_report = kernel.running_report()
        if running_report != "":
            logger.info(running_report)
        kernel.per_iteration_diagnostic()
        if tolerance is not None and kernel.loss < tolerance:
            break

    data = do_final_report(loop_start_time, run_start_time, kernel)
    return data
