import numpy as np
from typing import Optional, Union
import time
import logging
from lzcompression.kernels.kernel_base import KernelBase
from lzcompression.kernels.base_model_free import BaseModelFree
from lzcompression.kernels.rowwise_variance_gauss_model import RowwiseVarianceGaussianModelKernel
from lzcompression.kernels.single_variance_gauss_model import (
    SingleVarianceGaussianModelKernel,
)

from lzcompression.types import (
    FloatArrayType,
    InitializationStrategy,
    KernelInputType,
    KernelReturnDataType,
    KernelStrategy,
    SVDStrategy,
)
from lzcompression.util.util import (
    initialize_low_rank_candidate,
)

logger = logging.getLogger(__name__)


def instantiate_kernel(s: KernelStrategy, data_in: KernelInputType) -> KernelBase:
    if s == KernelStrategy.BASE_MODEL_FREE:
        return BaseModelFree(data_in)
    if s == KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE:
        return SingleVarianceGaussianModelKernel(data_in)
    if s == KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE:
        return RowwiseVarianceGaussianModelKernel(data_in)
    raise ValueError(f"Unsupported kernel strategy {s}")


def initialize_candidate(
    init_strat: InitializationStrategy,
    k_strat: KernelStrategy,
    sparse: FloatArrayType,
    guess: Optional[FloatArrayType],
) -> FloatArrayType:
    # Note: enforcing "COPY" strategy for base_model_free kernel may not be desirable
    _initialization_strategy = (
        InitializationStrategy.COPY
        if k_strat == KernelStrategy.BASE_MODEL_FREE
        else init_strat
    )
    input_matrix = (
        guess
        if (
            _initialization_strategy == InitializationStrategy.KNOWN_MATRIX
            and guess is not None
        )
        else sparse
    )
    return initialize_low_rank_candidate(input_matrix, _initialization_strategy)


def validate_sparse_matrix(sparse_matrix_X: FloatArrayType) -> None:
    if np.any(sparse_matrix_X < 0):
        raise ValueError("Sparse input matrix must be nonnegative.")


def compute_max_iterations(
    manual_max_iterations: Optional[int], target_rank: int
) -> int:
    if (manual_max_iterations) is not None:
        return manual_max_iterations
    return 100 * target_rank


def decompose(
    sparse_matrix_X: FloatArrayType,
    target_rank: int,
    kernel_strategy: KernelStrategy,
    *,
    svd_strategy: SVDStrategy = SVDStrategy.RANDOM_TRUNCATED,
    initialization: InitializationStrategy = InitializationStrategy.BROADCAST_MEAN,
    initial_guess_matrix: Union[FloatArrayType, None] = None,
    tolerance: Union[float, None] = None,
    manual_max_iterations: Union[int, None] = None,
    verbose: bool = False,
) -> KernelReturnDataType:
    if verbose:
        logger.setLevel(logging.INFO)
    logger.info(f"\tInitiating run, {target_rank=}, {tolerance=}")

    validate_sparse_matrix(sparse_matrix_X)
    max_iterations = compute_max_iterations(manual_max_iterations, target_rank)
    low_rank_candidate_L = initialize_candidate(
        initialization, kernel_strategy, sparse_matrix_X, initial_guess_matrix
    )
    kernel_input = KernelInputType(
        sparse_matrix_X, low_rank_candidate_L, target_rank, svd_strategy, tolerance
    )

    run_start_time = time.perf_counter()
    kernel = instantiate_kernel(kernel_strategy, kernel_input)

    loop_start_time = time.perf_counter()
    while kernel.elapsed_iterations < max_iterations:
        kernel.increment_elapsed()
        kernel.step()

        running_report = kernel.running_report()
        if running_report != "":
            logger.info(running_report)
        if tolerance is not None and kernel.loss < tolerance:
            break

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
