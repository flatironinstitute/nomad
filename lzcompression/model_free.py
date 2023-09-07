import numpy as np
import logging
import time

from lzcompression.types import (
    FloatArrayType,
    InitializationStrategy,
    SVDStrategy,
    LossType,
)
from lzcompression.util import (
    initialize_low_rank_candidate,
    find_low_rank,
    compute_loss,
)

logger = logging.getLogger(__name__)


def compress_sparse_matrix(
    sparse_matrix: FloatArrayType,
    target_rank: int,
    *,
    strategy: SVDStrategy = SVDStrategy.RANDOM_TRUNCATED,
    tolerance: float | None = None,
    verbose: bool = False,
    manual_max_iterations: int | None = None,
) -> FloatArrayType:
    if verbose:
        logger.setLevel(logging.INFO)
    logger.info(f"\tInitiating run, {target_rank=}, {tolerance=}")
    # TODO: Check actual sparsity?
    if np.any(sparse_matrix[sparse_matrix < 0]):
        raise ValueError("Sparse input matrix must be nonnegative.")

    run_start_time = time.perf_counter()

    low_rank_candidate_L = initialize_low_rank_candidate(
        sparse_matrix, InitializationStrategy.COPY
    )

    max_iterations = (
        manual_max_iterations
        if manual_max_iterations is not None
        else 100 * target_rank
    )  # TODO: check; should we really iterate longer for higher rank?

    elapsed_iterations = 0
    loss = float("inf")

    loop_start_time = time.perf_counter()
    while elapsed_iterations < max_iterations:
        elapsed_iterations += 1
        ## Alternate the utility-construction step and the low-rank estimation step
        utility_matrix_Z = construct_utility(low_rank_candidate_L, sparse_matrix)
        low_rank_candidate_L = find_low_rank(
            utility_matrix_Z, target_rank, low_rank_candidate_L, strategy
        )

        if tolerance is not None:
            loss = compute_loss(
                utility_matrix_Z, low_rank_candidate_L, LossType.FROBENIUS
            )
            logger.info(f"iteration: {elapsed_iterations} {loss=}")
            if loss < tolerance:
                break

    end_time = time.perf_counter()
    init_e = loop_start_time - run_start_time
    loop_e = end_time - loop_start_time
    per_loop_e = loop_e / (elapsed_iterations if elapsed_iterations > 0 else 1)
    logger.info(f"{elapsed_iterations} total, final loss {loss}")
    logger.info(
        f"\tInitialization took {init_e} loop took {loop_e} overall ({per_loop_e}/ea)"
    )
    return low_rank_candidate_L


def construct_utility(
    low_rank_matrix: FloatArrayType, base_matrix: FloatArrayType
) -> FloatArrayType:
    # The construction step creates Z from a 0-matrix by:
    #   copying the positive elements of S into the corresponding elements of Z
    #   and the negative elements of L into any corresponding elements of Z that are still 0
    # i.e., for each i, j: Z_ij = S_ij if S_ij > 0; else min(0, L_ij).
    conditions = [base_matrix > 0, low_rank_matrix < 0]
    choices = [base_matrix, low_rank_matrix]
    utility_matrix = np.select(conditions, choices, 0)
    return utility_matrix


# TODO:
# Consider secondary evaluation criteria, e.g. promoting bimodal/higher-variance ditsributions of negative values in answer
