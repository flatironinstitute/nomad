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
    manual_max_iterations: int | None = None,
    verbose: bool = False,
) -> FloatArrayType:
    """Estimate a low-rank approximation of a nonnegative sparse matrix.

    This is a more basic implementation than the one described in Saul (2022). The implementation
    in this method does not build an underlying model, but instead just alternates between
    constructing a utility matrix Z that enforces the non-zero values of the sparse matrix, and
    using SVD on Z to create a candidate low-rank matrix L. The algorithm terminates either when
    overall loss is below a tolerance (if specified) or when a maximum iteration count is reached,
    the maximum being either a manually-set value or 100 * the target rank.

    Args:
        sparse_matrix: The sparse nonnegative matrix to decompose ("X")
        target_rank: The target rank of the low-rank representation.
        strategy (optional): Strategy to use for SVD. Defaults to SVDStrategy.RANDOM_TRUNCATED.
        tolerance (optional): Loss tolerance for the reconstruction (defaults to None). If set,
            the algorithm will terminate once overall loss between Z and the candidate low-rank
            representation L has dropped below this level.
        manual_max_iterations (optional): If set, will override the default maximum iteration count
            (of 100 * the target rank). Defaults to None.
        verbose (optional): If True, will use a logger to report performance data. Default False.

    Raises:
        ValueError: If the input matrix is not nonnegative.

    Returns:
        A low-rank approximation of the input matrix.
    """
    if verbose:
        logger.setLevel(logging.INFO)
    logger.info(f"\tInitiating run, {target_rank=}, {tolerance=}")
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
    """Construct a utility matrix Z which enforces the invariants of the original
    sparse nonnegative matrix.

    Specifically, it creates Z from a 0-matrix by:
      - Copying the positive elements of the original sparse matrix X into the
        corresponding elements of Z
      - Copying any negative elements of the current low-rank approximation matrix
        L into the corresponding elements of Z, provided those elements of Z
        were not set in the first step
      - Any remaining elements remain 0

    i.e. for each i, j: Z_ij = X is X_ij > 0, else min(0, L_ij).

    Args:
        low_rank_matrix: The current low-rank approximation of the base matrix
        base_matrix: the sparse nonnegative matrix whose low-rank approximation
            is being sought

    Returns:
        A utility matrix whose only positive values are the positive values in
        the base_matrix
    """
    conditions = [base_matrix > 0, low_rank_matrix < 0]
    choices = [base_matrix, low_rank_matrix]
    utility_matrix = np.select(conditions, choices, 0)
    return utility_matrix


# TODO:
# Consider secondary evaluation criteria, e.g. promoting bimodal/higher-variance ditsributions of negative values in answer
