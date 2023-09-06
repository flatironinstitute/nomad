import numpy as np

from .types import FloatArrayType, InitializationStrategy, SVDStrategy, LossType
from .util import initialize_low_rank_candidate, find_low_rank, compute_loss


def compress_sparse_matrix(
    sparse_matrix: FloatArrayType,
    target_rank: int,
    *,
    strategy: SVDStrategy = SVDStrategy.RANDOM_TRUNCATED,
    tolerance: float | None = None,
) -> FloatArrayType:
    # TODO: Check actual sparsity?
    print(f"\tInitiating run, {target_rank=}, {tolerance=}")
    if np.any(sparse_matrix[sparse_matrix < 0]):
        raise ValueError("Sparse input matrix must be nonnegative.")

    # Initialize low_rank_candidate_L
    low_rank_candidate_L = initialize_low_rank_candidate(
        sparse_matrix, InitializationStrategy.COPY
    )

    elapsed_iterations = 0
    max_iterations = (
        100 * target_rank
    )  # TODO: check; should we really iterate longer for higher rank?

    loss = float("inf")
    while elapsed_iterations < max_iterations:
        elapsed_iterations += 1
        utility_matrix_Z = construct_utility(low_rank_candidate_L, sparse_matrix)
        low_rank_candidate_L = find_low_rank(
            utility_matrix_Z, target_rank, low_rank_candidate_L, strategy
        )

        if tolerance is not None:
            loss = compute_loss(
                utility_matrix_Z, low_rank_candidate_L, LossType.SQUARED_DIFFERENCE
            )
            if loss < tolerance:
                break
    # TODO: Check validity of solution?
    print(f"Returning after {elapsed_iterations} final loss {loss}")
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
