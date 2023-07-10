import numpy as np
import numpy.typing as npt
from enum import Enum


class InitializationStrategy(Enum):
    COPY = 1
    BROADCAST_MEAN = 2


def initialize_low_rank_candidate(
    sparse_matrix: npt.NDArray, method: InitializationStrategy
) -> npt.NDArray:
    low_rank_candidate = None
    if method == InitializationStrategy.COPY:
        low_rank_candidate = np.copy(sparse_matrix)
    elif method == InitializationStrategy.BROADCAST_MEAN:
        mean = np.mean(sparse_matrix)
        low_rank_candidate = np.ones(sparse_matrix.shape) * mean
    else:
        raise ValueError("Unsupported initialization strategy.")
    return low_rank_candidate


# TODO: Check type for NDArray--is this float or complex
def compress_sparse_matrix(
    sparse_matrix: npt.NDArray,
    target_rank: int,
    tolerance: float | None = None,
) -> npt.NDArray:
    # TODO: Check actual sparsity?
    if (np.any(sparse_matrix[sparse_matrix < 0])):
        raise ValueError("Sparse input matrix must be nonnegative.")

    # Initialize low_rank_candidate_L
    low_rank_candidate_L = initialize_low_rank_candidate(
        sparse_matrix, InitializationStrategy.COPY
    )

    elapsed_iterations = 0
    max_iterations = (
        100 * target_rank
    )  # TODO: check; should we really iterate longer for higher rank?

    while elapsed_iterations < max_iterations:
        elapsed_iterations += 1
        utility_matrix_Z = construct_utility(low_rank_candidate_L, sparse_matrix)
        low_rank_candidate_L = find_low_rank(
            utility_matrix_Z, target_rank, low_rank_candidate_L
        )

        if tolerance is not None:
            # TODO: do we really want the norm here, not the sum? Also, Frobenius norm ok?
            loss = np.linalg.norm(np.subtract(utility_matrix_Z, low_rank_candidate_L))
            if loss < tolerance:
                break
    # TODO: Check validity of solution?
    return low_rank_candidate_L


def find_low_rank(
    utility_matrix: npt.NDArray, target_rank: int, low_rank: npt.NDArray
) -> npt.NDArray:
    (U, S, Vh) = np.linalg.svd(utility_matrix)
    # SVD yields U, Sigma, and V-Transpose, with U, V orthogonal
    # and Sigma positive, diagonal, with entries in descending order.
    # The S from svd, however, is just a 1-d vector with the diagonal's values,
    # so we'll need to tweak a bit.

    # enforce rank r by zeroing out everything in S past the first r entries
    S[(target_rank + 1) :] = 0

    # recover a complete diagonal matrix
    Sigma = np.zeros((U.shape[0], Vh.shape[1]))
    np.fill_diagonal(Sigma, S)

    # and compute the new low-rank matrix candidate by regular matrix multiplication
    # TODO: fuss over parenthesization i.e. should we do U @ Sigma, Vh vs U, Sigma @ Vh
    np.matmul(U, Sigma @ Vh, out=low_rank)
    return low_rank
    # TODO: Consider whether we need to keep a history of L or might reject a new L,
    # such that we don't want to overwrite the old one; then we'd just do:
    # low_rank_matrix = np.matmul(U, Sigma @ Vh)
    # return low_rank_matrix


def construct_utility(
    low_rank_matrix: npt.NDArray, base_matrix: npt.NDArray
) -> npt.NDArray:
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


if __name__ == "__main__":
    pass
