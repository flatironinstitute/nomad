"""Utility functions for matrix decomposition.

Functions:
    find_low_rank: Compute low-rank approximation to input matrix using stated SVD strategy.
    two_part_factor: Factor M x N matrix of rank r into A (M x r), B (r x N)
    two_part_factor_known_rank: Factor M x N matrix into A (M x r), B (r x N) with known rank r.

"""

from typing import Tuple
import numpy as np
from sklearn.decomposition import TruncatedSVD  # type: ignore

from fi_nomad.types import FloatArrayType, SVDStrategy


def _find_low_rank_full(
    utility_matrix: FloatArrayType, target_rank: int, low_rank: FloatArrayType
) -> FloatArrayType:
    """Compute a low-rank approximation to a matrix by doing a full SVD, then
    manually trimming to the top n values.

    Args:
        utility_matrix: The matrix to approximate
        target_rank: The target rank of the low-rank approximation (this determines
            the number of values to keep)
        low_rank: An existing numpy array whose block of memory will be reused
            to store the result of the approximation

    Returns:
        The low_rank matrix (by reference, even though it is modified in-place).
    """
    (svd_U, svd_S, svd_V) = np.linalg.svd(utility_matrix)
    # SVD yields U, Sigma, and V-Transpose, with U, V orthogonal
    # and Sigma positive diagonal, with entries in descending order.
    # The S from this svd implementation, however, is just a 1-d vector
    # with the diagonal's values, so we'll need to manipulate it a bit.

    # enforce rank r by zeroing out everything in S past the first r entries
    svd_S[(target_rank + 1) :] = 0

    # Project that vector onto a full appropriately-sized matrix
    new_Sigma = np.zeros((svd_U.shape[0], svd_V.shape[1]))
    np.fill_diagonal(new_Sigma, svd_S)

    # and compute the new low-rank matrix candidate by regular matrix multiplication
    np.matmul(svd_U, new_Sigma @ svd_V, out=low_rank, casting="unsafe")
    return low_rank


def _find_low_rank_random_truncated(
    utility_matrix: FloatArrayType, target_rank: int
) -> FloatArrayType:
    """Compute a low-rank approximation to a matrix via random truncated SVD.

    Args:
        utility_matrix: The matrix to approximate
        target_rank: The target rank of the low-rank approximation. Note that
            in contrast to the full-decomposition method, the underlying algorithm
            will throw an error if the requested rank is equal to or greater than
            the smaller dimension of the matrix (meaning also that this algorithm
            cannot round-trip a full-rank matrix).

    Returns:
        The low-rank approximation.
    """
    svd = TruncatedSVD(n_components=target_rank)
    reduced = svd.fit_transform(utility_matrix)
    low_rank: FloatArrayType = svd.inverse_transform(reduced)
    return low_rank


def _find_low_rank_exact_truncated(
    utility_matrix: FloatArrayType, target_rank: int
) -> FloatArrayType:
    """Compute a low-rank approximation to a matrix via arpack algorithm, which
    performs an exact truncated SVD.

    Args:
        utility_matrix: The matrix to approximate
        target_rank: The target rank of the low-rank approximation. Note that
            in contrast to the full-decomposition method, the underlying algorithm
            will throw an error if the requested rank is equal to or greater than
            the smaller dimension of the matrix (meaning also that this algorithm
            cannot round-trip a full-rank matrix).

    Returns:
        The low-rank approximation.
    """
    svd = TruncatedSVD(n_components=target_rank, algorithm="arpack")
    reduced = svd.fit_transform(utility_matrix)
    low_rank: FloatArrayType = svd.inverse_transform(reduced)
    return low_rank


def find_low_rank(
    utility_matrix: FloatArrayType,
    target_rank: int,
    low_rank: FloatArrayType,
    strategy: SVDStrategy,
) -> FloatArrayType:
    """Compute a low-rank approximation to an input matrix, using the
    requested SVD strategy.

    Args:
        utility_matrix: The matrix to approximate
        target_rank: The target rank of the approximation
        low_rank: A numpy array that will be reused as a preallocated
            memory block by the full exact SVD deconmposition strategy
        strategy: The SVD strategy to use (full, exact truncated, or random truncated)

    Raises:
        ValueError: If an unsupported SVD strategy is requested

    Returns:
        A numpy array storing the low-rank approximation.
    """
    if strategy == SVDStrategy.RANDOM_TRUNCATED:
        return _find_low_rank_random_truncated(utility_matrix, target_rank)
    if strategy == SVDStrategy.EXACT_TRUNCATED:
        return _find_low_rank_exact_truncated(utility_matrix, target_rank)
    if strategy == SVDStrategy.FULL:
        return _find_low_rank_full(utility_matrix, target_rank, low_rank)
    raise ValueError("Unsupported SVD strategy.")


def two_part_factor(matrix: FloatArrayType) -> Tuple[FloatArrayType, FloatArrayType]:
    """Factor matrix into two rectangular matrices with inner dimension matching its rank.

    Args:
        matrix: Low-rank matrix to factor into two

    Returns:
        Two matrices whose product is the original matrix.
    """
    rank = np.linalg.matrix_rank(matrix)
    (svd_U, svd_S, svd_Vt) = np.linalg.svd(matrix)
    inner_a = np.zeros((matrix.shape[0], rank))
    np.fill_diagonal(inner_a, svd_S)
    part_A = svd_U @ inner_a
    part_B = np.pad(np.eye(rank), ((0, 0), (0, matrix.shape[1] - rank))) @ svd_Vt
    return (part_A, part_B)


def two_part_factor_known_rank(
    matrix: FloatArrayType, rank: int
) -> Tuple[FloatArrayType, FloatArrayType]:
    """Factor matrix into two rectangular matrices with inner dimension `rank`

    Args:
        matrix: Low-rank matrix to factor into two
        rank: The desired inner dimension

    Returns:
        Two matrices whose product is the original matrix.
    """
    (svd_U, svd_S, svd_Vt) = np.linalg.svd(matrix)
    inner_a = np.zeros((matrix.shape[0], rank))
    np.fill_diagonal(inner_a, svd_S)
    part_A = svd_U @ inner_a
    part_B = np.pad(np.eye(rank), ((0, 0), (0, matrix.shape[1] - rank))) @ svd_Vt
    return (part_A, part_B)
