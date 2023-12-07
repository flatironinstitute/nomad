import numpy as np
from sklearn.decomposition import TruncatedSVD  # type: ignore
from scipy.stats import norm as normal  # type: ignore
from typing import cast, Union

from lzcompression.types import (
    FloatArrayType,
    InitializationStrategy,
    SVDStrategy,
    LossType,
)


def initialize_low_rank_candidate(
    input_matrix: FloatArrayType, method: InitializationStrategy
) -> FloatArrayType:
    """Given a sparse matrix, create a starting-point guess for the low-rank representation.

    Currently supported strategies are:
     - BROADCAST_MEAN: fill the low-rank candidate with the mean of the sparse matrix's values
     - COPY: simply return a copy of the input sparse matrix

    Args:
        input_matrix: The sparse nonnegative matrix input ("X"). With the KNOWN_MATRIX
            strategy, this field is instead the desired initialization value (e.g. from checkpoint)
        method: The strategy to employ to generate the starting point

    Raises:
        ValueError: On request for an unsupported initialization strategy.

    Returns:
        Initial estimate for a low-rank representation.
    """
    low_rank_candidate = None
    if method == InitializationStrategy.COPY:
        low_rank_candidate = np.copy(input_matrix)
    elif method == InitializationStrategy.BROADCAST_MEAN:
        low_rank_candidate = np.full(input_matrix.shape, np.mean(input_matrix))
    elif method == InitializationStrategy.ROWWISE_MEAN:
        shape = input_matrix.shape
        row_means = np.mean(input_matrix, axis=1)
        low_rank_candidate = np.repeat(row_means, shape[1]).reshape(shape)
    elif method == InitializationStrategy.KNOWN_MATRIX:
        low_rank_candidate = np.copy(input_matrix)
    else:
        raise ValueError("Unsupported initialization strategy.")
    return low_rank_candidate


##### Stats Utility #####


# NOTE: As defined, this will underflow if x > ~37, or overflow if x < -2e9 or so,
# generating a warning.
# Unclear if these values are actually realistic in practice, and whether we even
# care, since they're only epsilon away from 0 or 1 (respectively).
# We might wish to avoid the warning by replacing the result for known out-of-bounds inputs,
# although numpy *should* be doing the right thing by replacing with 0/1 anyway.
def pdf_to_cdf_ratio_psi(x: Union[float, FloatArrayType]) -> FloatArrayType:
    """Compute the ratio of the probability density function to the
    cumulative distribution function, with respect to a normal distribution with
    zero mean and unit variance.

    This function is abbreviated "psi" in Saul (2022).

    Args:
        x: The value (or array of values) to compute

    Returns:
        A numpy array representing this value.
    """
    return cast(FloatArrayType, np.exp(normal.logpdf(x) - normal.logcdf(x)))


##### Losses and Likelihoods #####


# Included for historical reasons, but better to use the Frobenius norm instead.
def _squared_difference_loss(
    utility: FloatArrayType, candidate: FloatArrayType
) -> float:
    """Compute the square of the Frobenius norm of the difference between a target
    matrix and a utility matrix.

    Args:
        utility: A utility matrix ("Z" in algorithms from Saul (2022))
        candidate: The target matrix being evaluated

    Returns:
        Scalar loss estimate
    """
    loss = float(np.linalg.norm(np.subtract(utility, candidate)) ** 2)
    return loss


def _frobenius_norm_loss(utility: FloatArrayType, candidate: FloatArrayType) -> float:
    """Compute the Frobenius norm of the difference between a target
    matrix and a utility matrix.

    Args:
        utility: A utility matrix ("Z" in algorithms from Saul (2022))
        candidate: The target matrix being evaluated

    Returns:
        Scalar loss estimate
    """
    return float(np.linalg.norm(np.subtract(utility, candidate)))


def compute_loss(
    utility: FloatArrayType,
    candidate: FloatArrayType,
    type: LossType = LossType.FROBENIUS,
) -> float:
    """Compute a scalar estimate of a loss between the utility matrix (Z) and
    target matrix L.

    In the model-free algorithm, L is a direct low-rank approximation;
    Z's role is to rigorously enforce that X, the sparse nonnegative
    matrix being approximated, can be recovered from L by applying ReLU.

    In the Gaussian-model algorithm, L stores the prior means of the model
    while Z has the posterior means; lower loss between them represents
    improved fit to the data.

    Implemented losses are the Frobenius norm, and squared Frobenius norm.

    Args:
        utility: "Z" matrix
        candidate: "L" matrix
        type: Type of loss to use. Defaults to LossType.FROBENIUS.

    Raises:
        ValueError: If an unsupported loss type is requested.

    Returns:
        Scalar loss value representing how well L approximates Z (and,
        by proxy, X).
    """
    if type == LossType.FROBENIUS:
        return _frobenius_norm_loss(utility, candidate)
    if type == LossType.SQUARED_DIFFERENCE:
        return _squared_difference_loss(utility, candidate)
    raise ValueError(f"Unrecognized loss type {type} requested.")


##### Matrix decomposition (SVD) #####


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
    (U, S, Vh) = np.linalg.svd(utility_matrix)
    # SVD yields U, Sigma, and V-Transpose, with U, V orthogonal
    # and Sigma positive diagonal, with entries in descending order.
    # The S from this svd implementation, however, is just a 1-d vector
    # with the diagonal's values, so we'll need to manipulate it a bit.

    # enforce rank r by zeroing out everything in S past the first r entries
    S[(target_rank + 1) :] = 0

    # Project that vector onto a full appropriately-sized matrix
    Sigma = np.zeros((U.shape[0], Vh.shape[1]))
    np.fill_diagonal(Sigma, S)

    # and compute the new low-rank matrix candidate by regular matrix multiplication
    np.matmul(U, Sigma @ Vh, out=low_rank, casting="unsafe")
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
