import numpy as np
from sklearn.decomposition import TruncatedSVD  # type: ignore
from scipy.stats import norm as normal  # type: ignore
from typing import cast

from .types import FloatArrayType, InitializationStrategy, SVDStrategy, LossType


def initialize_low_rank_candidate(
    sparse_matrix: FloatArrayType, method: InitializationStrategy
) -> FloatArrayType:
    low_rank_candidate = None
    if method == InitializationStrategy.COPY:
        low_rank_candidate = np.copy(sparse_matrix)
    elif method == InitializationStrategy.BROADCAST_MEAN:
        mean = np.mean(sparse_matrix)
        low_rank_candidate = np.ones(sparse_matrix.shape) * mean
    else:
        raise ValueError("Unsupported initialization strategy.")
    return low_rank_candidate


##### Stats Utility #####


# TODO: As defined, this will underflow if x > ~37, or overflow if x < -2e9 or so,
# generating a warning.
# Unclear if these values are actually realistic in practice, and whether we even
# care, since they're only epsilon away from 0 or 1 (respectively).
# We might wish to avoid the warning by replacing the result for known out-of-bounds inputs,
# although numpy *should* be doing the right thing by replacing with 0/1 anyway.
def pdf_to_cdf_ratio_psi(x: float | FloatArrayType) -> FloatArrayType:
    return cast(FloatArrayType, np.exp(normal.logpdf(x) - normal.logcdf(x)))


def get_stddev_normalized_matrix_gamma(
    unnormalized_matrix: FloatArrayType, variance_sigma_sq: float
) -> FloatArrayType:
    return cast(FloatArrayType, unnormalized_matrix / np.sqrt(variance_sigma_sq))


def get_elementwise_posterior_variance_dZbar(
    sparse_matrix: FloatArrayType,
    model_variance: float,
    stddevnorm_model_matrix_gamma: FloatArrayType,
) -> FloatArrayType:
    ## Compute matrix of elementwise posterior variance dZ-bar.
    ## This will be:
    ##      0 when the underlying sparse matrix is nonzero, and
    ##      sigma^2[1 + gamma psi(-gamma) - psi(-gamma)^2] elsewhere.
    dZbar = np.zeros(stddevnorm_model_matrix_gamma.shape)

    psi_of_neg_gamma = pdf_to_cdf_ratio_psi(
        -1 * stddevnorm_model_matrix_gamma[sparse_matrix == 0]
    )
    dZbar[sparse_matrix == 0] = (
        1
        + (
            stddevnorm_model_matrix_gamma[sparse_matrix == 0] * psi_of_neg_gamma
            - psi_of_neg_gamma**2
        )
    ) * model_variance

    return dZbar


##### Losses and Likelihoods #####


# DON'T USE THIS--should've been using the Frobenius norm, which is this without the squaring
def _squared_difference_loss(
    utility: FloatArrayType, candidate: FloatArrayType
) -> float:
    loss = float(np.linalg.norm(np.subtract(utility, candidate)) ** 2)
    return loss


def _frobenius_norm_loss(utility: FloatArrayType, candidate: FloatArrayType) -> float:
    return float(np.linalg.norm(np.subtract(utility, candidate)))


def compute_loss(
    utility: FloatArrayType,
    candidate: FloatArrayType,
    type: LossType = LossType.FROBENIUS,
) -> float:
    if type == LossType.FROBENIUS:
        return _frobenius_norm_loss(utility, candidate)
    if type == LossType.SQUARED_DIFFERENCE:
        return _squared_difference_loss(utility, candidate)
    raise ValueError(f"Unrecognized loss type {type} requested.")


def low_rank_matrix_log_likelihood(
    sparse_matrix: FloatArrayType,
    low_rank_matrix: FloatArrayType,
    stddev_norm_lr_gamma: FloatArrayType,
    variance_sigma_sq: float,
) -> float:
    scale = np.sqrt(variance_sigma_sq)
    ### i.e. sum over log-likelihood matrix P_ij, where
    ##   P_ij = log cdf (-gamma) if S = 0
    ##   P_ij = log of 1/sqrt(2 pi sigma^2) * e^(-1/(2 sigma^2) * (S_ij - L_ij)^2) if S > 0
    ##### [which is logpdf[(S - L)/sigma], again over sigma.]
    ##### which in turn is just the logpdf of a gaussian of variance sigma-squared and center/mu L_ij.
    probability_matrix = np.empty(sparse_matrix.shape)
    probability_matrix[sparse_matrix == 0] = normal.logcdf(
        -1 * stddev_norm_lr_gamma[sparse_matrix == 0]
    )
    probability_matrix[sparse_matrix > 0] = normal.logpdf(
        sparse_matrix[sparse_matrix > 0],
        loc=low_rank_matrix[sparse_matrix > 0],
        scale=scale,
    )

    # # TODO: TEST THIS HARD
    # draws = normal.rvs(
    #     loc=low_rank_matrix[sparse_matrix > 0], scale=scale
    # )
    # probability_matrix[sparse_matrix > 0] = normal.logpdf(draws)
    return float(np.sum(probability_matrix))


##### Matrix decomposition (SVD) #####


def _find_low_rank_full(
    utility_matrix: FloatArrayType, target_rank: int, low_rank: FloatArrayType
) -> FloatArrayType:
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
    np.matmul(U, Sigma @ Vh, out=low_rank, casting="unsafe")
    return low_rank


# NOTE: This one actually requires you to not overestimate the real rank of the matrix.
def _find_low_rank_random_truncated(
    utility_matrix: FloatArrayType, target_rank: int
) -> FloatArrayType:
    svd = TruncatedSVD(n_components=target_rank)
    reduced = svd.fit_transform(utility_matrix)
    low_rank: FloatArrayType = svd.inverse_transform(reduced)
    return low_rank


# NOTE: This one actually requires you to not overestimate the real rank of the matrix.
def _find_low_rank_exact_truncated(
    utility_matrix: FloatArrayType, target_rank: int
) -> FloatArrayType:
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
    if strategy == SVDStrategy.RANDOM_TRUNCATED:
        return _find_low_rank_random_truncated(utility_matrix, target_rank)
    if strategy == SVDStrategy.EXACT_TRUNCATED:
        return _find_low_rank_exact_truncated(utility_matrix, target_rank)
    if strategy == SVDStrategy.FULL:
        return _find_low_rank_full(utility_matrix, target_rank, low_rank)
    raise ValueError("Unsupported SVD strategy.")
