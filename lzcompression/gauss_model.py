import numpy as np
from typing import Tuple
import time
import logging

from lzcompression.types import (
    FloatArrayType,
    InitializationStrategy,
    SVDStrategy,
    LossType,
)
from lzcompression.util import (
    initialize_low_rank_candidate,
    low_rank_matrix_log_likelihood,
    get_stddev_normalized_matrix_gamma,
    compute_loss,
    get_elementwise_posterior_variance_dZbar,
    find_low_rank,
    pdf_to_cdf_ratio_psi,
)

logger = logging.getLogger(__name__)


def compress_sparse_matrix_probabilistic(
    sparse_matrix: FloatArrayType,
    target_rank: int,
    *,
    svd_strategy: SVDStrategy = SVDStrategy.RANDOM_TRUNCATED,
    initialization: InitializationStrategy = InitializationStrategy.BROADCAST_MEAN,
    tolerance: float | None = None,
    manual_max_iterations: int | None = None,
    verbose: bool = False,
) -> Tuple[FloatArrayType, float]:
    if verbose:
        logger.setLevel(logging.INFO)
    logger.info(f"\tInitiating run, {target_rank=}, {tolerance=}")
    if np.any(sparse_matrix[sparse_matrix < 0]):
        raise ValueError("Sparse input matrix must be nonnegative.")

    run_start_time = time.perf_counter()

    low_rank_candidate_L = initialize_low_rank_candidate(sparse_matrix, initialization)
    model_variance_sigma_squared = float(np.var(sparse_matrix))
    gamma = get_stddev_normalized_matrix_gamma(
        low_rank_candidate_L, model_variance_sigma_squared
    )

    max_iterations = (
        manual_max_iterations
        if manual_max_iterations is not None
        else 100 * target_rank
    )
    elapsed_iterations = 0
    last_iter_likelihood = float("-inf")
    loss = float("inf")

    loop_start_time = time.perf_counter()
    while elapsed_iterations < max_iterations:
        elapsed_iterations += 1
        # Alternate constructing utility matrix Z and estimating low-rank model candidate L
        ## Construction step:
        utility_matrix_Z = construct_posterior_model_matrix_Z(
            low_rank_candidate_L, sparse_matrix, gamma, model_variance_sigma_squared
        )
        posterior_var_dZ = get_elementwise_posterior_variance_dZbar(
            sparse_matrix, model_variance_sigma_squared, gamma
        )

        ## L-recovery step:
        low_rank_candidate_L = find_low_rank(
            utility_matrix_Z, target_rank, low_rank_candidate_L, svd_strategy
        )
        model_variance_sigma_squared = estimate_new_model_variance(
            utility_matrix_Z, low_rank_candidate_L, posterior_var_dZ
        )
        gamma = get_stddev_normalized_matrix_gamma(
            low_rank_candidate_L, model_variance_sigma_squared
        )

        ### Monitor likelihood:
        likelihood = low_rank_matrix_log_likelihood(
            sparse_matrix, low_rank_candidate_L, gamma, model_variance_sigma_squared
        )
        if likelihood < last_iter_likelihood:
            logger.warning(
                f"Iteration {elapsed_iterations}: likelihood decreased, from {last_iter_likelihood} to {likelihood}"
            )
        last_iter_likelihood = likelihood

        ### Monitor loss:
        loss = compute_loss(utility_matrix_Z, low_rank_candidate_L, LossType.FROBENIUS)
        logger.info(f"\t\tIteration {elapsed_iterations}: {loss=} {likelihood=}")
        if tolerance is not None and loss < tolerance:
            break

    end_time = time.perf_counter()

    init_e = loop_start_time - run_start_time
    loop_e = end_time - loop_start_time
    per_loop_e = loop_e / (elapsed_iterations if elapsed_iterations > 0 else 1)
    logger.info(
        f"{elapsed_iterations} total iterations, final loss {loss} likelihood {last_iter_likelihood}"
    )
    logger.info(
        f"\tInitialization took {init_e} loop took {loop_e} overall ({per_loop_e}/ea)"
    )

    return (low_rank_candidate_L, model_variance_sigma_squared)


def construct_posterior_model_matrix_Z(
    low_rank_matrix: FloatArrayType,
    sparse_matrix: FloatArrayType,
    stddev_normalized_lowrank: FloatArrayType,
    variance_sigma_sq: float,
) -> FloatArrayType:
    ## compute posterior mean Z-bar, a matrix s.t.
    #### Z-bar_ij = S_ij if S_ij > 0
    #### Z-bar_ij = L_ij - sigma * psi(gamma) if S = 0.
    posterior_matrix = np.copy(sparse_matrix)
    sigma = np.sqrt(variance_sigma_sq)
    # fmt: off
    posterior_matrix[sparse_matrix == 0] = \
        low_rank_matrix[sparse_matrix == 0] - \
        sigma * pdf_to_cdf_ratio_psi(-1 * stddev_normalized_lowrank[sparse_matrix == 0])
    # fmt: on

    return posterior_matrix


def estimate_new_model_variance(
    utility_matrix_Z: FloatArrayType,
    low_rank_candidate_L: FloatArrayType,
    posterior_var_dZ: FloatArrayType,
) -> float:
    sigma_sq = np.mean(
        np.square(utility_matrix_Z - low_rank_candidate_L)
        + posterior_var_dZ  # this (being a variance) is already squared
    )
    return float(sigma_sq)
