import numpy as np
from scipy.stats import norm as normal  # type: ignore
from typing import cast, Tuple
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
    compute_loss,
    find_low_rank,
    pdf_to_cdf_ratio_psi,
)

logger = logging.getLogger(__name__)


def estimate_gaussian_model(
    sparse_matrix_X: FloatArrayType,
    target_rank: int,
    *,
    svd_strategy: SVDStrategy = SVDStrategy.RANDOM_TRUNCATED,
    initialization: InitializationStrategy = InitializationStrategy.BROADCAST_MEAN,
    tolerance: float | None = None,
    manual_max_iterations: int | None = None,
    verbose: bool = False,
) -> Tuple[FloatArrayType, float]:
    """Estimate a Gaussian model (L, v) for a sparse nonnegative matrix X.

    The algorithm uses an expectation-maximization strategy to learn the parameters of a
    Gaussian model with means L and variance v that maximizes the likelihood of the data X.
    We alternate between:
        - a posterior-evaluation step, in which we compute the posterior means Z and
          posterior variances dZ given the model (L, v) and the data X; and
        - an update step, in which we generate a new model given the posteriors and data

    Args:
        sparse_matrix_X: The sparse nonnegative matrix to decompose
        target_rank: The target rank of the low-rank representation
        svd_strategy (optional): Strategy to use for SVD. Defaults to SVDStrategy.RANDOM_TRUNCATED.
        initialization (optional): Strategy to use for initializing the low-rank representation.
            Defaults to InitializationStrategy.BROADCAST_MEAN.
        tolerance (optional): If set, the algorithm will terminate once the reconstruction loss of the
            estimate falls below this level. Defaults to None.
        manual_max_iterations (optional): If set, will override the default maximum iteration count
            (100 * target rank). Defaults to None.
        verbose (optional): If True, will use a logger to report performance data. Defaults to False.

    Raises:
        ValueError: If the input matrix is not nonnegative.

    Returns:
        A tuple containing the parameters (L, v) of a Gaussian model for X.
        L is the matrix of model means, and v is its variance.
    """
    if verbose:
        logger.setLevel(logging.INFO)
    logger.info(f"\tInitiating run, {target_rank=}, {tolerance=}")
    if np.any(sparse_matrix_X[sparse_matrix_X < 0]):
        raise ValueError("Sparse input matrix must be nonnegative.")

    run_start_time = time.perf_counter()

    model_means_L = initialize_low_rank_candidate(sparse_matrix_X, initialization)
    model_variance_sigma_squared = float(np.var(sparse_matrix_X))
    gamma = get_stddev_normalized_matrix_gamma(
        model_means_L, model_variance_sigma_squared
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

        ## Posterior-evaluation step:
        posterior_means_Z = get_posterior_means_Z(
            model_means_L, sparse_matrix_X, gamma, model_variance_sigma_squared
        )
        posterior_var_dZ = get_elementwise_posterior_variance_dZbar(
            sparse_matrix_X, model_variance_sigma_squared, gamma
        )

        ## L-update step:
        model_means_L = find_low_rank(
            posterior_means_Z, target_rank, model_means_L, svd_strategy
        )
        model_variance_sigma_squared = estimate_new_model_variance(
            posterior_means_Z, model_means_L, posterior_var_dZ
        )
        gamma = get_stddev_normalized_matrix_gamma(
            model_means_L, model_variance_sigma_squared
        )

        ### Monitor likelihood:
        # TODO: RENAME low_rank_matrix_log_likelihood
        likelihood = low_rank_matrix_log_likelihood(
            sparse_matrix_X, model_means_L, gamma, model_variance_sigma_squared
        )
        if likelihood < last_iter_likelihood:
            logger.warning(
                f"Iteration {elapsed_iterations}: likelihood decreased, from {last_iter_likelihood} to {likelihood}"
            )
        last_iter_likelihood = likelihood

        ### Monitor loss:
        loss = compute_loss(posterior_means_Z, model_means_L, LossType.FROBENIUS)
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

    return (model_means_L, model_variance_sigma_squared)


def get_posterior_means_Z(
    prior_means_L: FloatArrayType,
    sparse_matrix: FloatArrayType,
    stddev_normalized_lowrank: FloatArrayType,
    variance_sigma_sq: float,
) -> FloatArrayType:
    """Estimate the posterior means of the Gaussian model, based on the prior parameters
    (means L and variance sigma-squared) and the observed data sparse_matrix.

    This function implements Equation 4.8 from Saul (2022).

    Since the positive values of X should be exactly recoverable, those elements
    are treated as definitively known: the posterior mean must be the observed value
    of X (and the corresponding posterior variance should be 0).

    For elements which are 0 in X, the posterior mean is estimated in a way to minimize
    overall variance [EXPLAIN MORE].

    Args:
        prior_means_L: The current model means
        sparse_matrix: The nonnegative sparse matrix being approximated
        stddev_normalized_lowrank: The current model means, with each element
            divided by the overall standard deviation (i.e. L over sqrt(v))
        variance_sigma_sq: The variance parameter of the prior model

    Returns:
        An updated matrix of means for the model, given the data.
    """
    ## compute posterior mean Z-bar, a matrix s.t.
    #### Z-bar_ij = S_ij if S_ij > 0
    #### Z-bar_ij = L_ij - sigma * psi(gamma) if S = 0.
    posterior_matrix = np.copy(sparse_matrix)
    sigma = np.sqrt(variance_sigma_sq)
    # fmt: off
    posterior_matrix[sparse_matrix == 0] = \
        prior_means_L[sparse_matrix == 0] - \
        sigma * pdf_to_cdf_ratio_psi(-1 * stddev_normalized_lowrank[sparse_matrix == 0])
    # fmt: on

    return posterior_matrix


def estimate_new_model_variance(
    posterior_means_Z: FloatArrayType,
    prior_means_L: FloatArrayType,
    posterior_var_dZ: FloatArrayType,
) -> float:
    """Compute the updated overall variance estimate of the Gaussian model.

    This function implements Equation 4.12 from Saul (2022). It computes the
    mean of an implicit variance matrix, whose each element is composed of:
     - The square of the difference between the values of the posterior mean
       and prior mean at this element, and
     - The estimated posterior variance at this element

    In other words, given prior means L, posterior means Z and posterior
    variances dZ, create an implicit matrix S such that
        S_ij = (Z_ij - L_ij)^2 + dZ_ij
    then return the mean of the elements of S.

    Args:
        posterior_means_Z: Estimated posterior means, given the model and data
        prior_means_L: The current model means
        posterior_var_dZ: The elementwise posterior variance estimate, as
            computed by Equation 4.9 from Saul (2022).

    Returns:
        An updated estimate of the overall variance in the Gaussian model.
    """
    sigma_sq = np.mean(np.square(posterior_means_Z - prior_means_L) + posterior_var_dZ)
    return float(sigma_sq)


def get_stddev_normalized_matrix_gamma(
    unnormalized_matrix: FloatArrayType, variance_sigma_sq: float
) -> FloatArrayType:
    """Compute a utiliy matrix ("gamma" in Saul (2022)) in which each element is the
    corresponding element of prior means matrix, divided by the root
    of the model's variance parameter (hence "standard-deviation-normalized matrix").

    Args:
        unnormalized_matrix: The current model means parameter
        variance_sigma_sq: Variance parameter of the model (e.g. from equation 4.12 in
            Saul 2022)

    Returns:
        The prior means, scaled by the square root of prior variance.
    """
    return cast(FloatArrayType, unnormalized_matrix / np.sqrt(variance_sigma_sq))


def get_elementwise_posterior_variance_dZbar(
    sparse_matrix: FloatArrayType,
    model_variance: float,
    stddevnorm_matrix_gamma: FloatArrayType,
) -> FloatArrayType:
    """Estimate elementwise posterior variance, given the prior model and the
    observed matrix.

    This function implements equation 4.9 in Saul (2022).

    Per that formula:
     - where X has positive values, the model's variance should be 0
       (the non-zero values of X are known definitively)
     - where X has 0s, the posterior variance is estimated based on:
       - the prior model variance sigma^2, and
       - the value of gamma, i.e. the prior mean divided by sqrt(variance)
       Concisely, for X's 0s, return:
          sigma^2 * [ 1 + gamma * psi( -gamma) - psi( -gamma)^2 ]
       where psi is the ratio of pdf to cdf.

    Args:
        sparse_matrix: The obserevd data, a sparse nonnegative matrix ("X")
            (only really needed for the location of its zero-valued entries)
        model_variance: Variance parameter of the model
        stddevnorm_matrix_gamma: Prior means L, scaled by the root of
            prior variance sigma-squared

    Returns:
        A matrix containing the estimated posterior variance of the model.
    """
    ## Per Eqn 4.9, dZbar is:
    ##      0 when the sparse matrix is nonzero, and
    ##      sigma^2[1 + gamma psi(-gamma) - psi(-gamma)^2] elsewhere.
    # Initialize a 0 matrix (we will keep the 0s for those entries
    # where the sparse matrix has nonzero values):
    dZbar = np.zeros(stddevnorm_matrix_gamma.shape)

    # Cache the pdf-to-cdf ratio for entries corresponding to
    # sparse_matrix's zero values
    psi_of_neg_gamma = pdf_to_cdf_ratio_psi(
        -1 * stddevnorm_matrix_gamma[sparse_matrix == 0]
    )
    # And now populate those entries per the formula
    dZbar[sparse_matrix == 0] = (
        1
        + (
            stddevnorm_matrix_gamma[sparse_matrix == 0] * psi_of_neg_gamma
            - psi_of_neg_gamma**2
        )
    ) * model_variance

    return dZbar


def low_rank_matrix_log_likelihood(
    sparse_matrix: FloatArrayType,
    prior_means_L: FloatArrayType,
    stddev_norm_lr_gamma: FloatArrayType,
    variance_sigma_sq: float,
) -> float:
    """Compute the likelihood of observed nonnegative matrix X, with respect to
    a Gaussian model with means L and variance v.

    This implements Equation 4.6 in Saul (2022).

    Given:
      X (the observed sparse nonnegative matrix),
      L (the means of the current model),
      v (the current model variance), and
      gamma (L divided by sqrt(v))
    we compute a (log) likelihood estimate for each element in X; since the
    element values are independent, summing the log likelihoods yields the overall
    likelihood of X under this model.

    The likelihoods are defined as:
        For elements ij where X_ij is 0:
            the cdf of -1 * gamma_ij (with respect to a 0,1 gaussian)
        For elements X_ij > 0:
            the pdf of X_ij w/r/t a Gaussian with variance v centered at L_ij

    If the model is successfully adapting to better fit the data, the values
    of this likelihood should be nondecreasing over the course of the algorithm.

    Args:
        sparse_matrix: The matrix for which a low-rank approximation is sought
        prior_means_L: The means of the current model
        stddev_norm_lr_gamma: L, divided by the root of the model variance parameter
        variance_sigma_sq: The variance of the current model

    Returns:
        The sum of the elementwise log-likelihoods, representing the overall likelihood
        of observing the matrix X given the model.
    """
    scale = np.sqrt(variance_sigma_sq)
    probability_matrix = np.empty(sparse_matrix.shape)
    probability_matrix[sparse_matrix == 0] = normal.logcdf(
        -1 * stddev_norm_lr_gamma[sparse_matrix == 0]
    )
    probability_matrix[sparse_matrix > 0] = normal.logpdf(
        sparse_matrix[sparse_matrix > 0],
        loc=prior_means_L[sparse_matrix > 0],
        scale=scale,
    )

    return float(np.sum(probability_matrix))
