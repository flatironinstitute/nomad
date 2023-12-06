import numpy as np
from scipy.stats import norm as normal  # type: ignore
from typing import cast

from lzcompression.types import (
    FloatArrayType,
)
from lzcompression.util.util import (
    pdf_to_cdf_ratio_psi,
)


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

    For elements which are 0 in X, the model's expected value should be nonpositive.
    So we estimate the posterior mean using the formula for a right-truncated
    Gaussian.

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


def estimate_new_model_global_variance(
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
    zero_indices = sparse_matrix == 0
    psi_of_neg_gamma = pdf_to_cdf_ratio_psi(-1 * stddevnorm_matrix_gamma[zero_indices])
    # And now populate those entries per the formula
    dZbar[zero_indices] = (
        1
        + (
            stddevnorm_matrix_gamma[zero_indices] * psi_of_neg_gamma
            - psi_of_neg_gamma**2
        )
    ) * model_variance

    return dZbar


def target_matrix_log_likelihood(
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
    zero_indices = sparse_matrix == 0
    nonzero_indices = np.invert(zero_indices)
    # The following at least avoids *explicitly* creating & populating an empty matrix
    # just to sum over it
    sum = 0.0
    sum += np.sum(normal.logcdf(-1 * stddev_norm_lr_gamma[zero_indices]))
    sum += np.sum(
        normal.logpdf(
            sparse_matrix[nonzero_indices],
            loc=prior_means_L[nonzero_indices],
            scale=scale,
        )
    )

    return float(sum)
