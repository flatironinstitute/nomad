"""Utility functions for Gaussian-model kernels. Heavy computation should go here in
smaller functions (not in the classes themselves) to facilitate testing and incremental
improvement.
"""
from typing import Union, cast, Any
import numpy as np
from scipy.stats import norm as normal  # type: ignore

from fi_nomad.types.types import (
    FloatArrayType,
)
from fi_nomad.util.util import (
    pdf_to_cdf_ratio_psi,
)


def broadcast_rowwise_variance(
    variance: Union[float, FloatArrayType],
    target: FloatArrayType,
    *,
    limiter: Any = None,
) -> Union[float, FloatArrayType]:
    """Ensures consistent handling of scalar-variance and rowwise-variance stats functions
    for Gaussian-model kernels.

    When a single (scalar) variance is used, numpy will broadcast effectively; but when
    a rowwise variance is used, the matrix layouts don't naturally match. This function
    broadcasts the rowwise variance correctly, bringing it up to match the rank of the
    matrix it's operating on, along with optionally filtering to certain elements.

    Args:
        variance: The variance to broadcast (scalar or rowwise)
        target: The matrix to be modified by the resulting broadcast variance
        limiter: If set, determines the subset of values of the broadcast variance
            to return. Defaults to None.

    Returns:
        The variance, with all shape issues resolved.
    """
    if np.isscalar(variance):
        return cast(float, variance)
    var = cast(FloatArrayType, variance)
    shape = target.shape
    bcast = np.repeat(var, shape[1]).reshape(shape)
    if limiter is None:
        return bcast
    return cast(FloatArrayType, bcast[limiter])


def get_posterior_means_Z(
    prior_means_L: FloatArrayType,
    sparse_matrix: FloatArrayType,
    stddev_normalized_lowrank: FloatArrayType,
    variance_sigma_sq: Union[float, FloatArrayType],
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
        variance_sigma_sq: The variance parameter of the prior model; may be scalar
            or rowwise.

    Returns:
        An updated matrix of means for the model, given the data.
    """
    ## compute posterior mean Z-bar, a matrix s.t.
    #### Z-bar_ij = S_ij if S_ij > 0
    #### Z-bar_ij = L_ij - sigma * psi(gamma) if S = 0.
    posterior_matrix = np.copy(sparse_matrix)
    sigma: Union[float, FloatArrayType] = np.sqrt(variance_sigma_sq)
    element_filter = sparse_matrix == 0
    sigma = broadcast_rowwise_variance(sigma, sparse_matrix, limiter=element_filter)
    # fmt: off
    posterior_matrix[element_filter] = \
        prior_means_L[element_filter] - \
        sigma * pdf_to_cdf_ratio_psi(-1 * stddev_normalized_lowrank[element_filter])
    # fmt: on

    return posterior_matrix


def estimate_new_model_variance(
    posterior_means_Z: FloatArrayType,
    prior_means_L: FloatArrayType,
    posterior_var_dZ: FloatArrayType,
    *,
    rowwise: bool = False,
) -> Union[float, FloatArrayType]:
    """Compute the updated variance estimate of the Gaussian model.

    This function implements Equation 4.12 from Saul (2022). It computes a
    mean of an implicit variance matrix, whose each element is composed of:
     - The square of the difference between the values of the posterior mean
       and prior mean at this element, and
     - The estimated posterior variance at this element

    The variance is averaged over all elements by default, but if rowwise
    variance is requested, then variances will be averaged separately
    over each row.

    In other words, given prior means L, posterior means Z and posterior
    variances dZ, create an implicit matrix S such that
        S_ij = (Z_ij - L_ij)^2 + dZ_ij
    then return the mean of the elements (or, if selected, the rows) of S.

    Args:
        posterior_means_Z: Estimated posterior means, given the model and data
        prior_means_L: The current model means
        posterior_var_dZ: The (elementwise/rowwise) posterior variance estimate,
            as computed by Equation 4.9 from Saul (2022).

    Returns:
        An updated estimate of the overall variance in the Gaussian model.
    """
    axis = None if not rowwise else 1
    sigma_sq: Union[float, FloatArrayType] = np.mean(
        np.square(posterior_means_Z - prior_means_L) + posterior_var_dZ, axis=axis
    )
    return sigma_sq


def get_stddev_normalized_matrix_gamma(
    unnormalized_matrix: FloatArrayType, variance_sigma_sq: Union[float, FloatArrayType]
) -> FloatArrayType:
    """Compute a utiliy matrix ("gamma" in Saul (2022)) in which each element is the
    corresponding element of prior means matrix, divided by the root
    of the model's variance parameter (hence "standard-deviation-normalized matrix").

    Args:
        unnormalized_matrix: The current model means parameter
        variance_sigma_sq: Variance parameter of the model (e.g. from equation 4.12 in
            Saul 2022). May be scalar or row-wise.

    Returns:
        The prior means, scaled by the square root of prior variance.
    """
    stddev = broadcast_rowwise_variance(np.sqrt(variance_sigma_sq), unnormalized_matrix)
    return cast(FloatArrayType, unnormalized_matrix / stddev)


def scale_by_rowwise_stddev(
    to_scale: FloatArrayType, variance_sigma_sq: FloatArrayType
) -> FloatArrayType:
    """Scale up a matrix by the rowwise variance values.

    This is needed for the rowwise-variance model means update step, to undo the standard-deviation
    scaling that happens before SVD.

    Args:
        to_scale: Matrix to scale by the rowwise values. The intended use case is during
            the update step of the means for the rowwise-variance Gaussian model.
        variance_sigma_sq: Rowwise variance parameter of the model.

    Returns:
        The input matrix, scaled by multiplying by the rowwise variances.
    """
    stddev = broadcast_rowwise_variance(np.sqrt(variance_sigma_sq), to_scale)
    return cast(FloatArrayType, to_scale * stddev)


def get_elementwise_posterior_variance_dZbar(
    sparse_matrix: FloatArrayType,
    model_variance: Union[float, FloatArrayType],
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
        model_variance: Variance parameter of the model (rowwise or scalar)
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
    var = broadcast_rowwise_variance(model_variance, dZbar, limiter=zero_indices)
    psi_of_neg_gamma = pdf_to_cdf_ratio_psi(-1 * stddevnorm_matrix_gamma[zero_indices])
    # And now populate those entries per the formula
    dZbar[zero_indices] = (
        1
        + (
            stddevnorm_matrix_gamma[zero_indices] * psi_of_neg_gamma
            - psi_of_neg_gamma**2
        )
    ) * var

    return dZbar


def target_matrix_log_likelihood(
    sparse_matrix: FloatArrayType,
    prior_means_L: FloatArrayType,
    stddev_norm_lr_gamma: FloatArrayType,
    variance_sigma_sq: Union[float, FloatArrayType],
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
            the pdf of X_ij w/r/t a Gaussian with variance v (or possibly
            rowwise variance vi) centered at L_ij

    If the model is successfully adapting to better fit the data, the values
    of this likelihood should be nondecreasing over the course of the algorithm.
    It is expected that caller will monitor this.

    Args:
        sparse_matrix: The matrix for which a low-rank approximation is sought
        prior_means_L: The means of the current model
        stddev_norm_lr_gamma: L, divided by the root of the model variance parameter
        variance_sigma_sq: The variance of the current model

    Returns:
        The sum of the elementwise log-likelihoods, representing the overall likelihood
        of observing the matrix X given the model.
    """
    zero_indices = sparse_matrix == 0
    nonzero_indices = np.invert(zero_indices)
    scale: Union[float, FloatArrayType] = np.sqrt(variance_sigma_sq)
    scale = broadcast_rowwise_variance(scale, sparse_matrix, limiter=nonzero_indices)
    # The following at least avoids *explicitly* creating & populating an empty matrix
    # just to sum over it
    total = 0.0
    total += np.sum(normal.logcdf(-1 * stddev_norm_lr_gamma[zero_indices]))
    total += np.sum(
        normal.logpdf(
            sparse_matrix[nonzero_indices],
            loc=prior_means_L[nonzero_indices],
            scale=scale,
        )
    )

    return float(total)
