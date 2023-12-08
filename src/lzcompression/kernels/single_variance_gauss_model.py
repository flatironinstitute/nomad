from typing import cast
import numpy as np
import logging
from lzcompression.kernels.kernel_base import KernelBase

from lzcompression.types import (
    FloatArrayType,
    KernelInputType,
    KernelReturnType,
    LossType,
    SingleVarianceGaussianModelKernelReturnType,
)
from lzcompression.util.util import (
    compute_loss,
    find_low_rank,
)
from lzcompression.util.gauss_model_util import (
    get_stddev_normalized_matrix_gamma,
    get_posterior_means_Z,
    get_elementwise_posterior_variance_dZbar,
    estimate_new_model_variance,
    target_matrix_log_likelihood,
)

logger = logging.getLogger(__name__)


class SingleVarianceGaussianModelKernel(KernelBase):
    """Estimator for a Gaussian mdel (L, v) for a sparse nonnegative matrix X with shared variance across all entries.

    The algorithm uses an expectation-maximization strategy to learn the parameters of a
    Gaussian model with means L and universal variance v that maximizes the likelihood of the data X.
    We alternate between:
        - a posterior-evaluation step, in which we compute the posterior means Z and
          posterior variances dZ given the model (L, v) and the data X; and
        - an update step, in which we generate a new model given the posteriors and data
    """

    def __init__(self, input: KernelInputType) -> None:
        super().__init__(input)
        initial_variance = float(np.var(input.sparse_matrix_X))
        initial_gamma = get_stddev_normalized_matrix_gamma(
            input.low_rank_candidate_L, initial_variance
        )

        # We use a more semantic name for this use case
        self.model_means_L: FloatArrayType = input.low_rank_candidate_L
        self.model_variance_sigma_squared: float = initial_variance
        self.gamma: FloatArrayType = initial_gamma
        self.likelihood: float = float("-inf")

    def step(self) -> None:
        """Each step is comprised of three parts: posterior-evaluation, L-update, and likelihood monitoring.

        Posterior evaluation computes the means and elementwise variance of the current estimate, given the
        original data.

        L-update enforces low rank on the current posterior means, estimates the variance of that new model,
        and computes gamma (the means normalized by the model standard deviation).

        Likelihood monitoring computes the likelihood of the original input given the updated model. A warning
        is logged if this value decreases (no model update should reduce the likelihood).
        """
        ## Posterior-evaluation step:
        posterior_means_Z = get_posterior_means_Z(
            self.model_means_L,
            self.sparse_matrix_X,
            self.gamma,
            self.model_variance_sigma_squared,
        )
        posterior_var_dZ = get_elementwise_posterior_variance_dZbar(
            self.sparse_matrix_X, self.model_variance_sigma_squared, self.gamma
        )

        ## L-update step:
        self.model_means_L = find_low_rank(
            posterior_means_Z, self.target_rank, self.model_means_L, self.svd_strategy
        )
        self.model_variance_sigma_squared = cast(
            float,
            estimate_new_model_variance(
                posterior_means_Z, self.model_means_L, posterior_var_dZ
            ),
        )
        self.gamma = get_stddev_normalized_matrix_gamma(
            self.model_means_L, self.model_variance_sigma_squared
        )

        ### Monitor likelihood:
        likelihood = target_matrix_log_likelihood(
            self.sparse_matrix_X,
            self.model_means_L,
            self.gamma,
            self.model_variance_sigma_squared,
        )
        # TODO: don't hard-code this warning here
        if likelihood < self.likelihood:
            logger.warning(
                f"Iteration ${self.elapsed_iterations}: Likelihood decreased, from {self.likelihood} to {likelihood}"
            )
        self.likelihood = likelihood

        self.loss = compute_loss(
            posterior_means_Z, self.model_means_L, LossType.FROBENIUS
        )

    def running_report(self) -> str:
        txt = f"\t\tIteration {self.elapsed_iterations}: loss: {self.loss} likelihood: {self.likelihood}"
        return txt

    def report(self) -> KernelReturnType:
        text = f"{self.elapsed_iterations} total iterations, final loss {self.loss} likelihood {self.likelihood}"
        data = SingleVarianceGaussianModelKernelReturnType(
            self.model_means_L,
            self.model_variance_sigma_squared,
        )
        return KernelReturnType(text, data)
