"""Defines a variation of the Gaussian-model algorithm from Saul (2022) with per-row variance.

Classes:
    RowwiseVarianceGaussianModelKernel: Gaussian-model kernel with per-row variance.

"""
from typing import Tuple, cast
import logging
import numpy as np
from fi_nomad.kernels.kernel_base import KernelBase

from fi_nomad.types import (
    FloatArrayType,
    RowwiseVarianceGaussianModelKernelReturnType,
    KernelInputType,
    KernelReturnType,
    LossType,
)
from fi_nomad.util.util import (
    compute_loss,
    find_low_rank,
)
from fi_nomad.util.gauss_model_util import (
    get_stddev_normalized_matrix_gamma,
    get_posterior_means_Z,
    get_elementwise_posterior_variance_dZbar,
    estimate_new_model_variance,
    scale_by_rowwise_stddev,
    target_matrix_log_likelihood,
)

logger = logging.getLogger(__name__)

## Note:
# tested storing the rowwise variances as a full matrix rather than per-row.
# It might be marginally faster, but the results were inconclusive:
# average of maybe 2% speedup, and actually slower for 60% of the trials.
# Leave as is for now.


class RowwiseVarianceGaussianModelKernel(KernelBase):
    """Estimator for a Gaussian mdel (L, v) for a sparse nonnegative matrix X with shared
    variance per row,

    The algorithm uses an expectation-maximization strategy to learn the parameters of a
    Gaussian model with means L and row-wise variances V that maximizes the likelihood of
    the data X.
    We alternate between:
        - Updating the row-wise variance estimates, and
        - Updating the model mean estimates.
    Before each half-step, we evaluate the posterior means Z and rowwise variances dZ
    given the current model (L, V) and data X.
    After each update half-step, we compute the likelihood of the observed data, given
    the current parameters.
    """

    def __init__(self, indata: KernelInputType) -> None:
        super().__init__(indata)
        initial_variance: FloatArrayType = np.var(indata.sparse_matrix_X, axis=1)
        initial_gamma = get_stddev_normalized_matrix_gamma(
            indata.low_rank_candidate_L, initial_variance
        )

        # We use a more semantic name for this use case
        self.model_means_L: FloatArrayType = indata.low_rank_candidate_L
        self.model_variance_sigma_squared: FloatArrayType = initial_variance
        self.gamma: FloatArrayType = initial_gamma
        self.likelihood: float = float("-inf")
        self.half_step_likelihood: float = float("-inf")

    def do_pre_update(self) -> Tuple[FloatArrayType, FloatArrayType]:
        """Recompute posterior mean and variance, to be called before each half-step.

        Returns:
            Posterior mean and variance.
        """
        # pylint: disable=duplicate-code
        posterior_means_Z = get_posterior_means_Z(
            self.model_means_L,
            self.sparse_matrix_X,
            self.gamma,
            self.model_variance_sigma_squared,
        )
        posterior_var_dZ = get_elementwise_posterior_variance_dZbar(
            self.sparse_matrix_X, self.model_variance_sigma_squared, self.gamma
        )
        return (posterior_means_Z, posterior_var_dZ)

    def do_post_update(self) -> float:
        """Recompute variance-normalized matrix (gamma) and likelihood, to be called
        after each half-step.

        Returns:
            (log) likelihood under current model
        """
        # pylint: disable=duplicate-code
        gamma = get_stddev_normalized_matrix_gamma(
            self.model_means_L, self.model_variance_sigma_squared
        )
        self.gamma = gamma
        likelihood = target_matrix_log_likelihood(
            self.sparse_matrix_X,
            self.model_means_L,
            self.gamma,
            self.model_variance_sigma_squared,
        )
        return likelihood

    def make_updated_means_estimate(
        self, posterior_means_Z: FloatArrayType
    ) -> FloatArrayType:
        """Update current model means. Unlike in the scalar-global-variance version,
        with the rowwise variance we want to use SVD to find the best fit for the
        means, scaled by the rowwise standard deviation. After computing this matrix,
        we'll re-scale the result by multiplying by the rowwise stddev.

        Args:
            posterior_means_Z: Current posterior means estimates

        Returns:
            New means estimate with appropriate scaling.
        """
        # Unlike in the global-variance version, with rowwise variance we want to use SVD
        # to find the best fit for the rowwise-stddev-scaled means. Then we scale the
        # result back by multiplying by the rowwise standard deviation.
        normalized_means = get_stddev_normalized_matrix_gamma(
            posterior_means_Z, self.model_variance_sigma_squared
        )
        unscaled_new_means = find_low_rank(
            normalized_means, self.target_rank, self.model_means_L, self.svd_strategy
        )
        return scale_by_rowwise_stddev(
            unscaled_new_means, self.model_variance_sigma_squared
        )

    def step(self) -> None:
        """Each step combines two half-steps: first updating the model variance, then updating
        the model means. We re-evaluate the posterior means and elementwise variance using the
        current estimate before each half-step.

        The goal of proceeding in two steps is to decouple updates between the model means and
        variances.

        Posterior evaluation computes the means and elementwise variance of the current estimate,
        given the original data.

        L-update enforces low rank on the current posterior means, estimates the variance of that
        new model, and computes gamma (the means normalized by the model standard deviation).

        Likelihood monitoring computes the likelihood of the original input given the updated
        model. A warning is logged if this value decreases (no model update should reduce the
        likelihood).
        """
        # Evaluate posterior
        (posterior_means_Z, posterior_var_dZ) = self.do_pre_update()
        # Part A: Update variance
        self.model_variance_sigma_squared = cast(
            FloatArrayType,
            estimate_new_model_variance(
                posterior_means_Z, self.model_means_L, posterior_var_dZ, rowwise=True
            ),
        )
        # update gamma and record likelihood after this update
        self.half_step_likelihood = self.do_post_update()

        # Re-evaluate posterior
        (posterior_means_Z, posterior_var_dZ) = self.do_pre_update()

        ## Part B: Update means
        self.model_means_L = self.make_updated_means_estimate(posterior_means_Z)
        # Repeat gamma update and record new likelihood
        likelihood = self.do_post_update()

        ### Monitor likelihood:
        if likelihood < self.likelihood:
            logger.warning(
                f"Iteration {self.elapsed_iterations}: Likelihood decreased, from "
                + f"{self.likelihood} to {likelihood}"
            )
        if likelihood < self.half_step_likelihood:
            logger.warning(
                f"Iteration {self.elapsed_iterations - 1}.5: Likelihood decreased between "
                + "variance update and means update\n\t"
                + f"from {self.half_step_likelihood} to {likelihood}."
            )
        self.likelihood = likelihood

        self.loss = compute_loss(
            posterior_means_Z, self.model_means_L, LossType.FROBENIUS
        )

    def running_report(self) -> str:
        txt = (
            f"\t\tIteration {self.elapsed_iterations}: "
            + f"loss: {self.loss} Likelihoods {self.half_step_likelihood} -> {self.likelihood}"
        )
        return txt

    def report(self) -> KernelReturnType:
        text = (
            f"{self.elapsed_iterations} total iterations, final loss {self.loss} "
            + f"likelihood {self.likelihood}"
        )
        data = RowwiseVarianceGaussianModelKernelReturnType(
            self.model_means_L,
            self.model_variance_sigma_squared,
        )
        return KernelReturnType(text, data)
