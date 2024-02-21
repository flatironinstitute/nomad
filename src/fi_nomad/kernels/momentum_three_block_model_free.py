"""Defines the momentum 3-block model-free kernel.

Classes:
    Momentum3BlockModelFreeKernel: XXXXX.

"""

from fi_nomad.kernels.kernel_base import KernelBase

from fi_nomad.util.model_free_util import construct_utility, apply_momentum

from fi_nomad.util.momentum_three_block_model_free_util import (
    update_W,
    update_H,
)

from fi_nomad.types import (
    KernelInputType,
    Momentum3BlockModelFreeKernelReturnType,
    Momentum3BlockAdditionalParameters,
    KernelReturnType,
    LossType,
)
from fi_nomad.util import compute_loss, two_part_factor_known_rank


class Momentum3BlockModelFreeKernel(KernelBase):
    """Momentum 3-block model-free algorithm as described in Seraghiti et. al. (2023)"""

    def __init__(
        self, indata: KernelInputType, custom_params: Momentum3BlockAdditionalParameters
    ) -> None:
        super().__init__(indata)

        if (
            custom_params.candidate_factor_W0 is not None
            and custom_params.candidate_factor_H0 is not None
        ):
            self.candidate_factor_W = custom_params.candidate_factor_W0
            self.candidate_factor_H = custom_params.candidate_factor_H0
        else:
            # if W0, H0 are not given, factorize low_rank_candidate_L
            (W0, H0) = two_part_factor_known_rank(
                self.low_rank_candidate_L,
                self.target_rank,
            )
            self.candidate_factor_W = W0
            self.candidate_factor_H = H0

        self.momentum_beta = custom_params.momentum_beta
        self.utility_matrix_Z = indata.sparse_matrix_X.copy()
        self.previous_low_rank_candidate_L = None

    def step(self) -> None:
        """Single step of the momentum 3-block model-free low-rank approximation estimator.

        XXXXXXXXXXXXXXX

        Note that loss is only computed if an initial tolerance was set; this is an attempt to avoid
        unnecessary computation, but may be removed in a future version.

        Returns:
            None
        """
        if self.elapsed_iterations > 0:
            # Seraghiti et. al. (2023) apply momentum at the end of each step
            # but the last one, as this would alter the rank.  We omit an
            # additional parameter "max iterations", by moving the momentum step
            # on L to the beginning.
            self.low_rank_candidate_L = apply_momentum(
                self.low_rank_candidate_L,
                self.previous_low_rank_candidate_L,
                self.momentum_beta,
            )

        # BLOCK 1: construct utility Z and apply momentum term
        utility_matrix_Z = construct_utility(
            self.low_rank_candidate_L, self.sparse_matrix_X
        )
        self.utility_matrix_Z = apply_momentum(
            utility_matrix_Z, self.utility_matrix_Z, self.momentum_beta
        )

        # BLOCK 2: update candidate factor W
        self.candidate_factor_W = update_W(
            self.candidate_factor_H, self.utility_matrix_Z
        )

        # BLOCK 3: update candidate factor H
        self.candidate_factor_H = update_H(
            self.candidate_factor_W, self.utility_matrix_Z
        )

        self.previous_low_rank_candidate_L = self.low_rank_candidate_L.copy()
        self.low_rank_candidate_L = self.candidate_factor_W @ self.candidate_factor_H

        if self.tolerance is not None:
            self.loss = compute_loss(
                self.utility_matrix_Z, self.low_rank_candidate_L, LossType.FROBENIUS
            )

    def running_report(self) -> str:
        """Reports the current iteration number and loss. Only operative if a
        tolerance was set.

        Returns:
            Description of current iteration and loss.
        """
        txt = (
            ""
            if self.tolerance is None
            else f"iteration: {self.elapsed_iterations} loss: {self.loss}"
        )
        return txt

    def report(self) -> KernelReturnType:
        """Returns final low-rank approximation and descriptive string.
        The description indicates the total number of iterations completed,
        and the final loss (if a tolerance was set).

        Returns:
            Object containing results and summary
        """
        floss = str(self.loss) if self.loss != float("inf") else "Not Tracked"
        text = f"{self.elapsed_iterations} total, final loss {floss}"
        data = Momentum3BlockModelFreeKernelReturnType(
            (self.candidate_factor_W, self.candidate_factor_H)
        )
        return KernelReturnType(text, data)
