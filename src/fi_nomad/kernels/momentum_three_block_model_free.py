"""Defines the momentum 3-block model-free kernel.

Classes:
    Momentum3BlockModelFreeKernel: Momentum 3-block model-free algorithm as described 
        in Seraghiti et. al. (2023).

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
    """Momentum 3-block model-free algorithm as described in Seraghiti et. al. (2023)

    This is an extension of the base model-free algorithm described in Saul (2022).
    It makes use of the parametrization :math:`L=WH` to update `W` and `H` separately,
    thus avoiding computation of the rank-r tSVD, which reduces computational
    complexity from :math:`O(mnr^2)` to :math:`O(mnr)`. Further, it extrapolates `Z` and `L`
    using momentum terms with fixed momentum parameter `momentum_beta` to accelerate
    convergence.

    Note: one can instantiate the kernel by passing both `candidate_factor_W0` and
        `candidate_factor_H0` (as described in Seraghiti et. al.). In case you
        don't supply W0 and H0, or just one of them, `low_rank_candidate_L` is
        factored into W0, H0 which are in turn used to instantiate the kernel. By
        doing this, we can make use of intialization methods that return the low
        rank candidate directly.
    """

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
            (candidate_factor_W0, candidate_factor_H0) = two_part_factor_known_rank(
                self.low_rank_candidate_L,
                self.target_rank,
            )
            self.candidate_factor_W = candidate_factor_W0
            self.candidate_factor_H = candidate_factor_H0

        self.momentum_beta = custom_params.momentum_beta
        self.utility_matrix_Z = indata.sparse_matrix_X
        self.previous_low_rank_candidate_L = self.low_rank_candidate_L

    def step(self) -> None:
        """Single step of the momentum 3-block model-free low-rank approximator.

        Each step carries out one construction of the utility matrix (Z) and extrapolates
        it with a momentum term (Block 1). It then updates candidate factor W (Block 2)
        and H (Block 3). Finally the low-rank approximation (L) is calculated.
        Extrapolation of L is done at the very beginning, starting with the second
        iteration (instead of at the end of each step) to ensure L has rank r.

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
