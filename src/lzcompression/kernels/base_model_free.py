from lzcompression.kernels.kernel_base import KernelBase
from lzcompression.util.base_model_free_util import construct_utility

from lzcompression.types import (
    BaseModelFreeKernelReturnType,
    KernelInputType,
    KernelReturnType,
    LossType,
)
from lzcompression.util.util import (
    find_low_rank,
    compute_loss,
)


class BaseModelFree(KernelBase):
    """Most basic implementation of estimating a low-rank approximation of a sparse nonnegative matrix.

    This is a more basic implementation than the one described in Saul (2022). The implementation
    in this method does not build an underlying model, but instead just alternates between
    constructing a utility matrix Z that enforces the non-zero values of the sparse matrix, and
    using SVD on Z to create a candidate low-rank matrix L.
    """

    def __init__(self, input: KernelInputType) -> None:
        super().__init__(input)

    def step(self) -> None:
        """Single step of the basic model-free low-rank approximation estimator.

        Each iteration carries out one construction of the utility matrix (Z) and
        one update of the low-rank candidate (L).

        Note that loss is only computed if an initial tolerance was set; this is an attempt to avoid
        unnecessary computation, but may be removed in a future version.

        Returns:
            None
        """
        ## One iteration comprises one utility-construction step, then one low-rank estimation step
        utility_matrix_Z = construct_utility(
            self.low_rank_candidate_L, self.sparse_matrix_X
        )
        self.low_rank_candidate_L = find_low_rank(
            utility_matrix_Z,
            self.target_rank,
            self.low_rank_candidate_L,
            self.svd_strategy,
        )

        if self.tolerance is not None:
            self.loss = compute_loss(
                utility_matrix_Z, self.low_rank_candidate_L, LossType.FROBENIUS
            )

    def running_report(self) -> str:
        txt = (
            ""
            if self.tolerance is None
            else f"iteration: {self.elapsed_iterations} loss: {self.loss}"
        )
        return txt

    def report(self) -> KernelReturnType:
        text = f"{self.elapsed_iterations} total, final loss {self.loss if self.loss != float('inf') else 'Not Tracked'}"
        data = BaseModelFreeKernelReturnType(self.low_rank_candidate_L)
        return KernelReturnType(text, data)


# TODO:
# Consider secondary evaluation criteria, e.g. promoting bimodal/higher-variance ditsributions of negative values in answer
