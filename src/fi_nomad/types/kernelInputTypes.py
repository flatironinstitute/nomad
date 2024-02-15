"""Defines types for objects passed to kernels in instantiation. Kernel-specific parameter
sets will also be included here.

Classes:
    KernelInputType: Standard data object for initializing kernels.
    Momentum3BlockAdditionalParameters: Additional parameters used by momentum 3-block
        model-free kernel.

"""

from typing import NamedTuple, Union, Optional
from .types import FloatArrayType
from .enums import SVDStrategy


class KernelInputType(NamedTuple):
    """Standard data object for initializing kernels."""

    sparse_matrix_X: FloatArrayType
    low_rank_candidate_L: FloatArrayType
    target_rank: int
    svd_strategy: SVDStrategy
    tolerance: Union[float, None]


class Momentum3BlockAdditionalParameters(NamedTuple):
    """Additional parameters for momentum 3-block model-free kernel.
    W0 and H0 are candidate low rank factors (opposed to initialization using
    low-rank candidate matrix L). beta is the momentum hyperparameter."""

    momentum_beta: float
    candidate_factor_W0: Optional[FloatArrayType] = None
    candidate_factor_H0: Optional[FloatArrayType] = None


KernelSpecificParameters = Union[float, int, Momentum3BlockAdditionalParameters]
