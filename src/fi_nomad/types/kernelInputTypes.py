"""Defines types for objects passed to kernels in instantiation. Kernel-specific parameter
sets will also be included here.

Classes:
    KernelInputType: Standard data object for initializing kernels.

"""
from typing import NamedTuple, Union
from .types import FloatArrayType
from .enums import SVDStrategy


KernelSpecificParameters = Union[float, int]


class KernelInputType(NamedTuple):
    """Standard data object for initializing kernels."""

    sparse_matrix_X: FloatArrayType
    low_rank_candidate_L: FloatArrayType
    target_rank: int
    svd_strategy: SVDStrategy
    tolerance: Union[float, None]
