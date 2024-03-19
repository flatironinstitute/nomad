"""Defines an interface for data objects returned from decomposition kernels.

Classes:
    KernelReturnBase: Basic interface for kernel return values.
    BaseModelFreeKernelReturnType: Return from base naive kernel method.
    SingleVarianceGaussianModelKernelReturnType: Data with scalar variance.
    RowwiseVarianceGaussianModelKernelReturnType: Data with rowwise variance.
    Momentum3BlockModelFreeKernelReturnType: Return for momentum 3-block kernel method.
    KernelReturnType: Combination of data and summary string.

"""

from abc import ABC
from typing import Tuple, Union
from dataclasses import dataclass
from .types import FloatArrayType

SolutionType = Tuple[FloatArrayType, FloatArrayType]


@dataclass
class KernelReturnBase(ABC):
    """Base interface for returned kernel data. Enforces that every kernel
    must return consistent member(s) containing the reconstructed solution.
    """

    factors: SolutionType


@dataclass
class BaseModelFreeKernelReturnType(KernelReturnBase):
    """Base/naive method returns only the reconstruction. (Adds nothing to base.)"""


@dataclass
class SingleVarianceGaussianModelKernelReturnType(KernelReturnBase):
    """The simple Gaussian model returns a (scalar) variance in addition to
    the low-rank estimate.
    """

    variance: float


@dataclass
class RowwiseVarianceGaussianModelKernelReturnType(KernelReturnBase):
    """The rowwise-variance Gaussian model returns the estimated per-row
    variance of the model, in addition to the low-rank estimate (means).
    """

    variance: FloatArrayType


@dataclass
class Momentum3BlockModelFreeKernelReturnType(KernelReturnBase):
    """Momentum 3-block method returns only the reconstruction. (Adds nothing to base.)"""


KernelReturnDataType = Union[
    BaseModelFreeKernelReturnType,
    SingleVarianceGaussianModelKernelReturnType,
    RowwiseVarianceGaussianModelKernelReturnType,
    Momentum3BlockModelFreeKernelReturnType,
]


@dataclass
class KernelReturnType:
    """Enforces that kernels return a summary string and their actual data."""

    summary: str
    data: KernelReturnDataType
