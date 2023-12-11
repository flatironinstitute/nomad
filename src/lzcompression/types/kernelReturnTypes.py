from abc import ABC
from typing import Union
from dataclasses import dataclass
from .types import FloatArrayType

# This will facilitate a future conversion from returning the full
# low-rank matrix to returning a factored version.
SolutionType = FloatArrayType


@dataclass
class KernelReturnBase(ABC):
    reconstruction: SolutionType


@dataclass
class BaseModelFreeKernelReturnType(KernelReturnBase):
    pass


@dataclass
class SingleVarianceGaussianModelKernelReturnType(KernelReturnBase):
    variance: float


@dataclass
class RowwiseVarianceGaussianModelKernelReturnType(KernelReturnBase):
    variance: FloatArrayType


KernelReturnDataType = Union[
    BaseModelFreeKernelReturnType,
    SingleVarianceGaussianModelKernelReturnType,
    RowwiseVarianceGaussianModelKernelReturnType,
]


@dataclass
class KernelReturnType:
    summary: str
    data: KernelReturnDataType
