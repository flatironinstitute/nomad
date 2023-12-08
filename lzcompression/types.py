from typing import Tuple, NamedTuple, Union
import numpy as np
import numpy.typing as npt
from enum import Enum
from dataclasses import dataclass

FloatArrayType = npt.NDArray[np.float_]


class InitializationStrategy(Enum):
    COPY = 1
    BROADCAST_MEAN = 2
    ROWWISE_MEAN = 3
    KNOWN_MATRIX = 4


class SVDStrategy(Enum):
    FULL = 1
    RANDOM_TRUNCATED = 2
    EXACT_TRUNCATED = 3


class LossType(Enum):
    FROBENIUS = 1
    SQUARED_DIFFERENCE = 2


class KernelStrategy(Enum):
    TEST = 1
    BASE_MODEL_FREE = 2
    GAUSSIAN_MODEL_SINGLE_VARIANCE = 3
    GAUSSIAN_MODEL_ROWWISE_VARIANCE = 4


# This will facilitate a future conversion from returning the full
# low-rank matrix to returning a factored version.
SolutionType = FloatArrayType


class BaseModelFreeKernelReturnType(NamedTuple):
    reconstruction: SolutionType


class SingleVarianceGaussianModelKernelReturnType(NamedTuple):
    reconstruction: SolutionType
    variance: float


class RowwiseVarianceGaussianModelKernelReturnType(NamedTuple):
    reconstruction: SolutionType
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


@dataclass
class KernelInputType:
    sparse_matrix_X: FloatArrayType
    low_rank_candidate_L: FloatArrayType
    target_rank: int
    svd_strategy: SVDStrategy
    tolerance: Union[float, None]


# Currently unused--may be more trouble than it's worth
class DecomposeInput(NamedTuple):
    sparse_matrix_X: FloatArrayType
    target_rank: int
    kernel_strategy: KernelStrategy
    svd_strategy: Union[SVDStrategy, None]
    initialization: Union[InitializationStrategy, None]
    tolerance: Union[float, None]
    manual_max_ierations: Union[int, None]
    verbose: Union[bool, None]
