from typing import Tuple, NamedTuple, Union
import numpy as np
import numpy.typing as npt
from enum import Enum
from dataclasses import dataclass


class KernelStrategy(Enum):
    TEST = 1
    BASE_MODEL_FREE = 2
    GAUSSIAN_MODEL_SINGLE_VARIANCE = 3
    GAUSSIAN_MODEL_ROWWISE_VARIANCE = 4


class InitializationStrategy(Enum):
    COPY = 1
    BROADCAST_MEAN = 2
    KNOWN_MATRIX = 3


class SVDStrategy(Enum):
    FULL = 1
    RANDOM_TRUNCATED = 2
    EXACT_TRUNCATED = 3


class LossType(Enum):
    FROBENIUS = 1
    SQUARED_DIFFERENCE = 2


FloatArrayType = npt.NDArray[np.float_]

KernelReturnDataType = Union[
    FloatArrayType, Tuple[FloatArrayType, float], Tuple[FloatArrayType, FloatArrayType]
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


class DecomposeInput(NamedTuple):
    sparse_matrix_X: FloatArrayType
    target_rank: int
    kernel_strategy: KernelStrategy
    svd_strategy: Union[SVDStrategy, None]
    initialization: Union[InitializationStrategy, None]
    tolerance: Union[float, None]
    manual_max_ierations: Union[int, None]
    verbose: Union[bool, None]
