import numpy as np
import numpy.typing as npt
from enum import Enum

FloatArrayType = npt.NDArray[np.float_]


class InitializationStrategy(Enum):
    COPY = 1
    BROADCAST_MEAN = 2


class SVDStrategy(Enum):
    FULL = 1
    RANDOM_TRUNCATED = 2
    EXACT_TRUNCATED = 3


class LossType(Enum):
    FROBENIUS = 1
    SQUARED_DIFFERENCE = 2
