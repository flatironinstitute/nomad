from enum import Enum


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
