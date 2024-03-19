"""Defines enumerations for fixed-choice configuration values.

Classes:
    InitializationStrategy: Algorithm choice for setting the initial low-rank candidate.
    SVDStrategy: Algorithm choice for executing SVD within kernels.
    LossType: Types of loss.
    KernelStrategy: Choices of decomposition kernel.

"""

from enum import Enum


class InitializationStrategy(Enum):
    """Strategy to use for choosing the initial value of the low-rank candidate."""

    COPY = 1
    BROADCAST_MEAN = 2
    ROWWISE_MEAN = 3
    KNOWN_MATRIX = 4


class SVDStrategy(Enum):
    """Algorithms to use for SVD within kernels."""

    FULL = 1
    RANDOM_TRUNCATED = 2
    EXACT_TRUNCATED = 3


class LossType(Enum):
    """Known loss types for loss evaluation."""

    FROBENIUS = 1
    SQUARED_DIFFERENCE = 2


class KernelStrategy(Enum):
    """All currently-implemented decomposition kernels which can be instantiated
    in the main runner loop.
    """

    TEST = 1
    BASE_MODEL_FREE = 2
    GAUSSIAN_MODEL_SINGLE_VARIANCE = 3
    GAUSSIAN_MODEL_ROWWISE_VARIANCE = 4
    MOMENTUM_3_BLOCK_MODEL_FREE = 5


class DiagnosticLevel(Enum):
    """Defined verbosity or detail levels for per-iteration kernel reporting."""

    NONE = 0
    MINIMAL = 10
    MODERATE = 50
    HIGH = 100
    EXTREME = 200
