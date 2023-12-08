from abc import ABC, abstractmethod
from typing import Union

from lzcompression.types import (
    FloatArrayType,
    KernelInputType,
    KernelReturnType,
    SVDStrategy,
)


class KernelBase(ABC):
    """Base class for matrix decomposition kernels.

    Every kernel is expected to implement three methods:
    - A step method for a single iteration of its algorithm
    - A "running report" that returns a string (possibly empty) for per-iteration output
    - A report method for returning a summary string and the final data set

    Additionally, the class as a whole has members representing the initialization
    values, as well as the loss and elapsed iteration count, and implemented
    methods to set the loss and increment the iteration counter.
    """

    def __init__(self, input: KernelInputType) -> None:
        self.loss: float = float("inf")
        self.elapsed_iterations: int = 0
        self.sparse_matrix_X: FloatArrayType = input.sparse_matrix_X
        self.low_rank_candidate_L: FloatArrayType = input.low_rank_candidate_L
        self.target_rank: int = input.target_rank
        self.svd_strategy: SVDStrategy = input.svd_strategy
        self.tolerance: Union[float, None] = input.tolerance

    def increment_elapsed(self) -> None:
        self.elapsed_iterations += 1

    # The following lines are unreachable except through shenanigans
    # (the class can't even be insantiated for test without serious hacks)
    # so exclude from report
    # LCOV_EXCL_START
    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def running_report(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def report(self) -> KernelReturnType:
        raise NotImplementedError

    # LCOV_EXCL_STOP
