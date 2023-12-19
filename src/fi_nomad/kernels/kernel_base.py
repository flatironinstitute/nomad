"""Abstract base class defining the interface which all solution kernels
should follow.

Classes:
    KernelBase: Abstract base class for kernel classes.

"""
from abc import ABC, abstractmethod
from typing import Optional, Union
from pathlib import Path
import logging

from fi_nomad.types import (
    FloatArrayType,
    KernelInputType,
    KernelReturnType,
    SVDStrategy,
    DiagnosticLevel,
)

logger = logging.getLogger(__name__)


class KernelBase(ABC):
    """Base class for matrix decomposition kernels.

    Every kernel is expected to implement three methods:
    - A step method for a single iteration of its algorithm
    - A "running report" that returns a string (possibly empty) for per-iteration output
    - A per-iteration diagnostic data output method
    - A report method for returning a summary string and the final data set

    Additionally, the class as a whole has members representing the initialization
    values, as well as the loss and elapsed iteration count, and implemented
    methods to set the loss and increment the iteration counter.
    """

    def __init__(self, indata: KernelInputType) -> None:
        """Common data for all kernels, as initialized by decompose wrapper.

        Args:
            indata: Data from wrapper factory
        """
        self.loss: float = float("inf")
        self.elapsed_iterations: int = 0
        self.sparse_matrix_X: FloatArrayType = indata.sparse_matrix_X
        self.low_rank_candidate_L: FloatArrayType = indata.low_rank_candidate_L
        self.target_rank: int = indata.target_rank
        self.svd_strategy: SVDStrategy = indata.svd_strategy
        self.tolerance: Union[float, None] = indata.tolerance
        self.diagnostic_level: DiagnosticLevel = DiagnosticLevel.NONE
        self.out_dir: Optional[Path] = None

    def increment_elapsed(self) -> None:
        """Function to increment elapsed iterations."""
        self.elapsed_iterations += 1

    # The following lines are unreachable except through shenanigans
    # (the class can't even be insantiated for test without serious hacks)
    # so exclude from report
    @abstractmethod
    def step(self) -> None:  # pragma: no cover
        """Base method to execute one step of the kernel-defined algorithm.

        Raises:
            NotImplementedError: Method is abstract.
        """
        raise NotImplementedError

    @abstractmethod
    def running_report(self) -> str:  # pragma: no cover
        """Base method to return a string report on a per-step basis.

        Raises:
            NotImplementedError: Method is abstract.

        Returns:
            Summary for the just-completed algorithm iteration step.
        """
        raise NotImplementedError

    def per_iteration_diagnostic(self) -> None:
        """Base method for kernel-specific per-iteration diagnostic output.

        Implementation is not required as many kernels may not make use of
        this functionality.

        Note: If caller does not configure diagnostics or requests a
        diagnostic level of NONE, this method will be replaced in the
        factory by a no-op. If execution actually enters the implementation,
        then self.diagnostic_level should not be NONE and self.out_dir should
        be set to an appropriate output directory (unless the values have
        been changed mid-execution).
        """
        # We would prefer not to take the extra dependencies, but it's
        # probably a good thing to warn users if they're doing something
        # that results in a no-op
        if self.diagnostic_level != DiagnosticLevel.NONE:
            logger.warning(
                "Per-iteration diagnostic info requested to "
                + f"{str(self.out_dir)} but kernel does not support it."
            )

    @abstractmethod
    def report(self) -> KernelReturnType:  # pragma: no cover
        """Base method to return decomposition results and a summary.

        Raises:
            NotImplementedError: Method is abstract.

        Returns:
            Object containing decomposition result and summary description.
        """
        raise NotImplementedError
