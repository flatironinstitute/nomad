"""Defines most generally-visible types.

Classes:
    DecomposeInput: Nominal structured input for decompose loop. Not used.

"""

from typing import NamedTuple, Optional
import numpy as np
import numpy.typing as npt
from .enums import KernelStrategy, SVDStrategy, InitializationStrategy, DiagnosticLevel

FloatArrayType = npt.NDArray[np.float_]


class DiagnosticDataConfig(NamedTuple):
    """Data object for configuring per-iteration diagnostic data output."""

    diagnostic_level: DiagnosticLevel
    diagnostic_output_basedir: str
    use_exact_diagnostic_basepath: bool


# Currently unused--may be more trouble than it's worth
class DecomposeInput(NamedTuple):
    """Data object for input to the main decompose loop. Currently unused."""

    sparse_matrix_X: FloatArrayType
    target_rank: int
    kernel_strategy: KernelStrategy
    svd_strategy: Optional[SVDStrategy]
    initialization: Optional[InitializationStrategy]
    tolerance: Optional[float]
    manual_max_ierations: Optional[int]
    verbose: Optional[bool]
    diagnostic_config: Optional[DiagnosticDataConfig]
