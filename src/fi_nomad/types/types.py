"""Defines most generally-visible types."""
from typing import NamedTuple, Union
import numpy as np
import numpy.typing as npt
from .enums import KernelStrategy, SVDStrategy, InitializationStrategy

FloatArrayType = npt.NDArray[np.float_]


# Currently unused--may be more trouble than it's worth
class DecomposeInput(NamedTuple):
    """Data object for input to the main decompose loop. Currently unused."""

    sparse_matrix_X: FloatArrayType
    target_rank: int
    kernel_strategy: KernelStrategy
    svd_strategy: Union[SVDStrategy, None]
    initialization: Union[InitializationStrategy, None]
    tolerance: Union[float, None]
    manual_max_ierations: Union[int, None]
    verbose: Union[bool, None]
