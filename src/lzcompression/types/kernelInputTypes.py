from typing import NamedTuple, Union
from .types import FloatArrayType
from .enums import SVDStrategy


KernelSpecificParameters = Union[float, int]


class KernelInputType(NamedTuple):
    sparse_matrix_X: FloatArrayType
    low_rank_candidate_L: FloatArrayType
    target_rank: int
    svd_strategy: SVDStrategy
    tolerance: Union[float, None]
