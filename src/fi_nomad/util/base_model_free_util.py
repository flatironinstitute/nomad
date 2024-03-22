"""Utility functions for base model-free kernels. Most likely will not be expanded, even as other
naive kernels are added.

Functions:
    construct_utility: Construct candidate by enforcing base matrix constraints on an SVD result.

"""

import numpy as np
from fi_nomad.types import FloatArrayType


def construct_utility(
    low_rank_matrix: FloatArrayType, base_matrix: FloatArrayType
) -> FloatArrayType:
    """Construct a utility matrix Z which enforces the invariants of the original
    sparse nonnegative matrix.

    Specifically, it creates Z from a 0-matrix by:
      - Copying the positive elements of the original sparse matrix X into the
        corresponding elements of Z
      - Copying any negative elements of the current low-rank approximation matrix
        L into the corresponding elements of Z, provided those elements of Z
        were not set in the first step
      - Any remaining elements remain 0

    i.e. for each i, j: Z_ij = X is X_ij > 0, else min(0, L_ij).

    Args:
        low_rank_matrix: The current low-rank approximation of the base matrix
        base_matrix: the sparse nonnegative matrix whose low-rank approximation
            is being sought

    Returns:
        A utility matrix whose only positive values are the positive values in
        the base_matrix
    """
    conditions = [base_matrix > 0, low_rank_matrix < 0]
    choices = [base_matrix, low_rank_matrix]
    utility_matrix = np.select(conditions, choices, 0)
    return utility_matrix
