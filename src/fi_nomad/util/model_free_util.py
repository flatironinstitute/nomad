"""Utility functions for base model-free kernels. Most likely will not be expanded, even as other
naive kernels are added.

Functions:
    construct_utility: Construct candidate by enforcing base matrix constraints on an SVD result.
    apply_momentum: Applies momentum term on a matrix using the difference from the previous to the current solution.
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


def apply_momentum(
    X: FloatArrayType, previous_X: FloatArrayType, beta: float
) -> FloatArrayType:
    """Applies momentum on a matrix X

    Given the matrix from the current iteration and the one from
    the previous iteration, it calculates the difference and extrapolates
    the update accordingly. Hyperparameter beta controls the size of the
    extrapolation:

    X = X + beta(X - previous_X)

    Args:
        X: Matrix at the current iteration
        previous_X: Matrix at the previous iteration
        beta: Momentum parameter. Scalar between 0 and 1 that controls the
            extrapolation size.

    Returns:
        Updated matrix X after applying the momentum term.

    Example:
        # Setting beta to 1 will double the step taken from the last iteration to
        # the current.
        X = apply_momentum(X, X_previous, beta=1.0)

        # Setting beta to 0 will disable the momentum step and just return X
        post_momentum_X = apply_momentum(X, X_previous, beta=1.0)
        assert np.array_equal(X, post_momentum_X)
    """
    return X + beta * (X - previous_X)
