"""Utility functions for Momentum 3-Block Model-Free Kernel from Seraghiti et. al. 2023

Functions:
    _compute_least_squares_solution: Compute the least squares solution to the linear system Ax = B.
    update_W: Update candidate factor W of the low-rank solution Theta =  W @ H.
    update_H: Update candidate factor H of the low-rank solution Theta =  W @ H.

"""

import numpy as np
from fi_nomad.types import FloatArrayType


def _compute_least_squares_solution(
    A: FloatArrayType, B: FloatArrayType
) -> FloatArrayType:
    """Matrix least squares solver

    Solves argmin_{X} || B - A @ X ||^{2}_{F}

    Args:
        A: Coefficient matrix A of shape (m, n)
        B: Matrix of shape (m, k)

    Returns:
        Least squares solution X of shape (n, k)
    """
    return np.linalg.lstsq(A, B, rcond=None)[0]


def update_W(H: FloatArrayType, Z: FloatArrayType) -> FloatArrayType:
    """Update candidate factor W

    Solves argmin_{W} || Z - W @ H ||^{2}_{F}

    Constitutes the second block of the momentum 3-block algorithm, that updates
    the candidate factor matrix W.

    Args:
        H: Candidate factor for the low-rank approximation (r, p)
        Z: Utility matrix Z of shape (n, p)

    Returns:
        Updated candidate factor matrix W of shape (n, r)
    """
    return _compute_least_squares_solution(H @ H.T, H @ Z.T).T


def update_H(W: FloatArrayType, Z: FloatArrayType) -> FloatArrayType:
    """Update candidate factor H

    Solves argmin_{H} || Z - W @ H ||^{2}_{F}

    Constitutes the third block in the momentum 3-block algorithm, that updates
    the candidate factor matrix H

    Args:
        W: Candidate factor for the low-rank approximation (n, r)
        Z: Utility matrix Z of shape (n, p)

    Returns:
        Updated candidate factor matrix H of shape (r, p)
    """
    return _compute_least_squares_solution(W.T @ W, W.T @ Z)
