"""Utility functions for computing reconstruction loss.

Functions:
    compute_loss: Computes scalar loss estimate between utility and target matrices.

"""

import numpy as np

from fi_nomad.types import FloatArrayType, LossType


# Included for historical reasons, but better to use the Frobenius norm instead.
def _squared_difference_loss(
    utility: FloatArrayType, candidate: FloatArrayType
) -> float:
    """Compute the square of the Frobenius norm of the difference between a target
    matrix and a utility matrix.

    Args:
        utility: A utility matrix ("Z" in algorithms from Saul (2022))
        candidate: The target matrix being evaluated

    Returns:
        Scalar loss estimate
    """
    loss = float(np.linalg.norm(np.subtract(utility, candidate)) ** 2)
    return loss


def _frobenius_norm_loss(utility: FloatArrayType, candidate: FloatArrayType) -> float:
    """Compute the Frobenius norm of the difference between a target
    matrix and a utility matrix.

    Args:
        utility: A utility matrix ("Z" in algorithms from Saul (2022))
        candidate: The target matrix being evaluated

    Returns:
        Scalar loss estimate
    """
    return float(np.linalg.norm(np.subtract(utility, candidate)))


def compute_loss(
    utility: FloatArrayType,
    candidate: FloatArrayType,
    losstype: LossType = LossType.FROBENIUS,
) -> float:
    """Compute a scalar estimate of a loss between the utility matrix (Z) and
    target matrix L.

    In the model-free algorithm, L is a direct low-rank approximation;
    Z's role is to rigorously enforce that X, the sparse nonnegative
    matrix being approximated, can be recovered from L by applying ReLU.

    In the Gaussian-model algorithm, L stores the prior means of the model
    while Z has the posterior means; lower loss between them represents
    improved fit to the data.

    Implemented losses are the Frobenius norm, and squared Frobenius norm.

    Args:
        utility: "Z" matrix
        candidate: "L" matrix
        losstype: Type of loss to use. Defaults to LossType.FROBENIUS.

    Raises:
        ValueError: If an unsupported loss type is requested.

    Returns:
        Scalar loss value representing how well L approximates Z (and,
        by proxy, X).
    """
    if losstype == LossType.FROBENIUS:
        return _frobenius_norm_loss(utility, candidate)
    if losstype == LossType.SQUARED_DIFFERENCE:
        return _squared_difference_loss(utility, candidate)
    raise ValueError(f"Unrecognized loss type {losstype} requested.")
