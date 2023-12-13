"""Utility functions for initializations.

Functions:
    initialize_low_rank_candidate: Apply strategy to determine initial low-rank candidate matrix.

"""
import numpy as np

from fi_nomad.types import FloatArrayType, InitializationStrategy


def initialize_low_rank_candidate(
    input_matrix: FloatArrayType, method: InitializationStrategy
) -> FloatArrayType:
    """Given a sparse matrix, create a starting-point guess for the low-rank representation.

    Currently supported strategies are:
     - BROADCAST_MEAN: fill the low-rank candidate with the mean of the sparse matrix's values
     - COPY: simply return a copy of the input sparse matrix

    Args:
        input_matrix: The sparse nonnegative matrix input ("X"). With the KNOWN_MATRIX
            strategy, this field is instead the desired initialization value (e.g. from checkpoint)
        method: The strategy to employ to generate the starting point

    Raises:
        ValueError: On request for an unsupported initialization strategy.

    Returns:
        Initial estimate for a low-rank representation.
    """
    low_rank_candidate = None
    if method == InitializationStrategy.COPY:
        low_rank_candidate = np.copy(input_matrix)
    elif method == InitializationStrategy.BROADCAST_MEAN:
        low_rank_candidate = np.full(input_matrix.shape, np.mean(input_matrix))
    elif method == InitializationStrategy.ROWWISE_MEAN:
        shape = input_matrix.shape
        row_means = np.mean(input_matrix, axis=1)
        low_rank_candidate = np.repeat(row_means, shape[1]).reshape(shape)
    elif method == InitializationStrategy.KNOWN_MATRIX:
        low_rank_candidate = np.copy(input_matrix)
    else:
        raise ValueError("Unsupported initialization strategy.")
    return low_rank_candidate
