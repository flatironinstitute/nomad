"""Utility functions for initializations.

Functions:
    initialize_low_rank_candidate: Apply strategy to determine initial low-rank candidate matrix.
    initialize_candidate: Wrapper for initializing low-rank candidate with error handling.

"""

from typing import Optional
import numpy as np

from fi_nomad.types import (
    FloatArrayType,
    InitializationStrategy,
    KernelStrategy,
)


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


def initialize_candidate(
    init_strat: InitializationStrategy,
    k_strat: KernelStrategy,
    sparse: FloatArrayType,
    guess: Optional[FloatArrayType],
) -> FloatArrayType:
    """Initialization function for standard kernel inputs.

    Args:
        init_strat: Methodology for picking the initial decomposition candidate.
        k_strat: The type of kernel used (as a "COPY" strategy is enforced for the naive
            kernel)
        sparse: The input sparse matrix
        guess: An optional initial low-rank candidate, used for checkpointing when the
        "KNOWN_MATRIX" strategy is being followed

    Raises:
        ValueError: Raised if the shape of the guess parameter does not match the
            shape of the sparse input matrix

    Returns:
        An object of standard kernel inputs
    """
    # Note: enforcing "COPY" strategy for base_model_free kernel may not be desirable
    _initialization_strategy = (
        InitializationStrategy.COPY
        if k_strat == KernelStrategy.BASE_MODEL_FREE
        else init_strat
    )
    input_matrix = (
        guess
        if (
            _initialization_strategy == InitializationStrategy.KNOWN_MATRIX
            and guess is not None
        )
        else sparse
    )
    if (
        guess is not None
        and _initialization_strategy == InitializationStrategy.KNOWN_MATRIX
    ):
        if guess.shape != sparse.shape:
            raise ValueError(
                "A manual checkpoint matrix was submitted, but its shape"
                + f"{guess.shape} does not match the sparse matrix's {sparse.shape}."
            )
    return initialize_low_rank_candidate(input_matrix, _initialization_strategy)
