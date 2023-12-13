"""Utility functions related to statistical computations.

Functions:
    pdf_to_cdf_ratio_psi: Computes PDF/CDF for a (0, 1) Gaussian.

"""
from typing import cast, Union
import numpy as np
from scipy.stats import norm as normal  # type: ignore

from fi_nomad.types import FloatArrayType


# NOTE: As defined, this will underflow if x > ~37, or overflow if x < -2e9 or so,
# generating a warning.
# Unclear if these values are actually realistic in practice, and whether we even
# care, since they're only epsilon away from 0 or 1 (respectively).
# We might wish to avoid the warning by replacing the result for known out-of-bounds inputs,
# although numpy *should* be doing the right thing by replacing with 0/1 anyway.
def pdf_to_cdf_ratio_psi(x: Union[float, FloatArrayType]) -> FloatArrayType:
    """Compute the ratio of the probability density function to the
    cumulative distribution function, with respect to a normal distribution with
    zero mean and unit variance.

    This function is abbreviated "psi" in Saul (2022).

    Args:
        x: The value (or array of values) to compute

    Returns:
        A numpy array representing this value.
    """
    return cast(FloatArrayType, np.exp(normal.logpdf(x) - normal.logcdf(x)))
