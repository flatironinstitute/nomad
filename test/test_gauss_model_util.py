import numpy as np
from unittest.mock import Mock, patch
from pytest import approx

from lzcompression.util.gauss_model_util import (
    get_posterior_means_Z,
    estimate_new_model_global_variance,
    get_stddev_normalized_matrix_gamma,
    get_elementwise_posterior_variance_dZbar,
    target_matrix_log_likelihood,
)


# The import happens in lzcompression.util.gauss_model_util, we need to
# patch what *that* imported, not what the base implementation is.
@patch("lzcompression.util.gauss_model_util.pdf_to_cdf_ratio_psi")
def test_get_posterior_means_Z(mock_psi: Mock) -> None:
    # Mocked pdf-to-cdf-ratio fn to return its input,
    # so in the SUT it will return -1 * stddev_norm (i.e. -3)
    # Thus we should set the source 0s to low_rank - sqrt(sigma_sq) * -(stddev_norm)
    # i.e. 10 - 4 * -3 --> 22.
    mock_psi.side_effect = lambda x: x
    sparse = np.eye(3)
    low_rank = np.ones(sparse.shape) * 10
    stddev_norm = np.ones(sparse.shape) * 3
    sigma_sq = 16
    res = get_posterior_means_Z(low_rank, sparse, stddev_norm, sigma_sq)
    expected = np.eye(3)
    expected[expected == 0] = np.sqrt(sigma_sq) * 3 + 10
    np.testing.assert_allclose(res, expected)


def test_estimate_new_model_global_variance() -> None:
    utility = np.ones((3, 3)) * 7
    lr = np.ones((3, 3)) * 4
    var = np.ones((3, 3)) * 2
    # Should be taking the mean of a matrix, each of whose elements are
    # (7-4)^2 + 2 = 9 + 2 = 11
    expected = 11.0
    result = estimate_new_model_global_variance(utility, lr, var)
    assert result == expected


def test_get_stddev_normalized_matrix_gamma() -> None:
    input = np.eye(3) * 16
    sigma_squared = 4
    expected_result = np.eye(3) * 8
    res = get_stddev_normalized_matrix_gamma(input, sigma_squared)
    np.testing.assert_allclose(expected_result, res)


def test_get_elementwise_posterior_variance_dZbar_uses_0_variance_for_known_values() -> (
    None
):
    sparse = np.eye(3)
    var = 1.0
    model = np.array(range(9)).reshape((3, 3)) * 0.1
    res = get_elementwise_posterior_variance_dZbar(sparse, var, model)
    np.testing.assert_allclose(res[sparse > 0], np.zeros((3, 3))[sparse > 0])
    a = res[sparse == 0]
    b = sparse[sparse == 0]
    c = model[sparse == 0]
    for i in range(len(a)):
        assert a[i] != b[i]
        assert a[i] != c[i]


def test_get_elementwise_posterior_variance_dZbar_computes_variances_where_needed() -> (
    None
):
    sparse = np.eye(3)
    var = 2.0
    stddev_normalized_model_matrix = np.array(range(9)).reshape((3, 3)) * 0.1
    # fmt: off
    expected_result = np.array([
        [0.72676046, 0.68430569, 0.6441387],
        [0.60622898, 0.57052543, 0.53696081],
        [0.50545572, 0.47592198, 0.4482657]
    ])
    # fmt: on
    result = get_elementwise_posterior_variance_dZbar(
        sparse, var, stddev_normalized_model_matrix
    )
    np.testing.assert_allclose(result[sparse == 0], expected_result[sparse == 0])
    np.testing.assert_allclose(result[sparse > 0], np.zeros((3, 3))[sparse > 0])


def test_target_matrix_log_likelihood() -> None:
    sparse = np.eye(3)
    low_rank = (
        np.ones((3, 3)) * 2
    )  # this is not actually low rank but that doesn't matter
    sigma_sq = 9
    # this will result in us taking logpdf(1, loc=2, scale=3) for the 1s in the
    # identity matrix. That's 3 values at ~ -2.073106 each.
    stddev_norm_lr = np.ones(
        (3, 3)
    )  # obviously not the actual std-dev-normalized values of low_rank
    # we just need something identifiable for sparse's zero-valued entries
    # normal.logcdf of -1 is ~ -1.841021645, and we'll have 6 of those
    expected = (3 * -2.073106) + (6 * -1.841021645)
    result = target_matrix_log_likelihood(sparse, low_rank, stddev_norm_lr, sigma_sq)
    assert approx(result) == expected
