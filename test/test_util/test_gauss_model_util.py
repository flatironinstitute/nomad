from typing import cast
import numpy as np
from unittest.mock import Mock, patch
from pytest import approx
import pytest
from fi_nomad.types import FloatArrayType

from fi_nomad.util.gauss_model_util import (
    broadcast_rowwise_variance,
    get_posterior_means_Z,
    estimate_new_model_variance,
    get_stddev_normalized_matrix_gamma,
    get_elementwise_posterior_variance_dZbar,
    scale_by_rowwise_stddev,
    target_matrix_log_likelihood,
)


PKG = "fi_nomad.util.gauss_model_util"


@pytest.fixture
def threeByFourMatrix() -> FloatArrayType:
    # fmt: off
    matrix = np.array([
        [1, 2, 0, 0],
        [0, 1, 2, 0],
        [3, 0, 3, 1]
    ])
    # fmt: on
    return matrix


def test_broadcast_rowwise_variance_with_scalar() -> None:
    var = 1.0
    target = np.eye(3)
    result = broadcast_rowwise_variance(var, target)
    assert result == var
    assert np.isscalar(result)
    assert isinstance(result, float)


def test_broadcast_rowwise_variance_with_vector(
    threeByFourMatrix: FloatArrayType,
) -> None:
    matrix = threeByFourMatrix
    var = np.var(matrix, axis=1)
    result = cast(FloatArrayType, broadcast_rowwise_variance(var, matrix))
    assert result.shape == matrix.shape
    for k in range(matrix.shape[1]):
        np.testing.assert_array_equal(result[:, k], var)


def test_broadcast_rowwise_variance_with_scalar_filter(
    threeByFourMatrix: FloatArrayType,
) -> None:
    matrix = threeByFourMatrix
    var = np.var(matrix, axis=1)
    filter = matrix != 0
    result = cast(
        FloatArrayType, broadcast_rowwise_variance(var, matrix, limiter=filter)
    )
    assert result.shape == matrix[filter].shape


# The import happens in fi_nomad.util.gauss_model_util, we need to
# patch what *that* imported, not what the base implementation is.
@patch(f"{PKG}.pdf_to_cdf_ratio_psi")
def test_get_posterior_means_Z_scalar_variance(mock_psi: Mock) -> None:
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


@patch(f"{PKG}.pdf_to_cdf_ratio_psi")
def test_get_posterior_means_Z_rowwise_variance(mock_psi: Mock) -> None:
    # Mocked pdf-to-cdf-ratio fn to return its input,
    # so in the SUT it will return -1 * stddev_norm (i.e. -3)
    # Thus we should set the source 0s to low_rank - sqrt(sigma_sq) * -(stddev_norm)
    # i.e. 10 - 4 * -3 --> 22.
    mock_psi.side_effect = lambda x: x
    sparse = np.eye(3)
    low_rank = np.ones(sparse.shape) * 10
    stddev_norm = np.ones(sparse.shape) * 3
    sigma_sq = np.array([4, 9, 16])
    res = get_posterior_means_Z(low_rank, sparse, stddev_norm, sigma_sq)
    expected = np.eye(3)
    computed_expected = np.repeat(np.sqrt(sigma_sq) * 3 + 10, sparse.shape[1]).reshape(
        sparse.shape
    )
    expected[expected == 0] = computed_expected[expected == 0]
    np.testing.assert_allclose(res, expected)


def test_estimate_new_model_variance_scalar() -> None:
    utility = np.ones((3, 3)) * 7
    lr = np.ones((3, 3)) * 4
    var = np.ones((3, 3)) * 2
    # Should be taking the mean of a matrix, each of whose elements are
    # (7-4)^2 + 2 = 9 + 2 = 11
    expected = 11.0
    result = estimate_new_model_variance(utility, lr, var)
    assert result == expected


def test_estimate_new_model_rowwise_variance() -> None:
    posterior_means = np.ones((3, 3)) * 7
    prior_means = np.array(range(9)).reshape((3, 3))
    var = np.ones((3, 3)) * 2
    expected = [
        np.mean(np.square(posterior_means[0] - prior_means[0]) + var[0]),
        np.mean(np.square(posterior_means[1] - prior_means[1]) + var[1]),
        np.mean(np.square(posterior_means[2] - prior_means[2]) + var[2]),
    ]
    result = estimate_new_model_variance(
        posterior_means, prior_means, var, rowwise=True
    )
    np.testing.assert_array_almost_equal(expected, result)


def test_get_stddev_normalized_matrix_gamma_scalar_variance() -> None:
    input = np.eye(3) * 16
    sigma_squared = 4.0
    expected_result = np.eye(3) * 8
    res = get_stddev_normalized_matrix_gamma(input, sigma_squared)
    np.testing.assert_allclose(expected_result, res)


def test_get_stddev_normalized_matrix_gamma_rowwise_variance(
    threeByFourMatrix: FloatArrayType,
) -> None:
    sparse = threeByFourMatrix
    sigma_squared = np.var(sparse, axis=1)
    # intentionally simplistic--otherwise we'd just be duplicating the implementation code
    expected_result = [
        sparse[0] / np.sqrt(np.var(sparse[0])),
        sparse[1] / np.sqrt(np.var(sparse[1])),
        sparse[2] / np.sqrt(np.var(sparse[2])),
    ]
    res = get_stddev_normalized_matrix_gamma(sparse, sigma_squared)
    np.testing.assert_allclose(expected_result, res)


def test_scale_by_rowwise_stddev() -> None:
    to_scale = np.eye(3)
    rowwise_var = np.array([1, 4, 9])
    expected_result = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    res = scale_by_rowwise_stddev(to_scale, rowwise_var)
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


def test_get_elementwise_posterior_variance_dZbar_handles_vector_variance(
    threeByFourMatrix: FloatArrayType,
) -> None:
    sparse = threeByFourMatrix
    var = np.var(sparse, axis=1)
    stddev_normalized_model_matrix = np.array(range(12)).reshape((3, 4)) * 0.1
    # fmt: off
    expected_result = np.array([
        [0.,      0.,       0.22142,    0.20839],
        [0.19612, 0.,       0.,         0.1636],
        [0.,      0.35639,  0.,         0.]
    ])
    # fmt: on
    result = get_elementwise_posterior_variance_dZbar(
        sparse, var, stddev_normalized_model_matrix
    )
    np.testing.assert_allclose(result, expected_result, atol=0.00001)


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


def test_target_matrix_log_likelihood_rowwise_variance() -> None:
    sparse = np.eye(3)
    centers = np.ones((3, 3)) * 2
    sigma_sq = np.array([1, 2, 3])
    # this will result in us taking logpdf(1, loc=2, scale=sqrt([1, 2, 3])) for
    # the 1s in the identity matrix. So those values contribute
    # -1.41894 + -1.51555 + -1.63491 = -4.56936
    stddev_norm_lr = np.ones(
        (3, 3)
    )  # obviously not the actual std-dev-normalized values of low_rank
    # we just need something identifiable for sparse's zero-valued entries
    # normal.logcdf of -1 is ~ -1.841021645, and we'll have 6 of those
    expected = (-4.56936) + (6 * -1.841021645)
    result = target_matrix_log_likelihood(sparse, centers, stddev_norm_lr, sigma_sq)
    assert approx(result) == expected
