import numpy as np
import sys
from unittest.mock import Mock, patch
from pytest import raises, approx
from typing import cast

from lzcompression.types import InitializationStrategy, SVDStrategy, LossType
from lzcompression.util import (
    initialize_low_rank_candidate,
    _squared_difference_loss,
    _frobenius_norm_loss,
    _find_low_rank_full,
    _find_low_rank_random_truncated,
    _find_low_rank_exact_truncated,
    find_low_rank,
    pdf_to_cdf_ratio_psi,
    get_stddev_normalized_matrix_gamma,
    compute_loss,
    get_elementwise_posterior_variance_dZbar,
    low_rank_matrix_log_likelihood,
)


def test_initialize_returns_copy() -> None:
    # fmt: off
    base_matrix = np.array([
        [ 1,  5,  0],
        [ 0,  0,  0]
    ])
    # fmt: on
    res = initialize_low_rank_candidate(base_matrix, InitializationStrategy.COPY)
    np.testing.assert_allclose(base_matrix, res)


def test_initialize_returns_mean() -> None:
    # fmt: off
    base_matrix = np.array([
        [ 1,  5,  0],
        [ 0,  0,  0]
    ])
    # fmt: on
    res = initialize_low_rank_candidate(
        base_matrix, InitializationStrategy.BROADCAST_MEAN
    )
    expected = np.ones((2, 3))
    np.testing.assert_allclose(res, expected)


def test_initialize_throws_on_unknown_strategy() -> None:
    base_matrix = np.ones((2, 2))
    with raises(ValueError, match="Unsupported"):
        _ = initialize_low_rank_candidate(base_matrix, cast(InitializationStrategy, -5))


#### Stats


def test_pdf_to_cdf_ratio_psi() -> None:
    crossover = -0.30263083
    # float: example -0.30263083,
    # pdf: 0.3810856, cdf: 0.381056
    res1 = pdf_to_cdf_ratio_psi(crossover)
    assert approx(res1) == 1.0
    # Try with arrays as well
    # 0: cdf = 0.5, pdf = 0.3989, ratio 0.79788456...
    expected_scalar = 0.79788456
    expected = np.array([[1.0, expected_scalar], [expected_scalar, 1.0]])
    matrix = np.eye(2) * crossover
    res2 = pdf_to_cdf_ratio_psi(matrix)
    np.testing.assert_allclose(res2, expected)


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


# This isn't a great test by itself
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


#### Losses


def test_squared_difference_loss() -> None:
    # fmt: off
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([[1, 2, 0],
                  [4, 5, 0]])
    # fmt: on
    result = _squared_difference_loss(a, b)
    assert approx(result, 0.0001) == 45


def test_frobenius_norm_loss() -> None:
    # Frobenius norm = sum the squares of the elements, then take sqrt.
    # Since we're doing subtraction on the two input matrices, we'll take
    # two cases: case 1: 3x3 3s - 3x3 2s should give 3x3 1s = 9, sqrt = 3
    # case 2: identity x 8 - identity * 4 --> identity * 4
    # square the four nonzero elements, get 16s; sum them for 64;
    # sqrt = 8.0.
    a = np.ones((3, 3)) * 3
    b = np.ones((3, 3)) * 2
    res = _frobenius_norm_loss(a, b)
    assert approx(res) == 3.0

    a2 = np.eye(4) * 8
    b2 = np.eye(4) * 4
    res2 = _frobenius_norm_loss(a2, b2)
    assert approx(res2) == 8.0


@patch("lzcompression.util._squared_difference_loss")
@patch("lzcompression.util._frobenius_norm_loss")
def test_compute_loss_dispatches_correctly(mock_frob: Mock, mock_sqdiff: Mock) -> None:
    mock_frob_return = 5
    mock_sqdiff_return = 10
    mock_frob.return_value = mock_frob_return
    mock_sqdiff.return_value = mock_sqdiff_return
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    res1 = compute_loss(a, b, LossType.FROBENIUS)
    assert res1 == mock_frob_return
    assert mock_frob.call_count == 1
    assert mock_sqdiff.call_count == 0
    res2 = compute_loss(a, b, LossType.SQUARED_DIFFERENCE)
    assert res2 == mock_sqdiff_return
    assert mock_frob.call_count == 1
    assert mock_sqdiff.call_count == 1


def test_compute_loss_throws_on_bad_loss_type() -> None:
    a = np.ones((2, 2))
    b = a
    with raises(ValueError, match="Unrecognized"):
        _ = compute_loss(a, b, cast(LossType, -5))


def test_low_rank_matrix_log_likelihood() -> None:
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
    result = low_rank_matrix_log_likelihood(sparse, low_rank, stddev_norm_lr, sigma_sq)
    assert approx(result) == expected


#### SVD


def test_find_low_rank_full_uses_preallocated_memory() -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    allocated_memory = np.zeros(arbitrary_matrix.shape)
    result = _find_low_rank_full(arbitrary_matrix, 3, allocated_memory)
    assert result is allocated_memory


def test_find_low_rank_full_matches_shape_of_input_matrix() -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    allocated_memory = np.zeros(arbitrary_matrix.shape)
    result = _find_low_rank_full(arbitrary_matrix, 3, allocated_memory)
    assert result.shape == arbitrary_matrix.shape


def test_find_low_rank_full_recovers_known_result() -> None:
    # fmt: off
    low_rank_input = np.array([
        [1.0,   4.0,  -3.0],
        [2.0,   8.0,  -6.0],
        [3.0,  12.0,  -9.0]
    ])
    # fmt: on
    allocated_memory = np.zeros(low_rank_input.shape)
    result = _find_low_rank_full(low_rank_input, 1, allocated_memory)
    np.testing.assert_array_almost_equal(low_rank_input, result)


def test_find_low_rank_full_recovers_full_rank_input_when_allowed_full_rank() -> None:
    # fmt: off
    base_matrix = np.array([
        [3.0,  2.0,   2.0],
        [2.0,  3.0,  -2.0]
    ])
    # fmt: on
    allocated_memory = np.zeros(base_matrix.shape)
    returned_val = _find_low_rank_full(base_matrix, sys.maxsize, allocated_memory)
    np.testing.assert_array_almost_equal(base_matrix, returned_val)


def test_find_low_rank_rt_matches_shape_of_input_matrix() -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    result = _find_low_rank_random_truncated(arbitrary_matrix, 3)
    assert result.shape == arbitrary_matrix.shape


def test_find_low_rank_rt_recovers_known_result() -> None:
    # fmt: off
    low_rank_input = np.array([
        [1.0,   4.0,  -3.0],
        [2.0,   8.0,  -6.0],
        [3.0,  12.0,  -9.0]
    ])
    # fmt: on
    result = _find_low_rank_random_truncated(low_rank_input, 1)
    np.testing.assert_array_almost_equal(low_rank_input, result)


def test_find_low_rank_rt_recovers_full_rank_input_when_allowed_full_rank() -> None:
    # fmt: off
    base_matrix = np.array([
        [3.0,  2.0,   2.0],
        [2.0,  3.0,  -2.0]
    ])
    # fmt: on
    returned_val = _find_low_rank_random_truncated(base_matrix, max(base_matrix.shape))
    np.testing.assert_array_almost_equal(base_matrix, returned_val)


def test_find_low_rank_t_matches_shape_of_input_matrix() -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    result = _find_low_rank_exact_truncated(arbitrary_matrix, 3)
    assert result.shape == arbitrary_matrix.shape


def test_find_low_rank_t_recovers_known_result() -> None:
    # fmt: off
    low_rank_input = np.array([
        [1.0,   4.0,  -3.0],
        [2.0,   8.0,  -6.0],
        [3.0,  12.0,  -9.0]
    ])
    # fmt: on
    result = _find_low_rank_exact_truncated(low_rank_input, 1)
    np.testing.assert_array_almost_equal(low_rank_input, result)


def test_find_low_rank_t_recovers_full_rank_input_when_allowed_full_rank() -> None:
    # when using the arpack algorithm, the underlying SVD solver cannot recover a
    # full-rank matrix: you can only set the target to less than min(shape).
    pass


@patch("lzcompression.util._find_low_rank_random_truncated")
@patch("lzcompression.util._find_low_rank_exact_truncated")
@patch("lzcompression.util._find_low_rank_full")
def test_find_low_rank_dispatches_appropriately(
    mock_full: Mock, mock_lr_exact: Mock, mock_lr_rand: Mock
) -> None:
    util = np.ones((3, 2))
    _ = find_low_rank(util, 2, util, SVDStrategy.RANDOM_TRUNCATED)
    assert mock_full.call_count == 0
    assert mock_lr_exact.call_count == 0
    assert mock_lr_rand.call_count == 1
    _ = find_low_rank(util, 2, util, SVDStrategy.EXACT_TRUNCATED)
    assert mock_full.call_count == 0
    assert mock_lr_exact.call_count == 1
    assert mock_lr_rand.call_count == 1
    _ = find_low_rank(util, 2, util, SVDStrategy.FULL)
    assert mock_full.call_count == 1
    assert mock_lr_exact.call_count == 1
    assert mock_lr_rand.call_count == 1


def test_find_low_rank_throws_on_unknown_strategy() -> None:
    util = np.ones((3, 2))
    with raises(ValueError, match="Unsupported"):
        _ = find_low_rank(util, 2, util, cast(SVDStrategy, -5))
