import numpy as np
from unittest.mock import Mock, patch
from pytest import approx, raises
from typing import Callable, Tuple, cast

import pytest

from fi_nomad.types import InitializationStrategy, SVDStrategy, LossType
from fi_nomad.types.types import FloatArrayType
from fi_nomad.util.util import (
    initialize_low_rank_candidate,
    _squared_difference_loss,
    _frobenius_norm_loss,
    _find_low_rank_full,
    _find_low_rank_random_truncated,
    _find_low_rank_exact_truncated,
    find_low_rank,
    pdf_to_cdf_ratio_psi,
    compute_loss,
)

PKG = "fi_nomad.util.util"


@pytest.fixture
def init_matrix() -> FloatArrayType:
    # fmt: off
    base_matrix = np.array([
        [ 1,  5,  0],
        [ 0,  0,  0]
    ])
    # fmt: on
    return base_matrix


def test_initialize_returns_copy(init_matrix: FloatArrayType) -> None:
    res = initialize_low_rank_candidate(init_matrix, InitializationStrategy.COPY)
    np.testing.assert_allclose(init_matrix, res)


def test_initialize_returns_copy_on_known_matrix(init_matrix: FloatArrayType) -> None:
    res = initialize_low_rank_candidate(
        init_matrix, InitializationStrategy.KNOWN_MATRIX
    )
    np.testing.assert_allclose(init_matrix, res)


def test_initialize_returns_mean(init_matrix: FloatArrayType) -> None:
    res = initialize_low_rank_candidate(
        init_matrix, InitializationStrategy.BROADCAST_MEAN
    )
    expected = np.mean(init_matrix)
    np.testing.assert_allclose(res, expected)


def test_initialize_returns_rowwise_mean() -> None:
    # fmt: off
    base_matrix = np.array([
        [ 1, 2, 0],
        [ 4, 0, 5]
    ])
    # fmt: on
    expected = np.array([[1, 1, 1], [3, 3, 3]])
    res = initialize_low_rank_candidate(
        base_matrix, InitializationStrategy.ROWWISE_MEAN
    )
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


def test_normal_log_pdf() -> None:
    pass


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


@patch(f"{PKG}._squared_difference_loss")
@patch(f"{PKG}._frobenius_norm_loss")
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


#### SVD
SVDFnType = Callable[[FloatArrayType, int], FloatArrayType]


# Wrapper fn since this one has a different signature from the others
def _find_low_rank_full_wrapper(shape: Tuple[int, int]) -> SVDFnType:
    allocated_memory = np.zeros(shape)
    return lambda matrix, rank: _find_low_rank_full(matrix, rank, allocated_memory)


def test_find_low_rank_full_uses_preallocated_memory() -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    allocated_memory = np.zeros(arbitrary_matrix.shape)
    result = _find_low_rank_full(arbitrary_matrix, 3, allocated_memory)
    assert result is allocated_memory


@pytest.mark.parametrize(
    "svdfn",
    [
        _find_low_rank_full_wrapper((6, 7)),
        _find_low_rank_random_truncated,
        _find_low_rank_exact_truncated,
    ],
)
def test_svd_fns_match_shape_of_input_matrix(svdfn: SVDFnType) -> None:
    arbitrary_matrix = np.random.random_sample((6, 7))
    result = svdfn(arbitrary_matrix, 3)
    assert result.shape == arbitrary_matrix.shape


@pytest.mark.parametrize(
    "svdfn",
    [
        _find_low_rank_full_wrapper((3, 3)),
        _find_low_rank_random_truncated,
        _find_low_rank_exact_truncated,
    ],
)
def test_svd_fns_recover_known_result(svdfn: SVDFnType) -> None:
    # fmt: off
    low_rank_input = np.array([
        [1.0,   4.0,  -3.0],
        [2.0,   8.0,  -6.0],
        [3.0,  12.0,  -9.0]
    ])
    # fmt: on
    result = svdfn(low_rank_input, 1)
    np.testing.assert_array_almost_equal(low_rank_input, result)


# Note: "Exact Truncated" method omitted for this
# Since the underlying SVD solver cannot recover a full-rank matrix
@pytest.mark.parametrize(
    "svdfn", [_find_low_rank_full_wrapper((2, 3)), _find_low_rank_random_truncated]
)
def test_svd_fns_recover_full_rank_input_when_allowed_full_rank(
    svdfn: SVDFnType,
) -> None:
    # fmt: off
    base_matrix = np.array([
        [3.0,  2.0,   2.0],
        [2.0,  3.0,  -2.0]
    ])
    # fmt: on
    returned_val = svdfn(base_matrix, max(base_matrix.shape))
    np.testing.assert_array_almost_equal(base_matrix, returned_val)


@patch(f"{PKG}._find_low_rank_random_truncated")
@patch(f"{PKG}._find_low_rank_exact_truncated")
@patch(f"{PKG}._find_low_rank_full")
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
