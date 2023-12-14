import numpy as np
from unittest.mock import Mock, patch
from pytest import raises
from typing import Callable, Tuple, cast

import pytest

from fi_nomad.types import SVDStrategy
from fi_nomad.types.types import FloatArrayType
from fi_nomad.util.decomposition_util import (
    _find_low_rank_full,
    _find_low_rank_random_truncated,
    _find_low_rank_exact_truncated,
    find_low_rank,
    two_part_factor,
)

PKG = "fi_nomad.util.decomposition_util"

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
    np.testing.assert_allclose(low_rank_input, result)


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
    np.testing.assert_allclose(base_matrix, returned_val)


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


def test_two_part_factor() -> None:
    # fmt: off
    rank_two = np.array([
        [ 1.,  0.1,  2.,  -0.3],
        [ 5., -1.,  10.,   3. ],
        [ 7.,  1.,  14.,  -3. ],
        [ 9.,  2.,  18.,  -6. ]
    ])
    # fmt: on
    (A, B) = two_part_factor(rank_two)
    np.testing.assert_allclose(rank_two, A @ B)
    assert A.shape == (4, 2)
    assert B.shape == (2, 4)
