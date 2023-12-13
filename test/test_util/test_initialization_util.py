import numpy as np
from pytest import raises
from typing import cast

import pytest

from fi_nomad.types import InitializationStrategy
from fi_nomad.types.types import FloatArrayType
from fi_nomad.util.initialization_util import initialize_low_rank_candidate


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
