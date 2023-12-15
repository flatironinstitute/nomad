from unittest.mock import Mock, patch
import numpy as np
from pytest import raises
from typing import cast

import pytest

from fi_nomad.types import InitializationStrategy
from fi_nomad.types.enums import KernelStrategy
from fi_nomad.types.types import FloatArrayType
from fi_nomad.util.initialization_util import (
    initialize_candidate,
    initialize_low_rank_candidate,
)

PKG = "fi_nomad.util.initialization_util"


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


## low-rank candidate initialization
@patch(f"{PKG}.initialize_low_rank_candidate")
def test_initialize_candidate_honors_initialization_strategy(mock_init: Mock) -> None:
    mock_x = np.array([[0, 2], [2, 0]])
    mock_checkpoint = np.eye(2)
    with raises(AssertionError):
        np.testing.assert_array_equal(mock_x, mock_checkpoint)
    k_strat = KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE

    _ = initialize_candidate(
        InitializationStrategy.COPY, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.BROADCAST_MEAN, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.BROADCAST_MEAN)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(
        mock_checkpoint, InitializationStrategy.KNOWN_MATRIX
    )


@patch(f"{PKG}.initialize_low_rank_candidate")
def test_initialize_candidate_forces_copy_strategy_for_base_model_free(
    mock_init: Mock,
) -> None:
    mock_x = np.array([[0, 2], [2, 0]])
    mock_checkpoint = Mock()
    k_strat = KernelStrategy.BASE_MODEL_FREE

    _ = initialize_candidate(
        InitializationStrategy.COPY, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.BROADCAST_MEAN, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)
    mock_init.reset_mock()

    _ = initialize_candidate(
        InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.COPY)


@patch(f"{PKG}.initialize_low_rank_candidate")
def test_initialize_candidate_defaults_to_copy_on_null_checkpoint(
    mock_init: Mock,
) -> None:
    mock_x = np.array([[0, 2], [2, 0]])
    mock_checkpoint = None
    k_strat = KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE
    _ = initialize_candidate(
        InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
    )
    mock_init.assert_called_once_with(mock_x, InitializationStrategy.KNOWN_MATRIX)


def test_initialize_candidate_throws_on_checkpoint_size_mismatch() -> None:
    mock_x = np.eye(3)
    mock_checkpoint = np.eye(4)
    k_strat = KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE
    with raises(ValueError, match="shape"):
        _ = initialize_candidate(
            InitializationStrategy.KNOWN_MATRIX, k_strat, mock_x, mock_checkpoint
        )
