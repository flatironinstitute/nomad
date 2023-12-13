import numpy as np
from unittest.mock import Mock, patch
from pytest import approx, raises
from typing import cast

from fi_nomad.types import LossType
from fi_nomad.util.loss_util import (
    _squared_difference_loss,
    _frobenius_norm_loss,
    compute_loss,
)

PKG = "fi_nomad.util.loss_util"


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
