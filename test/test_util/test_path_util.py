from unittest.mock import Mock, patch
from pytest import raises
from datetime import datetime
from os import path, getcwd
from pathlib import Path

from fi_nomad.util.path_util import make_path

PKG = "fi_nomad.util.path_util"


@patch(f"{PKG}.makedirs")
@patch(f"{PKG}.datetime")
def test_make_path_creates_timestamped_path(mock_dt: Mock, mock_makedirs: Mock) -> None:
    # "Now" should return November 22, 1999, at 13:14:15
    # So returned subdirectory should be 19991122-131415
    mock_dt.now.return_value = datetime(1999, 11, 22, 13, 14, 15)
    expected_subdir = "19991122-131415"
    assert expected_subdir == mock_dt.now().strftime("%Y%m%d-%H%M%S")

    basepath = "test-base"
    expected_full_path = path.join(basepath, expected_subdir)
    expected_abs_path = path.join(getcwd(), expected_full_path)
    res = make_path(basepath)
    mock_makedirs.assert_called_with(Path(expected_full_path))
    assert str(res.absolute()) == expected_abs_path


@patch(f"{PKG}.makedirs")
@patch(f"{PKG}.Path")
def test_make_path_raises_on_non_directory_basepath(
    mock_Path: Mock, mock_makedirs: Mock
) -> None:
    mock_pathobj = Mock()
    mock_pathobj.exists = lambda: True
    mock_pathobj.is_dir = lambda: False
    mock_Path.return_value = mock_pathobj
    with raises(AssertionError):
        mock_makedirs.assert_any_call()
    with raises(ValueError, match="exists and is not a directory"):
        _ = make_path("test-base")


@patch(f"{PKG}.makedirs")
def test_make_path_honors_exact_path_request(mock_makedirs: Mock) -> None:
    basepath = "test-base"
    res = make_path(basepath, use_exact_path=True)
    mock_makedirs.assert_called_once()
    expected_abs_path = path.join(getcwd(), basepath)
    assert str(res.absolute()) == expected_abs_path


@patch(f"{PKG}.makedirs")
def test_make_path_creates_bp_if_not_exists(mock_makedirs: Mock) -> None:
    basepath = "test-base"
    assert not path.exists(basepath)
    _ = make_path(basepath)
    # get first Call to mock (first [0])
    # take its args from the Call object (second [0])
    # and then take the first argument from the args tuple (third [0])
    assert mock_makedirs.call_args_list[0][0][0] == Path(basepath)


@patch(f"{PKG}.makedirs")
def test_make_path_does_not_attempt_to_create_existing_bp(mock_makedirs: Mock) -> None:
    basepath = "."
    assert path.exists(basepath)
    _ = make_path()
    mock_makedirs.assert_called_once()


@patch(f"{PKG}.path")
@patch(f"{PKG}.makedirs")
@patch(f"{PKG}.Path")
@patch(f"{PKG}.datetime")
def test_make_path_increments_ordinal_if_needed(
    mock_dt: Mock, mock_Path: Mock, mock_makedirs: Mock, mock_path: Mock
) -> None:
    # defer joining to the system-appropriate os.path.join, but
    # knowing we have to get the pathname out of the mock path object
    mock_path.join = lambda x, y: path.join(x if isinstance(x, str) else x.str_val, y)

    mock_dt.now.return_value = datetime(1999, 11, 22, 13, 14, 15)
    expected_subdir = "19991122-131415"
    assert expected_subdir == mock_dt.now().strftime("%Y%m%d-%H%M%S")

    basepath = "test-base"
    expected_full_path = path.join(basepath, expected_subdir)
    expected_abs_path = path.join(getcwd(), expected_full_path, "-2")

    mock_pathobjs = [Mock(), Mock(), Mock(), Mock()]
    for i in range(3):
        mock_pathobjs[i].exists = lambda: True
        mock_pathobjs[i].str_val = path.join(basepath, expected_subdir)

    mock_pathobjs[3].exists = lambda: False
    mock_pathobjs[3].str_val = expected_full_path
    mock_Path.side_effect = mock_pathobjs

    res = make_path()
    mock_makedirs.assert_called_once_with(mock_pathobjs[3])
    assert res == mock_pathobjs[3]
