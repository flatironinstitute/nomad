"""Defines utilities for path management.

Functions:
    make_path: Makes an output path from a user-specified base.

"""
from pathlib import Path
from datetime import datetime
from os import path, makedirs


# Using os.makedirs because it makes testing easier--we can mock just one
# import rather than trying to mix mocks of the Path object
def make_path(base_path: str = ".", *, use_exact_path: bool = False) -> Path:
    """Get reference to guaranteed-existing output directory with known base.

    Unless otherwise requested, the returned path will be timestamped
    (to second resolution) and empty at the time it was returned.

    Args:
        base_path: If set, determine the root where any new output path
            will be created. Defaults to '.'.
        use_exact_path: If True, do not create a timestamped subdirectory.
            Defaults to False.

    Returns:
        A Path object guaranteed to exist (and, unless use_exact_path is set,
        to be empty) at return time.
    """
    now = datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S")

    bp = Path(base_path)
    if not bp.exists():
        makedirs(bp)
    elif not bp.is_dir():
        raise ValueError(
            f"Requested base path {base_path} exists and is not a directory."
        )
    if use_exact_path:
        return bp
    full_path = path.join(bp, time)
    fp = Path(full_path)
    ordinal = 0
    while fp.exists():
        ordinal += 1
        fp = Path(path.join(full_path, f"-${ordinal}"))
    makedirs(fp)
    return fp
