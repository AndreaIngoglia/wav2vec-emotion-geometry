"""
common.paths - two helpers used by every script:
  project_root()  locate the repo root (folder containing the common package),
                  used only to put it on sys.path.
  output_dir()    return/create an output/ folder next to the calling script.
"""

from __future__ import annotations

import sys
from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    """Return the repository root (the directory containing the ``common`` package).

    Walks upward from ``start`` (default: this file) until it finds a directory
    that contains ``common/__init__.py``. Falls back to the current working
    directory if nothing is found.
    """
    here = Path(start or __file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "common" / "__init__.py").exists():
            return parent
    return Path.cwd()


def ensure_importable(start: str | Path | None = None) -> Path:
    """Put the project root on ``sys.path`` so ``import common`` works."""
    root = project_root(start)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def output_dir(script_file: str | Path, *subdirs: str, create: bool = True) -> Path:
    """Return the ``output`` directory next to ``script_file``.

    Parameters
    ----------
    script_file:
        Pass ``__file__`` from the calling script.
    *subdirs:
        Optional sub-folders created inside ``output`` (e.g. a dataset name).
    create:
        Create the directory tree if it does not exist (default ``True``).

    Examples
    --------
    >>> out = output_dir(__file__)                 # .../<script_dir>/output
    >>> out = output_dir(__file__, "ESD")          # .../<script_dir>/output/ESD
    """
    base = Path(script_file).resolve().parent / "output"
    if subdirs:
        base = base.joinpath(*subdirs)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base
