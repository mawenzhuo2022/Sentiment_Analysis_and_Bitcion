# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Project Paths module

"""Simple helpers for locating project files and folders."""
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _detect_project_root() -> Path:
    """Return the project root directory (the folder containing this file)."""
    return Path(__file__).resolve().parent


PROJECT_ROOT = _detect_project_root()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = SRC_DIR / "models"
FIGURES_DIR = DATA_DIR / "d_linear_prediction"


def data_path(*parts: PathLike) -> Path:
    """Build a path inside the ``data`` directory from path parts."""
    return DATA_DIR.joinpath(*map(Path, parts))


def ensure_directory(path: PathLike) -> Path:
    """Create the directory if it does not exist and return the Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
