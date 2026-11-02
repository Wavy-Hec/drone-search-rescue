# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures of the kit-less test suite.

Nothing here launches Isaac Sim. Tests that need Isaac Lab, torch, ``pxr`` or the original 2.x sources skip
themselves when those are missing, so the suite also runs in a plain Python environment with only numpy and pytest:
the pure-Python checks of the tooling run there, everything else reports a skip reason.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _is_lfs_pointer(path: Path) -> bool:
    """Whether ``path`` is a Git LFS pointer file instead of the object it stands for."""
    return path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs")


@pytest.fixture(scope="session")
def project_dir() -> Path:
    """Root of the project checkout."""
    return PROJECT


@pytest.fixture(scope="session")
def load_script() -> Callable[[str], ModuleType]:
    """Import ``scripts/<name>.py`` as a module (once per session)."""

    def load(name: str) -> ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses in the scripts resolve their module through sys.modules
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
        return module

    return load


@pytest.fixture(scope="session")
def require_file() -> Callable[[Path], Path]:
    """Return ``path`` if it exists with real content; skip the test for a missing file or an LFS pointer."""

    def require(path: Path) -> Path:
        if not path.is_file() or _is_lfs_pointer(path):
            pytest.skip(f"{path.name} is missing or an un-smudged Git LFS pointer (run `git lfs pull`)")
        return path

    return require
