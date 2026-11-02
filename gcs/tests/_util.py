"""Shared test helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def short_tmpdir(prefix: str = "dgcs") -> str:
    """A temporary directory with a short path (UNIX socket paths must stay under 108 bytes)."""
    base = os.environ.get("TMPDIR") or "/tmp"
    if len(base) > 80:
        base = "/tmp"
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def rmtree(path) -> None:
    shutil.rmtree(path, ignore_errors=True)
