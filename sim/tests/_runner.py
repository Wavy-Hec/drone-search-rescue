"""Minimal test runner so the CPU tests also run without pytest (``python tests/test_x.py``).

Tests are plain ``test_*`` functions; they skip by raising :class:`unittest.SkipTest` (pytest honours it too).
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for p in (str(SRC_DIR), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def require(module_name: str):
    """Import a module or skip the calling test."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # ImportError, or a broken optional dependency
        raise unittest.SkipTest(f"{module_name} not available ({exc.__class__.__name__}: {exc})") from None


def run_functions(namespace: dict, title: str) -> tuple[int, int, int]:
    """Run every ``test_*`` callable in ``namespace``; returns (passed, failed, skipped)."""
    passed = failed = skipped = 0
    for name, fn in sorted(namespace.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        t0 = time.time()
        try:
            fn()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"  SKIP {title}::{name} ({exc})")
        except Exception:
            failed += 1
            print(f"  FAIL {title}::{name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"  PASS {title}::{name} ({time.time() - t0:.2f} s)")
    return passed, failed, skipped


def main_for(namespace: dict, title: str) -> None:
    """Entry point for ``if __name__ == "__main__"`` in a test module."""
    p, f, s = run_functions(namespace, title)
    print(f"{title}: {p} passed, {f} failed, {s} skipped")
    sys.exit(1 if f else 0)
