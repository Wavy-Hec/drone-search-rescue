"""Run every CPU test module without pytest (env_isaaclab has no pytest).

Usage (no GPU needed; hide it so nothing touches the device)::

    CUDA_VISIBLE_DEVICES= python tests/run_cpu_tests.py

With pytest available (e.g. ``~/anaconda3/bin/python -m pytest tests``) the same tests run under pytest; tests
whose dependencies are missing (torch, Isaac Lab) are skipped there.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner import run_functions  # noqa: E402

MODULES = (
    "test_paths",
    "test_mapping",
    "test_carrot",
    "test_reference",
    "test_plotting",
    "test_video",
    "test_terrain",
    "test_env_cfgs",
    "test_registration",
    "test_survey_map_offline",
    "test_sar_labels",
    "test_sar_scenario",
    "test_sar_env_cfg",
)


def main() -> int:
    totals = [0, 0, 0]
    for name in MODULES:
        print(f"== {name}")
        mod = importlib.import_module(name)
        for i, n in enumerate(run_functions(vars(mod), name)):
            totals[i] += n
    print(f"\nTOTAL: {totals[0]} passed, {totals[1]} failed, {totals[2]} skipped")
    return 1 if totals[1] else 0


if __name__ == "__main__":
    sys.exit(main())
