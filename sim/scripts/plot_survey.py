"""Plot a survey run directory (no Isaac Sim needed; any Python with numpy + matplotlib).

Writes into the run directory:

* ``map_with_trajectory.png`` -- elevation map from the ray hits, planned path (dashed) and flown trajectory
* ``map_error.png``           -- map minus ground truth, with the error histogram
* ``map_3d.png``              -- shaded 3D terrain with the flown trajectory
* ``altitude_profile.png``    -- altitude / terrain below, speed, and distance to the carrot over time
* ``summary.txt``             -- short text summary (re-generated from ``summary.json``)

Usage::

    python scripts/plot_survey.py runs/<name> [--theme light|dark]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from drone_lab.plotting import make_all_plots  # noqa: E402
from drone_lab.survey_io import format_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory written by survey_map.py or synthetic_survey.py.")
    parser.add_argument("--theme", choices=("light", "dark"), default="light", help="Colour theme of the figures.")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    if not (run_dir / "map_meta.json").exists():
        parser.error(f"{run_dir} is not a survey run directory (no map_meta.json)")
    paths = make_all_plots(run_dir, theme=args.theme)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        text = format_summary(json.loads(summary_path.read_text()))
        (run_dir / "summary.txt").write_text(text)
        print(text)
    for p in paths:
        print(f"[plot_survey] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
