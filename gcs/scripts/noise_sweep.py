#!/usr/bin/env python3
"""Replay one recorded mission at several levels of one sensor-noise setting and put the scorecards in a table.

In the simulator the drone's pose, the depth image and the ray hits are exact, so the pin error of a mission only
shows that the geometry is consistent. This script answers "what would a real drone get": it replays a link
recording (``gcs.py --record``) once per noise level and seed with ``gcs.py --replay FILE --headless --no-video
--noise SPEC``, one run at a time, and reads the scorecard of every run's ``summary.json``.

``--param`` is the swept setting and ``--values`` its levels; ``--fixed`` holds the other settings (all in the
``gcs.py --noise`` format, see ``drone_gcs/noise.py``; the swept value and the seed override ``--fixed``). Every
level is run once per seed in ``--seeds``. Arguments after ``--`` go to ``gcs.py`` unchanged, e.g. ``--detector gt``
or ``--weights FILE``.

Writes into ``--out`` (default: a time-stamped folder under ``runs/``): one run folder per level and seed
(``<param>_<value>_s<seed>``), ``sweep.csv`` (one row per run) and ``sweep.md`` (one row per level: mean and
standard deviation over the seeds), and prints that table. A run that fails is reported and left out of the means.

Example::

    python scripts/gcs.py --record --name flight          # once, with the simulator running
    python scripts/noise_sweep.py --replay runs/flight/link.dgcrec --param pos --values 0,0.1,0.25,0.5,1,2 \\
        --fixed att=0.5,depth=0.01 --seeds 0,1,2 -- --detector gt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_gcs.noise import KEYS, parse_noise  # noqa: E402

GCS_SCRIPT = PROJECT_ROOT / "scripts" / "gcs.py"

METRICS = (
    ("people_found", "People found", 2),
    ("false_alarms", "False alarms", 2),
    ("pin_error_m", "Pin error [m]", 3),
    ("people_pin_error_m", "Pin error, people [m]", 3),
    ("map_height_error_smooth_m", "Map height error away from steps [m]", 4),
)
"""Scorecard entries of the table: (key in ``summary.json`` -> ``scorecard``, column title, digits)."""

COLUMNS = ("param", "value", "seed", "noise", "run", "ok", "people_total", *(key for key, _, _ in METRICS))
"""Columns of ``sweep.csv``, in order."""


def read_scorecard(run_dir: Path) -> dict | None:
    """The scorecard in ``run_dir/summary.json``; None when the run did not write one."""
    try:
        card = json.loads((run_dir / "summary.json").read_text()).get("scorecard")
    except (OSError, json.JSONDecodeError):
        return None
    return card if isinstance(card, dict) else None


def run_one(replay: Path, spec: str, out_dir: Path, name: str, extra: list[str]) -> dict | None:
    """Replay the recording once with the noise ``spec`` into ``out_dir/name``; returns the scorecard, or None
    (after printing why) when the ground station failed or wrote no scorecard."""
    cmd = [sys.executable, str(GCS_SCRIPT), "--replay", str(replay), "--headless", "--no-video", "--noise", spec]
    # --linger 0: nothing is shown or filmed, so there is no reason to wait after the mission's END
    cmd += ["--out-dir", str(out_dir), "--name", name, "--linger", "0", *extra]
    done = subprocess.run(cmd, capture_output=True, text=True)
    card = read_scorecard(out_dir / name) if done.returncode == 0 else None
    if card is None:
        print(f"{name}: FAILED (exit code {done.returncode}); the end of its output:", file=sys.stderr)
        print("\n".join(done.stderr.splitlines()[-8:]), file=sys.stderr)
    return card


def mean_std(values: list[float]) -> tuple[float, float] | None:
    """Mean and standard deviation (sample; 0 for one value); None without values."""
    if not values:
        return None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    """One entry per swept value, in the order the values first appear: ``{"value", "runs", "ok", "people_total",
    <metric>: (mean, std) or None}`` over the runs that succeeded. A metric a run does not report (None) is left
    out of that metric's mean."""
    table = []
    for value in dict.fromkeys(r["value"] for r in rows):
        runs = [r for r in rows if r["value"] == value]
        ok = [r for r in runs if r["ok"]]
        entry = {"value": value, "runs": len(runs), "ok": len(ok)}
        entry["people_total"] = next((r["people_total"] for r in ok if r["people_total"] is not None), None)
        for key, _, _ in METRICS:
            entry[key] = mean_std([float(r[key]) for r in ok if r[key] is not None])
        table.append(entry)
    return table


def markdown(param: str, rows: list[dict], setup: str) -> str:
    """The sweep as a Markdown table: one row per value of ``param``, each cell mean ± std over the seeds.
    ``setup`` is a line of text about the recording and the other settings."""
    head = [f"{param} ({KEYS[param]})", "Runs", *(name for _, name, _ in METRICS)]
    lines = [f"# Noise sweep: {param}", "", setup, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for entry in summarize(rows):
        cells = [entry["value"], f"{entry['ok']} / {entry['runs']}"]
        for key, _, digits in METRICS:
            cell = "-" if entry[key] is None else f"{entry[key][0]:.{digits}f} ± {entry[key][1]:.{digits}f}"
            if key == "people_found" and entry[key] is not None and entry["people_total"] is not None:
                cell += f" of {entry['people_total']}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Each cell is the mean ± the standard deviation over the seeds (the runs that succeeded)."]
    failed = [r["run"] for r in rows if not r["ok"]]
    if failed:
        lines += ["", f"Failed runs (left out): {', '.join(failed)}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    extra: list[str] = []
    if "--" in argv:  # what follows goes to gcs.py unchanged
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", required=True, metavar="FILE", help="link recording made with gcs.py --record")
    p.add_argument("--param", required=True, choices=[k for k in KEYS if k != "seed"], help="the swept noise setting")
    p.add_argument("--values", required=True, help="comma-separated levels of --param, e.g. 0,0.1,0.25,0.5,1,2")
    p.add_argument("--fixed", default="", metavar="SPEC", help="the other noise settings, e.g. att=0.5,depth=0.01")
    p.add_argument("--seeds", default="0", help="comma-separated noise seeds, one run per level and seed (default 0)")
    p.add_argument("--out", default=None, metavar="DIR", help="output folder (default: runs/<time>_sweep_<param>)")
    a = p.parse_args(argv)
    replay = Path(a.replay).expanduser().resolve()
    if not replay.is_file():
        p.error(f"--replay: no such file: {a.replay}")
    values = [v.strip() for v in a.values.split(",") if v.strip()]
    try:
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    except ValueError:
        p.error("--seeds must be whole numbers, e.g. 0,1,2")
    if not values or not seeds:
        p.error("--values and --seeds need at least one entry each")
    for flag, items in (("--values", values), ("--seeds", seeds)):
        twice = sorted({str(v) for v in items if items.count(v) > 1})
        if twice:  # its second run would go to <name>_2 and the table would read the first one twice
            p.error(f"{flag}: {', '.join(twice)} listed twice")
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(a.out).expanduser() if a.out else PROJECT_ROOT / "runs" / f"{stamp}_sweep_{a.param}"
    rows = []
    for value in values:
        for seed in seeds:
            spec = ",".join(part for part in (a.fixed, f"{a.param}={value}", f"seed={seed}") if part)
            try:
                parse_noise(spec)  # a bad setting stops the sweep before its first run
            except ValueError as e:
                p.error(f"noise settings '{spec}': {e}")
            row = dict.fromkeys(COLUMNS)
            row.update(param=a.param, value=value, seed=seed, noise=spec, run=f"{a.param}_{value}_s{seed}", ok=0)
            rows.append(row)
    taken = [r["run"] for r in rows if (out_dir / r["run"]).exists()]
    if taken:  # gcs.py would write next to them (<name>_2) and the old results would be read
        p.error(f"{out_dir} already holds runs of this sweep ({', '.join(taken)}): choose another --out")
    for k, row in enumerate(rows, 1):
        card = run_one(replay, row["noise"], out_dir, row["run"], extra)
        if card is None:
            continue
        row["ok"] = 1
        row["people_total"] = card.get("people_total")
        for key, _, _ in METRICS:
            row[key] = card.get(key)
        shown = [f"{key}={'-' if row[key] is None else round(row[key], digits)}" for key, _, digits in METRICS]
        print(f"[{k}/{len(rows)}] {row['run']}: {'  '.join(shown)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    setup = (
        f"Recording {replay.name}; other noise settings: {a.fixed or 'none'}; seeds: {', '.join(map(str, seeds))}; "
        f"ground-station arguments: {' '.join(extra) or 'none'}."
    )
    text = markdown(a.param, rows, setup)
    (out_dir / "sweep.md").write_text(text)
    print(text)
    print(f"wrote {out_dir}/sweep.md and sweep.csv")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
