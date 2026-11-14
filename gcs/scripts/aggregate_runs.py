#!/usr/bin/env python3
"""Put the scorecards of several ground-station runs into one benchmark table.

Every run folder holds the ``summary.json`` the ground station writes after a mission. One mission on one site says
little about how often the system finds people: this script reads the scorecards of many missions (other site
seeds, other terrain seeds) and reports the rates and the spread.

For the people found it pools the counts over all missions and gives a 95 % Wilson score interval for that rate.
The interval treats every person as an independent trial; people on one site share the flight, so read it as a
lower bound on the uncertainty, next to the mean and the standard deviation over the missions.

Writes ``benchmark.csv`` (one row per run), ``benchmark.json`` (the rows and the aggregate) and ``benchmark.md``
(both as tables) into ``--out`` (default: a folder ``benchmark`` next to the first run), and prints the tables.

Example (after ``scripts/run_benchmark.sh --seeds 0-9`` at the repository root)::

    python scripts/aggregate_runs.py runs/bench_s*
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

LANDED = ("landed", "complete")
"""END reasons of a mission that flew its whole plan."""

Z95 = 1.959964
"""Normal quantile of a two-sided 95 % interval."""

COLUMNS = (
    "run",
    "mission",
    "seed",
    "terrain_seed",
    "end_reason",
    "people_found",
    "people_total",
    "outside_found",
    "outside_total",
    "false_alarms",
    "objects_found",
    "objects_total",
    "pin_error_m",
    "people_pin_error_m",
    "map_height_error_m",
    "map_height_error_smooth_m",
    "coverage_pct",
    "photo_coverage_pct",
    "flight_time_s",
    "closest_approach_m",
)
"""Columns of ``benchmark.csv``, in order."""


def wilson_interval(found: int, total: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval ``(low, high)`` of the rate ``found / total``; None without trials."""
    if total <= 0:
        return None
    p = found / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / denom
    # at 0 and at 1 the interval ends exactly there; say so instead of leaving a rounding error
    low = 0.0 if found == 0 else max(0.0, centre - half)
    high = 1.0 if found == total else min(1.0, centre + half)
    return low, high


def read_run(run_dir: Path) -> dict | None:
    """One row of the benchmark from ``run_dir/summary.json``; None when there is no usable scorecard."""
    path = run_dir / "summary.json"
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    card = summary.get("scorecard")
    if not isinstance(card, dict):
        return None
    mission = summary.get("mission") or {}
    evaluation = summary.get("evaluation") or {}
    outside = card.get("people_outside_area") or {}
    return {
        "run": run_dir.name,
        "mission": mission.get("name"),
        "seed": mission.get("seed"),
        "terrain_seed": mission.get("terrain_seed"),
        "end_reason": summary.get("end_reason"),
        "people_found": card.get("people_found"),
        "people_total": card.get("people_total"),
        "outside_found": outside.get("found"),
        "outside_total": outside.get("total"),
        "false_alarms": card.get("false_alarms"),
        "objects_found": evaluation.get("tp"),
        "objects_total": evaluation.get("ground_truth"),
        "pin_error_m": card.get("pin_error_m"),
        "people_pin_error_m": card.get("people_pin_error_m"),
        "map_height_error_m": card.get("map_height_error_m"),
        "map_height_error_smooth_m": card.get("map_height_error_smooth_m"),
        "coverage_pct": card.get("coverage_pct"),
        "photo_coverage_pct": card.get("photo_coverage_pct"),
        "flight_time_s": card.get("flight_time_s"),
        "closest_approach_m": card.get("closest_approach_m"),
    }


def spread(values: list[float]) -> dict | None:
    """Count, mean, standard deviation (sample; 0 for one value), minimum and maximum; None without values."""
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def column(rows: list[dict], key: str) -> list[float]:
    """The values of one column that are known (not None)."""
    return [float(r[key]) for r in rows if r.get(key) is not None]


def pooled(rows: list[dict], found_key: str, total_key: str) -> dict | None:
    """Counts summed over the runs that know both, their rate and its Wilson interval; None when no run does."""
    pairs = [(int(r[found_key]), int(r[total_key])) for r in rows if r.get(found_key) is not None and r.get(total_key)]
    if not pairs:
        return None
    found, total = sum(f for f, _ in pairs), sum(t for _, t in pairs)
    low, high = wilson_interval(found, total)
    return {
        "runs": len(pairs),
        "found": found,
        "total": total,
        "rate": found / total,
        "interval95": [low, high],
        "runs_all_found": sum(f == t for f, t in pairs),
        "per_run_rate": spread([f / t for f, t in pairs]),
    }


def aggregate(rows: list[dict]) -> dict:
    """The benchmark's summary numbers over all rows (metrics a run does not report are left out of their mean)."""
    alarms = column(rows, "false_alarms")
    return {
        "runs": len(rows),
        "landed": sum(r.get("end_reason") in LANDED for r in rows),
        "people": pooled(rows, "people_found", "people_total"),
        "objects": pooled(rows, "objects_found", "objects_total"),
        "false_alarms": None
        if not alarms
        else {**spread(alarms), "total": int(sum(alarms)), "runs_without": sum(a == 0 for a in alarms)},
        "pin_error_m": spread(column(rows, "pin_error_m")),
        "people_pin_error_m": spread(column(rows, "people_pin_error_m")),
        "map_height_error_m": spread(column(rows, "map_height_error_m")),
        "map_height_error_smooth_m": spread(column(rows, "map_height_error_smooth_m")),
        "coverage_pct": spread(column(rows, "coverage_pct")),
        "photo_coverage_pct": spread(column(rows, "photo_coverage_pct")),
        "flight_time_s": spread(column(rows, "flight_time_s")),
        "closest_approach_m": spread(column(rows, "closest_approach_m")),
    }


def _pm(s: dict | None, digits: int, unit: str = "") -> str:
    """``mean ± std unit (min .. max, n runs)`` of a :func:`spread`, or a dash."""
    if s is None:
        return "-"
    return (
        f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}{unit} "
        f"({s['min']:.{digits}f} .. {s['max']:.{digits}f}, {s['n']} runs)"
    )


def _rate(p: dict | None, what: str) -> str:
    """``found / total (rate; 95 % interval)`` of a :func:`pooled` count, or a dash."""
    if p is None:
        return "-"
    low, high = p["interval95"]
    return (
        f"{p['found']} / {p['total']} ({100 * p['rate']:.1f} %; 95 % interval {100 * low:.1f}-{100 * high:.1f} %, "
        f"every {what} counted as one trial)"
    )


def _cell(value, digits: int | None = None) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if digits is not None else str(value)


def markdown(rows: list[dict], agg: dict, failed: list[str]) -> str:
    """The benchmark as Markdown: the aggregate, then one line per run."""
    lines = [f"# Benchmark: {agg['runs']} missions", "", "| | |", "|---|---|"]
    people, alarms = agg["people"], agg["false_alarms"]
    summary = [("Missions", f"{agg['runs']} ({agg['landed']} flew the whole plan)")]
    summary.append(("People found", _rate(people, "person")))
    if people is not None:
        per_run = people["per_run_rate"]
        summary.append(("Missions with every person found", f"{people['runs_all_found']} / {people['runs']}"))
        summary.append(
            (
                "People found per mission",
                f"{100 * per_run['mean']:.1f} ± {100 * per_run['std']:.1f} % (lowest {100 * per_run['min']:.1f} %)",
            )
        )
    if alarms is not None:
        summary.append(
            (
                "False alarms",
                f"{alarms['total']} in total; {alarms['mean']:.2f} per mission (most {alarms['max']:.0f}); "
                f"none in {alarms['runs_without']} / {alarms['n']} missions",
            )
        )
    summary += [
        ("Objects found (all classes)", _rate(agg["objects"], "object")),
        ("Pin error", _pm(agg["pin_error_m"], 3, " m")),
        ("Pin error, people", _pm(agg["people_pin_error_m"], 3, " m")),
        ("Map height error (RMSE)", _pm(agg["map_height_error_m"], 3, " m")),
        ("Map height error away from steps", _pm(agg["map_height_error_smooth_m"], 4, " m")),
        ("Scan coverage", _pm(agg["coverage_pct"], 1, " %")),
        ("Photo coverage", _pm(agg["photo_coverage_pct"], 1, " %")),
        ("Flight time", _pm(agg["flight_time_s"], 1, " s")),
        ("Closest approach to obstacles", _pm(agg["closest_approach_m"], 2, " m")),
    ]
    lines += [f"| {k} | {v} |" for k, v in summary]
    lines += [
        "",
        "Each value is the mean ± the standard deviation over the missions (lowest .. highest). The interval of the "
        "pooled counts is a Wilson score interval.",
        "",
        "## Missions",
        "",
        "| Run | Site seed | Terrain seed | End | People | False alarms | Objects | Pin error [m] | "
        "Map error [m] | Scanned [%] | Flight [s] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        people_cell = "-" if r["people_found"] is None else f"{r['people_found']} / {_cell(r['people_total'])}"
        objects_cell = "-" if r["objects_found"] is None else f"{r['objects_found']} / {_cell(r['objects_total'])}"
        lines.append(
            f"| {r['run']} | {_cell(r['seed'])} | {_cell(r['terrain_seed'])} | {_cell(r['end_reason'])} | "
            f"{people_cell} | {_cell(r['false_alarms'])} | {objects_cell} | {_cell(r['pin_error_m'], 3)} | "
            f"{_cell(r['map_height_error_m'], 3)} | {_cell(r['coverage_pct'], 1)} | {_cell(r['flight_time_s'], 1)} |"
        )
    if failed:
        lines += ["", f"Without a scorecard (left out): {', '.join(failed)}."]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, rows: list[dict], agg: dict, failed: list[str]) -> None:
    """``benchmark.csv``, ``benchmark.json`` and ``benchmark.md`` in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "benchmark.json").write_text(json.dumps({"aggregate": agg, "runs": rows, "failed": failed}, indent=2))
    (out_dir / "benchmark.md").write_text(markdown(rows, agg, failed))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="ground-station run folders (each with a summary.json)")
    p.add_argument("--out", default=None, help="output folder (default: 'benchmark' next to the first run)")
    a = p.parse_args(argv)
    run_dirs = [Path(r).expanduser() for r in a.runs]
    rows, failed = [], []
    for run_dir in run_dirs:
        row = read_run(run_dir)
        if row is None:
            failed.append(run_dir.name)
        else:
            rows.append(row)
    if not rows:
        print(f"no run with a scorecard among: {', '.join(failed)}", file=sys.stderr)
        return 1
    agg = aggregate(rows)
    out_dir = Path(a.out).expanduser() if a.out else run_dirs[0].resolve().parent / "benchmark"
    write_outputs(out_dir, rows, agg, failed)
    print(markdown(rows, agg, failed))
    print(f"wrote {out_dir}/benchmark.md, benchmark.csv and benchmark.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
