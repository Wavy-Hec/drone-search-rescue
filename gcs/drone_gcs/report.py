"""Mission report: final maps (PNG + raw arrays), detections, trajectory, coverage timeline and summary."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .dashboard import render_report_map
from .mapping import evaluate_detections, path_length, terrain_error
from .protocol import CAMERA, SCAN, TELEMETRY, jsonable
from .state import GCSState

TRAJ_COLUMNS = (
    "drone_id", "t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz",
    "wp_index", "carrot_x", "carrot_y", "carrot_z", "agl",
)  # fmt: skip


def _round(v, n=4):
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, n) if math.isfinite(v) else None
    return v


def write_report(
    state: GCSState,
    run_dir: str | Path,
    *,
    video_path: str | Path | None = None,
    extra: dict | None = None,
    map_size: int = 1200,
    match_radius: float = 2.0,
) -> dict:
    """Write everything under ``run_dir``; returns the summary dict (also saved as ``summary.json``)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    name = (state.mission or {}).get("name", "survey")
    ce, cp = state.coverage()

    # ---- maps
    if state.grid is not None:
        titles = {
            "elevation": f"Elevation map (mean of ray hits)   {name}   scan coverage {100 * ce:.1f} %",
            "photo": f"Orthomosaic (camera pixels back-projected)   {name}   photo coverage {100 * cp:.1f} %",
            "coverage": f"Coverage (ray hits per cell; photo-only cells)   {name}",
        }
        for layer, fname in (
            ("elevation", "final_map_elevation.png"),
            ("photo", "final_map_photo.png"),
            ("coverage", "coverage.png"),
        ):
            img = render_report_map(state, layer, map_size, title=titles[layer])
            cv2.imwrite(str(run_dir / fname), img)
            files.append(fname)
        md = run_dir / "maps"
        md.mkdir(exist_ok=True)
        el = state.elevation
        np.save(md / "elevation_mean.npy", el.mean())
        np.save(md / "elevation_max.npy", el.max())
        np.save(md / "elevation_min.npy", el.min())
        np.save(md / "hit_count.npy", el.count)
        cv2.imwrite(str(md / "photo_rgb.png"), np.ascontiguousarray(state.mosaic.rgb[::-1, :, ::-1]))
        np.save(md / "photo_filled.npy", state.mosaic.filled)
        (md / "grid.json").write_text(
            json.dumps(
                {
                    "elevation_grid": state.grid.to_dict(),
                    "photo_grid": state.photo_grid.to_dict(),
                    "area": list(state.area),
                    "note": "npy arrays are (ny, nx), row 0 = southern edge (y_min); photo_rgb.png is north-up",
                },
                indent=2,
            )
        )
        files += [f"maps/{p.name}" for p in sorted(md.iterdir())]

    # ---- detections (+ evaluation against ground truth)
    tracker = state.tracker
    records = []
    for ob in tracker.objects:
        r = ob.to_dict()
        r["confirmed"] = ob.hits >= tracker.min_hits
        records.append(r)
    confirmed = [r for r in records if r["confirmed"]]
    evaluation = None
    if state.gt_objects is not None:
        evaluation = evaluate_detections(confirmed, state.gt_objects, match_radius)
        for m in evaluation["matches"]:
            r = confirmed[m["pred"]]
            g = state.gt_objects[m["gt"]]
            r["gt_id"] = g.get("id")
            r["err_xy_m"] = round(m["err_xy"], 4)
            r["err_z_m"] = round(m["err_z"], 4)
    det_cols = [
        "id", "class", "x", "y", "z", "confidence", "confidence_max", "hits", "first_seen", "last_seen",
        "spread_m", "confirmed", "gt_id", "err_xy_m", "err_z_m", "drone_id",
    ]  # fmt: skip
    with open(run_dir / "detections.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=det_cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    (run_dir / "detections.json").write_text(
        json.dumps(
            jsonable(
                {
                    "frame": "world (local ENU: x east, y north, z up) [m]",
                    "objects": records,
                    "ground_truth": state.gt_objects,
                    "evaluation": evaluation,
                }
            ),
            indent=2,
        )
    )
    files += ["detections.csv", "detections.json"]

    # ---- trajectory + coverage timeline
    with open(run_dir / "trajectory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRAJ_COLUMNS)
        for row in state.trajectory:
            w.writerow(
                [
                    _round(v, 5) if isinstance(v, float) else (round(float(v), 5) if isinstance(v, np.floating) else v)
                    for v in row
                ]
            )
    with open(run_dir / "coverage_timeline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("t", "elevation_coverage", "photo_coverage", "confirmed_objects"))
        for row in state.coverage_timeline:
            w.writerow([round(float(row[0]), 3), round(float(row[1]), 5), round(float(row[2]), 5), int(row[3])])
    files += ["trajectory.csv", "coverage_timeline.csv"]

    # ---- map height error against the true terrain (when the simulator sent it)
    terrain = None
    if state.gt_terrain is not None and state.elevation is not None:
        truth = state.gt_terrain.on_grid(state.grid)
        terrain = terrain_error(state.elevation.mean(), truth, state.area_mask)

    # ---- summary
    speeds = [math.hypot(r[9], r[10]) for r in state.trajectory if math.isfinite(r[9])]
    agls = [r[16] for r in state.trajectory if math.isfinite(r[16])]
    planned_len = path_length(state.planned_path[:, :3]) if state.planned_path is not None else None
    a = state.area
    by_class: dict[str, int] = {}
    for r in confirmed:
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1
    summary = {
        "run": run_dir.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mission": {
            "name": name,
            "mission_id": (state.mission or {}).get("mission_id"),
            "area": {"x_min": a[0], "x_max": a[1], "y_min": a[2], "y_max": a[3]} if a else None,
            "area_m2": (a[1] - a[0]) * (a[3] - a[2]) if a else None,
            "resolution_m": state.grid.resolution if state.grid else None,
            "photo_resolution_m": state.photo_grid.resolution if state.photo_grid else None,
            "survey_altitude": (state.mission or {}).get("survey_altitude"),
            "planned_path_length_m": planned_len,
            "waypoints": None if state.planned_path is None else len(state.planned_path),
            "auto_grid": state.auto_grid,
        },
        "ended": state.ended,
        "end_reason": state.end_reason or None,
        "flight": {
            "flight_time_s": state.mission_time,
            "path_length_m": state.flown_length(),
            "mean_speed_mps": float(np.mean(speeds)) if speeds else None,
            "max_speed_mps": float(np.max(speeds)) if speeds else None,
            "mean_agl_m": float(np.mean(agls)) if agls else None,
            "drones": sorted(state.drones),
        },
        "coverage": {
            "elevation_fraction": ce,
            "photo_fraction": cp,
            "elevation_pct": round(100 * ce, 2),
            "photo_pct": round(100 * cp, 2),
            "ray_hits": state.elevation.n_points if state.elevation else 0,
            "camera_frames_mapped": state.mosaic.n_frames if state.mosaic else 0,
        },
        "detections": {
            "confirmed": len(confirmed),
            "tentative": len(records) - len(confirmed),
            "raw_detections": tracker.n_detections,
            "by_class": by_class,
            "merge_radius_m": tracker.merge_radius,
            "merge_radius_by_class_m": tracker.class_radius,
            "min_hits": tracker.min_hits,
        },
        "evaluation": None
        if evaluation is None
        else {k: v for k, v in evaluation.items() if k not in ("matches", "unmatched_gt", "unmatched_pred")},
        "terrain_error": terrain,
        "drone_stats": state.end_fields.get("stats"),
        "messages": {
            "telemetry": state.counts[TELEMETRY],
            "scan": state.counts[SCAN],
            "camera": state.counts[CAMERA],
            **{f"errors_{k}": v for k, v in state.errors.items()},
            **({"ignored_other_mission": dict(state.stale)} if state.stale else {}),
        },
        "video": None if video_path is None else Path(video_path).name,
        "files": files,
    }
    summary["scorecard"] = scorecard(state, evaluation, terrain)
    if extra:
        summary.update(extra)
    if video_path is not None:
        files.append(Path(video_path).name)
    files += ["summary.json", "report.md"]
    (run_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    (run_dir / "report.md").write_text(report_markdown(summary, confirmed, state.gt_objects))
    return summary


def scorecard(state: GCSState, evaluation: dict | None, terrain: dict | None) -> dict:
    """The mission's headline numbers: people found, false alarms, pin error, map error, coverage, flight time
    and the closest approach to obstacles (reported by the drone in END)."""
    people = [o for o in state.tracker.confirmed() if o.cls == "person"]
    ce, cp = state.coverage()
    card: dict = {
        "people_found": len(people),
        "people_total": None,
        "false_alarms": None,
        "pin_error_m": None,
        "people_pin_error_m": None,
        "map_height_error_m": None,
        "map_height_error_smooth_m": None,
        "coverage_pct": round(100 * ce, 2),
        "photo_coverage_pct": round(100 * cp, 2),
        "flight_time_s": round(state.mission_time, 2),
        "closest_approach_m": None,
    }
    if evaluation is not None:
        person = evaluation["per_class"].get("person", {})
        card["people_found"] = person.get("tp", 0)
        card["people_total"] = person.get("ground_truth", 0)
        card["false_alarms"] = evaluation["fp"]
        card["pin_error_m"] = evaluation["mean_err_xy_m"]
        card["people_pin_error_m"] = person.get("mean_err_xy")
    if terrain is not None:
        card["map_height_error_m"] = (terrain["all"] or {}).get("rmse_m")
        card["map_height_error_smooth_m"] = (terrain["smooth"] or {}).get("rmse_m")
    stats = state.end_fields.get("stats") or {}
    card["closest_approach_m"] = (stats.get("closest_approach") or {}).get("closest_m")
    return card


def _fmt_duration(s: float) -> str:
    m, sec = divmod(round(float(s)), 60)
    return f"{m} min {sec:02d} s" if m else f"{sec} s"


def _fmt_m(v, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f} m"


def report_markdown(summary: dict, confirmed: list[dict], gt_objects: list[dict] | None) -> str:
    """The mission report as Markdown: scorecard, maps and the list of finds."""
    card = summary["scorecard"]
    m = summary["mission"]
    lines = [f"# Mission report: {m['name']}", ""]
    end = summary.get("end_reason") or "no END received"
    flight = summary["flight"]
    lines.append(
        f"{summary['created'].replace('T', ' ')} · end: {end} · flight {_fmt_duration(flight['flight_time_s'])}, "
        f"{flight['path_length_m']:.0f} m flown"
    )
    lines += ["", "| Scorecard | |", "|---|---|"]
    found = f"**{card['people_found']}**"
    if card["people_total"] is not None:
        found = f"**{card['people_found']} / {card['people_total']}**"
    pin = _fmt_m(card["pin_error_m"])
    if card["people_pin_error_m"] is not None:
        pin += f" (people {_fmt_m(card['people_pin_error_m'])})"
    height = _fmt_m(card["map_height_error_m"], 3)
    if card["map_height_error_m"] is not None:
        height += f" RMSE ({_fmt_m(card['map_height_error_smooth_m'], 3)} away from steps)"
    rows = [
        ("People found", found),
        ("False alarms", "-" if card["false_alarms"] is None else str(card["false_alarms"])),
        ("Average pin error", pin),
        ("Map height error", height),
        ("Coverage", f"{card['coverage_pct']:.1f} % scanned, {card['photo_coverage_pct']:.1f} % photographed"),
        ("Flight time", _fmt_duration(card["flight_time_s"])),
        ("Closest approach to obstacles", _fmt_m(card["closest_approach_m"], 1)),
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "![Photo map with the flight path and the pins](final_map_photo.png)",
        "",
        "![Elevation map with the flight path and the pins](final_map_elevation.png)",
        "",
        "## Finds",
        "",
        "| # | Class | x [m] | y [m] | z [m] | Confidence | Sightings | Ground truth | Pin error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(confirmed, key=lambda r: (r["class"] != "person", r["class"], r["id"]))
    for r in order:
        gt = r.get("gt_id")
        lines.append(
            f"| {r['id']} | {r['class']} | {r['x']:+.2f} | {r['y']:+.2f} | {r['z']:+.2f} | {r['confidence']:.2f} | "
            f"{r['hits']} | {gt if gt is not None else ('false alarm' if gt_objects is not None else '-')} | "
            f"{_fmt_m(r.get('err_xy_m'))} |"
        )
    if not order:
        lines.append("| | no confirmed finds | | | | | | | |")
    if gt_objects:
        matched = {r.get("gt_id") for r in confirmed}
        missed = [g for g in gt_objects if g.get("id") not in matched]
        people = [g for g in missed if g.get("class") == "person"]
        lines += ["", f"Not found: {len(missed)} of {len(gt_objects)} ground-truth objects"]
        if people:
            lines[-1] += ", people: " + ", ".join(str(g.get("id")) for g in people)
        lines[-1] += "."
    lines += [
        "",
        "Pins are matched to the ground truth by class within 2 m, nearest first; a confirmed pin without a match "
        "is a false alarm. The pin error is horizontal. The map height error compares the elevation map with the "
        "true terrain over the survey area; cells across a step are ambiguous, hence the second number. Coordinates "
        "are local ENU metres (x east, y north).",
        "",
    ]
    return "\n".join(lines)


def format_summary(s: dict) -> str:
    """Short human-readable summary."""
    lines = [f"run {s['run']}  mission '{s['mission']['name']}'  end: {s.get('end_reason')}"]
    f = s["flight"]
    lines.append(f"  flight {f['flight_time_s']:.1f} s, {f['path_length_m']:.1f} m flown")
    c = s["coverage"]
    lines.append(f"  coverage: scan {c['elevation_pct']:.1f} %, photo {c['photo_pct']:.1f} %  ({c['ray_hits']:,} hits)")
    d = s["detections"]
    lines.append(
        f"  objects: {d['confirmed']} confirmed ({d['tentative']} tentative) from {d['raw_detections']} detections"
    )
    e = s.get("evaluation")
    if e:
        p = "-" if e["precision"] is None else f"{e['precision']:.2f}"
        r = "-" if e["recall"] is None else f"{e['recall']:.2f}"
        err = "-" if e["mean_err_xy_m"] is None else f"{e['mean_err_xy_m']:.3f} m"
        lines.append(f"  vs ground truth: precision {p}, recall {r}, mean horizontal error {err}")
        card = s["scorecard"]
        lines.append(
            f"  people found {card['people_found']} / {card['people_total']}, false alarms {card['false_alarms']}"
        )
    t = (s.get("terrain_error") or {}).get("all")
    if t:
        lines.append(f"  map height error: RMSE {t['rmse_m']:.3f} m over {t['cells']:,} cells")
    return "\n".join(lines)
