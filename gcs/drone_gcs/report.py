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
from .mapping import evaluate_detections, path_length
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
        for layer, fname in (("elevation", "final_map_elevation.png"), ("photo", "final_map_photo.png"), ("coverage", "coverage.png")):
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
            w.writerow([_round(v, 5) if isinstance(v, float) else (round(float(v), 5) if isinstance(v, np.floating) else v) for v in row])
    with open(run_dir / "coverage_timeline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("t", "elevation_coverage", "photo_coverage", "confirmed_objects"))
        for row in state.coverage_timeline:
            w.writerow([round(float(row[0]), 3), round(float(row[1]), 5), round(float(row[2]), 5), int(row[3])])
    files += ["trajectory.csv", "coverage_timeline.csv"]

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
            "min_hits": tracker.min_hits,
        },
        "evaluation": None
        if evaluation is None
        else {k: v for k, v in evaluation.items() if k not in ("matches", "unmatched_gt", "unmatched_pred")},
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
    if extra:
        summary.update(extra)
    if video_path is not None:
        files.append(Path(video_path).name)
    (run_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    files.append("summary.json")
    return summary


def format_summary(s: dict) -> str:
    """Short human-readable summary."""
    lines = [f"run {s['run']}  mission '{s['mission']['name']}'  end: {s.get('end_reason')}"]
    f = s["flight"]
    lines.append(f"  flight {f['flight_time_s']:.1f} s, {f['path_length_m']:.1f} m flown")
    c = s["coverage"]
    lines.append(f"  coverage: scan {c['elevation_pct']:.1f} %, photo {c['photo_pct']:.1f} %  ({c['ray_hits']:,} hits)")
    d = s["detections"]
    lines.append(f"  objects: {d['confirmed']} confirmed ({d['tentative']} tentative) from {d['raw_detections']} detections")
    e = s.get("evaluation")
    if e:
        p = "-" if e["precision"] is None else f"{e['precision']:.2f}"
        r = "-" if e["recall"] is None else f"{e['recall']:.2f}"
        err = "-" if e["mean_err_xy_m"] is None else f"{e['mean_err_xy_m']:.3f} m"
        lines.append(f"  vs ground truth: precision {p}, recall {r}, mean horizontal error {err}")
    return "\n".join(lines)
