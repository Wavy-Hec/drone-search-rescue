"""Survey run directory: logging helpers, file formats and summary metrics (numpy only).

A run directory (``runs/<name>/``) contains:

* ``trajectory.csv``   -- one row per drone per policy step (see :data:`TRAJECTORY_FIELDS`).
* ``planned_path.csv`` -- planned waypoint polyline of every drone: ``env,index,x,y,z,s``.
* ``elevation_map.npy`` -- 2.5D map from the ray hits, (ny, nx) float32, NaN = no hit.
* ``hit_count.npy``    -- ray hits per cell, (ny, nx) int32.
* ``ground_truth.npy`` -- terrain height per cell (dense vertical ray cast of the terrain mesh), (ny, nx) float32.
* ``ground_truth_range.npy`` -- max - min terrain height inside each cell (flags steps and walls).
* ``hits.npy``         -- subsampled ray hits in the world frame, (N, 5) float32: ``time, env, x, y, z``.
* ``map_meta.json``    -- grid, survey area and map settings.
* ``summary.json`` / ``summary.txt`` -- metrics (coverage, map error, path length, tracking error, ...).
* ``survey.mp4``       -- optional video.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mapping import GridSpec, path_length, survey_map_metrics, tracking_stats
from .paths import Area, cumulative_length, distance_to_polyline, polyline_length

TRAJECTORY_FIELDS = (
    "step",
    "time",
    "env",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll",
    "pitch",
    "yaw",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "speed",
    "carrot_x",
    "carrot_y",
    "carrot_z",
    "carrot_dist",
    "path_s",
    "progress",
    "path_index",
    "obs_dx",
    "obs_dy",
    "obs_dz",
    "gate",
    "agl",
)
"""Columns of ``trajectory.csv``: world-frame position [m], orientation quaternion (x, y, z, w) and Euler XYZ
[rad], world linear velocity [m/s], body angular velocity [rad/s], carrot target [m], carrot arclength [m],
path progress [0..1], dense path segment index, the (clamped) position offset the policy observed [m], carrot
speed gate [0..1] and height above ground [m]."""

_INT_FIELDS = {"step", "env", "path_index"}


def quat_xyzw_to_euler(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll, pitch, yaw (XYZ convention, radians) of quaternions in (x, y, z, w) order."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class TrajectoryRecorder:
    """Collects per-step rows (any subset of :data:`TRAJECTORY_FIELDS`; missing columns become NaN)."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []

    def record(self, **columns):
        n = None
        for v in columns.values():
            v = np.asarray(v)
            if v.ndim > 0:
                n = len(v)
                break
        if n is None or n == 0:
            return
        block = np.full((n, len(TRAJECTORY_FIELDS)), np.nan)
        for i, name in enumerate(TRAJECTORY_FIELDS):
            if name in columns:
                block[:, i] = np.broadcast_to(np.asarray(columns[name], dtype=np.float64), (n,))
        unknown = set(columns) - set(TRAJECTORY_FIELDS)
        if unknown:
            raise KeyError(f"Unknown trajectory fields: {sorted(unknown)}")
        self._chunks.append(block)

    def __len__(self) -> int:
        return sum(len(c) for c in self._chunks)

    def as_dict(self) -> dict[str, np.ndarray]:
        arr = np.concatenate(self._chunks) if self._chunks else np.zeros((0, len(TRAJECTORY_FIELDS)))
        return {name: arr[:, i] for i, name in enumerate(TRAJECTORY_FIELDS)}


def write_trajectory_csv(path: str | Path, traj: dict[str, np.ndarray]) -> None:
    """Write a trajectory dict (columns of equal length) to CSV with the standard header."""
    names = [n for n in TRAJECTORY_FIELDS if n in traj]
    n = len(traj[names[0]]) if names else 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        cols = [np.asarray(traj[k]) for k in names]
        for i in range(n):
            row = []
            for name, col in zip(names, cols):
                v = col[i]
                if name in _INT_FIELDS and np.isfinite(v):
                    row.append(str(int(v)))
                elif not np.isfinite(v):
                    row.append("nan")
                else:
                    row.append(f"{v:.5f}")
            w.writerow(row)


def read_trajectory_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``trajectory.csv`` into a dict of float arrays."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, len(header))
    return {name: arr[:, i] for i, name in enumerate(header)}


def write_planned_path_csv(path: str | Path, waypoints: list[np.ndarray]) -> None:
    """Write the planned waypoint polylines (one per drone) with their arclength."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "index", "x", "y", "z", "s"])
        for env_id, wp in enumerate(waypoints):
            s = cumulative_length(wp)
            for i, (p, si) in enumerate(zip(wp, s)):
                w.writerow([env_id, i, f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{si:.4f}"])


def read_planned_path_csv(path: str | Path) -> list[np.ndarray]:
    """Read ``planned_path.csv`` into a list of (N_i, 3) arrays, indexed by env."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [(int(r["env"]), int(r["index"]), float(r["x"]), float(r["y"]), float(r["z"])) for r in reader]
    if not rows:
        return []
    n_env = max(r[0] for r in rows) + 1
    out = []
    for e in range(n_env):
        pts = sorted((r for r in rows if r[0] == e), key=lambda r: r[1])
        out.append(np.asarray([r[2:] for r in pts], dtype=np.float64).reshape(-1, 3))
    return out


@dataclass
class SurveyRun:
    """Everything stored in a run directory."""

    run_dir: Path
    trajectory: dict[str, np.ndarray]
    planned: list[np.ndarray]
    grid: GridSpec
    area: Area
    elevation: np.ndarray
    count: np.ndarray
    ground_truth: np.ndarray
    ground_truth_range: np.ndarray | None
    hits: np.ndarray | None
    meta: dict
    summary: dict = field(default_factory=dict)

    def env_ids(self) -> list[int]:
        """Env indices present in the trajectory (or in the plan)."""
        if len(self.trajectory.get("env", [])):
            return sorted({int(e) for e in self.trajectory["env"]})
        return list(range(len(self.planned)))

    def env_trajectory(self, env_id: int) -> dict[str, np.ndarray]:
        """Trajectory columns of one drone, sorted by step."""
        mask = self.trajectory["env"] == env_id
        order = np.argsort(self.trajectory["step"][mask], kind="stable")
        return {k: v[mask][order] for k, v in self.trajectory.items()}


def save_run(
    run_dir: str | Path,
    *,
    trajectory: dict[str, np.ndarray],
    planned: list[np.ndarray],
    grid: GridSpec,
    area: Area,
    elevation: np.ndarray,
    count: np.ndarray,
    ground_truth: np.ndarray,
    ground_truth_range: np.ndarray | None,
    hits: np.ndarray | None,
    meta: dict,
    summary: dict,
) -> Path:
    """Write all run files; returns the run directory."""
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_trajectory_csv(run_dir / "trajectory.csv", trajectory)
    write_planned_path_csv(run_dir / "planned_path.csv", planned)
    np.save(run_dir / "elevation_map.npy", np.asarray(elevation, dtype=np.float32))
    np.save(run_dir / "hit_count.npy", np.asarray(count, dtype=np.int32))
    np.save(run_dir / "ground_truth.npy", np.asarray(ground_truth, dtype=np.float32))
    if ground_truth_range is not None:
        np.save(run_dir / "ground_truth_range.npy", np.asarray(ground_truth_range, dtype=np.float32))
    if hits is not None:
        np.save(run_dir / "hits.npy", np.asarray(hits, dtype=np.float32))
    meta = dict(meta)
    meta["grid"] = grid.to_dict()
    meta["area"] = list(area.as_tuple())
    (run_dir / "map_meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (run_dir / "summary.txt").write_text(format_summary(summary))
    return run_dir


def load_run(run_dir: str | Path) -> SurveyRun:
    """Load a run directory written by :func:`save_run`."""
    run_dir = Path(run_dir).expanduser()
    meta = json.loads((run_dir / "map_meta.json").read_text())
    summary_path = run_dir / "summary.json"
    rng_path = run_dir / "ground_truth_range.npy"
    hits_path = run_dir / "hits.npy"
    return SurveyRun(
        run_dir=run_dir,
        trajectory=read_trajectory_csv(run_dir / "trajectory.csv"),
        planned=read_planned_path_csv(run_dir / "planned_path.csv"),
        grid=GridSpec.from_dict(meta["grid"]),
        area=Area(*meta["area"]),
        elevation=np.load(run_dir / "elevation_map.npy"),
        count=np.load(run_dir / "hit_count.npy"),
        ground_truth=np.load(run_dir / "ground_truth.npy"),
        ground_truth_range=np.load(rng_path) if rng_path.exists() else None,
        hits=np.load(hits_path) if hits_path.exists() else None,
        meta=meta,
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {},
    )


"""
Summary metrics.
"""


def compute_summary(
    trajectory: dict[str, np.ndarray],
    planned: list[np.ndarray],
    *,
    grid: GridSpec,
    area: Area,
    elevation: np.ndarray,
    count: np.ndarray,
    ground_truth: np.ndarray,
    ground_truth_range: np.ndarray | None,
    commanded_speed: float | None = None,
    crashed: list[bool] | None = None,
    extra: dict | None = None,
    ground_truth_at_hits: np.ndarray | None = None,
) -> dict:
    """Coverage, map error, path length and tracking metrics of a survey run.

    The map is scored against the cell-mean ground truth (``error_area``, ``error_area_smooth``) and, when
    ``ground_truth_at_hits`` is given (:func:`drone_lab.mapping.truth_at_hits_map`), also against the true
    terrain under the same rays (``error_area_at_hits``: sensor error only).
    """
    envs = sorted({int(e) for e in trajectory["env"]}) if len(trajectory.get("env", [])) else []
    per_env = []
    all_ct = []
    for e in envs:
        m = trajectory["env"] == e
        order = np.argsort(trajectory["step"][m], kind="stable")
        pos = np.column_stack([trajectory["x"][m], trajectory["y"][m], trajectory["z"][m]])[order]
        t = trajectory["time"][m][order]
        plan = planned[e] if e < len(planned) else planned[0]
        track = tracking_stats(pos, plan)
        speed = trajectory["speed"][m][order]
        carrot = trajectory["carrot_dist"][m][order]
        agl = trajectory["agl"][m][order]
        progress = trajectory["progress"][m][order]
        obs = np.column_stack([trajectory["obs_dx"][m], trajectory["obs_dy"][m], trajectory["obs_dz"][m]])[order]
        clamped = np.mean(np.linalg.norm(obs, axis=1) > 0.999) if len(obs) else float("nan")
        entry = {
            "env": e,
            "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            "samples": int(len(t)),
            "path_length_planned_m": polyline_length(plan),
            "path_length_flown_m": path_length(pos),
            "completion": float(np.nanmax(progress)) if len(progress) else 0.0,
            "crashed": bool(crashed[e]) if crashed is not None and e < len(crashed) else False,
            "speed_mean": _nanstat(np.nanmean, speed),
            "speed_max": _nanstat(np.nanmax, speed),
            "agl_min": _nanstat(np.nanmin, agl),
            "agl_mean": _nanstat(np.nanmean, agl),
            "carrot_dist_mean": _nanstat(np.nanmean, carrot),
            "carrot_dist_max": _nanstat(np.nanmax, carrot),
            "obs_offset_clamped_fraction": float(clamped),
            **track,
        }
        per_env.append(entry)
        if len(pos):
            all_ct.append(distance_to_polyline(pos, plan, planar=True)[0])
    map_stats = survey_map_metrics(elevation, count, ground_truth, ground_truth_range, grid, area,
                                   truth_at_hits=ground_truth_at_hits)
    summary = {
        "num_drones": len(envs),
        "commanded_speed_mps": commanded_speed,
        "area": list(area.as_tuple()),
        "map": {
            "resolution_m": grid.resolution,
            "shape": list(grid.shape),
            "coverage_area": map_stats["coverage_area"],
            "coverage_grid": map_stats["coverage_grid"],
            "error_area": map_stats["error_area"],
            "error_area_at_hits": map_stats.get("error_area_at_hits"),
            "error_area_smooth": map_stats.get("error_area_smooth"),
            "smooth_cell_fraction": map_stats.get("smooth_cell_fraction"),
            "step_threshold_m": map_stats.get("step_threshold"),
            "hits_total": int(np.asarray(count).sum()),
        },
        "flight": {
            "duration_s": max((p["duration_s"] for p in per_env), default=0.0),
            "path_length_flown_m": sum(p["path_length_flown_m"] for p in per_env),
            "path_length_planned_m": sum(p["path_length_planned_m"] for p in per_env),
            "completion_min": min((p["completion"] for p in per_env), default=0.0),
            "any_crash": any(p["crashed"] for p in per_env),
        },
        "tracking": {},
        "per_drone": per_env,
    }
    if all_ct:
        ct = np.concatenate(all_ct)
        summary["tracking"] = {
            "cross_track_mean_m": float(ct.mean()),
            "cross_track_rms_m": float(np.sqrt(np.mean(ct**2))),
            "cross_track_p95_m": float(np.percentile(ct, 95)),
            "cross_track_max_m": float(ct.max()),
            "carrot_dist_mean_m": float(np.nanmean(trajectory["carrot_dist"])),
            "carrot_dist_max_m": float(np.nanmax(trajectory["carrot_dist"])),
        }
    if extra:
        summary.update(extra)
    return summary


def _nanstat(fn, a) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0 or not np.any(np.isfinite(a)):
        return None
    return float(fn(a))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fmt(v, unit: str = "", digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{round(float(v), digits) + 0.0:.{digits}f}{unit}"  # + 0.0: no "-0.000"


def format_summary(summary: dict) -> str:
    """Short human-readable summary of :func:`compute_summary` output."""
    m = summary.get("map", {})
    err = m.get("error_area") or {}
    smooth = m.get("error_area_smooth") or {}
    at_hits = m.get("error_area_at_hits") or {}
    fl = summary.get("flight", {})
    tr = summary.get("tracking", {})
    lines = [f"Survey summary ({summary.get('source', 'run')})"]
    if summary.get("interrupted"):
        lines.append(f"  PARTIAL RUN        : {summary['interrupted']} (data up to the interruption)")
    lines += [
        f"  pattern            : {summary.get('pattern', 'n/a')}, {summary.get('num_drones', 0)} drone(s)",
        f"  flight altitude    : {_fmt(summary.get('flight_z_m'), ' m')} (terrain max "
        f"{_fmt(summary.get('terrain_z_max_m'), ' m')} + clearance {_fmt(summary.get('altitude_clearance_m'), ' m')})",
        f"  duration           : {_fmt(fl.get('duration_s'), ' s', 1)}",
        f"  path planned/flown : {_fmt(fl.get('path_length_planned_m'), ' m', 1)} / "
        f"{_fmt(fl.get('path_length_flown_m'), ' m', 1)}",
        f"  completion (min)   : {_fmt(100 * fl['completion_min'] if 'completion_min' in fl else None, ' %', 1)}"
        f"   crashed: {_fmt(fl.get('any_crash'))}",
        f"  coverage of area   : {_fmt(100 * m['coverage_area'] if 'coverage_area' in m else None, ' %', 1)}"
        f"   ({_fmt(m.get('resolution_m'), ' m')} cells, {m.get('hits_total', 0):,} hits)",
        f"  map error          : RMSE {_fmt(err.get('rmse'), ' m', 3)}, MAE {_fmt(err.get('mae'), ' m', 3)}, "
        f"max {_fmt(err.get('max_abs'), ' m', 2)}, bias {_fmt(err.get('bias'), ' m', 3)}",
        f"  map error (smooth) : RMSE {_fmt(smooth.get('rmse'), ' m', 3)} on "
        f"{_fmt(100 * m['smooth_cell_fraction'] if m.get('smooth_cell_fraction') is not None else None, ' %', 0)}"
        " of the cells (no step/wall inside the cell)",
    ]
    if at_hits:
        lines.append(
            f"  map error (sensor) : RMSE {_fmt(at_hits.get('rmse'), ' m', 3)}, "
            f"bias {_fmt(at_hits.get('bias'), ' m', 3)} (vs the true terrain under the same rays)"
        )
    lines += [
        f"  cross-track error  : mean {_fmt(tr.get('cross_track_mean_m'), ' m')}, RMS "
        f"{_fmt(tr.get('cross_track_rms_m'), ' m')}, max {_fmt(tr.get('cross_track_max_m'), ' m')}",
        f"  drone-carrot dist  : mean {_fmt(tr.get('carrot_dist_mean_m'), ' m')}, max "
        f"{_fmt(tr.get('carrot_dist_max_m'), ' m')}",
    ]
    return "\n".join(lines) + "\n"

