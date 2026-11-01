"""Survey path generators and polyline geometry (numpy only).

Everything in this module is plain numpy so it can be unit tested and used by the CPU-side tools
(synthetic survey, plotting) without Isaac Sim.

Conventions:
    * Points are ``(N, 3)`` float arrays in the world frame [m]; z is the absolute flight height.
    * An :class:`Area` is an axis-aligned rectangle in the world xy-plane.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PATTERNS = ("lawnmower", "spiral", "figure8", "waypoints")
"""Survey patterns understood by :func:`build_env_paths`."""


@dataclass(frozen=True)
class Area:
    """Axis-aligned survey rectangle in the world xy-plane [m]."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self):
        if not (self.x_max > self.x_min and self.y_max > self.y_min):
            raise ValueError(f"Degenerate area: {self}")

    @property
    def width(self) -> float:
        """Extent along x [m]."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Extent along y [m]."""
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        """Center (x, y) [m]."""
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    def inset(self, margin: float) -> Area:
        """Shrink the area by ``margin`` on every side (negative values grow it)."""
        return Area(self.x_min + margin, self.x_max - margin, self.y_min + margin, self.y_max - margin)

    def split(self, n: int, axis: str = "y") -> list[Area]:
        """Split the area into ``n`` equal strips along ``axis`` ("x" or "y")."""
        if n < 1:
            raise ValueError("n must be >= 1")
        if axis == "y":
            edges = np.linspace(self.y_min, self.y_max, n + 1)
            return [Area(self.x_min, self.x_max, float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]
        if axis == "x":
            edges = np.linspace(self.x_min, self.x_max, n + 1)
            return [Area(float(a), float(b), self.y_min, self.y_max) for a, b in zip(edges[:-1], edges[1:])]
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def contains(self, xy: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Boolean mask of points (..., 2+) whose xy lies inside the area (with tolerance)."""
        xy = np.asarray(xy)
        return (
            (xy[..., 0] >= self.x_min - tol)
            & (xy[..., 0] <= self.x_max + tol)
            & (xy[..., 1] >= self.y_min - tol)
            & (xy[..., 1] <= self.y_max + tol)
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max)."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)


"""
Pattern generators.
"""


def lawnmower(
    area: Area,
    spacing: float,
    altitude: float,
    *,
    direction: str = "x",
    turn: str = "round",
    points_per_turn: int = 16,
) -> np.ndarray:
    """Boustrophedon ("lawnmower") survey lines that cover ``area``.

    The area is cut into ``n = ceil(extent / spacing)`` swaths of equal width ``extent / n`` (<= ``spacing``)
    and one survey line runs through the middle of each swath, from one edge of the area to the other.
    A sensor whose footprint is at least ``spacing`` wide therefore covers the whole area.

    Args:
        area: Survey rectangle.
        spacing: Maximum distance between neighbouring lines [m].
        altitude: Absolute flight height z [m].
        direction: Axis the lines run along: "x" (lines step in y) or "y" (lines step in x).
        turn: "round" for semicircular U-turns outside the area (smoother for the drone) or
            "square" for straight connectors along the area edge.
        points_per_turn: Number of points on each semicircular turn.

    Returns:
        Waypoints, shape (N, 3).
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    if turn not in ("round", "square"):
        raise ValueError(f"turn must be 'round' or 'square', got {turn!r}")
    if direction not in ("x", "y"):
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")
    # work in a (u, v) frame: u along the lines, v across them
    if direction == "x":
        u_min, u_max, v_min, v_max = area.x_min, area.x_max, area.y_min, area.y_max
    else:
        u_min, u_max, v_min, v_max = area.y_min, area.y_max, area.x_min, area.x_max
    extent = v_max - v_min
    n_lines = max(1, math.ceil(extent / spacing - 1e-9))
    eff = extent / n_lines
    v_lines = v_min + eff * (np.arange(n_lines) + 0.5)

    pts: list[tuple[float, float]] = []
    for i, v in enumerate(v_lines):
        forward = i % 2 == 0
        u0, u1 = (u_min, u_max) if forward else (u_max, u_min)
        pts.append((u0, v))
        pts.append((u1, v))
        if i + 1 < n_lines:
            v_next = v_lines[i + 1]
            if turn == "round":
                # semicircle of radius eff/2 centred between the two lines, bulging outwards
                r = 0.5 * (v_next - v)
                vc = 0.5 * (v + v_next)
                sign = 1.0 if forward else -1.0  # bulge beyond u_max (forward) or u_min (backward)
                angles = np.linspace(-0.5 * math.pi, 0.5 * math.pi, points_per_turn + 2)[1:-1]
                for a in angles:
                    pts.append((u1 + sign * r * math.cos(a), vc + r * math.sin(a)))
            # square: next line start (u1, v_next) is appended by the next iteration
    uv = np.asarray(pts, dtype=np.float64)
    xy = uv if direction == "x" else uv[:, ::-1]
    return np.column_stack([xy, np.full(len(xy), float(altitude))])


def spiral(area: Area, spacing: float, altitude: float, *, inward: bool = True) -> np.ndarray:
    """Rectangular spiral ("expanding/contracting square") that covers ``area``.

    Concentric rectangles, ``spacing`` apart, joined into one continuous polyline. The outer ring runs
    ``spacing / 2`` inside the area boundary.

    Args:
        area: Survey rectangle.
        spacing: Distance between neighbouring rings [m].
        altitude: Absolute flight height z [m].
        inward: Start on the outer ring and spiral to the centre (True) or the reverse.

    Returns:
        Waypoints, shape (N, 3).
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    eps = 1e-6
    x0, x1 = area.x_min + 0.5 * spacing, area.x_max - 0.5 * spacing
    y0, y1 = area.y_min + 0.5 * spacing, area.y_max - 0.5 * spacing
    # tiny areas: collapse to the centre line
    if x1 < x0:
        x0 = x1 = 0.5 * (area.x_min + area.x_max)
    if y1 < y0:
        y0 = y1 = 0.5 * (area.y_min + area.y_max)
    pts: list[tuple[float, float]] = [(x0, y0)]
    while True:
        w, h = x1 - x0, y1 - y0
        if w <= eps and h <= eps:  # collapsed to a point
            break
        if h <= eps:  # remaining region is a horizontal line
            pts.append((x1, y0))
            break
        if w <= eps:  # remaining region is a vertical line
            pts.append((x0, y1))
            break
        # one ring: bottom edge, right edge, top edge
        pts += [(x1, y0), (x1, y1), (x0, y1)]
        # next ring, one spacing inside (collapse to the centre line if the ring would cross itself)
        x0n, x1n = x0 + spacing, x1 - spacing
        if x1n < x0n:
            x0n = x1n = 0.5 * (x0 + x1)
        y0n, y1n = y0 + spacing, y1 - spacing
        if y1n < y0n:
            y0n = y1n = 0.5 * (y0 + y1)
        # left edge down to the next ring's bottom edge, then step inwards
        pts += [(x0, y0n), (x0n, y0n)]
        x0, x1, y0, y1 = x0n, x1n, y0n, y1n
    xy = np.asarray(pts, dtype=np.float64)
    # drop consecutive duplicates
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9
    xy = xy[keep]
    if not inward:
        xy = xy[::-1]
    return np.column_stack([xy, np.full(len(xy), float(altitude))])


def figure8(
    area: Area, altitude: float, *, loops: int = 1, points_per_loop: int = 240, fill: float = 0.9
) -> np.ndarray:
    """Figure-eight (lemniscate of Gerono) scaled to ``area``.

    ``x = cx + a sin(t)``, ``y = cy + b sin(t) cos(t)`` with ``t`` in ``[0, 2 pi * loops]``. The curve is
    closed; the first and last point coincide.

    Args:
        area: Bounding rectangle.
        altitude: Absolute flight height z [m].
        loops: Number of times the eight is flown.
        points_per_loop: Sampling of one loop.
        fill: Fraction of the half-extents used by the curve.
    """
    cx, cy = area.center
    a = 0.5 * area.width * fill
    b = area.height * fill  # max of |sin t cos t| is 1/2
    t = np.linspace(0.0, 2.0 * math.pi * loops, points_per_loop * loops + 1)
    x = cx + a * np.sin(t)
    y = cy + b * np.sin(t) * np.cos(t)
    return np.column_stack([x, y, np.full(len(t), float(altitude))])


def resolve_pattern(pattern: str | None, waypoints_given: bool, default: str | None = "lawnmower") -> str | None:
    """Survey pattern from the ``--pattern`` / ``--waypoints`` command-line flags.

    A waypoint file implies ``"waypoints"``. Combining it with a generated pattern is an error: the generated path
    would silently ignore the file.

    Args:
        pattern: Value of ``--pattern`` (None when not given).
        waypoints_given: Whether ``--waypoints`` was given.
        default: Pattern when neither flag is given (None: leave the task config's pattern alone).

    Raises:
        ValueError: ``--waypoints`` with another ``--pattern``, or ``--pattern waypoints`` without a file.
    """
    if waypoints_given and pattern not in (None, "waypoints"):
        raise ValueError(
            f"--waypoints needs --pattern waypoints (or no --pattern), not --pattern {pattern}: "
            "a generated pattern would ignore the file"
        )
    resolved = "waypoints" if waypoints_given else (pattern or default)
    if resolved == "waypoints" and not waypoints_given:
        raise ValueError("--pattern waypoints needs --waypoints <file.csv|file.json>")
    return resolved


def read_waypoint_rows(path: str | Path) -> list[list[float]]:
    """Raw waypoint rows from a ``.csv`` (columns x,y[,z], optional header) or ``.json`` (list of [x, y(, z)])."""
    path = Path(path).expanduser()
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get("waypoints", rows.get("points"))
        return [[float(v) for v in row] for row in rows]
    rows = []
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                rows.append([float(v) for v in row if v.strip() != ""])
            except ValueError:
                continue  # header
    return rows


def load_waypoints(path: str | Path, default_z: float | None = None) -> np.ndarray:
    """Load custom waypoints (see :func:`read_waypoint_rows`); rows without z use ``default_z``."""
    return waypoints_array(read_waypoint_rows(path), default_z)


def waypoints_array(rows, default_z: float | None = None) -> np.ndarray:
    """Convert a list of [x, y] / [x, y, z] rows to an (N, 3) array, filling z with ``default_z``."""
    out = []
    for row in rows:
        row = [float(v) for v in row]
        if len(row) == 2:
            if default_z is None:
                raise ValueError("Waypoint without z and no default altitude given.")
            row = [row[0], row[1], float(default_z)]
        elif len(row) != 3:
            raise ValueError(f"Waypoints need 2 or 3 values, got {row}")
        out.append(row)
    if len(out) < 2:
        raise ValueError("At least two waypoints are needed.")
    return np.asarray(out, dtype=np.float64)


def build_env_paths(
    pattern: str,
    num_envs: int,
    area: Area,
    spacing: float,
    altitude: float,
    *,
    waypoints: np.ndarray | None = None,
    multi_env_mode: str = "split",
    direction: str = "x",
    turn: str = "round",
    figure8_loops: int = 1,
) -> list[np.ndarray]:
    """Waypoint polyline for every environment (drone).

    With several drones, ``multi_env_mode="split"`` gives each drone its own strip of the area for the
    area-covering patterns (lawnmower, spiral) and staggers the start of closed curves (figure8) by
    ``1/(2n)`` of the loop;
    ``"copy"`` gives every drone the same path.

    Args:
        pattern: One of :data:`PATTERNS`.
        num_envs: Number of drones.
        area: Survey rectangle.
        spacing: Line / ring spacing [m] for lawnmower and spiral.
        altitude: Absolute flight height z [m] (custom waypoints keep their own z).
        waypoints: (N, 3) array for ``pattern="waypoints"``.
        multi_env_mode: "split" or "copy".
        direction: Lawnmower line direction ("x" or "y").
        turn: Lawnmower turn style ("round" or "square").
        figure8_loops: Number of figure-eight loops.

    Returns:
        List of ``num_envs`` arrays of shape (N_i, 3).
    """
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern {pattern!r}; choose from {PATTERNS}")
    if multi_env_mode not in ("split", "copy"):
        raise ValueError(f"multi_env_mode must be 'split' or 'copy', got {multi_env_mode!r}")
    split = multi_env_mode == "split" and num_envs > 1
    if pattern in ("lawnmower", "spiral"):
        # strips across the line direction keep every line full length
        strips = area.split(num_envs, axis="y" if direction == "x" else "x") if split else [area] * num_envs
        if pattern == "lawnmower":
            return [lawnmower(a, spacing, altitude, direction=direction, turn=turn) for a in strips]
        return [spiral(a, spacing, altitude) for a in strips]
    if pattern == "figure8":
        base = figure8(area, altitude, loops=figure8_loops)
    else:
        if waypoints is None:
            raise ValueError("pattern='waypoints' needs a waypoint array")
        base = np.asarray(waypoints, dtype=np.float64)
    if not split:
        return [base.copy() for _ in range(num_envs)]
    closed = np.linalg.norm(base[0] - base[-1]) < 1e-6
    if not closed:
        return [base.copy() for _ in range(num_envs)]
    # stagger the start of a closed loop by 1/(2n) of its length: with half-loop offsets, two drones on a
    # figure eight would reach its crossing point at the same time
    ring = base[:-1]
    shifts = [int(round(i * len(ring) / (2 * num_envs))) for i in range(num_envs)]
    paths = []
    for k in shifts:
        rolled = np.roll(ring, -k, axis=0)
        paths.append(np.vstack([rolled, rolled[:1]]))
    return paths


"""
Polyline geometry.
"""


def cumulative_length(points: np.ndarray) -> np.ndarray:
    """Cumulative arclength at every vertex of a polyline, shape (N,), starting at 0."""
    points = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_length(points: np.ndarray) -> float:
    """Total length of a polyline [m]."""
    return float(cumulative_length(points)[-1]) if len(points) > 1 else 0.0


def resample_polyline(points: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Densify a polyline so consecutive points are at most ``ds`` apart.

    Every original vertex is kept (corners are not cut) and each segment is split into equal parts; duplicate
    consecutive vertices are removed. The length of the polyline is preserved exactly.

    Returns:
        ``(resampled_points (M, 3), arclength (M,))``.
    """
    if ds <= 0:
        raise ValueError("ds must be > 0")
    points = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9
    points = points[keep]
    if len(points) < 2:
        return points.copy(), np.zeros(len(points))
    seg = np.diff(points, axis=0)
    n = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / ds - 1e-9)).astype(np.int64)
    idx = np.repeat(np.arange(len(seg)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1  # 1..n_i within each segment
    out = np.vstack([points[:1], points[idx] + (k / n[idx])[:, None] * seg[idx]])
    return out, cumulative_length(out)


def point_at_arclength(points: np.ndarray, s_cum: np.ndarray, s: np.ndarray | float) -> np.ndarray:
    """Linear interpolation of a polyline at arclength(s) ``s`` (clamped to the ends)."""
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, s_cum[-1])
    return np.stack([np.interp(s, s_cum, points[:, k]) for k in range(points.shape[1])], axis=-1)


def distance_to_polyline(query: np.ndarray, polyline: np.ndarray, *, planar: bool = False, chunk: int = 4096):
    """Shortest distance from every query point to a polyline.

    Args:
        query: Points, shape (Q, 3) (or (Q, 2)).
        polyline: Vertices, shape (N, 3) (or (N, 2)), N >= 1.
        planar: Only use x and y (horizontal cross-track distance).
        chunk: Number of query points processed at once.

    Returns:
        ``(distance (Q,), segment_index (Q,))``.
    """
    q = np.asarray(query, dtype=np.float64)
    p = np.asarray(polyline, dtype=np.float64)
    dims = 2 if planar else min(q.shape[1], p.shape[1])
    q = q[:, :dims]
    p = p[:, :dims]
    if len(p) == 1:
        d = np.linalg.norm(q - p[0], axis=1)
        return d, np.zeros(len(q), dtype=np.int64)
    a = p[:-1]
    ab = p[1:] - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
    dist = np.empty(len(q))
    idx = np.empty(len(q), dtype=np.int64)
    for start in range(0, len(q), chunk):
        qq = q[start : start + chunk]
        ap = qq[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("qsd,sd->qs", ap, ab) / ab2[None, :], 0.0, 1.0)
        closest = a[None, :, :] + t[..., None] * ab[None, :, :]
        d2 = np.sum((qq[:, None, :] - closest) ** 2, axis=-1)
        k = np.argmin(d2, axis=1)
        idx[start : start + chunk] = k
        dist[start : start + chunk] = np.sqrt(d2[np.arange(len(qq)), k])
    return dist, idx
