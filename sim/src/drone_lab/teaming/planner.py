"""Cost-weighted 8-connected A* on a traversability cost grid, with start/goal snapping and path smoothing.

* **Graph:** every free cell (finite cost) connects to its 8 neighbours. A move costs its length times the mean
  cost of the two cells; diagonal moves may not cut the corner of a lethal cell. Without corner cutting, two
  free cells are connected exactly when they are 4-connected, which is what :func:`reachable_mask` uses.
* **A*:** octile-distance heuristic scaled by the cheapest cell cost, which is admissible and consistent, so the
  returned path is optimal on the graph (the tests check it against Dijkstra).
* **Snapping:** a start or goal on a lethal cell (inside an inflated obstacle, e.g. the person the robot is sent
  to) moves to the nearest free cell that is reachable from the start, so the robot stops next to the person
  instead of on top of them.
* **Smoothing:** line-of-sight shortcuts that never enter a lethal cell and never cost more than the stretch of
  the A* path they replace (plus a small tolerance), then a corner-rounding pass (Chaikin) that is kept only where
  it stays on free cells; finally the path is resampled at a fixed spacing for the follower.

Cells are ``(iy, ix)``; world points are ``(x, y)`` [m]. numpy + scipy only.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .traversability import TraversabilityMap

__all__ = [
    "PlanResult",
    "astar",
    "path_cost",
    "plan_path",
    "polyline_length",
    "reachable_mask",
    "resample_polyline",
    "segment_cost",
    "shortcut_path",
    "smooth_path",
    "snap_to_free",
]

SQRT2 = math.sqrt(2.0)


@dataclass
class PlanResult:
    """Outcome of :func:`plan_path`."""

    success: bool
    message: str
    start_xy: np.ndarray
    """Requested start [m]."""
    goal_xy: np.ndarray
    """Requested goal [m]."""
    start_snapped_xy: np.ndarray | None = None
    goal_snapped_xy: np.ndarray | None = None
    start_snap_m: float = 0.0
    """How far the start was moved to reach a free cell [m]."""
    goal_snap_m: float = 0.0
    """Distance between the requested goal and the reachable cell the robot is sent to [m]."""
    cells: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    """A* path, cells (K, 2) as (iy, ix)."""
    raw_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    """A* path through the cell centres [m]."""
    shortcut_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    """After line-of-sight shortcuts [m]."""
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    """Final smoothed path, resampled at a fixed spacing [m] (what the robot follows)."""
    path_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Terrain height along :attr:`path_xy` [m]."""
    graph_cost: float = math.inf
    """A* path cost on the grid graph (metres weighted by cell cost)."""
    path_cost: float = math.inf
    """Cost of the final path, integrated along it [m-equivalents]."""
    expanded: int = 0
    """Nodes expanded by A*."""
    failure: str = ""
    """Why planning failed (empty on success): ``start_outside`` / ``start_blocked`` (the start's problem, see
    :attr:`start_failed`), ``goal_outside``, ``unreachable`` (the closest reachable cell is farther from the goal
    than allowed) or ``no_path``."""

    @property
    def start_failed(self) -> bool:
        """The failure lies with the start (off the map, or no free cell near it): no other goal would work either."""
        return self.failure in ("start_outside", "start_blocked")

    @property
    def length_m(self) -> float:
        return polyline_length(self.path_xy)

    @property
    def raw_length_m(self) -> float:
        return polyline_length(self.raw_xy)

    def to_dict(self) -> dict:
        """JSON-friendly summary (the path itself is stored separately)."""

        def pt(a):
            return None if a is None else [round(float(v), 4) for v in np.asarray(a).ravel()[:2]]

        return {
            "success": self.success,
            "message": self.message,
            "failure": self.failure,
            "start_xy": pt(self.start_xy),
            "goal_xy": pt(self.goal_xy),
            "start_snapped_xy": pt(self.start_snapped_xy),
            "goal_snapped_xy": pt(self.goal_snapped_xy),
            "start_snap_m": round(self.start_snap_m, 3),
            "goal_snap_m": round(self.goal_snap_m, 3),
            "length_m": round(self.length_m, 3),
            "raw_length_m": round(self.raw_length_m, 3),
            "graph_cost": None if not math.isfinite(self.graph_cost) else round(self.graph_cost, 3),
            "path_cost": None if not math.isfinite(self.path_cost) else round(self.path_cost, 3),
            "num_points": int(len(self.path_xy)),
            "expanded": int(self.expanded),
        }


# ------------------------------------------------------------------------------------------------------ A*
def astar(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    resolution: float = 1.0,
    heuristic_weight: float = 1.0,
) -> tuple[np.ndarray, float, int]:
    """Cost-weighted 8-connected A* from ``start`` to ``goal`` (cells ``(iy, ix)``).

    A move between neighbours ``u -> v`` costs ``length * (cost[u] + cost[v]) / 2`` with length ``resolution``
    (axis) or ``sqrt(2) * resolution`` (diagonal). Cells with a non-finite cost are obstacles; a diagonal move is
    only allowed when both cells it squeezes past are free. ``heuristic_weight`` > 1 gives weighted A* (faster,
    no longer optimal).

    Returns:
        ``(path (K, 2) int cells from start to goal, path cost, nodes expanded)``; an empty path and ``inf`` when
        the goal cannot be reached.
    """
    cost = np.asarray(cost, dtype=float)
    ny, nx = cost.shape
    sy, sx = int(start[0]), int(start[1])
    gy, gx = int(goal[0]), int(goal[1])
    for (cy, cx), name in (((sy, sx), "start"), ((gy, gx), "goal")):
        if not (0 <= cy < ny and 0 <= cx < nx):
            raise ValueError(f"{name} cell {(cy, cx)} is outside the {ny}x{nx} grid")
    empty = np.zeros((0, 2), dtype=int)
    if not (np.isfinite(cost[sy, sx]) and np.isfinite(cost[gy, gx])):
        return empty, math.inf, 0
    finite = cost[np.isfinite(cost)]
    cmin = float(finite.min()) if finite.size else 1.0
    hscale = max(cmin, 0.0) * resolution * float(heuristic_weight)

    # padded flat layout: one ring of obstacles around the grid removes all bounds checks
    w = nx + 2
    padded = np.full((ny + 2, w), np.inf)
    padded[1:-1, 1:-1] = cost
    c = padded.ravel().tolist()
    n = len(c)
    free = [math.isfinite(v) for v in c]
    s = (sy + 1) * w + (sx + 1)
    g = (gy + 1) * w + (gx + 1)
    if s == g:
        return np.array([[sy, sx]], dtype=int), 0.0, 1

    half_axis = 0.5 * resolution
    half_diag = 0.5 * SQRT2 * resolution
    axis_moves = (1, -1, w, -w)
    # diagonal move and the two axis neighbours it squeezes past
    diag_moves = ((w + 1, w, 1), (w - 1, w, -1), (-w + 1, -w, 1), (-w - 1, -w, -1))
    gr, gc = divmod(g, w)
    k_diag = SQRT2 - 2.0

    def h(i: int) -> float:
        r, col = divmod(i, w)
        dy = abs(r - gr)
        dx = abs(col - gc)
        return hscale * (dx + dy + k_diag * (dx if dx < dy else dy))

    g_score = [math.inf] * n
    parent = [-1] * n
    closed = bytearray(n)
    g_score[s] = 0.0
    heap = [(h(s), 0.0, s)]
    expanded = 0
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        f, gs, u = pop(heap)
        if closed[u]:
            continue
        if gs > g_score[u]:
            continue
        closed[u] = 1
        expanded += 1
        if u == g:
            break
        cu = c[u]
        for d in axis_moves:
            v = u + d
            if not free[v] or closed[v]:
                continue
            ng = gs + half_axis * (cu + c[v])
            if ng < g_score[v]:
                g_score[v] = ng
                parent[v] = u
                push(heap, (ng + h(v), ng, v))
        for d, a1, a2 in diag_moves:
            v = u + d
            if not free[v] or closed[v] or not free[u + a1] or not free[u + a2]:
                continue
            ng = gs + half_diag * (cu + c[v])
            if ng < g_score[v]:
                g_score[v] = ng
                parent[v] = u
                push(heap, (ng + h(v), ng, v))
    if not closed[g]:
        return empty, math.inf, expanded
    path = []
    u = g
    while u != -1:
        r, col = divmod(u, w)
        path.append((r - 1, col - 1))
        u = parent[u]
    path.reverse()
    return np.asarray(path, dtype=int), float(g_score[g]), expanded


# ---------------------------------------------------------------------------------------------- snapping
def reachable_mask(cost: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Free cells reachable from ``start`` with the A* move rules (4-connectivity of the free cells)."""
    free = np.isfinite(cost)
    if not free[int(start[0]), int(start[1])]:
        return np.zeros_like(free)
    labels, _ = ndimage.label(free, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    return labels == labels[int(start[0]), int(start[1])]


def snap_to_free(allowed: np.ndarray, cell: tuple[int, int]) -> tuple[tuple[int, int] | None, float]:
    """Nearest ``allowed`` cell to ``cell`` (Euclidean, in cells); ``(None, inf)`` if no cell is allowed."""
    allowed = np.asarray(allowed, dtype=bool)
    iy, ix = int(cell[0]), int(cell[1])
    if allowed[iy, ix]:
        return (iy, ix), 0.0
    if not allowed.any():
        return None, math.inf
    _, (ny_idx, nx_idx) = ndimage.distance_transform_edt(~allowed, return_indices=True)
    ty, tx = int(ny_idx[iy, ix]), int(nx_idx[iy, ix])
    return (ty, tx), math.hypot(ty - iy, tx - ix)


# ----------------------------------------------------------------------------------------- path geometry
def polyline_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0


def resample_polyline(xy: np.ndarray, spacing: float) -> np.ndarray:
    """Points every ``spacing`` metres along the polyline, always including both end points."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 2:
        return xy.copy()
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    xy = xy[keep]
    if len(xy) < 2:
        return xy.copy()
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    n = max(1, int(math.ceil(s[-1] / max(spacing, 1e-6))))
    t = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])


def _densify(xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points along the polyline at most ``step`` apart, and the length each point stands for."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 2:
        return xy.copy(), np.zeros(len(xy))
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:], strict=False):
        L = float(np.linalg.norm(b - a))
        k = max(1, int(math.ceil(L / step)))
        t = (np.arange(1, k + 1) / k)[:, None]
        pts.append(a + t * (b - a))
    p = np.concatenate(pts)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    weights = np.zeros(len(p))
    weights[:-1] += 0.5 * d
    weights[1:] += 0.5 * d
    return p, weights


def path_cost(tmap: TraversabilityMap, xy: np.ndarray) -> float:
    """Cost integrated along a polyline (nearest-cell cost x length); ``inf`` if it touches a lethal cell or
    leaves the grid."""
    p, wts = _densify(xy, 0.25 * tmap.grid.resolution)
    if len(p) == 0:
        return math.inf
    c = tmap.sample(tmap.cost, p, outside=np.inf)
    if not np.all(np.isfinite(c)):
        return math.inf
    return float(np.sum(c * wts))


def segment_cost(tmap: TraversabilityMap, a: np.ndarray, b: np.ndarray) -> float:
    """:func:`path_cost` of the straight segment ``a -> b``."""
    return path_cost(tmap, np.vstack([a, b]))


def shortcut_path(
    tmap: TraversabilityMap, xy: np.ndarray, *, tolerance: float = 0.02, max_shortcut_m: float = 8.0
) -> np.ndarray:
    """Greedy line-of-sight shortcutting ("string pulling") that never raises the integrated cost.

    From each anchor the farthest later vertex is taken whose straight segment stays on free cells and costs at
    most ``(1 + tolerance)`` times the stretch of the current path it replaces. Segments are capped at
    ``max_shortcut_m`` to keep the check cheap and the path close to the planned corridor.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) <= 2:
        return xy.copy()
    # cumulative cost along the input path, per vertex
    cum = [0.0]
    for a, b in zip(xy[:-1], xy[1:], strict=False):
        cum.append(cum[-1] + segment_cost(tmap, a, b))
    cum = np.asarray(cum)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    out = [xy[0]]
    i = 0
    n = len(xy)
    while i < n - 1:
        best = i + 1
        # farthest first; stop at the first success
        j_max = int(np.searchsorted(s, s[i] + max_shortcut_m, side="right")) - 1
        j_max = min(max(j_max, i + 1), n - 1)
        for j in range(j_max, i + 1, -1):
            seg = segment_cost(tmap, xy[i], xy[j])
            if math.isfinite(seg) and seg <= (1.0 + tolerance) * (cum[j] - cum[i]) + 1e-9:
                best = j
                break
        out.append(xy[best])
        i = best
    return np.asarray(out)


def _chaikin(xy: np.ndarray, iterations: int) -> np.ndarray:
    p = np.asarray(xy, dtype=float)
    for _ in range(iterations):
        if len(p) < 3:
            break
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * (len(p) - 1), 2))
        mid[0::2] = q
        mid[1::2] = r
        p = np.vstack([p[:1], mid[1:-1], p[-1:]])
    return p


def smooth_path(
    tmap: TraversabilityMap,
    xy: np.ndarray,
    *,
    spacing: float = 0.1,
    chaikin_iterations: int = 3,
    tolerance: float = 0.02,
    max_shortcut_m: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Shortcut, round the corners where it is safe, resample every ``spacing`` metres.

    Returns:
        ``(shortcut polyline, final resampled path)``. Corner rounding is applied per corner and only kept when
        the rounded piece stays on free cells; otherwise the sharp corner is kept.
    """
    sc = shortcut_path(tmap, xy, tolerance=tolerance, max_shortcut_m=max_shortcut_m)
    if len(sc) < 3 or chaikin_iterations <= 0:
        return sc, resample_polyline(sc, spacing)
    rounded = _chaikin(sc, chaikin_iterations)
    if math.isfinite(path_cost(tmap, rounded)):
        return sc, resample_polyline(rounded, spacing)
    # round each corner separately; keep only the safe ones
    pieces = [sc[:1]]
    for k in range(1, len(sc) - 1):
        a, b, c = sc[k - 1], sc[k], sc[k + 1]
        # local corner between the midpoints of its two legs
        m1, m2 = 0.5 * (a + b), 0.5 * (b + c)
        local = _chaikin(np.vstack([m1, b, m2]), chaikin_iterations)
        if math.isfinite(path_cost(tmap, local)):
            pieces.append(local)
        else:
            pieces.append(np.vstack([m1, b, m2]))
    pieces.append(sc[-1:])
    merged = np.vstack(pieces)
    if not math.isfinite(path_cost(tmap, merged)):
        merged = sc
    return sc, resample_polyline(merged, spacing)


# ---------------------------------------------------------------------------------------------- top level
def plan_path(
    tmap: TraversabilityMap,
    start_xy,
    goal_xy,
    *,
    max_start_snap_m: float = 2.0,
    max_goal_snap_m: float = 6.0,
    smooth: bool = True,
    spacing: float = 0.1,
    heuristic_weight: float = 1.0,
) -> PlanResult:
    """Plan from ``start_xy`` to ``goal_xy`` (world, [m]) on ``tmap``.

    The start snaps to the nearest free cell (at most ``max_start_snap_m`` away). The goal snaps to the nearest
    free cell that is reachable from the start (at most ``max_goal_snap_m`` away): when the goal is a person, the
    inflated footprint makes the person's own cells lethal, so the robot is sent to the closest safe spot next to
    them. ``goal_snap_m`` reports how far that spot is from the requested goal.
    """
    start_xy = np.asarray(start_xy, dtype=float).ravel()[:2]
    goal_xy = np.asarray(goal_xy, dtype=float).ravel()[:2]
    res = PlanResult(success=False, message="", start_xy=start_xy, goal_xy=goal_xy)
    grid = tmap.grid
    free = np.isfinite(tmap.cost)
    iy, ix, inside = tmap.world_to_cell(np.array([start_xy[0], goal_xy[0]]), np.array([start_xy[1], goal_xy[1]]))
    if not inside[0]:
        res.message = f"start {start_xy.tolist()} is outside the map {grid.extent}"
        res.failure = "start_outside"
        return res
    if not inside[1]:
        res.message = f"goal {goal_xy.tolist()} is outside the map {grid.extent}"
        res.failure = "goal_outside"
        return res
    start_cell, d_start = snap_to_free(free, (iy[0], ix[0]))
    if start_cell is None or d_start * grid.resolution > max_start_snap_m:
        res.message = (f"no free cell within {max_start_snap_m} m of the start "
                       f"(nearest {d_start * grid.resolution:.2f} m)")
        res.failure = "start_blocked"
        return res
    reach = reachable_mask(tmap.cost, start_cell)
    goal_cell, d_goal = snap_to_free(reach, (iy[1], ix[1]))
    sx, sy = tmap.cell_to_world(start_cell[0], start_cell[1])
    res.start_snapped_xy = np.array([float(sx), float(sy)])
    res.start_snap_m = float(np.linalg.norm(res.start_snapped_xy - start_xy))
    if goal_cell is None:
        res.message = "nothing is reachable from the start"
        res.failure = "unreachable"
        return res
    gxw, gyw = tmap.cell_to_world(goal_cell[0], goal_cell[1])
    res.goal_snapped_xy = np.array([float(gxw), float(gyw)])
    res.goal_snap_m = float(np.linalg.norm(res.goal_snapped_xy - goal_xy))
    if res.goal_snap_m > max_goal_snap_m:
        res.message = (f"the goal is not reachable: the closest reachable cell is {res.goal_snap_m:.2f} m away "
                       f"(limit {max_goal_snap_m} m)")
        res.failure = "unreachable"
        return res
    cells, gcost, expanded = astar(tmap.cost, start_cell, goal_cell, resolution=grid.resolution,
                                   heuristic_weight=heuristic_weight)
    res.expanded = expanded
    if len(cells) == 0:  # cannot happen when the goal is in the start's component; kept as a guard
        res.message = "A* found no path"
        res.failure = "no_path"
        return res
    res.cells = cells
    res.graph_cost = gcost
    cx, cy = tmap.cell_to_world(cells[:, 0], cells[:, 1])
    raw = np.column_stack([cx, cy])
    # begin and end at the snapped points' cell centres (already the case) -- keep the exact centres
    res.raw_xy = raw
    if smooth and len(raw) >= 2:
        res.shortcut_xy, res.path_xy = smooth_path(tmap, raw, spacing=spacing)
    else:
        res.shortcut_xy = raw.copy()
        res.path_xy = resample_polyline(raw, spacing) if len(raw) >= 2 else raw.copy()
    res.path_cost = path_cost(tmap, res.path_xy) if len(res.path_xy) > 1 else 0.0
    res.path_z = tmap.height_at(res.path_xy) if len(res.path_xy) else np.zeros(0)
    res.success = True
    res.message = (f"path {res.length_m:.1f} m (A* {res.raw_length_m:.1f} m, {expanded} nodes expanded); "
                   f"goal moved {res.goal_snap_m:.2f} m to a reachable free cell")
    return res
