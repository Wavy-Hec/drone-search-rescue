"""Cost-weighted 8-connected A*: optimality against Dijkstra, obstacle avoidance, snapping and smoothing.

numpy + scipy only (runs in env_isaaclab with ``python tests/test_teaming_planner.py`` or under pytest).
"""

from __future__ import annotations

import math

import numpy as np
from _runner import main_for

SQ2 = math.sqrt(2.0)


def _dijkstra_cost(cost: np.ndarray, start, goal, res: float = 1.0) -> float:
    """Reference: scipy's Dijkstra on exactly the same graph as the A* (8-connected, no corner cutting)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    ny, nx = cost.shape
    free = np.isfinite(cost)
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for y in range(ny):
            for x in range(nx):
                y2, x2 = y + dy, x + dx
                if not (0 <= y2 < ny and 0 <= x2 < nx) or not (free[y, x] and free[y2, x2]):
                    continue
                if dy and dx and not (free[y, x2] and free[y2, x]):
                    continue  # no corner cutting
                length = res * (SQ2 if (dy and dx) else 1.0)
                c = 0.5 * length * (cost[y, x] + cost[y2, x2])
                rows += [idx[y, x], idx[y2, x2]]
                cols += [idx[y2, x2], idx[y, x]]
                w += [c, c]
    g = coo_matrix((w, (rows, cols)), shape=(ny * nx, ny * nx)).tocsr()
    d = dijkstra(g, indices=idx[start], directed=True)
    return float(d[idx[goal]])


def _path_cost(cost, path, res=1.0):
    total = 0.0
    for (y1, x1), (y2, x2) in zip(path[:-1], path[1:], strict=False):
        assert max(abs(y2 - y1), abs(x2 - x1)) == 1, "path must move between 8-neighbours"
        length = res * (SQ2 if (y1 != y2 and x1 != x2) else 1.0)
        total += 0.5 * length * (cost[y1, x1] + cost[y2, x2])
    return total


def _random_cost(rng, n=24, obstacle_fraction=0.25):
    cost = 1.0 + 4.0 * rng.random((n, n))
    cost[rng.random((n, n)) < obstacle_fraction] = np.inf
    cost[0, 0] = cost[-1, -1] = 1.0
    return cost


def test_astar_is_optimal_against_dijkstra():
    from drone_lab.teaming.planner import astar

    rng = np.random.default_rng(42)
    checked = 0
    for trial in range(16):
        cost = _random_cost(rng, n=20, obstacle_fraction=0.12 + 0.01 * trial)
        start, goal = (0, 0), (19, 19)
        res = 0.25
        ref = _dijkstra_cost(cost, start, goal, res)
        path, c, expanded = astar(cost, start, goal, resolution=res)
        if math.isinf(ref):
            assert len(path) == 0 and math.isinf(c)
            continue
        checked += 1
        assert abs(c - ref) < 1e-9, (trial, c, ref)
        assert tuple(path[0]) == start and tuple(path[-1]) == goal
        assert all(np.isfinite(cost[y, x]) for y, x in path)
        assert abs(_path_cost(cost, path, res) - c) < 1e-9
        assert expanded <= np.isfinite(cost).sum()
    assert checked >= 8  # most random maps are solvable


def test_astar_uniform_grid_gives_octile_distance():
    from drone_lab.teaming.planner import astar

    cost = np.ones((30, 40))
    path, c, _ = astar(cost, (2, 3), (20, 35), resolution=0.5)
    dy, dx = 18, 32
    octile = max(dy, dx) + (SQ2 - 1) * min(dy, dx)
    assert abs(c - 0.5 * octile) < 1e-9
    assert len(path) == max(dy, dx) + 1


def test_astar_goes_through_the_gap_and_never_cuts_corners():
    from drone_lab.teaming.planner import astar

    cost = np.ones((21, 21))
    cost[:, 10] = np.inf  # a wall ...
    cost[17, 10] = 1.0  # ... with a one-cell gap
    path, c, _ = astar(cost, (2, 2), (2, 18))
    assert len(path) > 0 and (17, 10) in [tuple(p) for p in path]
    assert all(np.isfinite(cost[y, x]) for y, x in path)
    # two lethal cells touching only at a corner block the diagonal squeeze between them
    cost = np.ones((5, 5))
    cost[1, 2] = cost[2, 1] = np.inf
    path, _, _ = astar(cost, (1, 1), (2, 2))
    steps = [tuple(b - a) for a, b in zip(path[:-1], path[1:], strict=False)]
    assert (1, 1) not in steps and len(path) > 2
    # fully walled-in goal: no path
    cost = np.ones((9, 9))
    cost[3:6, 3:6] = np.inf
    cost[4, 4] = 1.0
    path, c, _ = astar(cost, (0, 0), (4, 4))
    assert len(path) == 0 and math.isinf(c)


def test_astar_prefers_cheap_detours():
    from drone_lab.teaming.planner import astar

    cost = np.ones((15, 31))
    cost[:12, 15] = 50.0  # an expensive (but walkable) strip with a cheap way around at the bottom
    path, c, _ = astar(cost, (2, 2), (2, 28))
    assert max(y for y, _ in path) >= 12  # goes around
    straight = np.ones((15, 31))
    _, c_free, _ = astar(straight, (2, 2), (2, 28))
    assert c > c_free and c < c_free + 49.0


def _rubble_map():
    """10 m x 10 m flat map with two boxes (0.6 m high) leaving a corridor, as an elevation grid."""
    from drone_lab.mapping import GridSpec

    g = GridSpec(0.0, 0.0, 0.1, 100, 100)
    z = np.zeros(g.shape)
    z[0:55, 40:48] = 0.6  # wall from the south edge
    z[70:100, 40:48] = 0.6  # wall from the north edge: corridor y in [5.5, 7.0]
    return g, z


def test_plan_path_avoids_rubble_and_stops_next_to_the_person():
    from drone_lab.teaming.planner import plan_path
    from drone_lab.teaming.traversability import Footprint, TraversabilityCfg, compute_traversability

    g, z = _rubble_map()
    person = Footprint(8.0, 2.0, 0.6, "person")
    cfg = TraversabilityCfg(robot_radius_m=0.35)
    t = compute_traversability(z, g, cfg, footprints=[person])
    plan = plan_path(t, (1.0, 2.0), (person.x, person.y))
    assert plan.success, plan.message
    p = plan.path_xy
    # passes through the corridor between the two walls
    at_wall = p[(p[:, 0] > 4.0) & (p[:, 0] < 4.8)]
    assert len(at_wall) > 0 and np.all((at_wall[:, 1] > 5.5) & (at_wall[:, 1] < 7.0))
    # never on a lethal cell, and keeps the robot radius from the boxes' edge cells
    assert np.all(np.isfinite(t.sample(t.cost, p)))
    assert np.min(t.sample(t.clearance, p)) >= cfg.robot_radius_m - 1e-9
    # stops next to the person (outside their footprint + robot radius), not on them
    d_end = np.linalg.norm(p[-1] - [person.x, person.y])
    assert person.radius + cfg.robot_radius_m - 0.1 <= d_end <= person.radius + cfg.robot_radius_m + 0.3
    assert abs(plan.goal_snap_m - d_end) < 0.1
    # smoothing: shorter than the grid path, evenly resampled, same end points
    assert plan.length_m < plan.raw_length_m
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert np.all(seg <= 0.1 + 1e-6)
    assert np.allclose(p[0], plan.raw_xy[0]) and np.allclose(p[-1], plan.raw_xy[-1])
    assert plan.path_cost <= 1.05 * plan.graph_cost


def test_plan_path_snaps_the_start_and_reports_unreachable_goals():
    from drone_lab.teaming.planner import plan_path
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    g, z = _rubble_map()
    z[55:70, 40:48] = 0.6  # close the corridor: the east side is cut off
    t = compute_traversability(z, g, TraversabilityCfg())
    # start inside the wall: snapped to the nearest free cell (west side)
    plan = plan_path(t, (4.2, 2.0), (2.0, 8.0), max_start_snap_m=2.0)
    assert plan.success and plan.start_snap_m > 0.3 and plan.start_snapped_xy[0] < 4.0
    # goal on the cut-off side: the closest reachable cell is too far -> failure with a reason
    plan = plan_path(t, (1.0, 2.0), (8.0, 5.0), max_goal_snap_m=2.0)
    assert not plan.success and "not reachable" in plan.message
    assert plan.failure == "unreachable" and not plan.start_failed  # another goal may still work
    # with a generous snap distance it goes as close as it can (to the wall)
    plan = plan_path(t, (1.0, 2.0), (8.0, 5.0), max_goal_snap_m=10.0)
    assert plan.success and plan.path_xy[-1][0] < 4.0 and plan.goal_snap_m > 4.0
    assert plan.failure == "" and plan.to_dict()["failure"] == ""
    # start outside the map, or buried deep in no-go cells: the start's problem, whatever the goal
    plan = plan_path(t, (-5.0, 2.0), (2.0, 2.0))
    assert not plan.success and "outside" in plan.message
    assert plan.failure == "start_outside" and plan.start_failed
    plan = plan_path(t, (4.4, 6.2), (2.0, 2.0), max_start_snap_m=0.2)
    assert plan.failure == "start_blocked" and plan.start_failed and "no free cell" in plan.message
    plan = plan_path(t, (1.0, 2.0), (12.0, 2.0))
    assert plan.failure == "goal_outside" and not plan.start_failed


def test_shortcuts_never_raise_the_cost():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.planner import path_cost, plan_path, shortcut_path
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    rng = np.random.default_rng(3)
    g = GridSpec(0.0, 0.0, 0.25, 60, 60)
    xs, ys = g.cell_centers()
    # smooth hills (costly slopes) plus a few boxes
    z = 0.6 * np.sin(xs[None, :] / 2.0) * np.cos(ys[:, None] / 2.5)
    for _ in range(6):
        cx, cy = rng.integers(8, 52, 2)
        z[cy - 2 : cy + 2, cx - 2 : cx + 2] += 1.0
    t = compute_traversability(z, g, TraversabilityCfg())
    plan = plan_path(t, (1.5, 1.5), (13.5, 13.5), smooth=False)
    assert plan.success, plan.message
    raw = plan.raw_xy
    sc = shortcut_path(t, raw, tolerance=0.02)
    assert len(sc) <= len(raw) and np.allclose(sc[0], raw[0]) and np.allclose(sc[-1], raw[-1])
    assert math.isfinite(path_cost(t, sc)) and path_cost(t, sc) <= 1.02 * path_cost(t, raw) + 1e-9


def test_resample_polyline_spacing_and_ends():
    from drone_lab.teaming.planner import polyline_length, resample_polyline

    pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    r = resample_polyline(pts, 0.25)
    assert np.allclose(r[0], [0, 0]) and np.allclose(r[-1], [3, 4])
    assert abs(polyline_length(r) - 7.0) < 1e-6
    seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
    assert np.all(seg <= 0.25 + 1e-9) and np.all(seg > 0.2)


if __name__ == "__main__":
    main_for(globals(), "test_teaming_planner")
