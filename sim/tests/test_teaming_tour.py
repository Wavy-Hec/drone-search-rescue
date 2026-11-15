"""One route past several people: the visiting order, the stops, the legs, the figure and the script's tour mode.

* the visiting order on cost tables and on synthetic maps against the brute force over every order, with the costs
  between the stops from scipy's Dijkstra;
* a wall changes the order, unreachable goals are left out and reported, the legs connect, the totals are the sums;
* nearest neighbour + 2-opt (above 12 goals) is never worse than nearest neighbour alone;
* ``scripts/rescue_finale.py --tour`` on the ground truth and on a drone_gcs report (needs env_isaaclab; skipped
  elsewhere).

numpy + scipy (+ matplotlib for the figure): ``python tests/test_teaming_tour.py`` or under pytest.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from _runner import main_for, require

SQ2 = math.sqrt(2.0)
PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = str(PROJECT / "scripts" / "rescue_finale.py")


# ------------------------------------------------------------------------------------------------ references
def _cell(tmap, xy) -> tuple[int, int]:
    """Cell ``(iy, ix)`` of a world point."""
    g = tmap.grid
    return int(math.floor((xy[1] - g.y_min) / g.resolution)), int(math.floor((xy[0] - g.x_min) / g.resolution))


def _dijkstra(tmap, cells, direction_cost=None) -> np.ndarray:
    """Reference: scipy's Dijkstra from each of ``cells`` to every cell, on the planner's graph (8-connected, a move
    costs its length times the mean cost of its two cells, no corner cutting). Returns ``(len(cells), ny, nx)``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    cost, res = tmap.cost, tmap.grid.resolution
    ny, nx = cost.shape
    free = np.isfinite(cost)
    idx = np.arange(ny * nx).reshape(ny, nx)
    extra = np.zeros((4, ny, nx)) if direction_cost is None else direction_cost
    rows, cols, w = [], [], []
    # the moves along x, along y and along the two diagonals: the layers of the planner's direction costs
    for axis, (dy, dx) in enumerate(((0, 1), (1, 0), (1, 1), (1, -1))):
        for y in range(ny - dy):
            for x in range(max(0, -dx), nx - max(0, dx)):
                y2, x2 = y + dy, x + dx
                if not (free[y, x] and free[y2, x2]):
                    continue
                if dy and dx and not (free[y, x2] and free[y2, x]):
                    continue
                length = res * (SQ2 if (dy and dx) else 1.0)
                c = 0.5 * length * (cost[y, x] + cost[y2, x2] + extra[axis, y, x] + extra[axis, y2, x2])
                rows += [idx[y, x], idx[y2, x2]]
                cols += [idx[y2, x2], idx[y, x]]
                w += [c, c]
    graph = coo_matrix((w, (rows, cols)), shape=(ny * nx, ny * nx)).tocsr()
    return dijkstra(graph, indices=[idx[c] for c in cells], directed=True).reshape(len(cells), ny, nx)


def _cost_table(tmap, cells, direction_cost=None) -> np.ndarray:
    """Reference costs between every two of ``cells``."""
    dist = _dijkstra(tmap, cells, direction_cost)
    return np.array([[dist[a][cells[b]] for b in range(len(cells))] for a in range(len(cells))])


def _route_cost(table, order) -> float:
    """Cost of the route node 0 -> ``order[0]`` -> ``order[1]`` -> ... on a cost table."""
    nodes = [0, *order]
    return float(sum(table[a][b] for a, b in zip(nodes[:-1], nodes[1:], strict=True)))


def _brute_force(table) -> tuple[tuple[int, ...], float, bool]:
    """The cheapest order of the nodes ``1 .. n`` by trying every one, its cost, and whether no other order ties."""
    costs = sorted((_route_cost(table, p), p) for p in itertools.permutations(range(1, len(table))))
    unique = len(costs) == 1 or costs[1][0] - costs[0][0] > 1e-9
    return costs[0][1], costs[0][0], unique


def _nearest_neighbour(table) -> list[int]:
    left = list(range(1, len(table)))
    order, here = [], 0
    while left:
        here = min(left, key=lambda j: table[here][j])
        left.remove(here)
        order.append(here)
    return order


def _flat_map_with_boxes(rng, size_m=10.0, res=0.2, boxes=5):
    """A flat map with a few 0.6 m high boxes of 1 m x 1 m, as a traversability map."""
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    n = round(size_m / res)
    g = GridSpec(0.0, 0.0, res, n, n)
    z = np.zeros(g.shape)
    for _ in range(boxes):
        cx, cy = rng.integers(8, n - 8, 2)
        z[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.6
    return compute_traversability(z, g, TraversabilityCfg(robot_radius_m=0.35))


def _walled_map(gap: bool):
    """10 m x 10 m flat map with a 0.6 m high wall at x = 4.0 .. 4.8 m from the south edge up to y = 7 m (``gap``)
    or across the whole map."""
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    g = GridSpec(0.0, 0.0, 0.1, 100, 100)
    z = np.zeros(g.shape)
    z[0 : 70 if gap else 100, 40:48] = 0.6
    return compute_traversability(z, g, TraversabilityCfg(robot_radius_m=0.35))


# ------------------------------------------------------------------------------------------- the visiting order
def test_held_karp_order_is_the_cheapest_of_all_orders():
    from drone_lab.teaming.tour import held_karp_order, nearest_neighbour_order, route_cost, two_opt

    # stops on a line: the start at 0, the nodes 1, 2, 3 at 1.0, -1.5 and 3.0
    x = np.array([0.0, 1.0, -1.5, 3.0])
    line = np.abs(x[:, None] - x[None, :])
    assert route_cost(line, [0, 1, 3, 2]) == 1.0 + 2.0 + 4.5
    assert nearest_neighbour_order(line) == [1, 3, 2]  # always the nearest one next: 7.5 m
    assert held_karp_order(line) == [2, 1, 3]  # first the one behind the start: 1.5 + 2.5 + 2 = 6 m
    assert two_opt(line, [1, 3, 2]) == [1, 3, 2]  # no reversal of a stretch makes 7.5 m shorter
    assert two_opt(line, [3, 1, 2]) == [2, 1, 3]  # 3 + 2 + 2.5 = 7.5 m; reversed as a whole it is the 6 m route
    assert held_karp_order(np.zeros((1, 1))) == [] and held_karp_order(np.array([[0.0, 2.0], [2.0, 0.0]])) == [1]

    rng = np.random.default_rng(11)
    for n in range(2, 8):
        for _ in range(8):
            table = rng.random((n + 1, n + 1))
            table = table + table.T
            best, best_cost, unique = _brute_force(table)
            order = held_karp_order(table)
            assert unique and tuple(order) == best, (order, best)
            assert abs(_route_cost(table, order) - best_cost) < 1e-12


def test_two_opt_is_never_worse_than_nearest_neighbour():
    from drone_lab.teaming.tour import nearest_neighbour_order, two_opt

    rng = np.random.default_rng(5)
    improved = 0
    for trial in range(30):
        n = 5 + trial % 16
        points = rng.random((n + 1, 2)) * 20.0
        table = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        plain = nearest_neighbour_order(table)
        assert plain == _nearest_neighbour(table)
        order = two_opt(table, plain)
        assert sorted(order) == list(range(1, n + 1))  # every node once
        cost = _route_cost(table, order)
        assert cost <= _route_cost(table, plain) + 1e-12
        improved += cost < _route_cost(table, plain) - 1e-9
        # no reversal of a stretch is left that would make the route shorter
        for i in range(n):
            for j in range(i + 1, n):
                candidate = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                assert _route_cost(table, candidate) >= cost - 1e-9
        if n <= 7:
            assert cost >= _brute_force(table)[1] - 1e-12
    assert improved >= 10  # the reversals do find shorter routes


# ------------------------------------------------------------------------------------------------- on a map
def test_tour_order_is_the_brute_force_optimum_on_random_maps():
    from drone_lab.teaming.tour import plan_tour

    unique_optima = 0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        tmap = _flat_map_with_boxes(rng)
        n = 3 + seed  # 3 .. 6 goals
        start = np.array([0.7, 0.7])
        goals = rng.uniform(0.5, 9.5, (n, 2))
        tour = plan_tour(tmap, start, goals, max_goal_snap_m=3.0)
        assert tour.success and tour.method == "held_karp" and tour.unreachable == [], tour.message
        assert sorted(tour.order) == list(range(n)) and len(tour.legs) == n

        # every stop is a free cell reachable from the start, and no such cell is nearer to its goal's cell
        stops = {i: _cell(tmap, leg.goal_snapped_xy) for i, leg in zip(tour.order, tour.legs, strict=True)}
        cells = [_cell(tmap, start)] + [stops[i] for i in range(n)]
        reachable = np.isfinite(_dijkstra(tmap, cells[:1])[0])
        ry, rx = np.nonzero(reachable)
        for i in range(n):
            gy, gx = _cell(tmap, goals[i])
            assert reachable[stops[i]]
            nearest = np.hypot(ry - gy, rx - gx).min()
            assert abs(math.hypot(stops[i][0] - gy, stops[i][1] - gx) - nearest) < 1e-9

        # the order is the cheapest of all n! orders, and each leg costs what Dijkstra says
        table = _cost_table(tmap, cells)
        best, best_cost, unique = _brute_force(table)
        assert abs(tour.cost - best_cost) < 1e-6, (tour.cost, best_cost)
        if unique:
            unique_optima += 1
            assert tuple(i + 1 for i in tour.order) == best
        nodes = [0] + [i + 1 for i in tour.order]
        for leg, a, b in zip(tour.legs, nodes[:-1], nodes[1:], strict=True):
            assert abs(leg.graph_cost - table[a][b]) < 1e-6
    assert unique_optima >= 3


def test_a_wall_changes_the_order():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.tour import plan_tour
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    start = (1.0, 1.0)
    goals = np.array([[6.0, 1.0], [1.0, 8.0]])  # 5 m east of the start, and 7 m north of it
    # straight lines: east first is 5 + 8.6 m, north first 7 + 8.6 m
    straight = np.linalg.norm(np.vstack([start, goals])[:, None] - np.vstack([start, goals])[None, :], axis=2)
    assert _brute_force(straight)[0] == (1, 2)
    flat = compute_traversability(np.zeros((100, 100)), GridSpec(0.0, 0.0, 0.1, 100, 100), TraversabilityCfg())
    assert plan_tour(flat, start, goals).order == [0, 1]
    # a wall between the start and the eastern goal, open only north of y = 7 m: the way east leads past the
    # northern goal, so that one comes first
    tour = plan_tour(_walled_map(gap=True), start, goals)
    assert tour.success and tour.order == [1, 0]
    assert tour.leg_lengths_m[0] < 7.5 and 9.0 < tour.leg_lengths_m[1] < 13.0
    assert tour.legs[1].path_xy[:, 1].max() > 7.3  # through the gap


def test_unreachable_goals_are_left_out_and_reported():
    from drone_lab.teaming.tour import plan_tour

    tmap = _walled_map(gap=False)  # the east side is cut off
    goals = [(2.0, 8.0), (8.0, 5.0), (12.0, 2.0), (2.0, 3.0), (4.2, 5.0)]
    #        west        east: cut   off the map  west        inside the wall, 0.2 m from its west face
    tour = plan_tour(tmap, (1.0, 1.0), goals)
    assert tour.success and tour.unreachable == [1, 2]
    assert tour.order == [3, 4, 0] and len(tour.legs) == 3  # up the west side
    assert "2 unreachable" in tour.message
    wall_stop = tour.legs[1]  # west of the wall, the robot's radius and more from its face
    assert wall_stop.goal_snapped_xy[0] < 4.0 - 0.35 and 0.2 + 0.35 < wall_stop.goal_snap_m < 1.0
    d = tour.to_dict()
    assert json.loads(json.dumps(d)) == d  # plain JSON types
    assert d["order"] == [3, 4, 0] and d["unreachable"] == [1, 2] and [leg["goal"] for leg in d["legs"]] == [3, 4, 0]
    assert d["legs"][0]["goal_xy"] == [2.0, 3.0] and d["method"] == "held_karp"

    # a smaller reach leaves the goal inside the wall out as well
    assert plan_tour(tmap, (1.0, 1.0), goals, max_goal_snap_m=0.4).unreachable == [1, 2, 4]
    # nothing to visit: no goals, only goals that cannot be reached, or a start off the map
    for goals_xy, reason in (([], "no goals"), ([(8.0, 5.0), (12.0, 2.0)], "no goal is reachable")):
        none = plan_tour(tmap, (1.0, 1.0), goals_xy)
        assert not none.success and none.order == [] and none.legs == [] and none.method == ""
        assert none.unreachable == list(range(len(goals_xy))) and reason in none.message
        assert none.length_m == 0.0 and none.cost == 0.0 and none.to_dict()["legs"] == []
    off = plan_tour(tmap, (-5.0, 1.0), goals)
    assert not off.success and off.unreachable == [0, 1, 2, 3, 4] and "outside the map" in off.message


def test_legs_connect_and_the_totals_are_the_sums():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.planner import polyline_length
    from drone_lab.teaming.tour import plan_tour
    from drone_lab.teaming.traversability import Footprint, TraversabilityCfg, compute_traversability

    g = GridSpec(0.0, 0.0, 0.1, 100, 100)
    z = np.zeros(g.shape)
    z[0:55, 40:48] = 0.6  # two boxes leaving a corridor at y = 5.5 .. 7 m
    z[70:100, 40:48] = 0.6
    people = [
        Footprint(8.0, 2.0, 0.6, "person"),
        Footprint(2.0, 8.5, 0.6, "person"),
        Footprint(8.0, 8.0, 0.6, "person"),
    ]
    cfg = TraversabilityCfg(robot_radius_m=0.35)
    tmap = compute_traversability(z, g, cfg, footprints=people)
    goals = np.array([[p.x, p.y] for p in people])
    start = np.array([1.0, 2.0])
    tour = plan_tour(tmap, start, goals)
    assert tour.success and sorted(tour.order) == [0, 1, 2] and len(tour.legs) == 3

    here = start
    for i, leg in zip(tour.order, tour.legs, strict=True):
        p = leg.path_xy
        assert leg.success and np.allclose(leg.goal_xy, goals[i])
        assert np.linalg.norm(p[0] - here) < 0.1  # starts where the last leg stopped (the first: in the start's cell)
        assert np.allclose(p[-1], leg.goal_snapped_xy)
        assert np.all(np.isfinite(tmap.sample(tmap.cost, p)))  # never on a no-go cell
        assert np.linalg.norm(np.diff(p, axis=0), axis=1).max() <= 0.1 + 1e-6
        # the stop is next to the person: outside their footprint and the robot's radius, not on them
        d_stop = np.linalg.norm(p[-1] - goals[i])
        assert 0.6 + 0.35 - 0.1 <= d_stop <= 0.6 + 0.35 + 0.3 and abs(leg.goal_snap_m - d_stop) < 1e-9
        here = p[-1]
    for a, b in zip(tour.legs[:-1], tour.legs[1:], strict=True):
        assert np.array_equal(a.path_xy[-1], b.path_xy[0])  # exactly the same point

    lengths = [float(np.linalg.norm(np.diff(leg.path_xy, axis=0), axis=1).sum()) for leg in tour.legs]
    assert np.allclose(tour.leg_lengths_m, lengths) and abs(tour.length_m - sum(lengths)) < 1e-9
    assert tour.leg_costs == [leg.graph_cost for leg in tour.legs]
    assert abs(tour.cost - sum(leg.graph_cost for leg in tour.legs)) < 1e-9
    whole = np.vstack([leg.path_xy for leg in tour.legs])
    assert abs(polyline_length(whole) - tour.length_m) < 1e-9  # the legs joined are one route of that length
    d = tour.to_dict()
    assert abs(d["length_m"] - sum(lengths)) < 1e-3 and abs(d["cost"] - tour.cost) < 1e-3
    assert abs(sum(leg["length_m"] for leg in d["legs"]) - d["length_m"]) < 2e-3
    assert [leg["from_xy"] for leg in d["legs"][1:]] == [leg["stop_xy"] for leg in d["legs"][:-1]]

    # a coarser path spacing changes the points of the legs, not the stops
    coarse = plan_tour(tmap, start, goals, spacing=0.3)
    assert coarse.order == tour.order
    steps = np.linalg.norm(np.diff(coarse.legs[0].path_xy, axis=0), axis=1)
    assert steps.max() <= 0.3 + 1e-6 and steps.mean() > 0.2


def test_many_goals_use_nearest_neighbour_and_2opt():
    from drone_lab.teaming.tour import HELD_KARP_MAX_GOALS, plan_tour

    assert HELD_KARP_MAX_GOALS == 12
    rng = np.random.default_rng(21)
    tmap = _flat_map_with_boxes(rng, size_m=12.0, boxes=6)
    start = np.array([0.7, 0.7])
    goals = rng.uniform(0.5, 11.5, (13, 2))
    for n, method in ((13, "nearest_neighbour_2opt"), (12, "held_karp")):
        tour = plan_tour(tmap, start, goals[:n], max_goal_snap_m=3.0)
        assert tour.success and tour.method == method and sorted(tour.order) == list(range(n))
        stops = {i: _cell(tmap, leg.goal_snapped_xy) for i, leg in zip(tour.order, tour.legs, strict=True)}
        table = _cost_table(tmap, [_cell(tmap, start)] + [stops[i] for i in range(n)])
        assert abs(tour.cost - _route_cost(table, [i + 1 for i in tour.order])) < 1e-6
        # never worse than always walking on to the nearest person
        assert tour.cost <= _route_cost(table, _nearest_neighbour(table)) + 1e-6


def test_cross_slope_cost_counts_between_the_stops():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.tour import plan_tour
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    g = GridSpec(0.0, 0.0, 0.2, 50, 50)
    xs, _ = g.cell_centers()
    z = np.tile(math.tan(math.radians(15.0)) * xs, (g.ny, 1))  # a 15 deg slope, the fall line along x
    cfg = TraversabilityCfg(max_slope_deg=22.0, inflation_buffer_m=0.0)
    start = (2.1, 2.1)
    goals = np.array([[2.1, 8.1], [8.1, 2.1], [8.1, 8.1]])
    s = math.sqrt(0.5)
    plain = compute_traversability(z, g, cfg)
    cells = [_cell(plain, start)] + [_cell(plain, xy) for xy in goals]
    tour = plan_tour(plain, start, goals)
    assert abs(tour.cost - _brute_force(_cost_table(plain, cells))[1]) < 1e-6

    directional = compute_traversability(z, g, replace(cfg, w_cross_slope=6.0))
    extra = np.stack([directional.cross_slope_cost(ux, uy) for ux, uy in ((1.0, 0.0), (0.0, 1.0), (s, s), (s, -s))])
    # across the slope (along y) a metre costs 6 (15 / 22)^2 more, along the fall line (x) nothing more
    assert abs(extra[1, 25, 25] - 6.0 * (15.0 / 22.0) ** 2) < 1e-6 and extra[0].max() < 1e-9
    table = _cost_table(directional, cells, extra)
    best, best_cost, _ = _brute_force(table)
    tour_d = plan_tour(directional, start, goals)
    assert abs(tour_d.cost - best_cost) < 1e-6 and tuple(i + 1 for i in tour_d.order) == best
    assert tour_d.cost > tour.cost + 5.0  # the 12 m across the slope cost more than on the plain map
    assert abs(tour_d.leg_costs[0] - table[0][tour_d.order[0] + 1]) < 1e-6


def test_plot_tour_writes_a_file():
    from drone_lab.teaming.inputs import Pin
    from drone_lab.teaming.plots import plot_tour
    from drone_lab.teaming.tour import plan_tour

    tmap = _walled_map(gap=True)
    goals = [(6.0, 1.0), (1.0, 8.0), (8.0, 8.0)]
    pins = [Pin(id=str(k), label="person", x=x, y=y) for k, (x, y) in enumerate(goals)]
    with tempfile.TemporaryDirectory() as tmp:
        tour = plan_tour(tmap, (1.0, 1.0), goals)
        out = plot_tour(tmap, tour, Path(tmp) / "figs" / "tour.png", pins=pins, source="synthetic", robot="Test robot")
        assert out == Path(tmp) / "figs" / "tour.png" and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert out.stat().st_size > 30_000
        dark = plot_tour(tmap, tour, Path(tmp) / "dark.png", theme="dark")
        assert dark.stat().st_size > 30_000
        # a tour without a route still gives a figure (the map and the reason)
        none = plan_tour(_walled_map(gap=False), (1.0, 1.0), [(8.0, 5.0)])
        assert not none.success and plot_tour(tmap, none, Path(tmp) / "none.png").stat().st_size > 20_000


# --------------------------------------------------------------------------------------------------- the CLI
def _cli(out: Path, *args: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, SCRIPT, "--out", str(out), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(PROJECT),
    )
    tour = json.loads((out / "tour.json").read_text()) if (out / "tour.json").is_file() else None
    return r, tour


def _leg_commands(out: Path) -> list[list[str]]:
    """The commands of ``tour_legs.sh``, split into words."""
    lines = (out / "tour_legs.sh").read_text().splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    return [shlex.split(line) for line in lines[1:] if not line.startswith("#")]


def _option(words: list[str], flag: str, count: int = 1) -> list[str]:
    """The ``count`` words after ``flag`` in a command."""
    k = words.index(flag)
    return words[k + 1 : k + 1 + count]


def _check_tour_files(out: Path, tour: dict, start: list[float], speed: float) -> None:
    """What every tour's three files must show: connected legs, totals that are the sums, one command per leg."""
    legs = tour["legs"]
    assert [leg["leg"] for leg in legs] == list(range(1, len(legs) + 1))
    assert tour["order"] == [leg["pin"]["id"] for leg in legs] and tour["start_xy"] == start
    here = start
    for leg in legs:
        path = np.array(leg["path_xy"])
        assert np.linalg.norm(np.subtract(leg["from_xy"], here)) < 0.2 and len(path) == len(leg["path_z"])
        assert path[0].tolist() == leg["from_xy"] and path[-1].tolist() == leg["stop_xy"]
        assert abs(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() - leg["length_m"]) < 0.01
        to_person = np.hypot(leg["stop_xy"][0] - leg["pin"]["x"], leg["stop_xy"][1] - leg["pin"]["y"])
        assert abs(to_person - leg["stop_to_person_m"]) < 0.01 and to_person <= 2.5  # --max_goal_snap
        assert abs(leg["walk_time_s"] - leg["length_m"] / speed) < 0.06
        here = leg["stop_xy"]
    assert abs(tour["length_m"] - sum(leg["length_m"] for leg in legs)) < 0.01
    assert abs(tour["cost"] - sum(leg["cost"] for leg in legs)) < 0.01
    assert tour["cruise_speed_mps"] == speed and abs(tour["walk_time_s"] - tour["length_m"] / speed) < 0.06
    assert (out / "tour.png").stat().st_size > 50_000
    assert not (out / "plan.json").exists() and not (out / "summary.json").exists()  # no single plan, no walk

    commands = _leg_commands(out)
    assert len(commands) == len(legs)
    for k, (words, leg) in enumerate(zip(commands, legs, strict=True), start=1):
        assert words[:2] == ["python", SCRIPT] and words[-1] == "$@"
        leg_start = start if k == 1 else legs[k - 2]["stop_xy"]
        assert np.allclose([float(v) for v in _option(words, "--start", 2)], leg_start, atol=1e-3)
        assert _option(words, "--goal_pin") == [leg["pin"]["id"]]
        assert _option(words, "--out") == [f"{out}_leg{k}"]
        assert "--tour" not in words and "--plan_only" not in words


def test_rescue_finale_cli_tour_on_the_ground_truth():
    """``--tour 3`` on the ground truth of SAR site 0: the three reachable people nearest to the landing pad."""
    require("isaaclab")
    require("trimesh")
    from drone_lab.sar.scene import SarSiteCfg, scenario_for
    from drone_lab.teaming.inputs import pins_from_data

    truth = scenario_for(SarSiteCfg(seed=0)).ground_truth()
    people = [p for p in pins_from_data(truth["objects"]) if p.label == "person"]
    people.sort(key=lambda p: math.hypot(p.x + 23.0, p.y + 18.0))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tour"
        r, tour = _cli(out, "--tour", "3")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert tour["robot"] == "custom" and tour["map_mode"] == "ground_truth" and tour["sar_seed"] == 0
        assert tour["people"] == len(people) == 11 and tour["max_people"] == 3 and tour["method"] == "held_karp"
        # nearest first, the ones that cannot be reached left out
        assert set(tour["unreachable"]) < {p.id for p in people}
        nearest = [p.id for p in people if p.id not in tour["unreachable"]][:3]
        assert sorted(tour["order"]) == sorted(nearest) and len(tour["legs"]) == 3
        _check_tour_files(out, tour, [-23.0, -18.0], 0.7)
        # no option of this run to hand on: the start, the person, the tour's footprints, the output, "$@"
        assert all(len(words) == 11 and "--tour_footprints" in words for words in _leg_commands(out))
        assert " leg  person" in r.stdout and r.stdout.count("\n   ") >= 3  # the table of legs
        assert "tour.json and tour_legs.sh" in r.stdout

        # a tour visits the pins: it does not go with a goal of its own
        r, _ = _cli(Path(tmp) / "both", "--tour", "--goal_pin", "person_04")
        assert r.returncode != 0 and "--tour visits the person pins" in r.stderr


def _tour_report(tmp: Path) -> Path:
    """A drone_gcs mission report of a flat site with a surface map. Pinned: the landing pad at (-23, -18), three
    people the robot can reach (ids 4, 9 and 7: 10 m east of the pad, then 12 m north, then 12 m east; 10, 15.6 and
    25 m from the pad), one in the middle of 6 m x 6 m the drone did not map (id 5, 18.1 m from the pad) and one
    unconfirmed (id 3). Not pinned: a 2.25 m x 2.25 m box, 1 m tall, on the straight line from the pad to the first
    person."""
    from drone_lab.mapping import GridSpec
    from drone_lab.paths import Area

    grid = GridSpec.for_survey(Area(-20.0, 20.0, -20.0, 20.0), 0.25)
    xs, ys = grid.cell_centers()
    z = np.zeros(grid.shape, dtype=np.float32)
    z[np.ix_(np.abs(ys + 16.0) < 3.0, np.abs(xs + 5.0) < 3.0)] = np.nan
    surface = np.zeros(grid.shape, dtype=np.float32)
    surface[4:13, 4:13] = 1.0  # x and y from -19.125 to -16.875 m
    report = tmp / "mission"
    (report / "maps").mkdir(parents=True)
    np.save(report / "maps" / "elevation_mean.npy", z)
    np.save(report / "maps" / "surface_max.npy", surface)
    (report / "maps" / "grid.json").write_text(json.dumps({"elevation_grid": grid.to_dict(), "area": [-20, 20]}))
    (report / "detections.json").write_text(
        json.dumps(
            {
                "frame": "world (local ENU: x east, y north, z up) [m]",
                "objects": [
                    {"id": 1, "class": "landing_pad", "x": -23.0, "y": -18.0, "confirmed": True},
                    {"id": 7, "class": "person", "x": -1.0, "y": -6.0, "confirmed": True},
                    {"id": 5, "class": "person", "x": -5.0, "y": -16.0, "confirmed": True},
                    {"id": 4, "class": "person", "x": -13.0, "y": -18.0, "confirmed": True},
                    {"id": 3, "class": "person", "x": -13.0, "y": 0.0, "confidence": 0.4, "confirmed": False},
                    {"id": 9, "class": "person", "x": -13.0, "y": -6.0, "confirmed": True},
                ],
            }
        )
    )
    return report


def test_rescue_finale_cli_tour_on_a_drone_gcs_report():
    """``--tour`` on a drone_gcs report: all the reachable people in the cheapest order, every one of them with the
    pose-sized footprint, around the box only the surface map shows; ``--tour 3``: the three nearest of those the
    robot can reach, not the three nearest pins."""
    require("isaaclab")
    require("trimesh")
    from drone_lab.teaming.robots import ROBOTS

    radius = ROBOTS["custom"].traversability.robot_radius_m
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        report = _tour_report(tmp)

        r, tour = _cli(tmp / "all", "--report", str(report), "--tour")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert tour["map_mode"] == "drone_map" and tour["pins_source"].endswith("detections.json")
        assert tour["people"] == 4 and tour["max_people"] is None  # the confirmed pins
        assert tour["order"] == ["4", "9", "7"] and tour["unreachable"] == ["5"] and tour["method"] == "held_karp"
        assert "no route to person 5" in r.stdout and "3 of 4 people (1 not reachable)" in r.stdout
        _check_tour_files(tmp / "all", tour, [-23.0, -18.0], 0.7)
        # 10 + 12 + 12 m, give or take the stops' distance from the people and the ways round the box and person 9
        assert 32.0 < tour["length_m"] < 39.0
        # every person has the lying person's 1.2 m circle (a pin without a height has no pose), not the 0.6 m one
        cells = tour["grid"]["nx"] * tour["grid"]["ny"]
        assert tour["grid"]["nx"] == 209 and tour["traversability"]["num_footprints"] == 4
        assert abs(tour["traversability"]["object_fraction"] - 4 * math.pi * 1.2**2 / (0.25**2 * cells)) < 5e-4
        for leg in tour["legs"]:
            assert 1.2 + radius - 0.2 <= leg["stop_to_person_m"] <= 1.2 + radius + 0.36
        # the box the detector did not pin: 81 cells, and the first leg keeps the robot's radius from it
        assert tour["surface_obstacle_cells"] == 81 and "surface map: 81 cells" in r.stdout
        first = np.array(tour["legs"][0]["path_xy"])
        to_box = np.hypot(
            np.clip(np.abs(first[:, 0] + 18.0) - 1.125, 0.0, None),
            np.clip(np.abs(first[:, 1] + 18.0) - 1.125, 0.0, None),
        )
        assert to_box.min() >= radius - math.sqrt(2.0) * 0.25 and np.abs(first[:, 1] + 18.0).max() > 1.125 + radius
        # the legs hand this run's map on, and each leg command plans exactly the route of its leg in the tour
        assert all(_option(words, "--report") == [str(report)] for words in _leg_commands(tmp / "all"))
        for leg, words in zip(tour["legs"], _leg_commands(tmp / "all"), strict=True):
            assert words[-1] == "$@" and "--tour_footprints" in words
            command = [sys.executable, *words[1:-1], "--plan_only"]
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES="")
            )
            assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
            plan = json.loads((Path(_option(words, "--out")[0]) / "plan.json").read_text())
            assert plan["goal_pin"]["id"] == leg["pin"]["id"], leg["leg"]
            np.testing.assert_allclose(plan["path_xy"], leg["path_xy"], atol=1e-6, err_msg=f"leg {leg['leg']}")

        # person 5 is the third nearest pin, but nobody can get there: the third stop is the farther person 7
        options = ("--no_surface_obstacles", "--max_slope", "20")
        r, three = _cli(tmp / "three", "--report", str(report), "--tour", "3", *options)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert three["order"] == ["4", "9", "7"] and three["max_people"] == 3 and three["unreachable"] == ["5"]
        assert three["surface_obstacle_cells"] == 0 and three["traversability_cfg"]["max_slope_deg"] == 20.0
        _check_tour_files(tmp / "three", three, [-23.0, -18.0], 0.7)
        assert three["legs"][0]["length_m"] < tour["legs"][0]["length_m"] - 0.2  # straight through the box
        for words in _leg_commands(tmp / "three"):
            assert "--no_surface_obstacles" in words and _option(words, "--max_slope") == ["20.0"]


if __name__ == "__main__":
    main_for(globals(), "test_teaming_tour")
