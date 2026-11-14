"""Obstacles the detector did not pin: the ground station's surface map (camera depth) against the elevation map.

* the obstacle mask: both maps known and the surface more than the threshold above the terrain;
* the traversability takes the mask as no-go cells and inflates them by the robot radius: a route goes around a box
  that no pin marks, and straight through it without the mask;
* the mask keeps its place in the world when the map grows by the site's apron;
* the surface map of a report folder (present, absent, wrong shape);
* ``scripts/rescue_finale.py --plan_only`` on a drone_gcs report with a surface map (needs env_isaaclab; skipped
  elsewhere).

numpy + scipy only otherwise (``python tests/test_teaming_obstacles.py`` or under pytest).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _runner import main_for, require

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = str(PROJECT / "scripts" / "rescue_finale.py")


def _raises_value_error(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def _distance_to_box(xy: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Distance of the points ``xy`` (N, 2) to the rectangle [x0, x1] x [y0, y1] [m] (0 inside it)."""
    dx = np.maximum(np.maximum(x0 - xy[:, 0], xy[:, 0] - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - xy[:, 1], xy[:, 1] - y1), 0.0)
    return np.hypot(dx, dy)


def test_surface_obstacles_need_both_maps_and_the_height():
    from drone_lab.teaming.inputs import surface_obstacles

    nan = np.nan
    elevation = np.array([[0.0, 0.0, 0.0, 0.0, -2.0], [1.0, 1.0, nan, nan, 1.0]])
    surface = np.array([[0.0, 0.2, 0.25, -1.0, -1.5], [1.5, nan, 3.0, nan, 1.75]])
    # heights above the terrain: 0, 0.2, 0.25, -1, 0.5 | 0.5, not seen, unknown terrain, neither, 0.75
    mask = surface_obstacles(elevation, surface)
    assert mask.dtype == bool
    assert mask.tolist() == [[False, False, True, False, True], [True, False, False, False, True]]
    # exactly at the threshold is not an obstacle; a higher threshold keeps only what is taller
    assert surface_obstacles(elevation, surface, 0.5).tolist() == [[False] * 5, [False, False, False, False, True]]
    assert surface_obstacles(elevation, surface, min_height_m=1.0).sum() == 0
    assert _raises_value_error(surface_obstacles, elevation, surface[:, :4])


def _flat_site_with_a_box():
    """12 m x 8 m of flat ground in 0.1 m cells and the cells of a 1 m x 1 m box around (6, 4)."""
    from drone_lab.mapping import GridSpec

    g = GridSpec(0.0, 0.0, 0.1, 120, 80)
    xs, ys = g.cell_centers()
    box = (np.abs(xs[None, :] - 6.0) < 0.5) & (np.abs(ys[:, None] - 4.0) < 0.5)
    return g, np.zeros(g.shape), box


def test_obstacle_cells_are_no_go_and_inflated_by_the_robot_radius():
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    g, z, box = _flat_site_with_a_box()
    assert box.sum() == 100
    cfg = TraversabilityCfg(robot_radius_m=0.45)
    plain = compute_traversability(z, g, cfg)
    t = compute_traversability(z, g, cfg, obstacles=box)
    assert np.array_equal(t.lethal_obstacles, box) and not t.lethal_objects.any()
    assert t.summary()["surface_obstacle_fraction"] == round(100 / 9600, 4)
    # brute force: no-go = what was no-go already, or a cell centre closer than the robot radius to a box cell centre
    xs, ys = g.cell_centers()
    by, bx = np.nonzero(box)
    d = np.hypot(xs[None, :, None] - xs[bx], ys[:, None, None] - ys[by]).min(axis=2)
    assert np.array_equal(t.lethal, plain.lethal | (d < 0.45))
    assert np.isinf(t.cost[t.lethal]).all() and np.isfinite(t.cost[~t.lethal]).all()
    assert abs(t.sample(t.clearance, np.array([[8.05, 4.05]]))[0] - 1.6) < 1e-9  # 16 cells east of the box
    # beyond the soft band around the box the cost is what it was
    far = d > 0.45 + cfg.inflation_buffer_m + 0.1
    assert np.array_equal(t.cost[far], plain.cost[far])

    # without obstacles nothing changes: the layer is empty and the result is the one of an empty mask
    assert plain.lethal_obstacles.shape == g.shape and not plain.lethal_obstacles.any()
    assert plain.summary()["surface_obstacle_fraction"] == 0.0
    empty = compute_traversability(z, g, cfg, obstacles=np.zeros(g.shape, dtype=bool))
    assert np.array_equal(empty.cost, plain.cost) and np.array_equal(empty.lethal, plain.lethal)

    # the mask is checked: the grid's shape, booleans (not, say, the heights themselves)
    assert _raises_value_error(compute_traversability, z, g, cfg, obstacles=box[:, :-1])
    assert _raises_value_error(compute_traversability, z, g, cfg, obstacles=box[:1])  # one row would broadcast
    assert _raises_value_error(compute_traversability, z, g, cfg, obstacles=box.astype(float))


def test_route_goes_around_a_box_the_detector_did_not_pin():
    from drone_lab.teaming.planner import plan_path
    from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability

    g, z, box = _flat_site_with_a_box()
    cfg = TraversabilityCfg(robot_radius_m=0.45)
    start, goal = (2.0, 4.0), (10.0, 4.0)  # the box lies on the straight line between them
    edges = (5.5, 6.5, 3.5, 4.5)

    # the planner knows objects from the pins only: no pin, so the route runs straight through the box
    through = plan_path(compute_traversability(z, g, cfg), start, goal)
    assert through.success and abs(through.length_m - 8.0) < 1e-6
    assert np.abs(through.path_xy[:, 1] - 4.05).max() < 1e-9  # along the cell row of the start and the goal
    assert (_distance_to_box(through.path_xy, *edges) == 0.0).sum() >= 10

    t = compute_traversability(z, g, cfg, obstacles=box)
    around = plan_path(t, start, goal)
    assert around.success and around.length_m > through.length_m + 0.2
    assert np.allclose(around.path_xy[0], (2.05, 4.05)) and np.allclose(around.path_xy[-1], (10.05, 4.05))
    assert np.abs(around.path_xy[:, 1] - 4.0).max() > 0.5 + 0.45  # passes beside the box and its clearance
    # every cell the route passes through is at least the robot radius from every cell of the box (centre to
    # centre: the distance the inflation uses)
    xs, ys = g.cell_centers()
    by, bx = np.nonzero(box)
    iy, ix, inside = t.world_to_cell(around.path_xy[:, 0], around.path_xy[:, 1])
    assert inside.all()
    d_cells = np.hypot(xs[ix][:, None] - xs[bx], ys[iy][:, None] - ys[by]).min(axis=1)
    assert d_cells.min() >= 0.45
    # the route itself: a route point is up to half a cell diagonal from its cell's centre, and the box edge up to
    # half a cell diagonal from the centre of a box cell
    assert _distance_to_box(around.path_xy, *edges).min() >= 0.45 - math.sqrt(2.0) * g.resolution


def test_mask_keeps_its_place_on_a_grown_grid():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.inputs import extend_with_apron, mask_on_grid

    # by hand: two more columns in the west, one more row in the south, one column east, two rows north
    g = GridSpec(1.0, 2.0, 0.5, 4, 3)
    mask = np.zeros(g.shape, dtype=bool)
    mask[0, 0] = mask[1, 2] = mask[2, 3] = True
    out = mask_on_grid(mask, g, GridSpec(0.0, 1.5, 0.5, 7, 6))
    expected = np.zeros((6, 7), dtype=bool)
    expected[1, 2] = expected[2, 4] = expected[3, 5] = True
    assert out.dtype == bool and np.array_equal(out, expected)
    assert np.array_equal(mask_on_grid(mask, g, g), mask)  # the same grid: unchanged

    # a survey map grown by the site's apron: every marked cell stays at its world position, nothing is added
    grid = GridSpec(-20.125, -20.125, 0.25, 161, 161)
    mask = np.random.default_rng(3).random(grid.shape) < 0.02
    mask[0, 0] = mask[-1, -1] = True  # the map's corners
    _, new, _ = extend_with_apron(np.zeros(grid.shape), grid, (-20.0, 20.0, -20.0, 20.0), (-26.0, 26.0, -26.0, 26.0))
    assert new.shape == (209, 209)
    out = mask_on_grid(mask, grid, new)

    def marked(m, grd):
        xs, ys = grd.cell_centers()  # multiples of 0.125 m: exact in floating point
        iy, ix = np.nonzero(m)
        return sorted(zip(xs[ix].tolist(), ys[iy].tolist(), strict=True))

    assert out.shape == new.shape and out.sum() == mask.sum()
    assert marked(out, new) == marked(mask, grid)

    # only a grid with the same cells that contains the map will do
    assert _raises_value_error(mask_on_grid, mask[:, :-1], grid, new)  # not the mask of that grid
    assert _raises_value_error(mask_on_grid, mask, grid, GridSpec(-26.125, -26.125, 0.5, 105, 105))  # other cells
    assert _raises_value_error(mask_on_grid, mask, grid, GridSpec(-26.0, -26.125, 0.25, 209, 209))  # half a cell off
    assert _raises_value_error(mask_on_grid, mask, grid, GridSpec(-20.125, -20.125, 0.25, 160, 161))  # too small
    assert _raises_value_error(mask_on_grid, mask, grid, GridSpec(-19.125, -26.125, 0.25, 209, 209))  # cuts the west


def test_load_surface_from_a_report_folder():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.inputs import load_elevation, load_surface

    grid = GridSpec(-2.0, -1.0, 0.5, 8, 6)
    surface_max = np.arange(48, dtype=np.float32).reshape(6, 8)
    surface_max[2, 3] = np.nan  # the camera saw nothing there
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "mission"
        (report / "maps").mkdir(parents=True)
        np.save(report / "maps" / "elevation_mean.npy", np.zeros(grid.shape, dtype=np.float32))
        (report / "maps" / "grid.json").write_text(json.dumps({"elevation_grid": grid.to_dict()}))
        _, loaded_grid, _ = load_elevation(report)

        # an older report: no surface map
        assert load_surface(report, loaded_grid) is None
        # a drone_lab survey run never has one
        survey = Path(tmp) / "survey"
        survey.mkdir()
        np.save(survey / "elevation_map.npy", np.zeros(grid.shape, dtype=np.float32))
        assert load_surface(survey, grid) is None

        np.save(report / "maps" / "surface_max.npy", surface_max)
        np.save(report / "maps" / "surface_mean.npy", surface_max - 0.5)
        np.save(report / "maps" / "surface_count.npy", np.ones(grid.shape, dtype=np.int32))
        surface = load_surface(report, loaded_grid)
        assert surface.dtype == np.float64 and surface.shape == (6, 8)
        assert np.array_equal(surface, surface_max, equal_nan=True)  # the highest points, not the mean
        assert load_surface(str(report), loaded_grid) is not None  # a path as text works too

        # a surface map of another grid is refused
        np.save(report / "maps" / "surface_max.npy", surface_max.T)
        assert _raises_value_error(load_surface, report, loaded_grid)
        np.save(report / "maps" / "surface_max.npy", surface_max[:, :-1])
        assert _raises_value_error(load_surface, report, loaded_grid)


# --------------------------------------------------------------------------------------------------- the CLI
BOX = (-19.125, -16.875, -19.125, -16.875)
"""Edges (x0, x1, y0, y1) of the box in the synthetic report [m]: cells 4..12 of the survey grid in x and in y."""


def _flat_mission_report(tmp: Path) -> Path:
    """A drone_gcs mission report of a flat site: the landing pad and one person 20 m east of it are pinned. The
    surface map also shows a 2.25 m x 2.25 m box, 1 m tall, on the straight line between them (81 cells, not
    pinned), a 0.3 m tall object off the route (4 cells), cells the camera did not see, and something tall on
    terrain the drone did not map."""
    from drone_lab.mapping import GridSpec
    from drone_lab.paths import Area

    grid = GridSpec.for_survey(Area(-20.0, 20.0, -20.0, 20.0), 0.25)
    assert (grid.x_min, grid.y_min, grid.shape) == (-20.125, -20.125, (161, 161))
    z = np.zeros(grid.shape, dtype=np.float32)
    z[120:125, 120:125] = np.nan
    surface = np.zeros(grid.shape, dtype=np.float32)
    surface[4:13, 4:13] = 1.0
    surface[40:42, 40:42] = 0.3
    surface[100:110, 100:110] = np.nan
    surface[120:125, 120:125] = 2.0
    report = tmp / "mission"
    (report / "maps").mkdir(parents=True)
    np.save(report / "maps" / "elevation_mean.npy", z)
    np.save(report / "maps" / "surface_max.npy", surface)
    np.save(report / "maps" / "surface_mean.npy", 0.8 * surface)
    np.save(report / "maps" / "surface_count.npy", np.isfinite(surface).astype(np.int32))
    (report / "maps" / "grid.json").write_text(json.dumps({"elevation_grid": grid.to_dict(), "area": [-20, 20]}))
    (report / "detections.json").write_text(
        json.dumps(
            {
                "frame": "world (local ENU: x east, y north, z up) [m]",
                "objects": [
                    {"id": 1, "class": "landing_pad", "x": -23.0, "y": -18.0, "confirmed": True},
                    {"id": 2, "class": "person", "x": -3.0, "y": -18.0, "confirmed": True},
                ],
            }
        )
    )
    return report


def _cli(out: Path, *args: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, SCRIPT, "--plan_only", "--out", str(out), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(PROJECT),
    )
    plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").is_file() else None
    return r, plan


def test_rescue_finale_cli_plans_around_what_the_surface_map_shows():
    """A box no pin marks stands on the straight line from the landing pad to the person: with the report's surface
    map the route goes around it; with --no_surface_obstacles, or a report without the map, it runs through it."""
    require("isaaclab")
    require("trimesh")
    from drone_lab.teaming.robots import ROBOTS

    radius = ROBOTS["custom"].traversability.robot_radius_m
    box_centres = -19.0 + 0.25 * np.arange(9)  # cell centres of the box, the same in x and in y [m]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        report = _flat_mission_report(tmp)

        r, plan = _cli(tmp / "surface", "--report", str(report))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert plan["plan"]["success"] and plan["start_xy"] == [-23.0, -18.0] and plan["goal_pin"]["id"] == "2"
        # the box and the low object, counted on the map grown by the apron (209 x 209 cells)
        assert plan["apron_cells"] > 0 and plan["grid"]["nx"] == 209
        assert plan["surface_obstacle_cells"] == 81 + 4
        assert plan["traversability"]["surface_obstacle_fraction"] == round(85 / 209**2, 4)
        assert "surface map: 85 cells" in r.stdout and "85 of them outside every pin's footprint" in r.stdout
        path = np.array(plan["path_xy"])
        # every cell of the route is at least the robot radius from every cell of the box (centre to centre; the
        # box is a full square of cells, so its nearest cell is the nearest one in x and in y)
        g = plan["grid"]
        cx = g["x_min"] + (np.floor((path[:, 0] - g["x_min"]) / 0.25) + 0.5) * 0.25
        cy = g["y_min"] + (np.floor((path[:, 1] - g["y_min"]) / 0.25) + 0.5) * 0.25
        d_cells = np.hypot(np.abs(cx[:, None] - box_centres).min(axis=1), np.abs(cy[:, None] - box_centres).min(axis=1))
        assert d_cells.min() >= radius
        assert _distance_to_box(path, *BOX).min() >= radius - math.sqrt(2.0) * 0.25
        assert np.abs(path[:, 1] + 18.0).max() > 1.125 + radius  # beside the box and its clearance

        r, pins_only = _cli(tmp / "pins_only", "--report", str(report), "--no_surface_obstacles")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert "--no_surface_obstacles: obstacles come from the pins only" in r.stdout
        assert pins_only["plan"]["success"] and pins_only["surface_obstacle_cells"] == 0
        assert pins_only["traversability"]["surface_obstacle_fraction"] == 0.0
        through = np.array(pins_only["path_xy"])
        assert (_distance_to_box(through, *BOX) == 0.0).sum() >= 10  # straight through the box
        assert pins_only["plan"]["length_m"] < plan["plan"]["length_m"] - 0.2

        # a higher threshold leaves the 0.3 m object out
        r, tall = _cli(tmp / "tall", "--report", str(report), "--obstacle_height", "0.5")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert tall["surface_obstacle_cells"] == 81 and "more than 0.5 m above" in r.stdout

        # an older report (no surface map): one line in the log, the plan of the pins alone
        (report / "maps" / "surface_max.npy").unlink()
        r, old = _cli(tmp / "old", "--report", str(report))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert r.stdout.count("the report has no surface map") == 1 and "surface map: " not in r.stdout
        assert old["surface_obstacle_cells"] == 0 and old["path_xy"] == pins_only["path_xy"]


if __name__ == "__main__":
    main_for(globals(), "test_teaming_obstacles")
