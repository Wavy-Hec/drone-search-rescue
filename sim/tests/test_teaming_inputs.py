"""Finale inputs and figures: pins in the drone_gcs / plain formats, drone_lab and drone_gcs elevation maps, the flat
apron added around a survey map (the landing pad lies outside it), footprints, goal selection, and the plan / walk /
minimap figures (numpy + matplotlib, no simulator)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from _runner import main_for


def test_load_pins_drone_gcs_and_plain_formats():
    from drone_lab.teaming.inputs import load_pins

    with tempfile.TemporaryDirectory() as tmp:
        gcs = Path(tmp) / "detections.json"
        gcs.write_text(json.dumps({
            "frame": "world (local ENU) [m]",
            "objects": [
                {"id": 1, "class": "person", "x": 2.2, "y": -14.7, "z": 3.5, "confidence": 0.92, "confirmed": True},
                {"id": 2, "class": "car", "x": 7.5, "y": -14.8, "confirmed": True},
                {"id": 3, "class": "person", "x": 1.0, "y": 1.0, "confirmed": False},
                {"id": 4, "label": "Traffic Cone", "position": [3.0, 4.0, 0.1]},
                {"id": 5, "x": 1.0},  # malformed: skipped
            ],
        }))
        pins = load_pins(gcs)
        assert [p.id for p in pins] == ["1", "2", "4"]
        assert pins[0].label == "person" and abs(pins[0].confidence - 0.92) < 1e-12 and pins[0].z == 3.5
        assert pins[2].label == "traffic_cone" and (pins[2].x, pins[2].y) == (3.0, 4.0)
        assert len(load_pins(gcs, confirmed_only=False)) == 4
        plain = Path(tmp) / "pins.json"
        plain.write_text(json.dumps([{"class": "crate", "xy": [1, 2], "radius": 1.5}]))
        (p,) = load_pins(plain)
        assert p.label == "crate" and p.radius == 1.5 and p.id == "0"
        wrapped = Path(tmp) / "wrapped.json"
        wrapped.write_text(json.dumps({"pins": [{"cls": "barrel", "pos": [0, 0]}]}))
        assert load_pins(wrapped)[0].label == "barrel"
        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps({"nothing": 1}))
        try:
            load_pins(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a file without pins must be rejected")


def test_ground_truth_records_become_pins_with_exact_footprints():
    """drone_lab.sar ground-truth records: the box centre (not the asset origin at the feet) and the outline."""
    from drone_lab.teaming.inputs import footprints_from_pins, pins_from_data

    records = [
        {"id": 9, "name": "person_03", "class": "person", "category": "person", "pose": "lying_face_down",
         "x": -4.2, "y": -11.0, "position": [-4.2, -11.0, 0.0], "center": [-5.09, -10.97, 0.22],
         "footprint_xy": [[-6.0, -11.3], [-4.2, -11.3], [-4.2, -10.6], [-6.0, -10.6]]},
        {"id": 30, "name": "tree_01", "class": None, "category": "tree", "x": 3.0, "y": 4.0,
         "center": [3.0, 4.0, 2.0], "footprint_xy": [[2.5, 3.5], [3.5, 3.5], [3.5, 4.5], [2.5, 4.5]]},
        {"id": 0, "name": "landing_pad_00", "class": "landing_pad", "category": "landing_pad", "x": -23, "y": -18},
    ]
    pins = pins_from_data(records)
    assert [p.id for p in pins] == ["person_03", "tree_01", "landing_pad_00"]
    person, tree, pad = pins
    assert (person.x, person.y, person.pose) == (-5.09, -10.97, "lying_face_down")
    assert tree.label == "tree" and len(person.polygon) == 4
    fps = footprints_from_pins(pins)
    assert [f.label for f in fps] == ["person", "tree"]  # the pad is walkable
    assert fps[0].polygon == person.polygon and abs(fps[0].radius - np.hypot(0.91, 0.37)) < 0.01


def test_footprints_and_goal_selection():
    from drone_lab.teaming.inputs import DEFAULT_FOOTPRINT_RADII, Pin, footprints_from_pins, select_goal_pin

    pins = [
        Pin("a", "person", 5.0, 5.0),
        Pin("b", "person", -3.0, 0.0),
        Pin("c", "landing_pad", 0.0, 0.0),
        Pin("d", "vehicle", 8.0, 1.0),
        Pin("e", "mystery", 1.0, 1.0),
        Pin("f", "crate", 2.0, 2.0, radius=0.3),
    ]
    fps = footprints_from_pins(pins, radii={"vehicle": 3.0})
    by_label = {f.label: f.radius for f in fps}
    assert "landing_pad" not in by_label  # walkable
    assert by_label["vehicle"] == 3.0 and by_label["person"] == DEFAULT_FOOTPRINT_RADII["person"]
    assert by_label["mystery"] == 0.5 and by_label["crate"] == 0.3
    assert select_goal_pin(pins, near_xy=(0.0, 0.0)).id == "b"
    assert select_goal_pin(pins, pin_id="a").id == "a"
    assert select_goal_pin(pins, pin_id="zz") is None
    assert select_goal_pin(pins, label="tree") is None


def test_load_elevation_from_drone_lab_and_drone_gcs_runs():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.inputs import load_elevation

    grid = GridSpec(-20.125, -20.125, 0.25, 161, 161)
    z = np.random.default_rng(0).normal(size=grid.shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp) / "survey"
        lab.mkdir()
        np.save(lab / "elevation_map.npy", z)
        (lab / "map_meta.json").write_text(json.dumps({"source": "isaac_sim", "grid": grid.to_dict()}))
        z2, g2, desc = load_elevation(lab)
        assert g2 == grid and np.allclose(z2, z) and "elevation_map.npy" in desc
        gcs = Path(tmp) / "gcs"
        (gcs / "maps").mkdir(parents=True)
        np.save(gcs / "maps" / "elevation_max.npy", z)
        (gcs / "maps" / "grid.json").write_text(json.dumps({"elevation_grid": grid.to_dict(), "note": "..."}))
        z3, g3, _ = load_elevation(gcs, layer="max")
        assert g3 == grid and np.allclose(z3, z)
        bare = Path(tmp) / "bare.npy"
        np.save(bare, z[:10, :12])
        gj = Path(tmp) / "g.json"
        gj.write_text(json.dumps({"x_min": 0, "y_min": 0, "resolution": 0.5, "nx": 12, "ny": 10}))
        z4, g4, _ = load_elevation(bare, gj)
        assert g4.shape == (10, 12) and z4.shape == (10, 12)
        try:
            load_elevation(bare, lab / "map_meta.json")
        except ValueError:
            pass
        else:
            raise AssertionError("a grid that does not match the array must be rejected")


def test_extend_with_apron_keeps_the_drone_cells_and_the_lattice():
    """A survey map (40 m area, half-cell-shifted lattice) grows to the 52 m site: the drone's cells (NaNs too) are
    copied unchanged, the added cells on the flat border get its height, unmapped terrain stays unknown."""
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.inputs import extend_with_apron, on_map

    terrain, site = (-20.0, 20.0, -20.0, 20.0), (-26.0, 26.0, -26.0, 26.0)
    grid = GridSpec(-20.125, -20.125, 0.25, 161, 161)  # GridSpec.for_survey of the 40 m area
    z = np.random.default_rng(1).normal(size=grid.shape)
    z[40:60, 70:90] = np.nan  # a patch the drone did not see
    assert not on_map(grid, (-23.0, -18.0)) and on_map(grid, (-20.0, -18.0))
    z2, g2, apron = extend_with_apron(z, grid, terrain, site)
    assert g2 == GridSpec(-26.125, -26.125, 0.25, 209, 209)  # same cells edges, now covering the whole site
    assert on_map(g2, (-23.0, -18.0)) and z2.shape == g2.shape == apron.shape
    inner = z2[24:185, 24:185]
    assert np.array_equal(inner, z, equal_nan=True) and not apron[24:185, 24:185].any()
    assert np.all(z2[apron] == 0.0)
    # the apron ring is 6 m wide minus the drone's own 0.125 m strip; the outermost cells straddle the site edge
    # (half of them would be beyond the terrain mesh) and stay unknown
    xs, ys = g2.cell_centers()
    assert np.isnan(z2[0, :]).all() and np.isnan(z2[:, -1]).all()
    ix, iy = int(np.argmin(np.abs(xs + 23.0))), int(np.argmin(np.abs(ys + 18.0)))
    assert apron[iy, ix] and z2[iy, ix] == 0.0
    assert apron.sum() == 207 * 207 - 161 * 161
    # a map of only part of the terrain: the unmapped terrain between it and the apron stays unknown
    small = GridSpec(-16.125, -16.125, 0.25, 129, 129)
    z3, g3, apron3 = extend_with_apron(np.zeros(small.shape), small, terrain, site, height=0.5)
    assert g3 == g2 and np.all(z3[apron3] == 0.5)
    ix = int(np.argmin(np.abs(xs + 18.0)))  # x = -18: terrain, outside the small map
    assert np.isnan(z3[100, ix]) and not apron3[100, ix]
    # cells across the terrain edge (x = -20 inside the cell) are neither apron nor made up
    ix = int(np.argmin(np.abs(xs + 20.0)))
    assert xs[ix] - 0.125 < -20.0 < xs[ix] + 0.125 and np.isnan(z3[100, ix])
    # a map that already covers the site is returned unchanged
    z4, g4, apron4 = extend_with_apron(z2, g2, terrain, site)
    assert g4 == g2 and np.array_equal(z4, z2, equal_nan=True) and not apron4.any()
    try:
        extend_with_apron(z[:10], grid, terrain, site)
    except ValueError:
        pass
    else:
        raise AssertionError("a grid that does not match the array must be rejected")


def test_a_start_on_the_apron_plans_from_the_landing_pad():
    """The landing pad lies on the border, off the survey map: without the apron the start is off the map (the
    start's problem, whatever the goal); with it the route starts on the pad."""
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.inputs import extend_with_apron
    from drone_lab.teaming.planner import plan_path
    from drone_lab.teaming.traversability import Footprint, compute_traversability

    grid = GridSpec(-20.125, -20.125, 0.25, 161, 161)
    xs, ys = grid.cell_centers()
    z = 0.3 * np.sin(xs[None, :] / 3.0) * np.cos(ys[:, None] / 4.0)  # gentle hills (< 8 deg)
    z[np.ix_((ys > -14) & (ys < -10), (xs > -18) & (xs < -2))] += 1.0  # a 1 m high block: no-go
    pad, person = (-23.0, -18.0), (-4.5, -6.0)
    fps = [Footprint(*person, 1.0, "person")]
    plan = plan_path(compute_traversability(z, grid, footprints=fps), pad, person)
    assert not plan.success and plan.failure == "start_outside" and plan.start_failed
    z2, g2, apron = extend_with_apron(z, grid, (-20.0, 20.0, -20.0, 20.0), (-26.0, 26.0, -26.0, 26.0))
    t = compute_traversability(z2, g2, footprints=fps)
    plan = plan_path(t, pad, person)
    assert plan.success, plan.message
    assert plan.start_snap_m < 0.2 and np.linalg.norm(plan.path_xy[0] - pad) < 0.2
    # from the apron onto the drone's map and around the block, never on a no-go cell; stops next to the person
    assert np.all(np.isfinite(t.sample(t.cost, plan.path_xy))) and apron[t.world_to_cell(*plan.path_xy[0])[:2]]
    assert 1.0 <= plan.goal_snap_m <= 2.5


def test_ground_truth_elevation_matches_the_survey_terrain():
    from _runner import require

    require("isaaclab")
    require("trimesh")
    from drone_lab.teaming.inputs import ground_truth_elevation

    z, grid, desc = ground_truth_elevation(0.5)
    assert grid.shape == (104, 104) and (grid.x_min, grid.y_min) == (-26.0, -26.0)
    assert np.isfinite(z).all() and -1.6 < z.min() < -1.2 and 3.0 < z.max() < 3.6
    assert abs(float(z[2, 2])) < 1e-9  # the flat border


def test_plan_walk_and_minimap_figures():
    from drone_lab.mapping import GridSpec
    from drone_lab.teaming.inputs import Pin
    from drone_lab.teaming.planner import plan_path
    from drone_lab.teaming.plots import draw_disc, plot_plan, plot_walk, render_minimap
    from drone_lab.teaming.traversability import Footprint, compute_traversability

    g = GridSpec(0.0, 0.0, 0.2, 60, 60)
    z = np.zeros(g.shape)
    z[10:40, 25:30] = 1.0
    z[45:55, 5:15] = np.nan
    person = Pin("goal", "person", 10.0, 3.0)
    fps = [Footprint(10.0, 3.0, 0.6, "person"), Footprint(3.0, 9.0, 0.8, "crate")]
    t = compute_traversability(z, g, footprints=fps)
    plan = plan_path(t, (1.5, 3.0), (10.0, 3.0))
    assert plan.success
    with tempfile.TemporaryDirectory() as tmp:
        for theme in ("light", "dark"):
            p1 = plot_plan(t, plan, Path(tmp) / f"plan_{theme}.png", pins=[person, Pin("2", "crate", 3, 9)],
                           goal_pin=person, source="synthetic", theme=theme)
            assert p1.stat().st_size > 20_000
        walk = {"t": np.linspace(0, 20, 50), "x": plan.path_xy[::4, 0][:50], "y": plan.path_xy[::4, 1][:50],
                "speed": np.full(50, 0.6), "cmd_vx": np.full(50, 0.7), "dist_goal": np.linspace(9, 0.2, 50),
                "falls": [(7.0, 4.0, 3.0)]}
        n = min(len(walk["x"]), 50)
        walk = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in walk.items()}
        p2 = plot_walk(t, plan, walk, Path(tmp) / "walk.png", pins=[person], goal_pin=person,
                       summary={"success": True, "time_s": 20.0, "walked_length_m": 9.0, "planned_length_m": 9.2,
                                "falls": 1, "final_distance_to_person_m": 1.1, "physics": "newton_mjwarp"})
        assert p2.stat().st_size > 20_000
    img, to_px = render_minimap(t, plan, pins=[person], goal_pin=person, size_px=200)
    assert img.shape == (200, 200, 3) and img.dtype == np.uint8
    c, r = to_px(np.array([plan.path_xy[0, 0]]), np.array([plan.path_xy[0, 1]]))
    assert 0 <= c[0] < 200 and 0 <= r[0] < 200
    draw_disc(img, float(c[0]), float(r[0]), 4, (255, 0, 0), ring=(255, 255, 255))
    assert tuple(img[int(r[0]), int(c[0])]) == (255, 0, 0)


if __name__ == "__main__":
    main_for(globals(), "test_teaming_inputs")
