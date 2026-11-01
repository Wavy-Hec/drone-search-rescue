"""SAR scene generator: determinism, placement on the terrain (height queries), no objects inside blocks, spacing,
classes and the ground-truth JSON (CPU only; needs env_isaaclab for Isaac Lab's terrain generator, no Kit).

Run: ``CUDA_VISIBLE_DEVICES= python tests/test_sar_scenario.py`` (or with pytest).
"""

from __future__ import annotations

import json
import math

import numpy as np
from _runner import main_for, require

_CACHE: dict = {}


def _site():
    require("isaaclab")
    require("trimesh")
    if "site" not in _CACHE:
        from drone_lab.sar.scenario import TerrainSite

        _CACHE["site"] = TerrainSite()
    return _CACHE["site"]


def _scenario(seed: int = 0):
    key = ("scenario", seed)
    if key not in _CACHE:
        from drone_lab.sar.scenario import ScenarioCfg, generate_scenario

        _CACHE[key] = generate_scenario(ScenarioCfg(seed=seed), site=_site())
    return _CACHE[key]


def _rect_points(obj, shrink: float = 0.0, step: float = 0.05) -> np.ndarray:
    """Dense xy samples inside an object's oriented footprint rectangle (optionally shrunk)."""
    c = np.asarray(obj.footprint_xy, dtype=float)
    ctr = c.mean(axis=0)
    ax = c[1] - c[0]
    ay = c[3] - c[0]
    lx, ly = np.linalg.norm(ax), np.linalg.norm(ay)
    ux, uy = ax / lx, ay / ly
    s = np.linspace(-0.5 * lx + shrink, 0.5 * lx - shrink, max(2, int(lx / step)))
    t = np.linspace(-0.5 * ly + shrink, 0.5 * ly - shrink, max(2, int(ly / step)))
    ss, tt = np.meshgrid(s, t, indexing="ij")
    return ctr + ss.reshape(-1, 1) * ux + tt.reshape(-1, 1) * uy


def test_generator_is_deterministic():
    from drone_lab.sar.scenario import ScenarioCfg, generate_scenario

    site = _site()
    a = generate_scenario(ScenarioCfg(seed=7), site=site).ground_truth()
    b = generate_scenario(ScenarioCfg(seed=7), site=site).ground_truth()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = generate_scenario(ScenarioCfg(seed=8), site=site).ground_truth()
    assert json.dumps(a["objects"]) != json.dumps(c["objects"])
    # global numpy RNG untouched and irrelevant
    np.random.seed(123)
    state = np.random.get_state()[1].copy()
    d = generate_scenario(ScenarioCfg(seed=7), site=site).ground_truth()
    assert np.array_equal(state, np.random.get_state()[1])
    assert json.dumps(a, sort_keys=True) == json.dumps(d, sort_keys=True)


def test_class_list_and_counts():
    from drone_lab.sar.assets import ASSETS, CLASS_IDS, CLASSES, UNLABELED

    assert CLASSES == ("person", "vehicle", "crate", "barrel", "pallet", "cone", "landing_pad")
    assert [CLASS_IDS[c] for c in CLASSES] == list(range(7))
    assert {a.category for a in ASSETS.values()} == set(CLASSES) - {"landing_pad"} | {UNLABELED}
    for seed in (0, 1, 2, 3):
        sc = _scenario(seed)
        n = sc.counts()
        assert 6 <= n["person"] <= 10, (seed, n)
        assert n["vehicle"] == 1 and n["landing_pad"] == 1, (seed, n)
        assert n["crate"] >= 5 and n["barrel"] >= 3 and n["pallet"] >= 3 and n["cone"] >= 5, (seed, n)
        assert 3 <= n[UNLABELED] <= 6, (seed, n)
        people = sc.by_category("person")
        assert any(p.pose.startswith("lying") for p in people) and any(p.pose == "standing" for p in people)
        assert sum(p.near_block for p in people) >= 2, "some people must lie / stand next to a collapsed building"
        assert all(p.near_step for p in people if p.near_block)
        for o in sc.objects:
            assert o.labeled == (o.category != UNLABELED), o.name
            assert o.category in CLASSES or o.category == UNLABELED
        assert len({o.name for o in sc.objects}) == len(sc.objects)


def test_objects_rest_on_the_terrain_and_not_inside_blocks():
    """Height queries on the terrain mesh: nothing floats, nothing sinks into the ground or a block."""
    from drone_lab.sar.assets import ASSETS
    from drone_lab.sar.scenario import _person_parts

    site = _site()
    for seed in (0, 1, 2):
        sc = _scenario(seed)
        for o in sc.objects:
            bottom = o.aabb_min[2]
            if o.category == "tree":
                trunk = site.heights(np.array(o.position[:2])[None] + np.array([[0, 0], [0.3, 0], [0, 0.3]]))
                assert abs(o.position[2] - np.median(trunk)) < 0.35, o.name
                continue
            if o.category == "person":
                spec = ASSETS[o.asset]
                body_lo, body_hi, arm_lo, arm_hi = _person_parts(spec.bbox_min, spec.bbox_max)
                from drone_lab.sar.scenario import ScenarioCfg, _Placer

                placer = _Placer(site, ScenarioCfg(), np.random.default_rng(0))
                body = placer.posed_box_points(body_lo, body_hi, o.pose, o.yaw, o.scale, o.position[:2])
                arms = placer.posed_box_points(arm_lo, arm_hi, o.pose, o.yaw, o.scale, o.position[:2])
                zb, za = site.heights(body), site.heights(arms)
                assert np.all(np.isfinite(zb)) and np.all(np.isfinite(za)), o.name
                assert zb.max() <= bottom + 0.02 and zb.min() >= bottom - 0.16, (o.name, zb.min(), zb.max(), bottom)
                if o.pose == "standing":
                    assert za.max() <= bottom + 1.0, (o.name, "terrain reaches the arms")
                else:
                    assert za.max() <= bottom + 0.02 and za.min() >= bottom - 0.16, (o.name, "arms not on the ground")
                continue
            z = site.heights(_rect_points(o))
            assert np.all(np.isfinite(z)), o.name
            # rests on the highest point under it: no terrain (block, step, wall) pokes into the object ...
            assert z.max() <= bottom + 0.02, (seed, o.name, float(z.max()), bottom)
            # ... and it does not hang over an edge or float: the ground under it is flat within the tolerance
            assert z.min() >= bottom - 0.16, (seed, o.name, float(z.min()), bottom)
            assert abs(o.ground_z - bottom) < 1e-6, o.name


def test_spacing_and_area():
    from drone_lab.sar.scenario import ScenarioCfg

    cfg = ScenarioCfg()
    for seed in (0, 1, 2):
        sc = _scenario(seed)
        objs = [o for o in sc.objects if o.category != "tree"]
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                d = math.dist(a.center[:2], b.center[:2])
                gap = cfg.cluster_gap if (a.group is not None and a.group == b.group) else cfg.gap
                if a.group == "pad_cones" or b.group == "pad_cones":
                    gap = 0.0
                assert d >= a.radius + b.radius + gap - 1e-6, (seed, a.name, b.name, d)
        x0, x1, y0, y1 = sc.area
        for o in objs:
            if o.category == "landing_pad" or o.group == "pad_cones":
                continue
            fp = np.asarray(o.footprint_xy)
            m = cfg.area_margin - 1e-6
            assert fp[:, 0].min() >= x0 + m and fp[:, 0].max() <= x1 - m, o.name
            assert fp[:, 1].min() >= y0 + m and fp[:, 1].max() <= y1 - m, o.name
            assert o.in_area


def test_people_poses_and_trees_below_survey_altitude():
    for seed in (0, 1, 2):
        sc = _scenario(seed)
        for p in sc.by_category("person"):
            height = p.aabb_max[2] - p.aabb_min[2]
            if p.pose.startswith("lying"):
                assert height < 0.5, (p.name, height)  # flat on the ground
                assert max(p.half_extents[:2]) * 2 > 1.5  # body length lies horizontally
            else:
                assert 1.5 < height < 1.95, (p.name, height)
            # the quaternion is a unit quaternion and a lying person is rotated 90 deg about a horizontal axis
            q = np.asarray(p.quat_xyzw)
            assert abs(np.linalg.norm(q) - 1) < 1e-9
            x, y, z, w = q
            up = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])  # R @ e_z
            assert (abs(up[2]) < 1e-6) if p.pose.startswith("lying") else (abs(up[2] - 1) < 1e-6), p.name
        for t in sc.by_category("tree"):
            assert t.aabb_max[2] <= sc.flight_z - sc.cfg.tree_top_margin + 1e-6, (t.name, t.aabb_max[2])


def test_near_block_means_next_to_a_collapsed_building():
    """``near_step``: terrain >= 0.6 m above the person's ground within ~1.3 m of its footprint; ``near_block``: such
    a step on a blocks / stairs tile. Re-measured on a fine 5 cm grid around every person (with a margin either way
    for the generator's 25 cm grid): a pyramid terrace, the crater wall or a hill is a step but not a building."""
    from drone_lab.sar.scenario import _CLEARANCE, BUILDING_TILES

    site = _site()

    def steps(p, grow: float, rise: float) -> list[str]:
        """Tiles of the points >= ``rise`` above the person's ground, in its footprint grown by ~1.3 m + ``grow``."""
        reach = _CLEARANCE["person_lying" if p.pose.startswith("lying") else "person_standing"] + 1.3 + grow
        hx, hy = p.half_extents[0] + reach, p.half_extents[1] + reach
        u, w = np.meshgrid(np.arange(-hx, hx + 1e-9, 0.05), np.arange(-hy, hy + 1e-9, 0.05), indexing="ij")
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        xy = np.column_stack([u.ravel() * c - w.ravel() * s, u.ravel() * s + w.ravel() * c]) + np.asarray(p.center[:2])
        high = site.heights(xy) >= p.ground_z + rise
        return [site.tile_of(float(x), float(y)) for x, y in xy[high]]

    for seed in (0, 1, 2):
        for p in _scenario(seed).by_category("person"):
            loose, strict = steps(p, 0.3, 0.5), steps(p, -0.5, 0.7)
            assert (bool(loose) or not p.near_step) and (p.near_step or not strict), (seed, p.name, p.near_step)
            built_loose = any(t in BUILDING_TILES for t in loose)
            built_strict = any(t in BUILDING_TILES for t in strict)
            assert (built_loose or not p.near_block) and (p.near_block or not built_strict), (seed, p.name)
    people = _scenario(0).by_category("person")
    # seed 0: person_01 stands by a pyramid terrace, person_05 below a hill, person_08 in the crater
    assert {p.name for p in people if p.near_step and not p.near_block} == {"person_01", "person_05", "person_08"}
    assert sum(p.near_block for p in people) == 6


def test_seed0_layout_is_stable():
    """Seed 0 is the demo site: the drone mission, its figures and the rescue finale's documented runs
    (``--sar_seed 0 --goal_pin person_08``) refer to these people by name. A change here moves them all."""
    expected = {
        "landing_pad_00": ("upright", (-23.00, -18.00, 0.00)),
        "vehicle_00": ("upright", (-8.46, -13.16, 0.01)),
        "person_00": ("standing", (17.75, -13.35, 1.42)),
        "person_01": ("standing", (10.70, 18.73, 0.18)),
        "person_02": ("lying_face_up", (-10.52, -17.35, 0.00)),
        "person_03": ("lying_face_down", (-5.09, -10.97, 0.00)),
        "person_04": ("lying_face_up", (-11.99, -10.42, 0.00)),
        "person_05": ("standing", (-17.61, -0.42, -0.01)),
        "person_06": ("lying_face_up", (-9.35, 11.56, 0.09)),
        "person_07": ("lying_face_down", (-4.05, -14.90, 0.00)),
        "person_08": ("lying_face_down", (-14.77, -4.85, -1.31)),
    }
    sc = _scenario(0)
    got = {o.name: o for o in sc.objects if o.category in ("person", "vehicle", "landing_pad")}
    assert set(got) == set(expected), sorted(got)
    for name, (pose, (x, y, ground)) in expected.items():
        o = got[name]
        assert o.pose == pose, (name, o.pose)
        assert np.allclose([o.center[0], o.center[1], o.ground_z], [x, y, ground], atol=0.011), (name, o.center)
    assert len(sc.objects) == 43 and sc.counts()["crate"] == 9 and sc.counts()["tree"] == 4, sc.counts()


def test_landing_pad_and_home():
    from drone_lab.paths import Area, lawnmower

    sc = _scenario(0)
    pads = sc.by_category("landing_pad")
    assert len(pads) == 1
    pad = pads[0]
    assert pad.ground_spread < 0.01 and not pad.in_area
    assert np.allclose(sc.home[:2], pad.center[:2])
    x0, x1, y0, y1 = sc.area
    start = lawnmower(Area(x0, x1, y0, y1), sc.cfg.survey_spacing, sc.flight_z)[0]
    assert math.dist(start[:2], sc.home[:2]) <= 4.0, (start, sc.home)
    cones = [o for o in sc.objects if o.group == "pad_cones"]
    assert len(cones) == 4 and all(o.category == "cone" for o in cones)


def test_ground_truth_json():
    from drone_lab.sar.assets import CLASSES
    from drone_lab.sar.scenario import GROUND_TRUTH_FORMAT

    sc = _scenario(0)
    gt = json.loads(json.dumps(sc.ground_truth()))
    assert gt["format"] == GROUND_TRUTH_FORMAT and gt["classes"] == list(CLASSES)
    assert gt["home"]["landing_pad"] == "landing_pad_00"
    assert sum(gt["counts"].values()) == len(gt["objects"]) == len(sc.objects)
    for rec, o in zip(gt["objects"], sc.objects, strict=False):
        assert rec["prim_path"] == f"/World/sar/objects/{o.name}"
        assert (rec["x"], rec["y"], rec["z"]) == tuple(round(v, 3) for v in o.center)
        assert rec["class"] == (o.category if o.labeled else None)
        assert rec["url"] is None if o.category == "landing_pad" else rec["url"].startswith("http")
        for k in ("position", "center", "size", "aabb_min", "aabb_max", "quat_xyzw", "yaw_deg", "pose", "in_area"):
            assert k in rec, k
    assert gt["camera"]["width"] == 640 and gt["survey"]["flight_z"] > 7.0


def test_ground_material_assignment():
    """Every terrain face gets a ground material: grass hills / border, concrete block tops and walls, ..."""
    site, sc = _site(), _scenario(0)  # skips first when Isaac Lab is missing
    from drone_lab.sar.assets import GROUND_MATERIALS
    from drone_lab.sar.scene import terrain_face_materials

    keys = terrain_face_materials(site, sc, site.vertices, site.faces)
    assert len(keys) == len(site.faces) and set(keys.tolist()) <= set(GROUND_MATERIALS)
    tri = site.vertices[site.faces]
    c = tri.mean(axis=1)

    def at(x, y):  # material of the face nearest to (x, y) (top surface)
        i = np.argsort(np.hypot(c[:, 0] - x, c[:, 1] - y))[:20]
        return keys[i[np.argmax(c[i, 2])]]

    tiles = {name: (r, col) for (r, col), name in site.tiles.items()}
    for name, expected in (("hills", "grass"), ("pit", "soil"), ("stairs", "concrete"), ("rough", "gravel")):
        if name in tiles:
            r, col = tiles[name]
            x, y = site.area.x_min + (r + 0.5) * 10.0, site.area.y_min + (col + 0.5) * 10.0
            assert at(x, y) == expected, (name, at(x, y))
    assert at(-24.0, 10.0) == "grass"  # border
    blocks = [(r, col) for (r, col), name in site.tiles.items() if name == "blocks"]
    sel = np.zeros(len(c), bool)
    for r, col in blocks:
        x0, y0 = site.area.x_min + r * 10.0, site.area.y_min + col * 10.0
        sel |= (c[:, 0] >= x0) & (c[:, 0] < x0 + 10) & (c[:, 1] >= y0) & (c[:, 1] < y0 + 10)
    assert set(keys[sel & (c[:, 2] > 0.2)].tolist()) == {"concrete"}  # block tops / walls
    assert "gravel" in set(keys[sel & (c[:, 2] < 0.01)].tolist())  # rubble between the blocks


if __name__ == "__main__":
    main_for(globals(), "test_sar_scenario")
