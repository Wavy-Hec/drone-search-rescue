"""Natural people of the SAR site: pose data, placement on the terrain, the rest of the site unchanged, T-pose
sites byte for byte as before, and (with the assets in the local cache) the baked posed meshes.

CPU only; needs env_isaaclab for Isaac Lab's terrain generator, no Kit.
Run: ``CUDA_VISIBLE_DEVICES= python tests/test_sar_people.py`` (or with pytest).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import unittest

import numpy as np
from _runner import main_for, require

_CACHE: dict = {}

TPOSE_GT_SHA256 = {
    0: "b37eb044b7511fcc63f61e317c5f29d145ff9423d5308e3a01d0ab56b67da492",
    1: "d4505fb1fdde9ca0045171744bf735a4582731678d88067b8efac0306a900f0f",
    2: "54f4d9cf1ba9ca880efafd0848764609928d81abf14faab075f9b76ceb4303b0",
}
"""SHA-256 of ``ground_truth.json`` (``indent=1``) of the T-pose sites, taken before natural people existed."""


def _site():
    require("isaaclab")
    require("trimesh")
    if "site" not in _CACHE:
        from drone_lab.sar.scenario import TerrainSite

        _CACHE["site"] = TerrainSite()
    return _CACHE["site"]


def _scenario(seed: int, style: str = "natural"):
    key = (seed, style)
    if key not in _CACHE:
        from drone_lab.sar.scenario import ScenarioCfg, generate_scenario

        _CACHE[key] = generate_scenario(ScenarioCfg(seed=seed, people_style=style), site=_site())
    return _CACHE[key]


def test_pose_data_matches_the_pose_definitions():
    from drone_lab.sar.assets import assets_of
    from drone_lab.sar.people import CELL, POSES, data_digests, pose_geometry

    assert data_digests() == {k: p.digest() for k, p in POSES.items()}, "rerun: python -m drone_lab.sar.people"
    for spec in assets_of("person"):
        for pose in POSES:
            g = pose_geometry(spec.key, pose)
            lo, hi = np.asarray(g.bbox_min), np.asarray(g.bbox_max)
            # origin at the bottom centre of the posed box
            assert abs(lo[2]) < 1e-9 and np.allclose(lo[:2], -hi[:2], atol=2e-4), (spec.key, pose)
            under = g.underside
            assert np.nanmin(under) < 0.005 and np.nanmax(under) <= hi[2] + 1e-3, (spec.key, pose)
            # the height map covers the box
            nx, ny = under.shape
            assert g.origin[0] <= lo[0] + 1e-9 and g.origin[0] + nx * CELL >= hi[0] - 1e-9, (spec.key, pose)
            assert g.origin[1] <= lo[1] + 1e-9 and g.origin[1] + ny * CELL >= hi[1] - 1e-9, (spec.key, pose)
            height, family = hi[2], POSES[pose].key.split("_")[0]
            expect = {"lying": (0.25, 0.6), "sitting": (0.75, 1.1), "standing": (1.5, 2.05)}[family]
            assert expect[0] < height < expect[1], (spec.key, pose, height)


def test_tpose_sites_are_unchanged():
    if os.environ.get("ISAACSIM_ASSET_ROOT"):
        raise unittest.SkipTest("the ground truth records the asset root")
    for seed, digest in TPOSE_GT_SHA256.items():
        gt = _scenario(seed, "tpose").ground_truth("/World/sar")
        assert "people_style" not in gt["scenario_cfg"]
        assert hashlib.sha256(json.dumps(gt, indent=1).encode()).hexdigest() == digest, seed


def test_natural_keeps_the_rest_of_the_site():
    from drone_lab.sar.scenario import RESPONDER_GROUP

    for seed in (0, 1, 2):
        tpose, natural = _scenario(seed, "tpose"), _scenario(seed, "natural")
        a = [(o.name, o.asset, o.center, o.yaw) for o in tpose.objects if o.category != "person"]
        b = [(o.name, o.asset, o.center, o.yaw) for o in natural.objects if o.category != "person"]
        assert a == b, seed
        victims = [o for o in natural.objects if o.category == "person" and o.group != RESPONDER_GROUP]
        before = {o.name: o for o in tpose.by_category("person")}
        assert [o.name for o in victims] == list(before), seed
        for o in victims:  # same character, at most 1 m from its T-pose place
            old = before[o.name]
            assert o.asset == old.asset and math.dist(o.center[:2], old.center[:2]) <= 1.0 + 1e-6, (seed, o.name)
        responders = [o for o in natural.objects if o.group == RESPONDER_GROUP]
        assert 2 <= len(responders) <= 3, seed
        assert natural.objects[-len(responders) :] == responders  # appended after everything else
        gt = natural.ground_truth()
        assert gt["scenario_cfg"]["people_style"] == "natural"


def test_natural_poses_labels_and_roles():
    from drone_lab.sar.assets import CLASS_IDS
    from drone_lab.sar.people import POSES, RESPONDER_POSES, VICTIM_POSES
    from drone_lab.sar.scenario import RESPONDER_ASSETS, RESPONDER_GROUP, pose_family

    seen = set()
    for seed in range(6):
        sc = _scenario(seed)
        people = sc.by_category("person")
        for o in people:
            assert o.pose in POSES and o.labeled, (seed, o.name)
            responder = o.group == RESPONDER_GROUP
            assert o.pose in (RESPONDER_POSES if responder else VICTIM_POSES), (seed, o.name, o.pose)
            if responder:
                assert o.asset in RESPONDER_ASSETS
            height = o.aabb_max[2] - o.aabb_min[2]
            expect = {"lying": (0.25, 0.65), "sitting": (0.7, 1.15), "standing": (1.45, 2.15)}[pose_family(o.pose)]
            assert expect[0] < height < expect[1], (seed, o.name, o.pose, height)
            x, y, z, w = o.quat_xyzw  # posed meshes are baked: only a heading
            assert abs(x) < 1e-12 and abs(y) < 1e-12 and abs(math.atan2(2 * z * w, 1 - 2 * z * z) - o.yaw) < 1e-9
            seen.add(o.pose)
        names = [o.name for o in people]
        assert names == [f"person_{i:02d}" for i in range(len(people))], seed
        for rec in sc.ground_truth()["objects"]:
            if rec["category"] == "person":
                assert (rec["class"], rec["class_id"]) == ("person", CLASS_IDS["person"]) and rec["pose"] in POSES
    assert seen == set(POSES), seen


def test_natural_people_rest_on_the_terrain_and_not_inside_blocks():
    """Re-measured on a grid three times finer than the placement's: no terrain inside a body, the body rests on
    the parts that carry it (under-side <= 2 cm), and the ground drops at most the pose family's tolerance under
    them."""
    from drone_lab.sar.assets import ASSETS
    from drone_lab.sar.people import CELL, pose_geometry
    from drone_lab.sar.scenario import _SUPPORT_GAP, pose_family

    site = _site()
    sub = (np.arange(3) + 0.5) / 3.0 - 0.5
    for seed in (0, 1, 2, 3):
        for o in _scenario(seed).by_category("person"):
            g = pose_geometry(ASSETS[o.asset].key, o.pose)
            under = g.underside * o.scale
            ix, iy = np.nonzero(~np.isnan(under))
            cx = g.origin[0] + (ix + 0.5) * CELL
            cy = g.origin[1] + (iy + 0.5) * CELL
            du, dv = np.meshgrid(sub * CELL, sub * CELL, indexing="ij")
            px = (cx[:, None] + du.ravel()[None]) * o.scale
            py = (cy[:, None] + dv.ravel()[None]) * o.scale
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            wx, wy = o.center[0] + c * px - s * py, o.center[1] + s * px + c * py
            terrain = site.heights(np.column_stack([wx.ravel(), wy.ravel()])).reshape(px.shape)
            assert np.all(np.isfinite(terrain)), (seed, o.name)
            body = o.ground_z + under[ix, iy][:, None]
            assert float(np.max(terrain - body)) <= 0.01, (seed, o.name, float(np.max(terrain - body)))
            carry = under[ix, iy] <= 0.02 * o.scale
            drop = o.ground_z - terrain[carry, 4]  # sample 4 is the middle of the square
            assert float(np.min(body[carry] - terrain[carry])) <= 0.02, (seed, o.name, "does not rest on its support")
            limit = _SUPPORT_GAP[pose_family(o.pose)] + 1e-6
            assert float(np.max(drop)) <= limit, (seed, o.name, o.pose, float(np.max(drop)))
            assert abs(o.aabb_min[2] - o.ground_z) < 1e-9 and abs(o.position[2] - o.ground_z) < 1e-9


def test_natural_spacing_area_and_determinism():
    from drone_lab.sar.scenario import RESPONDER_GROUP, ScenarioCfg, generate_scenario

    cfg = ScenarioCfg()
    for seed in (0, 1, 2):
        sc = _scenario(seed)
        objs = [o for o in sc.objects if o.category != "tree"]
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                gap = cfg.cluster_gap if (a.group is not None and a.group == b.group) else cfg.gap
                if a.group == "pad_cones" or b.group == "pad_cones":
                    gap = 0.0
                assert math.dist(a.center[:2], b.center[:2]) >= a.radius + b.radius + gap - 1e-6, (seed, a.name, b.name)
        x0, x1, y0, y1 = sc.area
        pad = sc.by_category("landing_pad")[0]
        for o in sc.by_category("person"):
            if o.group == RESPONDER_GROUP and math.dist(o.center[:2], pad.center[:2]) < 8.0:
                assert math.dist(o.center[:2], pad.center[:2]) >= cfg.apron_half * 1.2 + o.radius - 1e-6
                continue  # by the pad, on the border
            fp = np.asarray(o.footprint_xy)
            m = cfg.area_margin - 1e-6
            assert fp[:, 0].min() >= x0 + m and fp[:, 0].max() <= x1 - m, (seed, o.name)
            assert fp[:, 1].min() >= y0 + m and fp[:, 1].max() <= y1 - m, (seed, o.name)
    site = _site()
    a = generate_scenario(ScenarioCfg(seed=4, people_style="natural"), site=site).ground_truth()
    np.random.seed(7)
    b = generate_scenario(ScenarioCfg(seed=4, people_style="natural"), site=site).ground_truth()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    try:
        generate_scenario(ScenarioCfg(seed=4, people_style="dancing"), site=site)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown people_style accepted")


def test_seed0_natural_people():
    """Seed 0 is the demo site: its victims keep their names and stay where the T-pose ones were (person_08 still
    lies in the crater), with every victim pose among them, plus responders by the forklift and the helipad."""
    sc, tpose = _scenario(0), _scenario(0, "tpose")
    people = {o.name: o for o in sc.by_category("person")}
    assert people["person_08"].pose.startswith("lying") and people["person_08"].ground_z < -1.2
    assert {o.pose for o in people.values() if o.group is None} == {
        "lying_back",
        "lying_side",
        "sitting",
        "standing_waving",
    }
    for o in tpose.by_category("person"):
        assert math.dist(people[o.name].center[:2], o.center[:2]) <= 1.0 + 1e-6
    responders = [o for o in people.values() if o.group]
    vehicle, pad = sc.by_category("vehicle")[0], sc.by_category("landing_pad")[0]
    assert min(math.dist(o.center[:2], vehicle.center[:2]) for o in responders) < 4.5
    assert min(math.dist(o.center[:2], pad.center[:2]) for o in responders) < 7.0


def test_baked_meshes_match_the_pose_data():
    """With the assets in the local cache: bake one person in two poses and compare with ``people_poses.json``;
    the static USD has no skeleton, the same meshes and textures that exist on disk, and later runs reuse it."""
    require("isaaclab")
    from drone_lab.sar.assets import ASSETS, asset_root
    from drone_lab.sar.people import PoseBaker, geometry_record, pose_geometry
    from drone_lab.sar.scene import asset_cache_dir

    cache = asset_cache_dir()
    mirror = os.path.join(cache or "", "https", asset_root().split("://", 1)[-1], "Isaac", "People", "Characters")
    if not cache or not os.path.isdir(mirror):
        raise unittest.SkipTest("Isaac People assets are not in the local cache")
    import tempfile

    from pxr import Sdf, Usd, UsdGeom

    from isaaclab.utils.assets import retrieve_file_path

    spec = ASSETS["female_business_02"]
    with tempfile.TemporaryDirectory() as out:
        try:
            baker = PoseBaker(lambda url: retrieve_file_path(url, download_dir=cache), asset_root(), out)
            for pose in ("sitting", "lying_side"):
                rec = geometry_record(baker.pose(spec.path, pose))
                ref = pose_geometry(spec.key, pose)
                assert np.allclose(rec["bbox_min"], ref.bbox_min, atol=1e-3) and np.allclose(
                    rec["bbox_max"], ref.bbox_max, atol=1e-3
                ), pose
            path = baker.usd(spec.key, spec.path, "sitting")
            # another process localises fresh copies of the same assets: the same baked file serves it
            again = PoseBaker(lambda url: retrieve_file_path(url, download_dir=cache), asset_root(), out)
            assert again.usd(spec.key, spec.path, "sitting") == path
        except (OSError, RuntimeError) as exc:  # offline and nothing cached
            raise unittest.SkipTest(f"assets not available ({exc})") from None
        stage = Usd.Stage.Open(path)
        types = {p.GetTypeName() for p in stage.Traverse()}
        assert "Skeleton" not in types and "SkelRoot" not in types, types
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert meshes and all("SkelBindingAPI" not in p.GetAppliedSchemas() for p in meshes)
        assets = [
            a.Get().path
            for p in stage.Traverse()
            for a in p.GetAttributes()
            if a.GetTypeName() == Sdf.ValueTypeNames.Asset and a.Get() and a.Get().path.endswith(".png")
        ]
        assert assets and all(os.path.isfile(a) for a in assets), assets[:3]
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(stage.GetDefaultPrim())
        lo, hi = box.ComputeAlignedRange().GetMin(), box.ComputeAlignedRange().GetMax()
        g = pose_geometry(spec.key, "sitting")
        assert np.allclose(lo, g.bbox_min, atol=2e-3) and np.allclose(hi, g.bbox_max, atol=2e-3), (lo, hi)


if __name__ == "__main__":
    main_for(globals(), "test_sar_people")
