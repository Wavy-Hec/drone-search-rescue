"""Views, lights, ground looks and the padded camera of the detector dataset (CPU only; needs env_isaaclab for
Isaac Lab's terrain generator, no Kit).

Run: ``CUDA_VISIBLE_DEVICES= python tests/test_sar_capture.py`` (or with pytest).
"""

from __future__ import annotations

import math

import numpy as np
from _runner import main_for, require

_CACHE: dict = {}


def _site_and_scenario(seed: int = 1000):
    require("isaaclab")
    require("trimesh")
    from drone_lab.sar.scenario import ScenarioCfg, TerrainSite, generate_scenario

    if "site" not in _CACHE:
        _CACHE["site"] = TerrainSite()
    if seed not in _CACHE:
        _CACHE[seed] = generate_scenario(ScenarioCfg(seed=seed), site=_CACHE["site"])
    return _CACHE["site"], _CACHE[seed]


def _pixel_of(view, point) -> tuple[float, float]:
    """Pixel of a world point in the SAR camera of a level drone (``view`` without tilt)."""
    from drone_lab.sar.camera import SAR_CAMERA, quat_xyzw_to_matrix

    c, s = math.cos(view.yaw), math.sin(view.yaw)
    r_world_body = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_world_cam = r_world_body @ quat_xyzw_to_matrix(SAR_CAMERA.mount_rot_ros)
    cam = np.array([view.x, view.y, view.z]) + r_world_body @ np.array(SAR_CAMERA.mount_pos)
    p = r_world_cam.T @ (np.asarray(point, dtype=float) - cam)
    k = np.array(SAR_CAMERA.intrinsic_matrix())
    uvw = k @ p
    return uvw[0] / uvw[2], uvw[1] / uvw[2]


def test_sample_views_heights_tilts_and_bounds():
    from drone_lab.sar.capture import footprint_radius, sample_views

    site, scenario = _site_and_scenario()
    views = sample_views(scenario, site, 400, np.random.default_rng(0))
    assert len(views) == 400
    assert views == sample_views(scenario, site, 400, np.random.default_rng(0))  # deterministic
    kinds = {v.kind for v in views}
    assert kinds == {"object", "site"}
    trees = [o for o in scenario.objects if not o.labeled]
    x0, x1, y0, y1 = site.bounds
    for v in views:
        assert v.agl >= 5.0 - 1e-6 and abs(v.z - v.agl - site.height(v.x, v.y)) < 1e-6
        assert math.degrees(math.hypot(v.roll, v.pitch)) <= 15.0 + 1e-6
        if any(math.hypot(v.x - t.center[0], v.y - t.center[1]) < t.radius + 1.0 for t in trees):
            continue  # may have been raised above the crown
        assert v.agl <= 12.0 + 1e-6
        inset = 0.75 * footprint_radius(v.agl) - 1e-6
        assert x0 + inset <= v.x <= x1 - inset and y0 + inset <= v.y <= y1 - inset
    aimed = sum(v.kind == "object" for v in views) / len(views)
    assert 0.55 < aimed < 0.75


def test_aimed_views_put_the_target_in_the_image():
    from drone_lab.sar.camera import SAR_CAMERA
    from drone_lab.sar.capture import sample_views

    site, scenario = _site_and_scenario()
    by_name = {o.name: o for o in scenario.objects}
    views = sample_views(scenario, site, 300, np.random.default_rng(1), max_tilt_deg=0.0)
    aimed = [v for v in views if v.kind == "object"]
    w, h = SAR_CAMERA.width, SAR_CAMERA.height

    def near_image(u: float, v: float) -> bool:  # within 10 % past the edges (plus the object's own height)
        return -0.12 * w <= u <= 1.12 * w and -0.12 * h <= v <= 1.12 * h

    hits = [near_image(*_pixel_of(v, by_name[v.target].center)) for v in aimed]
    # the rest were pulled back from the edge of the site, or their target stands on a block well above the ground
    # under the drone, so it comes out farther from the image centre than aimed
    assert sum(hits) / len(aimed) > 0.85


def test_mission_views_follow_the_survey_path():
    from drone_lab.sar.capture import mission_views

    site, scenario = _site_and_scenario(0)
    views = mission_views(scenario, site, 2.5, np.random.default_rng(0))
    xy = np.array([[v.x, v.y] for v in views])
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert len(views) > 150 and np.all(steps <= 2.5 + 1e-6) and np.median(steps) > 2.45
    assert all(abs(v.z - scenario.flight_z) < 1e-9 and v.kind == "mission" for v in views)
    assert max(abs(math.degrees(v.yaw)) for v in views) < 10.0
    assert max(math.degrees(math.hypot(v.roll, v.pitch)) for v in views) <= 4.0 + 1e-6
    x0, x1, y0, y1 = scenario.area
    assert xy[:, 1].min() > y0 and xy[:, 1].max() < y1  # the lines run inside the area, the turns just outside it


def test_lighting_and_ground_looks():
    from drone_lab.sar.capture import SUN_AZIMUTH_DEG, TERRAIN_LOOKS, Lighting, sample_ground_look, sample_lighting

    rng = np.random.default_rng(0)
    lights = [sample_lighting(rng) for _ in range(400)]
    names = [light.name for light in lights]
    assert 0.2 < names.count("default") / 400 < 0.4 and 0.08 < names.count("overcast") / 400 < 0.22
    assert all(light == Lighting() for light in lights if light.name == "default")
    for light in lights:
        assert math.isclose(light.sky_rotation_deg, light.sun_azimuth_deg - SUN_AZIMUTH_DEG)
        if light.name == "overcast":
            assert light.sun_intensity < Lighting().sun_intensity < light.sky_intensity * 2.5
    for _ in range(20):
        look = sample_ground_look(rng)
        assert sorted(look.materials) == sorted(TERRAIN_LOOKS) == sorted(look.materials.values())
        if look.name != "swapped":
            assert all(k == v for k, v in look.materials.items())
        assert all(0.8 <= look.brightness[k] <= 1.2 and 0.8 <= look.texture_scale[k] <= 1.25 for k in TERRAIN_LOOKS)


def test_padded_camera_crop_and_unclipped_boxes():
    from drone_lab.sar.camera import SAR_CAMERA
    from drone_lab.sar.capture import crop, padded_spec, unclipped_boxes

    spec = padded_spec(SAR_CAMERA, 160, 120)
    assert (spec.width, spec.height, spec.focal_px) == (960, 720, SAR_CAMERA.focal_px)
    assert (spec.cx - 160, spec.cy - 120) == (SAR_CAMERA.cx, SAR_CAMERA.cy)  # the crop is the mission camera
    seg = np.zeros((720, 960), dtype=np.int32)
    seg[300:340, 100:200] = 5  # cut by the crop's left edge (x = 160): 60 of 100 px outside
    seg[400:420, 400:450] = 6  # inside the crop
    seg[10:40, 10:60] = 7  # outside the crop
    seg[580:620, 500:540] = 8  # cut by the bottom edge (y = 600)
    assert crop(seg, 160, 120).shape == (480, 640)
    boxes = unclipped_boxes(seg, 160, 120)
    assert boxes[5] == (100 - 160, 300 - 120, 199 - 160, 339 - 120)
    assert boxes[8] == (500 - 160, 580 - 120, 539 - 160, 619 - 120)
    assert 6 not in boxes and 7 not in boxes


if __name__ == "__main__":
    main_for(globals(), "test_sar_capture")
