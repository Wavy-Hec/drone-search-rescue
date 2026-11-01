"""SAR task config: registered, policy observations identical to the survey task, camera / site / lights in the
scene, assets, camera math and the asset cache (CPU only; env_isaaclab, no Kit).

Run: ``CUDA_VISIBLE_DEVICES= python tests/test_sar_env_cfg.py`` (or with pytest).
"""

from __future__ import annotations

import math

import numpy as np
from _runner import main_for, require

SAR = "DroneLab-SAR-ARL-Robot-1-v0"
SURVEY = "DroneLab-Survey-ARL-Robot-1-v0"
_GROUP_SETTINGS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
    "history_order",
}


def _parse(task, overrides=()):
    require("isaaclab_tasks")
    from isaaclab_tasks.utils import parse_env_cfg

    import drone_lab.tasks  # noqa: F401

    return parse_env_cfg(task, device="cpu", overrides=list(overrides))


def _terms(group):
    return [k for k, v in group.__dict__.items() if k not in _GROUP_SETTINGS and v is not None]


def test_sar_task_registered():
    require("isaaclab_tasks")
    import gymnasium as gym

    import drone_lab.tasks  # noqa: F401

    spec = gym.spec(SAR)
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "drone_lab.tasks.sar.config.arl_robot_1.sar_env_cfg:SarEnvCfg"
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == gym.spec(SURVEY).kwargs["rsl_rl_cfg_entry_point"]


def test_policy_observations_identical_to_survey():
    cfg, ref = _parse(SAR), _parse(SURVEY)
    pol, rpol = cfg.observations.policy, ref.observations.policy
    assert _terms(pol) == _terms(rpol)
    for name in _terms(pol):
        a, b = getattr(pol, name), getattr(rpol, name)
        assert a.func is b.func and a.params == b.params and a.scale == b.scale and a.clip == b.clip, name
        assert a.noise is None and b.noise is None, name
    for key in _GROUP_SETTINGS:
        assert getattr(pol, key, None) == getattr(rpol, key, None), key
    assert [k for k, v in cfg.observations.__dict__.items() if v is not None] == ["policy"]
    # actions, timing, commands, events, terminations unchanged as well
    a, b = cfg.actions.thrust_command, ref.actions.thrust_command
    assert (a.scale, a.offset, a.clip) == (b.scale, b.offset, b.clip)
    timing, ref_timing = (
        (cfg.sim.dt, cfg.decimation, cfg.sim.render_interval),
        (ref.sim.dt, ref.decimation, ref.sim.render_interval),
    )
    assert timing == ref_timing
    assert cfg.commands.survey_target.to_dict() == ref.commands.survey_target.to_dict()
    assert cfg.terminations.to_dict() == ref.terminations.to_dict()
    assert cfg.events.to_dict() == ref.events.to_dict()


def test_scene_has_site_camera_and_lights():
    from drone_lab.sar.assets import SEMANTIC_TYPE
    from drone_lab.sar.camera import SAR_CAMERA

    cfg = _parse(SAR, ["physics=isaacsim_physx"])
    scene = cfg.scene
    site = scene.sar_site
    assert site.prim_path == "/World/sar" and type(site.spawn).__name__ == "SarSiteCfg"
    assert site.spawn.terrain_generator is scene.terrain.terrain_generator or (
        site.spawn.terrain_generator.to_dict() == scene.terrain.terrain_generator.to_dict()
    )
    assert site.spawn.survey_clearance == cfg.commands.survey_target.altitude
    cam = scene.sar_camera
    assert cam.prim_path.endswith("/Robot/base_link/sar_camera")
    assert (cam.width, cam.height) == (SAR_CAMERA.width, SAR_CAMERA.height) == (640, 480)
    assert cam.data_types == ["rgb", "distance_to_image_plane", "instance_segmentation"]
    assert cam.renderer_cfg.semantic_filter == f"{SEMANTIC_TYPE}:*"
    assert cam.renderer_cfg.colorize_instance_segmentation is False
    assert cam.renderer_cfg.global_settings.antialiasing_mode == "FXAA"
    assert cam.renderer_cfg.enable_scene_partitioning is False
    assert cam.offset.convention == "ros" and np.allclose(cam.offset.rot, SAR_CAMERA.mount_rot_ros)
    f_px = cam.spawn.focal_length / cam.spawn.horizontal_aperture * cam.width
    assert abs(f_px - SAR_CAMERA.focal_px) < 1e-6
    assert type(scene.sun_light.spawn).__name__ == "DistantLightCfg"
    assert scene.sky_light.spawn.texture_file.endswith(".hdr")
    assert scene.overview_camera is None and scene.topdown_camera is None
    # another seed through Hydra
    cfg3 = _parse(SAR, ["env.scene.sar_site.spawn.seed=3"])
    assert cfg3.scene.sar_site.spawn.seed == 3


def test_camera_mount_looks_down_with_nose_up_in_the_image():
    from drone_lab.sar.camera import SAR_CAMERA

    x, y, z, w = SAR_CAMERA.mount_rot_ros
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-6)  # optical axis (ROS +z) points down
    assert np.allclose(R @ [0, -1, 0], [1, 0, 0], atol=1e-6)  # image up (ROS -y) is the drone's nose (+x)
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-6)  # image right is the drone's right (-y)
    # Isaac Lab's own conversion agrees: ROS -> OpenGL camera frame
    require("isaaclab")
    import torch

    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    q = torch.tensor([SAR_CAMERA.mount_rot_ros], dtype=torch.float32)
    q_world = convert_camera_frame_orientation_convention(q, origin="ros", target="world")
    xw, yw, zw, ww = q_world[0].tolist()
    fwd = np.array([1 - 2 * (yw * yw + zw * zw), 2 * (xw * yw + zw * ww), 2 * (xw * zw - yw * ww)])  # R @ e_x
    assert np.allclose(fwd, [0, 0, -1], atol=1e-5)  # "world" convention looks along +x


def test_pixel_budget_lying_person_at_survey_altitude():
    """The camera math of drone_lab.sar.camera, checked on the real default site."""
    require("isaaclab")
    from drone_lab.sar.assets import assets_of
    from drone_lab.sar.camera import SAR_CAMERA
    from drone_lab.sar.scenario import ScenarioCfg, TerrainSite, generate_scenario

    cam = SAR_CAMERA
    assert abs(cam.hfov_deg - 67.38) < 0.05 and abs(cam.vfov_deg - 53.13) < 0.05
    site = TerrainSite()
    zmin, _ = site.terrain_z_range()
    flight_z = site.flight_z(clearance=4.0, spacing=4.0)
    assert abs(flight_z - 7.515) < 0.01
    cam_z = flight_z + cam.mount_pos[2]
    # over flat ground (z = 0) and over the lowest ground of the area (pit floor)
    shoulders, thinnest = 0.40, min(a.size[1] for a in assets_of("person"))
    assert cam.pixels(shoulders, cam_z - 0.0) >= 25 and cam.pixels(thinnest, cam_z - 0.0) >= 20
    assert cam.pixels(shoulders, cam_z - zmin) >= 20
    shortest = min(a.size[2] for a in assets_of("person"))
    assert cam.pixels(shortest, cam_z - zmin) >= 80
    # swath wider than the lawnmower spacing even over the highest point (4 m clearance)
    assert cam.footprint(4.0 + cam.mount_pos[2])[0] > 4.0
    # every lying person of the default scenes is at least 20 px across its shoulders
    for seed in (0, 1, 2):
        sc = generate_scenario(ScenarioCfg(seed=seed), site=site)
        for p in sc.by_category("person"):
            assert cam.pixels(shoulders, cam_z - p.ground_z) >= 20, (seed, p.name, p.ground_z)


def test_assets_units_and_urls():
    from drone_lab.sar.assets import ASSET_ROOT_6_2, ASSETS, GROUND_MATERIALS, UNLABELED, asset_root, assets_of

    assert asset_root().startswith("http") or asset_root() == ASSET_ROOT_6_2
    for a in ASSETS.values():
        assert a.url.startswith(asset_root() + "/") and a.url.endswith(".usd")
        assert all(lo < hi for lo, hi in zip(a.bbox_min, a.bbox_max, strict=False)), a.key
        cm = a.path.startswith("NVIDIA/Assets/")  # vegetation and DigitalTwin assets are in centimetres
        assert a.meters_per_unit == (0.01 if cm else 1.0), a.key
    for p in assets_of("person"):
        assert 1.55 < p.size[2] < 1.85 and p.bbox_min[2] == 0.0 and p.poses == ("standing", "lying")
    assert len(assets_of("person")) >= 8 and len(assets_of(UNLABELED)) >= 3
    assert all(m.url.endswith(".mdl") for m in GROUND_MATERIALS.values())


def test_projection_round_trip_and_image_orientation():
    """world_to_pixel / pixel_to_world with the real mount: nose up in the image, right side to the right."""
    require("isaaclab")
    from drone_lab.sar.camera import SAR_CAMERA, pixel_to_world, quat_xyzw_to_matrix, world_to_pixel
    from drone_lab.sar.scene import _matrix_to_quat

    cam = SAR_CAMERA
    K = np.array(cam.intrinsic_matrix())
    drone = np.array([3.0, -4.0, 7.515])
    for yaw in (0.0, 0.7, -2.0):
        R_body = quat_xyzw_to_matrix((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
        R_cam = R_body @ quat_xyzw_to_matrix(cam.mount_rot_ros)
        quat = _matrix_to_quat(R_cam)  # robust (Shepperd); these are 180 deg rotations, w = 0
        assert np.allclose(quat_xyzw_to_matrix(quat), R_cam, atol=1e-6)
        pos = drone + R_body @ np.array(cam.mount_pos)
        h = pos[2]  # ground at z = 0
        uv, d = world_to_pixel([[pos[0], pos[1], 0.0]], pos, quat, K)
        assert np.allclose(uv[0], [cam.cx, cam.cy], atol=1e-6) and abs(d[0] - h) < 1e-9
        nose = pos + R_body @ np.array([1.0, 0.0, 0.0])
        right = pos + R_body @ np.array([0.0, -1.0, 0.0])
        uv_n, _ = world_to_pixel([[nose[0], nose[1], 0.0]], pos, quat, K)
        uv_r, _ = world_to_pixel([[right[0], right[1], 0.0]], pos, quat, K)
        assert np.allclose(uv_n[0], [cam.cx, cam.cy - cam.focal_px / h], atol=1e-6)  # 1 m ahead -> above the centre
        assert np.allclose(uv_r[0], [cam.cx + cam.focal_px / h, cam.cy], atol=1e-6)  # 1 m right -> right
        pts = np.array([[1.0, 2.0, 0.3], [-2.0, -6.0, 1.7], [4.0, -1.0, -1.2]])
        uv, d = world_to_pixel(pts, pos, quat, K)
        assert np.allclose(pixel_to_world(uv, d, pos, quat, K), pts, atol=1e-9)


def test_look_at_quaternion():
    require("isaaclab")
    from drone_lab.sar.scene import look_at_quat_ros

    def rot(q):
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    R = rot(look_at_quat_ros((0, 0, 75), (0, 0, 0)))  # nadir photo camera: north (+y) up in the image
    assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-9) and np.allclose(R @ [0, -1, 0], [0, 1, 0], atol=1e-9)
    eye, tgt = np.array([-28.0, -38.0, 26.0]), np.array([1.0, 2.0, 0.0])
    R = rot(look_at_quat_ros(eye, tgt))
    f = (tgt - eye) / np.linalg.norm(tgt - eye)
    assert np.allclose(R[:, 2], f, atol=1e-9) and R[2, 1] < 0  # image down points down in the world
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-9)


def test_cache_pruning_keeps_layer_copies_that_a_live_process_uses():
    """Old per-run layer copies go, unless a live process (this one or another) still holds them; fresh copies and
    raw downloads always stay. A holder that exits releases its copies."""
    require("isaaclab")
    import os
    import subprocess
    import sys
    import tempfile
    import time

    from drone_lab.sar import scene

    def make_copy(cache: str, name: str) -> tuple[str, str]:
        folder = os.path.join(cache, f"{scene.CACHE_COPY_PREFIX}{name}")
        os.makedirs(os.path.join(folder, "1"))
        layer = os.path.join(folder, "1", "asset.usd")
        open(layer, "w").close()
        return folder, layer

    with tempfile.TemporaryDirectory() as cache:
        old_unused, _ = make_copy(cache, "old_unused")
        mine, mine_layer = make_copy(cache, "mine")
        theirs, _ = make_copy(cache, "theirs")
        fresh, _ = make_copy(cache, "fresh")
        raw = os.path.join(cache, "https", "example.org", "asset.usd")
        os.makedirs(os.path.dirname(raw))
        open(raw, "w").close()
        assert scene.hold_cache_copy(cache, mine_layer) and not scene.hold_cache_copy(cache, raw)
        holder = (
            "import fcntl, os, sys\n"
            "fd = os.open(sys.argv[1], os.O_RDONLY | os.O_CREAT)\n"
            "fcntl.flock(fd, fcntl.LOCK_SH)\n"
            "print('held', flush=True)\n"
            "sys.stdin.read()\n"
        )
        lock = os.path.join(theirs, scene.IN_USE_LOCK)
        other = subprocess.Popen(
            [sys.executable, "-c", holder, lock], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        try:
            assert other.stdout.readline().strip() == "held"
            old = time.time() - 7 * 3600.0
            for folder in (old_unused, mine, theirs):
                os.utime(folder, (old, old))
            assert scene.prune_cache_copies(cache, max_age_hours=6.0) == 1
            assert not os.path.exists(old_unused) and os.path.isfile(raw)
            assert all(os.path.isdir(f) for f in (mine, theirs, fresh))
        finally:
            other.stdin.close()
            other.wait(timeout=30)
        assert scene.prune_cache_copies(cache, max_age_hours=6.0) == 1  # the other process has exited
        assert not os.path.exists(theirs) and os.path.isdir(mine) and os.path.isdir(fresh)
        os.close(scene._HELD_COPIES.pop(os.path.abspath(mine)))  # this process lets go of its copy as well
        assert scene.prune_cache_copies(cache, max_age_hours=6.0) == 1
        assert not os.path.exists(mine) and os.path.isdir(fresh) and os.path.isfile(raw)


if __name__ == "__main__":
    main_for(globals(), "test_sar_env_cfg")
