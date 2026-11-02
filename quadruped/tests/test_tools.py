# Copyright (c) 2026 Hector Lugo
# SPDX-License-Identifier: MIT

"""The project's tools in ``scripts/``: joint-order check, depth-camera study, evaluation summary, Newton asset and
the 2.x parity check.

The log parser, the camera geometry and the evaluation summary need only numpy; the rest skips without ``pxr``,
Newton, Isaac Lab or the original 2.x sources.
"""

import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

DATA = Path(__file__).resolve().parent / "data"
LEGS = ("FL", "FR", "RL", "RR")
# Isaac Sim PhysX order (breadth-first: hips, thighs, calves), the one the 2.x checkpoint was trained with
BREADTH_FIRST = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf") for leg in LEGS]
# Newton's native order (per leg)
PER_LEG = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]


def test_check_joint_order_reads_isaaclab_logs(load_script: Callable[[str], ModuleType], tmp_path: Path) -> None:
    """Both joint-order records of a real Isaac Sim PhysX run log are found and checked correctly."""
    check = load_script("check_joint_order")
    # the action-term line and the articulation table of a `zero_agent --info` run on the RTX 5080
    text = (DATA / "zero_agent_rough_excerpt.log").read_text()
    assert check.parse_log(text) == {
        "action term JointPositionAction": BREADTH_FIRST,
        "articulation /World/envs/env_[^/]+/Robot": BREADTH_FIRST,
    }
    log = tmp_path / "run.log"
    log.write_text(text)
    assert check.main([str(log), "--expected", ",".join(BREADTH_FIRST)]) == 0
    assert check.main([str(log), "--expected", ",".join(PER_LEG)]) == 1  # e.g. Newton's order without the remap
    log.write_text("[INFO]: no joint records\n")
    assert check.main([str(log), "--expected", ",".join(BREADTH_FIRST)]) == 2


def test_check_joint_order_derives_order_from_usd(
    load_script: Callable[[str], ModuleType], require_file: Callable[[Path], Path]
) -> None:
    """The documented PhysX rule (breadth-first over links) applied to robot.usd gives the trained order."""
    pytest.importorskip("pxr", reason="needs USD (pxr)")
    check = load_script("check_joint_order")
    assert check.expected_joint_order(require_file(check.DEFAULT_USD)) == BREADTH_FIRST


def test_depth_camera_geometry(load_script: Callable[[str], ModuleType]) -> None:
    """Camera orientation helper, pinhole rays and the triangle ray cast behind the self-occlusion study."""
    study = load_script("depth_camera_study")
    for pitch in (0.0, 30.0, 50.0, 60.0):
        rot = study.quat_to_matrix(study.pitch_down_quat(pitch))
        # OpenGL camera: looks along -z, image right along +x; mounted forward-looking on the trunk
        forward = rot @ np.array([0.0, 0.0, -1.0])
        assert forward == pytest.approx([math.cos(math.radians(pitch)), 0.0, -math.sin(math.radians(pitch))])
        assert rot @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, -1.0, 0.0])
    # the 2.x quaternion is the same 50 deg pitch
    assert study.pitch_down_quat(50.0) == pytest.approx(study.MOUNT_2X[1], abs=2e-4)

    # 64x48 pixels, f = 18.14756 mm, aperture 20.955 mm: 60 deg horizontal field of view between the image edges
    width, height, focal, aperture = 64, 48, 18.14756, 20.955
    _, dirs = study.camera_rays((0.0, 0.0, 0.0), study.pitch_down_quat(0.0), width, height, focal, aperture)
    assert dirs.shape == (width * height, 3) and np.linalg.norm(dirs, axis=1) == pytest.approx(1.0)
    row = dirs[(height // 2) * width : (height // 2 + 1) * width]
    edge_angle = math.degrees(math.acos(float(row[0] @ row[-1])))
    half_fov = math.atan(aperture / 2 / focal)
    expected = 2 * math.degrees(math.atan((width / 2 - 0.5) / (width / 2) * math.tan(half_fov)))  # pixel centres
    assert math.degrees(2 * half_fov) == pytest.approx(60.0, abs=0.01)
    assert edge_angle == pytest.approx(expected, abs=0.05)

    # Moller-Trumbore: a ray through the triangle hits it at its distance, a ray beside it misses
    triangle = np.array([[[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]]], dtype=np.float32)
    rays = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert study._ray_triangles(np.zeros(3), rays, triangle).tolist() == [pytest.approx(1.0), math.inf]


def test_depth_camera_mount_sees_no_robot(
    load_script: Callable[[str], ModuleType], require_file: Callable[[Path], Path]
) -> None:
    """Ray-cast the configured 64x48 camera against the robot's own meshes at the default pose (no Kit)."""
    pytest.importorskip("pxr", reason="needs USD (pxr)")
    pytest.importorskip("isaaclab_tasks", reason="needs Isaac Lab 3.0 (the depth task cfg)")
    study = load_script("depth_camera_study")
    camera, usd_path, joint_pos = study.configured_setup()
    triangles, standing_height = study.posed_robot_triangles(str(require_file(Path(usd_path))), joint_pos)
    assert standing_height == pytest.approx(0.452, abs=0.005)  # trunk origin above flat ground, default pose
    spawn = camera.spawn
    results = {}
    for name, (pos, rot) in {"configured": (camera.offset.pos, camera.offset.rot), "2.x": study.MOUNT_2X}.items():
        origin, dirs = study.camera_rays(
            pos,
            rot,
            camera.width,
            camera.height,
            spawn.focal_length,
            spawn.horizontal_aperture,
            spawn.vertical_aperture,
        )
        for kind in study.GEOMETRY_KINDS:
            results[name, kind] = study.study_mount(
                origin, dirs, triangles, kind, standing_height, spawn.clipping_range[1]
            )
    for kind in study.GEOMETRY_KINDS:
        configured = results["configured", kind]
        assert configured["robot_pct"] == 0.0, kind
        assert configured["nearest_ground_centre"] < 0.7 and configured["beyond_far_clip_pct"] == 0.0
        assert results["2.x", kind]["robot_pct"] > 25.0  # the old mast mount sees its own trunk and legs


def _episode(**overrides: object) -> dict:
    """One per-episode record as ``scripts/eval_policy.py`` writes it."""
    record = {
        "length_s": 20.0,
        "base_contact": False,
        "time_out": True,
        "tilt_gt_60deg": False,
        "lin_err": 0.2,
        "ang_err": 0.1,
        "cmd_speed": 0.8,
        "speed": 0.7,
        "base_height": 0.65,
        "foot_contact_frac": 0.0,
        "calf_contact_frac": 0.5,
        "alive_steps": 1000,
        "reward_sum": {"track_lin_vel_xy_exp": 30.0, "lin_vel_z_l2": -1.0},
        "joint_pos_mean_deg": {"hip": 2.0, "thigh": -15.0, "calf": 42.0},
        "action_abs_mean": 2.5,
        "bc_trunk": False,
        "bc_thigh": False,
        "min_base_height": 0.5,
    }
    return record | overrides


def test_eval_summary(load_script: Callable[[str], ModuleType]) -> None:
    """Episode metrics: falls vs time-outs, step-weighted errors, rewards in the ``Episode_Reward`` convention."""
    evaluation = load_script("eval_policy")
    records = [
        _episode(),
        _episode(),
        _episode(),
        _episode(
            length_s=5.0,
            base_contact=True,
            time_out=False,
            tilt_gt_60deg=True,
            lin_err=0.6,
            alive_steps=250,
            reward_sum={"track_lin_vel_xy_exp": 5.0, "lin_vel_z_l2": -2.0},
            bc_thigh=True,
            min_base_height=0.1,
        ),
    ]
    terms = ["track_lin_vel_xy_exp", "lin_vel_z_l2"]
    summary = evaluation.summarize(records, terms, max_episode_length_s=20.0, dt=0.02)
    assert summary["episodes"] == 4
    assert summary["frac_base_contact"] == summary["frac_tilt_gt_60deg"] == summary["frac_bc_thigh"] == 0.25
    assert summary["frac_time_out"] == 0.75 and summary["frac_short_lt_5s"] == 0.0
    assert summary["mean_episode_length_s"] == pytest.approx(16.25)
    # tracking errors are means over alive steps, so the short episode weighs a quarter of a full one
    assert summary["lin_vel_err_mps"] == pytest.approx((3 * 1000 * 0.2 + 250 * 0.6) / 3250)
    # episode sum / 20 s, averaged over episodes
    rewards = summary["episode_reward_per_term"]
    assert rewards == pytest.approx({"track_lin_vel_xy_exp": 1.1875, "lin_vel_z_l2": -0.0625})
    assert summary["episode_reward_total"] == pytest.approx(1.125)
    assert summary["median_min_base_height_m"] == 0.5
    assert evaluation.summarize([], terms, 20.0, 0.02) == {"episodes": 0}


def test_newton_asset_matches_canonical_asset(
    load_script: Callable[[str], ModuleType], project_dir: Path, require_file: Callable[[Path], Path]
) -> None:
    """robot_newton.usd only moves the collision APIs onto the meshes; Newton imports one collider per body (CPU)."""
    pytest.importorskip("pxr", reason="needs USD (pxr)")
    pytest.importorskip("newton", reason="needs Newton")
    asset = load_script("make_newton_asset")
    for path in (asset.ROBOT_USD, asset.NEWTON_USD, asset.PHYSICS_LAYER, asset.NEWTON_PHYSICS_LAYER):
        require_file(path)
    assert asset.check() == 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # Newton's importer runs on the CPU; keep the GPU out of it
    out = subprocess.run(
        [sys.executable, str(project_dir / "scripts" / "make_newton_asset.py"), "--check", "--newton"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "robot.usd: 0 collision shapes" in out.stdout
    assert "robot_newton.usd: 17 collision shapes on 17 bodies" in out.stdout


def test_parity_with_2x_sources(load_script: Callable[[str], ModuleType], project_dir: Path) -> None:
    """Static 2.x config (2.3.2 base + the 2.x task overrides) vs the instantiated 3.0 cfg; needs the 2.x folders."""
    pytest.importorskip("isaaclab_tasks", reason="needs Isaac Lab 3.0")
    pytest.importorskip("torch")
    parity = load_script("check_obs_parity")
    if not (parity.DEFAULT_OLD_LAB / ".git").exists() or not (parity.DEFAULT_OLD_TASK / "rough_env_cfg.py").is_file():
        pytest.skip("the 2.x sources (~/IsaacLab, ~/quadruped_rough_terrain) are not on this machine")
    checkpoint = project_dir / "checkpoints_2x" / "height_scanner_model.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
        pytest.skip(f"{checkpoint.name} is missing or an un-smudged Git LFS pointer (run `git lfs pull`)")
    assert parity.main([]) == 0

    from quadruped_lab.tasks.locomotion.config.custom_quad.rough_env_cfg import CustomQuadRoughEnvCfg

    old = parity.load_old_sources(parity.DEFAULT_OLD_LAB, parity.DEFAULT_OLD_TASK)
    assert not old["skipped"]
    comparer = parity.compare_with_2x(old["state"], CustomQuadRoughEnvCfg(), old["defaults"])
    assert not comparer.problems and comparer.checked > 200
    checks = parity.execution_checks(old, CustomQuadRoughEnvCfg())
    assert all(expected == actual for expected, actual in checks.values()), checks
    # negative control: changed 3.0 values are reported, including a parameter the 2.x config leaves at its default
    cfg = CustomQuadRoughEnvCfg()
    cfg.rewards.dof_acc_l2.weight = -1.0e-6
    cfg.events.add_base_mass.params["recompute_inertia"] = False
    cfg.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.2)
    problems = "\n".join(parity.compare_with_2x(old["state"], cfg, old["defaults"]).problems)
    for path in (
        "rewards.dof_acc_l2.weight",
        "add_base_mass.params['recompute_inertia']",
        "['boxes'].grid_height_range",
    ):
        assert path in problems, path
    # ... and so are the Isaac Lab 3.0 execution defaults, which the 2.x policy was not trained with
    cfg.sim.physics.default.enable_external_forces_every_iteration = True
    cfg.sim.use_newton_actuators = True
    checks = parity.execution_checks(old, cfg)
    assert [name for name, (expected, actual) in checks.items() if expected != actual] == list(checks)
