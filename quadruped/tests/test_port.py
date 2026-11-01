# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less checks of the 2.x -> 3.0 port: configs, observation interface and checkpoint conversion.

Everything here runs on the CPU without launching Isaac Sim; the simulation-level checks are listed in PORT_NOTES.md.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

import quadruped_lab.tasks  # noqa: F401

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

PROJECT = Path(__file__).resolve().parents[1]
ROUGH = "Isaac-Velocity-Rough-Custom-Quad-v0"
FLAT = "Isaac-Velocity-Flat-Custom-Quad-v0"
DEPTH = "Isaac-Velocity-Rough-Depth-Custom-Quad-v0"
CHECKPOINT_2X = PROJECT / "checkpoints_2x" / "height_scanner_model.pt"
CHECKPOINT_RSLRL5 = PROJECT / "checkpoints_2x" / "height_scanner_model_rslrl5.pt"
# order and sizes of the policy observation the 2.x height-scan checkpoint was trained on (235 = 48 + 187)
POLICY_LAYOUT_2X = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
    ("height_scan", 187),
]


def _resolve(task_id, overrides=(), play_mode=False):
    from isaaclab_tasks.utils.hydra import resolve_task_config

    return resolve_task_config(task_id, "rsl_rl_cfg_entry_point", play_mode=play_mode, overrides=list(overrides))


def _script(name: str):
    """Import ``scripts/<name>.py`` as a module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses in the scripts resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def _converter():
    return _script("convert_rslrl_checkpoint")


def _is_lfs_pointer(path: Path) -> bool:
    return path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs")


@pytest.mark.parametrize("task_id", [FLAT, ROUGH, DEPTH])
def test_physics_defaults_to_isaacsim_physx(task_id):
    """The velocity base defaults to Newton in 3.0; the port defaults to Isaac Sim PhysX (and aliases `physx`)."""
    for overrides in ([], ["physics=isaacsim_physx"], ["physics=physx"]):
        env_cfg, _ = _resolve(task_id, overrides)
        assert type(env_cfg.sim.physics).__name__ == "PhysxCfg", overrides
    env_cfg, _ = _resolve(task_id, ["physics=newton_mjwarp"])
    assert type(env_cfg.sim.physics).__name__ == "NewtonCfg"


@pytest.mark.parametrize("task_id", [FLAT, ROUGH, DEPTH])
def test_play_mode_matches_2x_play_classes(task_id):
    env_cfg, _ = _resolve(task_id, ["physics=isaacsim_physx"], play_mode=True)
    assert env_cfg.scene.num_envs <= 50
    assert env_cfg.events.push_robot is None and env_cfg.events.base_external_force_torque is None
    for group in env_cfg.observations.__dataclass_fields__:
        assert getattr(env_cfg.observations, group).enable_corruption is False
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is not None:
        assert (generator.num_rows, generator.num_cols, generator.curriculum) == (5, 5, False)
        assert env_cfg.scene.terrain.max_init_terrain_level is None


def test_rough_policy_layout_matches_checkpoint():
    from pxr import Usd, UsdPhysics

    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors.ray_caster.patterns import grid_pattern

    env_cfg, agent_cfg = _resolve(ROUGH)
    group = env_cfg.observations.policy
    names = [n for n in group.__dataclass_fields__ if isinstance(getattr(group, n), ObservationTermCfg)]
    assert names == [name for name, _ in POLICY_LAYOUT_2X]
    assert group.concatenate_terms and group.enable_corruption
    assert group.height_scan.clip == (-1.0, 1.0)
    assert (group.joint_vel.noise.n_min, group.joint_vel.noise.n_max) == (-1.5, 1.5)
    assert all(getattr(group, n).scale is None for n in names)
    # sensor- and robot-dependent sizes
    rays = grid_pattern(env_cfg.scene.height_scanner.pattern_cfg, "cpu")[0].shape[0]
    stage = Usd.Stage.Open(env_cfg.scene.robot.spawn.usd_path)
    dofs = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert rays == 187 and len(dofs) == 12
    assert sum(size for _, size in POLICY_LAYOUT_2X) == 235
    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert env_cfg.actions.joint_pos.scale == 0.25
    assert (env_cfg.sim.dt, env_cfg.decimation) == (0.005, 4)


def test_events_and_actuator_keep_2x_semantics():
    env_cfg, _ = _resolve(ROUGH)
    mass = env_cfg.events.add_base_mass.params
    assert (mass["mass_distribution_params"], mass["operation"], mass["distribution"]) == (
        (-1.0, 3.0),
        "add",
        "uniform",
    )
    motor = env_cfg.scene.robot.actuators["base_legs"]
    assert (motor.actuator_effort_limit, motor.joint_effort_limit, motor.saturation_effort) == (45.0, 45.0, 45.0)
    assert (motor.actuator_velocity_limit, motor.stiffness, motor.damping) == (21.0, 60.0, 1.5)
    assert Path(env_cfg.scene.robot.spawn.usd_path).is_file()
    assert env_cfg.scene.robot.joint_ordering is None


@pytest.mark.parametrize("task_id", [FLAT, ROUGH, DEPTH])
def test_robot_preset_per_backend(task_id):
    """PhysX keeps robot.usd with native ordering; Newton gets robot_newton.usd in the PhysX joint/body order."""
    from quadruped_lab.assets.custom_quad import (
        CUSTOM_QUAD_CFG,
        CUSTOM_QUAD_NEWTON_USD_PATH,
        CUSTOM_QUAD_PHYSX_BODY_ORDER,
        CUSTOM_QUAD_PHYSX_JOINT_ORDER,
        CUSTOM_QUAD_USD_PATH,
    )

    for overrides in ([], ["physics=isaacsim_physx"], ["physics=physx"]):
        robot = _resolve(task_id, overrides)[0].scene.robot
        assert robot.spawn.usd_path == str(CUSTOM_QUAD_USD_PATH), overrides
        assert robot.joint_ordering is None and robot.body_ordering is None, overrides
    for overrides in (["physics=newton_mjwarp"], ["physics=newton_kamino"]):
        robot = _resolve(task_id, overrides)[0].scene.robot
        assert robot.spawn.usd_path == str(CUSTOM_QUAD_NEWTON_USD_PATH), overrides
        assert robot.joint_ordering == list(CUSTOM_QUAD_PHYSX_JOINT_ORDER)
        assert robot.body_ordering == list(CUSTOM_QUAD_PHYSX_BODY_ORDER)
        assert robot.prim_path == "{ENV_REGEX_NS}/Robot"
        # only the asset's collision layout and the ordering differ from the PhysX robot
        assert robot.actuators == CUSTOM_QUAD_CFG.actuators and robot.init_state == CUSTOM_QUAD_CFG.init_state
    # an explicit usd_path override is applied after the preset and still wins
    feb23 = str(CUSTOM_QUAD_USD_PATH.with_name("robot_feb23.usd"))
    robot = _resolve(task_id, ["physics=newton_mjwarp", f"env.scene.robot.spawn.usd_path={feb23}"])[0].scene.robot
    assert robot.spawn.usd_path == feb23


def test_newton_asset_matches_canonical_asset():
    """robot_newton.usd only moves the collision APIs onto the meshes; Newton imports one collider per body (CPU)."""
    import os
    import subprocess

    script = PROJECT / "scripts" / "make_newton_asset.py"
    assert _script("make_newton_asset").check() == 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # Newton's importer runs on the CPU; keep the GPU out of it
    out = subprocess.run(
        [sys.executable, str(script), "--check", "--newton"], env=env, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "robot.usd: 0 collision shapes" in out.stdout
    assert "robot_newton.usd: 17 collision shapes on 17 bodies" in out.stdout


def test_depth_task_layout():
    from isaaclab.utils.math import quat_apply

    env_cfg, agent_cfg = _resolve(DEPTH, ["physics=isaacsim_physx"])
    camera = env_cfg.scene.depth_camera
    # 2.x orientation (w, x, y, z) -> 3.0 (x, y, z, w), mounted on the trunk's front face instead of the 2.x mast
    w, x, y, z = (0.66446, 0.24184, -0.24184, -0.66446)
    assert camera.offset.rot == (x, y, z, w) and camera.offset.convention == "opengl"
    assert camera.offset.pos == (0.47, 0.0, 0.08)
    forward = quat_apply(torch.tensor([camera.offset.rot]), torch.tensor([[0.0, 0.0, -1.0]]))[0]  # OpenGL looks -Z
    assert math.degrees(math.asin(-forward[2].item())) == pytest.approx(50.0, abs=0.1) and forward[0] > 0
    # lazy sensor updates render on read (once per policy step); a positive period only adds host syncs
    assert camera.update_period == 0.0 and env_cfg.scene.lazy_sensor_update
    assert type(camera.renderer_cfg).__name__ == "NewtonWarpRendererCfg"
    term = env_cfg.observations.depth.depth_image
    # image_depth plus a pose refresh from the camera's parent link before every read: on Isaac Sim PhysX + Newton
    # Warp the camera otherwise renders from its spawn pose for the whole run (GPU check 2026-11-01)
    from isaaclab_tasks.core.velocity import mdp

    assert term.func.__name__ == "body_mounted_image_depth" and issubclass(term.func, mdp.image_depth)
    assert term.params["asset_cfg"].name == "robot" and term.params["asset_cfg"].body_names == "trunk"
    assert camera.prim_path.split("/")[-2] == term.params["asset_cfg"].body_names and camera.update_latest_camera_pose
    assert term.params["channel_first"] and term.params["data_type"] == "distance_to_camera"
    assert term.params["max_depth"] == camera.spawn.clipping_range[1]
    assert env_cfg.observations.policy.height_scan is None
    assert agent_cfg.obs_groups == {"actor": ["policy", "depth"], "critic": ["policy", "privileged"]}
    assert type(agent_cfg.actor).__name__ == "RslRlCNNModelCfg"
    rtx_cfg, _ = _resolve(DEPTH, ["physics=isaacsim_physx", "renderer=isaacsim_rtx"])
    assert type(rtx_cfg.scene.depth_camera.renderer_cfg).__name__ == "IsaacRtxRendererCfg"


def test_depth_cnn_model_builds():
    """The CNN actor builds from the agent cfg for a (N, 1, 48, 64) depth group plus 48 proprio values."""
    from rsl_rl.models import CNNModel
    from tensordict import TensorDict

    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

    env_cfg, agent_cfg = _resolve(DEPTH)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    cam = env_cfg.scene.depth_camera
    obs = TensorDict(
        {
            "policy": torch.zeros(2, 48),
            "depth": torch.zeros(2, 1, cam.height, cam.width),
            "privileged": torch.zeros(2, 187),
        },
        batch_size=[2],
    )
    actor_cfg = to_dict(agent_cfg.actor)
    actor_cfg.pop("class_name")
    actor = CNNModel(obs, agent_cfg.obs_groups, "actor", 12, **actor_cfg)
    assert actor(obs).shape == (2, 12)
    assert actor.cnn_latent_dim == 768


def test_convert_synthetic_checkpoint(tmp_path):
    """Round trip of a synthetic rsl-rl 3.x checkpoint, including an observation normalizer."""
    conv = _converter()
    torch.manual_seed(0)
    sd = {"std": torch.rand(12) + 0.1}
    for net, out in (("actor", 12), ("critic", 1)):
        dims = [235, 512, 256, 128, out]
        for i in range(4):
            sd[f"{net}.{2 * i}.weight"] = torch.randn(dims[i + 1], dims[i]) * 0.05
            sd[f"{net}.{2 * i}.bias"] = torch.randn(dims[i + 1]) * 0.05
        sd[f"{net}_obs_normalizer._mean"] = torch.randn(1, 235)
        sd[f"{net}_obs_normalizer._var"] = torch.rand(1, 235) + 0.5
        sd[f"{net}_obs_normalizer._std"] = sd[f"{net}_obs_normalizer._var"].sqrt()
        sd[f"{net}_obs_normalizer.count"] = torch.tensor(1000)
    new, info = conv.convert_checkpoint(
        {"model_state_dict": sd, "optimizer_state_dict": None, "iter": 7, "infos": None}
    )
    assert info["actor"]["obs_normalizer"] and info["num_actions"] == 12
    assert set(new["actor_state_dict"]) == {"distribution.std_param"} | {
        f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")
    } | {f"obs_normalizer.{b}" for b in ("_mean", "_var", "_std", "count")}

    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    obs = TensorDict({"policy": torch.randn(64, 235)}, batch_size=[64])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0}
    actor = MLPModel(obs, groups, "actor", 12, [512, 256, 128], "elu", obs_normalization=True, distribution_cfg=dist)
    critic = MLPModel(obs, groups, "critic", 1, [512, 256, 128], "elu", obs_normalization=True)
    actor.load_state_dict(new["actor_state_dict"], strict=True)
    critic.load_state_dict(new["critic_state_dict"], strict=True)
    actor.eval()
    critic.eval()
    with torch.inference_mode():
        assert torch.allclose(actor(obs), conv.old_forward(sd, "actor", obs["policy"]), atol=1e-6)
        assert torch.allclose(critic(obs), conv.old_forward(sd, "critic", obs["policy"]), atol=1e-6)


def test_converted_checkpoint_matches_original():
    """The committed rsl-rl 5.x checkpoint loads through the runner and reproduces the 2.x policy exactly."""
    for path in (CHECKPOINT_2X, CHECKPOINT_RSLRL5):
        if not path.is_file() or _is_lfs_pointer(path):
            pytest.skip(f"{path.name} is missing or an un-smudged Git LFS pointer (run `git lfs pull`)")
    report = _converter().verify(CHECKPOINT_2X, CHECKPOINT_RSLRL5, _converter().DEFAULT_AGENT_CFG, 1024, 0)
    assert report["obs_dim"] == 235 and report["num_actions"] == 12
    assert report["loaded_iteration"] == report["checkpoint_iteration"] == 13900
    assert max(v for k, v in report.items() if k.startswith("max_abs_diff")) == 0.0


def test_depth_runner_uses_camera_schedule():
    """CNN actor: fixed learning rate and 8 mini-batches (3.0 camera-actor settings); rough/flat PPO unchanged."""
    _, rough = _resolve(ROUGH)
    _, flat = _resolve(FLAT)
    _, depth = _resolve(DEPTH)
    for agent in (rough, flat):
        algo = agent.algorithm
        assert (algo.schedule, algo.learning_rate, algo.num_mini_batches) == ("adaptive", 1.0e-3, 4)
    algo = depth.algorithm
    assert (algo.schedule, algo.learning_rate, algo.num_mini_batches) == ("fixed", 7.0e-5, 8)
    for field in ("value_loss_coef", "entropy_coef", "clip_param", "num_learning_epochs", "gamma", "lam", "desired_kl"):
        assert getattr(algo, field) == getattr(rough.algorithm, field), field
    assert depth.check_for_nan  # turned off only after a NaN-free smoke run (agent.check_for_nan=false)
    _, overridden = _resolve(DEPTH, ["agent.check_for_nan=false"])
    assert overridden.check_for_nan is False


def test_depth_camera_mount_sees_no_robot():
    """Ray-cast the configured 64x48 camera against the robot's own meshes at the default pose (no Kit)."""
    study = _script("depth_camera_study")
    camera, usd_path, joint_pos = study.configured_setup()
    triangles, standing_height = study.posed_robot_triangles(usd_path, joint_pos)
    assert standing_height == pytest.approx(0.451, abs=0.005)  # trunk origin above flat ground, default pose
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
    # the pitch helper reproduces the 2.x quaternion
    assert study.pitch_down_quat(50.0) == pytest.approx(study.MOUNT_2X[1], abs=2e-4)


def _joint_log(names: list[str], table_names: list[str] | None = None) -> str:
    """Console log lines as Isaac Lab 3.0 writes them (joint_actions.py:70-73, PhysX articulation.py:4284-4317)."""
    import io
    import logging

    from prettytable import PrettyTable

    table = PrettyTable()
    table.title = "Simulation Joint Information (Prim path: /World/envs/env_.*/Robot)"
    table.field_names = ["Index", "Name", "Stiffness", "Damping", "Armature", "Static Friction", "Dynamic Friction"]
    table.align["Name"] = "l"
    for i, name in enumerate(table_names or names):
        table.add_row([i, name, "0.000", "0.000", "0.000", "0.000", "0.000"])
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("[INFO]: %(message)s"))  # the Kit launcher's Isaac Lab INFO handler
    logger = logging.getLogger("test_port.joint_log")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("Simulation parameters for joints in /World/envs/env_.*/Robot:\n" + table.get_string())
        logger.info(f"Resolved joint names for the action term JointPositionAction: {names} [ProxyArray(...)]")
    finally:
        logger.removeHandler(handler)
    return "[Info] [omni.kit] startup noise\n" + stream.getvalue()


def test_check_joint_order(tmp_path):
    check = _script("check_joint_order")
    expected = check.expected_joint_order(check.DEFAULT_USD)
    legs = ("FL", "FR", "RL", "RR")
    assert expected == [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf") for leg in legs]

    log = tmp_path / "ok.log"
    log.write_text(_joint_log(expected))
    found = check.parse_log(log.read_text())
    assert found == {"action term JointPositionAction": expected, "articulation /World/envs/env_.*/Robot": expected}
    assert check.main([str(log)]) == 0

    depth_first = [f"{leg}_{part}_joint" for leg in legs for part in ("hip", "thigh", "calf")]  # e.g. Newton's order
    log.write_text(_joint_log(depth_first))
    assert check.main([str(log)]) == 1
    log.write_text("no records\n")
    assert check.main([str(log)]) == 2


def test_convert_log_std_checkpoint(tmp_path):
    """A log-std checkpoint converts to ``distribution.log_std_param`` and says which agent cfg it needs."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    conv = _converter()
    torch.manual_seed(1)
    sd = {"log_std": torch.log(torch.rand(12) * 0.5 + 0.1)}
    for net, out in (("actor", 12), ("critic", 1)):
        dims = [48, 64, 64, out]
        for i in range(3):
            sd[f"{net}.{2 * i}.weight"] = torch.randn(dims[i + 1], dims[i]) * 0.1
            sd[f"{net}.{2 * i}.bias"] = torch.randn(dims[i + 1]) * 0.1
    old = {"model_state_dict": sd, "optimizer_state_dict": None, "iter": 3, "infos": None}
    new, info = conv.convert_checkpoint(old)
    assert info["std_type"] == "log" and "std_type='log'" in info["agent_cfg_hint"]
    assert "distribution.log_std_param" in new["actor_state_dict"]

    obs = TensorDict({"policy": torch.randn(16, 48)}, batch_size=[16])
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"}
    actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, [64, 64], "elu", distribution_cfg=dist)
    actor.load_state_dict(new["actor_state_dict"], strict=True)
    actor.eval()
    with torch.inference_mode():
        assert torch.allclose(actor(obs), conv.old_forward(sd, "actor", obs["policy"]), atol=1e-6)
        actor.distribution.update(actor(obs))
        assert torch.allclose(actor.distribution.std[0], sd["log_std"].exp(), atol=1e-6)

    # the default (scalar-std) agent cfg is refused up front, with the hint instead of a strict-load key error
    old_path, new_path = tmp_path / "old.pt", tmp_path / "new.pt"
    torch.save(old, old_path)
    conv.save_checkpoint(new, new_path, force=False)
    with pytest.raises(ValueError, match="std_type='log'"):
        conv.verify(old_path, new_path, conv.DEFAULT_AGENT_CFG, 8, 0)


def test_parity_with_2x_sources():
    """Static 2.x config (2.3.2 base + Hector's overrides) vs the instantiated 3.0 cfg; needs the 2.x folders."""
    parity = _script("check_obs_parity")
    if not (parity.DEFAULT_OLD_LAB / ".git").exists() or not (parity.DEFAULT_OLD_TASK / "rough_env_cfg.py").is_file():
        pytest.skip("the 2.x sources (~/IsaacLab, ~/quadruped_rough_terrain) are not on this machine")
    if not CHECKPOINT_2X.is_file() or _is_lfs_pointer(CHECKPOINT_2X):
        pytest.skip(f"{CHECKPOINT_2X.name} is missing or an un-smudged Git LFS pointer (run `git lfs pull`)")
    assert parity.main([]) == 0

    from quadruped_lab.tasks.locomotion.config.custom_quad.rough_env_cfg import CustomQuadRoughEnvCfg

    old = parity.load_old_sources(parity.DEFAULT_OLD_LAB, parity.DEFAULT_OLD_TASK)
    assert not old["skipped"]
    comparer = parity.compare_with_2x(old["state"], CustomQuadRoughEnvCfg(), old["defaults"])
    assert not comparer.problems and comparer.checked > 200
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
