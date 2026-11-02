# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolved task configs: physics and execution defaults per backend, the 2.x policy interface, the depth task.

Configs are resolved exactly like ``isaaclab train`` / ``play`` resolve them (presets, ``physics=`` selector, Hydra
overrides, play mode) on the CPU, without launching Isaac Sim.
"""

import math
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("isaaclab_tasks", reason="needs Isaac Lab 3.0")
torch = pytest.importorskip("torch")

import quadruped_lab.tasks  # noqa: F401  (registers the project tasks)

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

ROUGH = "Isaac-Velocity-Rough-Custom-Quad-v0"
FLAT = "Isaac-Velocity-Flat-Custom-Quad-v0"
DEPTH = "Isaac-Velocity-Rough-Depth-Custom-Quad-v0"
TASKS = [FLAT, ROUGH, DEPTH]
PHYSX_SELECTIONS = ([], ["physics=isaacsim_physx"], ["physics=physx"])
NEWTON_SELECTIONS = (["physics=newton_mjwarp"], ["physics=newton_kamino"])
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


def _resolve(task_id: str, overrides: list[str] | tuple[str, ...] = (), play_mode: bool = False) -> tuple:
    """``(env_cfg, agent_cfg)`` of a task as the ``isaaclab`` CLI resolves it."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    return resolve_task_config(task_id, "rsl_rl_cfg_entry_point", play_mode=play_mode, overrides=list(overrides))


@pytest.mark.parametrize("task_id", TASKS)
def test_physics_defaults_to_isaacsim_physx(task_id: str) -> None:
    """The velocity base defaults to Newton in 3.0; the project defaults to Isaac Sim PhysX (and aliases `physx`)."""
    for overrides in PHYSX_SELECTIONS:
        env_cfg, _ = _resolve(task_id, overrides)
        assert type(env_cfg.sim.physics).__name__ == "PhysxCfg", overrides
    env_cfg, _ = _resolve(task_id, ["physics=newton_mjwarp"])
    assert type(env_cfg.sim.physics).__name__ == "NewtonCfg"


@pytest.mark.parametrize("task_id", TASKS)
def test_execution_settings_per_backend(task_id: str) -> None:
    """Isaac Sim PhysX runs with the 2.x execution settings; Newton and OvPhysX keep the Isaac Lab 3.0 defaults."""
    from isaaclab.sim import SimulationCfg
    from isaaclab_ov.physics import OvPhysxCfg

    native_default = SimulationCfg().use_newton_actuators  # Isaac Lab 3.0: native actuators
    assert native_default is True
    for play_mode in (False, True):
        for overrides in PHYSX_SELECTIONS:
            env_cfg, _ = _resolve(task_id, overrides, play_mode=play_mode)
            assert env_cfg.sim.use_newton_actuators is False, overrides  # Isaac Lab's actuator path, as in 2.x
            assert env_cfg.sim.physics.enable_external_forces_every_iteration is False, overrides  # 2.3.2 TGS
        for overrides in NEWTON_SELECTIONS:
            env_cfg, _ = _resolve(task_id, overrides, play_mode=play_mode)
            assert env_cfg.sim.use_newton_actuators is native_default, overrides
        env_cfg, _ = _resolve(task_id, ["physics=ovphysx"], play_mode=play_mode)
        assert env_cfg.sim.use_newton_actuators is native_default
        assert (
            env_cfg.sim.physics.enable_external_forces_every_iteration
            is OvPhysxCfg().enable_external_forces_every_iteration
        )


@pytest.mark.parametrize("task_id", TASKS)
def test_3x_execution_defaults_are_one_override_away(task_id: str) -> None:
    """The documented overrides restore the Isaac Lab 3.0 execution defaults on Isaac Sim PhysX, per run."""
    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    overrides = ["env.sim.use_newton_actuators=true", "env.sim.physics.enable_external_forces_every_iteration=true"]
    env_cfg, _ = _resolve(task_id, ["physics=isaacsim_physx", *overrides])
    assert env_cfg.sim.use_newton_actuators is SimulationCfg().use_newton_actuators is True
    assert (
        env_cfg.sim.physics.enable_external_forces_every_iteration
        is PhysxCfg().enable_external_forces_every_iteration
        is True
    )
    # the other way round, a Newton run can still switch to Isaac Lab's actuator path
    env_cfg, _ = _resolve(task_id, ["physics=newton_mjwarp", "env.sim.use_newton_actuators=false"])
    assert env_cfg.sim.use_newton_actuators is False


@pytest.mark.parametrize("task_id", TASKS)
def test_play_mode_matches_2x_play_classes(task_id: str) -> None:
    """``play_mode()`` makes the changes of the 2.x ``*_PLAY`` classes."""
    env_cfg, _ = _resolve(task_id, ["physics=isaacsim_physx"], play_mode=True)
    assert env_cfg.scene.num_envs <= 50
    assert env_cfg.events.push_robot is None and env_cfg.events.base_external_force_torque is None
    for group in env_cfg.observations.__dataclass_fields__:
        assert getattr(env_cfg.observations, group).enable_corruption is False
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is not None:
        assert (generator.num_rows, generator.num_cols, generator.curriculum) == (5, 5, False)
        assert env_cfg.scene.terrain.max_init_terrain_level is None


def test_rough_policy_layout_matches_checkpoint(require_file: Callable[[Path], Path]) -> None:
    """Term order and sizes of the rough policy group equal the 2.x checkpoint's 235 inputs."""
    pytest.importorskip("pxr", reason="needs USD (pxr)")
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
    stage = Usd.Stage.Open(str(require_file(Path(env_cfg.scene.robot.spawn.usd_path))))
    dofs = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert rays == 187 and len(dofs) == 12
    assert sum(size for _, size in POLICY_LAYOUT_2X) == 235
    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert env_cfg.actions.joint_pos.scale == 0.25
    assert (env_cfg.sim.dt, env_cfg.decimation) == (0.005, 4)


def test_events_and_actuator_keep_2x_semantics() -> None:
    """Mass randomization and the DC-motor parameters mean what they meant in 2.x."""
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


@pytest.mark.parametrize("task_id", TASKS)
def test_robot_preset_per_backend(task_id: str) -> None:
    """PhysX keeps robot.usd with native ordering; Newton gets robot_newton.usd in the PhysX joint/body order."""
    from quadruped_lab.assets.custom_quad import (
        CUSTOM_QUAD_CFG,
        CUSTOM_QUAD_NEWTON_USD_PATH,
        CUSTOM_QUAD_PHYSX_BODY_ORDER,
        CUSTOM_QUAD_PHYSX_JOINT_ORDER,
        CUSTOM_QUAD_USD_PATH,
    )

    for overrides in PHYSX_SELECTIONS:
        robot = _resolve(task_id, overrides)[0].scene.robot
        assert robot.spawn.usd_path == str(CUSTOM_QUAD_USD_PATH), overrides
        assert robot.joint_ordering is None and robot.body_ordering is None, overrides
    for overrides in NEWTON_SELECTIONS:
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


def test_depth_task_layout() -> None:
    """Camera mount, renderer, observation groups and actor-critic split of the experimental depth task."""
    from isaaclab.utils.math import quat_apply

    from isaaclab_tasks.core.velocity import mdp

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
    # image_depth plus a pose refresh from the camera's parent link before every read: on Isaac Sim PhysX + Newton
    # Warp the camera would otherwise render from its spawn pose for the whole run
    term = env_cfg.observations.depth.depth_image
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


def test_depth_runner_uses_camera_schedule() -> None:
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
    assert depth.check_for_nan  # the per-step NaN scan stays on by default; agent.check_for_nan=false turns it off
    _, overridden = _resolve(DEPTH, ["agent.check_for_nan=false"])
    assert overridden.check_for_nan is False


def test_depth_cnn_model_builds() -> None:
    """The CNN actor builds from the agent cfg for a (N, 1, 48, 64) depth group plus 48 proprio values."""
    pytest.importorskip("rsl_rl")
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
