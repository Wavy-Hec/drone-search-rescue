# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less checks of the 2.x -> 3.0 port: configs, observation interface and checkpoint conversion.

Everything here runs on the CPU without launching Isaac Sim; the simulation-level checks are listed in PORT_NOTES.md.
"""

import importlib.util
import math
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


def _converter():
    spec = importlib.util.spec_from_file_location("convert_rslrl_checkpoint", PROJECT / "scripts/convert_rslrl_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors.ray_caster.patterns import grid_pattern
    from pxr import Usd, UsdPhysics

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
    assert (mass["mass_distribution_params"], mass["operation"], mass["distribution"]) == ((-1.0, 3.0), "add", "uniform")
    motor = env_cfg.scene.robot.actuators["base_legs"]
    assert (motor.actuator_effort_limit, motor.joint_effort_limit, motor.saturation_effort) == (45.0, 45.0, 45.0)
    assert (motor.actuator_velocity_limit, motor.stiffness, motor.damping) == (21.0, 60.0, 1.5)
    assert Path(env_cfg.scene.robot.spawn.usd_path).is_file()
    assert env_cfg.scene.robot.joint_ordering is None


def test_depth_task_layout():
    from isaaclab.utils.math import quat_apply

    env_cfg, agent_cfg = _resolve(DEPTH, ["physics=isaacsim_physx"])
    camera = env_cfg.scene.depth_camera
    # 2.x offset (w, x, y, z) -> 3.0 (x, y, z, w)
    w, x, y, z = (0.66446, 0.24184, -0.24184, -0.66446)
    assert camera.offset.rot == (x, y, z, w)
    forward = quat_apply(torch.tensor([camera.offset.rot]), torch.tensor([[0.0, 0.0, -1.0]]))[0]  # OpenGL looks -Z
    assert math.degrees(math.asin(-forward[2].item())) == pytest.approx(50.0, abs=0.1) and forward[0] > 0
    assert camera.update_period == pytest.approx(env_cfg.decimation * env_cfg.sim.dt)
    assert type(camera.renderer_cfg).__name__ == "NewtonWarpRendererCfg"
    term = env_cfg.observations.depth.depth_image
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
        {"policy": torch.zeros(2, 48), "depth": torch.zeros(2, 1, cam.height, cam.width), "privileged": torch.zeros(2, 187)},
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
    new, info = conv.convert_checkpoint({"model_state_dict": sd, "optimizer_state_dict": None, "iter": 7, "infos": None})
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
