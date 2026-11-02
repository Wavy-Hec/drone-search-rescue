"""Observation, event, reward and termination terms of the survey task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from drone_lab.carrot import clamp_offset

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

"""
Observations.
"""


def position_relative_to_target(
    env: ManagerBasedRLEnv,
    command_name: str = "survey_target",
    max_offset: float | None = 1.0,
    per_axis_limit: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Drone root position minus the commanded target position, world frame, clamped [m]. Shape (num_envs, 3).

    Drop-in replacement for ``mdp.root_pos_w`` of the built-in tracking task, which returns the root position
    relative to the env origin, i.e. relative to that task's (fixed) goal. The clamp keeps the value inside the
    +-1 m start box the policy was trained on.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, :3]
    return clamp_offset(asset.data.root_pos_w.torch - target, max_offset, per_axis_limit)


"""
Events.
"""


def reset_root_to_path_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    command_name: str = "survey_target",
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Deterministic reset: place the drone (at rest, level, given yaw) at the start of its survey path."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    num = len(env_ids)
    pos = term.path_start_w[env_ids] + torch.tensor(position_offset, device=env.device, dtype=torch.float32)
    zeros = torch.zeros(num, device=env.device)
    quat = math_utils.quat_from_euler_xyz(zeros, zeros, zeros + float(yaw))  # (x, y, z, w)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num, 6, device=env.device), env_ids=env_ids)


"""
Rewards (only logged in play; the survey task is not meant for training).
"""


def target_distance_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "survey_target",
    std: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-|p - target|^2 / std^2)``, the survey counterpart of the built-in ``distance_to_goal_exp``."""
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, :3]
    d2 = torch.sum(torch.square(asset.data.root_pos_w.torch - target), dim=1)
    return torch.exp(-d2 / std**2)


"""
Terminations.
"""


def survey_complete(env: ManagerBasedRLEnv, command_name: str = "survey_target", hold_s: float = 5.0) -> torch.Tensor:
    """True ``hold_s`` seconds after the carrot reached the end of the path (use as a time-out)."""
    term = env.command_manager.get_term(command_name)
    return term.done & (term.time_since_done >= hold_s)
