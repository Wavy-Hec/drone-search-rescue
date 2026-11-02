"""Observation, event, reward and termination terms of the trajectory-tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _term(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


"""
Observations (world frame; the policy is rewarded for holding yaw at zero, as in the built-in task).
"""


def reference_position_error(
    env: ManagerBasedRLEnv,
    command_name: str = "reference",
    clip: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Drone position minus reference position [m], clipped per axis. Shape (num_envs, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_pos_w.torch - _term(env, command_name).ref_pos_w, -clip, clip)


def reference_velocity_error(
    env: ManagerBasedRLEnv,
    command_name: str = "reference",
    clip: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Drone world linear velocity minus reference velocity [m/s], clipped per axis. Shape (num_envs, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w.torch - _term(env, command_name).ref_vel_w, -clip, clip)


def reference_lookahead(
    env: ManagerBasedRLEnv,
    command_name: str = "reference",
    clip: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Look-ahead reference points relative to the drone [m], clipped. Shape (num_envs, 3 * n_lookahead)."""
    asset: Articulation = env.scene[asset_cfg.name]
    rel = _term(env, command_name).lookahead_w - asset.data.root_pos_w.torch.unsqueeze(1)
    return torch.clamp(rel, -clip, clip).flatten(1)


"""
Events.
"""


def reset_near_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    command_name: str = "reference",
    position_noise: float = 0.3,
    attitude_noise: float = 0.26,
    velocity_noise: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Draw a new reference and place the drone near its start (position, attitude and velocity noise)."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = _term(env, command_name)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = env_ids.to(dtype=torch.long)
    n = len(env_ids)
    term.sample_reference(env_ids)
    p0, v0 = term.reference_at(torch.zeros(n, device=env.device), env_ids)

    def noise(scale, dims):
        return (2.0 * torch.rand(n, dims, device=env.device) - 1.0) * scale

    att = noise(attitude_noise, 3)
    quat = math_utils.quat_from_euler_xyz(att[:, 0], att[:, 1], att[:, 2])
    pose = torch.cat([p0 + noise(position_noise, 3), quat], dim=-1)
    vel = torch.cat([v0 + noise(velocity_noise, 3), noise(velocity_noise, 3)], dim=-1)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)


"""
Rewards.
"""


def reference_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "reference",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-|p - p_ref|^2 / std^2)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.sum(torch.square(asset.data.root_pos_w.torch - _term(env, command_name).ref_pos_w), dim=1)
    return torch.exp(-err / std**2)


def reference_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "reference",
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-|v - v_ref|^2 / std^2)`` (world frame)."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.sum(torch.square(asset.data.root_lin_vel_w.torch - _term(env, command_name).ref_vel_w), dim=1)
    return torch.exp(-err / std**2)


"""
Terminations.
"""


def reference_too_far(
    env: ManagerBasedRLEnv,
    command_name: str = "reference",
    max_distance: float = 2.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the drone is more than ``max_distance`` from the reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.root_pos_w.torch - _term(env, command_name).ref_pos_w, dim=-1) > max_distance
