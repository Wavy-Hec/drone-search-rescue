"""Event and termination terms of the rescue task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def reset_root_on_route(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    command_name: str = "base_velocity",
    xy_jitter: float = 0.0,
    yaw_jitter: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place the robot on its route, at rest, facing along it.

    At the first reset (and after a time-out) that is the start of the route; after a fall it is where the robot
    fell, a little behind (see :meth:`PathFollowCommand.prepare_respawn`). The root height is the terrain height
    under the spot plus the robot's standing height (``init_state.pos[2]``) plus a small margin.
    ``xy_jitter`` [m] and ``yaw_jitter`` [rad] add uniform noise per env (e.g. for several envs at once).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    s = term.prepare_respawn(env_ids)
    poses = np.array([term.route_pose(float(si)) for si in s], dtype=float).reshape(-1, 4)
    num = len(poses)
    pos = torch.tensor(poses[:, :3], dtype=torch.float32, device=env.device)
    pos[:, 2] += term.standing_height + term.cfg.spawn_height_offset
    yaw = torch.tensor(poses[:, 3], dtype=torch.float32, device=env.device)
    if xy_jitter > 0:
        pos[:, :2] += (torch.rand(num, 2, device=env.device) * 2.0 - 1.0) * xy_jitter
    if yaw_jitter > 0:
        yaw += (torch.rand(num, device=env.device) * 2.0 - 1.0) * yaw_jitter
    zeros = torch.zeros(num, device=env.device)
    quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)  # (x, y, z, w)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num, 6, device=env.device), env_ids=env_ids)


def route_complete(env: ManagerBasedRLEnv, command_name: str = "base_velocity", hold_s: float = 3.0) -> torch.Tensor:
    """True once the robot has stood at the end of its route for ``hold_s`` seconds (use as a time-out)."""
    term = env.command_manager.get_term(command_name)
    done = (term.time_at_goal >= hold_s) & term.goal_reached
    return torch.as_tensor(done, dtype=torch.bool, device=env.device)
