"""Reference-trajectory command: a smooth moving target with look-ahead points."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from drone_lab.reference import ReferenceBank

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import TrajectoryReferenceCommandCfg


class TrajectoryReferenceCommand(CommandTerm):
    """Evaluates one :class:`~drone_lab.reference.ReferenceBank` trajectory per env at the episode time.

    The reset event (:func:`reset_near_reference`) draws the new trajectory (:meth:`sample_reference`) before it
    places the drone, because events run before the command manager resets; :meth:`_resample_command` then only
    restarts the clock for those envs.
    """

    cfg: TrajectoryReferenceCommandCfg

    def __init__(self, cfg: TrajectoryReferenceCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.bank = ReferenceBank(self.num_envs, device=self.device)
        self.t = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lookahead = torch.tensor(cfg.lookahead_times, dtype=torch.float32, device=self.device)
        self.ref_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.lookahead_w = torch.zeros(self.num_envs, len(cfg.lookahead_times), 3, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)
        self.sample_reference(torch.arange(self.num_envs, device=self.device))
        self._evaluate(torch.arange(self.num_envs, device=self.device))

    def __str__(self) -> str:
        return (
            "TrajectoryReferenceCommand:\n"
            f"\tShapes        : {self.cfg.shape_weights if self.cfg.fixed_shape is None else self.cfg.fixed_shape}\n"
            f"\tSize [m]      : {self.cfg.size_range if self.cfg.fixed_size is None else self.cfg.fixed_size}\n"
            f"\tSpeed [m/s]   : {self.cfg.speed_range if self.cfg.fixed_speed is None else self.cfg.fixed_speed}\n"
            f"\tLook-ahead [s]: {self.cfg.lookahead_times}"
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.ref_pos_w, self.ref_vel_w, self.lookahead_w.flatten(1)], dim=-1)

    def sample_reference(self, env_ids: torch.Tensor):
        """Draw new trajectories for ``env_ids`` (and mark them so the next reset does not redraw them)."""
        c = self.cfg
        self.bank.sample(
            env_ids,
            shape_weights=c.shape_weights,
            size_range=c.size_range,
            speed_range=c.speed_range,
            z_offset_range=c.z_offset_range,
            z_amp_range=c.z_amp_range,
            fixed_shape=c.fixed_shape,
            fixed_size=c.fixed_size,
            fixed_speed=c.fixed_speed,
            fixed_z_amp=c.fixed_z_amp,
        )
        self._fresh[env_ids] = True

    def reference_at(self, t: torch.Tensor, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame reference position and velocity at times ``t`` for ``env_ids``."""
        origin = self._env.scene.env_origins[env_ids]
        return origin + self.bank.position(t, env_ids), self.bank.velocity(t, env_ids)

    def _evaluate(self, env_ids: torch.Tensor):
        t = self.t[env_ids]
        self.ref_pos_w[env_ids], self.ref_vel_w[env_ids] = self.reference_at(t, env_ids)
        n, k = len(env_ids), len(self._lookahead)
        t_ahead = (t.unsqueeze(1) + self._lookahead.unsqueeze(0)).reshape(-1)
        ids = env_ids.repeat_interleave(k)
        pos, _ = self.reference_at(t_ahead, ids)
        self.lookahead_w[env_ids] = pos.view(n, k, 3)

    def _update_metrics(self):
        self.metrics["position_error"][:] = torch.linalg.norm(self.robot.data.root_pos_w.torch - self.ref_pos_w, dim=-1)
        self.metrics["velocity_error"][:] = torch.linalg.norm(
            self.robot.data.root_lin_vel_w.torch - self.ref_vel_w, dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = env_ids.to(dtype=torch.long)
        stale = env_ids[~self._fresh[env_ids]]
        if len(stale) > 0:
            self.sample_reference(stale)
        self._fresh[env_ids] = False
        self.t[env_ids] = 0.0
        self._evaluate(env_ids)

    def _update_command(self):
        self.t += self._env.step_dt
        self._evaluate(torch.arange(self.num_envs, device=self.device))

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "reference_visualizer"):
                self.reference_visualizer = VisualizationMarkers(self.cfg.reference_visualizer_cfg)
            self.reference_visualizer.set_visibility(True)
        elif hasattr(self, "reference_visualizer"):
            self.reference_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized or not hasattr(self, "lookahead_w"):
            return
        k = self.lookahead_w.shape[1]
        points = torch.cat([self.ref_pos_w, self.lookahead_w.reshape(-1, 3)])
        indices = torch.cat(
            [
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.ones(self.num_envs * k, dtype=torch.long, device=self.device),
            ]
        )
        env_ids = torch.cat([self._env.scene._ALL_INDICES, self._env.scene._ALL_INDICES.repeat_interleave(k)])
        self.reference_visualizer.visualize(translations=points, marker_indices=indices, environment_ids=env_ids)
