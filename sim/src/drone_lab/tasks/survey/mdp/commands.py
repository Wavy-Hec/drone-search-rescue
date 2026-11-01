"""Survey carrot command: a target that moves along a survey path."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from drone_lab.carrot import advance_carrot, interpolate_path, pad_paths
from drone_lab.paths import Area, build_env_paths, waypoints_array

from .terrain_utils import terrain_mesh_from_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import SurveyCarrotCommandCfg

logger = logging.getLogger(__name__)


class SurveyCarrotCommand(CommandTerm):
    """Moves a target along a survey path at :attr:`SurveyCarrotCommandCfg.speed` (see :mod:`drone_lab.carrot`).

    Public state used by observations, events, terminations and the mapping script:

    * :attr:`command` / :attr:`carrot_w` -- carrot position in the world frame (num_envs, 3).
    * :attr:`path_start_w` -- first path point, where the drone is reset (num_envs, 3).
    * :attr:`s`, :attr:`progress`, :attr:`segment_index` -- carrot arclength [m], fraction of the path, and
      index of the dense path segment it is on.
    * :attr:`done`, :attr:`time_since_done` -- carrot reached the end of the path, and for how long [s].
    * :attr:`waypoints` -- planned waypoint polylines (list of (N_i, 3) numpy arrays), and the resampled
      tensors :attr:`path_points`, :attr:`path_arclength`, :attr:`path_length`.
    * :attr:`area`, :attr:`terrain_z_min`, :attr:`terrain_z_max`, :attr:`flight_altitude`.
    """

    cfg: SurveyCarrotCommandCfg

    def __init__(self, cfg: SurveyCarrotCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]

        # survey area and terrain height range
        self.area = self._resolve_area(env)
        self.terrain_z_min, self.terrain_z_max = self._terrain_height_range(self.area)
        self.flight_altitude = self.terrain_z_max + float(cfg.altitude)

        # planned paths (numpy, one per env) and their padded, resampled tensors
        custom = None
        if cfg.waypoints is not None:
            custom = waypoints_array(cfg.waypoints, default_z=self.flight_altitude)
        self.waypoints: list[np.ndarray] = build_env_paths(
            cfg.pattern,
            self.num_envs,
            self.area,
            cfg.spacing,
            self.flight_altitude,
            waypoints=custom,
            multi_env_mode=cfg.multi_env_mode,
            direction=cfg.direction,
            turn=cfg.turn,
            figure8_loops=cfg.figure8_loops,
        )
        self.path_points, self.path_arclength, self.path_length = pad_paths(
            self.waypoints, cfg.path_resolution, device=self.device
        )

        # carrot state
        self.s = torch.zeros(self.num_envs, device=self.device)
        self.carrot_w = self.path_points[:, 0].clone()
        self.segment_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gate = torch.ones(self.num_envs, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_since_done = torch.zeros(self.num_envs, device=self.device)

        # metrics logged by the command manager on reset
        self.metrics["carrot_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress"] = torch.zeros(self.num_envs, device=self.device)

        # path markers for debug visualization
        stride = max(1, int(round(cfg.marker_spacing / cfg.path_resolution)))
        marker_pts, marker_env = [], []
        for env_id in range(self.num_envs):
            n_valid = int(torch.count_nonzero(torch.diff(self.path_arclength[env_id]) > 0).item()) + 1
            pts = self.path_points[env_id, :n_valid:stride]
            marker_pts.append(pts)
            marker_env.append(torch.full((len(pts),), env_id, dtype=torch.long, device=self.device))
        self._path_marker_points = torch.cat(marker_pts)
        self._path_marker_env_ids = torch.cat(marker_env)

        logger.info(str(self))

    def __str__(self) -> str:
        lengths = ", ".join(f"{v:.1f}" for v in self.path_length.tolist())
        return (
            "SurveyCarrotCommand:\n"
            f"\tPattern          : {self.cfg.pattern} (spacing {self.cfg.spacing} m, turn {self.cfg.turn})\n"
            f"\tArea [m]         : {self.area.as_tuple()}\n"
            f"\tTerrain z [m]    : {self.terrain_z_min:.2f} .. {self.terrain_z_max:.2f}\n"
            f"\tFlight z [m]     : {self.flight_altitude:.2f}\n"
            f"\tPath length [m]  : {lengths}\n"
            f"\tSpeed [m/s]      : {self.cfg.speed} (max lead {self.cfg.max_lead} m)"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Carrot position in the world frame. Shape is (num_envs, 3)."""
        return self.carrot_w

    @property
    def path_start_w(self) -> torch.Tensor:
        """First point of every path in the world frame. Shape is (num_envs, 3)."""
        return self.path_points[:, 0]

    @property
    def progress(self) -> torch.Tensor:
        """Fraction of the path covered by the carrot, in [0, 1]. Shape is (num_envs,)."""
        return self.s / self.path_length.clamp(min=1e-6)

    """
    Implementation specific functions.
    """

    def _resolve_area(self, env: ManagerBasedRLEnv) -> Area:
        if self.cfg.area is not None:
            area = Area(*self.cfg.area)
        else:
            area = None
            terrain_cfg = getattr(env.cfg.scene, "terrain", None)
            gen_cfg = getattr(terrain_cfg, "terrain_generator", None)
            if terrain_cfg is not None and terrain_cfg.terrain_type == "generator" and gen_cfg is not None:
                # the terrain generator centres its sub-terrain grid on the world origin
                half_x = 0.5 * gen_cfg.size[0] * gen_cfg.num_rows
                half_y = 0.5 * gen_cfg.size[1] * gen_cfg.num_cols
                area = Area(-half_x, half_x, -half_y, half_y)
            if area is None:
                mesh = terrain_mesh_from_stage(self.cfg.terrain_prim_path)
                if mesh is not None:
                    lo, hi = mesh[0].min(axis=0), mesh[0].max(axis=0)
                    area = Area(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))
                else:
                    area = Area(-10.0, 10.0, -10.0, 10.0)
        return area.inset(self.cfg.area_margin) if self.cfg.area_margin else area

    def _terrain_height_range(self, area: Area) -> tuple[float, float]:
        mesh = terrain_mesh_from_stage(self.cfg.terrain_prim_path)
        if mesh is None:
            logger.warning("No terrain mesh under '%s'; assuming flat ground at z=0.", self.cfg.terrain_prim_path)
            return 0.0, 0.0
        vertices = mesh[0]
        # include the U-turn band around the area
        inside = area.inset(-0.5 * self.cfg.spacing - 0.5).contains(vertices[:, :2])
        z = vertices[inside, 2] if np.any(inside) else vertices[:, 2]
        return float(z.min()), float(z.max())

    def _update_metrics(self):
        pos = self.robot.data.root_pos_w.torch
        self.metrics["carrot_distance"][:] = torch.linalg.norm(pos - self.carrot_w, dim=-1)
        self.metrics["progress"][:] = self.progress

    def _resample_command(self, env_ids: Sequence[int] | slice):
        # (re)start the carrot at the beginning of the path
        self.s[env_ids] = 0.0
        self.carrot_w[env_ids] = self.path_points[env_ids, 0]
        self.segment_index[env_ids] = 0
        self.gate[env_ids] = 1.0
        self.done[env_ids] = False
        self.time_since_done[env_ids] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        pos = self.robot.data.root_pos_w.torch
        s, carrot, gate = advance_carrot(
            self.s,
            self.path_points,
            self.path_arclength,
            pos,
            self.cfg.speed,
            dt,
            self.cfg.max_lead,
            self.cfg.slow_lead,
        )
        _, seg = interpolate_path(self.path_points, self.path_arclength, s)
        self.time_since_done += self.done.float() * dt
        self.s[:] = s
        self.carrot_w[:] = carrot
        self.gate[:] = gate
        self.segment_index[:] = seg
        self.done |= s >= self.path_length - 1e-4

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "carrot_visualizer"):
                self.carrot_visualizer = VisualizationMarkers(self.cfg.carrot_visualizer_cfg)
                self.path_visualizer = VisualizationMarkers(self.cfg.path_visualizer_cfg)
            self.carrot_visualizer.set_visibility(True)
            self.path_visualizer.set_visibility(True)
        elif hasattr(self, "carrot_visualizer"):
            self.carrot_visualizer.set_visibility(False)
            self.path_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot may be de-initialized, and the buffers exist only after __init__ finished
        if not self.robot.is_initialized or not hasattr(self, "_path_marker_points"):
            return
        self.carrot_visualizer.visualize(translations=self.carrot_w, environment_ids=self._env.scene._ALL_INDICES)
        self.path_visualizer.visualize(
            translations=self._path_marker_points, environment_ids=self._path_marker_env_ids
        )
