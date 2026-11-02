"""Path-following velocity command: pure pursuit on a planned route -> ``(vx, vy, yaw_rate)`` in the base frame."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from drone_lab.teaming.follower import FollowerCfg, PurePursuitFollower, VelocityLimits

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import PathFollowCommandCfg

logger = logging.getLogger(__name__)


def _env_id_list(env_ids: torch.Tensor | Sequence[int] | slice | None, num_envs: int) -> list[int]:
    if env_ids is None:
        return list(range(num_envs))
    if isinstance(env_ids, slice):
        return list(range(num_envs))[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return [int(i) for i in env_ids.flatten().tolist()]
    return [int(i) for i in env_ids]


class PathFollowCommand(CommandTerm):
    """Follows a planned route with :class:`drone_lab.teaming.follower.PurePursuitFollower` (one per env).

    Public state (read by the reset event, the terminations and ``scripts/rescue_finale.py``):

    * :attr:`command` -- ``(num_envs, 3)`` body-frame ``(vx, vy, yaw_rate)``, clipped to the policy's ranges.
    * :attr:`path_xy`, :attr:`path_z`, :attr:`path_s`, :attr:`path_length` -- the route (numpy).
    * :attr:`person_xy` -- where the person is.
    * :attr:`progress_s`, :attr:`cross_track`, :attr:`distance_to_goal`, :attr:`goal_reached`,
      :attr:`time_at_goal`, :attr:`target_w` -- follower state per env.
    * :attr:`respawn_s` -- arclength where each env was (re)spawned last.
    """

    cfg: PathFollowCommandCfg

    def __init__(self, cfg: PathFollowCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self._terrain_query = None
        path_xy, path_z, person_xy, source = self._resolve_route()
        self.route_source = source
        self.path_xy = path_xy
        self.path_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))])
        self.path_length = float(self.path_s[-1])
        self.path_z = path_z
        self.person_xy = person_xy
        self.follower_cfg = FollowerCfg(
            cruise_speed=cfg.cruise_speed,
            min_speed=min(cfg.min_speed, cfg.cruise_speed),
            lookahead_m=cfg.lookahead_m,
            lateral_gain=cfg.lateral_gain,
            max_lateral_speed=cfg.max_lateral_speed,
            max_yaw_rate=cfg.max_yaw_rate,
            goal_tolerance_m=cfg.goal_tolerance_m,
            limits=VelocityLimits.from_ranges(cfg.ranges),
        )
        self.followers = [PurePursuitFollower(path_xy, self.follower_cfg) for _ in range(self.num_envs)]
        n = self.num_envs
        self.vel_command_b = torch.zeros(n, 3, device=self.device)
        self.progress_s = np.zeros(n)
        self.cross_track = np.zeros(n)
        self.distance_to_goal = np.full(n, self.path_length)
        self.heading_error = np.zeros(n)
        self.goal_reached = np.zeros(n, dtype=bool)
        self.time_at_goal = np.zeros(n)
        self.respawn_s = np.zeros(n)
        self.target_w = torch.zeros(n, 3, device=self.device)
        self.modes = ["track"] * n
        self.standing_height = float(self.robot.cfg.init_state.pos[2])
        self.metrics["progress"] = torch.zeros(n, device=self.device)
        self.metrics["cross_track"] = torch.zeros(n, device=self.device)
        self.metrics["distance_to_goal"] = torch.zeros(n, device=self.device)
        # markers along the route (every marker_spacing metres, slightly above the ground)
        k = max(
            1,
            int(
                round(cfg.marker_spacing / max(np.median(np.diff(self.path_s)) if len(self.path_s) > 1 else 1.0, 1e-3))
            ),
        )
        pts = np.column_stack([path_xy[::k], path_z[::k] + 0.06])
        self._route_marker_points = torch.tensor(pts, dtype=torch.float32, device=self.device)
        goal = np.array([[path_xy[-1, 0], path_xy[-1, 1], path_z[-1] + 0.03]])
        self._goal_marker_points = torch.tensor(goal, dtype=torch.float32, device=self.device)
        logger.info(str(self))

    def __str__(self) -> str:
        return (
            "PathFollowCommand:\n"
            f"\tRoute            : {len(self.path_xy)} points, {self.path_length:.1f} m ({self.route_source})\n"
            f"\tStart / end [m]  : {self.path_xy[0].round(2).tolist()} -> {self.path_xy[-1].round(2).tolist()}\n"
            f"\tPerson [m]       : {np.round(self.person_xy, 2).tolist()}\n"
            f"\tCruise speed     : {self.cfg.cruise_speed} m/s, look-ahead {self.cfg.lookahead_m} m\n"
            f"\tCommand ranges   : vx {self.cfg.ranges.lin_vel_x}, vy {self.cfg.ranges.lin_vel_y}, "
            f"yaw rate {self.cfg.ranges.ang_vel_z}"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Body-frame velocity command ``(vx, vy, yaw_rate)``. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Route
    """

    def _terrain(self):
        """Exact height query on the stage terrain mesh (built lazily)."""
        if self._terrain_query is None:
            from drone_lab.mapping import MeshHeightQuery
            from drone_lab.tasks.survey.mdp.terrain_utils import terrain_mesh_from_stage

            mesh = terrain_mesh_from_stage(self.cfg.terrain_prim_path)
            if mesh is None:
                raise RuntimeError(f"no terrain mesh under '{self.cfg.terrain_prim_path}'")
            self._terrain_mesh = mesh
            self._terrain_query = MeshHeightQuery(*mesh)
        return self._terrain_query

    def terrain_height(self, xy: np.ndarray) -> np.ndarray:
        """Terrain height under world points (N, 2) from the stage mesh [m]."""
        z = self._terrain()(np.asarray(xy, dtype=float).reshape(-1, 2))
        return np.nan_to_num(z, nan=0.0)

    def _resolve_route(self):
        cfg = self.cfg
        path_xy = path_z = person = None
        if cfg.path_xy is not None:
            path_xy = np.asarray(cfg.path_xy, dtype=float).reshape(-1, 2)
            path_z = None if cfg.path_z is None else np.asarray(cfg.path_z, dtype=float).ravel()
            source = "given route"
        elif cfg.path_file:
            data = json.loads(Path(cfg.path_file).expanduser().read_text())
            path_xy = np.asarray(data["path_xy"], dtype=float).reshape(-1, 2)
            path_z = np.asarray(data["path_z"], dtype=float).ravel() if data.get("path_z") is not None else None
            person = data.get("person_xy")
            source = f"route file {cfg.path_file}"
        else:
            path_xy, path_z = self._plan_on_stage(cfg.start_xy, cfg.goal_xy)
            person = cfg.goal_xy
            source = f"planned at start-up on the stage terrain ({cfg.plan_resolution} m)"
        if len(path_xy) == 0:
            raise ValueError("the route is empty")
        if path_z is None or len(path_z) != len(path_xy) or not np.all(np.isfinite(path_z)):
            path_z = self.terrain_height(path_xy)
        if cfg.person_xy is not None:
            person = cfg.person_xy
        person_xy = np.asarray(person if person is not None else path_xy[-1], dtype=float).ravel()[:2]
        return path_xy, path_z, person_xy, source

    def _plan_on_stage(self, start_xy, goal_xy):
        """Plan on the terrain read back from the stage (used when no route is given, e.g. ``isaaclab play``)."""
        from drone_lab.mapping import GridSpec, ground_truth_map
        from drone_lab.teaming.inputs import DEFAULT_FOOTPRINT_RADII
        from drone_lab.teaming.planner import plan_path
        from drone_lab.teaming.traversability import Footprint, compute_traversability

        self._terrain()
        vertices, faces = self._terrain_mesh
        lo, hi = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
        grid = GridSpec.from_bounds(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), self.cfg.plan_resolution)
        z, _ = ground_truth_map(vertices, faces, grid, supersample=3, query=self._terrain_query)
        person = Footprint(float(goal_xy[0]), float(goal_xy[1]), DEFAULT_FOOTPRINT_RADII["person"], "person")
        tmap = compute_traversability(z, grid, footprints=[person])
        plan = plan_path(tmap, start_xy, goal_xy)
        if not plan.success:
            raise RuntimeError(f"rescue route planning failed: {plan.message}")
        logger.info("rescue route: %s", plan.message)
        return plan.path_xy, plan.path_z

    def route_pose(self, s: float) -> tuple[float, float, float, float]:
        """``(x, y, terrain z, yaw)`` of the route at arclength ``s`` (yaw = route direction a little ahead)."""
        s = float(np.clip(s, 0.0, self.path_length))
        x = float(np.interp(s, self.path_s, self.path_xy[:, 0]))
        y = float(np.interp(s, self.path_s, self.path_xy[:, 1]))
        z = float(np.interp(s, self.path_s, self.path_z))
        s2 = min(s + 0.5, self.path_length)
        if s2 - s < 1e-3:
            s, s2 = max(0.0, s2 - 0.5), s2
        x0, y0 = np.interp(s, self.path_s, self.path_xy[:, 0]), np.interp(s, self.path_s, self.path_xy[:, 1])
        x1, y1 = np.interp(s2, self.path_s, self.path_xy[:, 0]), np.interp(s2, self.path_s, self.path_xy[:, 1])
        yaw = math.atan2(y1 - y0, x1 - x0) if (abs(x1 - x0) + abs(y1 - y0)) > 1e-6 else 0.0
        return x, y, z, yaw

    def prepare_respawn(self, env_ids: torch.Tensor | Sequence[int] | slice | None) -> np.ndarray:
        """Decide where the reset envs (re)start on the route and remember it (:attr:`respawn_s`).

        A fall (an env that terminated without a time-out) respawns where it fell, :attr:`cfg.respawn_backoff_m`
        behind the progress it had made, when :attr:`cfg.respawn_on_fall` is set; everything else starts at 0.
        """
        ids = _env_id_list(env_ids, self.num_envs)
        terminated = None
        tm = getattr(self._env, "termination_manager", None)
        if tm is not None:
            try:
                terminated = tm.terminated.detach().cpu().numpy()
            except Exception:  # before the first step
                terminated = None
        for i in ids:
            fell = bool(terminated[i]) if terminated is not None else False
            if self.cfg.respawn_on_fall and fell and self.progress_s[i] > 0.0:
                self.respawn_s[i] = max(0.0, self.progress_s[i] - self.cfg.respawn_backoff_m)
            else:
                self.respawn_s[i] = 0.0
        return self.respawn_s[ids]

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.metrics["progress"][:] = torch.as_tensor(
            self.progress_s / max(self.path_length, 1e-6), dtype=torch.float32, device=self.device
        )
        self.metrics["cross_track"][:] = torch.as_tensor(
            np.abs(self.cross_track), dtype=torch.float32, device=self.device
        )
        self.metrics["distance_to_goal"][:] = torch.as_tensor(
            self.distance_to_goal, dtype=torch.float32, device=self.device
        )

    def _resample_command(self, env_ids: Sequence[int] | slice):
        for i in _env_id_list(env_ids, self.num_envs):
            s0 = float(self.respawn_s[i])
            self.followers[i].reset(s0)
            self.progress_s[i] = s0
            self.goal_reached[i] = False
            self.time_at_goal[i] = 0.0
            self.cross_track[i] = 0.0
            self.modes[i] = "track"
        ids = torch.as_tensor(_env_id_list(env_ids, self.num_envs), dtype=torch.long, device=self.device)
        self.vel_command_b[ids] = 0.0

    def _update_command(self):
        pos = self.robot.data.root_pos_w.torch[:, :2].detach().cpu().numpy().astype(float)
        yaw = self.robot.data.heading_w.torch.detach().cpu().numpy().astype(float)
        cmd = np.zeros((self.num_envs, 3))
        target = np.zeros((self.num_envs, 2))
        for i, f in enumerate(self.followers):
            out = f.compute(pos[i, 0], pos[i, 1], yaw[i])
            cmd[i] = (out.vx, out.vy, out.wz)
            target[i] = out.target_xy
            self.progress_s[i] = out.progress_m
            self.cross_track[i] = out.cross_track_m
            self.distance_to_goal[i] = out.distance_to_goal_m
            self.heading_error[i] = out.heading_error_rad
            self.modes[i] = out.mode
            self.goal_reached[i] = out.done
        self.time_at_goal = np.where(self.goal_reached, self.time_at_goal + self._env.step_dt, 0.0)
        self.vel_command_b[:] = torch.as_tensor(cmd, dtype=torch.float32, device=self.device)
        tz = np.interp(self.progress_s + self.follower_cfg.lookahead_m, self.path_s, self.path_z) + 0.12
        self.target_w[:] = torch.as_tensor(np.column_stack([target, tz]), dtype=torch.float32, device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "route_visualizer"):
                self.route_visualizer = VisualizationMarkers(self.cfg.route_visualizer_cfg)
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.route_visualizer.set_visibility(True)
            self.target_visualizer.set_visibility(True)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "route_visualizer"):
            self.route_visualizer.set_visibility(False)
            self.target_visualizer.set_visibility(False)
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot may be de-initialized, and the buffers exist only after __init__ finished
        if not self.robot.is_initialized or not hasattr(self, "_route_marker_points"):
            return
        self.route_visualizer.visualize(translations=self._route_marker_points)
        self.goal_visualizer.visualize(translations=self._goal_marker_points)
        self.target_visualizer.visualize(translations=self.target_w)
