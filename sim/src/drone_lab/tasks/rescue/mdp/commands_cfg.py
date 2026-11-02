"""Configuration of the path-following velocity command."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

ROUTE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Rescue/route",
    markers={
        "route": sim_utils.SphereCfg(
            radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.41, 0.2))
        )
    },
)
"""Small orange spheres along the planned route."""

TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Rescue/target",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.07, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.47, 0.84))
        )
    },
)
"""Blue sphere at the pure-pursuit look-ahead point."""

GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Rescue/goal",
    markers={
        "goal": sim_utils.CylinderCfg(
            radius=0.35,
            height=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.11, 0.69, 0.48)),
        )
    },
)
"""Flat green disc where the robot is meant to stop (next to the person)."""


@configclass
class PathFollowCommandCfg(CommandTermCfg):
    """Body-frame velocity command ``(vx, vy, yaw_rate)`` that makes the robot follow a planned route.

    It replaces ``UniformVelocityCommandCfg`` under the same term name (``base_velocity``), so the policy's
    ``velocity_commands`` observation and the reward terms are unchanged; only the source of the command differs.
    The route comes from :attr:`path_xy` (set by ``scripts/rescue_finale.py``), from :attr:`path_file`, or is
    planned at start-up on the terrain read back from the stage (:attr:`start_xy` -> :attr:`goal_xy`).
    """

    class_type: type | str = "drone_lab.tasks.rescue.mdp.commands:PathFollowCommand"

    asset_name: str = "robot"
    """The robot that follows the route."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """The route is never resampled mid-episode."""

    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
    )
    """Command ranges of the locomotion policy; every output is clipped to them (copied from the replaced
    velocity command by :func:`drone_lab.tasks.rescue.rescue_env_cfg.apply_rescue_task`)."""

    # ------------------------------------------------------------------------------------------- route
    path_xy: list[list[float]] | None = None
    """Route waypoints ``[[x, y], ...]`` in the world frame [m] (dense, e.g. every 0.1 m)."""

    path_z: list[float] | None = None
    """Terrain height under each waypoint [m] (markers, respawn height). None: read from the stage terrain."""

    path_file: str | None = None
    """JSON file with ``path_xy`` (and optionally ``path_z``, ``person_xy``), e.g. ``plan.json`` of a finale run."""

    start_xy: tuple[float, float] = (-23.0, -18.0)
    """Start of the route planned at start-up when no route is given [m] (on the flat border, next to the base)."""

    goal_xy: tuple[float, float] = (-4.5, -12.5)
    """Person the route planned at start-up leads to when no route is given [m]."""

    person_xy: tuple[float, float] | None = None
    """Where the person is [m] (for the distance metric); defaults to :attr:`goal_xy` / the end of the route."""

    plan_resolution: float = 0.25
    """Cell size of the start-up plan on the stage terrain [m]."""

    terrain_prim_path: str = "/World/ground"
    """Terrain mesh prim (start-up planning and heights)."""

    # ----------------------------------------------------------------------------------------- follower
    cruise_speed: float = 0.7
    """Forward speed on open, straight stretches [m/s]."""

    min_speed: float = 0.3
    """Lowest forward speed while driving [m/s] (the policy stands still for much smaller commands)."""

    lookahead_m: float = 0.8
    """Pure-pursuit look-ahead distance [m]."""

    lateral_gain: float = 0.6
    """Sideways correction per metre of cross-track error [1/s]."""

    max_lateral_speed: float = 0.25
    """Limit of the sideways correction [m/s]."""

    max_yaw_rate: float = 0.9
    """Yaw rate used to slow down on tight arcs [rad/s]."""

    goal_tolerance_m: float = 0.35
    """The robot stops this close to the end of the route [m]."""

    # ----------------------------------------------------------------------------------------- respawn
    respawn_on_fall: bool = True
    """After a fall (a non-time-out termination) the robot respawns on the route where it fell, a little
    behind (:attr:`respawn_backoff_m`), instead of at the start."""

    respawn_backoff_m: float = 0.5
    """How far behind the fall progress the robot respawns [m]."""

    spawn_height_offset: float = 0.08
    """Extra height above the robot's standing height when it is placed on the route [m]."""

    # ----------------------------------------------------------------------------------------- display
    marker_spacing: float = 0.5
    """Spacing of the route markers [m]."""

    route_visualizer_cfg: VisualizationMarkersCfg = ROUTE_MARKER_CFG
    target_visualizer_cfg: VisualizationMarkersCfg = TARGET_MARKER_CFG
    goal_visualizer_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG
