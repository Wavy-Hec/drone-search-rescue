"""Configuration of the survey carrot command."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

CARROT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Survey/carrot",
    markers={
        "carrot": sim_utils.SphereCfg(
            radius=0.18, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.1))
        )
    },
)
"""Red sphere at the moving target."""

PATH_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Survey/path",
    markers={
        "path": sim_utils.SphereCfg(
            radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.75, 1.0))
        )
    },
)
"""Small cyan spheres along the planned path."""


@configclass
class SurveyCarrotCommandCfg(CommandTermCfg):
    """A target ("carrot") that slides along a survey path at a fixed speed.

    The command value is the carrot position in the world frame, shape (num_envs, 3). The path is generated once
    at start-up from the terrain area and the highest terrain point (see :mod:`drone_lab.paths`).
    """

    class_type: type | str = "drone_lab.tasks.survey.mdp.commands:SurveyCarrotCommand"

    asset_name: str = "robot"
    """Scene entity whose root follows the carrot."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Never resample mid-episode; the carrot restarts only when the env resets."""

    pattern: str = "lawnmower"
    """Survey pattern: "lawnmower", "spiral", "figure8" or "waypoints"."""

    spacing: float = 4.0
    """Distance between lawnmower lines / spiral rings [m]. Keep it below the mapping footprint (5 m)."""

    altitude: float = 4.0
    """Flight height above the highest terrain point inside the area [m]."""

    speed: float = 1.0
    """Nominal carrot speed along the path [m/s]."""

    max_lead: float = 1.0
    """Maximum drone-carrot distance [m]; the carrot waits beyond it (training starts were within +-1 m)."""

    slow_lead: float = 0.7
    """Drone-carrot distance [m] where the carrot starts slowing down (it stops at :attr:`max_lead`)."""

    path_resolution: float = 0.25
    """Arclength spacing of the resampled path [m]."""

    area: tuple[float, float, float, float] | None = None
    """Survey rectangle (x_min, x_max, y_min, y_max) [m]. None uses the terrain generator's sub-terrain grid."""

    area_margin: float = 0.0
    """Inset of the survey rectangle from :attr:`area` [m]."""

    direction: str = "x"
    """Lawnmower line direction, "x" or "y"."""

    turn: str = "round"
    """Lawnmower turn style, "round" (U-turns outside the area) or "square"."""

    figure8_loops: int = 1
    """Number of figure-eight loops."""

    waypoints: list[list[float]] | None = None
    """Custom waypoints for ``pattern="waypoints"``: [x, y] (flown at the survey altitude) or [x, y, z] (absolute)."""

    multi_env_mode: str = "split"
    """With several envs: "split" the area into strips (lawnmower/spiral) or stagger closed loops; "copy" the path."""

    terrain_prim_path: str = "/World/ground"
    """Prim under which the terrain mesh is searched (for the area height range)."""

    marker_spacing: float = 1.0
    """Spacing of the path markers [m] when debug visualization is on."""

    carrot_visualizer_cfg: VisualizationMarkersCfg = CARROT_MARKER_CFG
    """Marker for the carrot."""

    path_visualizer_cfg: VisualizationMarkersCfg = PATH_MARKER_CFG
    """Markers along the planned path."""
