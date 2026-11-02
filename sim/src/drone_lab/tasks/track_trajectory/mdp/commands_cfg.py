"""Configuration of the reference-trajectory command."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

REFERENCE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/TrackTrajectory/reference",
    markers={
        "now": sim_utils.SphereCfg(
            radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1))
        ),
        "ahead": sim_utils.SphereCfg(
            radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.1))
        ),
    },
)
"""Red sphere: reference now; orange spheres: look-ahead points."""


@configclass
class TrajectoryReferenceCommandCfg(CommandTermCfg):
    """A time-parameterised reference trajectory around every env origin (see :mod:`drone_lab.reference`).

    Command value (world frame), shape (num_envs, 6 + 3 * len(lookahead_times)):
    ``[reference position (3), reference velocity (3), look-ahead positions (3 each)]``.
    """

    class_type: type | str = "drone_lab.tasks.track_trajectory.mdp.commands:TrajectoryReferenceCommand"

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """A new trajectory is drawn at every reset, never mid-episode."""

    shape_weights: dict[str, float] = {"circle": 1.0, "figure8": 1.0, "lissajous": 1.0, "random": 1.0}
    """Sampling weights of the shapes."""

    size_range: tuple[float, float] = (0.8, 2.5)
    """Radius / half-width of the shapes [m]."""

    speed_range: tuple[float, float] = (0.3, 1.5)
    """Peak reference speed [m/s]."""

    z_offset_range: tuple[float, float] = (-0.5, 0.5)
    """Height of the trajectory relative to the env origin [m]."""

    z_amp_range: tuple[float, float] = (0.0, 0.5)
    """Amplitude of the vertical oscillation [m]."""

    lookahead_times: tuple[float, ...] = (0.5, 1.0)
    """Look-ahead times [s] of the extra reference points in the command / observations."""

    fixed_shape: str | None = None
    """Evaluation: force a shape ("circle", "figure8", "lissajous", "random") with deterministic phases."""

    fixed_size: float | None = None
    fixed_speed: float | None = None
    fixed_z_amp: float | None = None

    reference_visualizer_cfg: VisualizationMarkersCfg = REFERENCE_MARKER_CFG
