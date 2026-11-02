"""Ground robots of the rescue finale: task, policy, physics and what the planner may ask of each (no simulator).

* ``custom`` -- the custom quadruped (``quadruped_lab``): the converted Isaac Lab 2.x height-scan policy on Isaac
  Sim PhysX, the drone mission's backend, so the robot walks the very site the drone flew over.
* ``go2`` -- the Unitree Go2 with Isaac Lab's rough-terrain policy trained on Newton (the fallback; it flips within
  about 2 s on PhysX, so it walks on Newton).

The planner limits follow each policy's training terrain (Isaac Lab's rough-terrain curriculum) and the robot's
size; ``scripts/rescue_finale.py`` uses them unless a ``--max_*`` / ``--robot_radius`` flag overrides one.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from .traversability import TraversabilityCfg

__all__ = ["ROBOTS", "RobotProfile", "quadruped_lab_dir"]


def quadruped_lab_dir() -> Path:
    """Project folder of the custom quadruped: the installed ``quadruped_lab``, else the ``quadruped`` (or
    ``quadruped_lab``) folder next to this project."""
    spec = importlib.util.find_spec("quadruped_lab")
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parents[2]
    parent = Path(__file__).resolve().parents[4]
    for name in ("quadruped", "quadruped_lab"):
        if (parent / name / "pyproject.toml").is_file():
            return parent / name
    return parent / "quadruped"


@dataclass(frozen=True)
class RobotProfile:
    """One ground robot of the finale."""

    key: str
    """``--robot`` value."""
    name: str
    """Name in the figures and the video."""
    task: str
    """Rescue task id."""
    physics: str
    """Physics preset the policy walks on."""
    checkpoint: str
    """Default locomotion policy (RSL-RL checkpoint or TorchScript file); empty: ``--checkpoint`` is required."""
    traversability: TraversabilityCfg = field(default_factory=TraversabilityCfg)
    """Planner limits and weights."""
    cruise_speed: float = 0.7
    """Forward speed on open, straight stretches [m/s]."""
    lookahead_m: float = 0.8
    """Pure-pursuit look-ahead [m]."""
    camera_scale: float = 1.0
    """Follow-camera distance relative to a Go2-sized robot."""


CUSTOM_QUAD = RobotProfile(
    key="custom",
    name="Custom quadruped",
    task="DroneLab-Rescue-CustomQuad-v0",
    physics="isaacsim_physx",
    checkpoint=str(quadruped_lab_dir() / "checkpoints_2x" / "height_scanner_model_rslrl5.pt"),
    # trained on slopes up to 0.4 rise/run (21.8 deg), stairs up to 0.23 m, boxes up to 0.1 m and random rough
    # ground up to 0.06 m; 0.89 m long and 0.47 m wide (feet included), walking with the trunk about 0.7 m up.
    # 23 deg: the crater walls of the demo site fitted over half the body length; steeper walls are left out, since
    # the policy drifts downhill when it walks across them slowly. Turning on a steep descent makes it run away
    # downhill (yaw-rate commands of 0.4 rad/s or more on 17 deg and steeper: speed above 1.2 m/s in 5-74 % of the
    # cases, under 0.4 rad/s almost never), so the cross-slope term keeps routes on the fall line there
    traversability=TraversabilityCfg(
        max_slope_deg=23.0,
        max_step_m=0.15,
        max_roughness_m=0.05,
        footprint_radius_m=0.45,
        robot_radius_m=0.45,
        w_cross_slope=6.0,
    ),
    cruise_speed=0.7,
    lookahead_m=0.8,
    camera_scale=1.25,
)
"""The custom quadruped: ``quadruped_lab`` task ``Isaac-Velocity-Rough-Custom-Quad-v0``, converted 2.x checkpoint."""

GO2 = RobotProfile(
    key="go2",
    name="Go2",
    task="DroneLab-Rescue-Go2-v0",
    physics="newton_mjwarp",
    checkpoint="",  # no published policy: pass --checkpoint (a model_*.pt of Isaac-Velocity-Rough-UnitreeGo2)
    traversability=TraversabilityCfg(),
)
"""Unitree Go2: Isaac Lab's ``Isaac-Velocity-Rough-UnitreeGo2`` policy (600 + 900 iterations on Newton)."""

ROBOTS: dict[str, RobotProfile] = {p.key: p for p in (CUSTOM_QUAD, GO2)}
"""Robots by ``--robot`` key; the first one is the default."""
