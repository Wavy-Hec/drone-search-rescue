"""Air-ground teaming: the drone's map guides a ground robot to a person (the rescue finale).

After the survey flight, a quadruped (the custom one of quadruped_lab on Isaac Sim PhysX, or a Unitree Go2 on
Newton) gets the drone's elevation map and the pins of what it found, and walks to a person over the rubble:

* :mod:`.traversability` -- slope / step / roughness costs from the elevation grid, unknown cells, object
  footprints, obstacle cells, inflation by the robot radius -> a cost map.
* :mod:`.planner` -- cost-weighted 8-connected A* with start/goal snapping and path smoothing.
* :mod:`.tour` -- one route past several people: the cheapest visiting order and the route of every leg.
* :mod:`.follower` -- pure pursuit -> body-frame ``(vx, vy, yaw_rate)`` clipped to the locomotion policy's
  command ranges.
* :mod:`.inputs` -- drone maps (drone_lab / drone_gcs runs), pins, terrain ground truth, and the obstacles the
  detector did not pin (the drone_gcs surface map against the elevation map).
* :mod:`.robots` -- the ground robots: task, policy, physics and planner limits of each.
* :mod:`.plots` -- plan, tour and walked-path figures (matplotlib).

Everything here is CPU-only (numpy + scipy; matplotlib for the figures). The simulator side lives in
``drone_lab.tasks.rescue`` (tasks ``DroneLab-Rescue-CustomQuad-v0`` and ``DroneLab-Rescue-Go2-v0``) and
``scripts/rescue_finale.py``.
"""

from .follower import FollowerCfg, FollowerOutput, PurePursuitFollower, VelocityLimits
from .planner import PlanResult, astar, plan_path
from .tour import TourResult, plan_tour
from .traversability import Footprint, TraversabilityCfg, TraversabilityMap, compute_traversability

__all__ = [
    "Footprint",
    "FollowerCfg",
    "FollowerOutput",
    "PlanResult",
    "PurePursuitFollower",
    "TourResult",
    "TraversabilityCfg",
    "TraversabilityMap",
    "VelocityLimits",
    "astar",
    "compute_traversability",
    "plan_path",
    "plan_tour",
]
