"""Search-and-rescue mission: flight plan, ground truth for the ground station, obstacle clearance (numpy only).

The mission of ``scripts/sar_mission.py``: take off from the landing pad, fly a lawnmower survey over the site,
come back and land on the pad, while the ground station (the ``drone_gcs`` project) builds its map live.

Flight plan math (defaults; :func:`survey_geometry` gives the numbers for any setting). The survey flies at
z = 7.515 m (highest terrain 3.515 m + 4 m clearance), so the downward camera (0.1 m under the body, 640 x 480 px,
f = 480 px) is h = 7.4 m above the flat ground: its footprint is W h / f = 9.9 m across track by H h / f = 7.4 m
along track, 1.54 cm per pixel.

* **Line spacing 5 m** (8 lines over 40 m): neighbouring camera swaths overlap by 4.9 m (49 %) over flat ground,
  and over the highest hilltop (camera 4.0 m above it) the swath is still 5.3 m wide, so the photo map has no
  gaps. The 5 m wide ray grid of the mapping scanner tiles the area exactly (the outer rays of neighbouring lines
  meet), and every ray row lies on the 0.25 m map lattice.
* **Speed 1.5 m/s** for the carrot; the policy flies about 1.25 m/s. At 10 frames/s consecutive frames are 0.13 m
  apart (98 % forward overlap), so every point stays in view for about 6 s per pass.
* **Length** 8 x 40 m of lines + 7 half-circle turns (r = 2.5 m) = 375 m, plus about 55 m of take-off, transit,
  return and landing: about 6 minutes of flight.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..paths import Area, cumulative_length, lawnmower

if TYPE_CHECKING:
    from .scenario import SceneObject

PHASES: tuple[str, ...] = ("takeoff", "transit", "survey", "return", "descent", "touchdown")
"""Mission phases in flight order."""

FLIGHT_PHASES: tuple[str, ...] = ("transit", "survey", "return")
"""Phases away from the landing pad (obstacle clearance is measured in these)."""


@dataclass(frozen=True)
class MissionPlan:
    """Waypoints of the whole mission and the arclength ranges of its phases."""

    waypoints: np.ndarray
    """(N, 3) path from the start hover above the pad to the touchdown point [m, world]."""
    phase_ends: tuple[float, ...]
    """Arclength at which each phase of :data:`PHASES` ends [m]."""
    speeds: tuple[float, ...]
    """Carrot speed of each phase [m/s]."""
    home: tuple[float, float, float]
    """Centre of the landing pad's top surface [m]."""
    area: tuple[float, float, float, float]
    """Survey rectangle (x_min, x_max, y_min, y_max) [m]."""
    flight_z: float
    """Survey flight height [m, world]."""
    spacing: float
    """Lawnmower line spacing [m]."""
    survey_waypoints: tuple[int, int]
    """Index range ``[first, last]`` of the lawnmower waypoints."""

    @property
    def length(self) -> float:
        """Length of the whole path [m]."""
        return float(self.phase_ends[-1])

    @property
    def start(self) -> np.ndarray:
        """Spawn point: the hover above the pad."""
        return self.waypoints[0].copy()

    def phase_index(self, s: float) -> int:
        """Index into :data:`PHASES` of the phase at carrot arclength ``s``."""
        return min(int(np.searchsorted(self.phase_ends, s, side="right")), len(PHASES) - 1)

    def phase(self, s: float) -> str:
        """Name of the phase at carrot arclength ``s``."""
        return PHASES[self.phase_index(s)]

    def speed(self, s: float) -> float:
        """Carrot speed of the phase at arclength ``s`` [m/s]."""
        return self.speeds[self.phase_index(s)]

    def waypoint_index(self, s: float) -> int:
        """Index of the waypoint the carrot is heading to at arclength ``s``."""
        cum = cumulative_length(self.waypoints)
        return min(int(np.searchsorted(cum, s + 1e-6, side="left")), len(self.waypoints) - 1)

    def to_dict(self) -> dict[str, Any]:
        """Plan summary for ``mission.json``: home, area, length and the range and speed of every phase."""
        starts = (0.0, *self.phase_ends[:-1])
        return {
            "home": [round(v, 3) for v in self.home],
            "area": list(self.area),
            "flight_z": round(self.flight_z, 3),
            "spacing": self.spacing,
            "length_m": round(self.length, 2),
            "waypoints": len(self.waypoints),
            "survey_waypoints": list(self.survey_waypoints),
            "phases": {
                name: {"start_m": round(a, 2), "end_m": round(b, 2), "speed_mps": v}
                for name, a, b, v in zip(PHASES, starts, self.phase_ends, self.speeds, strict=True)
            },
        }


def plan_mission(
    area: tuple[float, float, float, float],
    home: tuple[float, float, float],
    flight_z: float,
    spacing: float = 5.0,
    *,
    speed: float = 1.5,
    climb_speed: float = 1.0,
    descent_speed: float = 0.8,
    touchdown_speed: float = 0.3,
    hover_height: float = 0.5,
    touchdown_height: float = 0.05,
    direction: str = "x",
    turn: str = "round",
) -> MissionPlan:
    """Take off from the pad, fly a lawnmower over ``area``, come back and land.

    The drone starts hovering ``hover_height`` above the pad (the reset puts it there at rest), climbs straight up to
    ``flight_z``, flies to the first survey line, covers the area, flies back over the pad, descends to the hover
    height and then slowly towards ``touchdown_height`` above the pad, which it touches before getting there.

    Args:
        area: Survey rectangle (x_min, x_max, y_min, y_max) [m].
        home: Centre of the pad's top surface [m].
        flight_z: Survey flight height (world z) [m].
        spacing: Maximum distance between lawnmower lines [m].
        speed: Carrot speed for transit, survey and return [m/s].
        climb_speed, descent_speed, touchdown_speed: Vertical carrot speeds [m/s].
        hover_height: Start and landing hover above the pad [m].
        touchdown_height: Height of the last waypoint above the pad [m].
        direction, turn: Lawnmower line direction and turn style (see :func:`drone_lab.paths.lawnmower`).
    """
    hx, hy, hz = (float(v) for v in home)
    if flight_z <= hz + hover_height:
        raise ValueError(f"flight_z {flight_z} must be above the hover point {hz + hover_height}")
    survey = lawnmower(Area(*area), spacing, flight_z, direction=direction, turn=turn)
    hover = np.array([hx, hy, hz + hover_height])
    top = np.array([hx, hy, float(flight_z)])
    touchdown = np.array([hx, hy, hz + touchdown_height])
    waypoints = np.vstack([hover, top, survey, top, hover, touchdown])
    cum = cumulative_length(waypoints)
    first, last = 2, 1 + len(survey)
    # phase ends: top of the climb, first survey point, last survey point, back over the pad, hover, touchdown
    ends = tuple(float(cum[i]) for i in (1, first, last, last + 1, last + 2, last + 3))
    speeds = (climb_speed, speed, speed, speed, descent_speed, touchdown_speed)
    return MissionPlan(
        waypoints=waypoints,
        phase_ends=ends,
        speeds=tuple(float(v) for v in speeds),
        home=(hx, hy, hz),
        area=tuple(float(v) for v in area),
        flight_z=float(flight_z),
        spacing=float(spacing),
        survey_waypoints=(first, last),
    )


def survey_geometry(
    flight_z: float,
    spacing: float,
    speed: float,
    focal_px: float,
    image_size: tuple[int, int],
    ground_z: float = 0.0,
    min_clearance: float | None = None,
    camera_offset: float = 0.1,
    frame_rate: float = 10.0,
) -> dict[str, float]:
    """Camera footprint, ground sample distance and overlaps of a lawnmower survey (flat ground at ``ground_z``).

    ``min_clearance`` is the smallest camera height above the terrain (e.g. over the highest hill); the swath there
    must stay at least ``spacing`` wide for a gap-free photo map. Lines run along the drone's nose, so the image
    width is across track.
    """
    w, h = image_size
    height = flight_z - camera_offset - ground_z
    gsd = height / focal_px
    across, along = w * gsd, h * gsd
    step = speed / frame_rate
    out = {
        "camera_height_m": height,
        "gsd_cm_per_px": 100.0 * gsd,
        "footprint_across_m": across,
        "footprint_along_m": along,
        "side_overlap_m": across - spacing,
        "side_overlap_pct": 100.0 * (across - spacing) / across,
        "frame_spacing_m": step,
        "forward_overlap_pct": 100.0 * (1.0 - step / along),
        "seconds_in_view": along / max(speed, 1e-9),
    }
    if min_clearance is not None:
        out["min_swath_m"] = w * min_clearance / focal_px
    return out


def gt_objects(objects: Iterable[SceneObject]) -> list[dict[str, Any]]:
    """Labelled objects of a :class:`~drone_lab.sar.scenario.Scenario` in the ground station's ``GT_OBJECTS`` format.

    ``pos`` is the footprint centre on the ground, ``top_z`` the top of the object's box. Trees are unlabelled
    occluders and left out.
    """
    out = []
    for o in objects:
        if not o.labeled:
            continue
        out.append(
            {
                "id": o.name,
                "class": o.category,
                "pos": [round(float(o.center[0]), 3), round(float(o.center[1]), 3), round(float(o.ground_z), 3)],
                "size": [round(2.0 * float(v), 3) for v in o.half_extents],
                "yaw": round(float(o.yaw), 4),
                "pose": o.pose,
                "top_z": round(float(o.aabb_max[2]), 3),
            }
        )
    return out


class ClearanceMonitor:
    """Closest approach of the drone to the terrain (its own mapping ray hits) and to the site's objects (boxes).

    Only steps in ``phases`` count: take-off and landing come close to the pad on purpose.
    """

    def __init__(
        self,
        objects: Iterable[SceneObject],
        phases: tuple[str, ...] = FLIGHT_PHASES,
        skip_categories: tuple[str, ...] = ("landing_pad",),
    ):
        kept = [o for o in objects if o.category not in skip_categories]
        self.names = [o.name for o in kept]
        self.lo = np.array([o.aabb_min for o in kept], dtype=np.float64).reshape(-1, 3)
        self.hi = np.array([o.aabb_max for o in kept], dtype=np.float64).reshape(-1, 3)
        self.phases = tuple(phases)
        self.terrain: dict[str, Any] = {"distance_m": math.inf}
        self.object: dict[str, Any] = {"distance_m": math.inf}

    def update(self, t: float, pos: ArrayLike, phase: str, hits: ArrayLike | None = None) -> None:
        """Record one step: drone position ``pos`` (3,), mission ``phase`` and the scanner's ray hits (N, 3)."""
        if phase not in self.phases:
            return
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        if hits is not None:
            h = np.asarray(hits, dtype=np.float64).reshape(-1, 3)
            h = h[np.isfinite(h).all(axis=1)]
            if len(h):
                d = np.linalg.norm(h - p, axis=1)
                i = int(np.argmin(d))
                if d[i] < self.terrain["distance_m"]:
                    self.terrain = {"distance_m": float(d[i]), "t": float(t), "drone": p.tolist()}
        if len(self.lo):
            gap = np.maximum(np.maximum(self.lo - p, 0.0), p - self.hi)
            d = np.linalg.norm(gap, axis=1)
            i = int(np.argmin(d))
            if d[i] < self.object["distance_m"]:
                self.object = {"distance_m": float(d[i]), "t": float(t), "drone": p.tolist(), "name": self.names[i]}

    def summary(self) -> dict[str, Any]:
        """Closest approaches for ``mission.json`` (rounded; None where nothing was measured)."""

        def clean(d: dict) -> dict:
            out = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
            if "drone" in out:
                out["drone"] = [round(v, 3) for v in out["drone"]]
            if not math.isfinite(d["distance_m"]):
                out["distance_m"] = None
            return out

        terrain, obj = clean(self.terrain), clean(self.object)
        ds = [d["distance_m"] for d in (terrain, obj) if d["distance_m"] is not None]
        return {"closest_m": min(ds) if ds else None, "terrain": terrain, "objects": obj, "phases": list(self.phases)}


def import_drone_gcs(root: str | Path | None = None) -> ModuleType:
    """Import the ground station's ``drone_gcs`` package (its sender side needs only numpy).

    Search order: an installed ``drone_gcs``, then ``root``, ``$DRONE_GCS_ROOT``, and the ground-station repository
    next to this one (``drone_gcs`` or ``gcs``).
    """
    candidates = [Path(p).expanduser() for p in (root, os.environ.get("DRONE_GCS_ROOT")) if p]
    if not candidates:
        try:
            return importlib.import_module("drone_gcs")
        except ImportError:
            pass
    repo_parent = Path(__file__).resolve().parents[4]
    candidates += [repo_parent / "drone_gcs", repo_parent / "gcs"]
    for path in candidates:
        if (path / "drone_gcs" / "link.py").is_file():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return importlib.import_module("drone_gcs")
    raise ImportError(f"drone_gcs not found (looked in {', '.join(map(str, candidates))}); set DRONE_GCS_ROOT")


__all__ = [
    "FLIGHT_PHASES",
    "PHASES",
    "ClearanceMonitor",
    "MissionPlan",
    "gt_objects",
    "import_drone_gcs",
    "plan_mission",
    "survey_geometry",
]
