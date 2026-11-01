"""Pure-pursuit path follower that turns a planned path into body-frame velocity commands (numpy only).

The locomotion policy (Isaac Lab's Go2 rough-terrain velocity task) takes a command ``(vx, vy, yaw_rate)`` in its
base frame and was trained with every component in [-1, 1] (``UniformVelocityCommandCfg.Ranges``), so every
output here is clipped to those ranges (:class:`VelocityLimits`).

Per step, given the robot pose ``(x, y, yaw)``:

1. **Progress:** the closest point of the path is searched only in a window around the previous progress
   (``search_back_m`` behind, ``search_ahead_m`` ahead), so the follower never skips ahead to a later part of
   the path that happens to pass nearby; progress never decreases.
2. **Look-ahead:** the target point lies ``lookahead_m`` further along the path (clamped to the end). In the body
   frame it is at angle ``alpha`` and distance ``L``; pure pursuit steers along the arc through it, curvature
   ``kappa = 2 sin(alpha) / L``.
3. **Speed:** cruise speed, reduced when the target is off to the side (``cos(alpha)``), on tight arcs (so the
   yaw rate ``v * kappa`` stays within its limit), and near the goal (``goal_slowdown_m``). When ``|alpha|`` is
   larger than ``turn_in_place_rad`` the robot turns on the spot first.
4. **Lateral:** an optional small sideways velocity pulls the robot back onto the path (``lateral_gain``).
5. **Goal:** within ``goal_tolerance_m`` of the last point the command becomes zero (stand still) and
   :attr:`FollowerOutput.done` is set; it stays done.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["FollowerCfg", "FollowerOutput", "PurePursuitFollower", "VelocityLimits", "wrap_to_pi"]


def wrap_to_pi(a):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class VelocityLimits:
    """Command ranges of the locomotion policy (Isaac Lab velocity task: all three are (-1, 1))."""

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    def clip(self, vx: float, vy: float, wz: float) -> tuple[float, float, float]:
        return (
            float(np.clip(vx, *self.lin_vel_x)),
            float(np.clip(vy, *self.lin_vel_y)),
            float(np.clip(wz, *self.ang_vel_z)),
        )

    @classmethod
    def from_ranges(cls, ranges) -> VelocityLimits:
        """From an object with ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z`` attributes (e.g. Isaac Lab's Ranges)."""
        return cls(tuple(ranges.lin_vel_x), tuple(ranges.lin_vel_y), tuple(ranges.ang_vel_z))


@dataclass
class FollowerCfg:
    """Pure-pursuit settings (defaults tuned for a Go2-sized robot walking over rough ground)."""

    cruise_speed: float = 0.7
    """Forward speed on straight, open stretches [m/s]."""
    min_speed: float = 0.3
    """Lowest forward speed while still driving (not turning in place) [m/s]. Locomotion policies trained with a
    lenient velocity-tracking reward tend to stand still for commands below ~0.2 m/s (the Go2 does), so the final
    approach keeps at least this speed until the goal tolerance is reached."""
    lookahead_m: float = 0.8
    """Distance of the pursuit point ahead of the progress point [m]."""
    max_yaw_rate: float = 0.9
    """Yaw-rate limit used to slow down on tight arcs [rad/s] (kept below the policy's 1.0)."""
    turn_in_place_rad: float = 1.0
    """Heading error to the pursuit point above which the robot stops and turns on the spot [rad]."""
    turn_exit_rad: float = 0.5
    """Heading error below which a turn on the spot ends and the robot drives again [rad]."""
    turn_gain: float = 1.2
    """Yaw rate per rad of heading error when turning on the spot [1/s]."""
    lateral_gain: float = 0.6
    """Sideways velocity per metre of cross-track error [1/s] (0 disables)."""
    max_lateral_speed: float = 0.25
    """Limit of the sideways correction [m/s]."""
    goal_tolerance_m: float = 0.35
    """Distance to the last path point at which the robot stops and the follower reports done [m]."""
    goal_slowdown_m: float = 1.2
    """Distance to the goal over which the speed ramps down linearly [m]."""
    search_back_m: float = 0.5
    """How far behind the current progress the closest-point search looks [m]."""
    search_ahead_m: float = 2.5
    """How far ahead of the current progress the closest-point search looks [m]."""
    limits: VelocityLimits = field(default_factory=VelocityLimits)
    """Final clip to the policy's command ranges."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FollowerOutput:
    """One follower step."""

    vx: float
    vy: float
    wz: float
    done: bool
    progress_m: float
    """Arclength of the closest path point [m]."""
    cross_track_m: float
    """Distance from the robot to the path (signed: + when the path is to the robot's left) [m]."""
    heading_error_rad: float
    """Angle of the pursuit point in the body frame [rad]."""
    distance_to_goal_m: float
    target_xy: tuple[float, float]
    """Pursuit point [m, world]."""
    mode: str
    """"track", "turn" (turning on the spot), or "goal"."""


class PurePursuitFollower:
    """Follows a 2D polyline with pure pursuit; see the module docstring."""

    def __init__(self, path_xy: np.ndarray, cfg: FollowerCfg | None = None):
        self.cfg = cfg or FollowerCfg()
        p = np.asarray(path_xy, dtype=float).reshape(-1, 2)
        if len(p) == 0:
            raise ValueError("empty path")
        if len(p) > 1:  # drop repeated points (zero-length segments)
            keep = np.concatenate([[True], np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9])
            p = p[keep]
        self.path = p
        seg = np.diff(p, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])
        self.reset()

    # ----------------------------------------------------------------------------------------------- state
    def reset(self, progress_m: float = 0.0) -> None:
        """Restart at ``progress_m`` along the path (e.g. after a respawn) and clear the done flag."""
        self.progress = float(np.clip(progress_m, 0.0, self.length))
        self.done = False
        self.turning = False

    def point_at(self, s: float) -> np.ndarray:
        """World point at arclength ``s`` (clamped to the path)."""
        if len(self.path) == 1:
            return self.path[0].copy()
        s = float(np.clip(s, 0.0, self.length))
        return np.array([np.interp(s, self.s, self.path[:, 0]), np.interp(s, self.s, self.path[:, 1])])

    def tangent_at(self, s: float) -> float:
        """Path direction [rad] at arclength ``s``."""
        if len(self.path) == 1:
            return 0.0
        k = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_len) - 1))
        d = self.path[k + 1] - self.path[k]
        return math.atan2(d[1], d[0])

    def closest(self, xy: np.ndarray, s_lo: float = 0.0, s_hi: float | None = None) -> tuple[float, float]:
        """Closest path point to ``xy`` among arclengths [s_lo, s_hi]: ``(arclength, distance)``."""
        xy = np.asarray(xy, dtype=float)
        if len(self.path) == 1:
            return 0.0, float(np.linalg.norm(xy - self.path[0]))
        s_hi = self.length if s_hi is None else s_hi
        k0 = int(np.clip(np.searchsorted(self.s, s_lo, side="right") - 1, 0, len(self.seg_len) - 1))
        k1 = int(np.clip(np.searchsorted(self.s, s_hi, side="left"), k0 + 1, len(self.seg_len)))
        a = self.path[k0:k1]
        d = self.path[k0 + 1 : k1 + 1] - a
        L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-18)
        t = np.clip(np.einsum("ij,ij->i", xy - a, d) / L2, 0.0, 1.0)
        proj = a + t[:, None] * d
        dist = np.linalg.norm(proj - xy, axis=1)
        s_cand = self.s[k0:k1] + t * self.seg_len[k0:k1]
        ok = (s_cand >= s_lo - 1e-9) & (s_cand <= s_hi + 1e-9)
        if not np.any(ok):
            ok[:] = True
        dist_ok = np.where(ok, dist, np.inf)
        i = int(np.argmin(dist_ok))
        return float(np.clip(s_cand[i], s_lo, s_hi)), float(dist_ok[i])

    # ------------------------------------------------------------------------------------------------ step
    def compute(self, x: float, y: float, yaw: float) -> FollowerOutput:
        """Velocity command for the robot at ``(x, y)`` with heading ``yaw`` [rad] (world frame)."""
        c = self.cfg
        pos = np.array([float(x), float(y)])
        goal = self.path[-1]
        dist_goal = float(np.linalg.norm(goal - pos))
        # progress: closest point in a window around the previous progress, never backwards
        s_new, _ = self.closest(pos, max(0.0, self.progress - c.search_back_m), self.progress + c.search_ahead_m)
        self.progress = max(self.progress, s_new)
        near = self.point_at(self.progress)
        tangent = self.tangent_at(self.progress)
        # signed cross-track error: (closest point - robot) along the path's left normal, i.e. + when the path is
        # to the robot's left
        off = near - pos
        cross_track = float(-math.sin(tangent) * off[0] + math.cos(tangent) * off[1])
        near_end = self.length - self.progress <= c.goal_tolerance_m + c.lookahead_m
        if self.done or (dist_goal <= c.goal_tolerance_m and near_end):
            self.done = True
            return FollowerOutput(0.0, 0.0, 0.0, True, self.progress, cross_track, 0.0, dist_goal,
                                  (float(goal[0]), float(goal[1])), "goal")
        target = self.point_at(self.progress + c.lookahead_m)
        rel = target - pos
        cy, sy = math.cos(yaw), math.sin(yaw)
        lx = cy * rel[0] + sy * rel[1]
        ly = -sy * rel[0] + cy * rel[1]
        L = max(math.hypot(lx, ly), 1e-6)
        alpha = math.atan2(ly, lx)
        # turn on the spot above turn_in_place_rad, resume driving below turn_exit_rad (hysteresis)
        self.turning = abs(alpha) > (c.turn_exit_rad if self.turning else c.turn_in_place_rad)
        if self.turning:
            wz = c.turn_gain * alpha
            vx, vy = 0.0, 0.0
            mode = "turn"
        else:
            kappa = 2.0 * math.sin(alpha) / L
            v = c.cruise_speed * max(math.cos(alpha), 0.0)
            if abs(kappa) > 1e-6:
                v = min(v, c.max_yaw_rate / abs(kappa))
            # ramp down over the last metres (distance to go along the path, or straight-line if smaller)
            to_go = min(dist_goal, max(self.length - self.progress, 0.0))
            if c.goal_slowdown_m > 0:
                v = min(v, c.cruise_speed * to_go / c.goal_slowdown_m)
            v = max(v, c.min_speed)
            vx = v
            wz = v * kappa
            # sideways pull towards the closest path point, in the body frame
            off_b_y = -sy * off[0] + cy * off[1]
            vy = float(np.clip(c.lateral_gain * off_b_y, -c.max_lateral_speed, c.max_lateral_speed))
            mode = "track"
        vx, vy, wz = c.limits.clip(vx, vy, wz)
        return FollowerOutput(vx, vy, wz, False, self.progress, cross_track, alpha, dist_goal,
                              (float(target[0]), float(target[1])), mode)
