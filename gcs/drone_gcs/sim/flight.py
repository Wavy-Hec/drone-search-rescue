"""Survey path planning and a kinematic multirotor that follows a carrot sliding along the path.

The carrot logic mirrors the Isaac Lab survey task (``drone_lab/carrot.py``): the target advances along the
planned polyline at the cruise speed but never gets more than ``max_lead`` metres ahead of the drone.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import quat_from_euler
from ..mapping import area_tuple

G = 9.81


def lawnmower(
    area, spacing: float, z: float, direction: str = "x", margin: float | None = None, overshoot: float = 2.0
):
    """Boustrophedon waypoints (N, 3): lines along ``direction`` spaced ``spacing`` apart, ``margin`` from the
    area edges (default ``spacing / 2``), extended ``overshoot`` metres past the area on both ends."""
    x0, x1, y0, y1 = area_tuple(area)
    if direction == "y":
        pts = lawnmower((y0, y1, x0, x1), spacing, z, "x", margin, overshoot)
        return pts[:, [1, 0, 2]]
    margin = spacing / 2 if margin is None else margin
    lo, hi = y0 + margin, y1 - margin
    n = max(1, int(math.ceil((hi - lo) / spacing - 1e-9)) + 1)
    ys = np.linspace(lo, hi, n) if n > 1 else np.array([0.5 * (y0 + y1)])
    wps = []
    for i, y in enumerate(ys):
        xa, xb = (x0 - overshoot, x1 + overshoot) if i % 2 == 0 else (x1 + overshoot, x0 - overshoot)
        wps += [(xa, y, z), (xb, y, z)]
    return np.array(wps, dtype=np.float64)


class CarrotFlight:
    """Kinematic quadrotor: first-order velocity response with acceleration limit, yaw towards the direction of
    travel, roll/pitch from the commanded acceleration (for realistic camera tilt with a fixed mount)."""

    def __init__(
        self,
        path,
        speed: float = 3.0,
        max_lead: float = 2.5,
        accel: float = 2.5,
        tau: float = 0.5,
        kp: float = 0.8,
        yaw_rate_deg: float = 90.0,
        start=None,
    ):
        self.path = np.asarray(path, dtype=np.float64)
        if len(self.path) < 2:
            raise ValueError("path needs at least 2 waypoints")
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.speed, self.max_lead, self.accel, self.tau, self.kp = speed, max_lead, accel, tau, kp
        self.yaw_rate = math.radians(yaw_rate_deg)
        self.p = np.array(self.path[0] if start is None else start, dtype=np.float64)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        d0 = seg[0]
        self.yaw = math.atan2(d0[1], d0[0])
        self.roll = self.pitch = 0.0
        self.s = 0.0
        self.t = 0.0
        self.done = False

    def carrot(self, s: float | None = None) -> tuple[np.ndarray, int]:
        """Point at arc length ``s`` and the index of the segment it lies on."""
        s = self.s if s is None else s
        k = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1))
        f = (s - self.cum[k]) / max(self.seg_len[k], 1e-9)
        return self.path[k] + np.clip(f, 0, 1) * (self.path[k + 1] - self.path[k]), k

    def step(self, dt: float) -> None:
        c, k = self.carrot()
        if np.linalg.norm(c - self.p) < self.max_lead:
            self.s = min(self.length, self.s + self.speed * dt)
            c, k = self.carrot()
        tangent = (self.path[k + 1] - self.path[k]) / max(self.seg_len[k], 1e-9)
        ff = self.speed * tangent if self.s < self.length else np.zeros(3)
        v_des = self.kp * (c - self.p) + ff
        vmax = 1.3 * self.speed
        n = np.linalg.norm(v_des)
        if n > vmax:
            v_des *= vmax / n
        a = (v_des - self.v) / self.tau
        an = np.linalg.norm(a)
        if an > self.accel:
            a *= self.accel / an
        self.a = a
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        self.t += dt
        vxy = math.hypot(self.v[0], self.v[1])
        if vxy > 1.0:
            want = math.atan2(self.v[1], self.v[0])
            err = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += float(np.clip(err, -self.yaw_rate * dt, self.yaw_rate * dt))
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        a_fwd = a[0] * cy + a[1] * sy
        a_left = -a[0] * sy + a[1] * cy
        alpha = min(1.0, dt / 0.15)
        self.pitch += alpha * (math.atan2(a_fwd, G) - self.pitch)
        self.roll += alpha * (-math.atan2(a_left, G) - self.roll)
        if self.s >= self.length and np.linalg.norm(self.path[-1] - self.p) < 0.3 and np.linalg.norm(self.v) < 0.3:
            self.done = True

    @property
    def quat(self) -> np.ndarray:
        """Body (FLU) -> world, (w, x, y, z)."""
        return quat_from_euler(self.roll, self.pitch, self.yaw)

    @property
    def wp_index(self) -> int:
        """Index of the waypoint the drone is heading to (end of the carrot's segment)."""
        return min(len(self.path) - 1, self.carrot()[1] + 1)
