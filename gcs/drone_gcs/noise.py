"""Sensor noise injected into the link messages as they arrive (numpy only).

In the simulator the drone's pose, the depth image and the ray hits are exact. A real drone has a state estimate
with an error, a depth camera with noise and holes, and sensors that are not perfectly in step. This module adds
such errors to TELEMETRY, SCAN and CAMERA messages on the ground station, so that one recorded mission can be
replayed at many noise levels (``gcs.py --replay FILE --noise SPEC``, ``scripts/noise_sweep.py``).

The navigation error (position [m, world] and attitude [rad]) is one value per drone and time step: the drone has
one state estimate and registers its scan and its camera frame with it, so the messages of one drone with the same
``t`` get the same error. It is the sum of a slowly varying part (a first-order Gauss-Markov process: its value
after a time ``dt`` is ``exp(-dt / tau)`` times the old value plus fresh noise, so it drifts like a GPS or
odometry error) and an independent part per time step. The ground truth (GT_OBJECTS, GT_TERRAIN, ``gt_boxes``)
is never changed.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .geometry import quat_from_euler, quat_multiply
from .protocol import CAMERA, SCAN, TELEMETRY, Message

KEYS = {
    "pos": "pos_bias_m",
    "pos_white": "pos_white_m",
    "att": "att_bias_deg",
    "att_white": "att_white_deg",
    "tau": "tau_s",
    "depth": "depth_rel",
    "dropout": "depth_dropout",
    "scan": "scan_z_m",
    "lag": "time_offset_s",
    "seed": "seed",
}
"""Short names of the :class:`NoiseCfg` settings in a noise specification (see :func:`parse_noise`)."""


@dataclass
class NoiseCfg:
    """Noise levels; 0 (the default) switches a noise source off. Every ``std`` is one standard deviation."""

    seed: int = 0  # the same seed and the same messages in the same order give the same noise
    pos_bias_m: float = 0.0  # slowly varying position error, std per axis [m]
    pos_white_m: float = 0.0  # position error drawn anew at every time step, std per axis [m]
    att_bias_deg: float = 0.0  # slowly varying attitude error, std of each of roll, pitch and yaw [deg]
    att_white_deg: float = 0.0  # attitude error drawn anew at every time step, std per angle [deg]
    tau_s: float = 30.0  # correlation time of the slowly varying errors [s]
    depth_rel: float = 0.0  # depth error per pixel, std as a fraction of the depth
    depth_dropout: float = 0.0  # fraction of the depth pixels that are lost (set to inf)
    scan_z_m: float = 0.0  # height error per ray hit, std [m]
    time_offset_s: float = 0.0  # the camera frame is paired with the pose of this much later [s]

    def __post_init__(self) -> None:
        stds = (self.pos_bias_m, self.pos_white_m, self.att_bias_deg, self.att_white_deg, self.depth_rel, self.scan_z_m)
        if min(stds) < 0 or not 0 <= self.depth_dropout <= 1 or self.tau_s <= 0 or self.seed < 0:
            raise ValueError("noise levels and the seed must be 0 or more, dropout between 0 and 1 and tau above 0")

    @property
    def active(self) -> bool:
        """Any noise source is switched on."""
        return any(value != 0 for name, value in self.to_dict().items() if name not in ("seed", "tau_s"))

    def to_dict(self) -> dict:
        """The settings as a plain dict (for ``summary.json``)."""
        return dataclasses.asdict(self)

    def spec(self) -> str:
        """The settings that differ from the defaults in the :func:`parse_noise` format, e.g. ``"pos=0.5,seed=1"``
        (for log lines: 6 significant digits)."""
        default = NoiseCfg().to_dict()
        mine = self.to_dict()
        return ",".join(f"{key}={mine[name]:g}" for key, name in KEYS.items() if mine[name] != default[name])


def parse_noise(spec: str) -> NoiseCfg:
    """``"pos=0.5,att=1,depth=0.02,dropout=0.1,scan=0.02,lag=0.05,tau=30,pos_white=0.05,att_white=0.2,seed=1"``
    -> :class:`NoiseCfg` (any subset of the keys, in any order; see :data:`KEYS`). Raises ValueError for an unknown
    key or a value that is not a number."""
    values: dict = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in KEYS:
            raise ValueError(f"unknown noise setting {key!r} (valid: {', '.join(KEYS)})")
        try:
            values[KEYS[key]] = int(value) if key == "seed" else float(value)
        except ValueError:
            raise ValueError(f"noise setting {item.strip()!r}: write {key}=NUMBER") from None
    return NoiseCfg(**values)


def _vec3(value: list | None) -> np.ndarray:
    """A JSON vector as an array; zeros when the message does not carry it."""
    return np.zeros(3) if value is None else np.asarray(value, dtype=np.float64).reshape(3)


def _move_pose(fields: dict, shift: np.ndarray, turn: np.ndarray) -> None:
    """Replace ``pos`` and ``quat`` in ``fields`` by the pose moved by ``shift`` [m, world] and turned by the small
    angles ``turn`` = (roll, pitch, yaw) [rad] about the world axes."""
    fields["pos"] = (np.asarray(fields["pos"], dtype=np.float64).reshape(3) + shift).tolist()
    fields["quat"] = quat_multiply(quat_from_euler(*turn), fields["quat"]).tolist()  # world rotation: on the left


class _DroneNoise:
    """The navigation error and the random streams of one drone."""

    def __init__(self, cfg: NoiseCfg, drone_id: int):
        self.cfg = cfg
        # Two streams, so that switching the pixel noise on or off leaves the navigation error as it was.
        self.nav_rng = np.random.default_rng([cfg.seed, drone_id, 0])
        self.sensor_rng = np.random.default_rng([cfg.seed, drone_id, 1])
        self.t: float | None = None  # the time step the error below belongs to
        self.pos_bias = np.zeros(3)  # slowly varying parts [m] and [rad]
        self.att_bias = np.zeros(3)
        self.pos_err = np.zeros(3)  # the error of time step t: slowly varying part + per-step part
        self.att_err = np.zeros(3)
        self.vel = np.zeros(3)  # true velocity [m/s] and angular velocity [rad/s] of the last telemetry
        self.ang_vel = np.zeros(3)

    def advance(self, t: float | None) -> None:
        """Draw the navigation error of time step ``t`` [s] (nothing to do while ``t`` stays the same)."""
        if t is None or t == self.t:
            return
        cfg = self.cfg
        # Gauss-Markov step: keep the fraction `keep` of the old value and add noise that restores the variance.
        # The first step has no old value. abs(): a sim clock that restarts (new mission) is a time gap like any.
        keep = 0.0 if self.t is None else math.exp(-abs(t - self.t) / cfg.tau_s)
        fresh = math.sqrt(1.0 - keep * keep)
        n = self.nav_rng.standard_normal((4, 3))
        self.pos_bias = keep * self.pos_bias + fresh * cfg.pos_bias_m * n[0]
        self.att_bias = keep * self.att_bias + fresh * math.radians(cfg.att_bias_deg) * n[1]
        self.pos_err = self.pos_bias + cfg.pos_white_m * n[2]
        self.att_err = self.att_bias + math.radians(cfg.att_white_deg) * n[3]
        self.t = t

    def noisy_points(self, points: np.ndarray) -> np.ndarray:
        """A copy of the ray hits (..., 3) with the finite ones shifted by the position error and their height
        disturbed; misses (inf / NaN) stay as they are."""
        out = np.array(points)
        rows = out.reshape(-1, 3)  # a view of the copy
        hit = np.isfinite(rows).all(axis=1)
        rows[hit] += self.pos_err
        if self.cfg.scan_z_m:
            rows[hit, 2] += self.cfg.scan_z_m * self.sensor_rng.standard_normal(int(hit.sum()))
        return out

    def noisy_depth(self, depth: np.ndarray) -> np.ndarray:
        """A copy of the depth image with an error proportional to the depth and with lost pixels (inf)."""
        out = np.array(depth)
        if self.cfg.depth_rel:
            out *= 1.0 + self.cfg.depth_rel * self.sensor_rng.standard_normal(out.shape)
        if self.cfg.depth_dropout:
            out[self.sensor_rng.random(out.shape) < self.cfg.depth_dropout] = np.inf
        return out


class NoiseInjector:
    """Adds the noise of a :class:`NoiseCfg` to the messages of a link, one drone at a time.

    The messages of one time step of a drone must arrive one after the other (they do: the simulator sends them
    in that order and the link keeps the order): a message with another ``t`` starts the next time step.
    """

    def __init__(self, cfg: NoiseCfg):
        self.cfg = cfg
        self._drones: dict[int, _DroneNoise] = {}

    def apply(self, msg: Message) -> Message:
        """``msg`` with the noise applied. The input is never changed: what changes is copied, the rest is shared.

        Messages other than TELEMETRY, SCAN and CAMERA, and all messages when every level is 0, are returned as
        they are. So is a message that does not have the fields of its type: the station reports it when it
        applies it."""
        if not self.cfg.active or msg.type not in (TELEMETRY, SCAN, CAMERA):
            return msg
        try:
            return self._apply(msg)
        except (KeyError, TypeError, ValueError):
            return msg

    def _apply(self, msg: Message) -> Message:
        cfg = self.cfg
        drone = self._drones.get(msg.drone_id)
        if drone is None:
            drone = self._drones[msg.drone_id] = _DroneNoise(cfg, msg.drone_id)
        drone.advance(msg.t)
        fields, arrays = dict(msg.fields), dict(msg.arrays)
        if msg.type == TELEMETRY:
            drone.vel = _vec3(fields.get("vel"))  # the true rates: the time offset of the camera needs them
            drone.ang_vel = _vec3(fields.get("ang_vel"))
            _move_pose(fields, drone.pos_err, drone.att_err)
        elif msg.type == SCAN:  # the scanner is gimballed: an attitude error does not move its points
            if arrays.get("points") is not None:
                arrays["points"] = drone.noisy_points(arrays["points"])
            if fields.get("origin") is not None:
                fields["origin"] = (_vec3(fields["origin"]) + drone.pos_err).tolist()
        else:  # CAMERA: paired with the pose the drone has time_offset_s later
            shift = drone.pos_err + cfg.time_offset_s * drone.vel
            turn = drone.att_err + cfg.time_offset_s * drone.ang_vel
            _move_pose(fields, shift, turn)
            if arrays.get("depth") is not None and (cfg.depth_rel or cfg.depth_dropout):
                arrays["depth"] = drone.noisy_depth(arrays["depth"])
        return dataclasses.replace(msg, fields=fields, arrays=arrays)
