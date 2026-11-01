"""Rotations, camera conventions and the pinhole camera model (numpy only).

Conventions (see PROTOCOL.md):

* quaternions are **(w, x, y, z)**, Hamilton, unit norm; ``q`` rotates frame-local vectors into the world:
  ``v_world = R(q) @ v_local``. Isaac Lab 3.0 stores (x, y, z, w): use :func:`quat_xyzw_to_wxyz`.
* world frame: Isaac Sim stage frame, Z up, treated as local ENU (+X east, +Y north).
* body frame: FLU (x forward, y left, z up).
* camera frame: ROS optical (x right, y down, z forward / into the scene).
* pixels: continuous image coordinates; pixel index ``(u, v)`` covers ``[u, u+1) x [v, v+1)`` so its centre
  is ``(u + 0.5, v + 0.5)`` (Isaac Lab's K has ``cx = W/2``). Use ``pixel_center=0`` for OpenCV-style K.
"""

from __future__ import annotations

import math

import numpy as np

# Rotation from Isaac Lab's "world" camera convention (x forward, y left, z up) to the ROS optical convention:
# R_world_ros = R_world_worldconv @ C_WORLDCONV_TO_ROS. Columns = ROS optical axes expressed in the "world"
# convention camera frame: x_right = -y, y_down = -z, z_fwd = +x.
C_WORLDCONV_TO_ROS = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
Q_WORLDCONV_TO_ROS = np.array([0.5, -0.5, 0.5, -0.5])  # (w, x, y, z) of C_WORLDCONV_TO_ROS
# OpenGL / USD camera (x right, y up, z backward) -> ROS optical: rotate 180 deg about x.
C_OPENGL_TO_ROS = np.diag([1.0, -1.0, -1.0])
Q_OPENGL_TO_ROS = np.array([0.0, 1.0, 0.0, 0.0])
# Body FLU -> downward camera looking along -Z_body with the image top towards the nose:
# x_right = -y_body, y_down = -x_body, z_fwd = -z_body.
C_BODY_TO_NADIR_CAM = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


# ---------------------------------------------------------------------------------------------------- quaternions
def quat_xyzw_to_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[..., [1, 2, 3, 0]]


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n > 0, n, 1.0)


def quat_to_matrix(q) -> np.ndarray:
    """(w, x, y, z) -> rotation matrix (..., 3, 3)."""
    w, x, y, z = np.moveaxis(quat_normalize(q), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def matrix_to_quat(R) -> np.ndarray:
    """Rotation matrix (3, 3) -> (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = quat_normalize(np.array(q))
    return -q if q[0] < 0 else q


def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product a * b, both (w, x, y, z)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) Euler angles [rad] -> (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def euler_from_quat(q) -> tuple[float, float, float]:
    """(w, x, y, z) -> ZYX ``(roll, pitch, yaw)`` [rad]."""
    w, x, y, z = quat_normalize(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_from_quat(q) -> float:
    """Yaw [rad] of a body->world quaternion: angle of the body x axis from world +X, counter-clockwise."""
    return euler_from_quat(q)[2]


def compass_heading_deg(yaw: float) -> float:
    """ENU yaw [rad] (0 = east, CCW) -> compass heading [deg] (0 = north, clockwise)."""
    return (90.0 - math.degrees(yaw)) % 360.0


def camera_quat_to_ros(q_wxyz, convention: str = "ros") -> np.ndarray:
    """Convert a camera orientation (w, x, y, z) given in Isaac Lab's ``"world"`` / ``"opengl"`` / ``"ros"``
    camera convention into the ROS optical convention used on the wire."""
    q = quat_normalize(q_wxyz)
    if convention == "ros":
        return q
    if convention == "world":
        return quat_multiply(q, Q_WORLDCONV_TO_ROS)
    if convention == "opengl":
        return quat_multiply(q, Q_OPENGL_TO_ROS)
    raise ValueError(f"unknown camera convention {convention!r}")


# ---------------------------------------------------------------------------------------------------- camera
def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Square-pixel K for a horizontal field of view, principal point at the image centre (W/2, H/2)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


class PinholeCamera:
    """Pinhole camera with pose ``R_wc`` (ROS optical -> world) and optical centre ``t_wc`` [m, world]."""

    def __init__(self, K, width: int, height: int, R_wc, t_wc, pixel_center: float = 0.5):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)
        self.width, self.height = int(width), int(height)
        self.R = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t_wc, dtype=np.float64).reshape(3)
        self.pixel_center = float(pixel_center)

    @classmethod
    def from_pose(cls, K, width, height, pos, quat_wxyz, pixel_center: float = 0.5) -> PinholeCamera:
        return cls(K, width, height, quat_to_matrix(quat_wxyz), pos, pixel_center)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def quat(self) -> np.ndarray:
        return matrix_to_quat(self.R)

    @property
    def optical_axis(self) -> np.ndarray:
        """Viewing direction in the world (unit)."""
        return self.R[:, 2].copy()

    def nadir_angle_deg(self) -> float:
        """Angle between the optical axis and straight down."""
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(self.R[2, 2])))))

    def rays(self, u, v) -> np.ndarray:
        """World directions (N, 3) through pixel indices (u, v), scaled so that camera-frame z = 1.

        ``origin + d * ray`` is the point at depth (distance to the image plane) ``d``.
        """
        u = np.asarray(u, dtype=np.float64).reshape(-1) + self.pixel_center
        v = np.asarray(v, dtype=np.float64).reshape(-1) + self.pixel_center
        pix = np.stack([u, v, np.ones_like(u)], 0)
        return (self.R @ (self.K_inv @ pix)).T

    def backproject(self, u, v, depth) -> np.ndarray:
        """World points (N, 3) of pixels (u, v) with distance-to-image-plane ``depth`` [m]."""
        d = np.asarray(depth, dtype=np.float64).reshape(-1, 1)
        return self.t + d * self.rays(u, v)

    def project(self, points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points (N, 3) -> pixel-index coordinates ``(u, v)`` and depth z.

        Pixel centres sit at integer values (``round(u)`` is the pixel index), the inverse of :meth:`rays`.
        """
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pc = (P - self.t) @ self.R  # = R^T (P - t)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (self.K @ (pc.T / z)).T
        return uv[:, 0] - self.pixel_center, uv[:, 1] - self.pixel_center, z

    def corner_pixels(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel indices of the four image corners (top-left, top-right, bottom-right, bottom-left)."""
        w, h = self.width - 1, self.height - 1
        return np.array([0, w, w, 0], float), np.array([0, 0, h, h], float)
