"""Downward RGB-D camera of the SAR drone: intrinsics, mount, footprint, the pixel budget and projections.

Choice: 640 x 480 px, square pixels, focal length f = 480 px, principal point at the image centre
(cx, cy) = (320, 240), i.e. a 67.4 deg x 53.1 deg field of view, mounted 0.1 m below ``base_link`` and looking
straight down (nadir) with the top of the image towards the drone's nose (+X body).

Why (the math, for the default survey of ``DroneLab-Survey-ARL-Robot-1-v0``):

* The survey flies at a fixed height ``z_flight = z_max + clearance``: the highest terrain point inside the area
  plus its U-turn band is z_max = 3.515 m and the default clearance is 4 m, so z_flight = 7.515 m. Most of the site
  (block-cluster ground, border, landing pad) is at z = 0, so the camera (0.1 m below the body) is h = 7.4 m above
  the ground there. The extremes inside the area are the pit floor (z = -1.43 m, h = 8.85 m) and the top of the
  highest hill (z = 3.41 m, h = 4.0 m).
* Ground sample distance GSD = h / f: 7.4 / 480 = 1.54 cm/px over flat ground, 1.84 cm/px over the pit floor.
* A lying adult of the asset set is 1.61-1.80 m long with shoulders about 0.40-0.45 m wide (the T-pose arms span
  1.56-1.87 m). At h = 7.4 m: length 104-117 px, shoulder width 26-29 px. Worst case (pit floor, h = 8.85 m):
  length 87-98 px, shoulders 22-24 px. So the narrowest part of a lying person is >= 20 px everywhere in the default
  site; even the front-to-back depth of the thinnest character (0.35 m) is 22.7 px over flat ground.
* Footprint = (W, H) * h / f: 9.9 m (across track, image width) x 7.4 m (along track) at h = 7.4 m, and
  5.3 m x 4.0 m at the minimum height of 4.0 m. The lawnmower lines are 4 m apart, so the camera swaths overlap
  (by >= 1.3 m even over the highest hill) and every point is seen from at least two lines over flat ground.
* f = 400 px (77 deg) would widen the swath to 11.8 m but give 18.6 px for the thinnest character at h = 7.4 m;
  f = 480 px keeps every person comfortably above 20 px while still covering the area without gaps.

Isaac Lab parameterises a pinhole camera by ``focal_length`` and ``horizontal_aperture`` (same unit), so
f_px = focal_length / horizontal_aperture * width; with the default aperture 20.955 the focal length is
480 / 640 * 20.955 = 15.716. The vertical aperture follows from the aspect ratio (square pixels).

Mount: Isaac Lab's ROS convention (x right, y down, z forward) with the rotation R_body<-cam whose columns are the
camera axes in the body frame: x_cam = -Y_body, y_cam = -X_body, z_cam = -Z_body. That is a 180 deg rotation about
(1, -1, 0)/sqrt(2), quaternion (x, y, z, w) = (0.70711, -0.70711, 0, 0). Image right is the drone's right side,
image up is its nose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

HORIZONTAL_APERTURE = 20.955
"""Isaac Lab's default ``PinholeCameraCfg.horizontal_aperture``; only the ratio to the focal length matters."""


@dataclass(frozen=True)
class CameraSpec:
    """Intrinsics and mount of the downward camera."""

    width: int = 640
    height: int = 480
    focal_px: float = 480.0
    """Focal length in pixels (fx = fy)."""
    clip_near: float = 0.05
    clip_far: float = 200.0
    mount_pos: tuple[float, float, float] = (0.0, 0.0, -0.1)
    """Camera position in the ``base_link`` frame [m]."""
    mount_rot_ros: tuple[float, float, float, float] = (0.70710678, -0.70710678, 0.0, 0.0)
    """Camera orientation in the ``base_link`` frame, ROS convention, quaternion (x, y, z, w)."""

    @property
    def cx(self) -> float:
        """Principal point, x [px]."""
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        """Principal point, y [px]."""
        return 0.5 * self.height

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view [deg]."""
        return math.degrees(2.0 * math.atan(0.5 * self.width / self.focal_px))

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view [deg]."""
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.focal_px))

    @property
    def focal_length(self) -> float:
        """``PinholeCameraCfg.focal_length`` for :data:`HORIZONTAL_APERTURE`."""
        return self.focal_px / self.width * HORIZONTAL_APERTURE

    def intrinsic_matrix(self) -> list[list[float]]:
        """3x3 pinhole matrix K (pixel centres at integer + 0.5 coordinates, origin top-left)."""
        return [[self.focal_px, 0.0, self.cx], [0.0, self.focal_px, self.cy], [0.0, 0.0, 1.0]]

    def gsd(self, height_above_ground: float) -> float:
        """Ground sample distance [m/px] of a nadir view from ``height_above_ground`` [m]."""
        return float(height_above_ground) / self.focal_px

    def pixels(self, size_m: float, height_above_ground: float) -> float:
        """Image size [px] of a ground-level object of ``size_m`` seen from ``height_above_ground``."""
        return float(size_m) / self.gsd(height_above_ground)

    def footprint(self, height_above_ground: float) -> tuple[float, float]:
        """Ground footprint (across track = image width, along track = image height) [m]."""
        g = self.gsd(height_above_ground)
        return self.width * g, self.height * g

    def to_dict(self) -> dict:
        """Intrinsics, mount and conventions for the ground-truth JSON."""
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.focal_px,
            "fy": self.focal_px,
            "cx": self.cx,
            "cy": self.cy,
            "K": self.intrinsic_matrix(),
            "hfov_deg": round(self.hfov_deg, 3),
            "vfov_deg": round(self.vfov_deg, 3),
            "clipping_range": [self.clip_near, self.clip_far],
            "mount_pos_base_link": list(self.mount_pos),
            "mount_rot_base_link_ros_xyzw": list(self.mount_rot_ros),
            "convention": "ROS optical frame: x right, y down, z forward; image top = drone nose (+X body)",
            "depth": "distance_to_image_plane [m] (z in the camera frame)",
        }


SAR_CAMERA = CameraSpec()
"""The camera used by ``DroneLab-SAR-ARL-Robot-1-v0``."""


def quat_xyzw_to_matrix(q: ArrayLike) -> np.ndarray:
    """Rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def world_to_pixel(
    points_w: ArrayLike, cam_pos_w: ArrayLike, cam_quat_w_ros: ArrayLike, K: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points (N, 3) into the image of a camera pose in the ROS convention.

    Args:
        points_w: World points [m].
        cam_pos_w: Camera position (3,) (Isaac Lab ``camera.data.pos_w[i]``).
        cam_quat_w_ros: Camera orientation (x, y, z, w), ROS convention (``camera.data.quat_w_ros[i]``).
        K: 3x3 intrinsics (``camera.data.intrinsic_matrices[i]``).

    Returns:
        ``(uv (N, 2) pixel coordinates (u right, v down), depth (N,) distance to the image plane [m])``.
    """
    R = quat_xyzw_to_matrix(cam_quat_w_ros)
    pc = (np.asarray(points_w, dtype=float).reshape(-1, 3) - np.asarray(cam_pos_w, dtype=float)) @ R
    K = np.asarray(K, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
    return uv, pc[:, 2]


def pixel_to_world(
    uv: ArrayLike, depth: ArrayLike, cam_pos_w: ArrayLike, cam_quat_w_ros: ArrayLike, K: ArrayLike
) -> np.ndarray:
    """Back-project pixels with their ``distance_to_image_plane`` depth to world points (N, 3).

    This is how a detection becomes a map pin: take the box centre (or the median of its mask pixels) and the
    depth image there.
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    d = np.asarray(depth, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * d
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * d
    pc = np.column_stack([x, y, d])
    return pc @ quat_xyzw_to_matrix(cam_quat_w_ros).T + np.asarray(cam_pos_w, dtype=float)
