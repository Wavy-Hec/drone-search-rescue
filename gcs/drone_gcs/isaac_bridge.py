"""Stream Isaac Lab 3.0 assets and sensors through a :class:`~drone_gcs.link.Sender`.

Standard library + numpy only and duck-typed: nothing here imports isaaclab, torch or warp, so it loads in
any Python. It reads the public Isaac Lab 3.0 data properties (``ProxyArray`` -> ``.torch`` -> numpy):

* ``robot.data.root_pos_w`` (N, 3), ``root_quat_w`` (N, 4) **(x, y, z, w)**, ``root_lin_vel_w``, ``root_ang_vel_w``
* ``ray_caster.data.ray_hits_w`` (N, B, 3) (``inf`` = miss), ``ray_caster.data.pos_w`` (N, 3)
* ``camera.data.output["rgb"]`` (N, H, W, 3) uint8, ``output["distance_to_image_plane"]`` (N, H, W, 1),
  ``camera.data.intrinsic_matrices`` (N, 3, 3), ``pos_w`` (N, 3), ``quat_w_ros`` (N, 4) (x, y, z, w)

Quaternions are converted to the wire order (w, x, y, z). ``origin`` (e.g. ``env.scene.env_origins[i]``) is
subtracted from every position so the GCS can work in the environment's local frame.

Example (inside the survey script, after ``env.step``)::

    from drone_gcs.link import Sender
    from drone_gcs.isaac_bridge import IsaacLabStreamer

    stream = IsaacLabStreamer(Sender(name="isaac-survey"), env_id=0)
    stream.mission(area=(-20, 20, -20, 20), resolution=0.25, planned_path=path_xyz, survey_altitude=8.0)
    ...
    stream.telemetry(sim_t, robot, wp_index=k, carrot=carrot_xyz)
    stream.scan(sim_t, scanner)
    if step % 3 == 0:
        stream.camera(sim_t, camera)
    ...
    stream.end()
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import quat_xyzw_to_wxyz
from .protocol import to_numpy

__all__ = ["IsaacLabStreamer", "row"]


def row(x: Any, i: int) -> np.ndarray:
    """Row ``i`` of a batched Isaac Lab quantity as numpy (works for ProxyArray, torch, warp, numpy)."""
    a = to_numpy(x)
    return np.asarray(a[i])


class IsaacLabStreamer:
    """Thin adapter: Isaac Lab objects -> ``Sender.send_*`` calls for one environment / drone."""

    def __init__(
        self,
        sender,
        env_id: int = 0,
        drone_id: int | None = None,
        origin: Any = None,
        jpeg: bool = False,
        rgb_key: str = "rgb",
        depth_key: str = "distance_to_image_plane",
    ):
        self.sender = sender
        self.env_id = int(env_id)
        self.drone_id = int(env_id if drone_id is None else drone_id)
        self.origin = np.zeros(3) if origin is None else np.asarray(to_numpy(origin), dtype=np.float64).reshape(3)
        self.jpeg = bool(jpeg)
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self._frame = 0

    def _p(self, p: Any) -> np.ndarray:
        return np.asarray(p, dtype=np.float64).reshape(3) - self.origin

    def mission(self, area, resolution, planned_path=None, survey_altitude=None, **kw) -> bool:
        """Positions given here must already be in the local frame (world minus ``origin``)."""
        return self.sender.send_mission(area, resolution, planned_path, survey_altitude, drone_id=self.drone_id, **kw)

    def telemetry(self, t: float, robot, wp_index: int | None = None, carrot: Any = None, **extra) -> bool:
        try:
            d, i = robot.data, self.env_id
            pos = self._p(row(d.root_pos_w, i))
            quat = quat_xyzw_to_wxyz(row(d.root_quat_w, i))
            vel = row(d.root_lin_vel_w, i)
            ang = row(d.root_ang_vel_w, i) if hasattr(d, "root_ang_vel_w") else None
            car = None if carrot is None else self._p(to_numpy(carrot))
            return self.sender.send_telemetry(
                t, pos, quat, vel, wp_index=wp_index, carrot=car, ang_vel=ang, drone_id=self.drone_id, **extra
            )
        except Exception as e:  # noqa: BLE001  (never raise into the sim loop)
            return self.sender._fail("IsaacLabStreamer.telemetry", e)

    def scan(self, t: float, ray_caster, **extra) -> bool:
        try:
            d, i = ray_caster.data, self.env_id
            hits = np.asarray(row(d.ray_hits_w, i), dtype=np.float32).reshape(-1, 3) - self.origin.astype(np.float32)
            return self.sender.send_scan(t, hits, origin=self._p(row(d.pos_w, i)), drone_id=self.drone_id, **extra)
        except Exception as e:  # noqa: BLE001
            return self.sender._fail("IsaacLabStreamer.scan", e)

    def camera(self, t: float, camera, frame_id: int | None = None, with_depth: bool = True, **extra) -> bool:
        try:
            d, i = camera.data, self.env_id
            out = d.output
            rgb = row(out[self.rgb_key], i)
            depth = None
            if with_depth and self.depth_key in out:
                depth = row(out[self.depth_key], i)
            K = row(d.intrinsic_matrices, i)
            pos = self._p(row(d.pos_w, i))
            quat = quat_xyzw_to_wxyz(row(d.quat_w_ros, i))
            if frame_id is None:
                frame_id = self._frame
            self._frame = frame_id + 1
            return self.sender.send_camera(
                t, rgb, K, pos, quat, depth=depth, jpeg=self.jpeg, frame_id=frame_id, drone_id=self.drone_id, **extra
            )
        except Exception as e:  # noqa: BLE001
            return self.sender._fail("IsaacLabStreamer.camera", e)

    def gt_objects(self, objects, **kw) -> bool:
        return self.sender.send_gt_objects(objects, **kw)

    def end(self, reason: str = "complete", t: float | None = None, **kw) -> bool:
        return self.sender.send_end(reason, t, **kw)
