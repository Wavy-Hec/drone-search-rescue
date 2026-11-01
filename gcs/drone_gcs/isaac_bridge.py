"""Stream Isaac Lab 3.0 assets and sensors through a :class:`~drone_gcs.link.Sender`.

Standard library + numpy only and duck-typed: nothing here imports isaaclab, torch or warp, so it loads in
any Python. It reads the public Isaac Lab 3.0 data properties (``ProxyArray`` -> ``.torch`` -> numpy):

* ``robot.data.root_pos_w`` (N, 3), ``root_quat_w`` (N, 4) **(x, y, z, w)**, ``root_lin_vel_w``, ``root_ang_vel_w``
* ``ray_caster.data.ray_hits_w`` (N, B, 3) (``inf`` = miss), ``ray_caster.data.pos_w`` (N, 3)
* ``camera.data.output["rgb"]`` (N, H, W, 3) uint8, ``output["distance_to_image_plane"]`` (N, H, W, 1),
  ``camera.data.intrinsic_matrices`` (N, 3, 3), ``pos_w`` (N, 3), ``quat_w_ros`` (N, 4) (x, y, z, w),
  ``camera.frame`` (N,) capture counter

Quaternions are converted to the wire order (w, x, y, z). Only row ``env_id`` is copied off the GPU.

Frames: ``origin`` (e.g. ``env.scene.env_origins[i]``) is subtracted from every position the streamer reads
from Isaac Lab, so the GCS works in the environment's local frame. Positions *you* pass (``mission`` area and
planned path, ``gt_objects``, ``carrot``) follow one rule: give them in the local frame, or pass
``world=True`` and the streamer subtracts ``origin`` for you (``carrot`` is always world, like the sim data).

Camera: ``camera()`` sends a frame only when the camera has a new capture (``camera.frame`` advanced), so a
``CameraCfg.update_period`` longer than the env step does not repeat images (repeats would count as extra
sightings in the GCS tracker). With OVRTX ``async_rendering`` the image is one capture old; its pose and
intrinsics are then taken from ``camera.data.info[<rgb key>]["capture"]``, not from the live ``pos_w``.

Example (inside the survey script, after ``env.step``)::

    from drone_gcs.link import Sender
    from drone_gcs.isaac_bridge import IsaacLabStreamer

    stream = IsaacLabStreamer(Sender(name="isaac-survey"), env_id=0)
    stream.mission(area=(-20, 20, -20, 20), resolution=0.25, planned_path=path_xyz, survey_altitude=8.0)
    ...
    stream.telemetry(sim_t, robot, wp_index=k, carrot=carrot_xyz)
    stream.scan(sim_t, scanner)
    stream.camera(sim_t, camera)  # every step is fine: only new captures are sent
    ...
    stream.end()
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import camera_quat_to_ros, quat_xyzw_to_wxyz
from .protocol import is_proxy_array, to_numpy

__all__ = ["IsaacLabStreamer", "row"]


def row(x: Any, i: int) -> np.ndarray:
    """Row ``i`` of a batched Isaac Lab quantity as numpy (ProxyArray, torch, warp, numpy).

    torch / numpy are indexed *before* the conversion, so only that row leaves the GPU."""
    if is_proxy_array(x):
        x = x.torch
    if isinstance(x, np.ndarray) or (hasattr(x, "detach") and hasattr(x, "cpu")):
        return np.asarray(to_numpy(x[i]))
    return np.asarray(to_numpy(x)[i])


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
        self._last_capture: int | None = None

    def _p(self, p: Any) -> np.ndarray:
        return np.asarray(p, dtype=np.float64).reshape(3) - self.origin

    def _local_xy(self, pts: Any) -> np.ndarray:
        """(N, 2|3) world points -> local (origin subtracted from the available columns)."""
        a = np.array(to_numpy(pts), dtype=np.float64, copy=True)
        a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
        return a - self.origin[: a.shape[1]]

    def mission(self, area, resolution, planned_path=None, survey_altitude=None, *, world: bool = False, **kw) -> bool:
        """Announce the survey. ``area`` ``(x_min, x_max, y_min, y_max)``, ``planned_path`` (N, 2|3) and
        ``survey_altitude`` are in the local frame, or in world coordinates with ``world=True``."""
        try:
            if world:
                if isinstance(area, dict):
                    a = [area[k] for k in ("x_min", "x_max", "y_min", "y_max")]
                elif hasattr(area, "x_min"):
                    a = [area.x_min, area.x_max, area.y_min, area.y_max]
                else:
                    a = list(np.asarray(to_numpy(area), dtype=np.float64).reshape(4))
                ox, oy, oz = self.origin
                area = (float(a[0]) - ox, float(a[1]) - ox, float(a[2]) - oy, float(a[3]) - oy)
                if planned_path is not None:
                    planned_path = self._local_xy(planned_path)
                if survey_altitude is not None:
                    survey_altitude = float(survey_altitude) - oz
                for k in ("ground_z",):
                    if kw.get(k) is not None:
                        kw[k] = float(kw[k]) - oz
                if kw.get("z_range") is not None:
                    kw["z_range"] = [float(v) - oz for v in np.asarray(to_numpy(kw["z_range"])).reshape(2)]
        except Exception as e:  # noqa: BLE001  (never raise into the sim loop)
            return self.sender._fail("IsaacLabStreamer.mission", e)
        return self.sender.send_mission(area, resolution, planned_path, survey_altitude, drone_id=self.drone_id, **kw)

    def telemetry(self, t: float, robot, wp_index: int | None = None, carrot: Any = None, **extra) -> bool:
        """Root state of ``robot`` for env ``env_id``. ``carrot`` (the tracking target) is in world
        coordinates, like the robot state."""
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

    def camera(
        self, t: float, camera, frame_id: int | None = None, with_depth: bool = True, force: bool = False, **extra
    ) -> bool:
        """Send env ``env_id``'s current image if it is a new capture. Returns ``False`` when nothing new was
        captured since the last call (``force=True`` sends anyway) or on error.

        ``frame_id`` defaults to Isaac Lab's capture counter (``camera.frame``), else a call counter."""
        try:
            d, i = camera.data, self.env_id  # reading .data first lets Isaac Lab update an outdated camera
            info = getattr(d, "info", None) or {}
            cap = info.get(self.rgb_key) if isinstance(info, dict) else None
            cap = cap.get("capture") if isinstance(cap, dict) else None
            if not isinstance(cap, dict) or not cap:
                cap = None
            if frame_id is None:
                src = cap.get("frame") if cap is not None else getattr(camera, "frame", None)
                if src is not None:
                    frame_id = int(np.asarray(row(src, i)).reshape(-1)[0])
                    if frame_id == self._last_capture and not force:
                        return False  # same capture as last time (update_period longer than the step)
                else:
                    frame_id = self._frame
            self._last_capture = int(frame_id)
            self._frame = int(frame_id) + 1
            out = d.output
            rgb = row(out[self.rgb_key], i)
            depth = None
            if with_depth and self.depth_key in out:
                depth = row(out[self.depth_key], i)
            if cap is not None and all(k in cap for k in ("pos_w", "quat_w_world", "intrinsic_matrices")):
                # async rendering: the pixels belong to the previous capture's pose and intrinsics
                K = row(cap["intrinsic_matrices"], i)
                pos = self._p(row(cap["pos_w"], i))
                quat = camera_quat_to_ros(quat_xyzw_to_wxyz(row(cap["quat_w_world"], i)), "world")
            else:
                K = row(d.intrinsic_matrices, i)
                pos = self._p(row(d.pos_w, i))
                quat = quat_xyzw_to_wxyz(row(d.quat_w_ros, i))
            return self.sender.send_camera(
                t, rgb, K, pos, quat, depth=depth, jpeg=self.jpeg, frame_id=frame_id, drone_id=self.drone_id, **extra
            )
        except Exception as e:  # noqa: BLE001
            return self.sender._fail("IsaacLabStreamer.camera", e)

    def gt_objects(self, objects, *, world: bool = False, **kw) -> bool:
        """Ground truth for evaluation: ``[{"class", "pos": [x, y, z]}, ...]`` in the local frame, or in world
        coordinates with ``world=True`` (e.g. straight from ``scene[name].data.root_pos_w``)."""
        if world:
            try:
                objs = []
                for o in objects:
                    if isinstance(o, dict):
                        d = dict(o)
                        d["pos"] = self._p(to_numpy(d["pos"])).tolist()
                    else:
                        d = (o[0], self._p(to_numpy(o[1])).tolist())
                    objs.append(d)
                objects = objs
            except Exception as e:  # noqa: BLE001
                return self.sender._fail("IsaacLabStreamer.gt_objects", e)
        return self.sender.send_gt_objects(objects, drone_id=self.drone_id, **kw)

    def end(self, reason: str = "complete", t: float | None = None, **kw) -> bool:
        return self.sender.send_end(reason, t, **kw)
