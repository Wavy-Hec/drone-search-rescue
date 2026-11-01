"""Local link between the simulator (``Sender``) and the GCS (``Receiver``). Standard library + numpy only.

The GCS listens; the simulator connects (and keeps reconnecting). Addresses:

* ``unix:/path/to.sock`` or an absolute path: UNIX domain socket (default
  ``$XDG_RUNTIME_DIR/drone_gcs.sock``, else ``/tmp/drone_gcs-<uid>.sock``)
* ``tcp://127.0.0.1:47800``, ``127.0.0.1:47800`` or ``:47800``: TCP
* environment variable ``DRONE_GCS_ADDR`` overrides the default on both ends.

``Sender`` is built for a simulation loop: every ``send_*`` call validates and copies its arrays, puts them in
a bounded queue and returns at once. A background thread encodes and writes them. When the GCS is slow or
absent the oldest data messages are dropped (CAMERA keeps only the 2 newest by default); MISSION / GT_OBJECTS /
END are never dropped, and MISSION / GT_OBJECTS are re-sent after every reconnect. ``send_*`` never raises:
it returns ``False`` and logs (rate limited) instead.
"""

from __future__ import annotations

import collections
import itertools
import logging
import os
import socket
import stat
import sys
import threading
import time
from typing import Any

import numpy as np

from .protocol import (
    CAMERA,
    CONTROL_TYPES,
    END,
    GT_OBJECTS,
    HELLO,
    LATCHED_TYPES,
    MISSION,
    PREFIX_SIZE,
    PROTOCOL_VERSION,
    SCAN,
    TELEMETRY,
    Message,
    ProtocolError,
    decode_message,
    encode_frame,
    parse_prefix,
    to_numpy,
)

__all__ = ["Sender", "Receiver", "BoundedMessageQueue", "default_address", "parse_address", "DEFAULT_TCP_PORT"]

log = logging.getLogger("drone_gcs.link")

DEFAULT_TCP_PORT = 47800
ENV_ADDR = "DRONE_GCS_ADDR"
_UNIX_PATH_MAX = 107
DROP_PRIORITY = (CAMERA, SCAN, TELEMETRY)  # what goes first when the byte budget is exceeded


def default_address() -> str:
    """``$DRONE_GCS_ADDR``, else a UNIX socket in ``$XDG_RUNTIME_DIR`` (or ``/tmp``)."""
    env = os.environ.get(ENV_ADDR)
    if env:
        return env
    run = os.environ.get("XDG_RUNTIME_DIR")
    if run and os.path.isdir(run) and os.access(run, os.W_OK):
        return "unix:" + os.path.join(run, "drone_gcs.sock")
    return f"unix:/tmp/drone_gcs-{os.getuid()}.sock"


def parse_address(address: str) -> tuple[int, Any, str]:
    """``(family, sockaddr, canonical string)`` for a link address."""
    a = str(address).strip()
    if a.startswith("unix:"):
        path = a[5:]
        if path.startswith("//"):
            path = path[2:]
    elif a.startswith(("/", "./", "~")) or a.endswith(".sock"):
        path = a
    else:
        if a.startswith("tcp://"):
            a = a[6:]
        host, sep, port = a.rpartition(":")
        if not sep:
            raise ValueError(f"cannot parse link address {address!r} (use unix:/path or tcp://host:port)")
        host = host.strip("[]") or "127.0.0.1"
        try:
            port_i = int(port)
        except ValueError as e:
            raise ValueError(f"bad port in link address {address!r}") from e
        return socket.AF_INET6 if ":" in host else socket.AF_INET, (host, port_i), f"tcp://{host}:{port_i}"
    path = os.path.abspath(os.path.expanduser(path))
    if len(path.encode()) > _UNIX_PATH_MAX:
        raise ValueError(f"UNIX socket path longer than {_UNIX_PATH_MAX} bytes: {path}")
    return socket.AF_UNIX, path, f"unix:{path}"


class _RateLimitedLog:
    """Log a given key at most once per ``interval`` seconds (and count the suppressed ones)."""

    def __init__(self, logger: logging.Logger, interval: float = 10.0):
        self.logger, self.interval = logger, interval
        self._last: dict[str, float] = {}
        self._suppressed: collections.Counter = collections.Counter()

    def __call__(self, level: int, key: str, msg: str, *args) -> None:
        now = time.monotonic()
        if now - self._last.get(key, -1e9) < self.interval:
            self._suppressed[key] += 1
            return
        n = self._suppressed.pop(key, 0)
        self._last[key] = now
        try:
            self.logger.log(level, msg + (f" (+{n} similar suppressed)" if n else ""), *args)
        except Exception:  # never let logging break the caller
            pass


# ---------------------------------------------------------------------------------------------------- queue
class BoundedMessageQueue:
    """FIFO with a size limit per message type and an optional total byte budget.

    Items need ``.type`` and ``.nbytes`` attributes. When a type exceeds its limit its *oldest* item is dropped;
    types in ``never_drop`` are unbounded. ``pop`` returns items in global insertion order.
    """

    def __init__(
        self,
        limits: dict[str, int] | None = None,
        default_limit: int = 1024,
        max_bytes: int | None = None,
        never_drop=CONTROL_TYPES,
    ):
        self.limits = dict(limits or {})
        self.default_limit = int(default_limit)
        self.max_bytes = max_bytes
        self.never_drop = frozenset(never_drop)
        self._qs: dict[str, collections.deque] = {}
        self._cond = threading.Condition()
        self._order = itertools.count()
        self._n = 0
        self._bytes = 0
        self.dropped: collections.Counter = collections.Counter()

    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return self._bytes

    def _drop_oldest(self, q: collections.deque, mtype: str) -> None:
        _, old = q.popleft()
        self._n -= 1
        self._bytes -= int(getattr(old, "nbytes", 0) or 0)
        self.dropped[mtype] += 1

    def put(self, item) -> int:
        """Append ``item``; returns how many older items were dropped to make room."""
        mtype = item.type
        dropped = 0
        with self._cond:
            q = self._qs.setdefault(mtype, collections.deque())
            q.append((next(self._order), item))
            self._n += 1
            self._bytes += int(getattr(item, "nbytes", 0) or 0)
            if mtype not in self.never_drop:
                limit = max(1, self.limits.get(mtype, self.default_limit))
                while len(q) > limit:
                    self._drop_oldest(q, mtype)
                    dropped += 1
            if self.max_bytes is not None:
                while self._bytes > self.max_bytes:
                    victim = next((t for t in DROP_PRIORITY if self._qs.get(t)), None)
                    if victim is None:
                        victim = next(
                            (t for t, qq in self._qs.items() if qq and t not in self.never_drop and t != mtype), None
                        )
                    if victim is None or (victim == mtype and len(q) <= 1):
                        break
                    self._drop_oldest(self._qs[victim], victim)
                    dropped += 1
            self._cond.notify_all()
        return dropped

    def push_front(self, item) -> None:
        """Put ``item`` back at the head (used to retry a control message after a failed send)."""
        with self._cond:
            q = self._qs.setdefault(item.type, collections.deque())
            q.appendleft((-1, item))
            self._n += 1
            self._bytes += int(getattr(item, "nbytes", 0) or 0)
            self._cond.notify_all()

    def _pop_locked(self):
        best_t, best = None, None
        for t, q in self._qs.items():
            if q and (best is None or q[0][0] < best):
                best_t, best = t, q[0][0]
        if best_t is None:
            return None
        _, item = self._qs[best_t].popleft()
        self._n -= 1
        self._bytes -= int(getattr(item, "nbytes", 0) or 0)
        return item

    def pop(self, timeout: float | None = None):
        """Oldest item, waiting up to ``timeout`` seconds; ``None`` if still empty."""
        with self._cond:
            if self._n == 0 and timeout != 0:
                self._cond.wait(timeout)
            return self._pop_locked() if self._n else None

    def drain(self, max_items: int | None = None) -> list:
        """Pop everything (or up to ``max_items``) without waiting."""
        out = []
        with self._cond:
            while self._n and (max_items is None or len(out) < max_items):
                out.append(self._pop_locked())
        return out

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def clear(self) -> None:
        with self._cond:
            self._qs.clear()
            self._n = 0
            self._bytes = 0


class _Item:
    __slots__ = ("type", "fields", "arrays", "drone_id", "encodings", "jpeg_quality", "nbytes")

    def __init__(self, mtype, fields, arrays, drone_id, encodings=None, jpeg_quality=90):
        self.type = mtype
        self.fields = fields
        self.arrays = arrays or {}
        self.drone_id = int(drone_id)
        self.encodings = encodings or {}
        self.jpeg_quality = jpeg_quality
        self.nbytes = sum(int(a.nbytes) for a in self.arrays.values()) + 256


# ---------------------------------------------------------------------------------------------------- helpers
def _vec(x: Any, n: int, name: str) -> list[float]:
    a = np.asarray(to_numpy(x), dtype=np.float64).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} must have {n} values, got shape {np.shape(to_numpy(x))}")
    return [float(v) for v in a]


def _owned(x: Any, dtype=None) -> np.ndarray:
    """A C-contiguous copy (the simulator may reuse its buffers after the call returns)."""
    return np.array(to_numpy(x), dtype=dtype, copy=True, order="C")


def _area_dict(area: Any) -> dict:
    if isinstance(area, dict):
        d = {k: float(area[k]) for k in ("x_min", "x_max", "y_min", "y_max")}
    elif all(hasattr(area, k) for k in ("x_min", "x_max", "y_min", "y_max")):  # e.g. drone_lab.paths.Area
        d = {k: float(getattr(area, k)) for k in ("x_min", "x_max", "y_min", "y_max")}
    else:
        v = _vec(area, 4, "area")
        d = dict(zip(("x_min", "x_max", "y_min", "y_max"), v))
    if not (d["x_max"] > d["x_min"] and d["y_max"] > d["y_min"]):
        raise ValueError(f"degenerate area {d}")
    return d


# ---------------------------------------------------------------------------------------------------- sender
class Sender:
    """Simulation-side endpoint: non-blocking, bounded, auto-reconnecting. ``send_*`` never raises.

    Args:
        address: link address (default :func:`default_address`).
        name: sender name shown by the GCS.
        drone_id: default ``drone_id`` for messages (override per call).
        camera_queue / scan_queue / telemetry_queue: max queued messages per type (oldest dropped first).
        max_queue_bytes: total byte budget of the queue.
        connect_timeout / send_timeout: socket timeouts [s] (used only by the background thread).
        jpeg_quality: default JPEG quality for ``send_camera(..., jpeg=True)``.
    """

    def __init__(
        self,
        address: str | None = None,
        *,
        name: str = "sim",
        drone_id: int = 0,
        camera_queue: int = 2,
        scan_queue: int = 256,
        telemetry_queue: int = 512,
        max_queue_bytes: int = 256 << 20,
        connect_timeout: float = 1.0,
        send_timeout: float = 5.0,
        reconnect_min: float = 0.2,
        reconnect_max: float = 2.0,
        jpeg_quality: int = 90,
        sndbuf: int = 8 << 20,
        autostart: bool = True,
    ):
        self.address = address or default_address()
        self._family, self._sockaddr, self.address_str = parse_address(self.address)
        self.name = str(name)
        self.drone_id = int(drone_id)
        self.jpeg_quality = int(jpeg_quality)
        self._connect_timeout = float(connect_timeout)
        self._send_timeout = float(send_timeout)
        self._reconnect_min = float(reconnect_min)
        self._reconnect_max = float(reconnect_max)
        self._sndbuf = int(sndbuf)
        self._queue = BoundedMessageQueue(
            {CAMERA: camera_queue, SCAN: scan_queue, TELEMETRY: telemetry_queue},
            default_limit=256,
            max_bytes=max_queue_bytes,
        )
        self._latched: dict[tuple[str, int], tuple[int, _Item]] = {}
        self._latched_dirty: set[tuple[str, int]] = set()
        self._latched_order = itertools.count()
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._in_flight = False
        self._closed = False
        self._seq = itertools.count()
        self._t0 = time.time()
        self._rlog = _RateLimitedLog(log)
        self._stats = collections.Counter()
        self.last_error: str = ""
        if autostart:
            self.start()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="drone_gcs-sender", daemon=True)
            self._thread.start()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def wait_connected(self, timeout: float = 5.0) -> bool:
        """Block until connected (or ``timeout``). For scripts that want the GCS up before flying."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connected:
                return True
            time.sleep(0.02)
        return self.connected

    def idle(self) -> bool:
        """Connected and nothing left to send."""
        return self.connected and len(self._queue) == 0 and not self._in_flight and not self._latched_dirty

    def flush(self, timeout: float = 2.0) -> bool:
        """Wait until everything queued has been written (or ``timeout``). Never raises."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self.idle():
                return True
            time.sleep(0.005)
        return self.idle()

    def close(self, flush_timeout: float = 1.0) -> None:
        """Flush (bounded by ``flush_timeout``), stop the thread and close the socket."""
        if self._closed:
            return
        try:
            if flush_timeout and flush_timeout > 0:
                self.flush(flush_timeout)
        finally:
            self._closed = True
            self._stop.set()
            self._queue.wake()
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=2.0)
            self._disconnect(quiet=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def stats(self) -> dict:
        """Counters: sent / dropped messages, bytes, connects, queue length, connected flag."""
        s = dict(self._stats)
        s["dropped"] = int(sum(self._queue.dropped.values()))
        s["dropped_by_type"] = dict(self._queue.dropped)
        s["queued"] = len(self._queue)
        s["queued_bytes"] = self._queue.nbytes
        s["connected"] = self.connected
        s["address"] = self.address_str
        return s

    # ------------------------------------------------------------------ public API (never raises)
    def send(
        self,
        msg_type: str,
        fields: dict | None = None,
        arrays: dict | None = None,
        *,
        drone_id: int | None = None,
        encodings: dict | None = None,
        jpeg_quality: int | None = None,
    ) -> bool:
        """Queue a message of any type. Arrays are copied now; encoding happens in the background thread."""
        try:
            arrs = {k: _owned(v) for k, v in (arrays or {}).items() if v is not None}
            return self._submit(msg_type, fields, arrs, drone_id, encodings, jpeg_quality)
        except Exception as e:  # noqa: BLE001  (the sim loop must never see an exception)
            return self._fail(f"send({msg_type})", e)

    def _submit(self, msg_type, fields, arrays, drone_id=None, encodings=None, jpeg_quality=None) -> bool:
        """Queue already-copied arrays (internal: the typed ``send_*`` methods copy once themselves)."""
        item = _Item(
            str(msg_type),
            dict(fields or {}),
            arrays,
            self.drone_id if drone_id is None else drone_id,
            encodings,
            self.jpeg_quality if jpeg_quality is None else jpeg_quality,
        )
        return self._enqueue(item)

    def send_mission(
        self,
        area: Any,
        resolution: float,
        planned_path: Any = None,
        survey_altitude: float | None = None,
        *,
        mission_id: str | None = None,
        name: str | None = None,
        ground_z: float | None = None,
        z_range: Any = None,
        padding: float | None = None,
        photo_resolution: float | None = None,
        geo_origin: dict | None = None,
        drone_id: int | None = None,
        **extra,
    ) -> bool:
        """Announce the survey: area ``(x_min, x_max, y_min, y_max)`` [m], map cell size [m], planned path
        (N,2|3) [m, world], survey altitude (world z [m]). Latched: re-sent on every reconnect."""
        try:
            f: dict[str, Any] = {
                "area": _area_dict(area),
                "resolution": float(resolution),
                "mission_id": mission_id or f"{name or 'mission'}-{int(time.time() * 1000)}",
                "name": name or "survey",
            }
            if f["resolution"] <= 0:
                raise ValueError("resolution must be > 0")
            arrays = {}
            if planned_path is not None:
                p = np.asarray(to_numpy(planned_path), dtype=np.float64)
                if p.ndim != 2 or p.shape[1] not in (2, 3):
                    raise ValueError(f"planned_path must be (N, 2) or (N, 3), got {p.shape}")
                if p.shape[1] == 2:
                    z = float(survey_altitude) if survey_altitude is not None else 0.0
                    p = np.column_stack([p, np.full(len(p), z)])
                arrays["planned_path"] = p
            for k, v in (
                ("survey_altitude", survey_altitude),
                ("ground_z", ground_z),
                ("padding", padding),
                ("photo_resolution", photo_resolution),
            ):
                if v is not None:
                    f[k] = float(v)
            if z_range is not None:
                f["z_range"] = _vec(z_range, 2, "z_range")
            if geo_origin is not None:
                f["geo_origin"] = dict(geo_origin)
            f.update(extra)
            return self._submit(MISSION, f, arrays, drone_id)
        except Exception as e:  # noqa: BLE001
            return self._fail("send_mission", e)

    def send_telemetry(
        self,
        t: float,
        pos: Any,
        quat: Any,
        vel: Any = None,
        *,
        wp_index: int | None = None,
        carrot: Any = None,
        ang_vel: Any = None,
        drone_id: int | None = None,
        **extra,
    ) -> bool:
        """Vehicle state: sim time [s], position [m, world], orientation quaternion **(w, x, y, z)** body->world,
        linear velocity [m/s, world], index of the current target waypoint, carrot target [m, world]."""
        try:
            f: dict[str, Any] = {"t": float(t), "pos": _vec(pos, 3, "pos"), "quat": _vec(quat, 4, "quat")}
            if vel is not None:
                f["vel"] = _vec(vel, 3, "vel")
            if ang_vel is not None:
                f["ang_vel"] = _vec(ang_vel, 3, "ang_vel")
            if wp_index is not None:
                f["wp_index"] = int(wp_index)
            if carrot is not None:
                f["carrot"] = _vec(carrot, 3, "carrot")
            f.update(extra)
            return self._submit(TELEMETRY, f, None, drone_id)
        except Exception as e:  # noqa: BLE001
            return self._fail("send_telemetry", e)

    def send_scan(
        self,
        t: float,
        points: Any,
        valid: Any = None,
        origin: Any = None,
        *,
        drone_id: int | None = None,
        **extra,
    ) -> bool:
        """Mapping ray hits: (N, 3) points [m, world] (inf/NaN = miss), optional (N,) valid mask, sensor origin."""
        try:
            p = _owned(points, np.float32).reshape(-1, 3)
            arrays: dict[str, np.ndarray] = {"points": p}
            if valid is not None:
                v = _owned(valid, bool).reshape(-1)
                if v.size != len(p):
                    raise ValueError(f"valid has {v.size} entries for {len(p)} points")
                arrays["valid"] = v
            f: dict[str, Any] = {"t": float(t)}
            if origin is not None:
                f["origin"] = _vec(origin, 3, "origin")
            f.update(extra)
            return self._submit(SCAN, f, arrays, drone_id)
        except Exception as e:  # noqa: BLE001
            return self._fail("send_scan", e)

    def send_camera(
        self,
        t: float,
        rgb: Any,
        K: Any,
        pos: Any,
        quat: Any,
        depth: Any = None,
        *,
        jpeg: bool = False,
        jpeg_quality: int | None = None,
        frame_id: int | None = None,
        camera: str = "cam0",
        pixel_center: float = 0.5,
        drone_id: int | None = None,
        **extra,
    ) -> bool:
        """Camera frame: RGB uint8 (H, W, 3), intrinsics K (3x3), optical-centre position [m, world] and
        orientation **(w, x, y, z)** of the ROS optical frame (x right, y down, z forward) -> world,
        optional depth float32 (H, W) = distance to the image plane [m]."""
        try:
            img = to_numpy(rgb)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            if img.dtype != np.uint8:
                f_img = np.asarray(img, dtype=np.float32)
                img = np.clip(f_img * (255.0 if f_img.max(initial=0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            img = _owned(img, np.uint8)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"rgb must be (H, W, 3), got {img.shape}")
            h, w = img.shape[:2]
            k = np.asarray(to_numpy(K), dtype=np.float64).reshape(3, 3)
            arrays: dict[str, np.ndarray] = {"rgb": img}
            if depth is not None:
                d = _owned(depth, np.float32)
                if d.ndim == 3 and d.shape[-1] == 1:
                    d = d[..., 0]
                if d.shape != (h, w):
                    raise ValueError(f"depth shape {d.shape} != image shape {(h, w)}")
                arrays["depth"] = np.ascontiguousarray(d)
            f: dict[str, Any] = {
                "t": float(t),
                "K": k.tolist(),
                "width": int(w),
                "height": int(h),
                "pos": _vec(pos, 3, "pos"),
                "quat": _vec(quat, 4, "quat"),
                "camera": str(camera),
                "convention": "ros_optical",
                "pixel_center": float(pixel_center),
            }
            if frame_id is not None:
                f["frame_id"] = int(frame_id)
            f.update(extra)
            return self._submit(CAMERA, f, arrays, drone_id, {"rgb": "jpeg"} if jpeg else None, jpeg_quality)
        except Exception as e:  # noqa: BLE001
            return self._fail("send_camera", e)

    def send_gt_objects(self, objects: Any, t: float | None = None, *, drone_id: int | None = None) -> bool:
        """Ground-truth objects for evaluation only: list of ``{"class", "pos": [x, y, z], "id"?, "size"?}``
        (or ``(class, (x, y, z))`` tuples). Latched."""
        try:
            objs = []
            for i, o in enumerate(objects):
                if isinstance(o, dict):
                    d = dict(o)
                    d["class"] = str(d.get("class", d.get("cls", "object")))
                    d.pop("cls", None)
                    d["pos"] = _vec(d["pos"], 3, "pos")
                else:
                    cls, pos = o[0], o[1]
                    d = {"class": str(cls), "pos": _vec(pos, 3, "pos")}
                d.setdefault("id", i)
                objs.append(d)
            f: dict[str, Any] = {"objects": objs}
            if t is not None:
                f["t"] = float(t)
            return self._submit(GT_OBJECTS, f, None, drone_id)
        except Exception as e:  # noqa: BLE001
            return self._fail("send_gt_objects", e)

    def send_end(
        self, reason: str = "complete", t: float | None = None, *, flush_timeout: float = 2.0, **extra
    ) -> bool:
        """End of mission. Waits up to ``flush_timeout`` s for the queue to drain (0 = don't wait)."""
        try:
            f: dict[str, Any] = {"reason": str(reason)}
            if t is not None:
                f["t"] = float(t)
            f.update(extra)
            ok = self._submit(END, f, None)
            if ok and flush_timeout and flush_timeout > 0:
                self.flush(flush_timeout)
            return ok
        except Exception as e:  # noqa: BLE001
            return self._fail("send_end", e)

    # ------------------------------------------------------------------ internals
    def _fail(self, what: str, e: Exception) -> bool:
        self._stats["send_errors"] += 1
        self.last_error = f"{what}: {type(e).__name__}: {e}"
        self._rlog(logging.WARNING, what, "drone_gcs.link: %s", self.last_error)
        return False

    def _enqueue(self, item: _Item) -> bool:
        if self._closed:
            return False
        if item.type in LATCHED_TYPES:
            key = (item.type, item.drone_id)
            with self._lock:
                self._latched[key] = (next(self._latched_order), item)
                self._latched_dirty.add(key)
            self._queue.wake()
            return True
        self._queue.put(item)
        return True

    def _next_dirty_latched(self) -> _Item | None:
        with self._lock:
            if not self._latched_dirty:
                return None
            key = min(self._latched_dirty, key=lambda k: self._latched[k][0])
            self._latched_dirty.discard(key)
            return self._latched[key][1]

    def _connect(self) -> bool:
        s = None
        try:
            if self._family == socket.AF_UNIX:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(self._connect_timeout)
                s.connect(self._sockaddr)
            else:
                s = socket.create_connection(self._sockaddr, timeout=self._connect_timeout)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._sndbuf)
            except OSError:
                pass
            s.settimeout(self._send_timeout)
            self._sock = s
            self._stats["connects"] += 1
            self._rlog(logging.INFO, "connected", "drone_gcs.link: connected to GCS at %s", self.address_str)
            return True
        except OSError as e:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
            self.last_error = f"connect: {e}"
            self._rlog(
                logging.INFO, "connect", "drone_gcs.link: GCS not reachable at %s (%s); retrying", self.address_str, e
            )
            return False

    def _disconnect(self, quiet: bool = False) -> None:
        s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass
            if not quiet:
                self._stats["disconnects"] += 1
                self._rlog(logging.INFO, "disconnect", "drone_gcs.link: lost the GCS link (%s)", self.last_error)

    def _write(self, item: _Item) -> None:
        """Encode + write one item. Encoding errors drop the item (ProtocolError); socket errors raise OSError."""
        chunks = encode_frame(
            item.type,
            item.fields,
            item.arrays,
            seq=next(self._seq),
            drone_id=item.drone_id,
            sender=self.name,
            wall_time=time.time(),
            encodings=item.encodings,
            jpeg_quality=item.jpeg_quality,
        )
        sock = self._sock
        if sock is None:
            raise OSError("not connected")
        flags = getattr(socket, "MSG_NOSIGNAL", 0)  # no SIGPIPE inside a host app that does not ignore it
        n = 0
        for c in chunks:
            if len(c):
                sock.sendall(c, flags)
                n += len(c)
        self._stats["sent"] += 1
        self._stats["sent_bytes"] += n
        self._stats[f"sent_{item.type}"] += 1

    def _handshake(self) -> None:
        hello = _Item(
            HELLO,
            {
                "name": self.name,
                "protocol": PROTOCOL_VERSION,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "started": self._t0,
            },
            None,
            self.drone_id,
        )
        self._write(hello)
        with self._lock:
            latched = sorted(self._latched.values(), key=lambda p: p[0])
            self._latched_dirty.clear()
        for _, item in latched:
            self._write(item)

    def _run(self) -> None:
        backoff = self._reconnect_min
        while not self._stop.is_set():
            item = None
            try:
                if self._sock is None:
                    if not self._connect():
                        self._stop.wait(backoff)
                        backoff = min(self._reconnect_max, backoff * 1.6)
                        continue
                    backoff = self._reconnect_min
                    self._handshake()
                item = self._next_dirty_latched()
                if item is None:
                    item = self._queue.pop(timeout=0.1)
                    if item is None:
                        continue
                    latched = self._next_dirty_latched()  # a MISSION set while we waited goes first
                    if latched is not None:
                        self._queue.push_front(item)
                        item = latched
                self._in_flight = True
                self._write(item)
            except (ProtocolError, TypeError, ValueError) as e:
                self._fail(f"encode {getattr(item, 'type', '?')}", e)  # bad data: drop it, keep the link
            except OSError as e:
                self.last_error = f"send: {e}"
                if item is not None and item.type in CONTROL_TYPES and item.type not in LATCHED_TYPES:
                    self._queue.push_front(item)  # END must not be lost
                self._disconnect()
                self._stop.wait(self._reconnect_min)  # no busy loop if the GCS accepts and drops at once
            except Exception as e:  # noqa: BLE001
                self._fail("sender thread", e)
            finally:
                self._in_flight = False
        self._disconnect(quiet=True)


# ---------------------------------------------------------------------------------------------------- receiver
class Receiver:
    """GCS-side endpoint: listens, accepts any number of senders, decodes frames in reader threads.

    ``drain()`` / ``get()`` return :class:`~drone_gcs.protocol.Message` objects in arrival order. The inbound
    queue keeps only the newest ``camera_queue`` CAMERA frames if the consumer falls behind.
    """

    def __init__(
        self,
        address: str | None = None,
        *,
        camera_queue: int = 3,
        max_queue: int = 200_000,
        decode_jpeg: bool = True,
        backlog: int = 8,
    ):
        self.address = address or default_address()
        self._family, self._sockaddr, self.address_str = parse_address(self.address)
        self._queue = BoundedMessageQueue({CAMERA: camera_queue}, default_limit=max_queue)
        self._decode_jpeg = decode_jpeg
        self._backlog = backlog
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._conns: dict[int, socket.socket] = {}
        self._conn_ids = itertools.count(1)
        self._lock = threading.Lock()
        self._unix_ino: int | None = None
        self._rate: collections.deque = collections.deque()  # (monotonic time, nbytes)
        self.counts: collections.Counter = collections.Counter()
        self.total_bytes = 0
        self.errors: list[str] = []
        self.last_msg_time: float | None = None
        self.senders: dict[int, dict] = {}  # conn_id -> HELLO fields

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> Receiver:
        if self._family == socket.AF_UNIX:
            path = self._sockaddr
            if os.path.exists(path):
                st = os.stat(path)
                if not stat.S_ISSOCK(st.st_mode):
                    raise RuntimeError(f"{path} exists and is not a socket")
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.settimeout(0.3)
                try:
                    probe.connect(path)
                    probe.close()
                    raise RuntimeError(f"another GCS is already listening at {path}")
                except OSError:
                    probe.close()
                    os.unlink(path)  # stale socket file from a crashed GCS
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(path)
            self._unix_ino = os.stat(path).st_ino
        else:
            s = socket.socket(self._family, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(self._sockaddr)
        s.listen(self._backlog)
        s.settimeout(0.25)
        self._listener = s
        t = threading.Thread(target=self._accept_loop, name="drone_gcs-accept", daemon=True)
        t.start()
        self._threads.append(t)
        return self

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        with self._lock:
            conns = list(self._conns.values())
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        if self._family == socket.AF_UNIX and self._unix_ino is not None:
            try:
                if os.stat(self._sockaddr).st_ino == self._unix_ino:
                    os.unlink(self._sockaddr)
            except OSError:
                pass
        self._queue.wake()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ consumer API
    def get(self, timeout: float | None = None) -> Message | None:
        return self._queue.pop(timeout)

    def drain(self, max_items: int | None = None) -> list[Message]:
        return self._queue.drain(max_items)

    @property
    def num_connections(self) -> int:
        with self._lock:
            return len(self._conns)

    @property
    def connected(self) -> bool:
        return self.num_connections > 0

    @property
    def dropped(self) -> dict:
        return dict(self._queue.dropped)

    def rates(self, window: float = 2.0) -> tuple[float, float]:
        """``(messages/s, bytes/s)`` over the last ``window`` seconds."""
        now = time.monotonic()
        with self._lock:
            while self._rate and now - self._rate[0][0] > window:
                self._rate.popleft()
            n = len(self._rate)
            b = sum(x[1] for x in self._rate)
        return n / window, b / window

    # ------------------------------------------------------------------ threads
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                time.sleep(0.05)
                continue
            cid = next(self._conn_ids)
            conn.settimeout(0.5)
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            except OSError:
                pass
            with self._lock:
                self._conns[cid] = conn
            t = threading.Thread(target=self._reader, args=(conn, cid), name=f"drone_gcs-rx{cid}", daemon=True)
            t.start()
            self._threads.append(t)
            log.info("drone_gcs.link: sender connected (connection %d)", cid)

    def _recv_exact(self, conn: socket.socket, n: int, buf: bytearray | None = None) -> bytearray | None:
        """Read exactly ``n`` bytes; ``None`` on a clean EOF before the first byte."""
        out = buf if buf is not None else bytearray(n)
        mv = memoryview(out)
        pos = 0
        while pos < n:
            if self._stop.is_set():
                raise ConnectionAbortedError("receiver closing")
            try:
                k = conn.recv_into(mv[pos:], n - pos)
            except TimeoutError:
                continue
            except InterruptedError:
                continue
            if k == 0:
                if pos == 0:
                    return None
                raise ConnectionError(f"connection closed mid-frame ({pos}/{n} bytes)")
            pos += k
        return out

    def _reader(self, conn: socket.socket, cid: int) -> None:
        try:
            while not self._stop.is_set():
                prefix = self._recv_exact(conn, PREFIX_SIZE)
                if prefix is None:
                    break
                hlen, plen = parse_prefix(bytes(prefix))
                header = self._recv_exact(conn, hlen) if hlen else bytearray()
                payload = self._recv_exact(conn, plen) if plen else bytearray()
                if header is None or payload is None:
                    raise ConnectionError("connection closed mid-frame")
                msg = decode_message(header, payload, decode_jpeg=self._decode_jpeg)
                now = time.monotonic()
                msg.recv_time = now
                msg.conn_id = cid
                msg.nbytes = PREFIX_SIZE + hlen + plen
                with self._lock:
                    self._rate.append((now, msg.nbytes))
                    self.counts[msg.type] += 1
                    self.total_bytes += msg.nbytes
                    self.last_msg_time = now
                    if msg.type == HELLO:
                        self.senders[cid] = dict(msg.fields)
                self._queue.put(msg)
        except ProtocolError as e:
            self.errors.append(f"connection {cid}: {e}")
            log.warning("drone_gcs.link: protocol error on connection %d: %s (closing it)", cid, e)
        except (ConnectionError, OSError) as e:
            if not self._stop.is_set():
                log.info("drone_gcs.link: connection %d closed: %s", cid, e)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"connection {cid}: {type(e).__name__}: {e}")
            log.exception("drone_gcs.link: reader %d crashed", cid)
        finally:
            with self._lock:
                self._conns.pop(cid, None)
            try:
                conn.close()
            except OSError:
                pass
