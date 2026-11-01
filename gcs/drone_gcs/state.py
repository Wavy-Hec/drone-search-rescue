"""GCS mission state: applies link messages to the maps, the object tracker and the per-drone tracks."""

from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import PinholeCamera, compass_heading_deg, yaw_from_quat
from .mapping import ElevationMap, Geolocator, GridSpec, ObjectTracker, Orthomosaic, area_tuple, intersect_rays_with_map
from .protocol import CAMERA, END, GT_OBJECTS, GT_TERRAIN, HELLO, MISSION, SCAN, TELEMETRY, Message

log = logging.getLogger("drone_gcs.state")

MISSION_SCOPED = frozenset({TELEMETRY, SCAN, CAMERA, END})
"""Message types the Sender stamps with the current ``mission_id``."""


@dataclass
class GCSConfig:
    photo_mode: str = "nadir"  # orthomosaic: nadir | latest | blend
    photo_subdiv: int = 2  # photo cells per elevation cell (per axis)
    depth_elevation: bool = False  # also feed back-projected depth pixels into the elevation map
    merge_radius: float = 2.0  # tracker: same class within this horizontal distance = same object [m]
    # per-class radius: piles of crates or barrels stand closer together than that, a forklift is larger
    class_radius: dict = field(
        default_factory=lambda: {
            "person": 1.5,
            "vehicle": 3.0,
            "crate": 0.8,
            "barrel": 0.8,
            "cone": 0.8,
            "pallet": 1.0,
        }
    )
    min_hits: int = 2  # tracker: detections needed to confirm an object
    det_ref: str = "auto"  # geolocation reference pixel: auto | center | bottom
    edge_margin: float = 3.0  # boxes this close to the image border are partial views: shown, not mapped [px]
    depth_min_range: float = 0.5  # depth closer than this is the drone itself (legs, props): not mapped [m]
    auto_grid_size: float = 120.0  # map size [m] when data arrives before any MISSION
    auto_grid_resolution: float = 0.5
    max_grid_cells: int = 4_000_000  # elevation cells (32 B each); larger missions get a coarser resolution
    max_photo_cells: int = 16_000_000


@dataclass
class CameraFrame:
    drone_id: int
    t: float
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray | None
    cam: PinholeCamera
    name: str = "cam0"
    recv_time: float = 0.0
    detections: list | None = None  # [(Detection, geo dict)] once the detector has run on this frame
    det_ms: float = 0.0
    gt_boxes: list | None = None  # boxes the simulator rendered for this frame (evaluation / --detector gt)


@dataclass
class DroneState:
    drone_id: int
    trail: list = field(default_factory=list)  # (t, x, y, z)
    t: float | None = None
    pos: np.ndarray | None = None
    quat: np.ndarray | None = None
    vel: np.ndarray | None = None
    wp_index: int | None = None
    carrot: np.ndarray | None = None
    phase: str = ""  # mission phase reported by the drone (take-off, survey, ...)
    yaw: float = 0.0
    camera: CameraFrame | None = None
    det_frame: CameraFrame | None = None
    scan_ground: float = float("nan")  # median z of the latest scan
    cam_times: collections.deque = field(default_factory=lambda: collections.deque(maxlen=20))
    _trail_np: np.ndarray | None = None
    _trail_n: int = -1

    @property
    def heading_deg(self) -> float:
        return compass_heading_deg(self.yaw)

    @property
    def speed_h(self) -> float:
        return float(math.hypot(self.vel[0], self.vel[1])) if self.vel is not None else float("nan")

    @property
    def speed_v(self) -> float:
        return float(self.vel[2]) if self.vel is not None else float("nan")

    def trail_array(self) -> np.ndarray:
        if self._trail_n != len(self.trail):
            self._trail_np = np.asarray(self.trail, dtype=np.float64).reshape(-1, 4)
            self._trail_n = len(self.trail)
        return self._trail_np

    @property
    def cam_fps(self) -> float:
        if len(self.cam_times) < 2:
            return 0.0
        dt = self.cam_times[-1] - self.cam_times[0]
        return (len(self.cam_times) - 1) / dt if dt > 0 else 0.0


@dataclass
class TerrainTruth:
    """True terrain height on a regular grid (GT_TERRAIN), for the map-error metric only."""

    heights: np.ndarray  # (ny, nx) [m], row 0 = south, NaN = unknown
    grid: GridSpec

    @classmethod
    def from_message(cls, msg: Message) -> TerrainTruth:
        h = np.asarray(msg.arrays["heights"], dtype=np.float32)
        f = msg.fields
        grid = GridSpec(float(f["x_min"]), float(f["y_min"]), float(f["resolution"]), h.shape[1], h.shape[0])
        return cls(h, grid)

    def on_grid(self, grid: GridSpec) -> np.ndarray:
        """Heights at the cell centres of ``grid`` (nearest truth cell; NaN outside the truth grid)."""
        xs, ys = np.meshgrid(*grid.cell_centers())
        ix, iy, ok = self.grid.cell_index(xs, ys)
        out = np.full(grid.shape, np.nan, np.float32)
        out[ok] = self.heights[iy[ok], ix[ok]]
        return out


class GCSState:
    """Everything the dashboard shows and the report writes."""

    def __init__(self, cfg: GCSConfig | None = None):
        self.cfg = cfg or GCSConfig()
        self.mission: dict | None = None
        self.auto_grid = False
        self.grid: GridSpec | None = None
        self.photo_grid: GridSpec | None = None
        self.area: tuple | None = None
        self.area_mask: np.ndarray | None = None
        self.area_mask_photo: np.ndarray | None = None
        self.elevation: ElevationMap | None = None
        self.mosaic: Orthomosaic | None = None
        self.geolocator: Geolocator | None = None
        self.tracker = ObjectTracker(self.cfg.merge_radius, self.cfg.min_hits, self.cfg.class_radius)
        self.planned_path: np.ndarray | None = None
        self.ground_z = 0.0
        self.z_range: tuple | None = None
        self.drones: dict[int, DroneState] = {}
        self.gt_objects: list[dict] | None = None
        self.gt_terrain: TerrainTruth | None = None
        self.home: np.ndarray | None = None  # landing pad (MISSION "home")
        self.end_fields: dict = {}
        self.ended = False
        self.end_reason = ""
        self.end_wall: float | None = None
        self.t_first: float | None = None
        self.t_now: float | None = None
        self.counts: collections.Counter = collections.Counter()
        self.errors: collections.Counter = collections.Counter()
        self.last_error = ""
        self.events: collections.deque = collections.deque(maxlen=6)  # (wall time, text)
        self.coverage_timeline: list[tuple] = []  # (t, elevation coverage, photo coverage, confirmed objects)
        self.trajectory: list[tuple] = []  # rows for trajectory.csv
        self.senders: dict[int, dict] = {}
        self.missions_seen = 0
        self.stale: collections.Counter = collections.Counter()  # mission_id -> ignored messages of other missions
        self._cov_cache = (None, 0.0, 0.0)

    # ------------------------------------------------------------------ helpers
    def event(self, text: str) -> None:
        self.events.append((time.time(), text))
        log.info(text)

    @property
    def has_data(self) -> bool:
        return bool(self.counts[TELEMETRY] or self.counts[SCAN] or self.counts[CAMERA])

    @property
    def mission_time(self) -> float:
        if self.t_first is None or self.t_now is None:
            return 0.0
        return max(0.0, self.t_now - self.t_first)

    def drone(self, drone_id: int) -> DroneState:
        d = self.drones.get(drone_id)
        if d is None:
            d = self.drones[drone_id] = DroneState(drone_id)
        return d

    def _touch_time(self, t) -> None:
        if t is None:
            return
        t = float(t)
        if self.t_first is None:
            self.t_first = t
        self.t_now = t if self.t_now is None else max(self.t_now, t)

    # ------------------------------------------------------------------ grid setup
    def setup_grid(self, area, resolution: float, padding: float | None = None, photo_resolution: float | None = None):
        x0, x1, y0, y1 = area_tuple(area)
        if padding is None:  # room for the turns outside the survey area
            padding = max(4.0, 0.15 * max(x1 - x0, y1 - y0))
        self.area = (x0, x1, y0, y1)
        requested = float(resolution)
        resolution = self._capped_resolution(x1 - x0, y1 - y0, float(padding), requested)
        if resolution != requested:
            cap = self.cfg.max_grid_cells
            self.event(f"map too large at {requested:g} m: using {resolution:g} m cells (max {cap:,} cells)")
        # Cell *centres* sit on the lattice x0 + k * resolution: the padding is rounded up to whole cells. Survey
        # paths and ray grids usually live on round numbers too, so rays land in the middle of cells instead of
        # on cell edges (where small lateral wobble would flip them between neighbouring cells and bias the
        # per-cell sampling).
        padding = math.ceil(max(0.0, float(padding)) / resolution - 1e-9) * resolution
        h = 0.5 * resolution
        self.grid = GridSpec.from_bounds(
            x0 - padding - h, x1 + padding + h, y0 - padding - h, y1 + padding + h, resolution
        )
        k = max(1, int(self.cfg.photo_subdiv))
        if photo_resolution:
            k = max(1, int(round(resolution / photo_resolution)))
        while self.grid.size * k * k > self.cfg.max_photo_cells and k > 1:
            k -= 1
        self.photo_grid = self.grid.subdivided(k)
        self.area_mask = self.grid.area_mask(self.area)
        self.area_mask_photo = self.photo_grid.area_mask(self.area)
        self.elevation = ElevationMap(self.grid)
        self.mosaic = Orthomosaic(self.photo_grid, self.cfg.photo_mode)
        self.geolocator = Geolocator(
            self.elevation, self.ground_z, self.cfg.det_ref, min_range=self.cfg.depth_min_range
        )

    def _capped_resolution(self, w: float, h: float, padding: float, resolution: float) -> float:
        """``resolution``, or the smallest whole multiple of it that keeps the grid under ``max_grid_cells``
        (a multiple keeps the cell lattice of the requested resolution)."""
        limit = max(1, int(self.cfg.max_grid_cells))
        k = 1
        while k < 1024:
            r = resolution * k
            p = math.ceil(padding / r - 1e-9) * r
            cells = (math.ceil((w + 2 * p) / r - 1e-9) + 1) * (math.ceil((h + 2 * p) / r - 1e-9) + 1)
            if cells <= limit:
                break
            k += 1
        return resolution * k

    def _ensure_grid(self, x: float, y: float) -> None:
        if self.grid is not None:
            return
        h = self.cfg.auto_grid_size / 2
        self.auto_grid = True
        self.setup_grid((x - h, x + h, y - h, y + h), self.cfg.auto_grid_resolution, padding=0.0)
        self.event(f"no MISSION yet: auto map {2 * h:.0f} m around ({x:.1f}, {y:.1f})")

    # ------------------------------------------------------------------ apply
    def is_stale(self, msg: Message) -> bool:
        """Data or END stamped with a ``mission_id`` other than the current MISSION's (left over from an
        earlier mission, e.g. another drone's stream still catching up)."""
        if msg.type not in MISSION_SCOPED or self.mission is None:
            return False
        mid = msg.fields.get("mission_id")
        return mid is not None and str(mid) != str(self.mission.get("mission_id"))

    def apply(self, msg: Message):
        """Apply one message. Returns a :class:`CameraFrame` for CAMERA messages (to feed the detector)."""
        if self.is_stale(msg):
            mid = str(msg.fields.get("mission_id"))
            self.stale[mid] += 1
            if self.stale[mid] == 1:
                self.event(f"ignoring {msg.type} of mission '{mid}' (current: '{self.mission.get('mission_id')}')")
            return None
        self.counts[msg.type] += 1
        try:
            if msg.type == TELEMETRY:
                self._on_telemetry(msg)
            elif msg.type == SCAN:
                self._on_scan(msg)
            elif msg.type == CAMERA:
                return self._on_camera(msg)
            elif msg.type == MISSION:
                self._on_mission(msg)
            elif msg.type == GT_OBJECTS:
                self.gt_objects = list(msg.fields.get("objects", []))
                self.event(f"ground truth: {len(self.gt_objects)} objects (evaluation only)")
            elif msg.type == GT_TERRAIN:
                self.gt_terrain = TerrainTruth.from_message(msg)
                g = self.gt_terrain.grid
                self.event(f"ground truth: terrain {g.nx}x{g.ny} @ {g.resolution:g} m (evaluation only)")
            elif msg.type == END:
                self._touch_time(msg.t)
                if not self.ended:
                    self.ended = True
                    self.end_reason = str(msg.fields.get("reason", ""))
                    self.end_fields = dict(msg.fields)
                    self.end_wall = time.time()
                    self.event(f"END received ({self.end_reason})")
            elif msg.type == HELLO:
                self.senders[msg.conn_id] = dict(msg.fields)
                self.event(f"link: '{msg.fields.get('name', msg.sender)}' connected")
            else:
                self.errors["unknown_type"] += 1
        except Exception as e:  # noqa: BLE001  (a bad message must not kill the GCS)
            self.errors[msg.type] += 1
            self.last_error = f"{msg.type}: {type(e).__name__}: {e}"
            log.exception("failed to apply %s", msg.type)
        return None

    def _on_mission(self, msg: Message) -> None:
        f = msg.fields
        mid = f.get("mission_id")
        if self.mission is not None and mid == self.mission.get("mission_id"):
            return  # the same mission re-sent after a reconnect
        # A new mission in this state (the app archives a previous mission that has data before calling us):
        # an END seen so far belonged to the previous mission, and so did its sim clock.
        self.ended, self.end_reason, self.end_wall = False, "", None
        if not self.has_data:
            self.t_first = self.t_now = None
        self.missions_seen += 1
        self.mission = dict(f)
        if msg.drone_id is not None:
            self.mission["drone_id"] = msg.drone_id
        self.ground_z = float(f.get("ground_z", 0.0) or 0.0)
        home = f.get("home")
        self.home = np.asarray(home, dtype=np.float64).reshape(3) if home is not None else None
        zr = f.get("z_range")
        self.z_range = (float(zr[0]), float(zr[1])) if zr else None
        pp = msg.arrays.get("planned_path")
        if pp is None and f.get("planned_path") is not None:
            pp = np.asarray(f["planned_path"], dtype=np.float64)
        self.planned_path = None if pp is None else np.asarray(pp, dtype=np.float64).reshape(-1, 3)
        self.auto_grid = False
        self.setup_grid(f["area"], float(f["resolution"]), f.get("padding"), f.get("photo_resolution"))
        g = self.grid
        self.event(
            f"MISSION '{f.get('name', '?')}': {self.area[1] - self.area[0]:.0f}x{self.area[3] - self.area[2]:.0f} m, "
            f"map {g.nx}x{g.ny} @ {g.resolution:g} m"
        )

    def _on_telemetry(self, msg: Message) -> None:
        f = msg.fields
        d = self.drone(msg.drone_id)
        pos = np.asarray(f["pos"], dtype=np.float64)
        d.t = msg.t
        d.pos = pos
        d.quat = np.asarray(f["quat"], dtype=np.float64)
        d.vel = np.asarray(f["vel"], dtype=np.float64) if f.get("vel") is not None else None
        d.wp_index = f.get("wp_index")
        d.carrot = np.asarray(f["carrot"], dtype=np.float64) if f.get("carrot") is not None else None
        d.yaw = yaw_from_quat(d.quat)
        d.phase = str(f.get("phase") or "")
        self._touch_time(msg.t)
        self._ensure_grid(pos[0], pos[1])
        t = msg.t if msg.t is not None else 0.0
        if (
            not d.trail
            or (pos[0] - d.trail[-1][1]) ** 2 + (pos[1] - d.trail[-1][2]) ** 2 > 0.01
            or t - d.trail[-1][0] > 1.0
        ):
            d.trail.append((t, pos[0], pos[1], pos[2]))
        v = d.vel if d.vel is not None else (np.nan, np.nan, np.nan)
        c = d.carrot if d.carrot is not None else (np.nan, np.nan, np.nan)
        self.trajectory.append(
            (msg.drone_id, t, *pos, *d.quat, *v, -1 if d.wp_index is None else int(d.wp_index), *c, self.agl(d))
        )

    def _on_scan(self, msg: Message) -> None:
        pts = msg.arrays.get("points")
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        self._touch_time(msg.t)
        finite = np.isfinite(pts).all(axis=1)
        if not finite.any():
            return
        if self.grid is None:
            c = pts[finite].mean(axis=0)
            self._ensure_grid(float(c[0]), float(c[1]))
        valid = msg.arrays.get("valid")
        self.elevation.add_points(pts, valid, msg.t)
        d = self.drone(msg.drone_id)
        zz = pts[finite, 2]
        d.scan_ground = float(np.median(zz)) if zz.size else float("nan")
        self._record_coverage()

    def _on_camera(self, msg: Message) -> CameraFrame | None:
        f = msg.fields
        rgb = msg.arrays.get("rgb")
        if rgb is None:
            return None
        cam = PinholeCamera.from_pose(
            f["K"],
            f.get("width", rgb.shape[1]),
            f.get("height", rgb.shape[0]),
            f["pos"],
            f["quat"],
            f.get("pixel_center", 0.5),
        )
        depth = msg.arrays.get("depth")
        self._touch_time(msg.t)
        self._ensure_grid(cam.t[0], cam.t[1])
        ground = self.ground_level()
        n, P = self.mosaic.add_camera_frame(cam, rgb, depth, self.elevation, ground, min_range=self.cfg.depth_min_range)
        if self.cfg.depth_elevation and depth is not None and len(P):
            self.elevation.add_points(P, None, msg.t)
        d = self.drone(msg.drone_id)
        frame = CameraFrame(
            msg.drone_id,
            msg.t or 0.0,
            int(f.get("frame_id", self.counts[CAMERA])),
            rgb,
            depth,
            cam,
            str(f.get("camera", "cam0")),
            msg.recv_time,
            gt_boxes=f.get("gt_boxes"),
        )
        d.camera = frame
        d.cam_times.append(time.monotonic())
        self._record_coverage()
        return frame

    def apply_detections(self, frame: CameraFrame, detections: list, ms: float) -> list:
        """Geolocate a frame's detections, update the tracker; returns ``[(Detection, geo), ...]``."""
        out = []
        if self.geolocator is None:
            return out
        self.geolocator.ground_z = self.ground_level()
        batch = []
        W, H = frame.cam.width, frame.cam.height
        m = self.cfg.edge_margin
        for det in detections:
            geo = self.geolocator.locate(det.box, frame.cam, frame.depth)
            x1, y1, x2, y2 = det.box
            geo["truncated"] = m > 0 and (x1 < m or y1 < m or x2 > W - m or y2 > H - m)
            out.append((det, geo))
            if geo["ok"] and not geo["truncated"]:
                batch.append(
                    {
                        "cls": det.cls,
                        "xyz": geo["xyz"],
                        "conf": det.conf,
                        "t": frame.t,
                        "box": det.box,
                        "drone_id": frame.drone_id,
                    }
                )
        self.tracker.update(batch)
        frame.detections = out
        frame.det_ms = ms
        self.drone(frame.drone_id).det_frame = frame
        return out

    # ------------------------------------------------------------------ derived values
    def ground_level(self) -> float:
        if self.elevation is not None and self.elevation.n_points:
            return self.elevation.ground_estimate(self.ground_z)
        return self.ground_z

    def agl(self, d: DroneState) -> float:
        """Height above the mapped ground under the drone (else the latest scan median, else MISSION ground_z)."""
        if d.pos is None:
            return float("nan")
        g = float("nan")
        if self.elevation is not None:
            g = self.elevation.height_at(d.pos[0], d.pos[1], radius=0.5)
        if not math.isfinite(g):
            g = d.scan_ground
        if not math.isfinite(g):
            g = self.ground_z
        return float(d.pos[2] - g)

    def coverage(self) -> tuple[float, float]:
        """(elevation coverage, photo coverage) of the mission area, fractions in [0, 1]."""
        if self.elevation is None:
            return 0.0, 0.0
        key = (self.elevation.version, self.mosaic.version)
        if self._cov_cache[0] != key:
            self._cov_cache = (key, self.elevation.coverage(self.area_mask), self.mosaic.coverage(self.area_mask_photo))
        return self._cov_cache[1], self._cov_cache[2]

    def _record_coverage(self) -> None:
        t = self.t_now if self.t_now is not None else 0.0
        if self.coverage_timeline and t - self.coverage_timeline[-1][0] < 0.25:
            return
        ce, cp = self.coverage()
        self.coverage_timeline.append((t, ce, cp, len(self.tracker.confirmed())))

    def flown_length(self) -> float:
        total = 0.0
        for d in self.drones.values():
            a = d.trail_array()
            if len(a) > 1:
                total += float(np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1).sum())
        return total

    def footprint(self, frame: CameraFrame) -> np.ndarray | None:
        """Ground polygon (4, 3) of a camera frame's image corners."""
        u, v = frame.cam.corner_pixels()
        P, ok = intersect_rays_with_map(frame.cam.t, frame.cam.rays(u, v), self.elevation, self.ground_level())
        if not ok.all():
            return None
        return P
