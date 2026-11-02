"""Incremental mapping on the GCS (numpy only).

* :class:`GridSpec` - regular xy grid over the mission area (same semantics as ``drone_lab.mapping.GridSpec``).
* :class:`ElevationMap` - 2.5D elevation map built from ray hits, with running statistics per cell (count,
  mean, min, max, std); ``count == 0`` is the *unknown* (fog-of-war) state.
* :class:`Orthomosaic` - per-cell colour from back-projected camera pixels (depth, or the elevation map when
  the frame has no depth); "nadir" keeps the most top-down view of each cell, "latest" the newest, "blend"
  a weighted mean.
* :class:`Geolocator` - detection box -> pixel -> ray -> world XYZ (depth, else ray / elevation-map
  intersection).
* :class:`ObjectTracker` - merges repeated detections of the same object (same class within a radius).
* :func:`evaluate_detections` - precision / recall / localisation error against ground truth.
* :func:`terrain_error` - height error of the elevation map against the true terrain.

Map arrays have shape ``(ny, nx)``: row ``iy`` is the y index (south first), column ``ix`` the x index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .geometry import PinholeCamera


def area_tuple(area: Any) -> tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` from a dict, an object with those attributes or a 4-sequence."""
    if isinstance(area, dict):
        return tuple(float(area[k]) for k in ("x_min", "x_max", "y_min", "y_max"))  # type: ignore[return-value]
    if hasattr(area, "x_min"):
        return (float(area.x_min), float(area.x_max), float(area.y_min), float(area.y_max))
    a = [float(v) for v in area]
    return (a[0], a[1], a[2], a[3])


# ---------------------------------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class GridSpec:
    """Regular xy grid. Cell ``(iy, ix)`` spans ``[x_min + ix*res, x_min + (ix+1)*res)`` (same for y)."""

    x_min: float
    y_min: float
    resolution: float
    nx: int
    ny: int

    @classmethod
    def from_bounds(cls, x_min: float, x_max: float, y_min: float, y_max: float, resolution: float) -> GridSpec:
        """Smallest grid with cell size ``resolution`` that covers the bounds (anchored at the minimum corner)."""
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        nx = max(1, int(math.ceil((x_max - x_min) / resolution - 1e-9)))
        ny = max(1, int(math.ceil((y_max - y_min) / resolution - 1e-9)))
        return cls(float(x_min), float(y_min), float(resolution), nx, ny)

    @classmethod
    def from_dict(cls, d: dict) -> GridSpec:
        """Inverse of :meth:`to_dict`."""
        return cls(float(d["x_min"]), float(d["y_min"]), float(d["resolution"]), int(d["nx"]), int(d["ny"]))

    def to_dict(self) -> dict:
        """JSON-ready fields (``maps/grid.json``)."""
        return {"x_min": self.x_min, "y_min": self.y_min, "resolution": self.resolution, "nx": self.nx, "ny": self.ny}

    @property
    def x_max(self) -> float:
        """Upper x edge of the last column [m]."""
        return self.x_min + self.nx * self.resolution

    @property
    def y_max(self) -> float:
        """Upper y edge of the last row [m]."""
        return self.y_min + self.ny * self.resolution

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape ``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        """Number of cells."""
        return self.nx * self.ny

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)``."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def subdivided(self, k: int) -> GridSpec:
        """Same bounds, ``k`` times finer."""
        return GridSpec(self.x_min, self.y_min, self.resolution / k, self.nx * k, self.ny * k)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell centre coordinates ``(xs (nx,), ys (ny,))``."""
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.resolution
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.resolution
        return xs, ys

    def cell_index(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integer indices ``(ix, iy)`` of points and the mask of points inside the grid."""
        ix = np.floor((np.asarray(x, dtype=np.float64) - self.x_min) / self.resolution).astype(np.int64)
        iy = np.floor((np.asarray(y, dtype=np.float64) - self.y_min) / self.resolution).astype(np.int64)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, valid

    def flat_index(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Row-major flat index ``iy * nx + ix`` (garbage where invalid) and the inside mask."""
        ix, iy, valid = self.cell_index(x, y)
        return iy * self.nx + ix, valid

    def area_mask(self, area: Any) -> np.ndarray:
        """Boolean ``(ny, nx)`` mask of cells whose centre lies inside the area rectangle."""
        x0, x1, y0, y1 = area_tuple(area)
        xs, ys = self.cell_centers()
        return ((ys >= y0) & (ys <= y1))[:, None] & ((xs >= x0) & (xs <= x1))[None, :]


# ---------------------------------------------------------------------------------------------------- elevation
class ElevationMap:
    """Incremental 2.5D elevation map with running per-cell statistics of the ray-hit heights."""

    STATS = ("mean", "max", "min", "std", "count")

    def __init__(self, grid: GridSpec, z_limits: tuple[float, float] = (-1e4, 1e4)):
        self.grid = grid
        self.z_limits = (float(z_limits[0]), float(z_limits[1]))
        shape = grid.shape
        self.count = np.zeros(shape, np.int32)
        self.sum = np.zeros(shape, np.float64)
        self.sumsq = np.zeros(shape, np.float64)
        self.zmax = np.full(shape, -np.inf, np.float32)
        self.zmin = np.full(shape, np.inf, np.float32)
        self.last_t = np.full(shape, np.nan, np.float32)
        self.version = 0
        self.n_points = 0

    def add_points(self, points: ArrayLike, valid: ArrayLike | None = None, t: float | None = None) -> int:
        """Add ray hits (N, 3) [m, world]. Non-finite points, ``valid == False`` and out-of-grid points are
        ignored. Returns the number of points used."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        ok = np.isfinite(P).all(axis=1)
        if valid is not None:
            ok &= np.asarray(valid, dtype=bool).reshape(-1)
        ok &= (P[:, 2] >= self.z_limits[0]) & (P[:, 2] <= self.z_limits[1])
        P = P[ok]
        if not len(P):
            return 0
        idx, inside = self.grid.flat_index(P[:, 0], P[:, 1])
        idx, z = idx[inside], P[inside, 2]
        if not idx.size:
            return 0
        u, inv = np.unique(idx, return_inverse=True)
        self.count.reshape(-1)[u] += np.bincount(inv).astype(np.int32)
        self.sum.reshape(-1)[u] += np.bincount(inv, weights=z)
        self.sumsq.reshape(-1)[u] += np.bincount(inv, weights=z * z)
        z32 = z.astype(np.float32)
        np.maximum.at(self.zmax.reshape(-1), idx, z32)
        np.minimum.at(self.zmin.reshape(-1), idx, z32)
        if t is not None:
            self.last_t.reshape(-1)[u] = t
        self.version += 1
        self.n_points += int(idx.size)
        return int(idx.size)

    # ------------------------------------------------------------------ layers
    @property
    def known(self) -> np.ndarray:
        """Cells with at least one hit (the complement is the fog of war)."""
        return self.count > 0

    def mean(self) -> np.ndarray:
        """Mean hit height per cell (NaN where unknown)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.sum / self.count
        m[self.count == 0] = np.nan
        return m.astype(np.float32)

    def max(self) -> np.ndarray:
        """Highest hit per cell (NaN where unknown)."""
        return np.where(self.count > 0, self.zmax, np.nan).astype(np.float32)

    def min(self) -> np.ndarray:
        """Lowest hit per cell (NaN where unknown)."""
        return np.where(self.count > 0, self.zmin, np.nan).astype(np.float32)

    def std(self) -> np.ndarray:
        """Standard deviation of the hit heights per cell (NaN where unknown)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.sum / self.count
            v = np.maximum(self.sumsq / self.count - m * m, 0.0)
        v[self.count == 0] = np.nan
        return np.sqrt(v).astype(np.float32)

    def layer(self, stat: str = "mean") -> np.ndarray:
        """One statistic of :data:`STATS` as a float32 ``(ny, nx)`` array."""
        if stat == "count":
            return self.count.astype(np.float32)
        if stat not in self.STATS:
            raise ValueError(f"unknown statistic {stat!r} (choose from {self.STATS})")
        return getattr(self, stat)()

    def coverage(self, mask: np.ndarray | None = None) -> float:
        """Fraction of cells (inside ``mask`` if given) with at least one hit."""
        k = self.count > 0
        if mask is None:
            return float(k.mean()) if k.size else 0.0
        n = int(mask.sum())
        return float((k & mask).sum() / n) if n else 0.0

    def heights_at(self, x: ArrayLike, y: ArrayLike, stat: str = "mean") -> np.ndarray:
        """Per-point height of the containing cell (NaN where unknown or outside the grid)."""
        x = np.asarray(x, dtype=np.float64)
        ix, iy, ok = self.grid.cell_index(x, y)
        out = np.full(x.shape, np.nan, np.float64)
        if np.any(ok):
            ixo, iyo = ix[ok], iy[ok]
            c = self.count[iyo, ixo]
            if stat == "max":
                vals = self.zmax[iyo, ixo].astype(np.float64)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals = self.sum[iyo, ixo] / c
            out[ok] = np.where(c > 0, vals, np.nan)
        return out

    def height_at(self, x: float, y: float, radius: float = 0.0) -> float:
        """Mean height of the known cells within ``radius`` [m] of (x, y) (NaN if none)."""
        g = self.grid
        r = max(0, int(math.ceil(radius / g.resolution)))
        ix, iy, _ = g.cell_index(np.array([x]), np.array([y]))
        ix0, iy0 = int(ix[0]), int(iy[0])
        x0, x1 = max(0, ix0 - r), min(g.nx, ix0 + r + 1)
        y0, y1 = max(0, iy0 - r), min(g.ny, iy0 + r + 1)
        if x0 >= x1 or y0 >= y1:
            return float("nan")
        c = self.count[y0:y1, x0:x1]
        if not np.any(c):
            return float("nan")
        return float(self.sum[y0:y1, x0:x1][c > 0].sum() / c[c > 0].sum())

    def ground_estimate(self, default: float = 0.0) -> float:
        """Median of the known cell means (a crude ground level), ``default`` if the map is empty."""
        k = self.count > 0
        if not np.any(k):
            return float(default)
        return float(np.median(self.sum[k] / self.count[k]))


# ---------------------------------------------------------------------------------------------------- rays
def intersect_rays_with_map(
    origin: ArrayLike,
    rays: ArrayLike,
    elevation: ElevationMap | None = None,
    ground_z: float = 0.0,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect rays ``origin + t * ray`` with the elevation map (unknown cells: plane ``z = ground_z``).

    Fixed-point iteration on ``t`` starting from the ground plane (accurate for near-nadir views of smooth
    terrain). Returns world points (N, 3) and a mask of rays that point downwards.
    """
    d = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    o = np.broadcast_to(np.asarray(origin, dtype=np.float64).reshape(-1, 3), d.shape)
    dz = d[:, 2]
    ok = dz < -1e-9
    safe_dz = np.where(ok, dz, -1.0)
    t = (ground_z - o[:, 2]) / safe_dz
    ok &= t > 0
    t = np.where(ok, t, 0.0)
    if elevation is not None and ok.any():
        for i in range(iterations):
            p = o + t[:, None] * d
            h = elevation.heights_at(p[:, 0], p[:, 1])
            h = np.where(np.isfinite(h), h, ground_z)
            t_new = (h - o[:, 2]) / safe_dz
            t_new = np.where(ok & (t_new > 0), t_new, t)
            if i >= 3:  # damp the oscillation at height discontinuities
                t_new = 0.5 * (t + t_new)
            done = np.max(np.abs(t_new - t)) < 1e-3 if t.size else True
            t = t_new
            if done:
                break
    return o + t[:, None] * d, ok


# ---------------------------------------------------------------------------------------------------- mosaic
class Orthomosaic:
    """Incremental orthomosaic: per-cell RGB colour from back-projected camera pixels."""

    MODES = ("nadir", "latest", "blend")

    def __init__(self, grid: GridSpec, mode: str = "nadir"):
        if mode not in self.MODES:
            raise ValueError(f"unknown mosaic mode {mode!r} (choose from {self.MODES})")
        self.grid = grid
        self.mode = mode
        ny, nx = grid.shape
        self.rgb = np.zeros((ny, nx, 3), np.uint8)
        self.filled = np.zeros((ny, nx), bool)
        self.quality = np.full((ny, nx), -1.0, np.float32)
        if mode == "blend":
            self._acc = np.zeros((ny * nx, 3), np.float64)
            self._wsum = np.zeros(ny * nx, np.float64)
        self.version = 0
        self.n_frames = 0
        self.last_stride = 0

    def add_samples(self, x: ArrayLike, y: ArrayLike, colors: ArrayLike, quality: ArrayLike) -> int:
        """Add coloured ground samples: positions (N,), (N,) [m], RGB (N, 3) uint8, quality (N,) in (0, 1]."""
        idx, ok = self.grid.flat_index(x, y)
        if not np.any(ok):
            return 0
        idx = idx[ok]
        col = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[ok]
        q = np.asarray(quality, dtype=np.float32).reshape(-1)[ok]
        flat_rgb = self.rgb.reshape(-1, 3)
        flat_q = self.quality.reshape(-1)
        if self.mode == "latest":
            flat_rgb[idx] = col
            flat_q[idx] = q
        elif self.mode == "nadir":
            order = np.argsort(q, kind="stable")  # ascending: the best sample of a cell is written last
            idx, col, q = idx[order], col[order], q[order]
            better = q >= flat_q[idx] - 0.02  # near-equal newer views may replace older ones
            idx, col, q = idx[better], col[better], q[better]
            flat_rgb[idx] = col
            flat_q[idx] = q
        else:  # blend: weighted running mean, weight = quality^2
            u, inv = np.unique(idx, return_inverse=True)
            w = q.astype(np.float64) ** 2
            for c in range(3):
                self._acc[u, c] += np.bincount(inv, weights=w * col[:, c])
            self._wsum[u] += np.bincount(inv, weights=w)
            flat_rgb[u] = np.clip(self._acc[u] / self._wsum[u, None], 0, 255).astype(np.uint8)
            np.maximum.at(flat_q, idx, q)
        self.filled.reshape(-1)[idx] = True
        self.version += 1
        return int(idx.size)

    def coverage(self, mask: np.ndarray | None = None) -> float:
        """Fraction of cells (inside ``mask`` if given) with a colour."""
        if mask is None:
            return float(self.filled.mean())
        n = int(mask.sum())
        return float((self.filled & mask).sum() / n) if n else 0.0

    def add_camera_frame(
        self,
        cam: PinholeCamera,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        elevation: ElevationMap | None = None,
        ground_z: float = 0.0,
        stride: int | None = None,
        max_range: float = 1000.0,
        min_quality: float = 0.25,
        min_range: float = 0.5,
    ) -> tuple[int, np.ndarray]:
        """Back-project a (sub-sampled) camera frame into the mosaic.

        ``depth`` (H, W) is the distance to the image plane [m]; without it the pixel rays are intersected
        with ``elevation`` (plane ``ground_z`` where unknown). Depth closer than ``min_range`` is the vehicle
        itself (landing gear, propellers) and is ignored. ``stride=None`` picks the pixel step from the
        ground sample distance so that samples are about half a mosaic cell apart.
        Returns ``(samples written, world points (M, 3) of the valid samples)``.
        """
        H, W = rgb.shape[:2]
        min_range = max(0.0, float(min_range))
        if stride is None:
            if depth is not None:
                c = depth[H // 2 - 2 : H // 2 + 3, W // 2 - 2 : W // 2 + 3]
                c = c[np.isfinite(c) & (c > min_range)]
                rng = float(np.median(c)) if c.size else float("nan")
            else:
                rng = float("nan")
            if not math.isfinite(rng):
                cosn = max(0.2, math.cos(math.radians(cam.nadir_angle_deg())))
                rng = max(0.5, (cam.t[2] - ground_z) / cosn)
            gsd = rng / max(cam.fx, 1e-6)
            stride = int(np.clip(math.floor(0.5 * self.grid.resolution / max(gsd, 1e-9)), 1, 32))
        self.last_stride = stride
        vs = np.arange(stride // 2, H, stride)
        us = np.arange(stride // 2, W, stride)
        uu, vv = np.meshgrid(us, vs)
        u, v = uu.ravel(), vv.ravel()
        rays = cam.rays(u, v)
        if depth is not None:
            d = np.asarray(depth, dtype=np.float64)[v, u]
            ok = np.isfinite(d) & (d > min_range) & (d > 0) & (d < max_range)
            P = cam.t + np.where(ok, d, 0.0)[:, None] * rays
        else:
            P, ok = intersect_rays_with_map(cam.t, rays, elevation, ground_z)
        q = -rays[:, 2] / np.linalg.norm(rays, axis=1)
        ok &= q > min_quality
        if not np.any(ok):
            return 0, np.zeros((0, 3))
        colors = np.asarray(rgb)[v[ok], u[ok]]
        n = self.add_samples(P[ok, 0], P[ok, 1], colors, q[ok])
        self.n_frames += 1
        return n, P[ok]


# ---------------------------------------------------------------------------------------------------- geolocation
class Geolocator:
    """Detection box (continuous pixel coordinates, pixel ``u`` spans ``[u, u+1)``) -> world XYZ.

    ``ref``: ``"center"`` (top-down views), ``"bottom"`` (bottom-centre = ground contact point, oblique views)
    or ``"auto"`` (centre when the optical axis is within ``nadir_threshold_deg`` of straight down).
    """

    def __init__(
        self,
        elevation: ElevationMap | None = None,
        ground_z: float = 0.0,
        ref: str = "auto",
        nadir_threshold_deg: float = 35.0,
        depth_window: float = 0.3,
        max_range: float = 1000.0,
        min_range: float = 0.5,
    ):
        if ref not in ("auto", "center", "bottom"):
            raise ValueError(f"unknown reference point {ref!r}")
        self.elevation = elevation
        self.ground_z = float(ground_z)
        self.ref = ref
        self.nadir_threshold_deg = float(nadir_threshold_deg)
        self.depth_window = float(depth_window)
        self.max_range = float(max_range)
        self.min_range = max(0.0, float(min_range))  # nearer depth = the vehicle itself: use the map instead

    def locate(self, box: ArrayLike, cam: PinholeCamera, depth: np.ndarray | None = None) -> dict:
        """World position of a detection box (x1, y1, x2, y2) [px].

        Returns ``{"xyz": (3,), "ok": bool, "method": "depth"|"map"|"plane", "ref": ..., "pixel": (u, v)}``.
        """
        x1, y1, x2, y2 = (float(b) for b in box)
        ref = self.ref
        if ref == "auto":
            ref = "center" if cam.nadir_angle_deg() <= self.nadir_threshold_deg else "bottom"
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        xc = 0.5 * (x1 + x2)
        yc = 0.5 * (y1 + y2) if ref == "center" else y2 - 0.05 * bh
        u_idx, v_idx = xc - 0.5, yc - 0.5  # continuous -> pixel-index coordinates
        out = {"ok": False, "method": "none", "ref": ref, "pixel": (xc, yc), "xyz": np.full(3, np.nan)}
        if depth is not None:
            H, W = depth.shape[:2]
            hw, hh = max(1.0, self.depth_window * bw / 2), max(1.0, self.depth_window * bh / 2)
            if ref == "center":
                r0, r1 = yc - hh, yc + hh
            else:
                r0, r1 = y2 - 2 * hh, y2
            c0, c1 = int(max(0, math.floor(xc - hw))), int(min(W, math.ceil(xc + hw)))
            r0i, r1i = int(max(0, math.floor(r0))), int(min(H, math.ceil(r1)))
            if c1 > c0 and r1i > r0i:
                win = np.asarray(depth[r0i:r1i, c0:c1], dtype=np.float64)
                win = win[np.isfinite(win) & (win > self.min_range) & (win > 0) & (win < self.max_range)]
                if win.size:
                    d = float(np.median(win))
                    out.update(xyz=cam.backproject([u_idx], [v_idx], [d])[0], ok=True, method="depth", depth=d)
                    return out
        ray = cam.rays([u_idx], [v_idx])
        P, ok = intersect_rays_with_map(cam.t, ray, self.elevation, self.ground_z)
        if ok[0]:
            known = self.elevation is not None and np.isfinite(self.elevation.heights_at(P[:, 0], P[:, 1])[0])
            out.update(xyz=P[0], ok=True, method="map" if known else "plane")
        return out


# ---------------------------------------------------------------------------------------------------- tracking
@dataclass
class TrackedObject:
    """An object built from repeated detections (position = confidence-weighted mean)."""

    id: int
    cls: str
    pos: np.ndarray
    conf_max: float
    conf_sum: float
    hits: int
    first_seen: float
    last_seen: float
    wsum: float
    drone_id: int = 0
    last_box: tuple | None = None
    spread: float = 0.0  # RMS distance of the detections from the mean [m]
    _sq: float = field(default=0.0, repr=False)

    @property
    def conf(self) -> float:
        """Mean confidence of the detections."""
        return self.conf_sum / max(1, self.hits)

    def to_dict(self) -> dict:
        """JSON record (``detections.json``; the matched ground truth is added by the report)."""
        return {
            "id": self.id,
            "class": self.cls,
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "z": float(self.pos[2]),
            "confidence": round(self.conf, 4),
            "confidence_max": round(self.conf_max, 4),
            "hits": self.hits,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "spread_m": round(self.spread, 3),
            "drone_id": self.drone_id,
        }


class ObjectTracker:
    """Associates geolocated detections with objects: same class, horizontal distance <= ``merge_radius``.

    Each frame is associated greedily one-to-one (nearest first), so two objects seen in the same frame are
    never merged. Objects with ``hits >= min_hits`` are *confirmed*.
    """

    def __init__(self, merge_radius: float = 2.0, min_hits: int = 2, class_radius: dict | None = None):
        self.merge_radius = float(merge_radius)
        self.min_hits = int(min_hits)
        self.class_radius = dict(class_radius or {})
        self.objects: list[TrackedObject] = []
        self._next_id = 1
        self.n_detections = 0

    def radius(self, cls: str) -> float:
        """Merge radius of class ``cls`` [m]."""
        return float(self.class_radius.get(cls, self.merge_radius))

    def update(self, detections: list[dict]) -> list[TrackedObject]:
        """``detections``: dicts with ``cls``, ``xyz`` (3,), ``conf`` and optionally ``t``, ``box``, ``drone_id``.
        Returns the objects touched by this update."""
        dets = [d for d in detections if d.get("xyz") is not None and np.all(np.isfinite(d["xyz"]))]
        if not dets:
            return []
        self.n_detections += len(dets)
        pairs = []
        for i, d in enumerate(dets):
            xy = np.asarray(d["xyz"], dtype=np.float64)[:2]
            r = self.radius(d["cls"])
            for j, ob in enumerate(self.objects):
                if ob.cls != d["cls"]:
                    continue
                dist = float(np.hypot(*(ob.pos[:2] - xy)))
                if dist <= r:
                    pairs.append((dist, i, j))
        pairs.sort()
        used_d, used_o, assign = set(), set(), {}
        for _, i, j in pairs:
            if i in used_d or j in used_o:
                continue
            used_d.add(i)
            used_o.add(j)
            assign[i] = j
        touched = []
        for i, d in enumerate(dets):
            xyz = np.asarray(d["xyz"], dtype=np.float64).reshape(3)
            conf = float(d.get("conf", 1.0))
            w = max(conf, 1e-3)
            t = float(d.get("t", 0.0) or 0.0)
            if i in assign:
                ob = self.objects[assign[i]]
                new_pos = (ob.pos * ob.wsum + xyz * w) / (ob.wsum + w)
                ob._sq += w * float(np.sum((xyz[:2] - new_pos[:2]) ** 2))
                ob.pos = new_pos
                ob.wsum += w
                ob.hits += 1
                ob.conf_sum += conf
                ob.conf_max = max(ob.conf_max, conf)
                ob.last_seen = max(ob.last_seen, t)
                ob.last_box = d.get("box")
                ob.spread = math.sqrt(ob._sq / ob.wsum)
            else:
                ob = TrackedObject(
                    id=self._next_id,
                    cls=str(d["cls"]),
                    pos=xyz.copy(),
                    conf_max=conf,
                    conf_sum=conf,
                    hits=1,
                    first_seen=t,
                    last_seen=t,
                    wsum=w,
                    drone_id=int(d.get("drone_id", 0)),
                    last_box=d.get("box"),
                )
                self._next_id += 1
                self.objects.append(ob)
            touched.append(ob)
        self._merge_close()
        return touched

    def _merge_close(self) -> None:
        """Merge objects of the same class that drifted within the merge radius of each other."""
        merged = True
        while merged:
            merged = False
            for a in range(len(self.objects)):
                for b in range(a + 1, len(self.objects)):
                    oa, ob = self.objects[a], self.objects[b]
                    if oa.cls != ob.cls or np.hypot(*(oa.pos[:2] - ob.pos[:2])) > 0.5 * self.radius(oa.cls):
                        continue
                    w = oa.wsum + ob.wsum
                    oa.pos = (oa.pos * oa.wsum + ob.pos * ob.wsum) / w
                    oa.wsum, oa.hits = w, oa.hits + ob.hits
                    oa.conf_sum += ob.conf_sum
                    oa.conf_max = max(oa.conf_max, ob.conf_max)
                    oa.first_seen = min(oa.first_seen, ob.first_seen)
                    oa.last_seen = max(oa.last_seen, ob.last_seen)
                    del self.objects[b]
                    merged = True
                    break
                if merged:
                    break

    def confirmed(self) -> list[TrackedObject]:
        """Objects seen at least ``min_hits`` times."""
        return [o for o in self.objects if o.hits >= self.min_hits]


# ---------------------------------------------------------------------------------------------------- metrics
def evaluate_detections(predictions: list[dict], ground_truth: list[dict], match_radius: float = 2.0) -> dict:
    """Match predicted objects to ground truth (same class, horizontal distance <= ``match_radius``, greedy
    nearest first) and report precision, recall and localisation error.

    ``predictions``: dicts with ``class`` and ``x, y, z`` (or ``pos``); ``ground_truth``: dicts with ``class``
    and ``pos`` (or ``x, y, z``).
    """

    def _xyz(d):
        if "pos" in d:
            return np.asarray(d["pos"], dtype=np.float64).reshape(3)
        return np.array([d["x"], d["y"], d.get("z", 0.0)], dtype=np.float64)

    P = [(str(d.get("class")), _xyz(d)) for d in predictions]
    G = [(str(d.get("class")), _xyz(d)) for d in ground_truth]
    pairs = []
    for i, (pc, pp) in enumerate(P):
        for j, (gc, gp) in enumerate(G):
            if pc == gc:
                dist = float(np.hypot(*(pp[:2] - gp[:2])))
                if dist <= match_radius:
                    pairs.append((dist, i, j))
    pairs.sort()
    used_p, used_g, matches = set(), set(), []
    for dist, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        dz = float(P[i][1][2] - G[j][1][2])
        matches.append({"pred": i, "gt": j, "class": P[i][0], "err_xy": dist, "err_z": dz})
    tp = len(matches)
    exy = np.array([m["err_xy"] for m in matches]) if matches else np.zeros(0)
    exyz = np.array([math.hypot(m["err_xy"], m["err_z"]) for m in matches]) if matches else np.zeros(0)
    classes = sorted({c for c, _ in P} | {c for c, _ in G})
    per_class = {}
    for c in classes:
        n_p = sum(1 for pc, _ in P if pc == c)
        n_g = sum(1 for gc, _ in G if gc == c)
        mc = [m for m in matches if m["class"] == c]
        per_class[c] = {
            "predicted": n_p,
            "ground_truth": n_g,
            "tp": len(mc),
            "precision": len(mc) / n_p if n_p else None,
            "recall": len(mc) / n_g if n_g else None,
            "mean_err_xy": float(np.mean([m["err_xy"] for m in mc])) if mc else None,
        }
    return {
        "match_radius_m": match_radius,
        "predicted": len(P),
        "ground_truth": len(G),
        "tp": tp,
        "fp": len(P) - tp,
        "fn": len(G) - tp,
        "precision": tp / len(P) if P else None,
        "recall": tp / len(G) if G else None,
        "mean_err_xy_m": float(exy.mean()) if tp else None,
        "median_err_xy_m": float(np.median(exy)) if tp else None,
        "max_err_xy_m": float(exy.max()) if tp else None,
        "mean_err_xyz_m": float(exyz.mean()) if tp else None,
        "per_class": per_class,
        "matches": matches,
        "unmatched_gt": [j for j in range(len(G)) if j not in used_g],
        "unmatched_pred": [i for i in range(len(P)) if i not in used_p],
    }


def terrain_error(estimate: ArrayLike, truth: ArrayLike, mask: ArrayLike | None = None, step: float = 0.1) -> dict:
    """Height error of a map against the true terrain on the same grid: cells known in both (inside ``mask``).

    ``smooth`` repeats the statistics away from steps: cells whose true height is within ``step`` of all eight
    neighbours (a cell across a wall is ambiguous by nature). Errors are estimate - truth [m].
    """
    est = np.asarray(estimate, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    ok = np.isfinite(est) & np.isfinite(tru)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    padded = np.pad(tru, 1, mode="edge")
    stepped = np.zeros(tru.shape, bool)
    ny, nx = tru.shape
    with np.errstate(invalid="ignore"):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    stepped |= np.abs(padded[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx] - tru) > step

    def stats(sel: np.ndarray) -> dict | None:
        e = (est - tru)[sel]
        if not e.size:
            return None
        return {
            "cells": int(e.size),
            "rmse_m": float(np.sqrt(np.mean(e * e))),
            "mae_m": float(np.mean(np.abs(e))),
            "bias_m": float(np.mean(e)),
            "max_abs_m": float(np.max(np.abs(e))),
        }

    n = int(ok.sum())
    return {
        "all": stats(ok),
        "smooth": stats(ok & ~stepped),
        "smooth_fraction": float((ok & ~stepped).sum() / n) if n else None,
        "step_threshold_m": step,
    }


def path_length(points: ArrayLike) -> float:
    """Length of a polyline (N, 2|3) [m]."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
