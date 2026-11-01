"""2.5D elevation mapping, mesh ground truth and survey metrics (numpy only).

Map arrays have shape ``(ny, nx)``: row ``iy`` is the y index and column ``ix`` the x index, so
``plt.imshow(m, origin="lower", extent=grid.extent)`` shows them in world orientation. Empty cells are NaN.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np

from .paths import Area, distance_to_polyline

REDUCERS = ("median", "mean", "max", "min")
"""Per-cell reductions supported by :func:`elevation_map`."""


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
    def from_area(cls, area: Area, resolution: float, padding: float = 0.0) -> GridSpec:
        """Grid covering ``area`` grown by ``padding`` on each side."""
        return cls.from_bounds(
            area.x_min - padding, area.x_max + padding, area.y_min - padding, area.y_max + padding, resolution
        )

    @property
    def x_max(self) -> float:
        return self.x_min + self.nx * self.resolution

    @property
    def y_max(self) -> float:
        return self.y_min + self.ny * self.resolution

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape ``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` for ``matplotlib.imshow(..., origin="lower")``."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell centre coordinates ``(xs (nx,), ys (ny,))``."""
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.resolution
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.resolution
        return xs, ys

    def cell_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integer cell indices of points and a mask of points that fall inside the grid."""
        ix = np.floor((np.asarray(x) - self.x_min) / self.resolution).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.y_min) / self.resolution).astype(np.int64)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, valid

    def area_mask(self, area: Area) -> np.ndarray:
        """Boolean ``(ny, nx)`` mask of cells whose centre lies inside ``area``."""
        xs, ys = self.cell_centers()
        mx = (xs >= area.x_min) & (xs <= area.x_max)
        my = (ys >= area.y_min) & (ys <= area.y_max)
        return my[:, None] & mx[None, :]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GridSpec:
        return cls(float(d["x_min"]), float(d["y_min"]), float(d["resolution"]), int(d["nx"]), int(d["ny"]))


"""
Elevation map from ray hits.
"""


def _clean_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts[np.all(np.isfinite(pts), axis=1)]


def elevation_map(points: np.ndarray, grid: GridSpec, reducer: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Bin 3D ray hits into a 2.5D elevation grid.

    Args:
        points: Hit points (N, 3) in the world frame. Non-finite rows (missed rays) are ignored.
        grid: Map grid.
        reducer: Per-cell statistic of the hit heights: "median", "mean", "max" or "min".

    Returns:
        ``(height (ny, nx) float32 with NaN for empty cells, count (ny, nx) int64)``.
    """
    if reducer not in REDUCERS:
        raise ValueError(f"reducer must be one of {REDUCERS}, got {reducer!r}")
    pts = _clean_points(points)
    height = np.full(grid.shape, np.nan, dtype=np.float32)
    count = np.zeros(grid.shape, dtype=np.int64)
    if len(pts) == 0:
        return height, count
    ix, iy, valid = grid.cell_index(pts[:, 0], pts[:, 1])
    cell = (iy * grid.nx + ix)[valid]
    z = pts[valid, 2]
    n_cells = grid.nx * grid.ny
    cnt = np.bincount(cell, minlength=n_cells)
    count[:] = cnt.reshape(grid.shape)
    hit = cnt > 0
    flat = np.full(n_cells, np.nan)
    if reducer == "mean":
        flat[hit] = np.bincount(cell, weights=z, minlength=n_cells)[hit] / cnt[hit]
    elif reducer == "max":
        acc = np.full(n_cells, -np.inf)
        np.maximum.at(acc, cell, z)
        flat[hit] = acc[hit]
    elif reducer == "min":
        acc = np.full(n_cells, np.inf)
        np.minimum.at(acc, cell, z)
        flat[hit] = acc[hit]
    else:  # median: sort by (cell, z) and average the middle element(s) of every group
        order = np.lexsort((z, cell))
        cs, zs = cell[order], z[order]
        uniq, start, n = np.unique(cs, return_index=True, return_counts=True)
        lo = start + (n - 1) // 2
        hi = start + n // 2
        flat[uniq] = 0.5 * (zs[lo] + zs[hi])
    height[:] = flat.reshape(grid.shape)
    return height, count


class ElevationMapper:
    """Accumulates ray hits batch by batch and builds the elevation map at the end.

    Hits are filtered to the grid when added (finite and in bounds), stored as float32, and reduced with
    :func:`elevation_map` in :meth:`finalize`. A running hit count is available at any time for coverage.
    """

    def __init__(self, grid: GridSpec):
        self.grid = grid
        self._chunks: list[np.ndarray] = []
        self.count = np.zeros(grid.shape, dtype=np.int64)
        self.num_points = 0

    def add(self, points: np.ndarray) -> int:
        """Add hit points (N, 3); returns how many were inside the grid."""
        pts = _clean_points(points)
        if len(pts) == 0:
            return 0
        ix, iy, valid = self.grid.cell_index(pts[:, 0], pts[:, 1])
        if not np.any(valid):
            return 0
        pts = pts[valid]
        np.add.at(self.count, (iy[valid], ix[valid]), 1)
        self._chunks.append(pts.astype(np.float32))
        self.num_points += len(pts)
        return len(pts)

    def points(self) -> np.ndarray:
        """All accumulated points (N, 3) float32."""
        if not self._chunks:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(self._chunks, axis=0)

    def finalize(self, reducer: str = "median") -> tuple[np.ndarray, np.ndarray]:
        """Elevation map and hit count, see :func:`elevation_map`."""
        return elevation_map(self.points(), self.grid, reducer)


"""
Ground truth from the terrain mesh.
"""


class MeshHeightQuery:
    """Top-surface height queries on a triangle mesh (a vertical ray cast from above), with a reusable index.

    Near-vertical triangles (zero area in the xy projection) are dropped: they never define the top surface.
    The remaining triangles are binned once into a uniform bucket grid over their xy bounding boxes; every
    query then only tests the triangles of its bucket (barycentric test, highest hit wins).
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, bucket_size: float | None = None):
        """Build the index.

        Args:
            vertices: Mesh vertices (V, 3) in the world frame.
            faces: Triangle vertex indices (F, 3).
            bucket_size: Bucket edge length [m]. Defaults to twice the median triangle bbox size.
        """
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        self.empty = len(f) == 0
        if self.empty:
            return
        tri = v[f]  # (T, 3, 3)
        x0, y0 = tri[:, 0, 0], tri[:, 0, 1]
        x1, y1 = tri[:, 1, 0], tri[:, 1, 1]
        x2, y2 = tri[:, 2, 0], tri[:, 2, 1]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)  # twice the signed projected area
        bb_min = tri[:, :, :2].min(axis=1)
        bb_max = tri[:, :, :2].max(axis=1)
        size = np.maximum(bb_max - bb_min, 0.0).max(axis=1)
        keep = np.abs(den) > 1e-12 * np.maximum(size, 1e-6) ** 2 + 1e-14
        self.empty = not np.any(keep)
        if self.empty:
            return
        tri, den, bb_min, bb_max, size = tri[keep], den[keep], bb_min[keep], bb_max[keep], size[keep]
        if bucket_size is None:
            bucket_size = float(max(2.0 * np.median(size), 1e-3))
        origin = bb_min.min(axis=0)
        nb = np.maximum(np.ceil((bb_max.max(axis=0) - origin) / bucket_size).astype(np.int64), 1) + 1
        b0 = np.floor((bb_min - origin) / bucket_size).astype(np.int64)
        b1 = np.floor((bb_max - origin) / bucket_size).astype(np.int64)
        nbx = b1[:, 0] - b0[:, 0] + 1
        nby = b1[:, 1] - b0[:, 1] + 1
        per_tri = nbx * nby
        # expand (triangle, bucket) pairs and sort them by bucket (CSR layout)
        tri_id = np.repeat(np.arange(len(tri)), per_tri)
        local = np.arange(per_tri.sum()) - np.repeat(np.cumsum(per_tri) - per_tri, per_tri)
        bx = b0[tri_id, 0] + local % nbx[tri_id]
        by = b0[tri_id, 1] + local // nbx[tri_id]
        bucket = bx * nb[1] + by
        order = np.argsort(bucket, kind="stable")
        bucket_sorted = bucket[order]
        n_buckets = int(nb[0] * nb[1])
        self._starts = np.searchsorted(bucket_sorted, np.arange(n_buckets))
        self._counts = np.diff(np.append(self._starts, len(bucket_sorted)))
        self._tri_sorted = tri_id[order]
        self._tri = tri
        self._den = den
        self._origin = origin
        self._nb = nb
        self.bucket_size = bucket_size

    def __call__(self, query_xy: np.ndarray, chunk_size: int = 65536) -> np.ndarray:
        """Heights (Q,) float64 at the query points (Q, 2+); NaN where no triangle covers the point."""
        q = np.asarray(query_xy, dtype=np.float64)
        q = q.reshape(len(q), -1)[:, :2] if q.size else q.reshape(0, 2)
        out = np.full(len(q), np.nan)
        if self.empty or len(q) == 0:
            return out
        tri, den, origin, nb, bs = self._tri, self._den, self._origin, self._nb, self.bucket_size
        for s in range(0, len(q), chunk_size):
            qq = q[s : s + chunk_size]
            qb = np.floor((qq - origin) / bs).astype(np.int64)
            inside = (qb[:, 0] >= 0) & (qb[:, 0] < nb[0]) & (qb[:, 1] >= 0) & (qb[:, 1] < nb[1])
            qid = np.nonzero(inside)[0]
            if len(qid) == 0:
                continue
            b = qb[qid, 0] * nb[1] + qb[qid, 1]
            c = self._counts[b]
            pair_q = np.repeat(qid, c)
            pair_off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
            pair_t = self._tri_sorted[np.repeat(self._starts[b], c) + pair_off]
            px, py = qq[pair_q, 0], qq[pair_q, 1]
            t = tri[pair_t]
            d = den[pair_t]
            l0 = ((t[:, 1, 1] - t[:, 2, 1]) * (px - t[:, 2, 0]) + (t[:, 2, 0] - t[:, 1, 0]) * (py - t[:, 2, 1])) / d
            l1 = ((t[:, 2, 1] - t[:, 0, 1]) * (px - t[:, 2, 0]) + (t[:, 0, 0] - t[:, 2, 0]) * (py - t[:, 2, 1])) / d
            l2 = 1.0 - l0 - l1
            tol = -1e-9
            hit = (l0 >= tol) & (l1 >= tol) & (l2 >= tol)
            if not np.any(hit):
                continue
            z = l0 * t[:, 0, 2] + l1 * t[:, 1, 2] + l2 * t[:, 2, 2]
            best = np.full(len(qq), -np.inf)
            np.maximum.at(best, pair_q[hit], z[hit])
            found = np.isfinite(best)
            out[s : s + chunk_size][found] = best[found]
        return out


def mesh_top_heights(
    vertices: np.ndarray,
    faces: np.ndarray,
    query_xy: np.ndarray,
    *,
    bucket_size: float | None = None,
    chunk_size: int = 65536,
) -> np.ndarray:
    """One-shot :class:`MeshHeightQuery` (build the index, query once). Reuse the class for repeated queries.

    Returns:
        Heights (Q,) float64, NaN where no triangle lies above/below the point.
    """
    return MeshHeightQuery(vertices, faces, bucket_size)(query_xy, chunk_size)


def ground_truth_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: GridSpec,
    supersample: int = 3,
    query: MeshHeightQuery | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth elevation of every grid cell from the terrain mesh.

    Each cell is sampled on a ``supersample x supersample`` sub-grid with :class:`MeshHeightQuery` (a dense
    vertical ray-cast pass); the cell value is the mean of the samples, comparable to a median/mean of many
    ray hits inside the cell.

    Returns:
        ``(gt_mean (ny, nx) float32, gt_range (ny, nx) float32)`` where ``gt_range`` (max - min of the sub-samples)
        flags cells that straddle a step or wall.
    """
    k = max(1, int(supersample))
    offs = (np.arange(k) + 0.5) / k * grid.resolution
    xs = grid.x_min + np.arange(grid.nx)[:, None] * grid.resolution + offs[None, :]  # (nx, k)
    ys = grid.y_min + np.arange(grid.ny)[:, None] * grid.resolution + offs[None, :]  # (ny, k)
    # sample order (iy, sy, ix, sx)
    yy, xx = np.meshgrid(ys.reshape(-1), xs.reshape(-1), indexing="ij")
    query = query if query is not None else MeshHeightQuery(vertices, faces)
    z = query(np.column_stack([xx.ravel(), yy.ravel()]))
    z = z.reshape(grid.ny, k, grid.nx, k)
    with warnings.catch_warnings():
        # all-NaN cells (outside the mesh) stay NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gt = np.nanmean(z, axis=(1, 3))
        rng = np.nanmax(z, axis=(1, 3)) - np.nanmin(z, axis=(1, 3))
    return gt.astype(np.float32), rng.astype(np.float32)


def heightfield_to_mesh(xs: np.ndarray, ys: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh of a height field sampled at ``xs`` (nx,) x ``ys`` (ny,), ``z`` shape (ny, nx)."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ny, nx = z.shape
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])
    iy, ix = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    a = (iy * nx + ix).ravel()
    b = a + 1
    c = a + nx
    d = c + 1
    faces = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    return vertices, faces


"""
Metrics.
"""


def map_error_stats(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Error statistics of an elevation map against ground truth over cells valid in both.

    Returns:
        Dict with ``rmse``, ``mae``, ``max_abs``, ``p95_abs``, ``bias`` [m] and ``n_cells``. Values are None
        when no cell can be compared.
    """
    est = np.asarray(estimate, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(est) & np.isfinite(gt)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    n = int(valid.sum())
    if n == 0:
        return {"rmse": None, "mae": None, "max_abs": None, "p95_abs": None, "bias": None, "n_cells": 0}
    err = est[valid] - gt[valid]
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "p95_abs": float(np.percentile(abs_err, 95)),
        "bias": float(np.mean(err)),
        "n_cells": n,
    }


def coverage_fraction(count_or_map: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of (masked) cells that received at least one hit (pass a count array or a NaN map)."""
    a = np.asarray(count_or_map)
    covered = np.isfinite(a) if np.issubdtype(a.dtype, np.floating) else a > 0
    if mask is None:
        return float(covered.mean())
    m = np.asarray(mask, dtype=bool)
    return float(covered[m].mean()) if m.any() else 0.0


def path_length(points: np.ndarray) -> float:
    """Length of a sampled trajectory or path (N, 3) [m]."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def tracking_stats(trajectory: np.ndarray, planned: np.ndarray) -> dict:
    """Distance between the flown trajectory and the planned polyline.

    Returns:
        ``cross_track_*``: horizontal distance of each flown sample to the closest point of the plan;
        ``distance3d_*``: the same in 3D (includes altitude error) [m].
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    plan = np.asarray(planned, dtype=np.float64)
    if len(traj) == 0 or len(plan) == 0:
        return {}
    d2, _ = distance_to_polyline(traj, plan, planar=True)
    d3, _ = distance_to_polyline(traj, plan, planar=False)
    return {
        "cross_track_mean": float(d2.mean()),
        "cross_track_rms": float(np.sqrt(np.mean(d2**2))),
        "cross_track_p95": float(np.percentile(d2, 95)),
        "cross_track_max": float(d2.max()),
        "distance3d_mean": float(d3.mean()),
        "distance3d_rms": float(np.sqrt(np.mean(d3**2))),
        "distance3d_max": float(d3.max()),
    }


def survey_map_metrics(
    elevation: np.ndarray,
    count: np.ndarray,
    gt: np.ndarray,
    gt_range: np.ndarray | None,
    grid: GridSpec,
    area: Area,
    step_threshold: float = 0.2,
) -> dict:
    """Coverage and map error inside the survey area (and away from steps/walls)."""
    in_area = grid.area_mask(area)
    stats = {
        "coverage_area": coverage_fraction(count, in_area),
        "coverage_grid": coverage_fraction(count),
        "error_area": map_error_stats(elevation, gt, in_area),
    }
    if gt_range is not None:
        smooth = in_area & np.isfinite(gt_range) & (gt_range <= step_threshold)
        stats["error_area_smooth"] = map_error_stats(elevation, gt, smooth)
        stats["smooth_cell_fraction"] = float(smooth.sum() / max(1, in_area.sum()))
        stats["step_threshold"] = step_threshold
    return stats
