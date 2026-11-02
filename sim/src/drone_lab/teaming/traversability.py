"""Traversability of a 2.5D elevation grid for a legged robot (numpy + scipy only).

The elevation grid comes from the drone (``elevation_map.npy``, NaN where nothing was seen) or from the terrain
ground truth. Three terrain features are computed per cell, in the spirit of the slope / step / roughness filters
used for legged robots (Wermelinger et al., "Navigation planning for legged robots in challenging terrain",
IROS 2016):

* **slope** -- inclination of a least-squares plane fitted to the cells within ``footprint_radius_m`` (about the
  size of the robot body). Reported in degrees.
* **roughness** -- RMS residual of that plane fit [m]: bumps the plane cannot explain.
* **step** -- the largest height jump to a cell within ``step_radius_m`` once the local slope is taken out [m].
  The local slope is the *minmod* of the one-sided differences (the smaller one when both have the same sign,
  zero otherwise), so on a smooth incline the step is ~0, while at the edge of a ledge the flat side's zero
  difference wins and the full ledge height is reported.

A cell is **lethal** when any feature exceeds its limit, when it is unknown (``unknown_is_lethal``), or when it
lies inside a known object footprint (crates, vehicles, people, ... pinned by the drone). Lethal cells are then
inflated by the robot radius (the robot centre must stay that far away), and a soft cost band beyond that keeps
the route away from edges when there is room. The traversal cost of a free cell is::

    1 + w_slope (slope/max_slope)^2 + w_step (step/max_step)^2 + w_roughness (roughness/max_roughness)^2
      + w_inflation (1 - (clearance - robot_radius) / inflation_buffer)^2      (inside the buffer only)

so flat open ground costs 1 per metre and the planner pays extra for every degree of slope and every centimetre
of step it takes. Lethal cells cost ``inf``. With ``w_cross_slope`` the cost also depends on the direction of
travel: a move pays ``w_cross_slope (cross_slope/max_slope)^2`` per metre for the slope *across* its direction
(:meth:`TraversabilityMap.cross_slope_cost`), so the planner prefers to go straight up or down steep ground and
to turn where it is flat.

Arrays are ``(ny, nx)`` with row ``iy`` = y index (the :class:`drone_lab.mapping.GridSpec` convention).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from drone_lab.mapping import GridSpec

__all__ = [
    "Footprint",
    "TraversabilityCfg",
    "TraversabilityMap",
    "compute_traversability",
    "fill_small_holes",
    "minmod",
    "plane_fit_features",
    "plane_fit_gradient",
    "points_in_polygon",
    "rasterize_footprints",
    "step_height",
]


@dataclass(frozen=True)
class TraversabilityCfg:
    """Limits and weights of the traversability analysis (defaults: a Go2-class quadruped)."""

    max_slope_deg: float = 22.0
    """Steepest plane slope the robot may walk on [deg] (the Go2 rough-terrain curriculum tops out at 0.4 rise/run,
    i.e. 21.8 deg; today's policy got about half-way up the curriculum)."""

    max_step_m: float = 0.12
    """Highest ledge the robot may step on or off [m] (the Go2 stairs curriculum tops out at 0.12 m)."""

    max_roughness_m: float = 0.05
    """Largest RMS bumpiness around the plane fit [m] (the Go2 random-rough curriculum tops out at 0.06 m)."""

    footprint_radius_m: float = 0.35
    """Radius of the plane-fit window for slope and roughness [m] (about half the body length)."""

    step_radius_m: float = 0.25
    """Radius of the step window [m] (at least one cell)."""

    robot_radius_m: float = 0.35
    """Clearance the robot centre needs from anything lethal [m]; lethal cells are inflated by it."""

    inflation_buffer_m: float = 0.8
    """Width of the soft-cost band beyond the lethal inflation [m]."""

    w_slope: float = 6.0
    """Cost weight of the slope term (at 15 deg a metre costs about 3.8 instead of 1)."""

    w_cross_slope: float = 0.0
    """Cost weight of the cross-slope term: the slope across the direction of travel, normalised like the slope
    term (0: the cost does not depend on the direction). With it, routes run straight up or down steep ground and
    turn where it is flat."""

    w_step: float = 4.0
    """Cost weight of the step term."""

    w_roughness: float = 3.0
    """Cost weight of the roughness term."""

    w_inflation: float = 3.0
    """Cost weight of the soft clearance band."""

    unknown_is_lethal: bool = True
    """Treat cells with no elevation (never seen by the drone) as obstacles."""

    unknown_cost: float = 6.0
    """Extra cost of an unknown cell when :attr:`unknown_is_lethal` is False."""

    fill_holes_iterations: int = 2
    """Isolated NaN cells with at least :attr:`fill_holes_min_neighbors` known 8-neighbours are filled with the
    neighbour mean, this many times (0 disables). Larger gaps stay unknown."""

    fill_holes_min_neighbors: int = 5
    """Known 8-neighbours needed to fill a hole cell."""

    min_valid_fraction: float = 0.6
    """Fraction of the plane-fit window that must be known; otherwise the cell is unknown."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Footprint:
    """Footprint of a known obstacle (a pinned crate, a vehicle, a person) [m, world frame].

    A circle of ``radius`` around ``(x, y)``, or, when ``polygon`` is given (e.g. the rotated box outline of a
    ground-truth object), that polygon; ``radius`` then only describes its size (e.g. its circumradius).
    """

    x: float
    y: float
    radius: float
    label: str = ""
    polygon: tuple[tuple[float, float], ...] | None = None


@dataclass
class TraversabilityMap:
    """Result of :func:`compute_traversability`; every array is ``(ny, nx)``."""

    grid: GridSpec
    cfg: TraversabilityCfg
    elevation: np.ndarray
    """Elevation after hole filling [m] (NaN = unknown)."""
    slope_deg: np.ndarray
    step: np.ndarray
    roughness: np.ndarray
    unknown: np.ndarray
    """No elevation, or too little of the plane-fit window known."""
    lethal_terrain: np.ndarray
    """Lethal because of the terrain (slope, step, roughness) or unknown (when unknown is lethal)."""
    lethal_objects: np.ndarray
    """Inside a known object footprint."""
    lethal: np.ndarray
    """Final obstacle mask: terrain or objects, inflated by the robot radius."""
    clearance: np.ndarray
    """Distance from the cell centre to the nearest terrain/object-lethal cell centre [m] (inf if none)."""
    cost: np.ndarray
    """Traversal cost per metre (>= 1), ``inf`` where lethal."""
    footprints: list[Footprint] = field(default_factory=list)
    gradient: np.ndarray | None = None
    """Plane-fit gradient ``(dz/dx, dz/dy)``, shape ``(2, ny, nx)`` (NaN = unknown)."""

    # ------------------------------------------------------------------------------------------- direction
    def _cross_term(self, gx, gy, ux, uy) -> np.ndarray:
        """Cross-slope cost for gradients ``(gx, gy)`` and unit directions ``(ux, uy)`` (broadcast)."""
        c = self.cfg
        across = np.abs(np.nan_to_num(gy, nan=0.0) * ux - np.nan_to_num(gx, nan=0.0) * uy)
        deg = np.degrees(np.arctan(across))
        return c.w_cross_slope * np.square(np.clip(deg / max(c.max_slope_deg, 1e-9), 0.0, 1.0))

    def cross_slope_cost(self, ux: float, uy: float) -> np.ndarray:
        """Extra cost per metre of moving in the direction ``(ux, uy)`` (unit vector) through each cell:
        ``w_cross_slope (cross_slope / max_slope)^2``, zero along the fall line; 0 where unknown or without
        :attr:`TraversabilityCfg.w_cross_slope`."""
        if self.cfg.w_cross_slope <= 0.0 or self.gradient is None:
            return np.zeros(self.grid.shape)
        return self._cross_term(self.gradient[0], self.gradient[1], float(ux), float(uy))

    def cross_slope_cost_at(self, xy: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """:meth:`cross_slope_cost` at world points ``xy`` (N, 2) (nearest cell) for unit ``directions`` (N, 2)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if self.cfg.w_cross_slope <= 0.0 or self.gradient is None:
            return np.zeros(len(xy))
        u = np.asarray(directions, dtype=float).reshape(-1, 2)
        gx = self.sample(self.gradient[0], xy)
        gy = self.sample(self.gradient[1], xy)
        return self._cross_term(gx, gy, u[:, 0], u[:, 1])

    # ---------------------------------------------------------------------------------------------- queries
    def world_to_cell(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell indices ``(iy, ix)`` of world points and a mask of points inside the grid."""
        ix, iy, valid = self.grid.cell_index(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return iy, ix, valid

    def cell_to_world(self, iy, ix) -> tuple[np.ndarray, np.ndarray]:
        """World coordinates of cell centres."""
        res = self.grid.resolution
        return (self.grid.x_min + (np.asarray(ix) + 0.5) * res, self.grid.y_min + (np.asarray(iy) + 0.5) * res)

    def sample(self, layer: np.ndarray, xy: np.ndarray, outside=np.nan) -> np.ndarray:
        """Nearest-cell value of ``layer`` at world points ``xy`` (N, 2); ``outside`` beyond the grid."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        iy, ix, valid = self.world_to_cell(xy[:, 0], xy[:, 1])
        out = np.full(len(xy), outside, dtype=float)
        out[valid] = layer[iy[valid], ix[valid]]
        return out

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear elevation at world points (N, 2) (NaN where unknown or outside the grid)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        res = self.grid.resolution
        fx = (xy[:, 0] - self.grid.x_min) / res - 0.5
        fy = (xy[:, 1] - self.grid.y_min) / res - 0.5
        x0 = np.clip(np.floor(fx).astype(int), 0, self.grid.nx - 1)
        y0 = np.clip(np.floor(fy).astype(int), 0, self.grid.ny - 1)
        x1 = np.clip(x0 + 1, 0, self.grid.nx - 1)
        y1 = np.clip(y0 + 1, 0, self.grid.ny - 1)
        tx = np.clip(fx - x0, 0.0, 1.0)
        ty = np.clip(fy - y0, 0.0, 1.0)
        z = self.elevation
        top = (1 - tx) * z[y0, x0] + tx * z[y0, x1]
        bot = (1 - tx) * z[y1, x0] + tx * z[y1, x1]
        out = (1 - ty) * top + ty * bot
        # fall back to the nearest cell where a bilinear corner is unknown
        bad = ~np.isfinite(out)
        if np.any(bad):
            out[bad] = self.sample(z, xy[bad])
        inside = (xy[:, 0] >= self.grid.x_min) & (xy[:, 0] < self.grid.x_max)
        inside &= (xy[:, 1] >= self.grid.y_min) & (xy[:, 1] < self.grid.y_max)
        out[~inside] = np.nan
        return out

    def free_fraction(self, mask: np.ndarray | None = None) -> float:
        """Fraction of (masked) cells that are not lethal."""
        m = np.ones_like(self.lethal) if mask is None else mask
        return float(np.mean(~self.lethal[m])) if np.any(m) else 0.0

    def summary(self) -> dict:
        """Counts of the lethal causes, for logs and the run summary."""
        n = self.lethal.size
        c = self.cfg
        return {
            "cells": int(n),
            "resolution_m": self.grid.resolution,
            "free_fraction": round(float(np.mean(~self.lethal)), 4),
            "lethal_slope_fraction": round(float(np.mean(self.slope_deg > c.max_slope_deg)), 4),
            "lethal_step_fraction": round(float(np.mean(self.step > c.max_step_m)), 4),
            "lethal_roughness_fraction": round(float(np.mean(self.roughness > c.max_roughness_m)), 4),
            "unknown_fraction": round(float(np.mean(self.unknown)), 4),
            "object_fraction": round(float(np.mean(self.lethal_objects)), 4),
            "num_footprints": len(self.footprints),
        }


# --------------------------------------------------------------------------------------------------- features
def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise minmod: the argument of smaller magnitude when both have the same sign, else 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a * b) > 0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def fill_small_holes(z: np.ndarray, iterations: int = 2, min_neighbors: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Fill isolated NaN cells with the mean of their known 8-neighbours.

    A NaN cell is filled when at least ``min_neighbors`` of its 8 neighbours are known; this repeats
    ``iterations`` times, so only small gaps (a missed ray here and there) close while large unseen areas stay
    unknown.

    Returns:
        ``(filled copy, mask of the cells that were filled)``.
    """
    out = np.array(z, dtype=float, copy=True)
    filled = np.zeros(out.shape, dtype=bool)
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0.0
    for _ in range(max(0, int(iterations))):
        known = np.isfinite(out)
        if known.all():
            break
        vals = np.where(known, out, 0.0)
        s = ndimage.correlate(vals, kernel, mode="constant", cval=0.0)
        n = ndimage.correlate(known.astype(float), kernel, mode="constant", cval=0.0)
        fill = (~known) & (n >= min_neighbors)
        if not fill.any():
            break
        out[fill] = s[fill] / n[fill]
        filled |= fill
    return out, filled


def _disk_offsets(radius_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = int(radius_cells)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (xx**2 + yy**2) <= (r + 0.5) ** 2
    return disk.astype(float), xx.astype(float), yy.astype(float)


def plane_fit_features(
    z: np.ndarray, resolution: float, radius_m: float, min_valid_fraction: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares plane over a disk window around every cell (NaN cells are left out of the fit).

    Returns:
        ``(slope (rise/run), roughness = RMS residual [m], valid)``; slope and roughness are NaN where fewer than
        ``min_valid_fraction`` of the window cells are known.
    """
    gradient, rough, valid = plane_fit_gradient(z, resolution, radius_m, min_valid_fraction)
    return np.hypot(gradient[0], gradient[1]), rough, valid


def plane_fit_gradient(
    z: np.ndarray, resolution: float, radius_m: float, min_valid_fraction: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`plane_fit_features` with the plane's gradient instead of its slope.

    Returns:
        ``(gradient (2, ny, nx) = (dz/dx, dz/dy), roughness = RMS residual [m], valid)``; NaN where fewer than
        ``min_valid_fraction`` of the window cells are known.
    """
    r = max(1, int(round(radius_m / resolution)))
    w, ox, oy = _disk_offsets(r)
    ox = ox * resolution
    oy = oy * resolution
    known = np.isfinite(z)
    m = known.astype(float)
    # centre the heights for conditioning (the fit is translation invariant in z)
    z0 = float(np.nanmean(z)) if known.any() else 0.0
    zz = np.where(known, z - z0, 0.0)

    def corr(img, kern):
        return ndimage.correlate(img, kern, mode="constant", cval=0.0)

    s0 = corr(m, w)
    sx = corr(m, w * ox)
    sy = corr(m, w * oy)
    sxx = corr(m, w * ox * ox)
    syy = corr(m, w * oy * oy)
    sxy = corr(m, w * ox * oy)
    sz = corr(zz, w)
    sxz = corr(zz, w * ox)
    syz = corr(zz, w * oy)
    szz = corr(zz * zz, w)

    valid = known & (s0 >= max(3.0, min_valid_fraction * w.sum()))
    shape = z.shape
    gradient = np.full((2, *shape), np.nan)
    rough = np.full(shape, np.nan)
    if valid.any():
        idx = np.nonzero(valid)
        a = np.empty((len(idx[0]), 3, 3))
        a[:, 0, 0], a[:, 0, 1], a[:, 0, 2] = s0[idx], sx[idx], sy[idx]
        a[:, 1, 0], a[:, 1, 1], a[:, 1, 2] = sx[idx], sxx[idx], sxy[idx]
        a[:, 2, 0], a[:, 2, 1], a[:, 2, 2] = sy[idx], sxy[idx], syy[idx]
        b = np.stack([sz[idx], sxz[idx], syz[idx]], axis=1)
        # a tiny relative ridge keeps collinear windows (e.g. a one-cell-wide strip of known cells) solvable
        diag = np.einsum("nii->ni", a)
        ridge = 1e-10 * np.maximum(diag, 1e-12)
        a_r = a + ridge[:, :, None] * np.eye(3)[None]
        theta = np.linalg.solve(a_r, b[..., None])[..., 0]
        # residual sum of squares of that plane, sum w (z - theta . phi)^2 = Szz - 2 theta.b + theta' A theta (exact
        # for any theta, so the ridge only enters at second order)
        rss = szz[idx] - 2.0 * np.einsum("ij,ij->i", theta, b) + np.einsum("ni,nij,nj->n", theta, a, theta)
        gradient[0][idx] = theta[:, 1]
        gradient[1][idx] = theta[:, 2]
        rough[idx] = np.sqrt(np.maximum(rss, 0.0) / s0[idx])
    return gradient, rough, valid


def step_height(z: np.ndarray, resolution: float, radius_m: float) -> np.ndarray:
    """Largest slope-corrected height jump from each cell to the cells within ``radius_m`` [m].

    ``step = max_j |z_j - z_i - g_i . (p_j - p_i)|`` over the square window of half-size
    ``max(1, round(radius_m / resolution))`` cells, where ``g_i`` is the minmod gradient at cell ``i``. A plane
    gives 0 everywhere; a ledge of height ``h`` gives ``h`` on the cells next to its edge. Unknown neighbours are
    ignored; cells that are unknown themselves get NaN.
    """
    ny, nx = z.shape
    zp = np.pad(np.asarray(z, dtype=float), 1, constant_values=np.nan)
    fx = zp[1:-1, 2:] - zp[1:-1, 1:-1]
    bx = zp[1:-1, 1:-1] - zp[1:-1, :-2]
    fy = zp[2:, 1:-1] - zp[1:-1, 1:-1]
    by = zp[1:-1, 1:-1] - zp[:-2, 1:-1]
    gx = minmod(np.nan_to_num(fx), np.nan_to_num(bx))  # per cell (height change per cell step)
    gy = minmod(np.nan_to_num(fy), np.nan_to_num(by))
    r = max(1, int(round(radius_m / resolution)))
    zr = np.pad(np.asarray(z, dtype=float), r, constant_values=np.nan)
    step = np.zeros((ny, nx))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            zn = zr[r + dy : r + dy + ny, r + dx : r + dx + nx]
            resid = np.abs(zn - (z + gx * dx + gy * dy))
            step = np.fmax(step, np.nan_to_num(resid, nan=0.0))
    step[~np.isfinite(z)] = np.nan
    return step


def points_in_polygon(px: np.ndarray, py: np.ndarray, polygon) -> np.ndarray:
    """Even-odd rule point-in-polygon test for arrays of points (broadcast shapes)."""
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    inside = np.zeros(np.broadcast(px, py).shape, dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= crosses & (px < x_cross)
        xj, yj = xi, yi
    return inside


def _rasterize_polygon(mask: np.ndarray, grid: GridSpec, polygon) -> None:
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    xs, ys = grid.cell_centers()
    res = grid.resolution
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    ix0 = max(0, int(math.floor((lo[0] - grid.x_min) / res)))
    ix1 = min(grid.nx, int(math.ceil((hi[0] - grid.x_min) / res)) + 1)
    iy0 = max(0, int(math.floor((lo[1] - grid.y_min) / res)))
    iy1 = min(grid.ny, int(math.ceil((hi[1] - grid.y_min) / res)) + 1)
    if ix0 < ix1 and iy0 < iy1:
        mask[iy0:iy1, ix0:ix1] |= points_in_polygon(xs[ix0:ix1][None, :], ys[iy0:iy1][:, None], poly)
    # thin or small polygons still block the cells their outline passes through
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=False):
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / (0.5 * res))) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = a + t * (b - a)
        ix, iy, ok = grid.cell_index(pts[:, 0], pts[:, 1])
        mask[iy[ok], ix[ok]] = True


def rasterize_footprints(grid: GridSpec, footprints) -> np.ndarray:
    """Boolean ``(ny, nx)`` mask of the cells covered by the footprints (circle: cell centre inside the circle;
    polygon: cell centre inside the polygon, plus every cell its outline passes through)."""
    mask = np.zeros(grid.shape, dtype=bool)
    xs, ys = grid.cell_centers()
    res = grid.resolution
    for fp in footprints:
        if getattr(fp, "polygon", None) is not None and len(fp.polygon) >= 3:
            _rasterize_polygon(mask, grid, fp.polygon)
            continue
        if not (np.isfinite(fp.x) and np.isfinite(fp.y)) or fp.radius <= 0:
            continue
        r = float(fp.radius)
        ix0 = max(0, int(math.floor((fp.x - r - grid.x_min) / res)))
        ix1 = min(grid.nx, int(math.ceil((fp.x + r - grid.x_min) / res)) + 1)
        iy0 = max(0, int(math.floor((fp.y - r - grid.y_min) / res)))
        iy1 = min(grid.ny, int(math.ceil((fp.y + r - grid.y_min) / res)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        dx = xs[ix0:ix1][None, :] - fp.x
        dy = ys[iy0:iy1][:, None] - fp.y
        inside = dx * dx + dy * dy <= r * r
        # a footprint smaller than a cell still blocks the cell it sits in
        if not inside.any():
            cx = int(np.clip(math.floor((fp.x - grid.x_min) / res), 0, grid.nx - 1))
            cy = int(np.clip(math.floor((fp.y - grid.y_min) / res), 0, grid.ny - 1))
            mask[cy, cx] = True
        mask[iy0:iy1, ix0:ix1] |= inside
    return mask


# ----------------------------------------------------------------------------------------------------- main
def compute_traversability(
    elevation: np.ndarray,
    grid: GridSpec,
    cfg: TraversabilityCfg | None = None,
    footprints=(),
) -> TraversabilityMap:
    """Slope / step / roughness costs, unknown cells and object footprints -> inflated cost map.

    Args:
        elevation: ``(ny, nx)`` heights [m], NaN where unknown.
        grid: Grid of ``elevation`` (cell size, origin).
        cfg: Limits and weights (default: :class:`TraversabilityCfg`).
        footprints: Known obstacles (:class:`Footprint`), e.g. from
            :func:`drone_lab.teaming.inputs.footprints_from_pins`.
    """
    cfg = cfg or TraversabilityCfg()
    z = np.asarray(elevation, dtype=float)
    if z.shape != grid.shape:
        raise ValueError(f"elevation shape {z.shape} does not match the grid {grid.shape}")
    res = grid.resolution
    z, _ = fill_small_holes(z, cfg.fill_holes_iterations, cfg.fill_holes_min_neighbors)
    gradient, rough, fit_valid = plane_fit_gradient(z, res, cfg.footprint_radius_m, cfg.min_valid_fraction)
    step = step_height(z, res, cfg.step_radius_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gradient[0], gradient[1])))

    unknown = ~np.isfinite(z) | ~fit_valid
    with np.errstate(invalid="ignore"):
        bad = (
            (np.nan_to_num(slope_deg, nan=0.0) > cfg.max_slope_deg)
            | (np.nan_to_num(step, nan=0.0) > cfg.max_step_m)
            | (np.nan_to_num(rough, nan=0.0) > cfg.max_roughness_m)
        )
    lethal_terrain = bad | (unknown if cfg.unknown_is_lethal else np.zeros_like(bad))
    footprints = list(footprints)
    lethal_objects = rasterize_footprints(grid, footprints)
    core = lethal_terrain | lethal_objects
    clearance = ndimage.distance_transform_edt(~core) * res if core.any() else np.full(z.shape, np.inf)
    lethal = core | (clearance < cfg.robot_radius_m)

    def term(v, limit):
        return np.square(np.clip(np.nan_to_num(v, nan=0.0) / max(limit, 1e-9), 0.0, 1.0))

    cost = 1.0 + cfg.w_slope * term(slope_deg, cfg.max_slope_deg)
    cost += cfg.w_step * term(step, cfg.max_step_m)
    cost += cfg.w_roughness * term(rough, cfg.max_roughness_m)
    if cfg.inflation_buffer_m > 0:
        band = 1.0 - (clearance - cfg.robot_radius_m) / cfg.inflation_buffer_m
        cost += cfg.w_inflation * np.square(np.clip(np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0))
    if not cfg.unknown_is_lethal:
        cost += np.where(unknown, cfg.unknown_cost, 0.0)
    cost[lethal] = np.inf
    return TraversabilityMap(
        grid=grid,
        cfg=cfg,
        elevation=z,
        slope_deg=slope_deg,
        step=step,
        roughness=rough,
        unknown=unknown,
        lethal_terrain=lethal_terrain,
        lethal_objects=lethal_objects,
        lethal=lethal,
        clearance=clearance,
        cost=cost,
        footprints=footprints,
        gradient=gradient,
    )
