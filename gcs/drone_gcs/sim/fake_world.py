"""Synthetic survey world: height field (hills, buildings, trees, objects), albedo/shading texture, a
downward grid ray-caster (like Isaac Lab's ``RayCaster`` + ``GridPatternCfg``) and a camera renderer that
ray-marches the height field (RGB + distance-to-image-plane depth).

numpy + OpenCV (``cv2.remap`` does the bilinear look-ups). Everything is deterministic for a given seed.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import PinholeCamera
from ..mapping import GridSpec, area_tuple

# Saturated colours that the terrain palette avoids, so the colour-blob stand-in detector can find them.
OBJECT_CLASSES: dict[str, dict] = {
    "car": {"color": (214, 38, 38), "size": (4.2, 1.9), "height": 1.5, "shape": "rect"},
    "tent": {"color": (36, 92, 222), "size": (2.6, 2.6), "height": 1.3, "shape": "pyramid"},
    "person": {"color": (210, 40, 190), "size": (0.7, 0.7), "height": 1.75, "shape": "disc"},
    "tarp": {"color": (238, 206, 30), "size": (3.0, 2.0), "height": 0.05, "shape": "rect"},
}
DEFAULT_COUNTS = {"car": 3, "tent": 2, "person": 3, "tarp": 2}


@dataclass
class FakeObject:
    id: int
    cls: str
    x: float
    y: float
    z: float  # ground height at the centre
    yaw: float
    size: tuple
    height: float
    shape: str

    def to_gt(self) -> dict:
        return {
            "id": self.id,
            "class": self.cls,
            "pos": [round(self.x, 4), round(self.y, 4), round(self.z, 4)],
            "size": [self.size[0], self.size[1], self.height],
            "yaw": round(self.yaw, 4),
        }


def _smooth_noise(rng: np.random.Generator, shape: tuple, cells: int) -> np.ndarray:
    """Smooth value noise in [0, 1] (bicubic upsampling of a ``cells`` x ``cells`` random grid)."""
    small = rng.random((cells, cells)).astype(np.float32)
    big = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(big.min()), float(big.max())
    return (big - lo) / max(hi - lo, 1e-6)


class FakeWorld:
    """Synthetic terrain over ``area`` (x_min, x_max, y_min, y_max) [m] plus a ``margin`` around it."""

    def __init__(
        self,
        area=(-20.0, 20.0, -20.0, 20.0),
        margin: float = 16.0,
        resolution: float = 0.05,
        seed: int = 0,
        n_hills: int = 7,
        n_buildings: int = 4,
        n_trees: int = 10,
        object_counts: dict | None = None,
        sun_azimuth_deg: float = 225.0,
        sun_elevation_deg: float = 50.0,
    ):
        self.area = area_tuple(area)
        x0, x1, y0, y1 = self.area
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.grid = GridSpec.from_bounds(x0 - margin, x1 + margin, y0 - margin, y1 + margin, resolution)
        g = self.grid
        xs, ys = g.cell_centers()
        X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        self._X, self._Y = X, Y

        # --- ground: gaussian hills + gentle slope
        ground = np.zeros(g.shape, np.float32)
        for k in range(n_hills):
            cx, cy = rng.uniform(x0 - 5, x1 + 5), rng.uniform(y0 - 5, y1 + 5)
            amp = rng.uniform(0.6, 2.2) * (1 if k % 4 else -0.4)
            sig = rng.uniform(4.0, 9.0)
            ground += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig * sig)).astype(np.float32)
        ang = rng.uniform(0, 2 * math.pi)
        ground += 0.015 * (np.cos(ang) * X + np.sin(ang) * Y)
        ground -= float(np.percentile(ground, 1))
        self.ground = ground
        height = ground.copy()

        # --- albedo: grass with low/high-frequency variation (hue kept away from the object colours)
        n1 = _smooth_noise(rng, g.shape, 9)[..., None]
        n2 = _smooth_noise(rng, g.shape, 37)[..., None]
        grass_a = np.array([84, 124, 58], np.float32)
        grass_b = np.array([118, 146, 74], np.float32)
        albedo = grass_a + (grass_b - grass_a) * (0.6 * n1 + 0.4 * n2)
        albedo += rng.normal(0, 5.0, g.shape + (1,)).astype(np.float32)

        # dirt road: a sinusoidal band through the area
        phase = float(rng.uniform(0, 3))

        def road_y(x):
            return 0.5 * (y0 + y1) + 0.25 * (y1 - y0) * np.sin((x - x0) / max(1.0, (x1 - x0)) * 2.4 + phase)

        self.road_y = road_y
        d_road = np.abs(Y - road_y(X))
        road = np.clip(1.6 - d_road, 0.0, 1.0)[..., None]
        dirt = np.array([166, 147, 116], np.float32) + rng.normal(0, 6.0, g.shape + (1,)).astype(np.float32)
        albedo = albedo * (1 - road) + dirt * road

        occupied: list[tuple[float, float, float]] = []  # (x, y, radius) of placed things

        def free(x, y, r, inside_margin=1.0, road_clearance=None):
            if not (x0 + r + inside_margin <= x <= x1 - r - inside_margin):
                return False
            if not (y0 + r + inside_margin <= y <= y1 - r - inside_margin):
                return False
            if road_clearance is not None and abs(y - float(road_y(x))) < r + road_clearance:
                return False
            return all(math.hypot(x - ox, y - oy) > r + orad + 0.8 for ox, oy, orad in occupied)

        def local_window(cx, cy, r):
            ix0, iy0, _ = g.cell_index(np.array([cx - r]), np.array([cy - r]))
            ix1, iy1, _ = g.cell_index(np.array([cx + r]), np.array([cy + r]))
            sx = slice(max(0, int(ix0[0])), min(g.nx, int(ix1[0]) + 2))
            sy = slice(max(0, int(iy0[0])), min(g.ny, int(iy1[0]) + 2))
            return sy, sx

        def footprint(cx, cy, yaw, sx, sy, shape):
            r = 0.5 * math.hypot(sx, sy) + 0.1
            wy, wx = local_window(cx, cy, r)
            lx = (X[wy, wx] - cx) * math.cos(yaw) + (Y[wy, wx] - cy) * math.sin(yaw)
            ly = -(X[wy, wx] - cx) * math.sin(yaw) + (Y[wy, wx] - cy) * math.cos(yaw)
            if shape == "disc":
                rr = np.hypot(lx, ly) / (0.5 * sx)
                return wy, wx, rr <= 1.0, np.clip(1 - rr, 0, 1)
            m = (np.abs(lx) <= sx / 2) & (np.abs(ly) <= sy / 2)
            prof = 1 - np.maximum(np.abs(lx) / (sx / 2), np.abs(ly) / (sy / 2))
            return wy, wx, m, np.clip(prof, 0, 1)

        # --- buildings (flat roofs, grey / slate / beige)
        roof_cols = [(132, 132, 136), (98, 106, 120), (172, 162, 142), (112, 112, 108)]
        self.buildings = []
        for _ in range(n_buildings):
            for _attempt in range(60):
                sx, sy = rng.uniform(5.0, 9.0), rng.uniform(4.0, 7.0)
                r = 0.5 * math.hypot(sx, sy)
                cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if free(cx, cy, r, 0.5, road_clearance=2.0):
                    break
            else:
                continue
            yaw = rng.uniform(-0.6, 0.6)
            h = rng.uniform(2.5, 5.0)
            wy, wx, m, prof = footprint(cx, cy, yaw, sx, sy, "rect")
            roof = float(ground[wy, wx][m].max()) + h
            height[wy, wx][m] = roof
            col = np.array(roof_cols[len(self.buildings) % len(roof_cols)], np.float32)
            edge = (prof < 0.08) & m
            a = albedo[wy, wx]
            a[m] = col + rng.normal(0, 3.0, (int(m.sum()), 3))
            a[edge] = col * 0.75
            occupied.append((cx, cy, r))
            self.buildings.append({"x": cx, "y": cy, "size": (sx, sy), "yaw": yaw, "roof_z": roof})

        # --- trees (rounded cones, dark canopy)
        self.trees = []
        for _ in range(n_trees):
            for _attempt in range(60):
                rad = rng.uniform(1.2, 2.2)
                cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if free(cx, cy, rad, -1.5, road_clearance=1.0):
                    break
            else:
                continue
            h = rng.uniform(3.0, 6.0)
            wy, wx, m, prof = footprint(cx, cy, 0.0, 2 * rad, 2 * rad, "disc")
            base = float(ground[wy, wx][m].mean())
            canopy = base + h * np.power(prof, 0.6)
            sub = height[wy, wx]
            sub[m] = np.maximum(sub[m], canopy[m])
            col = np.array([46, 92, 40], np.float32)
            a = albedo[wy, wx]
            a[m] = col * (0.75 + 0.35 * prof[m, None]) + rng.normal(0, 4.0, (int(m.sum()), 3))
            occupied.append((cx, cy, rad))
            self.trees.append({"x": cx, "y": cy, "radius": rad, "height": h})

        # --- survey objects
        counts = dict(DEFAULT_COUNTS if object_counts is None else object_counts)
        self.objects: list[FakeObject] = []
        oid = 1
        for cls, n in counts.items():
            spec = OBJECT_CLASSES[cls]
            for _ in range(int(n)):
                sx, sy = spec["size"]
                r = 0.5 * math.hypot(sx, sy)
                for _attempt in range(200):
                    cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
                    if free(cx, cy, r, 1.5):
                        break
                else:
                    continue
                yaw = rng.uniform(-math.pi, math.pi)
                wy, wx, m, prof = footprint(cx, cy, yaw, sx, sy, spec["shape"])
                gz = float(ground[wy, wx][m].mean()) if m.any() else 0.0
                col = np.array(spec["color"], np.float32)
                a = albedo[wy, wx]
                sub = height[wy, wx]
                if spec["shape"] == "pyramid":
                    sub[m] = np.maximum(sub[m], gz + spec["height"] * prof[m])
                    a[m] = col * (0.8 + 0.25 * prof[m, None])
                else:
                    sub[m] = np.maximum(sub[m], gz + spec["height"])
                    a[m] = col
                    if cls == "car":  # darker windscreen band, same hue
                        lx = (X[wy, wx] - cx) * math.cos(yaw) + (Y[wy, wx] - cy) * math.sin(yaw)
                        band = m & (lx > 0.15 * sx) & (lx < 0.32 * sx)
                        a[band] = col * 0.55
                occupied.append((cx, cy, r))
                self.objects.append(FakeObject(oid, cls, cx, cy, gz, yaw, (sx, sy), spec["height"], spec["shape"]))
                oid += 1

        self.height = height.astype(np.float32)
        self.hmin = float(self.height.min())
        self.hmax = float(self.height.max())

        # --- shading: hillshade + cast shadows from a fixed sun
        az, el = math.radians(sun_azimuth_deg), math.radians(sun_elevation_deg)
        sun = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], np.float32)
        gy_, gx_ = np.gradient(self.height, g.resolution)
        nrm = np.stack([-gx_, -gy_, np.ones_like(gx_)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        lam = np.clip(nrm @ sun, 0, 1)
        shadow = self._shadows(az, el)
        shade = (0.42 + 0.58 * lam) * (1.0 - 0.38 * shadow)
        self.texture = np.clip(albedo * shade[..., None], 0, 255).astype(np.uint8)
        self._ray_cache: dict = {}
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fake-render")

    # ------------------------------------------------------------------ helpers
    def _shadows(self, az: float, el: float) -> np.ndarray:
        """Soft cast-shadow mask in [0, 1] (1 = in shadow), by marching every texel towards the sun."""
        h = self.height
        res = self.grid.resolution
        dx, dy = math.cos(az), math.sin(az)
        tan_el = math.tan(el)
        n = int(math.ceil((self.hmax - self.hmin) / (res * tan_el))) + 1
        shadow = np.zeros(h.shape, bool)
        ny, nx = h.shape
        for k in range(1, min(n, 400) + 1, 2):
            ox, oy = int(round(k * dx)), int(round(k * dy))
            if abs(ox) >= nx or abs(oy) >= ny:
                break
            src = h[max(0, oy) : ny + min(0, oy), max(0, ox) : nx + min(0, ox)]
            dst = (slice(max(0, -oy), ny - max(0, oy)), slice(max(0, -ox), nx - max(0, ox)))
            dist = math.hypot(ox, oy) * res
            shadow[dst] |= src > h[dst] + dist * tan_el + 0.02
        return cv2.GaussianBlur(shadow.astype(np.float32), (5, 5), 1.2)

    def _map_coords(self, x, y):
        g = self.grid
        return ((x - g.x_min) / g.resolution - 0.5).astype(np.float32), ((y - g.y_min) / g.resolution - 0.5).astype(
            np.float32
        )

    def height_at(self, x, y) -> np.ndarray:
        """Bilinear terrain height at world (x, y) (same shape as x)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mx, my = self._map_coords(x.reshape(1, -1), y.reshape(1, -1))
        if mx.shape[1] > 30000:  # cv2.remap needs map dimensions < 32767
            n = mx.shape[1]
            w = int(math.ceil(n / 30000))
            pad = w * 30000 - n
            mx2 = np.pad(mx, ((0, 0), (0, pad))).reshape(w, 30000)
            my2 = np.pad(my, ((0, 0), (0, pad))).reshape(w, 30000)
            out = cv2.remap(self.height, mx2, my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return out.reshape(-1)[:n].reshape(x.shape).astype(np.float64)
        out = cv2.remap(self.height, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out.reshape(x.shape).astype(np.float64)

    def truth_on_grid(self, grid: GridSpec) -> np.ndarray:
        """Mean terrain height over each cell of ``grid`` (k x k samples per cell)."""
        k = max(1, int(math.ceil(grid.resolution / self.grid.resolution)))
        off = (np.arange(k) + 0.5) / k * grid.resolution
        xs = grid.x_min + np.arange(grid.nx)[:, None] * grid.resolution + off[None, :]
        ys = grid.y_min + np.arange(grid.ny)[:, None] * grid.resolution + off[None, :]
        X = np.broadcast_to(xs.reshape(1, 1, grid.nx, k), (grid.ny, k, grid.nx, k))
        Y = np.broadcast_to(ys.reshape(grid.ny, k, 1, 1), (grid.ny, k, grid.nx, k))
        h = self.height_at(X.reshape(-1), Y.reshape(-1)).reshape(grid.ny, k, grid.nx, k)
        return h.mean(axis=(1, 3)).astype(np.float32)

    def gt_objects(self) -> list[dict]:
        return [o.to_gt() for o in self.objects]

    # ------------------------------------------------------------------ sensors
    def scan_grid(self, origin, yaw: float, size=(5.0, 5.0), resolution: float = 0.25, noise: float = 0.0, rng=None):
        """Vertical rays on a yaw-aligned grid around ``origin`` (Isaac Lab ``GridPatternCfg``, ray_alignment
        "yaw"). Returns hits (N, 3) float32 and the valid mask (rays that start above the terrain)."""
        o = np.asarray(origin, dtype=np.float64)
        gx = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution)
        gy = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution)
        GX, GY = np.meshgrid(gx, gy)
        c, s = math.cos(yaw), math.sin(yaw)
        x = o[0] + c * GX.ravel() - s * GY.ravel()
        y = o[1] + s * GX.ravel() + c * GY.ravel()
        z = self.height_at(x, y)
        if noise > 0:
            rng = rng or np.random.default_rng()
            z = z + rng.normal(0, noise, z.shape)
        valid = z < o[2]
        pts = np.stack([x, y, np.where(valid, z, np.inf)], -1).astype(np.float32)
        return pts, valid

    def _camera_rays(self, cam: PinholeCamera) -> np.ndarray:
        key = (cam.width, cam.height, cam.pixel_center, *np.round(cam.K.ravel(), 6))
        rc = self._ray_cache.get(key)
        if rc is None:
            u, v = np.meshgrid(np.arange(cam.width) + cam.pixel_center, np.arange(cam.height) + cam.pixel_center)
            pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
            rc = (pix @ cam.K_inv.T).reshape(cam.height, cam.width, 3).astype(np.float32)
            self._ray_cache[key] = rc
        return rc

    def render(self, cam: PinholeCamera, step_xy: float = 0.2, bisect: int = 6) -> tuple[np.ndarray, np.ndarray]:
        """Render RGB (H, W, 3) uint8 and depth (H, W) float32 [m, distance to the image plane; inf = no hit]."""
        rays_c = self._camera_rays(cam)
        H = cam.height
        bands = [(r0, min(H, r0 + (H + 3) // 4)) for r0 in range(0, H, (H + 3) // 4)]
        futs = [self._pool.submit(self._render_band, cam, rays_c[r0:r1], step_xy, bisect) for r0, r1 in bands]
        parts = [f.result() for f in futs]
        rgb = np.concatenate([p[0] for p in parts], 0)
        depth = np.concatenate([p[1] for p in parts], 0)
        return rgb, depth

    def _render_band(self, cam, rays_c, step_xy, bisect):
        R = cam.R.astype(np.float32)
        o = cam.t.astype(np.float32)
        d = rays_c @ R.T  # world directions with camera z = 1, so t is the depth
        dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
        down = dz < -1e-6
        sdz = np.where(down, dz, -1.0).astype(np.float32)
        top = self.hmax + 0.05
        bot = self.hmin - 0.05
        t0 = np.maximum((top - o[2]) / sdz, 0.0).astype(np.float32)
        t1 = ((bot - o[2]) / sdz).astype(np.float32)
        t1 = np.where(down, t1, t0)
        span_xy = (t1 - t0) * np.hypot(dx, dy)
        n = int(np.clip(math.ceil(float(span_xy.max(initial=0.0)) / step_xy), 2, 64))
        hgt = self.height
        g = self.grid
        sx, sy = 1.0 / g.resolution, 1.0 / g.resolution
        ox_, oy_ = -g.x_min / g.resolution - 0.5, -g.y_min / g.resolution - 0.5

        def h_at(t):
            mx = (o[0] + t * dx) * sx + ox_
            my = (o[1] + t * dy) * sy + oy_
            return cv2.remap(
                hgt, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

        lo = t0.copy()
        hi = np.full_like(t0, np.inf)
        active = down.copy()
        prev = t0
        for k in range(1, n + 1):
            t = t0 + (t1 - t0) * (k / n)
            below = active & (o[2] + t * dz <= h_at(t))
            lo = np.where(below, prev, lo)
            hi = np.where(below, t, hi)
            active &= ~below
            prev = t
            if not active.any():
                break
        hit = np.isfinite(hi)
        lo_b, hi_b = lo.copy(), np.where(hit, hi, lo)
        for _ in range(bisect):
            mid = 0.5 * (lo_b + hi_b)
            below = o[2] + mid * dz <= h_at(mid)
            hi_b = np.where(hit & below, mid, hi_b)
            lo_b = np.where(hit & ~below, mid, lo_b)
        t_hit = np.where(hit, hi_b, np.inf).astype(np.float32)
        tt = np.where(hit, t_hit, 0.0)
        mx = ((o[0] + tt * dx) * sx + ox_).astype(np.float32)
        my = ((o[1] + tt * dy) * sy + oy_).astype(np.float32)
        rgb = cv2.remap(self.texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Height discontinuities (walls, roof edges): bilinear colour would blend e.g. a red car with green
        # grass into a yellow streak. There, take the colour of the neighbouring texel at the matching level:
        # the highest one if the hit is in the upper half of the step (wall/roof), else the lowest (ground).
        ny, nx = hgt.shape
        ix0 = np.clip(np.floor(mx).astype(np.int32), 0, nx - 2)
        iy0 = np.clip(np.floor(my).astype(np.int32), 0, ny - 2)
        hs = np.stack([hgt[iy0, ix0], hgt[iy0, ix0 + 1], hgt[iy0 + 1, ix0], hgt[iy0 + 1, ix0 + 1]], -1)
        hmax_, hmin_ = hs.max(-1), hs.min(-1)
        step = hit & (hmax_ - hmin_ > 0.15)
        if step.any():
            zhit = o[2] + tt * dz
            upper = zhit > 0.5 * (hmax_ + hmin_)
            pick = np.where(upper, hs.argmax(-1), hs.argmin(-1))[step]
            jx = ix0[step] + (pick % 2)
            jy = iy0[step] + (pick // 2)
            col = self.texture[jy, jx].astype(np.float32)
            wall = (zhit[step] < hmax_[step] - 0.05) & upper[step]
            col[wall] *= 0.72  # vertical faces are darker
            rgb[step] = col.astype(np.uint8)
        rgb[~hit] = (150, 175, 205)  # sky (never seen by a nadir camera)
        return rgb, t_hit

    def close(self):
        self._pool.shutdown(wait=False)
