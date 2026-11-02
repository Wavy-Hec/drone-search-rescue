"""Ground-control dashboard (OpenCV, BGR canvas).

Layout (default 1600 x 900)::

    +-- top bar: title, mission, link status, REC, keys, clock --------------------------------------+
    | CAMERA (feed + detection boxes)  | MAP: fog of war (unknown = dark grey), layer elevation | photo  |
    |                                  | | coverage, planned path (dashed), flown trail, drone, camera  |
    +----------------------------------+ footprint, waypoint, objects, scale bar, north arrow | legend  |
    | DETECTED OBJECTS table            |                                                               |
    +-- HUD: mission time, altitude AGL, speed, heading, waypoint, coverage, objects, link, detector --+

:class:`MapRenderer` is shared with the report (final maps).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import cv2
import numpy as np

from .mapping import GridSpec
from .state import CameraFrame, DroneState, GCSState
from .text import TextRenderer

LAYERS = ("elevation", "photo", "coverage")
STATS = ("mean", "max")


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# Tokyo Night palette (BGR)
PAL = {
    "bg": _hex("#16161e"),
    "panel": _hex("#1f2335"),
    "panel2": _hex("#24283b"),
    "border": _hex("#3b4261"),
    "fg": _hex("#c0caf5"),
    "fg2": _hex("#a9b1d6"),
    "dim": _hex("#737aa2"),
    "blue": _hex("#7aa2f7"),
    "cyan": _hex("#7dcfff"),
    "green": _hex("#9ece6a"),
    "orange": _hex("#ff9e64"),
    "red": _hex("#f7768e"),
    "yellow": _hex("#e0af68"),
    "magenta": _hex("#bb9af7"),
    "white": (245, 245, 245),
    "black": (18, 18, 18),
    "fog": (62, 60, 60),
    "fog_grid": (72, 70, 70),
    "photo_only": (160, 110, 88),
}

SAR_CLASSES = ("person", "vehicle", "crate", "barrel", "pallet", "cone", "landing_pad")
"""Classes of the search-and-rescue site (and of the fine-tuned detector), in legend order."""
CLASS_RGB = {
    # search-and-rescue site, the same colours as the simulator's label previews
    "person": (230, 25, 75),
    "vehicle": (245, 130, 48),
    "crate": (255, 225, 25),
    "barrel": (60, 180, 75),
    "pallet": (145, 30, 180),
    "cone": (70, 240, 240),
    "landing_pad": (0, 130, 200),
    # objects of the stand-in fake world
    "car": (214, 38, 38),
    "tent": (36, 92, 222),
    "tarp": (238, 206, 30),
}
_HASH_COLORS = [PAL["orange"], PAL["cyan"], PAL["green"], PAL["magenta"], PAL["yellow"], PAL["blue"], PAL["red"]]


def class_color(name: str) -> tuple[int, int, int]:
    if name in CLASS_RGB:
        r, g, b = CLASS_RGB[name]
        return (b, g, r)
    return _HASH_COLORS[sum(ord(c) for c in name) % len(_HASH_COLORS)]


def _lut(stops) -> np.ndarray:
    """(256, 3) BGR lookup table from RGB colour stops [(t, (r, g, b)), ...]."""
    t = np.linspace(0, 1, 256)
    ts = [s[0] for s in stops]
    out = np.zeros((256, 3), np.float32)
    for c in range(3):
        out[:, 2 - c] = np.interp(t, ts, [s[1][c] for s in stops])
    return out.astype(np.uint8)


HYPSO_LUT = _lut(
    [
        (0.00, (34, 102, 92)),
        (0.18, (64, 145, 96)),
        (0.38, (150, 188, 108)),
        (0.58, (226, 206, 132)),
        (0.78, (190, 128, 80)),
        (1.00, (246, 240, 232)),
    ]
)
HITS_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(256, 3)


def fmt_time(s: float) -> str:
    s = max(0.0, float(s))
    m, sec = divmod(s, 60)
    return f"{int(m):02d}:{sec:04.1f}"


def nice_length(max_m: float) -> float:
    """Largest 1-2-5 x 10^k length not above ``max_m``."""
    if max_m <= 0:
        return 1.0
    k = 10 ** math.floor(math.log10(max_m))
    for m in (5, 2, 1):
        if m * k <= max_m:
            return m * k
    return k


def dashed_polyline(img, pts, color, thickness=1, dash=9.0, gap=6.0) -> None:
    pts = np.asarray(pts, dtype=np.float64)
    on, remaining = True, dash
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        seg = b - a
        L = float(np.hypot(*seg))
        if L < 1e-6:
            continue
        u = seg / L
        pos = 0.0
        while pos < L - 1e-9:
            step = min(remaining, L - pos)
            if on:
                p0 = a + u * pos
                p1 = a + u * (pos + step)
                cv2.line(
                    img,
                    (int(round(p0[0])), int(round(p0[1]))),
                    (int(round(p1[0])), int(round(p1[1]))),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            pos += step
            remaining -= step
            if remaining <= 1e-9:
                on = not on
                remaining = dash if on else gap


def blend_poly(img, pts, color, alpha: float) -> None:
    x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.fillPoly(over, [np.round(pts - [x0, y0]).astype(np.int32)], color, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)


# ---------------------------------------------------------------------------------------------------- label placement
Box = tuple[int, int, int, int]
"""Pixel box (x0, y0, x1, y1); x1 and y1 are exclusive."""

MAP_LABEL_SPOTS = (
    (12, 0, "lm"),
    (-12, 0, "rm"),
    (12, -16, "lm"),
    (12, 16, "lm"),
    (-12, -16, "rm"),
    (-12, 16, "rm"),
    (12, -32, "lm"),
    (12, 32, "lm"),
    (-12, -32, "rm"),
    (-12, 32, "rm"),
    (12, -48, "lm"),
    (12, 48, "lm"),
    (-12, -48, "rm"),
    (-12, 48, "rm"),
)
"""Where the label of a map pin may go, nearest first: offset (px) of the text anchor from the pin, and the anchor."""


def overlap_area(a: Box, b: Box) -> int:
    """Number of pixels two boxes share."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def pick_spot(options: Sequence[Sequence[Box]], taken: Sequence[Box], size: tuple[int, int]) -> int:
    """Index of the first option (the boxes of one way to place a label) whose boxes lie inside an image of
    ``size`` (w, h) and clear of ``taken``; when every option is blocked, the one that hides the fewest pixels."""
    w, h = size
    for i, boxes in enumerate(options):
        if all(
            b[0] >= 0
            and b[1] >= 0
            and b[2] <= w
            and b[3] <= h
            and not any(b[0] < t[2] and t[0] < b[2] and b[1] < t[3] and t[1] < b[3] for t in taken)
            for b in boxes
        ):
            return i

    def hidden(b: Box) -> int:
        outside = (b[2] - b[0]) * (b[3] - b[1]) - overlap_area(b, (0, 0, w, h))
        return outside + sum(overlap_area(b, t) for t in taken)

    return min(range(len(options)), key=lambda i: sum(hidden(b) for b in options[i]))


def layout_map_labels(
    pins: Sequence[tuple[int, int]], sizes: Sequence[tuple[int, int]], taken: Sequence[Box], size: tuple[int, int]
) -> list[tuple[int, int, str, Box]]:
    """Spots for the labels of map pins, placed in order, each at the nearest of :data:`MAP_LABEL_SPOTS` that lies
    inside the map image of ``size`` (w, h) and clear of ``taken`` (markers, drones) and of the labels placed before
    it. ``sizes`` are the text sizes. Returns, per pin, the text position, its anchor and the label's box."""
    taken = list(taken)
    spots = []
    for (cx, cy), (lw, lh) in zip(pins, sizes, strict=True):
        options = []
        for dx, dy, anchor in MAP_LABEL_SPOTS:
            x0 = cx + dx - (lw if anchor == "rm" else 0) - 2  # the text background box (pad 2)
            y0 = cy + dy - lh // 2 - 1
            options.append((cx + dx, cy + dy, anchor, (x0, y0, x0 + lw + 4, y0 + lh + 2)))
        spot = options[pick_spot([(o[3],) for o in options], taken, size)]
        taken.append(spot[3])
        spots.append(spot)
    return spots


class MapView:
    """Fits a grid into a pixel rectangle (north up). ``to_px`` maps world (x, y) to local pixels."""

    def __init__(self, grid: GridSpec, w: int, h: int, pad: int = 6):
        self.grid = grid
        wm, hm = grid.nx * grid.resolution, grid.ny * grid.resolution
        self.ppm = min((w - 2 * pad) / wm, (h - 2 * pad) / hm)
        self.w = max(1, int(round(wm * self.ppm)))
        self.h = max(1, int(round(hm * self.ppm)))
        self.x0 = (w - self.w) // 2
        self.y0 = (h - self.h) // 2

    def local(self) -> MapView:
        """Same mapping, but relative to the map image itself (for drawing clipped to the map)."""
        v = object.__new__(MapView)
        v.grid, v.ppm, v.w, v.h, v.x0, v.y0 = self.grid, self.ppm, self.w, self.h, 0, 0
        return v

    def to_px(self, x, y):
        g = self.grid
        return self.x0 + (np.asarray(x) - g.x_min) * self.ppm, self.y0 + (g.y_max - np.asarray(y)) * self.ppm

    def pt(self, x, y) -> tuple[int, int]:
        px, py = self.to_px(x, y)
        return int(round(float(px))), int(round(float(py)))

    def pts(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        px, py = self.to_px(xy[:, 0], xy[:, 1])
        return np.stack([px, py], -1)


# ---------------------------------------------------------------------------------------------------- map renderer
class MapRenderer:
    """Renders the map layers and overlays into a region of an image."""

    REBUILD_FACTOR = 3.0  # a changed base layer is rebuilt at most every REBUILD_FACTOR x its build time ...
    REBUILD_MAX_S = 0.5  # ... but at least every REBUILD_MAX_S seconds (0 = rebuild on every change)

    def __init__(self, text: TextRenderer):
        self.text = text
        self._range: tuple[float, float] | None = None
        self._base_cache: dict = {}

    # ------------------------------------------------------------------ base layers (grid resolution, BGR)
    def elevation_range(self, state: GCSState, z: np.ndarray) -> tuple[float, float]:
        if state.z_range is not None:
            return state.z_range
        k = np.isfinite(z)
        if not k.any():
            return (0.0, 1.0)
        lo, hi = np.percentile(z[k], [1, 99.5])
        if hi - lo < 1.0:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.5, mid + 0.5
        if self._range is None:
            self._range = (float(lo), float(hi))
        else:  # smooth, so colours do not flicker while the map grows
            a = 0.15
            self._range = ((1 - a) * self._range[0] + a * lo, (1 - a) * self._range[1] + a * hi)
        return self._range

    def elevation_bgr(self, state: GCSState, stat: str = "mean") -> tuple[np.ndarray, tuple[float, float]]:
        elev = state.elevation
        z = elev.layer(stat)
        known = elev.known
        lo, hi = self.elevation_range(state, z)
        norm = np.clip((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
        col = HYPSO_LUT[(norm * 255).astype(np.uint8)].astype(np.float32)
        # hillshade (light from the north-west), only where all 4 neighbours are known
        zm = elev.mean()
        zf = np.where(known, zm, np.nanmean(zm) if known.any() else 0.0)
        gy, gx = np.gradient(zf.astype(np.float64), elev.grid.resolution)
        az, alt = math.radians(135.0), math.radians(45.0)  # azimuth from +x CCW: north-west
        lx, ly, lz = math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)
        shade = (-gx * lx - gy * ly + lz) / np.sqrt(gx * gx + gy * gy + 1.0)
        inner = known.copy()
        inner[1:, :] &= known[:-1, :]
        inner[:-1, :] &= known[1:, :]
        inner[:, 1:] &= known[:, :-1]
        inner[:, :-1] &= known[:, 1:]
        shade = np.where(inner, np.clip(shade, 0, 1), lz)
        col *= (0.55 + 0.6 * shade)[..., None]
        img = np.clip(col, 0, 255).astype(np.uint8)
        img[~known] = PAL["fog"]
        return img, (lo, hi)

    def photo_bgr(self, state: GCSState) -> np.ndarray:
        m = state.mosaic
        img = np.ascontiguousarray(m.rgb[..., ::-1])
        img[~m.filled] = PAL["fog"]
        return img

    def coverage_bgr(self, state: GCSState) -> np.ndarray:
        elev, mos = state.elevation, state.mosaic
        cnt = elev.count.astype(np.float32)
        k = cnt > 0
        img = np.empty(elev.grid.shape + (3,), np.uint8)
        img[:] = PAL["fog"]
        # photo coverage on the elevation grid: any photo sub-cell filled
        f = mos.grid.nx // elev.grid.nx
        photo = mos.filled.reshape(elev.grid.ny, f, elev.grid.nx, f).any(axis=(1, 3)) if f >= 1 else mos.filled
        img[photo & ~k] = PAL["photo_only"]
        if k.any():
            lv = np.log1p(cnt[k])
            top = max(np.log1p(64.0), float(lv.max()))
            img[k] = HITS_LUT[np.clip(40 + 215 * lv / top, 0, 255).astype(np.uint8)]
        return img

    def base(self, state: GCSState, layer: str, stat: str, w: int, h: int) -> tuple[np.ndarray, tuple | None]:
        """Layer image resized to (w, h), north up; cached per map version.

        When only the map data changed, the layer is rebuilt at most every ``3 x`` its own build time (capped at
        0.5 s): a 40 m map (~6 ms) still updates on every dashboard frame, a 250 m one (~60 ms) about 5x/s
        instead of slowing the whole loop down."""
        ver = (layer, stat, w, h, state.elevation.version, state.mosaic.version)
        hit = self._base_cache.get(layer)
        if hit is not None:
            if hit[0] == ver:
                return hit[1], hit[2]
            if hit[0][:4] == ver[:4] and time.monotonic() - hit[3] < hit[4]:
                return hit[1], hit[2]
        t_build = time.monotonic()
        rng = None
        if layer == "photo":
            img = self.photo_bgr(state)
            interp = cv2.INTER_AREA if img.shape[1] > w else cv2.INTER_LINEAR
        elif layer == "coverage":
            img, interp = self.coverage_bgr(state), cv2.INTER_NEAREST
        else:
            img, rng = self.elevation_bgr(state, stat)
            interp = cv2.INTER_NEAREST if w >= img.shape[1] else cv2.INTER_AREA
        out = cv2.resize(np.ascontiguousarray(img[::-1]), (w, h), interpolation=interp)
        now = time.monotonic()
        self._base_cache[layer] = (ver, out, rng, now, min(self.REBUILD_MAX_S, self.REBUILD_FACTOR * (now - t_build)))
        return out, rng

    # ------------------------------------------------------------------ full map
    def draw(
        self,
        img: np.ndarray,
        state: GCSState,
        layer: str = "elevation",
        stat: str = "mean",
        show_gt: bool = False,
        active_drone: int | None = None,
        legend_w: int = 170,
        report: bool = False,
    ) -> MapView | None:
        """Draw the map into ``img`` (a region of the canvas, modified in place)."""
        H, W = img.shape[:2]
        txt = self.text
        if state.grid is None:
            txt.put(img, "waiting for MISSION / telemetry ...", W // 2, H // 2, 18, PAL["dim"], anchor="mm")
            return None
        mw = W - legend_w
        view = MapView(state.grid, mw, H)
        base, rng = self.base(state, layer, stat, view.w, view.h)
        img[view.y0 : view.y0 + view.h, view.x0 : view.x0 + view.w] = base
        mimg = img[view.y0 : view.y0 + view.h, view.x0 : view.x0 + view.w]  # overlays are clipped to the map
        lv = view.local()
        self._grid_lines(mimg, lv, state)
        self._overlays(mimg, lv, state, show_gt, active_drone, report)
        self._scale_and_north(mimg, lv)
        cv2.rectangle(img, (view.x0 - 1, view.y0 - 1), (view.x0 + view.w, view.y0 + view.h), PAL["border"], 1)
        if legend_w > 0:
            self._legend(img[:, mw:], state, layer, stat, rng, report)
        return view

    def _grid_lines(self, img, view: MapView, state: GCSState) -> None:
        g = state.grid
        span = max(g.nx, g.ny) * g.resolution
        step = nice_length(span / 6)
        x0, y0 = view.x0, view.y0
        x1, y1 = view.x0 + view.w - 1, view.y0 + view.h - 1
        roi = img[y0 : y1 + 1, x0 : x1 + 1]
        lines = roi.copy()
        for x in np.arange(math.ceil(g.x_min / step) * step, g.x_max, step):
            px = int(round(view.to_px(x, 0)[0])) - x0
            cv2.line(lines, (px, 0), (px, view.h - 1), (255, 255, 255), 1)
        for y in np.arange(math.ceil(g.y_min / step) * step, g.y_max, step):
            py = int(round(view.to_px(0, y)[1])) - y0
            cv2.line(lines, (0, py), (view.w - 1, py), (255, 255, 255), 1)
        cv2.addWeighted(lines, 0.07, roi, 0.93, 0, dst=roi)

    def _overlays(self, img, view: MapView, state: GCSState, show_gt: bool, active_drone, report: bool) -> None:
        txt = self.text
        aa = cv2.LINE_AA
        # mission area
        if state.area is not None:
            ax0, ax1, ay0, ay1 = state.area
            p0, p1 = view.pt(ax0, ay1), view.pt(ax1, ay0)
            cv2.rectangle(img, p0, p1, PAL["fg2"], 1, aa)
            txt.put(
                img, "survey area", p0[0] + 4, p0[1] + 3, 12, PAL["fg2"], "sans", bg=PAL["black"], pad=2, bg_alpha=0.45
            )
        # planned path (dashed) + waypoint markers
        if state.planned_path is not None and len(state.planned_path) > 1:
            pp = view.pts(state.planned_path[:, :2])
            dashed_polyline(img, pp, (0, 0, 0), 3, 9, 6)
            dashed_polyline(img, pp, PAL["cyan"], 1, 9, 6)
            for p in pp:
                cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 2, PAL["cyan"], -1, aa)
        # camera footprints
        for d in state.drones.values():
            fr = d.camera
            if fr is not None and not report and state.elevation is not None:
                P = state.footprint(fr)
                if P is not None:
                    poly = view.pts(P[:, :2])
                    blend_poly(img, poly, PAL["cyan"], 0.18)
                    cv2.polylines(img, [np.round(poly).astype(np.int32)], True, PAL["cyan"], 1, aa)
        # trails
        for d in state.drones.values():
            tr = d.trail_array()
            if len(tr) > 1:
                pts = np.round(view.pts(tr[:, 1:3])).astype(np.int32)
                cv2.polylines(img, [pts], False, (0, 0, 0), 4, aa)
                cv2.polylines(img, [pts], False, PAL["orange"], 2, aa)
        # current waypoint + carrot, while the drone flies
        for d in state.drones.values():
            if report or state.ended or d.phase == "landed":
                continue
            if state.planned_path is not None and d.wp_index is not None and 0 <= d.wp_index < len(state.planned_path):
                wp = state.planned_path[int(d.wp_index)]
                c = view.pt(wp[0], wp[1])
                cv2.circle(img, c, 9, (0, 0, 0), 3, aa)
                cv2.circle(img, c, 9, PAL["yellow"], 2, aa)
                txt.put(
                    img,
                    f"WP {int(d.wp_index) + 1}",
                    c[0] + 12,
                    c[1],
                    12,
                    PAL["yellow"],
                    "medium",
                    anchor="lm",
                    bg=PAL["black"],
                    pad=2,
                    bg_alpha=0.55,
                )
            if d.carrot is not None:
                c = view.pt(d.carrot[0], d.carrot[1])
                cv2.drawMarker(img, c, PAL["magenta"], cv2.MARKER_CROSS, 9, 2, aa)
        # landing pad
        if state.home is not None:
            c = view.pt(state.home[0], state.home[1])
            cv2.circle(img, c, 11, (0, 0, 0), 3, aa)
            cv2.circle(img, c, 11, PAL["white"], 1, aa)
            txt.put(img, "H", c[0], c[1], 13, PAL["white"], "bold", anchor="mm")
        # ground truth (evaluation)
        if show_gt and state.gt_objects:
            for o in state.gt_objects:
                p = o.get("pos")
                if p is None:
                    continue
                c = view.pt(p[0], p[1])
                cv2.rectangle(img, (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), PAL["white"], 1, aa)
        # objects: markers first, then the labels where they hide no marker, no other label and neither the scale bar
        # nor the north arrow. Once the mission has ended the drones stay put, so the labels keep clear of them too
        # (a landed drone sits on the landing pad's pin).
        taken = self._furniture_boxes(view)
        if report or state.ended:
            r = 22  # the drone icon's ring: 17 px, 20 px in the report
            for d in state.drones.values():
                if d.pos is not None:
                    x, y = view.pt(d.pos[0], d.pos[1])
                    taken.append((x - r, y - r, x + r, y + r))
        labels: list[tuple[str, tuple[int, int]]] = []
        for ob in sorted(state.tracker.objects, key=lambda o: o.id):
            c = view.pt(ob.pos[0], ob.pos[1])
            col = class_color(ob.cls)
            if ob.hits >= state.tracker.min_hits:
                cv2.circle(img, c, 8, (0, 0, 0), -1, aa)
                cv2.circle(img, c, 6, col, -1, aa)
                cv2.circle(img, c, 8, PAL["white"], 1, aa)
                taken.append((c[0] - 9, c[1] - 9, c[0] + 9, c[1] + 9))
                labels.append((f"{ob.cls} {ob.conf:.2f}", c))
            else:
                cv2.circle(img, c, 6, col, 2, aa)
        sizes = [txt.size(label, 12, "medium") for label, _ in labels]
        spots = layout_map_labels([c for _, c in labels], sizes, taken, (img.shape[1], img.shape[0]))
        for (label, c), (x, y, anchor, _) in zip(labels, spots, strict=True):
            if y != c[1]:
                cv2.line(img, c, (x, y), PAL["white"], 1, aa)
            txt.put(img, label, x, y, 12, PAL["white"], "medium", anchor=anchor, bg=(20, 20, 20), pad=2, bg_alpha=0.6)
        # drones
        for d in state.drones.values():
            if d.pos is None:
                continue
            c = np.array(view.to_px(d.pos[0], d.pos[1]))
            f = np.array([math.cos(d.yaw), -math.sin(d.yaw)])
            l_ = np.array([-f[1], f[0]])
            s = 1.0 if not report else 1.2
            poly = np.array([c + f * 16 * s, c - f * 9 * s + l_ * 10 * s, c - f * 4 * s, c - f * 9 * s - l_ * 10 * s])
            col = PAL["white"] if active_drone is None or d.drone_id == active_drone else PAL["fg2"]
            cv2.circle(img, (int(c[0]), int(c[1])), int(17 * s), (0, 0, 0), 2, aa)
            cv2.fillPoly(img, [np.round(poly).astype(np.int32)], col, aa)
            cv2.polylines(img, [np.round(poly).astype(np.int32)], True, (0, 0, 0), 1, aa)
            if len(state.drones) > 1:
                txt.put(img, f"d{d.drone_id}", c[0] + 14, c[1] - 14, 12, col, "medium")

    @staticmethod
    def _furniture(view: MapView) -> tuple[float, int, tuple[int, int], tuple[int, int]]:
        """Placement of the scale bar (bottom left: its length in m and px, the left end of the bar) and of the north
        arrow (top right: its top centre)."""
        length = nice_length(150 / view.ppm)
        bar = (view.x0 + 14, view.y0 + view.h - 18)
        return length, int(round(length * view.ppm)), bar, (view.x0 + view.w - 24, view.y0 + 18)

    def _furniture_boxes(self, view: MapView) -> list[Box]:
        """Pixels the scale bar and the north arrow cover; they are drawn last, so the labels keep clear of them."""
        _, px, (x0, y0), (nx, ny) = self._furniture(view)
        return [(x0 - 8, y0 - 22, x0 + px + 11, y0 + 10), (nx - 18, ny - 15, nx + 18, ny + 32)]

    def _scale_and_north(self, img, view: MapView) -> None:
        txt = self.text
        L, px, (x0, y0), (nx, ny) = self._furniture(view)
        cv2.rectangle(img, (x0 - 8, y0 - 22), (x0 + px + 10, y0 + 9), (20, 20, 20), -1)
        cv2.line(img, (x0, y0), (x0 + px, y0), PAL["white"], 2)
        for xx in (x0, x0 + px // 2, x0 + px):
            cv2.line(img, (xx, y0 - 5), (xx, y0 + 4), PAL["white"], 1)
        label = f"{L:g} m"
        txt.put(img, label, x0 + px // 2, y0 - 6, 12, PAL["white"], "medium", anchor="mb")
        cv2.circle(img, (nx, ny + 14), 17, (20, 20, 20), -1, cv2.LINE_AA)
        pts = np.array([[nx, ny + 2], [nx - 7, ny + 24], [nx, ny + 18], [nx + 7, ny + 24]], np.int32)
        cv2.fillPoly(img, [pts], PAL["white"], cv2.LINE_AA)
        txt.put(img, "N", nx, ny - 1, 12, PAL["white"], "bold", anchor="mb")

    def _legend(self, img, state: GCSState, layer: str, stat: str, rng, report: bool) -> None:
        txt = self.text
        H, W = img.shape[:2]
        x = 10
        y = 8
        title = {"elevation": f"ELEVATION ({stat})", "photo": "ORTHOMOSAIC", "coverage": "COVERAGE"}[layer]
        txt.put(img, title, x, y, 13, PAL["fg"], "bold")
        y += 24
        if layer == "elevation" and rng is not None:
            bar_h, bar_w = 150, 16
            bar = HYPSO_LUT[np.linspace(255, 0, bar_h).astype(np.uint8)][:, None, :].repeat(bar_w, axis=1)
            img[y : y + bar_h, x : x + bar_w] = bar
            cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), PAL["border"], 1)
            lo, hi = rng
            for frac, val in ((0.0, hi), (0.5, 0.5 * (lo + hi)), (1.0, lo)):
                yy = int(y + frac * (bar_h - 1))
                cv2.line(img, (x + bar_w, yy), (x + bar_w + 4, yy), PAL["fg2"], 1)
                txt.put(img, f"{val:+.1f} m", x + bar_w + 8, yy, 12, PAL["fg2"], "mono", anchor="lm")
            y += bar_h + 10
            sw = PAL["fog"]
            cv2.rectangle(img, (x, y), (x + 16, y + 12), sw, -1)
            txt.put(img, "unknown", x + 24, y + 6, 12, PAL["fg2"], anchor="lm")
            y += 22
        elif layer == "coverage":
            for col, label in (
                (tuple(int(c) for c in HITS_LUT[250]), "many ray hits"),
                (tuple(int(c) for c in HITS_LUT[60]), "few ray hits"),
                (PAL["photo_only"], "photo only"),
                (PAL["fog"], "unknown"),
            ):
                cv2.rectangle(img, (x, y), (x + 16, y + 12), col, -1)
                txt.put(img, label, x + 24, y + 6, 12, PAL["fg2"], anchor="lm")
                y += 20
            y += 4
        else:
            mode = state.mosaic.mode if state.mosaic is not None else "-"
            txt.put(img, "camera pixels on", x, y, 12, PAL["fg2"])
            txt.put(img, f"the map ({mode} view)", x, y + 15, 12, PAL["fg2"])
            y += 36
            cv2.rectangle(img, (x, y), (x + 16, y + 12), PAL["fog"], -1)
            txt.put(img, "unknown", x + 24, y + 6, 12, PAL["fg2"], anchor="lm")
            y += 22
        # key
        y += 6
        cv2.line(img, (x, y), (W - 10, y), PAL["border"], 1)
        y += 10
        items = [("path", "planned path"), ("trail", "flown trail"), ("drone", "drone + heading")]
        if not report:
            items += [("wp", "current waypoint"), ("fov", "camera footprint")]
        items += [("obj", "object (confirmed)"), ("tent", "tentative")]
        if state.gt_objects:
            items.append(("gt", "ground truth (g)"))
        for kind, label in items:
            cy = y + 7
            if kind == "path":
                dashed_polyline(img, [(x, cy), (x + 22, cy)], PAL["cyan"], 1, 5, 3)
            elif kind == "trail":
                cv2.line(img, (x, cy), (x + 22, cy), PAL["orange"], 2, cv2.LINE_AA)
            elif kind == "drone":
                pts = np.array([[x + 22, cy], [x + 6, cy - 6], [x + 10, cy], [x + 6, cy + 6]], np.int32)
                cv2.fillPoly(img, [pts], PAL["white"], cv2.LINE_AA)
            elif kind == "wp":
                cv2.circle(img, (x + 11, cy), 6, PAL["yellow"], 2, cv2.LINE_AA)
            elif kind == "fov":
                cv2.rectangle(img, (x + 3, cy - 6), (x + 19, cy + 6), PAL["cyan"], 1)
            elif kind == "obj":
                cv2.circle(img, (x + 11, cy), 6, PAL["red"], -1, cv2.LINE_AA)
                cv2.circle(img, (x + 11, cy), 7, PAL["white"], 1, cv2.LINE_AA)
            elif kind == "tent":
                cv2.circle(img, (x + 11, cy), 5, PAL["red"], 2, cv2.LINE_AA)
            elif kind == "gt":
                cv2.rectangle(img, (x + 5, cy - 6), (x + 17, cy + 6), PAL["white"], 1)
            txt.put(img, label, x + 30, cy, 12, PAL["fg2"], anchor="lm")
            y += 19
        # confirmed objects per class: the site's classes first, people on top
        counts: dict[str, int] = {}
        for o in state.tracker.confirmed():
            counts[o.cls] = counts.get(o.cls, 0) + 1
        classes = sorted(counts, key=lambda c: (SAR_CLASSES.index(c) if c in SAR_CLASSES else len(SAR_CLASSES), c))
        if classes:
            y += 6
            cv2.line(img, (x, y), (W - 10, y), PAL["border"], 1)
            y += 10
            for c in classes[:9]:
                cv2.circle(img, (x + 11, y + 7), 6, class_color(c), -1, cv2.LINE_AA)
                bold = c == "person"
                col = PAL["fg"] if bold else PAL["fg2"]
                txt.put(img, c.replace("_", " "), x + 30, y + 7, 12, col, "medium" if bold else "sans", anchor="lm")
                txt.put(img, str(counts[c]), W - 14, y + 7, 12, col, "mono", anchor="rm")
                y += 19
        # stats at the bottom
        if state.elevation is not None:
            ce, cp = state.coverage()
            g = state.grid
            lines = [
                f"grid {g.nx}x{g.ny} @ {g.resolution:g} m",
                f"scan cover {100 * ce:5.1f} %",
                f"photo cover {100 * cp:5.1f} %",
                f"hits {state.elevation.n_points:,}",
            ]
            if state.auto_grid:
                lines.insert(0, "AUTO GRID (no MISSION)")
            yy = H - 10 - 16 * len(lines)
            for s in lines:
                txt.put(img, s, x, yy, 12, PAL["dim"], "mono")
                yy += 16


# ---------------------------------------------------------------------------------------------------- dashboard
class Dashboard:
    """Composes the full canvas. ``render(state, ui)`` returns a BGR image of ``size``."""

    def __init__(self, size=(1600, 900)):
        self.W, self.H = int(size[0]), int(size[1])
        self.text = TextRenderer()
        self.maps = MapRenderer(self.text)
        m = 12
        self.m = m
        self.top_h = 44
        self.hud_h = 78
        cam_w = int(min(640, 0.42 * (self.W - 3 * m)))
        cam_h = int(cam_w * 3 / 4)
        y0 = self.top_h + m // 2
        self.r_cam = (m, y0, cam_w, 26 + cam_h)
        oy = y0 + 26 + cam_h + m
        hud_y = self.H - self.hud_h - m // 2
        self.r_obj = (m, oy, cam_w, hud_y - m - oy)
        mx = 2 * m + cam_w
        self.r_map = (mx, y0, self.W - mx - m, hud_y - m - y0)
        self.r_hud = (m, hud_y, self.W - 2 * m, self.hud_h)
        self._bg = self._background()

    def _background(self) -> np.ndarray:
        bg = np.empty((self.H, self.W, 3), np.uint8)
        bg[:] = PAL["bg"]
        cv2.rectangle(bg, (0, 0), (self.W, self.top_h), PAL["panel"], -1)
        cv2.line(bg, (0, self.top_h), (self.W, self.top_h), PAL["border"], 1)
        for r in (self.r_cam, self.r_obj, self.r_map, self.r_hud):
            x, y, w, h = r
            cv2.rectangle(bg, (x, y), (x + w, y + h), PAL["panel"], -1)
            cv2.rectangle(bg, (x, y), (x + w, y + h), PAL["border"], 1)
        for r in (self.r_cam, self.r_obj, self.r_map):
            x, y, w, h = r
            cv2.rectangle(bg, (x + 1, y + 1), (x + w - 1, y + 25), PAL["panel2"], -1)
            cv2.line(bg, (x, y + 26), (x + w, y + 26), PAL["border"], 1)
        return bg

    def render(self, state: GCSState, ui: dict) -> np.ndarray:
        img = self._bg.copy()
        self._top_bar(img, state, ui)
        self._camera(img, state, ui)
        self._objects(img, state, ui)
        self._map(img, state, ui)
        self._hud(img, state, ui)
        return img

    # ------------------------------------------------------------------ pieces
    def _top_bar(self, img, state: GCSState, ui: dict) -> None:
        t = self.text
        t.put(img, "DRONE GCS", 16, 22, 20, PAL["cyan"], "bold", anchor="lm")
        name = (state.mission or {}).get("name", "no mission")
        x = t.put(img, f"mission  {name}", 140, 22, 15, PAL["fg"], "medium", anchor="lm")[2]
        link = ui.get("link", {})
        if state.ended:
            pill, col = f"MISSION END ({state.end_reason or 'end'})", PAL["blue"]
        elif link.get("connected"):
            pill, col = "LINK LIVE", PAL["green"]
        elif state.has_data:
            pill, col = "LINK LOST", PAL["red"]
        else:
            pill, col = "WAITING FOR SIM", PAL["yellow"]
        bx = x + 18
        w, _ = t.size(pill, 13, "bold")
        cv2.rectangle(img, (bx, 11), (bx + w + 26, 33), col, -1)
        cv2.circle(img, (bx + 11, 22), 4, PAL["bg"], -1, cv2.LINE_AA)
        t.put(img, pill, bx + 20, 22, 13, PAL["bg"], "bold", anchor="lm")
        x = bx + w + 40
        addr = link.get("label") or link.get("address", "")
        if addr:
            t.put(img, addr, x, 22, 12, PAL["dim"], "mono", anchor="lm")
        rx = self.W - 16
        clock = time.strftime("%H:%M:%S")
        rx = t.put(img, clock, rx, 22, 15, PAL["fg"], "mono", anchor="rm")[0] - 18
        keys = "m layer   e stat   g truth   c camera   s shot   q quit"
        rx = t.put(img, keys, rx, 22, 12, PAL["dim"], anchor="rm")[0] - 18
        if ui.get("recording"):
            on = int(time.time() * 2) % 2 == 0
            rx = t.put(
                img, f"REC {fmt_time(ui.get('rec_seconds', 0))[:-2]}", rx, 22, 13, PAL["red"], "bold", anchor="rm"
            )[0]
            if on:
                cv2.circle(img, (rx - 10, 22), 5, PAL["red"], -1, cv2.LINE_AA)

    def _camera(self, img, state: GCSState, ui: dict) -> None:
        t = self.text
        x, y, w, h = self.r_cam
        did = ui.get("active_drone")
        d = state.drones.get(did) if did is not None else None
        if d is not None and (d.phase == "landed" or (state.ended and state.end_reason == "landed")):
            self._landed_card(img, state, d)
            return
        frame: CameraFrame | None = None
        if d is not None:
            frame = d.camera
            if ui.get("detector_on") and d.det_frame is not None and (frame is None or frame.t - d.det_frame.t < 1.0):
                frame = d.det_frame
        hdr = "CAMERA"
        if frame is not None:
            h, w = frame.rgb.shape[:2]
            hdr += f"   drone {frame.drone_id}  {frame.name}  {w}x{h}  {d.cam_fps:4.1f} fps"
            hdr += f"   nadir {frame.cam.nadir_angle_deg():4.1f}°"
        t.put(img, hdr, x + 10, y + 13, 13, PAL["fg"], "medium", anchor="lm")
        ix, iy, iw, ih = x + 1, y + 27, w - 1, h - 27
        if frame is None:
            img[iy : iy + ih, ix : ix + iw] = (24, 20, 20)
            t.put(img, "NO VIDEO", ix + iw // 2, iy + ih // 2, 22, PAL["dim"], "bold", anchor="mm")
            return
        rgb = frame.rgb
        sh, sw = rgb.shape[:2]
        s = min(iw / sw, ih / sh)
        dw, dh = int(round(sw * s)), int(round(sh * s))
        ox, oy = ix + (iw - dw) // 2, iy + (ih - dh) // 2
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        if (dw, dh) != (sw, sh):
            bgr = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        img[iy : iy + ih, ix : ix + iw] = (24, 20, 20)
        img[oy : oy + dh, ox : ox + dw] = bgr
        roi = img[oy : oy + dh, ox : ox + dw]
        cx, cy = dw // 2, dh // 2
        for a, b in (
            ((cx - 14, cy), (cx - 5, cy)),
            ((cx + 5, cy), (cx + 14, cy)),
            ((cx, cy - 14), (cx, cy - 5)),
            ((cx, cy + 5), (cx, cy + 14)),
        ):
            cv2.line(roi, a, b, PAL["white"], 1, cv2.LINE_AA)
        if frame.detections:
            # the boxes first, then each label with the map position of its find, where they hide no other label
            boxes = [tuple(int(round(v * s)) for v in det.box) for det, _ in frame.detections]
            for (det, geo), (x1, y1, x2, y2) in zip(frame.detections, boxes, strict=True):
                cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 0, 0), 4, cv2.LINE_AA)
                width = 1 if geo.get("truncated") else 2
                cv2.rectangle(roi, (x1, y1), (x2, y2), class_color(det.cls), width, cv2.LINE_AA)
            taken: list[Box] = []
            for (det, geo), (x1, y1, _, y2) in zip(frame.detections, boxes, strict=True):
                col = class_color(det.cls)
                label = f"{det.cls} {det.conf:.2f}" + (" (edge)" if geo.get("truncated") else "")
                lw, lh = t.size(label, 13, "bold")
                pos = ""
                if geo.get("ok") and not geo.get("truncated"):
                    pos = f"({geo['xyz'][0]:.1f}, {geo['xyz'][1]:.1f})"
                pw, ph = t.size(pos, 11, "mono") if pos else (0, 0)
                lx = int(np.clip(x1, 1, max(1, dw - lw - 8)))
                # the label above the box, below it or inside its top; the position under the box or the label
                rows = ((y1 - lh - 2, y2 + 3), (y2 + 2, y2 + lh + 6), (y1 + 3, y2 + 3))
                options = [
                    [(lx - 1, ly, lx + lw + 7, ly + lh + 2)] + ([(lx, py - 1, lx + pw + 4, py + ph + 1)] if pos else [])
                    for ly, py in rows
                ]
                i = pick_spot(options, taken, (dw, dh))
                ly, py = rows[i]
                taken += options[i]
                cv2.rectangle(roi, (lx - 1, ly), (lx + lw + 6, ly + lh + 1), col, -1)
                t.put(roi, label, lx + 3, ly, 13, PAL["black"] if sum(col) > 380 else PAL["white"], "bold")
                if pos:
                    t.put(roi, pos, lx + 2, py, 11, PAL["white"], "mono", bg=(20, 20, 20), pad=2, bg_alpha=0.6)
        info = f"t {frame.t:7.2f} s   frame {frame.frame_id}   cam z {frame.cam.t[2]:5.1f} m"
        if frame.depth is None:
            info += "   (no depth)"
        t.put(roi, info, 8, dh - 8, 12, PAL["white"], "mono", anchor="lb", bg=(20, 20, 20), pad=3, bg_alpha=0.55)

    def _landed_card(self, img, state: GCSState, d: DroneState) -> None:
        """The camera panel once the drone sits on the ground (its camera a hand's width above the pad sees only
        noise): where it landed, when, and how far it flew."""
        t = self.text
        x, y, w, h = self.r_cam
        t.put(img, f"CAMERA   drone {d.drone_id}   motors off", x + 10, y + 13, 13, PAL["fg"], "medium", anchor="lm")
        ix, iy, iw, ih = x + 1, y + 27, w - 1, h - 27
        img[iy : iy + ih, ix : ix + iw] = (24, 20, 20)
        cx, cy = ix + iw // 2, iy + ih // 2
        cv2.circle(img, (cx, cy - 62), 42, PAL["green"], 3, cv2.LINE_AA)
        t.put(img, "H", cx, cy - 62, 44, PAL["green"], "bold", anchor="mm")
        t.put(img, "LANDED", cx, cy + 14, 40, PAL["green"], "bold", anchor="mm")
        if state.home is not None and d.pos is not None:
            off = math.hypot(d.pos[0] - state.home[0], d.pos[1] - state.home[1])
            dist = f"{1000 * off:.0f} mm" if off < 0.1 else f"{off:.2f} m"
            t.put(img, f"{dist} from the centre of the landing pad", cx, cy + 58, 16, PAL["fg"], "medium", anchor="mm")
        touchdown = d.landed_t
        if touchdown is None:
            touchdown = (state.end_fields.get("stats") or {}).get("touchdown_t")
        if touchdown is not None and state.t_first is not None:
            sub = f"touchdown T+{fmt_time(touchdown - state.t_first)}   {state.flown_length():.0f} m flown"
            t.put(img, sub, cx, cy + 86, 13, PAL["dim"], "medium", anchor="mm")

    def _objects(self, img, state: GCSState, ui: dict) -> None:
        t = self.text
        x, y, w, h = self.r_obj
        conf = state.tracker.confirmed()
        n_tent = len(state.tracker.objects) - len(conf)
        title = f"DETECTED OBJECTS   {len(conf)} confirmed   {n_tent} tentative"
        x_end = t.put(img, title, x + 10, y + 13, 13, PAL["fg"], "medium", anchor="lm")[2]
        det_s = ui.get("detector", {}).get("name", "detector off")
        det_s = self._fit(det_s, x + w - 10 - (x_end + 20), 12)
        if det_s:
            t.put(img, det_s, x + w - 10, y + 13, 12, PAL["dim"], anchor="rm")
        cols = [
            ("ID", 10),
            ("CLASS", 46),
            ("CONF", 150),
            ("HITS", 208),
            ("X [m]", 262),
            ("Y [m]", 338),
            ("Z [m]", 414),
            ("SEEN [s]", 486),
            ("SPREAD", 566),
        ]
        ty = y + 36
        for name, cx in cols:
            t.put(img, name, x + cx, ty, 11, PAL["dim"], "medium")
        ty += 17
        row_h = 17
        max_rows = max(0, (h - (ty - y) - 4) // row_h)
        # confirmed before tentative, people first
        objs = sorted(state.tracker.objects, key=lambda o: (o.hits < state.tracker.min_hits, o.cls != "person", o.id))
        more = 0
        if len(objs) > max_rows:
            more = len(objs) - (max_rows - 1)
            objs = objs[: max_rows - 1]
        for i, ob in enumerate(objs):
            ok = ob.hits >= state.tracker.min_hits
            col = PAL["fg"] if ok else PAL["dim"]
            if i % 2 == 0:
                cv2.rectangle(img, (x + 4, ty - 2), (x + w - 4, ty + row_h - 2), PAL["panel2"], -1)
            cv2.circle(img, (x + 36, ty + 7), 5, class_color(ob.cls), -1 if ok else 1, cv2.LINE_AA)
            vals = [
                (f"{ob.id}", 10),
                (ob.cls, 46),
                (f"{ob.conf:.2f}", 150),
                (f"{ob.hits}", 208),
                (f"{ob.pos[0]:+7.2f}", 262),
                (f"{ob.pos[1]:+7.2f}", 338),
                (f"{ob.pos[2]:+6.2f}", 414),
                (f"{ob.first_seen:6.1f}", 486),
                (f"{ob.spread:4.2f}", 566),
            ]
            for j, (s, cx) in enumerate(vals):
                t.put(img, s, x + cx + (14 if j == 1 else 0), ty - 1, 13, col, "mono" if j != 1 else "sans")
            ty += row_h
        if more:
            t.put(img, f"+ {more} more (see detections.csv at the end)", x + 46, ty, 12, PAL["dim"])
        if not state.tracker.objects:
            t.put(img, "no objects yet", x + w // 2, y + (h + 40) // 2, 14, PAL["dim"], anchor="mm")

    def _map(self, img, state: GCSState, ui: dict) -> None:
        t = self.text
        x, y, w, h = self.r_map
        layer, stat = ui.get("layer", "elevation"), ui.get("stat", "mean")
        hdr = f"MAP   layer: {layer.upper()}"
        if layer == "elevation":
            hdr += f" ({stat})"
        if state.grid is not None:
            hdr += f"   {state.area[1] - state.area[0]:.0f} x {state.area[3] - state.area[2]:.0f} m survey area"
        t.put(img, hdr, x + 10, y + 13, 13, PAL["fg"], "medium", anchor="lm")
        roi = img[y + 27 : y + h, x + 1 : x + w]
        self.maps.draw(roi, state, layer, stat, ui.get("show_gt", False), ui.get("active_drone"))
        # latest event toast
        if state.events:
            when, msg = state.events[-1]
            if time.time() - when < 5.0:
                t.put(img, msg, x + 16, y + 40, 13, PAL["fg"], "medium", bg=(20, 20, 20), pad=5, bg_alpha=0.75)

    def _fit(self, text: str, width: int, size: int, kind: str = "sans") -> str:
        """``text`` if it fits in ``width`` px, else its part before " (", else shortened with an ellipsis."""
        for candidate in (text, text.split(" (")[0]):
            if self.text.size(candidate, size, kind)[0] <= width:
                return candidate
        s = text.split(" (")[0]
        while s and self.text.size(s + "…", size, kind)[0] > width:
            s = s[:-1]
        return s + "…" if s else ""

    @staticmethod
    def _detector_cell(det: dict) -> tuple[str, str, str]:
        if not det.get("on"):
            return ("DETECTOR", "off", "")
        if det.get("kind") == "gt":
            return ("DETECTOR", "truth", "boxes from the sim")
        return ("DETECTOR", f"{det.get('ms', 0):4.0f} ms", f"{det.get('fps', 0):.1f} fps  {det.get('kind', '')}")

    @staticmethod
    def _link_cell(link: dict, ended: bool) -> tuple[str, str, str]:
        """Message rate while the mission runs; after its END, the data received (the rate only decays to 0)."""
        drop = f"drop {link.get('dropped', 0)}"
        if ended:
            return ("LINK", f"{link.get('bytes', 0) / 1e9:.1f} GB", f"received  {drop}")
        return ("LINK", f"{link.get('msg_rate', 0):4.0f} Hz", f"{link.get('byte_rate', 0) / 1e6:.1f} MB/s  {drop}")

    def _hud(self, img, state: GCSState, ui: dict) -> None:
        t = self.text
        x, y, w, h = self.r_hud
        did = ui.get("active_drone")
        d = state.drones.get(did) if did is not None else None
        ce, cp = state.coverage()
        link = ui.get("link", {})
        det = ui.get("detector", {})
        agl = state.agl(d) if d is not None else float("nan")
        n_wp = len(state.planned_path) if state.planned_path is not None else 0
        wp = f"{int(d.wp_index) + 1}/{n_wp}" if d is not None and d.wp_index is not None and n_wp else "-"
        confirmed = state.tracker.confirmed()
        cells = [
            (
                "MISSION TIME",
                f"T+{fmt_time(state.mission_time)}",
                f"sim t {state.t_now:.1f} s" if state.t_now is not None else "",
            ),
            (
                "ALTITUDE AGL",
                f"{agl:5.1f} m" if math.isfinite(agl) else "--",
                f"z {d.pos[2]:.1f} m" if d is not None and d.pos is not None else "",
            ),
            (
                "SPEED",
                f"{d.speed_h:4.1f} m/s" if d is not None and d.vel is not None else "--",
                f"vs {d.speed_v:+.1f} m/s" if d is not None and d.vel is not None else "",
            ),
            ("HEADING", f"{d.heading_deg:03.0f}°" if d is not None and d.pos is not None else "--", "0 = north"),
            (
                "WAYPOINT",
                wp,
                f"{d.phase + '  ' if d is not None and d.phase else ''}{state.flown_length():.0f} m flown",
            ),
            ("COVERAGE", f"{100 * ce:5.1f} %", f"photo {100 * cp:.1f} %"),
            (
                "PEOPLE FOUND",
                f"{sum(1 for o in confirmed if o.cls == 'person')}",
                f"{len(confirmed)} objects  {state.tracker.n_detections} dets",
            ),
            self._link_cell(link, state.ended),
            self._detector_cell(det),
        ]
        cw = w / len(cells)
        for i, (label, value, sub) in enumerate(cells):
            cx = int(x + i * cw)
            if i:
                cv2.line(img, (cx, y + 10), (cx, y + h - 10), PAL["border"], 1)
            t.put(img, label, cx + 14, y + 10, 11, PAL["dim"], "medium")
            vcol = PAL["fg"]
            if label == "PEOPLE FOUND" and value != "0":
                vcol = class_color("person")
            if label == "COVERAGE":
                vcol = PAL["green"] if ce > 0.95 else PAL["fg"]
            if label == "LINK" and not link.get("connected"):
                vcol = PAL["red"] if state.has_data and not state.ended else PAL["dim"]
            t.put(img, value, cx + 14, y + 26, 24, vcol, "mono_bold")
            if sub:
                t.put(img, sub, cx + 14, y + h - 18, 11, PAL["dim"], "sans")


def render_report_map(state: GCSState, layer: str, size: int = 1200, stat: str = "mean", title: str = "") -> np.ndarray:
    """Standalone map image for the report: title strip + map + legend (flown trail, planned path, objects,
    ground truth if known)."""
    text = TextRenderer()
    mr = MapRenderer(text)
    legend_w = 200
    W, H = size + legend_w, size + 48
    img = np.empty((H, W, 3), np.uint8)
    img[:] = PAL["panel"]
    text.put(img, title or layer.upper(), 16, 24, 20, PAL["fg"], "bold", anchor="lm")
    sub = img[48:, :]
    mr.draw(sub, state, layer, stat, show_gt=bool(state.gt_objects), legend_w=legend_w, report=True)
    return img
