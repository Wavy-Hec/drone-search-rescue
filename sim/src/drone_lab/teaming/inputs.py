"""Inputs of the rescue finale: the drone's elevation map, its pins, or the terrain ground truth (numpy only).

Elevation sources (all give ``(elevation (ny, nx), GridSpec, description)``):

* a ``drone_lab`` survey run directory: ``elevation_map.npy`` + ``map_meta.json`` (``scripts/survey_map.py``);
* a ``drone_gcs`` run directory: ``maps/elevation_mean.npy`` (or ``_max`` / ``_min``) + ``maps/grid.json``;
* a bare ``.npy`` file with a ``map_meta.json`` / ``grid.json`` next to it (or an explicit grid JSON);
* the terrain ground truth: :func:`ground_truth_elevation` rasterises the exact terrain mesh (the same seeded
  ``SURVEY_TERRAIN_CFG`` the simulator builds) -- for when no drone run exists.

A ``drone_lab`` survey map covers the surveyed 40 m x 40 m terrain only, while the base (the landing pad the ground
robot deploys from) sits on the flat 6 m apron around it; a ``drone_gcs`` map spans the whole site but knows only what
the drone scanned. :func:`extend_with_apron` grows a map to the whole site and gives the apron cells the drone did not
map the apron's (flat) height: the drone's cells stay as measured, and terrain the drone did not see stays unknown.

Pins (``load_pins``): the ``drone_gcs`` ``detections.json`` (``{"objects": [{"class", "x", "y", "z",
"confirmed", ...}]}``), a ``{"pins": [...]}`` / ``{"detections": [...]}`` dict, or a plain list. Each pin needs a
class (``class`` / ``label`` / ``cls`` / ``name``) and a position (``x``, ``y`` or ``position`` / ``pos`` /
``xy``); optional ``id``, ``z``, ``confidence``, ``confirmed``, ``radius`` (footprint radius [m]).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from drone_lab.mapping import GridSpec

from .traversability import Footprint

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainGeneratorCfg

    from drone_lab.paths import Area

__all__ = [
    "DEFAULT_FOOTPRINT_RADII",
    "Pin",
    "extend_with_apron",
    "footprints_from_pins",
    "ground_truth_elevation",
    "infer_person_poses",
    "load_elevation",
    "load_pins",
    "on_map",
    "pins_from_data",
    "select_goal_pin",
    "survey_site_bounds",
]

DEFAULT_FOOTPRINT_RADII: dict[str, float] = {
    "person": 0.6,
    "vehicle": 2.2,
    "car": 2.2,
    "truck": 3.0,
    "forklift": 1.8,
    "crate": 0.8,
    "box": 0.6,
    "barrel": 0.45,
    "pallet": 0.8,
    "cone": 0.25,
    "traffic_cone": 0.25,
    "tree": 0.8,
    "tent": 1.6,
    "tarp": 1.2,
    "rubble": 0.8,
    "landing_pad": 0.0,
}
"""Footprint radius per object class [m] (the object's own extent; the robot radius is added by the inflation).
``landing_pad`` is flat and walkable, so it gets no footprint. A person lying down is ~1.8 m long, hence 0.6 m
(their own cells are lethal: the robot stops next to them, not on them)."""

DEFAULT_FOOTPRINT_RADIUS = 0.5
"""Footprint radius of a class missing from :data:`DEFAULT_FOOTPRINT_RADII` [m]."""


@dataclass(frozen=True)
class Pin:
    """An object pinned on the map by the drone (or a ground-truth object)."""

    id: str
    label: str
    x: float
    y: float
    z: float = math.nan
    confidence: float = math.nan
    confirmed: bool = True
    radius: float | None = None
    """Footprint radius override [m]."""
    polygon: tuple[tuple[float, float], ...] | None = None
    """Exact footprint outline [m, world] (e.g. ``footprint_xy`` of a ground-truth object)."""
    pose: str = ""
    """Pose of the object when known (e.g. ``lying_face_up``)."""

    @property
    def xy(self) -> np.ndarray:
        """Position (x, y) [m]."""
        return np.array([self.x, self.y])

    def to_dict(self) -> dict:
        """JSON record (``class`` for ``label``; unknown and empty fields are left out)."""
        d = {"id": self.id, "class": self.label, "x": round(self.x, 3), "y": round(self.y, 3)}
        if math.isfinite(self.z):
            d["z"] = round(self.z, 3)
        if math.isfinite(self.confidence):
            d["confidence"] = round(self.confidence, 4)
        d["confirmed"] = bool(self.confirmed)
        if self.radius is not None:
            d["radius"] = self.radius
        if self.pose:
            d["pose"] = self.pose
        if self.polygon is not None:
            d["footprint_xy"] = [[round(float(a), 3), round(float(b), 3)] for a, b in self.polygon]
        return d


# ------------------------------------------------------------------------------------------------------ pins
def _pin_from_dict(d: dict, index: int) -> Pin | None:
    label = None
    # "category" before "name": ground-truth records carry a unique name ("person_03") and the class in "category"
    # (unlabeled objects such as trees have class None but a category)
    for key in ("class", "label", "cls", "category", "name"):
        if key in d and d[key] is not None:
            label = str(d[key])
            break
    pos = None
    # the centre of the object's box when known (ground truth: "position" is the asset origin, e.g. the feet)
    for key in ("center", "centre"):
        if key in d and d[key] is not None:
            v = list(d[key])
            pos = (float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else math.nan)
            break
    if pos is None and "x" in d and "y" in d:
        pos = (float(d["x"]), float(d["y"]), float(d.get("z", math.nan)))
    if pos is None:
        for key in ("position", "pos", "xy", "xyz", "world"):
            if key in d and d[key] is not None:
                v = list(d[key])
                pos = (float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else math.nan)
                break
    if label is None or pos is None:
        return None
    radius = d.get("radius", d.get("footprint_radius"))
    conf = d.get("confidence", d.get("conf", math.nan))
    polygon = None
    fxy = d.get("footprint_xy", d.get("footprint"))
    if isinstance(fxy, list) and len(fxy) >= 3 and all(isinstance(q, list | tuple) and len(q) >= 2 for q in fxy):
        polygon = tuple((float(q[0]), float(q[1])) for q in fxy)
    pin_id = d.get("name") if ("name" in d and "id" in d and d.get("category") is not None) else d.get("id", index)
    return Pin(
        id=str(pin_id),
        label=label.strip().lower().replace(" ", "_"),
        x=pos[0],
        y=pos[1],
        z=pos[2],
        confidence=float(conf) if conf is not None else math.nan,
        confirmed=bool(d.get("confirmed", True)),
        radius=None if radius is None else float(radius),
        polygon=polygon,
        pose=str(d.get("pose", "") or ""),
    )


def load_pins(path: str | Path, *, confirmed_only: bool = True) -> list[Pin]:
    """Read pins from a JSON file (see the module docstring for the accepted layouts)."""
    return pins_from_data(
        json.loads(Path(path).expanduser().read_text()), confirmed_only=confirmed_only, source=str(path)
    )


def pins_from_data(data: dict | list, *, confirmed_only: bool = True, source: str = "data") -> list[Pin]:
    """Pins from already-parsed JSON data (a list of records or a dict holding one)."""
    path = source
    items = data
    if isinstance(data, dict):
        for key in ("objects", "pins", "detections", "finds", "items"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        else:
            raise ValueError(f"{path}: no list of pins found (expected 'objects', 'pins' or 'detections')")
    if not isinstance(items, list):
        raise ValueError(f"{path}: pins must be a list")
    pins = []
    for i, d in enumerate(items):
        if not isinstance(d, dict):
            continue
        pin = _pin_from_dict(d, i)
        if pin is None:
            continue
        if confirmed_only and not pin.confirmed:
            continue
        pins.append(pin)
    return pins


def footprints_from_pins(
    pins: Iterable[Pin], *, radii: dict[str, float] | None = None, default_radius: float = DEFAULT_FOOTPRINT_RADIUS
) -> list[Footprint]:
    """Circular footprints of the pinned objects (zero-radius classes such as the landing pad are skipped)."""
    table = dict(DEFAULT_FOOTPRINT_RADII)
    if radii:
        table.update({k.lower(): float(v) for k, v in radii.items()})
    out = []
    for p in pins:
        r = p.radius if p.radius is not None else table.get(p.label, default_radius)
        if r is None or r <= 0:
            continue
        if p.polygon is not None:
            poly = np.asarray(p.polygon, dtype=float)
            r_poly = float(np.max(np.linalg.norm(poly - p.xy, axis=1)))
            out.append(Footprint(float(p.x), float(p.y), r_poly, p.label, polygon=tuple(map(tuple, poly))))
        else:
            out.append(Footprint(float(p.x), float(p.y), float(r), p.label))
    return out


def select_goal_pin(
    pins: Sequence[Pin], *, pin_id: str | None = None, label: str = "person", near_xy: ArrayLike | None = None
) -> Pin | None:
    """The pin the robot is sent to: by id, else the ``label`` pin closest to ``near_xy`` (else the first one)."""
    if pin_id is not None:
        for p in pins:
            if str(p.id) == str(pin_id):
                return p
        return None
    cands = [p for p in pins if p.label == label]
    if not cands:
        return None
    if near_xy is None:
        return cands[0]
    near = np.asarray(near_xy, dtype=float)
    return min(cands, key=lambda p: float(np.linalg.norm(p.xy - near)))


def infer_person_poses(
    pins: Iterable[Pin],
    z: np.ndarray,
    grid: GridSpec,
    *,
    ring_m: tuple[float, float] = (1.2, 2.0),
    max_lying_height_m: float = 0.7,
) -> list[Pin]:
    """Give person pins without a pose one from the map: lying or standing, by the pin's height above the ground.

    The drone's camera looks straight down, so a pin geolocated on a standing person sits on their upper body
    (0.8-1.9 m above the ground on the demo site), and one on a person lying or sitting stays low (at most 0.5 m):
    sitting people count as lying. The ground is the lower quartile of the known map cells ``ring_m`` away from the
    pin (the person's own cells and objects right next to them stay out of it). Pins with a pose, without a height
    or without known ground around them are returned unchanged.
    """
    xs, ys = grid.cell_centers()
    out = []
    for p in pins:
        if p.label != "person" or p.pose or not math.isfinite(p.z):
            out.append(p)
            continue
        r = np.hypot(xs[None, :] - p.x, ys[:, None] - p.y)
        ring = z[(r >= ring_m[0]) & (r <= ring_m[1])]
        ring = ring[np.isfinite(ring)]
        if ring.size == 0:
            out.append(p)
            continue
        height = p.z - float(np.percentile(ring, 25))
        out.append(replace(p, pose="lying" if height < max_lying_height_m else "standing"))
    return out


# ------------------------------------------------------------------------------------------------ elevation
def _grid_from_json(path: Path) -> GridSpec:
    d = json.loads(path.read_text())
    for key in ("grid", "elevation_grid"):
        if key in d and isinstance(d[key], dict):
            return GridSpec.from_dict(d[key])
    return GridSpec.from_dict(d)


def load_elevation(
    source: str | Path, grid_json: str | Path | None = None, *, layer: str = "mean"
) -> tuple[np.ndarray, GridSpec, str]:
    """Elevation map of a drone run (see the module docstring).

    Args:
        source: Run directory (drone_lab or drone_gcs) or a ``.npy`` file.
        grid_json: Grid description when it cannot be found next to ``source``.
        layer: drone_gcs layer (``mean``, ``max`` or ``min``).

    Returns:
        ``(elevation (ny, nx) float, GridSpec, description)``.
    """
    src = Path(source).expanduser()
    npy = meta = None
    if src.is_dir():
        if (src / "elevation_map.npy").is_file():
            npy, meta = src / "elevation_map.npy", src / "map_meta.json"
        elif (src / "maps" / f"elevation_{layer}.npy").is_file():
            npy, meta = src / "maps" / f"elevation_{layer}.npy", src / "maps" / "grid.json"
        else:
            raise FileNotFoundError(
                f"{src}: no elevation_map.npy (drone_lab) or maps/elevation_{layer}.npy (drone_gcs)"
            )
    else:
        npy = src
        for cand in (src.parent / "map_meta.json", src.parent / "grid.json"):
            if cand.is_file():
                meta = cand
                break
    if grid_json is not None:
        meta = Path(grid_json).expanduser()
    if meta is None or not meta.is_file():
        raise FileNotFoundError(f"no grid description for {npy} (map_meta.json / grid.json); pass grid_json")
    z = np.load(npy).astype(float)
    grid = _grid_from_json(meta)
    if z.shape != grid.shape:
        raise ValueError(f"{npy}: shape {z.shape} does not match the grid {grid.shape} from {meta}")
    return z, grid, f"drone map {npy}"


def ground_truth_elevation(
    resolution: float = 0.25,
    *,
    terrain_cfg: TerrainGeneratorCfg | None = None,
    padding: float = 0.0,
    supersample: int = 3,
) -> tuple[np.ndarray, GridSpec, str]:
    """Terrain ground truth: the exact survey terrain mesh rasterised to a grid (CPU, needs Isaac Lab's terrain
    generator but not the simulator).

    The grid covers the whole terrain including the flat border (52 m x 52 m by default), anchored at the mesh's
    minimum corner, so the cell edges fall on the 0.1 m height-field lattice for resolutions that are multiples
    of it.

    Returns:
        ``(elevation (ny, nx), GridSpec, description)``; each cell holds the mean surface height inside it.
    """
    from drone_lab.mapping import ground_truth_map
    from drone_lab.terrain import SURVEY_TERRAIN_CFG, generate_terrain_mesh

    cfg = terrain_cfg if terrain_cfg is not None else SURVEY_TERRAIN_CFG
    vertices, faces = generate_terrain_mesh(cfg)
    lo = vertices[:, :2].min(axis=0) - padding
    hi = vertices[:, :2].max(axis=0) + padding
    grid = GridSpec.from_bounds(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), resolution)
    z, _ = ground_truth_map(vertices, faces, grid, supersample=supersample)
    return z, grid, f"terrain ground truth (seed {getattr(cfg, 'seed', None)}, {resolution} m cells)"


# --------------------------------------------------------------------------------------------- site apron
def on_map(grid: GridSpec, xy: Sequence[float] | np.ndarray) -> bool:
    """Whether the world point ``xy`` [m] lies on the grid."""
    _, _, valid = grid.cell_index(np.array([float(xy[0])]), np.array([float(xy[1])]))
    return bool(valid[0])


def _bounds(b) -> tuple[float, float, float, float]:
    if all(hasattr(b, k) for k in ("x_min", "x_max", "y_min", "y_max")):
        return float(b.x_min), float(b.x_max), float(b.y_min), float(b.y_max)
    x0, x1, y0, y1 = (float(v) for v in b)
    return x0, x1, y0, y1


def survey_site_bounds(terrain_cfg: TerrainGeneratorCfg | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(terrain, site)`` bounds ``(x_min, x_max, y_min, y_max)`` [m] of the survey terrain: the sub-terrains
    (what the drone surveys) and the whole site with the flat border around them. Needs Isaac Lab's terrain
    configs, not the simulator."""
    from drone_lab.terrain import SURVEY_TERRAIN_CFG, terrain_area

    cfg = terrain_cfg if terrain_cfg is not None else SURVEY_TERRAIN_CFG
    inner = terrain_area(cfg)
    return _bounds(inner), _bounds(inner.inset(-float(cfg.border_width)))


def extend_with_apron(
    z: np.ndarray,
    grid: GridSpec,
    terrain: Area | Sequence[float],
    site: Area | Sequence[float],
    *,
    height: float = 0.0,
) -> tuple[np.ndarray, GridSpec, np.ndarray]:
    """Grow a map to the whole site and fill in the flat apron (the border) around the surveyed terrain.

    The grid grows to cover ``site`` with the same cell size and the same cell edges, so the map's own cells are
    copied unchanged (nothing is resampled). A cell without a height (added, or not seen by the drone) that lies
    entirely on the apron (inside ``site``, outside ``terrain``) gets ``height``; every other unknown cell (terrain
    the map does not cover, or a cell across the site edge) stays unknown (NaN). The apron is known flat ground; a
    measured cell is never changed and the terrain is never made up.

    Args:
        z: ``(ny, nx)`` elevation [m], NaN = unknown.
        grid: Its grid.
        terrain: Bounds ``(x_min, x_max, y_min, y_max)`` of the surveyed terrain [m] (or an object with those
            attributes, e.g. :class:`drone_lab.paths.Area`).
        site: Bounds of the whole site, apron included [m].
        height: Apron height [m]; Isaac Lab's terrain generator puts the top of its flat border at z = 0.

    Returns:
        ``(elevation, grid, apron mask)``; the mask marks the apron cells that got ``height``. A map that already
        covers the site and its apron comes back as a copy with an empty mask.
    """
    z = np.asarray(z, dtype=float)
    if z.shape != grid.shape:
        raise ValueError(f"elevation shape {z.shape} does not match the grid {grid.shape}")
    tx0, tx1, ty0, ty1 = _bounds(terrain)
    sx0, sx1, sy0, sy1 = _bounds(site)
    res = grid.resolution
    # whole cells on each side keep the lattice (cell edges stay at x_min + k * res)
    west = max(0, math.ceil((grid.x_min - sx0) / res - 1e-9))
    east = max(0, math.ceil((sx1 - grid.x_max) / res - 1e-9))
    south = max(0, math.ceil((grid.y_min - sy0) / res - 1e-9))
    north = max(0, math.ceil((sy1 - grid.y_max) / res - 1e-9))
    new = GridSpec(
        grid.x_min - west * res, grid.y_min - south * res, res, grid.nx + west + east, grid.ny + south + north
    )
    out = np.full(new.shape, np.nan)
    out[south : south + grid.ny, west : west + grid.nx] = z
    xs, ys = new.cell_centers()
    lo_x, hi_x, lo_y, hi_y = xs - 0.5 * res, xs + 0.5 * res, ys - 0.5 * res, ys + 0.5 * res
    eps = 1e-9
    in_site_x = (lo_x >= sx0 - eps) & (hi_x <= sx1 + eps)
    in_site_y = (lo_y >= sy0 - eps) & (hi_y <= sy1 + eps)
    off_terrain_x = (hi_x <= tx0 + eps) | (lo_x >= tx1 - eps)
    off_terrain_y = (hi_y <= ty0 + eps) | (lo_y >= ty1 - eps)
    in_site = in_site_y[:, None] & in_site_x[None, :]
    off_terrain = off_terrain_y[:, None] | off_terrain_x[None, :]
    apron = np.isnan(out) & in_site & off_terrain
    out[apron] = float(height)
    return out, new, apron
