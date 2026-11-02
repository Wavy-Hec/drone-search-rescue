"""Seeded generator of the dressed disaster site: which object goes where, plus the ground-truth JSON.

The generator works on a CPU copy of the survey terrain (:class:`TerrainSite`): Isaac Lab's terrain generator is
deterministic for :data:`drone_lab.terrain.SURVEY_TERRAIN_CFG`, so the mesh here is the mesh the simulator imports.
Every object is placed with height queries on that mesh (:class:`drone_lab.mapping.MeshHeightQuery`):

* its footprint (the oriented rectangle of its posed bounding box, plus a small clearance) must be flat within a
  per-class tolerance, so nothing straddles a block edge, a stair step or a wall ("not inside blocks");
* it rests on the highest terrain point under that footprint, so no terrain pokes into it;
* it keeps a minimum free distance to every other object (circumscribed circles of the footprints);
* labelled objects stay inside the survey area (with a margin), trees stay below the survey altitude.

People are "standing" or "lying" (rotated 90 deg about their left-right axis, face up or down, flat on the
ground); some are placed right next to a step at least 0.6 m high ("near_step": a block, wall, stair, pyramid
terrace or crater wall), and "near_block" marks the people next to a collapsed building (that step is on a blocks
or stairs tile). The landing pad sits on the flat border next to the start of the default lawnmower path, with
traffic cones at its corners.

``people_style="natural"`` keeps that layout and then re-poses every person (:mod:`drone_lab.sar.people`): the
victims lie on their back or side, sit or wave, at their place or slid towards their step (a sitting victim leans
its back towards it), and a few responders look around, point or walk by the vehicle and the landing pad; each
person shows its pose in one of its variants (mirror image, clip frame). A posed person rests on the
terrain through the under-side height map of its pose: nothing pokes into the body, and the parts that carry it
(back, seat, feet) are at most a few centimetres above the ground. The rest of the site is the T-pose site's.

Only numpy (+ Isaac Lab's CPU terrain generator through :mod:`drone_lab.terrain`) is needed; no simulator.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .assets import (
    ASSETS,
    CLASS_IDS,
    CLASSES,
    SEMANTIC_TYPE,
    UNLABELED,
    AssetSpec,
    asset_root,
    assets_of,
)
from .camera import SAR_CAMERA
from .people import CELL, PEOPLE_STYLES, POSES, RESPONDER_TAGS, pose_geometry, variants

GROUND_TRUTH_FORMAT = "drone_lab.sar.ground_truth/v1"
"""Identifier written into the ground-truth JSON."""

OBJECTS_SUBPATH = "objects"
"""Objects are spawned at ``<site prim>/objects/<name>`` (e.g. ``/World/sar/objects/person_03``)."""

LANDING_PAD_KEY = "landing_pad"
"""Pseudo asset key of the procedural landing pad (not a USD file)."""


@dataclass(frozen=True)
class ScenarioCfg:
    """What to put into the site. Ranges are inclusive and drawn per seed."""

    seed: int = 0
    num_people: tuple[int, int] = (6, 10)
    lying_fraction: float = 0.5
    near_block_fraction: float = 0.4
    """Fraction of people that must stand or lie right next to a step at least 0.6 m high (``near_step``). On this
    terrain that is mostly a block of a collapsed building (``near_block``), sometimes a pyramid terrace or the
    crater wall; the rule is kept as is so that every seed keeps its layout."""
    people_style: str = "tpose"
    """``tpose``: the characters' rest T-pose, standing or lying flat. ``natural``: the same places, re-posed as
    victims lying on their back or side, sitting or waving, plus responders (see the module docstring)."""
    num_responders: tuple[int, int] = (2, 3)
    """Responders by the vehicle and the landing pad (``natural`` only)."""
    num_vehicles: int = 1
    num_crates: tuple[int, int] = (7, 10)
    num_barrels: tuple[int, int] = (4, 6)
    num_pallets: tuple[int, int] = (3, 5)
    num_site_cones: tuple[int, int] = (2, 4)
    pad_cones: bool = True
    """Four traffic cones at the corners of the landing-pad apron."""
    num_trees: tuple[int, int] = (4, 6)
    trees_inside_area: int = 1
    """How many of the trees stand inside the survey area (occluders); the others are on the border."""
    area_margin: float = 0.75
    """Labelled objects keep this distance from the edge of the survey area [m]."""
    gap: float = 0.6
    """Minimum free distance between the footprint circles of two objects [m]."""
    cluster_gap: float = 0.12
    """Minimum free distance inside a crate / barrel cluster [m]."""
    home_xy: tuple[float, float] | None = None
    """Landing-pad centre. None: 3 m outside the area edge, beside the first lawnmower waypoint."""
    pad_radius: float = 1.25
    apron_half: float = 3.0
    """Half size of the asphalt apron around the pad [m] (the pad and its cones live on it)."""
    survey_clearance: float = 4.0
    """Survey altitude above the highest terrain point (``SurveyCarrotCommandCfg.altitude``) [m]."""
    survey_spacing: float = 4.0
    """Lawnmower spacing (sets the U-turn band whose terrain also counts for the flight height) [m]."""
    tree_top_margin: float = 2.0
    """Tree tops stay at least this far below the survey flight height [m]."""
    max_tries: int = 3000
    """Rejection-sampling attempts per object."""


# per-class placement rules: (flatness tolerance [m], footprint clearance [m])
_FLAT_TOL = {
    "person_standing": 0.15,
    "person_lying": 0.12,
    "vehicle": 0.12,
    "crate": 0.10,
    "barrel": 0.10,
    "pallet": 0.08,
    "cone": 0.15,
    UNLABELED: 0.5,
}
_CLEARANCE = {"vehicle": 0.25, "person_lying": 0.15, "person_standing": 0.1}
_NEAR_STEP = 0.6
"""A step next to a person is terrain at least this much higher than the person's ground [m]."""
_NATURAL_ONLY = ("people_style", "num_responders")
"""Scenario settings that only exist for natural people (T-pose ground truth leaves them out, unchanged)."""
_VICTIM_MIX = {
    "lying": {"lying": 0.4, "on side": 0.35, "sitting": 0.25},
    "standing": {"waving": 0.65, "sitting": 0.35},
}
"""Natural pose of a victim (a :attr:`drone_lab.sar.people.PoseSpec.tag`) by the pose of its T-pose place
(probabilities); which variant of it (mirror image, clip frame) is drawn per victim."""
_VICTIM_FALLBACK = {"lying": ("lying", "on side", "sitting"), "standing": ("waving", "sitting")}
_SUPPORT_GAP = {"lying": 0.06, "sitting": 0.05, "standing": 0.05}
"""Largest drop of the ground under the parts that carry a posed person (back, seat, feet) [m]."""
_SUPPORT = 0.02
"""Under-side cells at most this high carry the body (they touch the ground on a flat floor) [m]."""
_RIM_GRAZE = 0.01
"""How far the rim of an under-side square may reach into the body [m] (see :meth:`_Placer.natural_fit`)."""
_SUPPORT_LIFT = 0.01
"""A posed person must rest on the parts that carry it: within this of the terrain [m] (not, say, on its hand
over the foot of a wall with its back in the air)."""
RESPONDER_GROUP = "responders"
"""``group`` of the responders (victims have none)."""
RESPONDER_ASSETS = (
    "female_medical_01",
    "male_medical_01",
    "female_police_01",
    "female_police_02",
    "female_police_03",
    "male_police_04",
)
"""Characters in uniform that the responders wear."""
_NEAR_RING = (0.35, 0.8, 1.3)
"""The step must be within these distances of the person's footprint (sampled on a 0.25 m grid) [m]."""
BUILDING_TILES = ("blocks", "stairs")
"""Sub-terrains that stand for collapsed buildings (block clusters, concrete stairs): ``near_block``."""


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_family(pose: str) -> str:
    """``lying``, ``sitting`` or ``standing`` (``upright`` objects and walking people count as standing)."""
    if pose.startswith("lying"):
        return "lying"
    return "sitting" if pose.startswith("sitting") else "standing"


def pose_quaternion(pose: str) -> np.ndarray:
    """Quaternion (x, y, z, w) that puts an asset into ``pose`` before its yaw.

    The people face -Y, so a rotation about X by -90 deg lays them on their back (front up, body along +Y from the
    feet) and +90 deg on their front (body along -Y); the T-pose arms stay flat on the ground either way. Natural
    poses (:data:`drone_lab.sar.people.POSES`) are baked into their meshes: no rotation.
    """
    if pose in ("upright", "standing") or pose in POSES:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if pose == "lying_face_up":
        return np.array([-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    if pose == "lying_face_down":
        return np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    raise ValueError(f"unknown pose {pose!r}")


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def _box_corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


@dataclass
class SceneObject:
    """One placed object. Positions are world coordinates [m] (z up); the survey env origin is the world origin."""

    name: str
    category: str
    """Semantic class, or ``"tree"`` for unlabelled occluders."""
    asset: str
    """Key of :data:`drone_lab.sar.assets.ASSETS`, or ``"landing_pad"`` (procedural)."""
    position: tuple[float, float, float]
    """Where the asset origin goes."""
    quat_xyzw: tuple[float, float, float, float]
    yaw: float
    """Heading about world z [rad] (applied after the pose rotation)."""
    pose: str
    """"upright", "standing", "lying_face_up" or "lying_face_down"; natural people: a key of
    :data:`drone_lab.sar.people.POSES` ("lying_back", "lying_side_mirrored", "sitting", "walking", ...)."""
    scale: float
    """Size multiplier; the asset's unit scale (0.01 for centimetre assets) comes on top."""
    unit_scale: float
    center: tuple[float, float, float]
    """Centre of the object's posed bounding box in the world."""
    half_extents: tuple[float, float, float]
    """Half size of the posed box along its yawed axes (length along local x, width along local y, height)."""
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    footprint_xy: list[list[float]]
    """Four corners of the oriented footprint rectangle (counter-clockwise)."""
    ground_z: float
    """Terrain height the object rests on."""
    ground_spread: float
    """Max - min terrain height under the footprint (flatness) [m]."""
    labeled: bool
    in_area: bool
    near_block: bool = False
    """Person right next to a collapsed building: terrain at least 0.6 m higher within ~1.3 m, on a blocks or
    stairs tile."""
    group: str | None = None
    near_step: bool = False
    """Person right next to any step at least 0.6 m high (block, wall, stair, pyramid terrace, crater wall)."""

    @property
    def label(self) -> str | None:
        return self.category if self.labeled else None

    @property
    def radius(self) -> float:
        """Radius of the circle around the footprint (spacing checks) [m]."""
        return float(math.hypot(self.half_extents[0], self.half_extents[1]))

    def prim_path(self, site_prim_path: str = "/World/sar") -> str:
        return f"{site_prim_path}/{OBJECTS_SUBPATH}/{self.name}"


def object_geometry(spec_lo, spec_hi, pose: str, yaw: float, scale: float, center_xy=None, origin=None):
    """Posed box geometry of an asset.

    Either ``center_xy`` (footprint centre, z ignored) or ``origin`` (asset origin, world) must be given.

    Returns:
        dict with ``quat`` (x, y, z, w), ``origin`` (x, y, 0 when placed by centre), ``local_center``
        (posed + yawed box centre relative to the origin), ``half`` (half extents along yawed axes),
        ``z_min`` (lowest point relative to the origin), ``corners`` (8 world-offset corners relative to origin).
    """
    q_pose = pose_quaternion(pose)
    q = _quat_mul(yaw_quaternion(yaw), q_pose)
    corners = _box_corners(spec_lo, spec_hi) * scale
    posed = corners @ _quat_to_matrix(q_pose).T  # posed, before yaw
    plo, phi = posed.min(axis=0), posed.max(axis=0)
    half = 0.5 * (phi - plo)
    c_posed = 0.5 * (phi + plo)
    rz = _quat_to_matrix(yaw_quaternion(yaw))
    c_local = rz @ c_posed
    world_corners = corners @ _quat_to_matrix(q).T
    if origin is None:
        origin = np.array([center_xy[0] - c_local[0], center_xy[1] - c_local[1], 0.0])
    return {
        "quat": q,
        "origin": np.asarray(origin, float),
        "local_center": c_local,
        "half": half,
        "z_min": float(plo[2]),
        "corners": world_corners,
        "rz": rz,
    }


def _person_parts(lo, hi):
    """Body and arm boxes of a T-pose character (asset frame): a 0.5 m wide trunk/legs column and the arm span
    at 74-88 % of the height (shoulders to wrists)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    h = hi[2] - lo[2]
    body_lo = np.array([max(lo[0], -0.25), lo[1], lo[2]])
    body_hi = np.array([min(hi[0], 0.25), hi[1], hi[2]])
    arm_lo = np.array([lo[0], lo[1], lo[2] + 0.74 * h])
    arm_hi = np.array([hi[0], hi[1], lo[2] + 0.88 * h])
    return body_lo, body_hi, arm_lo, arm_hi


class Rest(NamedTuple):
    """How a posed person rests on the terrain (:meth:`_Placer.natural_fit`)."""

    ground: float
    """Height of the asset origin (the lowest point of the body) [m]."""
    gap: float
    """Largest drop of the ground under the parts that carry the body (0 on a flat floor) [m]."""
    lift: float
    """Smallest height of those parts above the terrain: 0 when the body rests on them [m]."""
    spread: float
    """Max - min terrain height under the body [m]."""


class TerrainSite:
    """CPU copy of the survey terrain with height queries and the sub-terrain type of every tile."""

    def __init__(self, terrain_cfg=None):
        from drone_lab.mapping import MeshHeightQuery
        from drone_lab.terrain import SURVEY_TERRAIN_CFG, terrain_area

        self.cfg = terrain_cfg if terrain_cfg is not None else SURVEY_TERRAIN_CFG
        self.vertices, self.faces, self.tiles = generate_terrain_with_tiles(self.cfg)
        self.query = MeshHeightQuery(self.vertices, self.faces)
        self.area = terrain_area(self.cfg)
        b = float(self.cfg.border_width)
        self.bounds = (self.area.x_min - b, self.area.x_max + b, self.area.y_min - b, self.area.y_max + b)
        self.tile_size = (float(self.cfg.size[0]), float(self.cfg.size[1]))

    # --- terrain queries -------------------------------------------------------------------------------------
    def heights(self, xy: np.ndarray) -> np.ndarray:
        return self.query(np.asarray(xy, dtype=np.float64).reshape(-1, 2))

    def height(self, x: float, y: float) -> float:
        return float(self.heights(np.array([[x, y]]))[0])

    def tile_of(self, x: float, y: float) -> str:
        """Sub-terrain name under (x, y), or "border"."""
        a = self.area
        if not (a.x_min <= x < a.x_max and a.y_min <= y < a.y_max):
            return "border"
        row = int((x - a.x_min) // self.tile_size[0])
        col = int((y - a.y_min) // self.tile_size[1])
        return self.tiles.get((row, col), "unknown")

    def terrain_z_range(self, inset: float = 0.0) -> tuple[float, float]:
        """Min / max vertex height inside the area grown by ``-inset`` (negative inset grows it)."""
        inside = self.area.inset(inset).contains(self.vertices[:, :2])
        z = self.vertices[inside, 2] if np.any(inside) else self.vertices[:, 2]
        return float(z.min()), float(z.max())

    def flight_z(self, clearance: float = 4.0, spacing: float = 4.0) -> float:
        """Survey flight height, computed like ``SurveyCarrotCommand`` (highest point of area + U-turn band)."""
        return self.terrain_z_range(inset=-0.5 * spacing - 0.5)[1] + float(clearance)

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        x0, x1, y0, y1 = self.bounds
        return x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin


def generate_terrain_with_tiles(cfg) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], str]]:
    """Run Isaac Lab's :class:`TerrainGenerator` on the CPU and record the sub-terrain name of each tile.

    Returns:
        ``(vertices (V, 3), faces (F, 3), {(row, col): sub_terrain_name})``; row indexes x and col indexes y
        (tile (row, col) spans x in [x_min + row * size_x, ...), y in [y_min + col * size_y, ...)).
    """
    from isaaclab.terrains import TerrainGenerator
    from isaaclab.utils import clone

    class _RecordingGenerator(TerrainGenerator):
        def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
            names = [k for k, v in self.cfg.sub_terrains.items() if v is sub_terrain_cfg]
            self._tile_names[(int(row), int(col))] = names[0] if names else type(sub_terrain_cfg).__name__
            return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    _RecordingGenerator._tile_names = {}
    gen = _RecordingGenerator(clone(cfg), device="cpu")
    tiles = dict(_RecordingGenerator._tile_names)
    mesh = gen.terrain_mesh
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64), tiles


@dataclass
class Scenario:
    """Result of :func:`generate_scenario`."""

    cfg: ScenarioCfg
    objects: list[SceneObject]
    home: tuple[float, float, float]
    """Landing-pad centre on the ground [m]."""
    area: tuple[float, float, float, float]
    bounds: tuple[float, float, float, float]
    flight_z: float
    terrain_z: tuple[float, float]
    tiles: dict[tuple[int, int], str] = field(default_factory=dict)
    terrain_seed: int | None = None

    def by_category(self, category: str) -> list[SceneObject]:
        return [o for o in self.objects if o.category == category]

    def labeled(self) -> list[SceneObject]:
        return [o for o in self.objects if o.labeled]

    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in CLASSES}
        out[UNLABELED] = 0
        for o in self.objects:
            out[o.category] = out.get(o.category, 0) + 1
        return out

    def ground_truth(self, site_prim_path: str = "/World/sar") -> dict:
        """JSON-ready ground truth (see the module docstring of :mod:`drone_lab.sar`)."""
        objs = []
        for i, o in enumerate(self.objects):
            spec = ASSETS.get(o.asset)
            objs.append(
                {
                    "id": i,
                    "name": o.name,
                    "class": o.category if o.labeled else None,
                    "class_id": CLASS_IDS.get(o.category) if o.labeled else None,
                    "category": o.category,
                    "labeled": o.labeled,
                    "asset": o.asset,
                    "url": spec.url if spec is not None else None,
                    "prim_path": o.prim_path(site_prim_path),
                    # pin position = centre of the object's box (a lying person's origin is at the feet)
                    "x": round(float(o.center[0]), 3),
                    "y": round(float(o.center[1]), 3),
                    "z": round(float(o.center[2]), 3),
                    "pose": o.pose,
                    "position": _r(o.position),
                    "center": _r(o.center),
                    "yaw_deg": round(math.degrees(o.yaw), 2),
                    "quat_xyzw": _r(o.quat_xyzw, 6),
                    "scale": round(o.scale, 4),
                    "unit_scale": o.unit_scale,
                    "size": _r([2 * h for h in o.half_extents]),
                    "aabb_min": _r(o.aabb_min),
                    "aabb_max": _r(o.aabb_max),
                    "footprint_xy": [_r(p) for p in o.footprint_xy],
                    "ground_z": round(o.ground_z, 3),
                    "in_area": o.in_area,
                    "near_block": o.near_block,
                    "near_step": o.near_step,
                    "group": o.group,
                }
            )
        x0, x1, y0, y1 = self.area
        bx0, bx1, by0, by1 = self.bounds
        pad = next((o for o in self.objects if o.category == "landing_pad"), None)
        cfg = _jsonable(asdict(self.cfg))
        if self.cfg.people_style == "tpose":  # T-pose sites keep their original records byte for byte
            for key in _NATURAL_ONLY:
                cfg.pop(key)
        return {
            "format": GROUND_TRUTH_FORMAT,
            "seed": self.cfg.seed,
            "frame": "world: x/y horizontal, z up, metres; the survey env origin is the world origin",
            "asset_root": asset_root(),
            "semantic_type": SEMANTIC_TYPE,
            "classes": list(CLASSES),
            "class_ids": dict(CLASS_IDS),
            "unlabeled_category": UNLABELED,
            "site_prim_path": site_prim_path,
            "area": {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1},
            "site_bounds": {"x_min": bx0, "x_max": bx1, "y_min": by0, "y_max": by1},
            "terrain": {
                "generator_seed": self.terrain_seed,
                "z_min": round(self.terrain_z[0], 3),
                "z_max": round(self.terrain_z[1], 3),
                "tiles": {f"{r},{c}": n for (r, c), n in sorted(self.tiles.items())},
            },
            "home": {
                "x": round(self.home[0], 3),
                "y": round(self.home[1], 3),
                "z": round(self.home[2], 3),
                "landing_pad": pad.name if pad else None,
            },
            "survey": {
                "flight_z": round(self.flight_z, 3),
                "clearance": self.cfg.survey_clearance,
                "spacing": self.cfg.survey_spacing,
            },
            "camera": SAR_CAMERA.to_dict(),
            "counts": {k: v for k, v in self.counts().items()},
            "counts_in_area": {c: sum(1 for o in self.objects if o.category == c and o.in_area) for c in CLASSES},
            "scenario_cfg": cfg,
            "objects": objs,
        }

    def save_ground_truth(self, path: str | Path, site_prim_path: str = "/World/sar") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.ground_truth(site_prim_path), indent=1))
        return path


def _r(v, nd: int = 3):
    return [round(float(x), nd) for x in v]


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, list | tuple):
        return [_jsonable(v) for v in d]
    return d


class _Placer:
    """Rejection sampler with the shared placement rules."""

    def __init__(self, site: TerrainSite, cfg: ScenarioCfg, rng: np.random.Generator):
        self.site = site
        self.cfg = cfg
        self.rng = rng
        self.objects: list[SceneObject] = []
        self.counter: dict[str, int] = {}
        self.flight_z = site.flight_z(cfg.survey_clearance, cfg.survey_spacing)
        self.keepout: list[tuple[float, float, float]] = []  # (x, y, r) circles nothing else may enter

    def name(self, category: str) -> str:
        n = self.counter.get(category, 0)
        self.counter[category] = n + 1
        return f"{category}_{n:02d}"

    # footprint sampling of an oriented rectangle (centre, yaw, half extents), 0.1 m grid
    def footprint_points(self, cx, cy, rz, hx, hy, step=0.1):
        nx = max(2, int(math.ceil(2 * hx / step)) + 1)
        ny = max(2, int(math.ceil(2 * hy / step)) + 1)
        u, v = np.meshgrid(np.linspace(-hx, hx, nx), np.linspace(-hy, hy, ny), indexing="ij")
        local = np.column_stack([u.ravel(), v.ravel()])
        return local @ rz[:2, :2].T + np.array([cx, cy])

    def posed_box_points(self, lo, hi, pose, yaw, scale, origin_xy, clear=0.0, step=0.1):
        """Grid of xy points under a box given in the asset frame, after pose + yaw, at the asset origin."""
        posed = _box_corners(lo, hi) * scale @ _quat_to_matrix(pose_quaternion(pose)).T
        plo, phi = posed.min(axis=0), posed.max(axis=0)
        nx = max(2, int(math.ceil((phi[0] - plo[0] + 2 * clear) / step)) + 1)
        ny = max(2, int(math.ceil((phi[1] - plo[1] + 2 * clear) / step)) + 1)
        u, v = np.meshgrid(
            np.linspace(plo[0] - clear, phi[0] + clear, nx),
            np.linspace(plo[1] - clear, phi[1] + clear, ny),
            indexing="ij",
        )
        rz = _quat_to_matrix(yaw_quaternion(yaw))[:2, :2]
        return np.column_stack([u.ravel(), v.ravel()]) @ rz.T + np.asarray(origin_xy, float)[:2]

    def free(self, cx, cy, r, group=None, gap=None, skip=None) -> bool:
        gap = self.cfg.gap if gap is None else gap
        for o in self.objects:
            if o is skip:
                continue
            g = self.cfg.cluster_gap if (group is not None and o.group == group) else gap
            if math.hypot(o.center[0] - cx, o.center[1] - cy) < o.radius + r + g:
                return False
        return all(math.hypot(kx - cx, ky - cy) >= kr + r for kx, ky, kr in self.keepout)

    def try_place(
        self,
        spec: AssetSpec | None,
        category: str,
        pose: str,
        yaw: float,
        scale: float,
        cx: float,
        cy: float,
        *,
        bbox: tuple | None = None,
        rule: str | None = None,
        labeled: bool = True,
        group: str | None = None,
        require_near_step: bool = False,
        region: str = "area",
        gap: float | None = None,
        name: str | None = None,
    ) -> SceneObject | None:
        """Place one object with its footprint centred at (cx, cy) if every rule passes.

        All sizes are metres: the asset boxes of :mod:`drone_lab.sar.assets` already include the unit fix; the
        unit scale is only recorded (the spawner multiplies it into the USD scale).
        """
        site, cfg = self.site, self.cfg
        lo, hi = (spec.bbox_min, spec.bbox_max) if spec is not None else bbox
        unit = spec.unit_scale if spec is not None else 1.0
        geo = object_geometry(lo, hi, pose, yaw, scale, center_xy=(cx, cy))
        hx, hy, hz = (float(v) for v in geo["half"])
        rz = geo["rz"]
        rule = rule or category
        a = site.area
        corners_xy = self.footprint_points(cx, cy, rz, hx, hy, step=1e9)  # the 4 footprint corners
        if region == "area":
            m = cfg.area_margin
            if (
                corners_xy[:, 0].min() < a.x_min + m
                or corners_xy[:, 0].max() > a.x_max - m
                or corners_xy[:, 1].min() < a.y_min + m
                or corners_xy[:, 1].max() > a.y_max - m
            ):
                return None
        elif region == "border":
            if not all(site.in_bounds(px, py, 0.3) for px, py in corners_xy):
                return None
            if a.inset(-0.5).contains(np.array([[cx, cy]]))[0]:
                return None
        elif region == "any":
            if not all(site.in_bounds(px, py, 0.1) for px, py in corners_xy):
                return None
        r = math.hypot(hx, hy)
        if not self.free(cx, cy, r if category != UNLABELED else 0.6 * r, group=group, gap=gap):
            return None
        # flatness and resting height
        clear = _CLEARANCE.get(rule, 0.05)
        origin_xy = geo["origin"][:2]
        if category == UNLABELED:
            pts = self.footprint_points(cx, cy, rz, 0.4, 0.4)  # trunk
        elif category == "person":
            # a T-pose person touches the ground along a cross (body and arms), not with the corners of its box
            body_lo, body_hi, arm_lo, arm_hi = _person_parts(lo, hi)
            body = self.posed_box_points(body_lo, body_hi, pose, yaw, scale, origin_xy, 0.05)
            arms = self.posed_box_points(arm_lo, arm_hi, pose, yaw, scale, origin_xy, clear)
            pts = body if rule == "person_standing" else np.concatenate([body, arms])
        else:
            pts = self.footprint_points(cx, cy, rz, hx + clear, hy + clear)
        z = site.heights(pts)
        if np.any(np.isnan(z)):
            return None
        ground, spread = float(z.max()), float(z.max() - z.min())
        if spread > _FLAT_TOL.get(rule, 0.1):
            return None
        if rule == "person_standing":
            # nothing under the outstretched arms (~1.3 m up) may reach them: keep the terrain 1 m below
            arm = site.heights(arms)
            if np.any(np.isnan(arm)) or float(arm.max()) > ground + 1.0:
                return None
        near_step = near_block = False
        if category == "person":
            near_step, near_block = self.near_flags(cx, cy, rz, hx + clear, hy + clear, ground)
            if require_near_step and not near_step:
                return None
        if category == UNLABELED:
            ground = float(np.median(z))
            if ground + float(geo["corners"][:, 2].max()) > self.flight_z - cfg.tree_top_margin:
                return None
        origin = geo["origin"].copy()
        origin[2] = ground if category == UNLABELED else ground - geo["z_min"]
        world = geo["corners"] + origin
        center = origin + geo["local_center"]
        center[2] = 0.5 * (world[:, 2].min() + world[:, 2].max())
        ang = np.arctan2(corners_xy[:, 1] - cy, corners_xy[:, 0] - cx)
        rect = corners_xy[np.argsort(ang)]  # counter-clockwise
        obj = SceneObject(
            name=name or self.name(category),
            category=category,
            asset=spec.key if spec is not None else LANDING_PAD_KEY,
            position=tuple(float(v) for v in origin),
            quat_xyzw=tuple(float(v) for v in geo["quat"]),
            yaw=float(yaw),
            pose=pose,
            scale=float(scale),
            unit_scale=float(unit),
            center=tuple(float(v) for v in center),
            half_extents=(hx, hy, hz),
            aabb_min=tuple(float(v) for v in world.min(axis=0)),
            aabb_max=tuple(float(v) for v in world.max(axis=0)),
            footprint_xy=[[float(px), float(py)] for px, py in rect],
            ground_z=ground,
            ground_spread=spread,
            labeled=labeled and category != UNLABELED,
            in_area=bool(site.area.contains(np.array([[cx, cy]]))[0]),
            near_block=near_block,
            group=group,
            near_step=near_step,
        )
        self.objects.append(obj)
        return obj

    def near_flags(self, cx, cy, rz, hx, hy, ground) -> tuple[bool, bool]:
        """(``near_step``, ``near_block``) of a person whose footprint (with clearance) has half sizes hx, hy."""
        ring = np.concatenate([self.footprint_points(cx, cy, rz, hx + d, hy + d, step=0.25) for d in _NEAR_RING])
        high = self.site.heights(ring) >= ground + _NEAR_STEP
        near_block = any(self.site.tile_of(float(x), float(y)) in BUILDING_TILES for x, y in ring[high])
        return bool(np.any(high)), near_block

    # --- natural people -----------------------------------------------------------------------------------
    def natural_fit(self, spec: AssetSpec, pose: str, yaw: float, scale: float, cx: float, cy: float) -> Rest | None:
        """How a posed person whose box is centred at (cx, cy) rests on the terrain (None: off the terrain).

        The under-side height map of the pose is laid over the terrain, sampled at the centre, corners and edge
        midpoints of every square; the body rests on the highest sample relative to its under-side, so no terrain
        pokes into it. The parts that carry the body are the squares whose under-side is at most 2 cm high (back,
        seat, soles): see :class:`Rest` for the two numbers that say whether it stands on them.
        """
        geo = pose_geometry(spec.key, pose)
        under = geo.underside * scale
        nx, ny = under.shape
        half = 0.5 * CELL * scale
        ii, jj = np.meshgrid(np.arange(2 * nx + 1), np.arange(2 * ny + 1), indexing="ij")
        local = np.column_stack([geo.origin[0] * scale + ii.ravel() * half, geo.origin[1] * scale + jj.ravel() * half])
        c, s = math.cos(yaw), math.sin(yaw)
        t = self.site.heights(local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])).reshape(ii.shape)
        if np.any(np.isnan(t)):
            return None
        near = np.max([t[di : di + 2 * nx : 2, dj : dj + 2 * ny : 2] for di in range(3) for dj in range(3)], axis=0)
        body = ~np.isnan(under)
        floor = t[1::2, 1::2]
        # the square centres stay clear of the body; a square's rim may graze it by 1 cm (the lowest vertex of a
        # square is rarely right above its highest rim point: on a slope this keeps bodies from floating by cm)
        ground = float(max(np.max(floor[body] - under[body]), np.max(near[body] - under[body]) - _RIM_GRAZE))
        support = body & (np.nan_to_num(under, nan=np.inf) <= _SUPPORT * scale)
        return Rest(
            ground=ground,
            gap=float(np.max(ground - floor[support])),
            lift=float(np.min(ground + under[support] - near[support])),
            spread=float(floor[body].max() - floor[body].min()),
        )

    @staticmethod
    def rests_well(fit: Rest | None, pose: str) -> bool:
        return fit is not None and fit.gap <= _SUPPORT_GAP[pose_family(pose)] and fit.lift <= _SUPPORT_LIFT

    def natural_object(
        self, spec: AssetSpec, pose: str, yaw: float, scale: float, cx: float, cy: float, fit, name: str, group=None
    ) -> SceneObject:
        """The :class:`SceneObject` of a posed person (origin at the bottom centre of its box)."""
        geo = pose_geometry(spec.key, pose)
        ground, spread = fit.ground, fit.spread
        lo, hi = np.asarray(geo.bbox_min) * scale, np.asarray(geo.bbox_max) * scale
        hx, hy, hz = (float(v) for v in 0.5 * (hi - lo))
        rz = _quat_to_matrix(yaw_quaternion(yaw))
        origin = np.array([cx, cy, ground])
        world = _box_corners(lo, hi) @ rz.T + origin
        corners = self.footprint_points(cx, cy, rz, hx, hy, step=1e9)
        rect = corners[np.argsort(np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx))]
        clear = _CLEARANCE["person_lying" if pose_family(pose) == "lying" else "person_standing"]
        near_step, near_block = self.near_flags(cx, cy, rz, hx + clear, hy + clear, ground)
        return SceneObject(
            name=name,
            category="person",
            asset=spec.key,
            position=tuple(float(v) for v in origin),
            quat_xyzw=tuple(float(v) for v in yaw_quaternion(yaw)),
            yaw=float(yaw),
            pose=pose,
            scale=float(scale),
            unit_scale=float(spec.unit_scale),
            center=(float(cx), float(cy), float(ground + hz)),
            half_extents=(hx, hy, hz),
            aabb_min=tuple(float(v) for v in world.min(axis=0)),
            aabb_max=tuple(float(v) for v in world.max(axis=0)),
            footprint_xy=[[float(px), float(py)] for px, py in rect],
            ground_z=ground,
            ground_spread=spread,
            labeled=True,
            in_area=bool(self.site.area.contains(np.array([[cx, cy]]))[0]),
            near_block=near_block,
            group=group,
            near_step=near_step,
        )

    def natural_half(self, spec: AssetSpec, pose: str, scale: float) -> tuple[float, float]:
        """Half length and width of a posed person's footprint [m]."""
        geo = pose_geometry(spec.key, pose)
        return 0.5 * scale * (geo.bbox_max[0] - geo.bbox_min[0]), 0.5 * scale * (geo.bbox_max[1] - geo.bbox_min[1])

    def in_area_margin(self, cx: float, cy: float, yaw: float, hx: float, hy: float) -> bool:
        """Is the footprint (centre, heading, half sizes) inside the survey area by the labelled-object margin?"""
        a, m = self.site.area, self.cfg.area_margin
        rz = _quat_to_matrix(yaw_quaternion(yaw))
        c = self.footprint_points(cx, cy, rz, hx, hy, step=1e9)
        return bool(
            c[:, 0].min() >= a.x_min + m
            and c[:, 0].max() <= a.x_max - m
            and c[:, 1].min() >= a.y_min + m
            and c[:, 1].max() <= a.y_max - m
        )

    def step_direction(self, cx: float, cy: float, ground: float, reach: float = 2.5):
        """Unit vector and distance from (cx, cy) to the nearest terrain at least 0.5 m above ``ground``."""
        ang = np.linspace(-math.pi, math.pi, 48, endpoint=False)
        rad = np.arange(0.3, reach + 1e-9, 0.1)
        rr, aa = np.meshgrid(rad, ang, indexing="ij")
        pts = np.column_stack([cx + (rr * np.cos(aa)).ravel(), cy + (rr * np.sin(aa)).ravel()])
        high = np.nonzero(self.site.heights(pts) >= ground + 0.5)[0]
        if len(high) == 0:
            return None
        k = int(high[0])  # rows are radii: the first hit is the nearest
        return np.array([math.cos(aa.ravel()[k]), math.sin(aa.ravel()[k])]), float(rr.ravel()[k])

    def repose(self, index: int, rng: np.random.Generator, variant: int = 0) -> SceneObject:
        """Natural pose for the T-pose person ``objects[index]``.

        Tries the poses of the victim mix (the drawn one first) inside the old footprint circle, so every spacing
        rule still holds; near a step, slid towards it (a sitting victim turns its back to it). If none rests within
        the support tolerance there, it moves up to 1 m away (with the spacing checked again); failing that, it
        takes the best fit. ``variant`` picks the variant of every pose (mirror image, clip frame).
        """
        old = self.objects[index]
        spec = ASSETS[old.asset]
        slot = "lying" if old.pose.startswith("lying") else "standing"
        mix = _VICTIM_MIX[slot]
        first = list(mix)[int(rng.choice(len(mix), p=np.array(list(mix.values())) / sum(mix.values())))]
        tags = [first] + [q for q in _VICTIM_FALLBACK[slot] if q != first]
        order = [variants(t)[variant % len(variants(t))] for t in tags]
        step = self.step_direction(old.center[0], old.center[1], old.ground_z) if old.near_step else None
        jitter = float(rng.uniform(-0.35, 0.35))
        best = None
        for far in (False, True):
            for pose in order:
                hx, hy = self.natural_half(spec, pose, old.scale)
                r = math.hypot(hx, hy)
                room = old.radius - r
                if room < 0 and not far:
                    continue
                for yaw, (dx, dy) in self._repose_candidates(old, pose, step, room, jitter, far):
                    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                    cx, cy = old.center[0] + dx, old.center[1] + dy
                    if old.in_area and not self.in_area_margin(cx, cy, yaw, hx, hy):
                        continue
                    if far and not self.free(cx, cy, r, group=old.group, skip=old):
                        continue
                    fit = self.natural_fit(spec, pose, yaw, old.scale, cx, cy)
                    if fit is None:
                        continue
                    args = (spec, pose, yaw, old.scale, cx, cy, fit, old.name, old.group)
                    if self.rests_well(fit, pose):
                        return self.natural_object(*args)
                    score = max(fit.gap - _SUPPORT_GAP[pose_family(pose)], fit.lift - _SUPPORT_LIFT)
                    if best is None or score < best[0]:
                        best = (score, args)
        if best is None:
            raise RuntimeError(f"no natural pose fits the place of {old.name}")
        return self.natural_object(*best[1])

    @staticmethod
    def _repose_candidates(old: SceneObject, pose: str, step, room: float, jitter: float, far: bool):
        """(yaw, offset) to try, best first. Inside the old circle: the old heading (a sitting victim: back to the
        step) slid towards the step, then eight headings at moves of up to 0.3 m. ``far``: moves of 0.4-1 m."""
        headings = [old.yaw + k * math.pi / 4 for k in range(8)]
        if far:
            for r in (0.4, 0.6, 0.8, 1.0):
                for a in np.arange(8) * math.pi / 4:
                    for yaw in headings:
                        yield yaw, (r * math.cos(a), r * math.sin(a))
            return
        preferred = [old.yaw, old.yaw + math.pi]
        if pose_family(pose) == "sitting" and step is not None:  # the sitting body's back is its local +Y
            back = math.atan2(-step[0][0], step[0][1])
            preferred = [back + jitter, back - jitter, back]
        slides = [0.0]
        if step is not None:
            slides = [float(d) for d in np.arange(min(room, step[1]), 0.0, -0.05)] + [0.0]
        for yaw in preferred:
            for d in slides:
                yield yaw, ((d * step[0][0], d * step[0][1]) if step is not None else (0.0, 0.0))
        moves = [(0.0, 0.0)] + [
            (r * math.cos(a), r * math.sin(a)) for r in (0.1, 0.2, 0.3) if r <= room for a in np.arange(8) * math.pi / 4
        ]
        for move in moves:
            for yaw in headings:
                yield yaw, move

    def place_responders(
        self, n: int, rng: np.random.Generator, tags: tuple[str, ...] = RESPONDER_TAGS, tries: int = 600
    ) -> list[SceneObject]:
        """Up to ``n`` responders: the first by the vehicle, the others around the landing pad (outside its apron),
        on the same level as what they stand by and facing it (the vehicle) or the survey area (the pad). Responder
        ``k`` shows ``tags[k]`` (looks around, points, walks) and each wears another uniform (if possible not a
        victim's)."""
        vehicle = next((o for o in self.objects if o.category == "vehicle"), None)
        pad = next(o for o in self.objects if o.category == "landing_pad")
        victims = {o.asset for o in self.objects if o.category == "person"}
        a = self.site.area
        area_c = np.array([0.5 * (a.x_min + a.x_max), 0.5 * (a.y_min + a.y_max)])
        keep = self.cfg.apron_half * 1.2
        placed: list[SceneObject] = []
        for k in range(n):
            options = variants(tags[k % len(tags)])
            taken = {o.asset for o in placed}
            keys = [c for c in RESPONDER_ASSETS if c not in victims | taken] or [
                c for c in RESPONDER_ASSETS if c not in taken
            ]
            anchors = ([vehicle] if k == 0 and vehicle is not None else []) + [pad]
            for anchor in anchors:
                obj = None
                for _ in range(tries):
                    spec = ASSETS[keys[int(rng.integers(len(keys)))]]
                    pose = options[int(rng.integers(len(options)))]
                    scale = float(rng.uniform(*spec.scale_range))
                    hx, hy = self.natural_half(spec, pose, scale)
                    r = math.hypot(hx, hy)
                    lo = (keep if anchor is pad else anchor.radius) + r + 0.3
                    dist, ang = float(rng.uniform(lo, lo + 2.2)), float(rng.uniform(-math.pi, math.pi))
                    cx, cy = anchor.center[0] + dist * math.cos(ang), anchor.center[1] + dist * math.sin(ang)
                    look = (area_c if anchor is pad else np.asarray(anchor.center[:2])) - (cx, cy)
                    yaw = math.atan2(look[0], -look[1]) + float(rng.uniform(-0.4, 0.4))  # faces its local -Y
                    if anchor is pad and not self.site.in_bounds(cx, cy, r + 0.3):
                        continue
                    if anchor is not pad and not self.in_area_margin(cx, cy, yaw, hx, hy):
                        continue
                    if not self.free(cx, cy, r, group=RESPONDER_GROUP):
                        continue
                    fit = self.natural_fit(spec, pose, yaw, scale, cx, cy)
                    if not self.rests_well(fit, pose) or abs(fit.ground - anchor.ground_z) > 0.15:
                        continue
                    obj = self.natural_object(spec, pose, yaw, scale, cx, cy, fit, self.name("person"), RESPONDER_GROUP)
                    break
                if obj is not None:
                    self.objects.append(obj)
                    placed.append(obj)
                    break
        return placed

    def sample_xy(self, region: str = "area", margin: float = 1.0) -> tuple[float, float]:
        a = self.site.area
        if region == "area":
            return (
                float(self.rng.uniform(a.x_min + margin, a.x_max - margin)),
                float(self.rng.uniform(a.y_min + margin, a.y_max - margin)),
            )
        x0, x1, y0, y1 = self.site.bounds
        while True:
            x = float(self.rng.uniform(x0 + margin, x1 - margin))
            y = float(self.rng.uniform(y0 + margin, y1 - margin))
            if region == "any" or not a.inset(-0.5).contains(np.array([[x, y]]))[0]:
                return x, y

    def place_random(
        self,
        spec_choices,
        category,
        *,
        pose_fn=None,
        scale=None,
        region="area",
        group=None,
        near=None,
        require_near_step=False,
        gap=None,
        rule_fn=None,
        tries: int | None = None,
    ) -> SceneObject | None:
        """Rejection-sample one object of ``category`` anywhere in ``region`` (or within ``near=(x, y, r)``)."""
        for _ in range(tries or self.cfg.max_tries):
            spec = self.pick(spec_choices)
            pose = pose_fn() if pose_fn else "upright"
            yaw = float(self.rng.uniform(-math.pi, math.pi))
            s = scale if scale is not None else float(self.rng.uniform(*spec.scale_range))
            if near is not None:
                nx, ny, nr = near
                ang, rad = self.rng.uniform(-math.pi, math.pi), nr * math.sqrt(self.rng.uniform())
                cx, cy = nx + rad * math.cos(ang), ny + rad * math.sin(ang)
            else:
                cx, cy = self.sample_xy(region)
            rule = rule_fn(pose) if rule_fn else category
            obj = self.try_place(
                spec,
                category,
                pose,
                yaw,
                s,
                cx,
                cy,
                rule=rule,
                group=group,
                region=region,
                require_near_step=require_near_step,
                gap=gap,
            )
            if obj is not None:
                return obj
        return None

    def pick(self, specs: list[AssetSpec]) -> AssetSpec:
        w = np.array([s.weight for s in specs], dtype=float)
        return specs[int(self.rng.choice(len(specs), p=w / w.sum()))]


def default_home(site: TerrainSite, cfg: ScenarioCfg) -> tuple[float, float]:
    """Landing-pad centre: 3 m outside the area's x_min edge, level with the first default lawnmower line."""
    if cfg.home_xy is not None:
        return float(cfg.home_xy[0]), float(cfg.home_xy[1])
    a = site.area
    n_lines = max(1, math.ceil((a.y_max - a.y_min) / cfg.survey_spacing - 1e-9))
    first_line = a.y_min + 0.5 * (a.y_max - a.y_min) / n_lines
    return a.x_min - 3.0, first_line


def generate_scenario(cfg: ScenarioCfg | None = None, site: TerrainSite | None = None, **overrides) -> Scenario:
    """Place every object of the site for ``cfg.seed`` (deterministic for a given cfg and terrain).

    Args:
        cfg: Scenario settings (``overrides`` are applied on top, e.g. ``seed=3``).
        site: Terrain to place on; built from :data:`drone_lab.terrain.SURVEY_TERRAIN_CFG` when None. Pass one
            in to reuse it (building it costs about a second).
    """
    cfg = replace(cfg or ScenarioCfg(), **overrides)
    if cfg.people_style not in PEOPLE_STYLES:
        raise ValueError(f"people_style must be one of {PEOPLE_STYLES}, not {cfg.people_style!r}")
    site = site if site is not None else TerrainSite()
    rng = np.random.default_rng([int(cfg.seed), 0x5A12])
    p = _Placer(site, cfg, rng)

    def n_of(rng_range) -> int:
        lo, hi = rng_range
        return int(rng.integers(int(lo), int(hi) + 1))

    # 1) landing pad on its apron, cones at the apron corners; then nothing else may enter the apron
    hx_, hy_ = default_home(site, cfg)
    r = cfg.pad_radius
    pad = p.try_place(
        None,
        "landing_pad",
        "upright",
        0.0,
        1.0,
        hx_,
        hy_,
        bbox=((-r, -r, 0.0), (r, r, 0.049)),
        rule="pallet",
        region="any",
        name="landing_pad_00",
    )
    if pad is None:
        raise RuntimeError(f"landing pad does not fit at {hx_, hy_}: the ground there is not flat")
    home = (pad.center[0], pad.center[1], pad.ground_z)
    if cfg.pad_cones:
        cone = ASSETS["traffic_cone"]
        d = cfg.apron_half - 0.8
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            p.try_place(
                cone,
                "cone",
                "upright",
                float(rng.uniform(-math.pi, math.pi)),
                cone.scale_range[0],
                home[0] + sx * d,
                home[1] + sy * d,
                region="any",
                group="pad_cones",
                gap=0.0,
            )
    p.keepout.append((home[0], home[1], cfg.apron_half * 1.2))

    # 2) vehicle(s): the biggest footprint first
    for _ in range(cfg.num_vehicles):
        p.place_random(assets_of("vehicle"), "vehicle")

    # 3) people, standing or lying; the first ones right next to a step (mostly the blocks of collapsed buildings)
    people = assets_of("person")
    order = rng.permutation(len(people))
    n_people = n_of(cfg.num_people)
    n_near = int(round(cfg.near_block_fraction * n_people))
    for i in range(n_people):
        lying = bool(rng.uniform() < cfg.lying_fraction)
        face = "lying_face_up" if rng.uniform() < 0.6 else "lying_face_down"
        pose = face if lying else "standing"
        spec = people[int(order[i % len(order)])]
        obj = p.place_random(
            [spec],
            "person",
            pose_fn=lambda pose=pose: pose,
            rule_fn=lambda q: "person_lying" if q.startswith("lying") else "person_standing",
            require_near_step=i < n_near,
        )
        if obj is None and i < n_near:  # no spot next to a step left: anywhere will do
            obj = p.place_random(
                [spec],
                "person",
                pose_fn=lambda pose=pose: pose,
                rule_fn=lambda q: "person_lying" if q.startswith("lying") else "person_standing",
            )

    # 4) clutter: crates and barrels in small clusters, pallets and a few cones on their own
    def clusters(category: str, total: int, sizes=(1, 2, 3, 4)):
        specs = assets_of(category)
        placed, k = 0, 0
        while placed < total and k < 4 * total:
            k += 1
            size = int(min(total - placed, rng.choice(sizes)))
            first = p.place_random(specs, category, group=f"{category}_pile_{k:02d}")
            if first is None:
                break
            placed += 1
            for _ in range(size - 1):
                o = p.place_random(
                    specs, category, group=first.group, near=(first.center[0], first.center[1], 1.6), tries=300
                )
                placed += o is not None

    clusters("crate", n_of(cfg.num_crates))
    clusters("barrel", n_of(cfg.num_barrels), sizes=(1, 2, 3))
    for _ in range(n_of(cfg.num_pallets)):
        p.place_random(assets_of("pallet"), "pallet")
    for _ in range(n_of(cfg.num_site_cones)):
        p.place_random(assets_of("cone"), "cone")

    # 5) trees (unlabelled occluders): a few on the border, some inside the area
    n_trees = n_of(cfg.num_trees)
    trees = assets_of(UNLABELED)
    for i in range(n_trees):
        region = "area" if i < cfg.trees_inside_area else "border"
        p.place_random(trees, UNLABELED, region=region, gap=0.3)

    # 6) natural people: the same places, re-posed, plus responders (own random streams: the layout above is kept,
    #    and drawing the variants does not change the pose drawn for each victim)
    if cfg.people_style == "natural":
        rng_people = np.random.default_rng([int(cfg.seed), 0x5A12, 0x9E0])
        rng_variants = np.random.default_rng([int(cfg.seed), 0x5A12, 0x9E1])
        for i, o in enumerate(list(p.objects)):
            if o.category == "person":
                p.objects[i] = p.repose(i, rng_people, int(rng_variants.integers(1 << 16)))
        lo, hi = cfg.num_responders
        tags = tuple(RESPONDER_TAGS[int(k)] for k in rng_variants.permutation(len(RESPONDER_TAGS)))
        p.place_responders(int(rng_people.integers(int(lo), int(hi) + 1)), rng_people, tags)

    zmin, zmax = site.terrain_z_range()
    return Scenario(
        cfg=cfg,
        objects=p.objects,
        home=home,
        area=site.area.as_tuple(),
        bounds=site.bounds,
        flight_z=p.flight_z,
        terrain_z=(zmin, zmax),
        tiles=dict(site.tiles),
        terrain_seed=getattr(site.cfg, "seed", None),
    )


def load_ground_truth(path: str | Path) -> dict:
    """Read a ground-truth JSON written by :meth:`Scenario.save_ground_truth`."""
    gt = json.loads(Path(path).read_text())
    if gt.get("format") != GROUND_TRUTH_FORMAT:
        raise ValueError(f"{path}: not a {GROUND_TRUTH_FORMAT} file")
    return gt
