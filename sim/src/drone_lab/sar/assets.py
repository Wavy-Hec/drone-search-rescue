"""Verified assets of the search-and-rescue (SAR) scene: URLs, units, measured sizes and semantic classes.

Every entry below was downloaded from the Isaac Sim 6.2 asset bucket on 2026-11-01 and measured with ``pxr``
(stage ``metersPerUnit`` / ``upAxis`` and the extents of the mesh points in the rest pose). The measurement
scripts and their raw output live in ``~/isaac_ws/runs/2026-11-01_drones/sar_scene/asset_measurements/``.

Findings that shape the scene code:

* People (``Isaac/People/Characters/*``): metres, Z-up, origin between the feet, feet at z = 0, facing -Y, in a
  T-pose (arm span 1.56-1.87 m, height 1.61-1.80 m, front-to-back 0.35-0.43 m). No semantics, no colliders.
  :mod:`drone_lab.sar.people` bakes them into natural poses (lying, sitting, waving, standing).
* Simple_Warehouse props and ``Isaac/Props``: metres, Z-up, origin at the bottom centre. Most carry *legacy*
  ``Semantics`` labels ("box", "barel", "crate", "pallet", "cone") and the forklift carries a ``LabelsAPI``
  label "forklift" on its body; the spawner replaces all of them with the classes of :data:`CLASSES`.
* ``NVIDIA/Assets/Vegetation/Trees/*`` and ``NVIDIA/Assets/DigitalTwin/*``: **centimetres** (metersPerUnit 0.01),
  Z-up, so they need a 0.01 unit scale on top of any size scale (:attr:`AssetSpec.unit_scale`). The tree sizes
  below come from ``UsdGeom.BBoxCache`` because their foliage is not plain mesh points.
* ``Isaac/Props/TrafficCone/traffic_cone.usd`` and ``Isaac/Vehicles/`` do not exist in 6.2 (HTTP 404); the cone
  is ``Simple_Warehouse/Props/S_TrafficCone.usd`` (0.46 m tall, scaled to a 0.69 m road cone here).

Only the standard library is imported, so the module works in any Python (e.g. the ground station).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ASSET_ROOT_6_2 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2"
"""Isaac Sim 6.2 asset root (also the ``persistent.isaac.asset_root.default`` of Isaac Lab's kit file)."""

CLASSES: tuple[str, ...] = ("person", "vehicle", "crate", "barrel", "pallet", "cone", "landing_pad")
"""Semantic classes of the scene, in detector class-id order (``CLASS_IDS``). Trees are unlabeled occluders."""

CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
"""Class name -> integer id (YOLO order)."""

UNLABELED = "tree"
"""Category of objects that are spawned without a semantic label (they only occlude)."""

SEMANTIC_TYPE = "class"
"""Semantic type used for every label (``UsdSemantics.LabelsAPI`` instance name, filter ``class:*``)."""


def asset_root() -> str:
    """Asset root URL: ``ISAACSIM_ASSET_ROOT`` if set (Isaac Lab honours it too), else the 6.2 bucket."""
    root = os.environ.get("ISAACSIM_ASSET_ROOT", "").rstrip("/\\")
    return root or ASSET_ROOT_6_2


@dataclass(frozen=True)
class AssetSpec:
    """One verified asset.

    Sizes are in metres at ``unit_scale`` (i.e. after converting the asset's own units), in the asset frame
    (Z-up, origin as authored).
    """

    key: str
    """Short unique name."""
    category: str
    """Semantic class from :data:`CLASSES`, or :data:`UNLABELED` for occluders."""
    path: str
    """Path below the asset root."""
    bbox_min: tuple[float, float, float]
    """Measured lower corner of the geometry [m] (asset frame, unit scale applied)."""
    bbox_max: tuple[float, float, float]
    """Measured upper corner of the geometry [m]."""
    meters_per_unit: float = 1.0
    """The asset's own ``metersPerUnit`` (0.01 for centimetre assets)."""
    scale_range: tuple[float, float] = (1.0, 1.0)
    """Uniform size multiplier drawn per instance (applied on top of the unit fix)."""
    poses: tuple[str, ...] = ("upright",)
    """Allowed poses: "upright", or "standing" / "lying" for people."""
    weight: float = 1.0
    """Relative probability of picking this asset within its category."""
    notes: str = ""

    @property
    def unit_scale(self) -> float:
        """Scale that converts the asset's units to metres (the stage is in metres, Z-up)."""
        return float(self.meters_per_unit)

    @property
    def url(self) -> str:
        return f"{asset_root()}/{self.path}"

    @property
    def size(self) -> tuple[float, float, float]:
        """Native size (dx, dy, dz) [m]."""
        return tuple(float(b - a) for a, b in zip(self.bbox_min, self.bbox_max, strict=False))  # type: ignore[return-value]


def _people() -> list[AssetSpec]:
    rows = [
        # key, folder/file, bbox_min, bbox_max  (mesh points, rest T-pose)
        (
            "male_construction_05",
            "male_adult_construction_05_new/male_adult_construction_05_new.usd",
            (-0.920, -0.225, 0.0),
            (0.919, 0.195, 1.773),
        ),
        ("female_business_02", "F_Business_02/F_Business_02.usd", (-0.781, -0.192, 0.0), (0.781, 0.157, 1.607)),
        (
            "female_police_01",
            "female_adult_police_01_new/female_adult_police_01_new.usd",
            (-0.841, -0.212, 0.0),
            (0.841, 0.165, 1.687),
        ),
        ("female_medical_01", "F_Medical_01/F_Medical_01.usd", (-0.812, -0.221, 0.0), (0.813, 0.198, 1.680)),
        ("male_medical_01", "M_Medical_01/M_Medical_01.usd", (-0.936, -0.244, 0.0), (0.936, 0.175, 1.788)),
        (
            "female_police_02",
            "female_adult_police_02/female_adult_police_02.usd",
            (-0.830, -0.204, 0.0),
            (0.830, 0.167, 1.688),
        ),
        (
            "female_police_03",
            "female_adult_police_03_new/female_adult_police_03_new.usd",
            (-0.834, -0.212, 0.0),
            (0.834, 0.193, 1.671),
        ),
        (
            "male_construction_01",
            "male_adult_construction_01_new/male_adult_construction_01_new.usd",
            (-0.899, -0.223, 0.0),
            (0.898, 0.179, 1.797),
        ),
        (
            "male_construction_03",
            "male_adult_construction_03/male_adult_construction_03.usd",
            (-0.876, -0.209, 0.0),
            (0.876, 0.193, 1.781),
        ),
        (
            "male_police_04",
            "male_adult_police_04/male_adult_police_04.usd",
            (-0.904, -0.232, 0.0),
            (0.904, 0.197, 1.771),
        ),
    ]
    return [
        AssetSpec(
            key,
            "person",
            f"Isaac/People/Characters/{path}",
            lo,
            hi,
            scale_range=(0.96, 1.04),
            poses=("standing", "lying"),
            notes="T-pose, faces -Y",
        )
        for key, path, lo, hi in rows
    ]


_WAREHOUSE = "Isaac/Environments/Simple_Warehouse/Props"

ASSETS: dict[str, AssetSpec] = {
    a.key: a
    for a in [
        *_people(),
        AssetSpec(
            "forklift",
            "vehicle",
            "Isaac/Props/Forklift/forklift.usd",
            (-0.607, -1.747, -0.002),
            (0.607, 1.747, 2.153),
            notes="forks along +Y; body label 'forklift'",
        ),
        AssetSpec("card_box_a", "crate", f"{_WAREHOUSE}/SM_CardBoxA_01.usd", (-0.35, -0.25, 0.0), (0.35, 0.25, 0.5)),
        AssetSpec("card_box_b", "crate", f"{_WAREHOUSE}/SM_CardBoxB_01.usd", (-0.25, -0.25, 0.0), (0.25, 0.25, 0.5)),
        AssetSpec(
            "plastic_crate_a",
            "crate",
            f"{_WAREHOUSE}/SM_CratePlastic_A_01.usd",
            (-0.308, -0.240, 0.0),
            (0.308, 0.259, 0.435),
        ),
        AssetSpec(
            "plastic_crate_c",
            "crate",
            f"{_WAREHOUSE}/SM_CratePlastic_C_01.usd",
            (-0.252, -0.346, 0.0),
            (0.252, 0.421, 0.338),
        ),
        AssetSpec(
            "wood_crate",
            "crate",
            "NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Shipping/Wood_Crates/Plywood_B/"
            "PlywoodCrate_B02_100x100x100cm_PR_NV_01.usd",
            (-0.501, -0.504, 0.0),
            (0.501, 0.504, 1.017),
            meters_per_unit=0.01,
            weight=1.5,
            notes="centimetre asset",
        ),
        AssetSpec(
            "barrel_a",
            "barrel",
            f"{_WAREHOUSE}/SM_BarelPlastic_A_01.usd",
            (-0.3, -0.354, 0.0),
            (0.3, 0.354, 0.901),
            weight=2.0,
        ),
        AssetSpec(
            "barrel_b", "barrel", f"{_WAREHOUSE}/SM_BarelPlastic_B_01.usd", (-0.3, -0.354, 0.0), (0.3, 0.354, 0.588)
        ),
        AssetSpec("pallet", "pallet", "Isaac/Props/Pallet/pallet.usd", (-0.605, -0.400, 0.0), (0.608, 0.402, 0.143)),
        AssetSpec(
            "pallet_warehouse",
            "pallet",
            f"{_WAREHOUSE}/SM_PaletteA_01.usd",
            (-0.605, -0.500, 0.0),
            (0.608, 0.503, 0.211),
        ),
        AssetSpec(
            "traffic_cone",
            "cone",
            f"{_WAREHOUSE}/S_TrafficCone.usd",
            (-0.168, -0.167, 0.0),
            (0.168, 0.167, 0.463),
            scale_range=(1.5, 1.5),
            notes="0.46 m warehouse cone, scaled to a 0.69 m road cone",
        ),
        # trees: centimetre assets; sizes from UsdGeom.BBoxCache (foliage included); the scenario scales them so
        # their top stays well below the survey altitude
        AssetSpec(
            "douglas_fir",
            UNLABELED,
            "NVIDIA/Assets/Vegetation/Trees/Douglas_Fir.usd",
            (-1.393, -1.366, 0.0),
            (1.630, 1.405, 6.026),
            meters_per_unit=0.01,
            scale_range=(0.7, 0.9),
        ),
        AssetSpec(
            "red_ash",
            UNLABELED,
            "NVIDIA/Assets/Vegetation/Trees/Red_Ash.usd",
            (-3.053, -2.918, 0.0),
            (2.680, 3.138, 8.435),
            meters_per_unit=0.01,
            scale_range=(0.55, 0.65),
        ),
        AssetSpec(
            "scarlet_oak",
            UNLABELED,
            "NVIDIA/Assets/Vegetation/Trees/Scarlet_Oak.usd",
            (-4.902, -5.431, -0.344),
            (5.507, 4.826, 12.409),
            meters_per_unit=0.01,
            scale_range=(0.38, 0.45),
        ),
        AssetSpec(
            "white_pine",
            UNLABELED,
            "NVIDIA/Assets/Vegetation/Trees/White_Pine.usd",
            (-1.008, -1.038, -0.351),
            (0.992, 1.085, 1.999),
            meters_per_unit=0.01,
            scale_range=(1.2, 1.5),
        ),
    ]
}
"""All verified assets by key."""


def assets_of(category: str) -> list[AssetSpec]:
    """Assets of one category (a class of :data:`CLASSES` or :data:`UNLABELED`)."""
    return [a for a in ASSETS.values() if a.category == category]


@dataclass(frozen=True)
class GroundMaterial:
    """An OmniPBR-based MDL material used on the terrain (world-space projected texture)."""

    key: str
    path: str
    texture_scale: float = 0.25
    """OmniPBR ``texture_scale`` with ``project_uvw`` and world coordinates (0.4: one texture tile per 2.5 m)."""
    albedo_brightness: float | None = None
    """OmniPBR albedo multiplier (None: the material's own); darkens the bright concrete / gravel in the sun."""

    @property
    def url(self) -> str:
        return f"{asset_root()}/{self.path}"


GROUND_MATERIALS: dict[str, GroundMaterial] = {
    m.key: m
    for m in [
        GroundMaterial("grass", "NVIDIA/Materials/Base/Natural/Grass_Countryside.mdl", 0.4),
        GroundMaterial("soil", "NVIDIA/Materials/Base/Natural/Soil_Rocky.mdl", 0.4, 0.8),
        GroundMaterial("gravel", "NVIDIA/Materials/Base/Stone/Gravel.mdl", 0.5, 0.6),
        GroundMaterial("concrete", "NVIDIA/Materials/Base/Masonry/Concrete_Rough.mdl", 0.4, 0.45),
        GroundMaterial("asphalt", "NVIDIA/Materials/Base/Natural/Asphalt.mdl", 0.5, 0.7),
    ]
}
"""Terrain materials (all ``OmniPBR(*)`` wrappers, so ``project_uvw`` / ``world_or_object`` are inputs)."""

SKY_HDRI = "Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
"""Clear-sky HDRI of the dome light (the same one the built-in drone tasks use)."""


@dataclass(frozen=True)
class AssetCacheInfo:
    """Where remote assets are cached (see :func:`drone_lab.sar.scene.asset_cache_dir`)."""

    env_var: str = "DRONE_LAB_ASSET_CACHE"
    default: str = field(
        default_factory=lambda: os.path.join(os.path.expanduser("~"), ".cache", "drone_lab", "isaac_assets")
    )


ASSET_CACHE = AssetCacheInfo()
