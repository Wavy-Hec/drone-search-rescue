"""Drone views, lighting and ground looks for rendering a detector dataset of the SAR site (numpy only).

``scripts/gen_yolo_dataset.py`` renders the site from many drone poses; this module decides what it renders:

* :func:`sample_views` -- training views: 5-12 m above the ground, up to 15 deg off nadir, any heading; most of
  them aimed at a labelled object (anywhere in the image, sometimes cut by its edge), the rest anywhere over
  the site.
* :func:`mission_views` -- views along the survey path at the survey height, nose along +X and the small tilts
  of the real flight: what the drone sees during the mission.
* :func:`sample_lighting` / :func:`sample_ground_look` -- sun azimuth and strength (the elevation stays at the
  sky image's 43 deg), overcast skies, and swapped or re-tinted ground materials.
* :func:`padded_spec` / :func:`crop` / :func:`unclipped_boxes` -- the camera renders a larger image with the
  same focal length and principal point offset, so the centre crop is exactly the mission camera's image and
  the margin shows how much of a cut-off object lies outside it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .assets import CLASSES, GROUND_MATERIALS
from .camera import SAR_CAMERA, CameraSpec
from .scenario import Scenario, TerrainSite

SUN_AZIMUTH_DEG = 136.0
"""Azimuth of the sun inside the sky HDRI (counter-clockwise from +X), see :func:`drone_lab.sar.scene.sun_light_cfg`."""

SUN_ELEVATION_DEG = 43.0
SUN_INTENSITY = 2000.0
SKY_INTENSITY = 800.0
SUN_COLOR = (1.0, 0.96, 0.9)

TERRAIN_LOOKS = ("grass", "soil", "gravel", "concrete")
"""Ground materials bound to the terrain mesh (the asphalt apron is its own prim and keeps its look)."""


@dataclass(frozen=True)
class View:
    """One drone pose: ``base_link`` position [m] and attitude [rad] (roll, pitch, yaw about world X, Y, Z)."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    agl: float
    """Height of the drone above the terrain under it [m]."""
    kind: str
    """"object" (aimed at ``target``), "site" (anywhere over the site) or "mission" (along the survey path)."""
    target: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("x", "y", "z", "agl"):
            d[key] = round(d[key], 3)
        for key in ("roll", "pitch", "yaw"):
            d[key] = round(d[key], 5)
        return d


@dataclass(frozen=True)
class Lighting:
    """Sun and sky of one frame. Turning the sky dome by ``sky_rotation_deg`` keeps its own sun on the light's."""

    sun_azimuth_deg: float = SUN_AZIMUTH_DEG
    sun_elevation_deg: float = SUN_ELEVATION_DEG
    sun_intensity: float = SUN_INTENSITY
    sky_intensity: float = SKY_INTENSITY
    sun_color: tuple[float, float, float] = SUN_COLOR
    name: str = "default"

    @property
    def sky_rotation_deg(self) -> float:
        return self.sun_azimuth_deg - SUN_AZIMUTH_DEG

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass(frozen=True)
class GroundLook:
    """Material of each terrain look and per-material albedo / texture-scale multipliers."""

    materials: dict[str, str] = field(default_factory=lambda: {k: k for k in TERRAIN_LOOKS})
    """Terrain look (face subset ``ground_<look>``) -> ground material bound to it."""
    brightness: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TERRAIN_LOOKS, 1.0))
    texture_scale: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TERRAIN_LOOKS, 1.0))
    name: str = "default"

    def albedo_brightness(self, material: str) -> float:
        base = GROUND_MATERIALS[material].albedo_brightness
        return (1.0 if base is None else base) * self.brightness[material]

    def scaled_texture(self, material: str) -> float:
        return GROUND_MATERIALS[material].texture_scale * self.texture_scale[material]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "materials": dict(self.materials),
            "brightness": {k: round(v, 3) for k, v in self.brightness.items()},
            "texture_scale": {k: round(v, 3) for k, v in self.texture_scale.items()},
        }


# ---------------------------------------------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------------------------------------------


def footprint_radius(agl: float, spec: CameraSpec = SAR_CAMERA) -> float:
    """Distance from the image centre to an image corner on flat ground ``agl`` metres below a nadir camera."""
    return agl * math.hypot(0.5 * spec.width, 0.5 * spec.height) / spec.focal_px


def random_tilt(rng: np.random.Generator, max_tilt_deg: float) -> tuple[float, float]:
    """Roll and pitch [rad] of a tilt with a uniform magnitude in [0, max] and a uniform direction."""
    tilt = math.radians(float(rng.uniform(0.0, max_tilt_deg)))
    direction = float(rng.uniform(-math.pi, math.pi))
    return tilt * math.cos(direction), tilt * math.sin(direction)


def object_weights(scenario: Scenario, person_weight: float = 1.5) -> np.ndarray:
    """Aim weights of the labelled objects: 1 / sqrt(objects of the class), so rare classes get more views."""
    objs = scenario.labeled()
    counts = {c: sum(1 for o in objs if o.category == c) for c in CLASSES}
    w = np.array([1.0 / math.sqrt(counts[o.category]) for o in objs])
    w *= np.array([person_weight if o.category == "person" else 1.0 for o in objs])
    return w / w.sum()


def _clear_of_trees(scenario: Scenario, x: float, y: float, z: float, margin: float = 1.5) -> float:
    """Raise ``z`` above any tree crown the drone would sit in."""
    for o in scenario.objects:
        if o.labeled:
            continue
        if math.hypot(x - o.center[0], y - o.center[1]) < o.radius + 1.0:
            z = max(z, float(o.aabb_max[2]) + margin)
    return z


def sample_views(
    scenario: Scenario,
    site: TerrainSite,
    n: int,
    rng: np.random.Generator,
    agl: tuple[float, float] = (5.0, 12.0),
    max_tilt_deg: float = 15.0,
    object_fraction: float = 0.65,
    spec: CameraSpec = SAR_CAMERA,
) -> list[View]:
    """Random training views of one site (see the module docstring).

    Aimed views put the target anywhere in the image, up to 10 % past its edges (cut-off objects are part of
    the data). The view centre stays far enough inside the site that at most the corners of a high view show
    the void beyond the terrain.
    """
    objs = scenario.labeled()
    weights = object_weights(scenario) if objs else None
    x0, x1, y0, y1 = site.bounds
    ax0, ax1, ay0, ay1 = scenario.area
    views = []
    for _ in range(n):
        h = float(rng.uniform(*agl))
        roll, pitch = random_tilt(rng, max_tilt_deg)
        yaw = float(rng.uniform(-math.pi, math.pi))
        if objs and rng.uniform() < object_fraction:
            target = objs[int(rng.choice(len(objs), p=weights))]
            # where the target lands in the image: (u, v) in [-0.6, 0.6] of the half footprint, then into the world
            gsd = h / spec.focal_px
            du = float(rng.uniform(-0.6, 0.6)) * spec.width * gsd
            dv = float(rng.uniform(-0.6, 0.6)) * spec.height * gsd
            # image up is the nose (+X body), image right the drone's right (-Y body)
            c, s = math.cos(yaw), math.sin(yaw)
            x = target.center[0] - (dv * c + du * s)
            y = target.center[1] - (dv * s - du * c)
            kind, name = "object", target.name
        else:
            x = float(rng.uniform(ax0 - 4.0, ax1 + 4.0))
            y = float(rng.uniform(ay0 - 4.0, ay1 + 4.0))
            kind, name = "site", None
        inset = 0.75 * footprint_radius(h, spec)
        x = float(np.clip(x, x0 + inset, x1 - inset))
        y = float(np.clip(y, y0 + inset, y1 - inset))
        ground = site.height(x, y)
        z = _clear_of_trees(scenario, x, y, ground + h)
        views.append(View(x, y, z, roll, pitch, yaw, z - ground, kind, name))
    return views


def mission_views(
    scenario: Scenario,
    site: TerrainSite,
    step: float,
    rng: np.random.Generator,
    max_tilt_deg: float = 4.0,
    yaw_sigma_deg: float = 2.0,
) -> list[View]:
    """Views every ``step`` metres along the default lawnmower path at the survey height (the mission's camera)."""
    from drone_lab.paths import Area, cumulative_length, lawnmower, point_at_arclength

    x0, x1, y0, y1 = scenario.area
    path = lawnmower(Area(x0, x1, y0, y1), scenario.cfg.survey_spacing, scenario.flight_z)
    s_cum = cumulative_length(path)
    points = point_at_arclength(path, s_cum, np.arange(0.0, s_cum[-1], step))
    views = []
    for x, y, z in points:
        roll, pitch = random_tilt(rng, max_tilt_deg)
        yaw = math.radians(float(rng.normal(0.0, yaw_sigma_deg)))
        ground = site.height(float(x), float(y))
        views.append(View(float(x), float(y), float(z), roll, pitch, yaw, float(z) - ground, "mission"))
    return views


# ---------------------------------------------------------------------------------------------------------------
# lighting and ground looks
# ---------------------------------------------------------------------------------------------------------------


def sample_lighting(rng: np.random.Generator, default_fraction: float = 0.3, overcast_fraction: float = 0.15):
    """The mission's sun (``default_fraction``), an overcast sky, or a sun from another azimuth and strength."""
    u = float(rng.uniform())
    if u < default_fraction:
        return Lighting()
    warm = float(rng.uniform(0.0, 1.0))
    color = (1.0, 1.0 - 0.08 * warm, 1.0 - 0.2 * warm)
    azimuth = float(rng.uniform(0.0, 360.0))
    if u < default_fraction + overcast_fraction:
        return Lighting(
            sun_azimuth_deg=azimuth,
            sun_intensity=SUN_INTENSITY * float(rng.uniform(0.1, 0.3)),
            sky_intensity=SKY_INTENSITY * float(rng.uniform(1.2, 1.6)),
            sun_color=color,
            name="overcast",
        )
    return Lighting(
        sun_azimuth_deg=azimuth,
        sun_intensity=SUN_INTENSITY * float(rng.uniform(0.7, 1.3)),
        sky_intensity=SKY_INTENSITY * float(rng.uniform(0.7, 1.3)),
        sun_color=color,
        name="sun",
    )


def sample_ground_look(rng: np.random.Generator, swap_fraction: float = 0.5) -> GroundLook:
    """The site's own ground materials, or (``swap_fraction``) a shuffled assignment; both with random tints."""
    looks = list(TERRAIN_LOOKS)
    swapped = bool(rng.uniform() < swap_fraction)
    order = [looks[i] for i in rng.permutation(len(looks))] if swapped else looks
    return GroundLook(
        materials=dict(zip(looks, order, strict=True)),
        brightness={k: float(rng.uniform(0.8, 1.2)) for k in looks},
        texture_scale={k: float(rng.uniform(0.8, 1.25)) for k in looks},
        name="swapped" if swapped else "tinted",
    )


# ---------------------------------------------------------------------------------------------------------------
# padded camera
# ---------------------------------------------------------------------------------------------------------------


def padded_spec(spec: CameraSpec, pad_x: int, pad_y: int) -> CameraSpec:
    """Camera with ``pad`` extra pixels on every side, same focal length and centred principal point."""
    return CameraSpec(
        width=spec.width + 2 * pad_x,
        height=spec.height + 2 * pad_y,
        focal_px=spec.focal_px,
        clip_near=spec.clip_near,
        clip_far=spec.clip_far,
        mount_pos=spec.mount_pos,
        mount_rot_ros=spec.mount_rot_ros,
    )


def crop(image: np.ndarray, pad_x: int, pad_y: int, spec: CameraSpec = SAR_CAMERA) -> np.ndarray:
    """Centre crop of a padded render: exactly the image of ``spec``."""
    return image[pad_y : pad_y + spec.height, pad_x : pad_x + spec.width]


def unclipped_boxes(
    seg_padded: np.ndarray, pad_x: int, pad_y: int, spec: CameraSpec = SAR_CAMERA
) -> dict[int, tuple[int, int, int, int]]:
    """Full boxes (crop pixel coordinates, inclusive, may lie outside the crop) of the instances cut by its edge.

    Only instances that are visible in the crop and touch its border are returned: the others are whole. An
    instance that also touches the edge of the padded render is cut there too, so its box is a lower bound.
    """
    seg = np.asarray(seg_padded)
    seg = seg.reshape(seg.shape[:2])
    inner = crop(seg, pad_x, pad_y, spec)
    edge = np.unique(np.concatenate([inner[0], inner[-1], inner[:, 0], inner[:, -1]]))
    boxes = {}
    for i in edge.tolist():
        rows = np.flatnonzero((seg == i).any(axis=1))
        cols = np.flatnonzero((seg == i).any(axis=0))
        boxes[int(i)] = (int(cols[0]) - pad_x, int(rows[0]) - pad_y, int(cols[-1]) - pad_x, int(rows[-1]) - pad_y)
    return boxes
