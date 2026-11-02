"""Isaac Lab spawn helpers of the SAR site: the whole dressed site as one scene entity, plus camera / light configs.

Use in a scene config (any scene that also imports the survey terrain at ``/World/ground``)::

    sar_site = AssetBaseCfg(prim_path="/World/sar", spawn=SarSiteCfg(seed=0))
    sar_camera = sar_camera_cfg()
    sun_light = sun_light_cfg()

:func:`spawn_sar_site` regenerates the seeded :class:`~drone_lab.sar.scenario.Scenario` on the CPU (the terrain
generator is deterministic, so it matches the imported terrain; this is checked against the stage mesh), then

* textures the terrain with several MDL ground materials (grass / soil / gravel / concrete / asphalt, bound to face
  subsets of the terrain mesh by tile type and slope),
* references every object USD at ``<prim_path>/objects/<name>`` with its pose and scale (unit fix included),
  replaces any semantic labels the asset brought along and applies ``class: <category>`` (trees get none); with
  ``people_style="natural"`` each person is a static posed copy of its character, baked on first use into the
  asset cache (:class:`drone_lab.sar.people.PoseBaker`, ``<cache>/posed_people``),
* builds the procedural landing pad (base disk with a collider, yellow ring, white "H"),
* remembers the scenario (:func:`get_spawned_scenario`) and stores its ground truth as custom data on the prim.

Remote assets are localised with Isaac Lab's ``retrieve_file_path(..., download_dir=...)`` into a persistent cache
(:func:`asset_cache_dir`, default ``~/.cache/drone_lab/isaac_assets``, override with ``DRONE_LAB_ASSET_CACHE``;
set it to ``tmp`` for Isaac Lab's default ``/tmp``). Every copy there is checked against the server's revision
before use, so a stale cache cannot win; offline runs fall back to the local copy with a warning. The rewritten
per-run layer copies are pruned after a few hours unless a live process still uses them
(:func:`prune_cache_copies`, :func:`hold_cache_copy`).

Kit / Isaac Lab modules are imported inside the functions, so this module imports without a simulator.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

try:
    import fcntl
except ImportError:  # not POSIX: no in-use locks, the age rule alone protects the cached layer copies
    fcntl = None

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from .assets import ASSET_CACHE, ASSETS, GROUND_MATERIALS, SEMANTIC_TYPE, SKY_HDRI, UNLABELED, asset_root
from .camera import SAR_CAMERA, CameraSpec
from .people import POSES, PoseBaker
from .scenario import OBJECTS_SUBPATH, Scenario, ScenarioCfg, TerrainSite, generate_scenario

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import Camera, CameraCfg

logger = logging.getLogger(__name__)

_SPAWNED: dict[str, Scenario] = {}
"""Scenarios spawned in this process, by site prim path."""

_LOCALIZED: dict[str, str] = {}
"""Remote URL -> local localised copy (one download / rewrite per asset per process)."""

_BAKERS: dict[str, PoseBaker] = {}
"""Pose bakers by cache folder (they keep the animation rig and the characters loaded)."""

CACHE_COPY_PREFIX = "isaaclab_usd_"
"""Folders of the rewritten USD layers that ``retrieve_file_path`` writes for every retrieval."""

IN_USE_LOCK = ".drone_lab_in_use.lock"
"""Lock file inside a layer-copy folder; every process that references the folder holds a shared lock on it."""

_HELD_COPIES: dict[str, int] = {}
"""Layer-copy folders this process references -> descriptor holding the shared lock (released at process exit)."""

GROUND_TRUTH_CUSTOM_DATA_KEY = "drone_lab:sar_ground_truth"


# ---------------------------------------------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------------------------------------------


@configclass
class SarSiteCfg(SpawnerCfg):
    """Spawner of the whole SAR site (see the module docstring). Fields mirror :class:`ScenarioCfg`."""

    func: Callable | str = "drone_lab.sar.scene:spawn_sar_site"

    seed: int = 0
    """Scenario seed: same seed, same site."""
    num_people: tuple[int, int] = (6, 10)
    lying_fraction: float = 0.5
    near_block_fraction: float = 0.4
    people_style: str = "tpose"
    """``tpose``: the characters' rest T-pose; ``natural``: victims lying, sitting or waving and responders
    looking around, pointing or walking (``env.scene.sar_site.spawn.people_style=natural``). The other objects are
    the same either way."""
    num_responders: tuple[int, int] = (2, 3)
    """Responders by the vehicle and the landing pad (``natural`` only)."""
    num_crates: tuple[int, int] = (7, 10)
    num_barrels: tuple[int, int] = (4, 6)
    num_pallets: tuple[int, int] = (3, 5)
    num_site_cones: tuple[int, int] = (2, 4)
    num_trees: tuple[int, int] = (4, 6)
    trees_inside_area: int = 1
    home_xy: tuple[float, float] | None = None
    survey_clearance: float = 4.0
    """Must match ``commands.survey_target.altitude`` (trees are kept below the resulting flight height)."""
    survey_spacing: float = 4.0

    terrain_generator: object | None = None
    """Terrain generator config of the scene's terrain (None: ``SURVEY_TERRAIN_CFG``). Keep it in sync."""
    terrain_prim_path: str = "/World/ground"
    ground_materials: bool = True
    """Texture the terrain with MDL ground materials (False keeps the generator's vertex colours)."""
    label_objects: bool = True
    """Apply ``class`` semantic labels (and strip the labels the assets came with)."""
    asset_cache_dir: str | None = None
    """Download directory for remote assets; None: :func:`asset_cache_dir`."""

    def scenario_cfg(self) -> ScenarioCfg:
        """The scenario settings of this site (what :func:`scenario_for` and the spawner place)."""
        return ScenarioCfg(
            seed=int(self.seed),
            num_people=tuple(self.num_people),
            lying_fraction=float(self.lying_fraction),
            near_block_fraction=float(self.near_block_fraction),
            people_style=str(self.people_style),
            num_responders=tuple(self.num_responders),
            num_crates=tuple(self.num_crates),
            num_barrels=tuple(self.num_barrels),
            num_pallets=tuple(self.num_pallets),
            num_site_cones=tuple(self.num_site_cones),
            num_trees=tuple(self.num_trees),
            trees_inside_area=int(self.trees_inside_area),
            home_xy=None if self.home_xy is None else tuple(self.home_xy),
            survey_clearance=float(self.survey_clearance),
            survey_spacing=float(self.survey_spacing),
        )


def scenario_for(cfg: SarSiteCfg, site: TerrainSite | None = None) -> Scenario:
    """The scenario a :class:`SarSiteCfg` spawns (CPU only; identical to what :func:`spawn_sar_site` places)."""
    site = site if site is not None else TerrainSite(cfg.terrain_generator)
    return generate_scenario(cfg.scenario_cfg(), site=site)


def sar_camera_cfg(
    spec: CameraSpec = SAR_CAMERA,
    prim_path: str = "{ENV_REGEX_NS}/Robot/base_link/sar_camera",
    data_types: tuple[str, ...] = ("rgb", "distance_to_image_plane", "instance_segmentation"),
    antialiasing: str | None = "FXAA",
) -> CameraCfg:
    """Downward RGB-D camera with instance segmentation that carries the ``class`` labels.

    ``instance_segmentation`` (Replicator's ``instance_segmentation_fast``) gives one instance id per labelled
    object (unlabelled children merge into their labelled parent; terrain, trees and the drone are BACKGROUND /
    UNLABELLED). It is returned raw (int32 ids, not colourised); ``camera.data.info["instance_segmentation"]``
    holds ``idToLabels`` (id -> prim path of the object) and ``idToSemantics`` (id -> {"class": ...}).
    (``instance_id_segmentation_fast`` is the renderer's raw per-mesh id in Isaac Sim 6.2: every prim, terrain
    included, and no semantics, so it is not used.) Scene partitioning is off, so the camera sees the global site
    prims like any other geometry.

    ``antialiasing`` is a process-wide RTX setting. The kit's default DLSS renders at a lower internal resolution
    and upscales: it blurs the ground textures and, with a teleported or jittering camera, smears high-contrast
    details into screen-aligned bands (seen over the landing pad in the GPU preview). FXAA renders at native
    resolution (sharper, no streaks) for ~20 % more time per step; None keeps the kit's setting. ("DLAA" fails in
    Isaac Sim 6.2 on this machine: "Failed to create feature: 13".)
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=spec.width,
        height=spec.height,
        data_types=list(data_types),
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=spec.focal_length,
            horizontal_aperture=20.955,
            clipping_range=(spec.clip_near, spec.clip_far),
        ),
        offset=CameraCfg.OffsetCfg(pos=spec.mount_pos, rot=spec.mount_rot_ros, convention="ros"),
        renderer_cfg=IsaacRtxRendererCfg(
            global_settings=IsaacRtxRendererGlobalSettingsCfg(antialiasing_mode=antialiasing),
            semantic_filter=f"{SEMANTIC_TYPE}:*",
            colorize_instance_segmentation=False,
            colorize_instance_id_segmentation=False,
            colorize_semantic_segmentation=False,
            enable_scene_partitioning=False,
        ),
    )


def look_at_quat_ros(
    eye: ArrayLike, target: ArrayLike, up: ArrayLike = (0.0, 0.0, 1.0)
) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) of a ROS-convention camera at ``eye`` looking at ``target``."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:  # looking straight up/down: keep image-up along world +Y
        r = np.array([1.0, 0.0, 0.0]) if f[2] < 0 else np.array([-1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return _matrix_to_quat(np.column_stack([r, d, f]))


def _matrix_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    t = np.trace(m)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w, x, y, z = 0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def fixed_camera_cfg(
    prim_path: str,
    eye: ArrayLike,
    target: ArrayLike,
    width: int,
    height: int,
    hfov_deg: float,
    data_types: Sequence[str] = ("rgb", "instance_segmentation"),
    antialiasing: str | None = "FXAA",
) -> CameraCfg:
    """A world-fixed camera (e.g. overview or top-down photo camera) looking from ``eye`` at ``target``.

    Its RTX global settings must equal those of every other camera in the process (Isaac Lab checks it); the
    default matches :func:`sar_camera_cfg`.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

    f_px = 0.5 * width / math.tan(math.radians(0.5 * hfov_deg))
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=width,
        height=height,
        data_types=list(data_types),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=f_px / width * 20.955, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e4)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in eye), rot=look_at_quat_ros(eye, target), convention="ros"
        ),
        renderer_cfg=IsaacRtxRendererCfg(
            global_settings=IsaacRtxRendererGlobalSettingsCfg(antialiasing_mode=antialiasing),
            semantic_filter=f"{SEMANTIC_TYPE}:*",
            colorize_instance_segmentation=False,
            colorize_instance_id_segmentation=False,
            colorize_semantic_segmentation=False,
            enable_scene_partitioning=False,
        ),
    )


def sun_light_cfg(intensity: float = 2000.0, elevation_deg: float = 43.0, azimuth_deg: float = 136.0) -> AssetBaseCfg:
    """Distant "sun" light. Elevation above the horizon; azimuth of the sun, counter-clockwise from +X.

    The defaults match the sun inside :data:`~drone_lab.sar.assets.SKY_HDRI` (dome light at identity rotation):
    with any other direction every object casts a second shadow.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    # a distant light shines along its local -Z: tilt it away from the zenith by (90 - elevation) towards the sun
    tilt = math.radians(90.0 - elevation_deg)
    az = math.radians(azimuth_deg)
    # rotate about the horizontal axis perpendicular to the sun direction (counter-clockwise view from above)
    axis = np.array([-math.sin(az), math.cos(az), 0.0])
    q = (*(axis * math.sin(0.5 * tilt)), math.cos(0.5 * tilt))
    return AssetBaseCfg(
        prim_path="/World/sunLight",
        spawn=sim_utils.DistantLightCfg(intensity=intensity, angle=0.53, color=(1.0, 0.96, 0.9)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=tuple(float(v) for v in q)),
    )


def sky_light_cfg(intensity: float = 1000.0) -> AssetBaseCfg:
    """Dome light with the clear-sky HDRI (same texture as the built-in drone tasks)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=intensity, texture_file=f"{asset_root()}/{SKY_HDRI}"),
    )


# ---------------------------------------------------------------------------------------------------------------
# asset cache
# ---------------------------------------------------------------------------------------------------------------


def asset_cache_dir(override: str | None = None) -> str | None:
    """Persistent download directory for remote assets, or None for Isaac Lab's default (the temp dir)."""
    value = override if override is not None else os.environ.get(ASSET_CACHE.env_var, ASSET_CACHE.default)
    if not value or value.lower() in ("tmp", "none", "default"):
        return None
    path = os.path.abspath(os.path.expanduser(value))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        logger.warning("Asset cache '%s' is not usable (%s); using Isaac Lab's default.", path, exc)
        return None
    return path


def _copy_folder(cache_dir: str | None, local_path: str) -> str | None:
    """The ``isaaclab_usd_*`` folder of the cache that holds ``local_path`` (None for raw downloads)."""
    if not cache_dir:
        return None
    rel = os.path.relpath(os.path.abspath(local_path), os.path.abspath(cache_dir))
    top = rel.split(os.sep, 1)[0]
    if rel.startswith("..") or not top.startswith(CACHE_COPY_PREFIX):
        return None
    return os.path.join(os.path.abspath(cache_dir), top)


def hold_cache_copy(cache_dir: str | None, local_path: str) -> bool:
    """Mark the layer-copy folder behind ``local_path`` as used by this process, for the rest of its life.

    Takes a shared ``flock`` on the folder's :data:`IN_USE_LOCK` and keeps the descriptor open; the OS drops the
    lock when the process ends, crashed or not. :func:`prune_cache_copies` (in any process) skips locked folders,
    so a demo or training run keeps its layers however long it runs. Returns whether the folder is held.
    """
    folder = _copy_folder(cache_dir, local_path)
    if folder is None or fcntl is None:
        return False
    if folder in _HELD_COPIES:
        return True
    try:
        fd = os.open(os.path.join(folder, IN_USE_LOCK), os.O_RDONLY | os.O_CREAT, 0o644)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    _HELD_COPIES[folder] = fd
    return True


def prune_cache_copies(cache_dir: str | None, max_age_hours: float = 6.0) -> int:
    """Delete the rewritten layer copies (``isaaclab_usd_*``) older than ``max_age_hours`` that no process uses.

    ``retrieve_file_path`` writes every rewritten USD layer into a fresh ``isaaclab_usd_<uuid>`` folder (about
    30 MB per site spawn); under ``/tmp`` a reboot clears them, in a persistent cache they would pile up. A folder
    that a live process holds (:func:`hold_cache_copy`) is kept at any age; raw downloads are always kept.
    Returns the number of folders removed.
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    removed, now = 0, time.time()
    for name in os.listdir(cache_dir):
        path = os.path.join(os.path.abspath(cache_dir), name)
        if not name.startswith(CACHE_COPY_PREFIX) or path in _HELD_COPIES or not os.path.isdir(path):
            continue
        try:
            if now - os.path.getmtime(path) <= max_age_hours * 3600.0:
                continue
        except OSError:
            continue
        fd = None
        lock = os.path.join(path, IN_USE_LOCK)
        if fcntl is not None and os.path.exists(lock):
            try:
                fd = os.open(lock, os.O_RDONLY)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # fails while any process holds its shared lock
            except OSError:
                if fd is not None:
                    os.close(fd)
                continue
        try:
            shutil.rmtree(path, ignore_errors=True)  # under the exclusive lock: nobody can take the folder now
            removed += 1
        finally:
            if fd is not None:
                os.close(fd)
    return removed


def localize(url: str, cache_dir: str | None) -> str:
    """Local copy of a remote USD (dependencies rewritten to local copies), cached and held per process."""
    from isaaclab.utils.assets import retrieve_file_path

    key = f"{cache_dir}|{url}"
    if key not in _LOCALIZED:
        _LOCALIZED[key] = retrieve_file_path(url, download_dir=cache_dir)
        hold_cache_copy(cache_dir, _LOCALIZED[key])
    return _LOCALIZED[key]


def pose_baker(cache_dir: str | None) -> PoseBaker:
    """Baker of the natural people, writing to ``<cache>/posed_people`` (one per cache, kept for the process)."""
    key = f"baker|{cache_dir}"
    if key not in _BAKERS:
        out = os.path.join(cache_dir, "posed_people") if cache_dir else None
        _BAKERS[key] = PoseBaker(lambda url: localize(url, cache_dir), asset_root(), out)
    return _BAKERS[key]


# ---------------------------------------------------------------------------------------------------------------
# spawning
# ---------------------------------------------------------------------------------------------------------------


def read_camera(cam: Camera, env_index: int = 0, data_types: tuple[str, ...] | None = None) -> dict:
    """One environment's frame of an Isaac Lab camera as numpy arrays.

    Returns ``{"rgb" (H, W, 3) uint8, "distance_to_image_plane" (H, W) float32, "instance_segmentation"
    (H, W) int32, ...}`` for the requested (default: all configured) data types, plus ``"info"`` (the
    segmentation's ``idToLabels`` / ``idToSemantics``), ``"pos_w"``, ``"quat_w_ros"`` (x, y, z, w) and ``"K"``.
    """
    import torch

    def to_np(x):
        x = getattr(x, "torch", x)
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    data = cam.data
    names = data_types or tuple(cam.cfg.data_types)
    out = {}
    for name in names:
        if name in data.output:
            arr = to_np(data.output[name])[env_index]
            out[name] = arr[..., 0] if arr.ndim == 3 and arr.shape[-1] == 1 else arr
    if "rgb" in out:
        out["rgb"] = out["rgb"][..., :3].astype(np.uint8)
    info = data.info if isinstance(data.info, dict) else (data.info[env_index] if data.info else {})
    seg_info = {}
    for key in ("instance_segmentation", "instance_id_segmentation_fast", "semantic_segmentation"):
        val = info.get(key) if isinstance(info, dict) else None
        if isinstance(val, list):
            val = val[env_index] if val else None
        if val:
            seg_info = val
            break
    out["info"] = seg_info or {}
    out["pos_w"] = to_np(data.pos_w)[env_index].astype(float)
    out["quat_w_ros"] = to_np(data.quat_w_ros)[env_index].astype(float)
    out["K"] = to_np(data.intrinsic_matrices)[env_index].astype(float)
    return out


def get_spawned_scenario(prim_path: str = "/World/sar") -> Scenario | None:
    """The scenario spawned at ``prim_path`` in this process (None if nothing was spawned there)."""
    return _SPAWNED.get(prim_path)


def spawn_sar_site(
    prim_path: str,
    cfg: SarSiteCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn the dressed site (see the module docstring). Returns the site root prim."""
    from pxr import UsdGeom

    import isaaclab.sim as sim_utils

    t0 = time.time()
    stage = sim_utils.get_current_stage()
    site = TerrainSite(cfg.terrain_generator)
    scenario = generate_scenario(cfg.scenario_cfg(), site=site)
    _check_stage_terrain(site, cfg.terrain_prim_path)

    root = stage.DefinePrim(prim_path, "Xform")
    stage.DefinePrim(f"{prim_path}/{OBJECTS_SUBPATH}", "Xform")
    cache_dir = asset_cache_dir(cfg.asset_cache_dir)
    pruned = prune_cache_copies(cache_dir)

    if cfg.ground_materials:
        try:
            apply_ground_materials(site, scenario, cfg.terrain_prim_path, f"{prim_path}/Looks")
            spawn_apron(f"{prim_path}/apron", scenario.home, scenario.cfg.apron_half)
        except Exception as exc:  # textures are cosmetic: never lose the scene over them
            logger.warning("Ground materials failed (%s); keeping the vertex colours.", exc)

    for obj in scenario.objects:
        path = obj.prim_path(prim_path)
        if obj.asset == "landing_pad":
            prim = spawn_landing_pad(path, obj.position, radius=obj.half_extents[0])
        else:
            spec = ASSETS[obj.asset]
            if obj.category == "person" and obj.pose in POSES:
                local = pose_baker(cache_dir).usd(spec.key, spec.path, obj.pose)
            else:
                local = localize(spec.url, cache_dir)
            prim = stage.DefinePrim(path, "Xform")
            if not prim.GetReferences().AddReference(local):
                raise RuntimeError(f"could not reference {local} at {path}")
            s = obj.scale * obj.unit_scale
            sim_utils.standardize_xform_ops(prim, translation=obj.position, orientation=obj.quat_xyzw, scale=(s, s, s))
        if cfg.label_objects:
            clear_semantics(prim)
            if obj.labeled:
                sim_utils.add_labels(prim, labels=[obj.category], instance_name=SEMANTIC_TYPE)
        UsdGeom.Imageable(prim).MakeVisible()

    gt = scenario.ground_truth(prim_path)
    root.SetCustomDataByKey(GROUND_TRUTH_CUSTOM_DATA_KEY, json.dumps(gt))
    _SPAWNED[prim_path] = scenario
    counts = {k: v for k, v in scenario.counts().items() if v}
    logger.info(
        "SAR site seed %d (%s people) spawned at %s in %.1f s: %s (asset cache %s, %d stale copies pruned)",
        scenario.cfg.seed,
        scenario.cfg.people_style,
        prim_path,
        time.time() - t0,
        counts,
        cache_dir or "Isaac Lab default",
        pruned,
    )
    print(
        f"[drone_lab.sar] site seed {scenario.cfg.seed} ({scenario.cfg.people_style} people): {counts} spawned in "
        f"{time.time() - t0:.1f} s (asset cache: {cache_dir or 'Isaac Lab default (/tmp)'})",
        flush=True,
    )
    return root


def clear_semantics(prim: Usd.Prim) -> int:
    """Remove the semantic labels an asset brought along (new ``LabelsAPI`` and legacy ``Semantics``)."""
    from pxr import Usd

    n = 0
    for p in Usd.PrimRange(prim):
        for schema in list(p.GetAppliedSchemas()):
            if schema.startswith("SemanticsLabelsAPI:") or schema.startswith("SemanticsAPI:"):
                p.RemoveAppliedSchema(schema)
                n += 1
        for attr in p.GetAttributes():
            name = attr.GetName()
            if name.startswith("semantics:labels:") or (name.startswith("semantic:") and ":params:" in name):
                attr.Block()
    return n


def spawn_landing_pad(path: str, position: Sequence[float], radius: float = 1.25) -> Usd.Prim:
    """Procedural helipad: dark disk (collider) with a yellow ring and a white "H"; returns its root prim."""
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    root = stage.DefinePrim(path, "Xform")
    sim_utils.standardize_xform_ops(root, translation=tuple(position), orientation=(0.0, 0.0, 0.0, 1.0))
    h = 0.04

    def mat(rgb, rough=0.7):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=rough, metallic=0.0)

    parts = [
        (
            "base",
            sim_utils.CylinderCfg(
                radius=radius,
                height=h,
                visual_material=mat((0.06, 0.07, 0.08)),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
            (0.0, 0.0, 0.5 * h),
        ),
        (
            "ring",
            sim_utils.CylinderCfg(radius=0.9 * radius, height=0.004, visual_material=mat((0.95, 0.72, 0.05))),
            (0.0, 0.0, h + 0.002),
        ),
        (
            "inner",
            sim_utils.CylinderCfg(radius=0.8 * radius, height=0.004, visual_material=mat((0.06, 0.07, 0.08))),
            (0.0, 0.0, h + 0.004),
        ),
        (
            "h_left",
            sim_utils.CuboidCfg(size=(0.16, 0.9, 0.004), visual_material=mat((0.95, 0.95, 0.95))),
            (-0.3, 0.0, h + 0.007),
        ),
        (
            "h_right",
            sim_utils.CuboidCfg(size=(0.16, 0.9, 0.004), visual_material=mat((0.95, 0.95, 0.95))),
            (0.3, 0.0, h + 0.007),
        ),
        (
            "h_bar",
            sim_utils.CuboidCfg(size=(0.6, 0.16, 0.004), visual_material=mat((0.95, 0.95, 0.95))),
            (0.0, 0.0, h + 0.007),
        ),
    ]
    for name, cfg, offset in parts:
        cfg.func(f"{path}/{name}", cfg, translation=offset)
    return root


def spawn_apron(path: str, home: Sequence[float], half: float, thickness: float = 0.004) -> Usd.Prim:
    """Asphalt square around the landing pad (visual only, a few millimetres above the flat border)."""
    import isaaclab.sim as sim_utils

    mat = GROUND_MATERIALS["asphalt"]
    cfg = sim_utils.CuboidCfg(
        size=(2.0 * half, 2.0 * half, thickness),
        visual_material=_ground_mdl_cfg(mat.url, mat.texture_scale, mat.albedo_brightness),
    )
    return cfg.func(path, cfg, translation=(float(home[0]), float(home[1]), float(home[2]) + 0.5 * thickness))


def _check_stage_terrain(site: TerrainSite, terrain_prim_path: str) -> None:
    """Warn if the imported terrain differs from the CPU copy the objects were placed on."""
    try:
        from drone_lab.tasks.survey.mdp.terrain_utils import terrain_mesh_from_stage

        mesh = terrain_mesh_from_stage(terrain_prim_path)
    except Exception as exc:
        logger.warning("Could not read the terrain from the stage (%s); placement uses the CPU terrain.", exc)
        return
    if mesh is None:
        logger.warning("No terrain mesh under %s yet; placement uses the CPU terrain.", terrain_prim_path)
        return
    v, f = mesh
    if v.shape != site.vertices.shape or f.shape != site.faces.shape:
        logger.warning(
            "Stage terrain %s/%s differs from the CPU terrain %s/%s: objects may float or sink.",
            v.shape,
            f.shape,
            site.vertices.shape,
            site.faces.shape,
        )
        return
    err = float(np.abs(v - site.vertices).max())
    if err > 1e-3:
        logger.warning("Stage terrain differs from the CPU terrain by up to %.3f m.", err)


def terrain_face_materials(
    site: TerrainSite, scenario: Scenario, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Ground material key of every terrain face (array of str), from its tile type, height and slope."""
    tri = vertices[faces]
    c = tri.mean(axis=1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nz = np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    a = site.area
    tsx, tsy = site.tile_size
    inside = (c[:, 0] >= a.x_min) & (c[:, 0] < a.x_max) & (c[:, 1] >= a.y_min) & (c[:, 1] < a.y_max)
    row = np.clip(((c[:, 0] - a.x_min) // tsx).astype(int), 0, 10_000)
    col = np.clip(((c[:, 1] - a.y_min) // tsy).astype(int), 0, 10_000)
    out = np.full(len(faces), "grass", dtype=object)
    by_tile = {
        "hills": "grass",
        "waves": "grass",
        "pyramid": "soil",
        "pit": "soil",
        "rough": "gravel",
        "stairs": "concrete",
        "blocks": "gravel",
    }
    for (r, cc), name in site.tiles.items():
        sel = inside & (row == r) & (col == cc)
        out[sel] = by_tile.get(name, "soil")
        if name == "blocks":  # block tops and walls are concrete, the ground between them is rubble
            out[sel & ((c[:, 2] > 0.05) | (nz < 0.9))] = "concrete"
    # the border is a few huge triangles, so the asphalt apron around the pad is its own mesh (spawn_apron)
    return out


def _ground_mdl_cfg(mdl_path: str, texture_scale: float, albedo_brightness: float | None = None):
    """OmniPBR-based MDL material with a world-space projected texture (``project_uvw`` + ``world_or_object``)."""
    from isaaclab.sim.spawners.materials import MdlFileCfg

    @configclass
    class GroundMdlCfg(MdlFileCfg):
        world_or_object: bool | None = True
        """OmniPBR input: project the texture in world space, so neighbouring faces line up."""

    return GroundMdlCfg(
        mdl_path=mdl_path,
        project_uvw=True,
        texture_scale=(texture_scale, texture_scale),
        albedo_brightness=albedo_brightness,
    )


def apply_ground_materials(site: TerrainSite, scenario: Scenario, terrain_prim_path: str, looks_path: str) -> dict:
    """Bind the ground materials to face subsets of the terrain mesh; returns {material: face count}."""
    from pxr import Usd, UsdGeom, UsdShade

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    mesh_prim = sim_utils.get_first_matching_child_prim(terrain_prim_path, lambda p: p.GetTypeName() == "Mesh")
    if mesh_prim is None or not mesh_prim.IsValid():
        raise RuntimeError(f"no terrain mesh under {terrain_prim_path}")
    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    world = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    points = points @ world[:3, :3].T + world[:3, 3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise RuntimeError("terrain mesh is not triangulated")
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    keys = terrain_face_materials(site, scenario, points, faces)
    stage.DefinePrim(looks_path, "Scope")
    binding = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    used = {}
    for key in sorted(set(keys.tolist())):
        idx = np.nonzero(keys == key)[0]
        mat = GROUND_MATERIALS[key]
        mat_path = f"{looks_path}/ground_{key}"
        mcfg = _ground_mdl_cfg(mat.url, mat.texture_scale, mat.albedo_brightness)
        mcfg.func(mat_path, mcfg)
        subset = binding.CreateMaterialBindSubset(f"ground_{key}", idx.astype(np.int32).tolist(), UsdGeom.Tokens.face)
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)), UsdShade.Tokens.strongerThanDescendants
        )
        used[key] = int(len(idx))
    binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)
    return used


def ground_truth_from_stage(prim_path: str = "/World/sar") -> dict | None:
    """Read the ground truth the spawner stored on the site prim (works for a saved stage too)."""
    import isaaclab.sim as sim_utils

    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    raw = prim.GetCustomDataByKey(GROUND_TRUTH_CUSTOM_DATA_KEY)
    return json.loads(raw) if raw else None


__all__ = [
    "SarSiteCfg",
    "apply_ground_materials",
    "asset_cache_dir",
    "clear_semantics",
    "fixed_camera_cfg",
    "get_spawned_scenario",
    "ground_truth_from_stage",
    "hold_cache_copy",
    "localize",
    "look_at_quat_ros",
    "pose_baker",
    "prune_cache_copies",
    "read_camera",
    "sar_camera_cfg",
    "scenario_for",
    "sky_light_cfg",
    "spawn_apron",
    "spawn_landing_pad",
    "spawn_sar_site",
    "sun_light_cfg",
    "terrain_face_materials",
    "UNLABELED",
]
