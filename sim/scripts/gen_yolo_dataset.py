"""Render a detector dataset of the SAR site in Isaac Sim: drone-camera RGB + instance segmentation (GPU).

One simulator run renders many sites. A site is the survey terrain dressed by one scenario seed: the seed places the
people and objects, the terrain is the same for every seed. For every seed the objects are respawned in place and
the drone is put at sampled views (:mod:`drone_lab.sar.capture`) under varied light and ground looks. Frames go to
one folder per split, in the capture layout of the ground station's dataset tools
(``drone_gcs/scripts/seg_to_yolo.py``):

    <out>/train/   --train_seeds (default 1000-1035), --frames random views per site
    <out>/val/     --val_seeds   (default 2000-2006): other sites, held out from training
    <out>/test/    --test_seeds  (default 0, the demo mission's site): views along the survey path, mission light

Seeds below 1000 never reach train or val, so the layout of any site picked with ``--seed`` in the other scripts is
unseen.

Per frame ``rgb_<key>.png``, ``instance_<key>.png`` (16-bit instance ids) and ``instance_<key>.json`` (id -> prim
path and class, full boxes of the objects cut by the image edge, drone view, light). Per site ``seed_<n>.json``
(objects, ground look, label checks, timing); a site that has it is skipped when the run is repeated.

The camera renders 960 x 720 px with the focal length of the mission camera. The centre 640 x 480 crop is exactly
the mission camera's image; the margin shows the full extent of every object cut by the crop's edge, so the
converter can drop the ones that are mostly outside.

Examples (env_isaaclab, from the project root)::

    python scripts/gen_yolo_dataset.py                       # full dataset: 2118 frames, about 7 min
    python scripts/gen_yolo_dataset.py --train_seeds 1000 --val_seeds 2000 --frames 6 --out /tmp/sar_capture_smoke

Then, in the ``vision`` env (classes in the order of :data:`drone_lab.sar.assets.CLASSES`)::

    python ~/isaac_ws/projects/drone_gcs/scripts/seg_to_yolo.py --src <out> --out <yolo dataset> \\
        --classes person,vehicle,crate,barrel,pallet,cone,landing_pad
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from drone_lab.sar.capture import (  # noqa: E402
    TERRAIN_LOOKS,
    GroundLook,
    Lighting,
    View,
    crop,
    mission_views,
    padded_spec,
    sample_ground_look,
    sample_lighting,
    sample_views,
    unclipped_boxes,
)

TASK = "DroneLab-SAR-ARL-Robot-1-v0"
SEG = "instance_segmentation"
SITE = "/World/sar"
SITES_ROOT = "/World/dataset_sites"
"""Parent of the respawned sites (the first site stays at :data:`SITE`, whose ground materials every site uses)."""
SUN_PRIM = "/World/sunLight"
SKY_PRIM = "/World/skyLight"
SPLITS = ("train", "val", "test")


def parse_seeds(text: str) -> list[int]:
    """``"1000-1003,1010"`` -> ``[1000, 1001, 1002, 1003, 1010]``."""
    seeds: list[int] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="~/isaac_ws/datasets/drone_sar_yolo/capture", help="Capture folder.")
    parser.add_argument("--train_seeds", default="1000-1035", help="Sites of the training split, e.g. 1000-1035.")
    parser.add_argument("--val_seeds", default="2000-2006", help="Held-out sites of the validation split.")
    parser.add_argument("--test_seeds", default="0", help="Sites rendered along the mission path ('' for none).")
    parser.add_argument("--frames", type=int, default=45, help="Random views per train / val site.")
    parser.add_argument("--test_step", type=float, default=2.5, help="Distance between test views [m].")
    parser.add_argument("--agl", type=float, nargs=2, default=(5.0, 12.0), metavar=("MIN", "MAX"))
    parser.add_argument("--max_tilt", type=float, default=15.0, help="Largest tilt off nadir [deg].")
    parser.add_argument("--pad", type=int, default=160, help="Render margin left and right [px]; 3/4 of it on top.")
    parser.add_argument("--warmup", type=int, default=2, help="Env steps per view before the frame is read.")
    parser.add_argument("--settle", type=int, default=12, help="Env steps after a site is spawned.")
    parser.add_argument("--rng_seed", type=int, default=0, help="Seed of the views, lights and ground looks.")
    parser.add_argument("--fixed_look", action="store_true", help="Mission light and ground materials everywhere.")
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


# ---------------------------------------------------------------------------------------------------------------
# the renderer: one simulator, many sites
# ---------------------------------------------------------------------------------------------------------------


class SiteRenderer:
    """Holds the drone at views, swaps sites, light and ground looks, and reads the downward camera."""

    def __init__(self, base, site_cfg, pad: tuple[int, int], warmup: int, settle: int):
        import torch

        from drone_lab.sar.scene import get_spawned_scenario

        self.base = base
        self.site_cfg = site_cfg
        self.pad_x, self.pad_y = pad
        self.warmup = warmup
        self.settle = settle
        self.device = base.device
        self.robot = base.scene["robot"]
        self.camera = base.scene.sensors["sar_camera"]
        self.zero = torch.zeros(1, base.action_manager.total_action_dim, device=self.device)
        self.rest = torch.zeros(1, 6, device=self.device)
        self.env_ids = torch.tensor([0], device=self.device)
        self.objects_root = f"{SITE}/objects"
        self.scenario = get_spawned_scenario(SITE)
        self.light: Lighting | None = None

    @property
    def seed(self) -> int:
        return int(self.scenario.cfg.seed)

    def hold(self, view: View) -> None:
        import torch

        import isaaclab.utils.math as math_utils

        angles = [torch.tensor([v], device=self.device) for v in (view.roll, view.pitch, view.yaw)]
        quat = math_utils.quat_from_euler_xyz(*angles)
        pos = torch.tensor([[view.x, view.y, view.z]], device=self.device, dtype=torch.float32)
        self.robot.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=self.env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.rest, env_ids=self.env_ids)

    def step_at(self, view: View, steps: int) -> None:
        for _ in range(steps):
            self.hold(view)
            self.base.step(self.zero)

    def load_site(self, seed: int, park: View) -> float:
        """Replace the spawned objects with the site of ``seed`` (ground materials stay); returns the seconds."""
        import isaaclab.sim as sim_utils

        from drone_lab.sar.scene import get_spawned_scenario, spawn_sar_site

        t0 = time.time()
        if seed != self.seed:
            sim_utils.delete_prim(self.objects_root)
            root = f"{SITES_ROOT}/seed_{seed:04d}"
            spawn_sar_site(root, self.site_cfg.replace(seed=seed, ground_materials=False))
            self.objects_root = root
            self.scenario = get_spawned_scenario(root)
        self.step_at(park, self.settle)
        return time.time() - t0

    def set_lighting(self, light: Lighting) -> None:
        """Sun direction, strength and colour; the sky dome turns with the sun so its own sun stays aligned."""
        if light == self.light:
            return
        from pxr import Gf, UsdLux

        import isaaclab.sim as sim_utils

        from drone_lab.sar.scene import sun_light_cfg

        stage = sim_utils.get_current_stage()
        sun, sky = stage.GetPrimAtPath(SUN_PRIM), stage.GetPrimAtPath(SKY_PRIM)
        rot = sun_light_cfg(light.sun_intensity, light.sun_elevation_deg, light.sun_azimuth_deg).init_state.rot
        sim_utils.standardize_xform_ops(sun, orientation=rot)
        UsdLux.DistantLight(sun).GetIntensityAttr().Set(light.sun_intensity)
        UsdLux.DistantLight(sun).GetColorAttr().Set(Gf.Vec3f(*light.sun_color))
        half = 0.5 * math.radians(light.sky_rotation_deg)
        sim_utils.standardize_xform_ops(sky, orientation=(0.0, 0.0, math.sin(half), math.cos(half)))
        UsdLux.DomeLight(sky).GetIntensityAttr().Set(light.sky_intensity)
        self.light = light

    def set_ground_look(self, look: GroundLook) -> None:
        """Bind each terrain face subset to the look's material and set the materials' tint and texture scale."""
        from pxr import Sdf, UsdGeom, UsdShade

        import isaaclab.sim as sim_utils

        stage = sim_utils.get_current_stage()
        looks = f"{SITE}/Looks"
        terrain = sim_utils.get_first_matching_child_prim(
            self.site_cfg.terrain_prim_path, lambda p: p.GetTypeName() == "Mesh"
        )
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(terrain)):
            key = subset.GetPrim().GetName().removeprefix("ground_")
            if key in look.materials:
                material = UsdShade.Material(stage.GetPrimAtPath(f"{looks}/ground_{look.materials[key]}"))
                binding = UsdShade.MaterialBindingAPI.Apply(subset.GetPrim())
                binding.Bind(material, UsdShade.Tokens.strongerThanDescendants)
        for name in TERRAIN_LOOKS:
            shader = UsdShade.Shader(stage.GetPrimAtPath(f"{looks}/ground_{name}/Shader"))
            if not shader:
                continue
            shader.CreateInput("albedo_brightness", Sdf.ValueTypeNames.Float).Set(look.albedo_brightness(name))
            scale = look.scaled_texture(name)
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set((scale, scale))

    def capture(self, view: View) -> dict:
        """Render ``view`` and return the mission-camera crop with its instance ids and the full boxes."""
        from drone_lab.sar.labels import semantic_label
        from drone_lab.sar.scene import read_camera

        self.step_at(view, self.warmup)
        fr = read_camera(self.camera, 0, ("rgb", SEG))
        seg_full = fr[SEG].astype(np.int64)
        seg = crop(seg_full, self.pad_x, self.pad_y)
        info = fr["info"]
        labelled = {int(k) for k, v in (info.get("idToSemantics") or {}).items() if semantic_label(v)}
        cut = unclipped_boxes(seg_full, self.pad_x, self.pad_y)
        return {
            "rgb": np.ascontiguousarray(crop(fr["rgb"], self.pad_x, self.pad_y)),
            "seg": np.ascontiguousarray(seg),
            "info": info,
            "unclipped": {i: b for i, b in cut.items() if i in labelled},
            "pos_w": fr["pos_w"],
            "quat_w_ros": fr["quat_w_ros"],
            "K": fr["K"],
        }


# ---------------------------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------------------------


def write_frame(folder: Path, key: str, frame: dict, record: dict) -> None:
    from PIL import Image

    if int(frame["seg"].max(initial=0)) > np.iinfo(np.uint16).max:
        raise ValueError(f"{key}: instance ids do not fit in a 16-bit PNG")
    Image.fromarray(frame["rgb"]).save(folder / f"rgb_{key}.png", compress_level=3)
    Image.fromarray(frame["seg"].astype(np.uint16)).save(folder / f"instance_{key}.png", compress_level=3)
    info = frame["info"]
    meta = {
        "idToLabels": info.get("idToLabels") or {},
        "idToSemantics": info.get("idToSemantics") or {},
        "bboxUnclipped": {str(i): list(b) for i, b in frame["unclipped"].items()},
        **record,
        "camera_pos_w": [round(float(v), 4) for v in frame["pos_w"]],
        "camera_quat_w_ros_xyzw": [round(float(v), 6) for v in frame["quat_w_ros"]],
    }
    (folder / f"instance_{key}.json").write_text(json.dumps(meta, separators=(",", ":")))


def label_check(frame: dict, scenario, pad: tuple[int, int]) -> tuple[int, list[str], list[str]]:
    """Project every labelled object's centre into the frame and compare its class with the instance under it.

    Replicator reports no prim paths for prims created after the camera started, so the check is geometric.
    Returns the matches, the objects whose centre shows another class, and the instances under the centres of
    several objects (merged instances). Centres on unlabelled pixels (a tree in front, the gap between pallet
    slats) are skipped. Objects that touch in the image (a pile of crates, an arm over a barrel) can put a
    centre on their neighbour, so a few hits per thousand centres are expected.
    """
    from drone_lab.sar.camera import world_to_pixel
    from drone_lab.sar.labels import semantic_label

    objs = scenario.labeled()
    uv, depth = world_to_pixel(np.array([o.center for o in objs]), frame["pos_w"], frame["quat_w_ros"], frame["K"])
    classes = {int(k): semantic_label(v) for k, v in (frame["info"].get("idToSemantics") or {}).items()}
    seg = frame["seg"]
    owners: dict[int, list[str]] = {}
    ok, wrong = 0, []
    for obj, (u, v), d in zip(objs, uv, depth, strict=True):
        col, row = math.floor(u) - pad[0], math.floor(v) - pad[1]
        if d <= 0 or not (0 <= row < seg.shape[0] and 0 <= col < seg.shape[1]):
            continue
        instance = int(seg[row, col])
        label = classes.get(instance)
        if label is None:
            continue
        owners.setdefault(instance, []).append(obj.name)
        if label == obj.category:
            ok += 1
        else:
            wrong.append(f"{obj.name}:{label}")
    merged = [f"{i}:{'+'.join(names)}" for i, names in owners.items() if len(names) > 1]
    return ok, wrong, merged


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------


def main() -> int:  # noqa: C901
    args, hydra_overrides = parse_args()
    out = Path(args.out).expanduser().resolve()
    plan = {
        "train": parse_seeds(args.train_seeds),
        "val": parse_seeds(args.val_seeds),
        "test": parse_seeds(args.test_seeds),
    }
    seen: dict[int, str] = {}
    for split, seeds in plan.items():
        for s in seeds:
            if s in seen:
                sys.exit(f"[gen_yolo_dataset] seed {s} is in both {seen[s]} and {split}")
            seen[s] = split
    todo = [(split, s) for split in SPLITS for s in plan[split] if not (out / split / f"seed_{s:04d}.json").exists()]
    if not todo:
        print(f"[gen_yolo_dataset] every site is already in {out}")
        return 0
    for split in SPLITS:
        (out / split).mkdir(parents=True, exist_ok=True)

    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.utils import parse_env_cfg

    import drone_lab.tasks  # noqa: F401  (registers the DroneLab gym ids)
    from drone_lab.sar.camera import SAR_CAMERA
    from drone_lab.sar.scenario import TerrainSite
    from drone_lab.sar.scene import sar_camera_cfg

    first_seed = todo[0][1]
    overrides = list(hydra_overrides) + ["physics=isaacsim_physx", f"env.scene.sar_site.spawn.seed={first_seed}"]
    env_cfg = parse_env_cfg(TASK, device=getattr(args, "device", None) or "cpu", num_envs=1, overrides=overrides)
    pad = (args.pad, round(args.pad * SAR_CAMERA.height / SAR_CAMERA.width))
    env_cfg.scene.sar_camera = sar_camera_cfg(padded_spec(SAR_CAMERA, *pad), data_types=("rgb", SEG))
    for name in list(vars(env_cfg.terminations)):  # the drone is placed, not flown: nothing may reset it
        setattr(env_cfg.terminations, name, None)
    env_cfg.seed = args.rng_seed
    args.headless = True

    t_start = time.time()
    totals: dict[str, int] = dict.fromkeys(SPLITS, 0)
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        base = env.unwrapped
        base.reset()
        print(f"[gen_yolo_dataset] simulator ready in {time.time() - t_start:.0f} s; {len(todo)} sites to render")
        site = TerrainSite(env_cfg.scene.sar_site.spawn.terrain_generator)
        renderer = SiteRenderer(base, env_cfg.scene.sar_site.spawn, pad, args.warmup, args.settle)
        for split, seed in todo:
            folder = out / split
            rng_views, rng_light, rng_look = (np.random.default_rng([args.rng_seed, seed, k]) for k in range(3))
            mission = split == "test"
            hx, hy, hz = renderer.scenario.home
            spawn_s = renderer.load_site(seed, View(hx, hy, hz + 8.0, 0.0, 0.0, 0.0, 8.0, "site"))
            scenario = renderer.scenario
            fixed = mission or args.fixed_look
            look = GroundLook() if fixed else sample_ground_look(rng_look)
            renderer.set_ground_look(look)
            if mission:
                views = mission_views(scenario, site, args.test_step, rng_views)
            else:
                views = sample_views(
                    scenario, site, args.frames, rng_views, agl=tuple(args.agl), max_tilt_deg=args.max_tilt
                )
            t_seed = time.time()
            matches, wrong, merged = 0, [], []
            for k, view in enumerate(views):
                light = Lighting() if fixed else sample_lighting(rng_light)
                renderer.set_lighting(light)
                frame = renderer.capture(view)
                key = f"s{seed:04d}_{k:03d}"
                record = {"seed": seed, "split": split, "view": view.to_dict(), "light": light.to_dict()}
                write_frame(folder, key, frame, record)
                ok, bad, both = label_check(frame, scenario, pad)
                matches += ok
                wrong += [f"{key}:{b}" for b in bad]
                merged += [f"{key}:{b}" for b in both]
            render_s = time.time() - t_seed
            meta = {
                "seed": seed,
                "split": split,
                "frames": len(views),
                "views": "mission path" if mission else f"{args.frames} random",
                "objects": {k: v for k, v in scenario.counts().items() if v},
                "ground_look": look.to_dict(),
                "centres_checked": matches + len(wrong),
                "class_matches": matches,
                "class_mismatches": wrong[:50],
                "merged_instances": merged[:50],
                "spawn_s": round(spawn_s, 1),
                "render_s": round(render_s, 1),
                "seconds_per_frame": round(render_s / max(1, len(views)), 3),
            }
            (folder / f"seed_{seed:04d}.json").write_text(json.dumps(meta, indent=1))
            totals[split] += len(views)
            print(
                f"[gen_yolo_dataset] {split} seed {seed}: {len(views)} frames, object centres {matches} with the "
                f"right class / {len(wrong)} wrong / {len(merged)} merged, look {look.name}, "
                f"spawn {spawn_s:.1f} s, {render_s / max(1, len(views)):.2f} s/frame",
                flush=True,
            )
        summary = {
            "task": TASK,
            "seeds": {k: v for k, v in plan.items()},
            "frames_this_run": totals,
            "frames_per_site": args.frames,
            "test_step_m": args.test_step,
            "agl_m": list(args.agl),
            "max_tilt_deg": args.max_tilt,
            "render_px": [SAR_CAMERA.width + 2 * pad[0], SAR_CAMERA.height + 2 * pad[1]],
            "crop_px": [SAR_CAMERA.width, SAR_CAMERA.height],
            "warmup_steps": args.warmup,
            "fixed_look": args.fixed_look,
            "rng_seed": args.rng_seed,
            "wall_s": round(time.time() - t_start, 1),
        }
        (out / "capture.json").write_text(json.dumps(summary, indent=1))
        print(f"[gen_yolo_dataset] {totals} frames in {time.time() - t_start:.0f} s -> {out}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
