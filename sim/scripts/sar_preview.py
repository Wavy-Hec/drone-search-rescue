"""Spawn the dressed SAR site in Isaac Sim and save what it looks like, plus automatic scale / label checks (GPU).

Writes to ``runs/<out>/`` (default ``runs/sar_preview_seed<seed>``):

* ``ground_truth.json``     -- every object: class, position / centre, size, pose, prim path (CPU, no Isaac Sim)
* ``object_map.png``        -- top-down object map over the hill-shaded terrain, with the survey path (CPU)
* ``overview.png`` / ``overview_classes.png`` -- high oblique RTX render of the site and its semantic classes
* ``photo_map.png`` / ``photo_map_pins.png``  -- nadir photo of the whole site from 75 m, and with GT pins on it
* ``drone/NN_<target>_{rgb,depth,classes,boxes}.png`` + ``.npz`` + ``_labels.json`` -- drone-camera frames above
  chosen objects (RGB, distance-to-image-plane, instance-id segmentation, boxes from the segmentation)
* ``checks.json``           -- stage vs scenario boxes (scale / placement), labels on the prims and in the frames,
  stage terrain vs CPU terrain, pixel size of people in the drone frames
* ``timing.json``           -- Kit start, site spawn, per-frame render time, env.step rate, VRAM of this process
* ``summary.txt``

Examples (env_isaaclab, from the project root)::

    python scripts/sar_preview.py                      # seed 0, everything
    python scripts/sar_preview.py --seed 3 --out sar_seed3
    python scripts/sar_preview.py --no_sim             # ground truth + object map only (CPU, seconds)
    python scripts/sar_preview.py --bench_only --bench_steps 200   # drone camera only: render time / VRAM

Trailing ``key=value`` tokens are Hydra overrides of the task config
(e.g. ``env.scene.sar_site.spawn.num_trees=[2,3]``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

TASK = "DroneLab-SAR-ARL-Robot-1-v0"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--seed", type=int, default=0, help="Scenario seed (same seed, same site).")
parser.add_argument("--out", type=str, default=None, help="Run name under runs/ (or a path).")
parser.add_argument("--no_sim", action="store_true", help="Only the CPU outputs (ground truth + object map).")
parser.add_argument(
    "--bench_only",
    action="store_true",
    help="No overview cameras / drone frames: only the drone camera's render time and VRAM.",
)
parser.add_argument(
    "--no_camera",
    action="store_true",
    help="With --bench_only: remove the drone camera too (physics-only reference for the render cost).",
)
parser.add_argument(
    "--aa",
    choices=("task", "Off", "FXAA", "DLSS", "TAA", "DLAA"),
    default="task",
    help="RTX anti-aliasing mode for all cameras (process-wide; 'task' keeps the task's setting).",
)
parser.add_argument("--warmup", type=int, default=8, help="Env steps per drone pose before the frame is saved.")
parser.add_argument("--bench_steps", type=int, default=60, help="Env steps of the render-time benchmark.")
parser.add_argument("--overview_size", type=int, nargs=2, default=(1600, 900), metavar=("W", "H"))
parser.add_argument("--photo_size", type=int, default=1600, help="Square size of the nadir photo map [px].")
args_cli, hydra_overrides = parser.parse_known_args()
if not args_cli.no_sim:
    from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

    add_launcher_args(parser)
    args_cli, hydra_overrides = parser.parse_known_args()

import numpy as np  # noqa: E402

from drone_lab.sar.labels import CLASS_COLORS, class_map, colorize_classes, instance_boxes  # noqa: E402
from drone_lab.sar.scenario import Scenario, TerrainSite  # noqa: E402
from drone_lab.sar.topdown import plot_object_map  # noqa: E402

SITE = "/World/sar"
SEG = "instance_segmentation"
"""Segmentation data type of the cameras (one id per labelled object, with ``idToSemantics``)."""


def run_dir_for(out: str | None, seed: int) -> Path:
    if out and ("/" in out or out.startswith("~")):
        return Path(out).expanduser().resolve()
    return PROJECT_ROOT / "runs" / (out or f"sar_preview_seed{seed}")


def lawnmower_xy(scenario: Scenario) -> np.ndarray:
    from drone_lab.paths import Area, lawnmower

    x0, x1, y0, y1 = scenario.area
    return lawnmower(Area(x0, x1, y0, y1), scenario.cfg.survey_spacing, scenario.flight_z)[:, :2]


def save_png(path: Path, img: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(img)).save(path)


def depth_png(depth: np.ndarray) -> np.ndarray:
    import matplotlib

    d = np.asarray(depth, dtype=np.float32)
    ok = np.isfinite(d) & (d > 0)
    lo, hi = (np.percentile(d[ok], 1), np.percentile(d[ok], 99)) if ok.any() else (0.0, 1.0)
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (matplotlib.colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)
    rgb[~ok] = 0
    return rgb


def draw_boxes(rgb: np.ndarray, boxes, extra_points=None) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
    dr = ImageDraw.Draw(img)
    for b in boxes:
        color = tuple(CLASS_COLORS.get(b.label, (255, 255, 255)))
        x0, y0, x1, y1 = b.bbox_xyxy
        dr.rectangle([x0, y0, x1, y1], outline=color, width=2)
        tag = f"{b.label} {b.name.split('_')[-1]} {x1 - x0 + 1}x{y1 - y0 + 1}"
        dr.text((x0 + 2, max(0, y0 - 11)), tag, fill=color)
    for u, v, color in extra_points or []:
        dr.line([u - 5, v, u + 5, v], fill=color, width=2)
        dr.line([u, v - 5, u, v + 5], fill=color, width=2)
    return np.asarray(img)


def project(points_w: np.ndarray, pos_w: np.ndarray, quat_w_ros_xyzw: np.ndarray, K: np.ndarray):
    """World points (N, 3) -> pixels (N, 2) and depth (N,) (see :func:`drone_lab.sar.camera.world_to_pixel`)."""
    from drone_lab.sar.camera import world_to_pixel

    return world_to_pixel(points_w, pos_w, quat_w_ros_xyzw, K)


class VramMonitor:
    """Polls ``nvidia-smi`` for this process's GPU memory (MiB): last and peak."""

    def __init__(self, period: float = 1.0):
        self.pid = os.getpid()
        self.period = period
        self.peak = 0
        self.last = 0
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def query(self) -> int:
        try:
            cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return -1
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid):
                return int(parts[1])
        return 0

    def _run(self):
        while not self._stop.is_set():
            v = self.query()
            if v >= 0:
                self.last = v
                self.peak = max(self.peak, v)
                self.samples.append((round(time.time() - self._t0, 1), v))
            self._stop.wait(self.period)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


def cpu_outputs(run_dir: Path, site_cfg, site: TerrainSite | None = None):
    """Ground truth + object map without Isaac Sim, from the task's (Hydra-resolved) site config."""
    from drone_lab.sar.scene import scenario_for

    site = site if site is not None else TerrainSite(site_cfg.terrain_generator)
    scenario = scenario_for(site_cfg, site)
    return scenario, site, *write_scenario_outputs(run_dir, scenario, site)


def write_scenario_outputs(run_dir: Path, scenario: Scenario, site: TerrainSite):
    gt_path = scenario.save_ground_truth(run_dir / "ground_truth.json", SITE)
    map_path = plot_object_map(scenario, site, run_dir / "object_map.png", path_xy=lawnmower_xy(scenario))
    return gt_path, map_path


def camera_frame(cam, data_types=("rgb", "distance_to_image_plane", SEG)) -> dict:
    """Env-0 frame of a camera in the layout used below (see :func:`drone_lab.sar.scene.read_camera`)."""
    from drone_lab.sar.scene import read_camera

    fr = read_camera(cam, 0, tuple(data_types))
    out = {k: fr[k] for k in data_types if k in fr}
    return {"out": out, "seg_info": fr["info"], "pos_w": fr["pos_w"], "quat_w_ros": fr["quat_w_ros"], "K": fr["K"]}


def main():  # noqa: C901
    from isaaclab_tasks.utils import parse_env_cfg

    import drone_lab.tasks  # noqa: F401  (registers the DroneLab gym ids)

    run_dir = run_dir_for(args_cli.out, args_cli.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = list(hydra_overrides) + ["physics=isaacsim_physx", f"env.scene.sar_site.spawn.seed={args_cli.seed}"]
    env_cfg = parse_env_cfg(TASK, device=getattr(args_cli, "device", None) or "cpu", num_envs=1, overrides=overrides)
    t_cpu = time.time()
    scenario, site, gt_path, map_path = cpu_outputs(run_dir, env_cfg.scene.sar_site.spawn)
    print(
        f"[sar_preview] seed {args_cli.seed}: {len(scenario.objects)} objects {scenario.counts()} "
        f"(CPU {time.time() - t_cpu:.1f} s)\n[sar_preview] wrote {gt_path}\n[sar_preview] wrote {map_path}"
    )
    if args_cli.no_sim:
        return

    import gymnasium as gym

    from drone_lab.sar.scene import fixed_camera_cfg, get_spawned_scenario

    x0, x1, y0, y1 = site.bounds
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    span = max(x1 - x0, y1 - y0)
    if not args_cli.bench_only:
        w, h = args_cli.overview_size
        env_cfg.scene.overview_camera = fixed_camera_cfg(
            "/World/sar_preview/overview_cam",
            eye=(cx - 26.0, cy - 58.0, 42.0),
            target=(cx, cy - 6.0, -6.0),
            width=w,
            height=h,
            hfov_deg=60.0,
        )
        hp = 75.0
        env_cfg.scene.topdown_camera = fixed_camera_cfg(
            "/World/sar_preview/topdown_cam",
            eye=(cx, cy, hp),
            target=(cx, cy, 0.0),
            width=args_cli.photo_size,
            height=args_cli.photo_size,
            hfov_deg=2.0 * math.degrees(math.atan(0.5 * span / hp)),
        )
    if args_cli.no_camera:
        if not args_cli.bench_only:
            sys.exit("[sar_preview] --no_camera only makes sense with --bench_only")
        env_cfg.scene.sar_camera = None
    # RTX global settings are process-wide and Isaac Lab requires them to be identical on every camera
    sar_cam = getattr(env_cfg.scene, "sar_camera", None)
    for name in ("overview_camera", "topdown_camera"):
        cam_cfg = getattr(env_cfg.scene, name, None)
        if cam_cfg is not None and sar_cam is not None:
            cam_cfg.renderer_cfg.global_settings = sar_cam.renderer_cfg.global_settings.copy()
    if args_cli.aa != "task":
        for name in ("sar_camera", "overview_camera", "topdown_camera"):
            cam_cfg = getattr(env_cfg.scene, name, None)
            if cam_cfg is not None:
                cam_cfg.renderer_cfg.global_settings.antialiasing_mode = args_cli.aa
    env_cfg.seed = args_cli.seed
    if "kit" not in (getattr(args_cli, "visualizer", None) or []):
        args_cli.headless = True

    timing: dict = {"seed": args_cli.seed, "bench_only": args_cli.bench_only, "antialiasing": args_cli.aa}
    checks: dict = {}
    vram = VramMonitor().start()
    t_launch = time.time()
    with launch_simulation(env_cfg, args_cli):
        timing["kit_start_s"] = round(time.time() - t_launch, 1)
        t_env = time.time()
        env = gym.make(TASK, cfg=env_cfg)
        base = env.unwrapped
        timing["env_create_s"] = round(time.time() - t_env, 1)
        t_reset = time.time()
        obs, _ = base.reset()
        timing["reset_s"] = round(time.time() - t_reset, 1)
        timing["vram_after_reset_mib"] = vram.query()
        import torch

        import isaaclab.utils.math as math_utils

        device = base.device
        robot = base.scene["robot"]
        cam = base.scene.sensors.get("sar_camera")
        spawned = get_spawned_scenario(SITE)
        checks["scenario_matches_cpu"] = bool(
            spawned is not None and json.dumps(spawned.ground_truth(SITE)) == json.dumps(scenario.ground_truth(SITE))
        )
        if spawned is not None and not checks["scenario_matches_cpu"]:
            print("[sar_preview] WARNING: the spawned site differs from the CPU one; saving the spawned one")
            scenario = spawned
            write_scenario_outputs(run_dir, scenario, site)
        flight_z = float(base.command_manager.get_term("survey_target").flight_altitude)
        checks["flight_z"] = {"task": round(flight_z, 3), "scenario": round(scenario.flight_z, 3)}
        zero = torch.zeros(1, base.action_manager.total_action_dim, device=device)
        env_ids = torch.tensor([0], device=device)

        def hold(x, y, z, yaw=0.0):
            quat = math_utils.quat_from_euler_xyz(
                torch.zeros(1, device=device),
                torch.zeros(1, device=device),
                torch.full((1,), float(yaw), device=device),
            )
            pose = torch.cat([torch.tensor([[x, y, z]], device=device, dtype=torch.float32), quat], dim=-1)
            robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids)

        def step_at(x, y, z, n):
            for _ in range(n):
                hold(x, y, z)
                base.step(zero)

        # ---------------------------------------------------------------- stage checks (scale / placement / labels)
        checks.update(stage_checks(scenario, site))
        # ---------------------------------------------------------------- benchmark: env.step with cameras
        hx, hy, _ = scenario.home
        step_at(hx + 3.0, hy, flight_z, 5)
        torch.cuda.synchronize()
        t_b = time.time()
        times = []
        for _ in range(args_cli.bench_steps):
            t1 = time.time()
            hold(hx + 3.0, hy, flight_z)
            base.step(zero)
            if cam is not None:
                _ = cam.data.output["rgb"]
            torch.cuda.synchronize()
            times.append(time.time() - t1)
        times = np.asarray(times[3:]) * 1000.0
        timing["bench"] = {
            "steps": int(args_cli.bench_steps),
            "render_products": (0 if cam is None else 1) if args_cli.bench_only else 3,
            "env_step_ms_mean": round(float(times.mean()), 1),
            "env_step_ms_p50": round(float(np.percentile(times, 50)), 1),
            "env_step_ms_p95": round(float(np.percentile(times, 95)), 1),
            "env_step_ms_max": round(float(times.max()), 1),
            "loop_hz": round(1000.0 / float(times.mean()), 1),
            "wall_s": round(time.time() - t_b, 1),
            "note": "one env.step = 10 physics steps + one RTX render of every camera + data fetch",
        }
        timing["vram_after_bench_mib"] = vram.query()

        frames = []
        if not args_cli.bench_only:
            # ------------------------------------------------------------ overview + nadir photo map
            ov = base.scene.sensors["overview_camera"]
            td = base.scene.sensors["topdown_camera"]
            step_at(hx, hy, scenario.home[2] + 0.8, 4)  # drone hovering over its pad for the overview
            fo, ft = camera_frame(ov, ("rgb", SEG)), camera_frame(td, ("rgb", SEG))
            save_png(run_dir / "overview.png", fo["out"]["rgb"])
            save_png(run_dir / "overview_classes.png", colorize_classes(class_map(fo["out"][SEG], fo["seg_info"])))
            save_png(run_dir / "photo_map.png", ft["out"]["rgb"])
            pins = []
            for o in scenario.objects:
                if not o.labeled:
                    continue
                uv, zc = project(np.array([o.center]), ft["pos_w"], ft["quat_w_ros"], ft["K"])
                pins.append((float(uv[0, 0]), float(uv[0, 1]), tuple(CLASS_COLORS.get(o.category, (255, 255, 255)))))
            boxes_td = instance_boxes(ft["out"][SEG], ft["seg_info"], SITE)
            save_png(run_dir / "photo_map_pins.png", draw_boxes(ft["out"]["rgb"], boxes_td, pins))
            checks["photo_map_labels"] = label_check(boxes_td, scenario)
            checks["overview_labels"] = label_check(instance_boxes(fo["out"][SEG], fo["seg_info"], SITE), scenario)
            # ------------------------------------------------------------ drone frames above chosen objects
            for old_file in (run_dir / "drone").glob("*"):  # frames of an earlier run of the same name
                if old_file.is_file():
                    old_file.unlink()
            targets = choose_targets(scenario)
            t_frames = []
            for k, (tag, obj, (dx, dy), yaw) in enumerate(targets):
                x, y = obj.center[0] + dx, obj.center[1] + dy
                t1 = time.time()
                for _ in range(args_cli.warmup):
                    hold(x, y, flight_z, yaw)
                    base.step(zero)
                fr = camera_frame(cam)
                t_frames.append(time.time() - t1)
                frames.append(save_drone_frame(run_dir / "drone", k, tag, obj, fr, scenario))
            timing["drone_frames"] = {
                "count": len(t_frames),
                "warmup_steps_per_frame": args_cli.warmup,
                "seconds_per_saved_frame": round(float(np.mean(t_frames)), 2) if t_frames else None,
            }
            checks["drone_frames"] = frames
        timing["vram_peak_mib"] = vram.peak
        timing["vram_last_mib"] = vram.last
        vram.stop()
        timing["vram_samples"] = vram.samples[-200:]
        (run_dir / "checks.json").write_text(json.dumps(checks, indent=1, default=_json_default))
        (run_dir / "timing.json").write_text(json.dumps(timing, indent=1, default=_json_default))
        summary = make_summary(scenario, checks, timing, run_dir)
        (run_dir / "summary.txt").write_text(summary)
        print(summary, flush=True)
        try:
            env.close()
        except Exception as exc:
            print(f"[sar_preview] WARNING: env.close() failed ({exc})")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def stage_checks(scenario: Scenario, site: TerrainSite) -> dict:
    """Compare the spawned prims with the scenario: world boxes, resting height, labels; stage vs CPU terrain."""
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    from drone_lab.tasks.survey.mdp.terrain_utils import terrain_mesh_from_stage

    stage = sim_utils.get_current_stage()
    out: dict = {}
    mesh = terrain_mesh_from_stage("/World/ground")
    if mesh is not None:
        v, f = mesh
        same = v.shape == site.vertices.shape and f.shape == site.faces.shape
        out["terrain"] = {
            "stage_vertices": int(len(v)),
            "cpu_vertices": int(len(site.vertices)),
            "same_shape": same,
            "max_abs_diff_m": float(np.abs(v - site.vertices).max()) if same else None,
        }
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def points_box(prim, frame=None):
        """Box of the mesh points (rest pose), in the world or in ``frame``'s yawed box frame (centre, yaw).
        Skinned characters' extent attributes are padded, so the BBoxCache box of a person is ~0.15 m too big;
        the points are what the asset measurements used."""
        lo_, hi_ = np.full(3, np.inf), np.full(3, -np.inf)
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
                if pts is None or len(pts) == 0:
                    continue
                m = np.array(xcache.GetLocalToWorldTransform(p)).T
                w = np.asarray(pts, dtype=np.float64) @ m[:3, :3].T + m[:3, 3]
                if frame is not None:
                    c, cy_, sy_ = np.array(frame.center), math.cos(frame.yaw), math.sin(frame.yaw)
                    d = w - c
                    w = np.column_stack([d[:, 0] * cy_ + d[:, 1] * sy_, -d[:, 0] * sy_ + d[:, 1] * cy_, d[:, 2]])
                lo_, hi_ = np.minimum(lo_, w.min(0)), np.maximum(hi_, w.max(0))
        return lo_, hi_

    rows, worst = [], 0.0
    for o in scenario.objects:
        prim = stage.GetPrimAtPath(o.prim_path(SITE))
        if not prim.IsValid():
            rows.append({"name": o.name, "error": "missing prim"})
            continue
        if o.category == "person":
            # compare in the person's own (yawed) frame: the world AABB of a rotated cross-shaped body is smaller
            # than the AABB of its rotated box, so only the yawed frame gives a like-for-like comparison
            lo, hi = points_box(prim, frame=o)
            exp_lo, exp_hi = -np.array(o.half_extents), np.array(o.half_extents)
        else:
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
            exp_lo, exp_hi = np.array(o.aabb_min), np.array(o.aabb_max)
        dlo, dhi = lo - exp_lo, hi - exp_hi
        err = float(max(np.abs(dlo).max(), np.abs(dhi).max()))
        labels = sim_utils.get_labels(prim)
        n_desc_labels = sum(1 for p in Usd.PrimRange(prim) if p != prim and sim_utils.get_labels(p))
        bottom = float(lo[2] + o.center[2]) if o.category == "person" else float(lo[2])
        row = {
            "name": o.name,
            "asset": o.asset,
            "pose": o.pose,
            "frame": "yawed object frame (mesh points)" if o.category == "person" else "world AABB (USD bounds)",
            "expected_size": [round(float(v), 3) for v in exp_hi - exp_lo],
            "stage_size": [round(float(v), 3) for v in hi - lo],
            "max_box_error_m": round(err, 3),
            "bottom_minus_ground_m": round(bottom - o.ground_z, 3),
            "labels": labels,
            "descendant_labels": n_desc_labels,
        }
        if o.category != "tree":  # tree boxes include foliage instancers; checked by eye in the renders
            worst = max(worst, err)
        rows.append(row)
    out["objects"] = rows
    out["max_box_error_m_excluding_trees"] = round(worst, 3)
    bad_labels = [
        r["name"]
        for r in rows
        if "labels" in r
        and (
            (r["name"].startswith("tree") and r["labels"])
            or (not r["name"].startswith("tree") and r["labels"].get("class") != [r["name"].rsplit("_", 1)[0]])
        )
    ]
    out["label_mismatches"] = bad_labels
    out["objects_with_descendant_labels"] = [r["name"] for r in rows if r.get("descendant_labels")]
    return out


def label_check(boxes, scenario: Scenario) -> dict:
    """Do the classes in a segmentation frame agree with the ground truth of the same objects?"""
    by_name = {o.name: o for o in scenario.objects}
    ok, wrong, unknown = 0, [], []
    for b in boxes:
        o = by_name.get(b.name)
        if o is None:
            unknown.append({"name": b.name, "label": b.label})
        elif o.category == b.label:
            ok += 1
        else:
            wrong.append({"name": b.name, "label": b.label, "expected": o.category})
    return {"objects_seen": len(boxes), "correct": ok, "wrong": wrong, "not_in_site": unknown}


def choose_targets(scenario: Scenario):
    """Interesting views: ``(tag, object, (dx, dy) offset of the drone from the object's centre, yaw)``."""
    objs = scenario.objects
    picks = []

    def first(pred):
        return next((o for o in objs if pred(o) and all(o is not p[1] for p in picks)), None)

    for tag, pred in [
        ("landing_pad", lambda o: o.category == "landing_pad"),
        ("person_lying", lambda o: o.category == "person" and o.pose.startswith("lying") and o.near_block),
        ("person_lying_2", lambda o: o.category == "person" and o.pose.startswith("lying")),
        ("person_standing", lambda o: o.category == "person" and o.pose == "standing"),
        ("vehicle", lambda o: o.category == "vehicle"),
        ("crates", lambda o: o.category == "crate" and o.group is not None),
        ("barrels", lambda o: o.category == "barrel"),
        ("tree_inside", lambda o: o.category == "tree" and o.in_area),
    ]:
        o = first(pred)
        if o is not None:
            picks.append((tag, o, (0.0, 0.0), 0.0))
    pad = next((o for o in objs if o.category == "landing_pad"), None)
    if pad is not None:
        x0, _, y0, _ = scenario.area
        n_lines = max(1, math.ceil((scenario.area[3] - y0) / scenario.cfg.survey_spacing - 1e-9))
        start = (x0, y0 + 0.5 * (scenario.area[3] - y0) / n_lines)  # first lawnmower waypoint
        picks.insert(1, ("survey_start", pad, (start[0] - pad.center[0], start[1] - pad.center[1]), 0.0))
        picks.insert(2, ("landing_pad_yaw90", pad, (0.0, 0.0), 0.5 * math.pi))
    return picks


def save_drone_frame(out_dir: Path, k: int, tag: str, obj, fr: dict, scenario: Scenario) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{k:02d}_{tag}"
    rgb = fr["out"]["rgb"][..., :3].astype(np.uint8)
    depth = fr["out"]["distance_to_image_plane"].astype(np.float32)
    seg = fr["out"][SEG].astype(np.int32)
    boxes = instance_boxes(seg, fr["seg_info"], SITE)
    pts = []
    for o in scenario.objects:
        if not o.labeled:
            continue
        uv, zc = project(np.array([o.center]), fr["pos_w"], fr["quat_w_ros"], fr["K"])
        if zc[0] > 0 and 0 <= uv[0, 0] < rgb.shape[1] and 0 <= uv[0, 1] < rgb.shape[0]:
            pts.append((float(uv[0, 0]), float(uv[0, 1]), (255, 255, 255)))
    save_png(out_dir / f"{stem}_rgb.png", rgb)
    save_png(out_dir / f"{stem}_depth.png", depth_png(depth))
    save_png(out_dir / f"{stem}_classes.png", colorize_classes(class_map(seg, fr["seg_info"])))
    save_png(out_dir / f"{stem}_boxes.png", draw_boxes(rgb, boxes, pts))
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        rgb=rgb,
        depth=depth,
        instance_id=seg,
        K=fr["K"],
        pos_w=fr["pos_w"],
        quat_w_ros=fr["quat_w_ros"],
    )
    target_box = next((b for b in boxes if b.name == obj.name), None)
    # rendered geometry check: how far the target rises above the ground around it, from the depth image
    # (nadir camera, so depth differences near the image centre are heights)
    visible_height = None
    if target_box is not None:
        mask = np.isin(seg, list(target_box.ids))
        x0, y0, x1, y1 = target_box.bbox_xyxy
        pad = 25
        ring = np.zeros_like(mask)
        ring[max(0, y0 - pad) : y1 + pad + 1, max(0, x0 - pad) : x1 + pad + 1] = True
        ring &= ~mask & (seg <= 1)  # background / unlabelled pixels next to the object (ground)
        if mask.any() and ring.sum() > 50:
            visible_height = round(float(np.median(depth[ring]) - np.percentile(depth[mask], 2)), 3)
    rec = {
        "frame": stem,
        "target": obj.name,
        "target_class": obj.category,
        "target_pose": obj.pose,
        "camera_pos_w": [round(float(v), 3) for v in fr["pos_w"]],
        "camera_height_above_target_ground_m": round(float(fr["pos_w"][2] - obj.ground_z), 3),
        "depth_at_center_m": round(float(depth[depth.shape[0] // 2, depth.shape[1] // 2]), 3),
        "target_height_from_depth_m": visible_height,
        "target_expected_height_m": round(float(obj.aabb_max[2] - obj.aabb_min[2]), 3),
        "target_box": target_box.to_dict() if target_box else None,
        "target_box_min_side_px": (
            min(
                target_box.bbox_xyxy[2] - target_box.bbox_xyxy[0] + 1,
                target_box.bbox_xyxy[3] - target_box.bbox_xyxy[1] + 1,
            )
            if target_box
            else None
        ),
        "boxes": [b.to_dict() for b in boxes],
        "label_check": label_check(boxes, scenario),
        "id_to_labels": {str(k_): v for k_, v in (fr["seg_info"].get("idToLabels") or {}).items()},
        "id_to_semantics": {str(k_): v for k_, v in (fr["seg_info"].get("idToSemantics") or {}).items()},
    }
    (out_dir / f"{stem}_labels.json").write_text(json.dumps(rec, indent=1, default=_json_default))
    return {k_: v for k_, v in rec.items() if k_ not in ("id_to_labels", "id_to_semantics", "boxes")}


def make_summary(scenario: Scenario, checks: dict, timing: dict, run_dir: Path) -> str:
    lines = [f"SAR preview, seed {scenario.cfg.seed}: {run_dir}"]
    lines.append("objects: " + ", ".join(f"{k} {v}" for k, v in scenario.counts().items() if v))
    lines.append(
        f"scenario spawned == CPU scenario: {checks.get('scenario_matches_cpu')}; flight z {checks.get('flight_z')}"
    )
    t = checks.get("terrain")
    if t:
        lines.append(
            f"terrain: stage {t['stage_vertices']} vertices vs CPU {t['cpu_vertices']}, max diff {t['max_abs_diff_m']}"
        )
    lines.append(
        f"stage boxes vs scenario: max error {checks.get('max_box_error_m_excluding_trees')} m "
        "(trees excluded); "
        f"label mismatches {checks.get('label_mismatches')}; prims with leftover asset labels "
        f"{checks.get('objects_with_descendant_labels')}"
    )
    for key in ("photo_map_labels", "overview_labels"):
        if key in checks:
            c = checks[key]
            lines.append(
                f"{key}: {c['objects_seen']} objects seen, {c['correct']} correct, wrong {c['wrong']}, "
                f"not in site {c['not_in_site'][:4]}"
            )
    for fr in checks.get("drone_frames", []):
        lc = fr["label_check"]
        lines.append(
            f"frame {fr['frame']}: target {fr['target']} ({fr['target_pose']}) box min side "
            f"{fr['target_box_min_side_px']} px, height from depth {fr['target_height_from_depth_m']} m "
            f"(expected {fr['target_expected_height_m']}), "
            f"camera {fr['camera_height_above_target_ground_m']} m "
            f"above its ground; labels {lc['correct']}/{lc['objects_seen']} correct, wrong {lc['wrong']}"
        )
    b = timing.get("bench", {})
    lines.append(
        f"timing: Kit start {timing.get('kit_start_s')} s, env create {timing.get('env_create_s')} s, reset "
        f"{timing.get('reset_s')} s; env.step with {b.get('render_products')} camera(s): mean "
        f"{b.get('env_step_ms_mean')} ms (p95 {b.get('env_step_ms_p95')}, {b.get('loop_hz')} Hz)"
    )
    lines.append(
        f"VRAM (this process): after reset {timing.get('vram_after_reset_mib')} MiB, peak "
        f"{timing.get('vram_peak_mib')} MiB"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
