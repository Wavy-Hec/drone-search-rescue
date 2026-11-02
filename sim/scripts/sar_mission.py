"""Fly the search-and-rescue mission in Isaac Sim and stream it live to the drone_gcs ground station.

The trained tracking policy takes off from the landing pad, flies a lawnmower survey over the dressed site of
``DroneLab-SAR-ARL-Robot-1-v0``, comes back and lands on the pad. Every policy step (10 Hz) it streams through
``drone_gcs.link.Sender`` (non-blocking: a slow or absent ground station only costs dropped frames):

* MISSION (area, map grid, planned path, landing pad) and, for the scorecard only, GT_OBJECTS (the site's ground
  truth) and GT_TERRAIN (the true terrain height),
* TELEMETRY (pose, velocity, waypoint, carrot, mission phase), SCAN (ray hits of the mapping scanner),
* CAMERA (RGB, distance to the image plane, K and the ROS-optical pose of the downward camera),
* END (landed / crashed / timeout, with the closest approach to obstacles).

The flight plan math (spacing, height, overlaps) is in :mod:`drone_lab.sar.mission`. Start the ground station
first (``python scripts/gcs.py`` in the drone_gcs project, conda env ``vision``); either side may start first.

The Isaac Sim window (``--viz kit``) and the video (``--video``) show the same camera: by default a smooth follow
camera that keeps the drone and its scan in view, with the mapping rays' hits glowing on the ground and the ground
running on to the horizon (:mod:`drone_lab.sar.chase_view`). Only that camera sees these extras; the drone's own
camera renders the site alone.

Outputs in ``runs/<out>/`` (default ``runs/sar_mission_seed<seed>``): ``mission.json`` (plan, outcome, loop rates,
link and GPU statistics), ``trajectory.csv``, ``ground_truth.json`` and, with ``--video``, ``mission.mp4`` plus
``mission_frames.csv`` (the sim time of every frame, for drone_gcs ``scripts/compose_video.py``).

Examples (env_isaaclab, from the project root)::

    python scripts/sar_mission.py --checkpoint <run>/exported/policy.pt                     # headless, full mission
    python scripts/sar_mission.py --checkpoint <policy.pt> --viz kit --video                # live window + recording
    python scripts/sar_mission.py --checkpoint <policy.pt> --max_time 90 --out sar_short    # short test flight
    python scripts/sar_mission.py --checkpoint <policy.pt> --video --view overview          # fixed view of the site

Trailing ``key=value`` tokens are Hydra overrides of the task config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import add_launcher_args, launch_simulation

TASK = "DroneLab-SAR-ARL-Robot-1-v0"
SITE = "/World/sar"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=str, required=True, help="TorchScript tracking policy (exported/policy.pt).")
parser.add_argument("--seed", type=int, default=0, help="Site seed (same seed, same site; 0 is the demo site).")
parser.add_argument("--spacing", type=float, default=5.0, help="Lawnmower line spacing [m].")
parser.add_argument("--altitude", type=float, default=4.0, help="Clearance above the highest terrain point [m].")
parser.add_argument("--speed", type=float, default=1.5, help="Carrot speed for transit and survey [m/s].")
parser.add_argument("--settle", type=float, default=1.5, help="Hover above the pad before the take-off [s].")
parser.add_argument("--max_time", type=float, default=None, help="End the mission after this much sim time [s].")
parser.add_argument(
    "--gcs", type=str, default=None, help="Ground-station address (default: drone_gcs's, e.g. $DRONE_GCS_ADDR)."
)
parser.add_argument("--gcs_root", type=str, default=None, help="drone_gcs repository (default: next to this one).")
parser.add_argument("--no_gcs", action="store_true", help="Fly without the live link.")
parser.add_argument(
    "--wait_gcs", type=float, default=10.0, help="Wait this long on the pad for the ground station [s]."
)
parser.add_argument("--camera_every", type=int, default=1, help="Send every N-th camera frame (1 = 10 Hz).")
parser.add_argument("--jpeg", action="store_true", help="Send camera images as JPEG instead of raw RGB.")
parser.add_argument(
    "--gt_boxes",
    action="store_true",
    help="Attach the boxes of the instance segmentation to every frame (for the ground station's --detector gt).",
)
parser.add_argument(
    "--view",
    choices=("chase", "overview"),
    default="chase",
    help="Camera of the window / video: a smooth follow camera, or a fixed view of the whole site.",
)
parser.add_argument("--no_scan_rays", action="store_true", help="Do not draw the mapping rays in the window / video.")
parser.add_argument("--video", action="store_true", help="Record runs/<out>/mission.mp4 of the window's camera.")
parser.add_argument(
    "--video_fps", type=int, default=10, help="Video frame rate; one frame per 0.1 s step, so 10 is real time."
)
parser.add_argument("--video_size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
parser.add_argument(
    "--window_pos",
    type=int,
    nargs=2,
    default=None,
    metavar=("X", "Y"),
    help="Position of the Isaac Sim window (--viz kit); it shows the viewport only.",
)
parser.add_argument("--kit_ui", action="store_true", help="Keep Isaac Sim's menus and panels around the viewport.")
parser.add_argument(
    "--markers",
    action="store_true",
    help="Show the carrot and path markers. The drone camera sees the ones below it (take-off, landing), and the "
    "detector may pin them.",
)
parser.add_argument("--out", type=str, default=None, help="Run name under runs/ (or a path).")
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

import csv
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch

import drone_lab.tasks  # noqa: F401  (registers the DroneLab gym ids)
from drone_lab.sar.camera import SAR_CAMERA
from drone_lab.sar.chase_view import (
    FollowCamera,
    ScanRays,
    ground_below,
    isolate_chase_view,
    rim_points,
    set_focal_length,
    spawn_ground_skirt,
)
from drone_lab.sar.mission import (
    ClearanceMonitor,
    MissionPlan,
    gt_objects,
    import_drone_gcs,
    plan_mission,
    survey_geometry,
)
from drone_lab.sar.scenario import Scenario, TerrainSite
from drone_lab.video import capture_frame, find_kit_visualizer, open_mp4_writer, playback_speedup

ZERO_THRUST = -1.0
"""Raw action that the thrust action term maps to 0 N (scale 3, offset 3)."""
VIEWER_CAMERA = "/OmniverseKit_Persp"
"""Camera prim of the Kit viewport; the video recorder renders the same prim."""


def run_dir_for(out: str | None, seed: int) -> Path:
    if out and ("/" in out or out.startswith("~")):
        return Path(out).expanduser().resolve()
    return PROJECT_ROOT / "runs" / (out or f"sar_mission_seed{seed}")


def gpu_memory_mib(pid: int | None = None) -> int | None:
    """GPU memory of process ``pid`` (default: this one) from ``nvidia-smi``."""
    pid = os.getpid() if pid is None else pid
    try:
        cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return None


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "max": round(float(a.max()), 2),
    }


def configure(env_cfg, plan) -> None:
    """Fly the mission plan instead of the task's lawnmower; the script ends the mission itself."""
    cmd = env_cfg.commands.survey_target
    cmd.pattern = "waypoints"
    cmd.waypoints = plan.waypoints.tolist()
    cmd.speed = plan.speeds[0]
    cmd.debug_vis = args_cli.markers
    # a touchdown on the pad is a contact, and the end of the path is not the end of the mission
    env_cfg.terminations.collision = None
    env_cfg.terminations.survey_complete = None
    env_cfg.seed = args_cli.seed
    if args_cli.video or "kit" in (args_cli.visualizer or []):
        from isaaclab_visualizers.kit import KitVisualizerCfg

        from drone_lab.tasks.survey.config.arl_robot_1.survey_env_cfg import overview_camera

        open_window = "kit" in (args_cli.visualizer or [])
        w, h = args_cli.video_size
        if args_cli.view == "chase":  # placed every step by FollowCamera; this is the first frame's view
            lookat = tuple(plan.waypoints[0])
            eye = tuple(np.add(lookat, FollowCamera.offset))
        else:
            eye, lookat = overview_camera(env_cfg.scene.terrain.terrain_generator)
        # background_color=None: the sky dome is the background, not a flat colour behind the terrain
        viz = KitVisualizerCfg(
            headless=not open_window, window_width=w, window_height=h, eye=eye, lookat=lookat, background_color=None
        )
        env_cfg.sim.visualizer_cfgs = [viz]
        if open_window:
            args_cli.kit_args = f"{args_cli.kit_args or ''} {window_kit_args()}".strip()
        else:
            args_cli.headless = True
    else:
        args_cli.headless = True


def window_kit_args() -> str:
    """Kit settings for the main window: its position and (unless --kit_ui) the viewport filling it."""
    kit = []
    if args_cli.window_pos is not None:
        kit += [f"--/app/window/x={args_cli.window_pos[0]}", f"--/app/window/y={args_cli.window_pos[1]}"]
    if not args_cli.kit_ui:
        kit.append("--/app/window/hideUi=true")
    return " ".join(kit)


def terrain_truth(site: TerrainSite, resolution: float = 0.25) -> tuple[np.ndarray, float, float]:
    """True terrain height over the whole site, cell centres on the ``resolution`` lattice of the site bounds.

    Returns ``(heights (ny, nx), x_min, y_min)`` with ``x_min`` / ``y_min`` the lower cell edges."""
    x0, x1, y0, y1 = site.bounds
    xs = x0 + resolution * np.arange(round((x1 - x0) / resolution) + 1)
    ys = y0 + resolution * np.arange(round((y1 - y0) / resolution) + 1)
    xx, yy = np.meshgrid(xs, ys)
    heights = site.heights(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return heights.astype(np.float32), x0 - 0.5 * resolution, y0 - 0.5 * resolution


def contact_force(contact_sensor) -> float:
    """Largest contact force on any body of the drone over the sensor history [N]."""
    forces = contact_sensor.data.net_normal_forces_w_history.torch[0]
    return float(torch.linalg.norm(forces, dim=-1).max())


def frame_boxes(camera, site_prim_path: str) -> list[dict]:
    """Labelled objects in the current frame of env 0, boxes in continuous pixel coordinates."""
    from drone_lab.sar.labels import instance_boxes
    from drone_lab.sar.scene import read_camera

    fr = read_camera(camera, 0, ("instance_segmentation",))
    out = []
    for b in instance_boxes(fr["instance_segmentation"], fr["info"], site_prim_path):
        x0, y0, x1, y1 = b.bbox_xyxy
        out.append({"class": b.label, "name": b.name, "box": [x0, y0, x1 + 1, y1 + 1], "pixels": b.pixels})
    return out


@dataclass
class Mission:
    """The site, the flight plan and its camera numbers: everything known before the simulator starts."""

    scenario: Scenario
    site: TerrainSite
    plan: MissionPlan
    geometry: dict[str, float]
    terrain_z: tuple[float, float]
    """Lowest and highest terrain point of the area and its turn band [m]."""


@dataclass
class Flight:
    """What the flight loop measured."""

    outcome: str = "timeout"
    t: float = 0.0
    wall_s: float = 0.0
    touchdown_t: float | None = None
    rows: list[list] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)
    stream_ms: list[float] = field(default_factory=list)
    viewer_ms: list[float] = field(default_factory=list)
    video_ms: list[float] = field(default_factory=list)
    video_t: list[float] = field(default_factory=list)
    """Sim time of every video frame [s]."""
    camera_frames: int = 0
    vram_peak_mib: int = 0


@dataclass
class Viewer:
    """The camera of the window and the video, and the extras only it sees (see :mod:`drone_lab.sar.chase_view`)."""

    viz: Any
    """The Kit visualizer (window or headless recorder)."""
    camera: FollowCamera | None
    """None: the fixed overview."""
    rays: ScanRays | None

    def update(self, pos: np.ndarray, hits: np.ndarray | None, dt: float) -> None:
        """Place the camera for the next render and show the latest scan (``hits`` None before the first one)."""
        if self.camera is not None:
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

            eye, target = self.camera.update(pos, None if hits is None else ground_below(hits), dt)
            eye, target = tuple(eye.tolist()), tuple(target.tolist())
            # as the visualizer's own asset tracking does: its view, then the renderer's camera (needed headless)
            self.viz.set_camera_view(eye, target)
            set_kit_renderer_camera_view(eye=eye, target=target, camera_prim_path=VIEWER_CAMERA)
        if self.rays is not None and hits is not None:
            self.rays.update(hits)


def setup_viewer(base, mission: Mission) -> Viewer | None:
    """Follow camera, ground skirt and scan rays for the Kit visualizer (None without one).

    The skirt and the rays are drawn only once the drone camera is in its own render partition: the detector
    must never see them.
    """
    import isaaclab.sim as sim_utils

    viz = find_kit_visualizer(base.sim)
    if viz is None:
        return None
    x0, x1, y0, y1 = mission.site.bounds
    camera = None
    if args_cli.view == "chase":
        area = mission.plan.area
        camera = FollowCamera(bounds=(x0 + 1.0, x1 - 1.0, y0 + 1.0, y1 - 1.0), swing_x=(area[0], area[1]))
    stage = base.sim.stage
    if camera is not None:
        set_focal_length(stage, VIEWER_CAMERA, camera.focal_length)
    drone_cameras = sim_utils.find_matching_prim_paths(base.scene.sensors["sar_camera"].cfg.prim_path)
    try:
        isolate_chase_view(stage, VIEWER_CAMERA, drone_cameras)
    except RuntimeError as exc:
        print(f"[sar_mission] WARNING: {exc}: the window shows the site only")
        return Viewer(viz, camera, None)
    rim_z = float(np.min(mission.site.heights(rim_points(mission.site.bounds))))  # the flat border of the site
    spawn_ground_skirt(stage, mission.site.bounds, rim_z - 0.005, f"{SITE}/Looks/ground_grass")
    rays = None
    if not args_cli.no_scan_rays:
        rays = ScanRays(stage, base.scene.sensors["mapping_scanner"].num_rays)
    return Viewer(viz, camera, rays)


def prepare_mission(env_cfg) -> Mission:
    """Regenerate the site on the CPU (it matches what the simulator spawns) and plan the flight over it."""
    from drone_lab.sar.scene import scenario_for

    site_cfg = env_cfg.scene.sar_site.spawn
    site = TerrainSite(site_cfg.terrain_generator)
    scenario = scenario_for(site_cfg, site)
    pad = next(o for o in scenario.objects if o.category == "landing_pad")
    home = (float(pad.center[0]), float(pad.center[1]), float(pad.aabb_max[2]))
    band = 0.5 * args_cli.spacing + 0.5  # the U-turns run this far outside the area
    terrain_z = site.terrain_z_range(inset=-band)
    flight_z = terrain_z[1] + args_cli.altitude
    plan = plan_mission(scenario.area, home, flight_z, args_cli.spacing, speed=args_cli.speed)
    geometry = survey_geometry(
        flight_z,
        args_cli.spacing,
        min(args_cli.speed, 1.25),  # flown speed: the policy tops out near 1.3 m/s
        SAR_CAMERA.focal_px,
        (SAR_CAMERA.width, SAR_CAMERA.height),
        min_clearance=args_cli.altitude + SAR_CAMERA.mount_pos[2],  # camera above the highest terrain point
    )
    return Mission(scenario, site, plan, geometry, terrain_z)


def announce(stream, mission: Mission) -> None:
    """MISSION, then the ground truth for the scorecard (GT_OBJECTS, GT_TERRAIN)."""
    plan = mission.plan
    stream.mission(
        area=plan.area,
        resolution=0.25,
        planned_path=plan.waypoints,
        survey_altitude=plan.flight_z,
        mission_id=f"sar-seed{args_cli.seed}-{datetime.now():%Y%m%d-%H%M%S}",
        name=f"sar_seed{args_cli.seed}",
        ground_z=plan.home[2],
        z_range=mission.terrain_z,
        home=list(plan.home),
        pattern="lawnmower",
        spacing=args_cli.spacing,
        speed=args_cli.speed,
        phases=plan.to_dict()["phases"],
        seed=args_cli.seed,
    )
    stream.gt_objects(gt_objects(mission.scenario.objects))
    heights, x_min, y_min = terrain_truth(mission.site)
    stream.gt_terrain(heights, x_min, y_min, 0.25)


def fly(
    base, policy, plan: MissionPlan, stream, clearance: ClearanceMonitor, viewer: Viewer | None = None, video=None
) -> Flight:
    """Fly the plan with the policy, stream every step, stop after the touchdown (or a crash or the time limit).

    ``viewer`` places the window / video camera and the scan rays before every step (so the step's render shows
    them); ``video`` is the mp4 writer or None."""
    step_dt = base.step_dt
    term = base.command_manager.get_term("survey_target")
    robot = base.scene["robot"]
    scanner = base.scene.sensors["mapping_scanner"]
    camera = base.scene.sensors["sar_camera"]
    contacts = base.scene.sensors["contact_forces"]
    settle_steps = max(0, round(args_cli.settle / step_dt))
    max_time = args_cli.max_time or (1.5 * plan.length / min(args_cli.speed, 1.0) + 60.0)
    motors_off = torch.full((base.num_envs, base.action_manager.total_action_dim), ZERO_THRUST, device=base.device)
    flight = Flight(vram_peak_mib=gpu_memory_mib() or 0)
    obs, _ = base.reset()
    pos = robot.data.root_pos_w.torch[0].cpu().numpy()
    hits = None  # no scan before the first step
    t_wall = time.time()
    print(f"[sar_mission] flying: {plan.length:.0f} m planned, time limit {max_time:.0f} s", flush=True)
    try:
        with torch.inference_mode():
            for step in range(math.ceil(max_time / step_dt)):
                s = float(term.s[0])
                phase = plan.phase(s)
                term.cfg.speed = 0.0 if step < settle_steps else plan.speed(s)
                actions = motors_off if flight.touchdown_t is not None else policy(obs["policy"])
                if viewer is not None:
                    t0 = time.perf_counter()
                    viewer.update(pos, hits, step_dt)
                    flight.viewer_ms.append(1000.0 * (time.perf_counter() - t0))
                t1 = time.perf_counter()
                obs, _, terminated, _, _ = base.step(actions)
                _ = camera.data  # a headless camera renders when its data is first read: count that as the step
                torch.cuda.synchronize()  # rendering done: the time after this is the link's own cost
                flight.step_ms.append(1000.0 * (time.perf_counter() - t1))
                t = flight.t = (step + 1) * step_dt
                if bool(terminated[0]):  # flipped or fell: the env has already been reset, do not stream
                    flight.outcome = "crashed"
                    print(f"[sar_mission] the drone crashed (flip / fall) at t={t:.1f} s in phase {phase}")
                    break
                force = contact_force(contacts)
                if force > 1.0 and flight.touchdown_t is None:
                    if phase not in ("descent", "touchdown"):
                        flight.outcome = "crashed"
                        print(f"[sar_mission] collision ({force:.1f} N) at t={t:.1f} s in phase {phase}")
                        break
                    flight.touchdown_t = t
                    print(f"[sar_mission] touchdown at t={t:.1f} s: motors off", flush=True)
                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                hits = scanner.data.ray_hits_w.torch[0].cpu().numpy()
                clearance.update(t, pos, phase, hits)
                if stream is not None:
                    t2 = time.perf_counter()
                    stream.telemetry(t, robot, wp_index=plan.waypoint_index(s), carrot=term.carrot_w[0], phase=phase)
                    stream.scan(t, scanner)
                    if step % max(1, args_cli.camera_every) == 0:
                        extra = {"gt_boxes": frame_boxes(camera, SITE)} if args_cli.gt_boxes else {}
                        flight.camera_frames += int(stream.camera(t, camera, **extra))
                    flight.stream_ms.append(1000.0 * (time.perf_counter() - t2))
                quat = robot.data.root_quat_w.torch[0].cpu().numpy()
                vel = robot.data.root_lin_vel_w.torch[0].cpu().numpy()
                carrot = term.carrot_w[0].cpu().numpy()
                flight.rows.append([round(t, 3), phase, *pos.round(4), *quat.round(5), *vel.round(4), *carrot.round(4)])
                if video is not None and step >= settle_steps:
                    t3 = time.perf_counter()
                    frame = capture_frame(base.sim, viewer.viz)
                    if frame is not None and frame.any():
                        video.append_data(frame)
                        flight.video_t.append(round(t, 3))
                    flight.video_ms.append(1000.0 * (time.perf_counter() - t3))
                if step % 100 == 0:
                    flight.vram_peak_mib = max(flight.vram_peak_mib, gpu_memory_mib() or 0)
                if step % 200 == 0:
                    rate = (step + 1) / max(time.time() - t_wall, 1e-6)
                    print(
                        f"[sar_mission] t={t:6.1f} s  {phase:9s} {100 * s / plan.length:5.1f} %  "
                        f"z {pos[2]:5.2f} m  sim {rate:4.1f} steps/s",
                        flush=True,
                    )
                if flight.touchdown_t is not None and t - flight.touchdown_t >= 1.0:
                    flight.outcome = "landed"
                    break
                if bool(term.done[0]) and float(term.time_since_done[0]) > 5.0:
                    flight.outcome = "complete"  # hovering over the pad without touching it
                    break
    except KeyboardInterrupt:
        flight.outcome = "aborted"
        print(f"\n[sar_mission] Ctrl+C at t={flight.t:.1f} s: ending the mission")
    flight.wall_s = time.time() - t_wall
    flight.vram_peak_mib = max(flight.vram_peak_mib, gpu_memory_mib() or 0)
    return flight


def save_outputs(run_dir: Path, mission: Mission, flight: Flight, stats: dict, link: dict | None) -> dict:
    """``mission.json`` (plan, outcome, timing, link, GPU memory) and ``trajectory.csv``; returns the summary."""
    wall = max(flight.wall_s, 1e-6)
    summary = {
        "task": TASK,
        "seed": args_cli.seed,
        "checkpoint": str(Path(args_cli.checkpoint).expanduser().resolve()),
        "outcome": flight.outcome,
        "sim_time_s": round(flight.t, 2),
        "wall_time_s": round(flight.wall_s, 1),
        "real_time_factor": round(flight.t / wall, 2),
        "plan": mission.plan.to_dict(),
        "survey_geometry": {k: round(v, 3) for k, v in mission.geometry.items()},
        "stats": stats,
        "timing_ms": {
            "env_step": percentiles(flight.step_ms),
            "streaming": percentiles(flight.stream_ms),
            "viewer": percentiles(flight.viewer_ms),
            "video": percentiles(flight.video_ms),
        },
        "loop_hz": round(len(flight.step_ms) / wall, 2),
        "camera_frames_sent": flight.camera_frames,
        "link": link,
        "link_mb_per_s": round(link.get("sent_bytes", 0) / 1e6 / wall, 2) if link else None,
        "vram_peak_mib": flight.vram_peak_mib,
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }
    (run_dir / "mission.json").write_text(json.dumps(summary, indent=1))
    with open(run_dir / "trajectory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "phase", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "cx", "cy", "cz"])
        w.writerows(flight.rows)
    if flight.video_t:  # lines the video up with the ground station's recording (drone_gcs compose_video.py)
        with open(run_dir / "mission_frames.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "t"])
            w.writerows(enumerate(flight.video_t))
    return summary


def main() -> None:
    import gymnasium as gym

    from isaaclab_tasks.utils import parse_env_cfg

    from drone_lab.sar.scene import get_spawned_scenario

    overrides = [*hydra_overrides, "physics=isaacsim_physx", f"env.scene.sar_site.spawn.seed={args_cli.seed}"]
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, overrides=overrides)
    t0 = time.time()
    mission = prepare_mission(env_cfg)
    plan, g = mission.plan, mission.geometry
    print(
        f"[sar_mission] site seed {args_cli.seed}: {len(mission.scenario.objects)} objects, pad at "
        f"({plan.home[0]:.1f}, {plan.home[1]:.1f}); flight z {plan.flight_z:.2f} m, {args_cli.spacing:g} m lines, "
        f"path {plan.length:.0f} m (CPU {time.time() - t0:.1f} s)"
    )
    print(
        f"[sar_mission] camera footprint {g['footprint_across_m']:.1f} x {g['footprint_along_m']:.1f} m "
        f"({g['gsd_cm_per_px']:.2f} cm/px), side overlap {g['side_overlap_pct']:.0f} %, "
        f"narrowest swath {g['min_swath_m']:.1f} m, forward overlap {g['forward_overlap_pct']:.0f} %"
    )
    configure(env_cfg, plan)
    run_dir = run_dir_for(args_cli.out, args_cli.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    sender = None
    if not args_cli.no_gcs:
        import_drone_gcs(args_cli.gcs_root)
        from drone_gcs.isaac_bridge import IsaacLabStreamer
        from drone_gcs.link import Sender

        sender = Sender(args_cli.gcs, name=f"isaac-sar-seed{args_cli.seed}")
        print(f"[sar_mission] ground-station link: {sender.address_str}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        base = env.unwrapped
        policy = torch.jit.load(str(Path(args_cli.checkpoint).expanduser()), map_location=base.device)
        policy.eval()
        spawned = get_spawned_scenario(SITE)
        if spawned is not None and spawned.ground_truth(SITE) != mission.scenario.ground_truth(SITE):
            print("[sar_mission] WARNING: the spawned site differs from the CPU one; using the spawned one")
            mission.scenario = spawned
        mission.scenario.save_ground_truth(run_dir / "ground_truth.json", SITE)

        stream = None
        if sender is not None:
            stream = IsaacLabStreamer(sender, env_id=0, origin=base.scene.env_origins[0], jpeg=args_cli.jpeg)
            announce(stream, mission)
            if args_cli.wait_gcs > 0 and not sender.wait_connected(args_cli.wait_gcs):
                print(f"[sar_mission] no ground station after {args_cli.wait_gcs:g} s: flying anyway (it may join)")
        viewer = setup_viewer(base, mission)
        video = None
        if args_cli.video:
            if viewer is None:
                print("[sar_mission] WARNING: no Kit visualizer is active; no video will be recorded.")
            else:
                video = open_mp4_writer(run_dir / "mission.mp4", args_cli.video_fps)
                speedup = playback_speedup(args_cli.video_fps, base.step_dt)
                print(f"[sar_mission] video: {args_cli.video_fps} fps = {speedup:g}x real time")

        clearance = ClearanceMonitor(mission.scenario.objects)
        flight = fly(base, policy, plan, stream, clearance, viewer, video)
        stats = {
            "closest_approach": clearance.summary(),
            "flight_time_s": round(flight.t, 2),
            "touchdown_t": flight.touchdown_t,
            "progress_pct": round(100.0 * float(base.command_manager.get_term("survey_target").s[0]) / plan.length, 1),
        }
        link = None
        if stream is not None:
            stream.end(flight.outcome, t=flight.t, flush_timeout=5.0, stats=stats)
            link = sender.stats()
            sender.close()
        if video is not None:
            video.close()
        summary = save_outputs(run_dir, mission, flight, stats, link)
        steps = summary["timing_ms"]
        print(
            f"[sar_mission] {flight.outcome} after {flight.t:.1f} s sim time ({flight.wall_s:.0f} s wall, "
            f"{summary['real_time_factor']:.2f}x real time, {summary['loop_hz']:.1f} steps/s); env.step "
            f"{steps['env_step']['mean']} ms mean, streaming {steps['streaming']['mean']} ms; "
            f"{flight.camera_frames} camera frames sent; closest approach {stats['closest_approach']['closest_m']} m; "
            f"VRAM peak {flight.vram_peak_mib} MiB",
            flush=True,
        )
        if link:
            print(
                f"[sar_mission] link: {link.get('sent', 0)} messages, {link.get('sent_bytes', 0) / 1e6:.0f} MB "
                f"({summary['link_mb_per_s']} MB/s), dropped {link.get('dropped', 0)} {link.get('dropped_by_type')}"
            )
        print(f"[sar_mission] wrote {run_dir}", flush=True)
        try:
            env.close()
        except Exception as exc:
            print(f"[sar_mission] WARNING: env.close() failed ({exc})")


if __name__ == "__main__":
    main()
