"""Rescue finale: a quadruped gets the drone's map and pins, plans a safe route over the rubble, walks to a person.

1. **Map:** the drone's elevation map (``--run``: a ``drone_lab`` survey run or a ``drone_gcs`` run, or
   ``--elevation`` + ``--grid``) and its pins (``--pins``, e.g. drone_gcs ``detections.json``). Without a drone
   run, ``--ground_truth`` (the default) rasterises the exact terrain the simulator builds, and ``--goal x y``
   places the person. A drone survey map covers the surveyed terrain only: when the start (the landing pad on the
   flat border) lies off it, the site's flat apron is added around the map (``--no_apron``: the start moves onto
   the map edge instead).
2. **Plan (CPU):** slope / step / roughness traversability + object footprints + robot-radius inflation, then
   cost-weighted A* with start/goal snapping and smoothing (``drone_lab.teaming``). Writes ``plan.png`` and
   ``plan.json``. ``--plan_only`` stops here (no simulator).
3. **Walk (GPU):** task ``DroneLab-Rescue-Go2-v0``: Isaac Lab's Go2 rough-terrain policy on the same survey
   terrain, its velocity command replaced by pure pursuit on the route. Physics ``newton_mjwarp`` (the backend
   the checkpoint was trained on) or ``isaacsim_physx``. After a fall the robot respawns where it fell (up to
   ``--retries`` times).
4. **Outputs** (``runs/<out>/``): ``plan.png``, ``plan.json``, ``walk.csv``, ``walk.png``, ``summary.json`` /
   ``summary.txt`` (success, time, path lengths, falls, ...), and with ``--video`` ``rescue.mp4``: a title card
   with the plan, the follow-cam walk with a minimap and a HUD, and an end card with the result; plus
   ``rescue_share.mp4``, a ~5x smaller copy for sharing.

Examples (env_isaaclab, from the project root)::

    python scripts/rescue_finale.py --plan_only                                  # CPU: plan on ground truth
    python scripts/rescue_finale.py --video --out rescue_gt                      # walk it (Go2, Newton)
    python scripts/rescue_finale.py --goal -16 5 --video --out hill              # another person
    python scripts/rescue_finale.py --run runs/lawnmower_v1 --pins <detections.json> --video --out from_drone
    python scripts/rescue_finale.py --physics isaacsim_physx --video --out physx  # same scene as the drone

Trailing ``key=value`` tokens are Hydra overrides of the task config, e.g.
``env.commands.base_velocity.lookahead_m=1.0``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

DEFAULT_CHECKPOINT = (
    "~/isaac_ws/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-11-01_15-38-28_demo_quadruped_cont/model_1498.pt"
)
"""Go2 rough-terrain policy: the 2026-11-01 demo run (600 iterations, Newton) resumed for 900 more iterations. On the
training curriculum it falls 4x less often than the 600-iteration checkpoint (base-contact terminations 5.1 % vs
19.2 %); on the finale routes both reach the person 8/8 (runs/eval_*, runs/stress_*)."""
DEFAULT_START = (-23.0, -18.0)
DEFAULT_GOAL = (-4.5, -12.5)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
src = parser.add_argument_group("map and pins")
src.add_argument("--run", type=str, default=None,
                 help="Drone run directory: drone_lab survey run (elevation_map.npy) or drone_gcs run (maps/).")
src.add_argument("--elevation", type=str, default=None, help="Elevation .npy (with --grid or a grid JSON next to it).")
src.add_argument("--grid", type=str, default=None, help="Grid JSON for --elevation (map_meta.json / grid.json).")
src.add_argument("--gcs_layer", choices=("mean", "max", "min"), default="mean", help="drone_gcs elevation layer.")
src.add_argument("--pins", type=str, default=None,
                 help="Pins JSON (drone_gcs detections.json, {'pins': [...]} or a list); "
                      "default: <run>/detections.json.")
src.add_argument("--all_pins", action="store_true", help="Also use unconfirmed pins.")
src.add_argument("--ground_truth", action="store_true",
                 help="Plan on the terrain ground truth (the default when no --run / --elevation is given).")
src.add_argument("--sar_seed", type=int, default=None,
                 help="Dressed SAR site of this seed (drone_lab.sar, the scene the drone mission flies over): its "
                      "people and props become the pins (exact footprints), the robot starts at its landing pad, "
                      "and the site is spawned in the simulator.")
src.add_argument("--resolution", type=float, default=0.25, help="Ground-truth cell size [m].")
src.add_argument("--no_apron", action="store_true",
                 help="Plan on the drone map alone. By default, when the start (e.g. the landing pad on the flat "
                      "border) lies off a drone map, the site's flat apron around the surveyed terrain is added to the "
                      "map; with this flag the start moves onto the map edge instead.")
plan_g = parser.add_argument_group("plan")
plan_g.add_argument("--start", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help=f"Robot start [m] (default {DEFAULT_START}, or next to the landing pad pin).")
plan_g.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help=f"Person to reach [m] (ground truth default {DEFAULT_GOAL}; with pins: --goal_pin).")
plan_g.add_argument("--goal_pin", type=str, default=None,
                    help="Pin id of the person to reach (default: the person pin closest to the start).")
plan_g.add_argument("--max_goal_snap", type=float, default=2.5,
                    help="The robot must get within this distance of the person's position (centre) [m]; a person "
                         "who cannot be reached that close counts as unreachable (auto-selection tries the next).")
plan_g.add_argument("--person_pose", choices=("lying", "standing"), default="lying",
                    help="How the person is placed in the scene (also sets the footprint radius).")
plan_g.add_argument("--max_slope", type=float, default=None, help="Steepest walkable slope [deg] (default 22).")
plan_g.add_argument("--max_step", type=float, default=None, help="Highest walkable step [m] (default 0.12).")
plan_g.add_argument("--max_roughness", type=float, default=None,
                    help="Roughest walkable ground [m RMS] (default 0.05).")
plan_g.add_argument("--robot_radius", type=float, default=None,
                    help="Clearance of the robot centre [m] (default 0.35).")
plan_g.add_argument("--plan_only", action="store_true", help="Plan, write plan.png / plan.json and stop (CPU only).")
walk_g = parser.add_argument_group("walk")
walk_g.add_argument("--task", type=str, default="DroneLab-Rescue-Go2-v0", help="Rescue task id.")
walk_g.add_argument("--register", type=str, default=None,
                    help="Python module (dotted name or .py file) imported before the task is resolved, e.g. one that "
                         "registers a rescue task for another robot built with apply_rescue_task().")
walk_g.add_argument("--robot_name", type=str, default="Go2", help="Robot name shown in the figures and the video.")
walk_g.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                    help="Locomotion policy: RSL-RL checkpoint (model_*.pt) or TorchScript (exported/policy.pt).")
walk_g.add_argument("--physics", dest="physics_backend", choices=("newton_mjwarp", "isaacsim_physx"),
                    default="newton_mjwarp", help="Physics backend (the Go2 checkpoint was trained on newton_mjwarp).")
walk_g.add_argument("--speed", type=float, default=0.7, help="Cruise speed [m/s] (policy range 0..1).")
walk_g.add_argument("--lookahead", type=float, default=0.8, help="Pure-pursuit look-ahead [m].")
walk_g.add_argument("--retries", type=int, default=2,
                    help="Falls allowed (respawn where it fell); 0 = stop at the first.")
walk_g.add_argument("--max_time", type=float, default=None, help="Give up after this much sim time [s].")
walk_g.add_argument("--hold", type=float, default=2.0, help="Keep recording this long after arriving [s].")
walk_g.add_argument("--num_envs", type=int, default=1,
                    help="Robots walking the same route at once (small spawn jitter; the video follows robot 0).")
walk_g.add_argument("--seed", type=int, default=0, help="Environment seed.")
vid_g = parser.add_argument_group("video")
vid_g.add_argument("--video", dest="record_video", action="store_true", help="Record runs/<out>/rescue.mp4.")
vid_g.add_argument("--camera", choices=("follow", "overview"), default="follow", help="Video camera.")
vid_g.add_argument("--video_fps", type=int, default=25, help="Playback frame rate (25 = real time with stride 2).")
vid_g.add_argument("--video_stride", type=int, default=2, help="Capture every N-th 0.02 s policy step.")
vid_g.add_argument("--video_size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
vid_g.add_argument("--no_cards", action="store_true", help="No title / end cards in the video.")
parser.add_argument("--out", type=str, default=None, help="Run name under runs/ (or a path).")
parser.add_argument("--theme", choices=("light", "dark"), default="light", help="Figure theme.")
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

"""Rest everything follows."""

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402

from drone_lab.teaming.inputs import (  # noqa: E402
    DEFAULT_FOOTPRINT_RADII,
    Pin,
    extend_with_apron,
    footprints_from_pins,
    ground_truth_elevation,
    load_elevation,
    load_pins,
    on_map,
    select_goal_pin,
    survey_site_bounds,
)
from drone_lab.teaming.planner import PlanResult, plan_path  # noqa: E402
from drone_lab.teaming.traversability import TraversabilityCfg, compute_traversability  # noqa: E402

PERSON_RADIUS = {"lying": 1.0, "standing": 0.45}
"""Footprint radius of the person the robot walks to, by pose [m] (lying: 1.8 m long body)."""


def log(msg: str) -> None:
    print(f"[rescue] {msg}", flush=True)


def resolve_run_dir(out: str | None) -> Path:
    if out and ("/" in out or out.startswith("~")):
        return Path(out).expanduser().resolve()
    return PROJECT_ROOT / "runs" / (out or f"{datetime.now():%Y%m%d_%H%M%S}_rescue")


# ------------------------------------------------------------------------------------------------------ plan
def build_plan(run_dir: Path):
    """Map + pins -> traversability -> route. Returns a dict with everything the walk and the figures need."""
    a = args_cli
    if a.run or a.elevation:
        z, grid, source = load_elevation(a.run or a.elevation, a.grid, layer=a.gcs_layer)
        mode = "drone_map"
    else:
        t0 = time.time()
        z, grid, source = ground_truth_elevation(a.resolution)
        mode = "ground_truth"
        log(f"ground truth rasterised in {time.time() - t0:.1f} s")
    log(f"map: {source}, {grid.nx} x {grid.ny} cells of {grid.resolution} m, "
        f"{100 * np.mean(np.isfinite(z)):.1f} % known")

    pins: list[Pin] = []
    sar_home = None
    if a.sar_seed is not None:
        from drone_lab.sar.scene import SarSiteCfg, scenario_for
        from drone_lab.teaming.inputs import pins_from_data

        gt = scenario_for(SarSiteCfg(seed=a.sar_seed)).ground_truth()
        pins = pins_from_data(gt["objects"], source=f"SAR site seed {a.sar_seed}")
        sar_home = np.array([gt["home"]["x"], gt["home"]["y"]], dtype=float)
        counts = {k: v for k, v in gt["counts"].items() if v}
        log(f"SAR site seed {a.sar_seed}: {counts}, landing pad at {sar_home.tolist()}")
    pins_path = a.pins
    if pins_path is None and a.run:
        for cand in ("detections.json", "pins.json"):
            if (Path(a.run).expanduser() / cand).is_file():
                pins_path = str(Path(a.run).expanduser() / cand)
                break
    if pins_path:
        pins = pins + load_pins(pins_path, confirmed_only=not a.all_pins)
        log(f"pins: {len(pins)} from {pins_path} ({', '.join(sorted({p.label for p in pins})) or 'none'})")

    # start: --start, else the SAR site's landing pad, else a landing pad pin, else the default spot on the border
    if a.start is not None:
        start = np.array(a.start, dtype=float)
    elif sar_home is not None:
        start = sar_home.copy()  # on the landing pad: the ground robot deploys from the drone's base
    else:
        pad = next((p for p in pins if p.label == "landing_pad"), None)
        start = pad.xy.copy() if pad is not None else np.array(DEFAULT_START, dtype=float)
    requested_start = start.copy()
    apron_cells = 0
    if mode == "drone_map" and not on_map(grid, start) and not a.no_apron:
        # a drone_lab survey map covers the surveyed terrain only, while the base sits on the flat border around it
        old_extent = grid.extent
        z, grid, apron = extend_with_apron(z, grid, *survey_site_bounds())
        apron_cells = int(apron.sum())
        log(f"the start {start.round(2).tolist()} is off the drone map {old_extent}: added the site's flat apron "
            f"around the surveyed terrain ({apron_cells} cells at z = 0; the drone's cells are unchanged) -> "
            f"{grid.nx} x {grid.ny} cells, {grid.extent}")
    if not on_map(grid, start):
        start = np.clip(start, [grid.x_min + 0.5, grid.y_min + 0.5], [grid.x_max - 0.5, grid.y_max - 0.5])
        log(f"WARNING: the start {requested_start.round(2).tolist()} is outside the map {grid.extent}: moved "
            f"{np.linalg.norm(start - requested_start):.1f} m onto the map edge, to {start.round(2).tolist()}")
    # goal: --goal (a person placed there), else a person pin (by id, or the reachable one about 20 m away)
    if a.goal is not None:
        candidates = [Pin(id="goal", label="person", x=float(a.goal[0]), y=float(a.goal[1]),
                          radius=PERSON_RADIUS[a.person_pose], pose=a.person_pose)]
        pins = [p for p in pins if not (p.label == "person" and np.linalg.norm(p.xy - candidates[0].xy) < 0.5)]
        pins = pins + candidates
    elif a.goal_pin is not None:
        chosen = select_goal_pin(pins, pin_id=a.goal_pin)
        if chosen is None:
            sys.exit(f"[rescue] no pin with id {a.goal_pin}")
        candidates = [chosen]
    else:
        people = [p for p in pins if p.label == "person"]
        if people:
            candidates = sorted(people, key=lambda p: abs(float(np.linalg.norm(p.xy - start)) - 20.0))
        else:
            if pins:
                log("no person pin; using the default goal")
            candidates = [Pin(id="goal", label="person", x=DEFAULT_GOAL[0], y=DEFAULT_GOAL[1],
                              radius=PERSON_RADIUS[a.person_pose], pose=a.person_pose)]
            pins = pins + candidates

    overrides = {k: v for k, v in (("max_slope_deg", a.max_slope), ("max_step_m", a.max_step),
                                   ("max_roughness_m", a.max_roughness), ("robot_radius_m", a.robot_radius))
                 if v is not None}
    tcfg = TraversabilityCfg(**overrides)
    from drone_lab.teaming.traversability import Footprint

    t0 = time.time()
    for goal_pin in candidates:
        footprints = footprints_from_pins(pins, radii={"person": DEFAULT_FOOTPRINT_RADII["person"]})
        if goal_pin.polygon is None:
            # a person without an exact outline: their footprint follows the pose (a lying body is 1.8 m long)
            footprints = [f for f in footprints if not (abs(f.x - goal_pin.x) < 1e-6 and abs(f.y - goal_pin.y) < 1e-6)]
            radius = PERSON_RADIUS.get(goal_pin.pose or a.person_pose, PERSON_RADIUS["lying"])
            footprints.append(Footprint(goal_pin.x, goal_pin.y, radius, "person (goal)"))
        tmap = compute_traversability(z, grid, tcfg, footprints=footprints)
        plan: PlanResult = plan_path(tmap, start, goal_pin.xy, max_goal_snap_m=a.max_goal_snap)
        if plan.success:
            break
        if plan.start_failed:  # the start's problem, not this person's: no other person is reachable either
            log(f"no route from the start {start.round(2).tolist()}: {plan.message}")
            break
        log(f"no route to {goal_pin.label} {goal_pin.id} at {goal_pin.xy.round(2).tolist()}: {plan.message}")
    log(f"traversability {tmap.summary()}")
    log(f"goal: {goal_pin.label} {goal_pin.id} at {goal_pin.xy.round(2).tolist()} "
        f"({np.linalg.norm(goal_pin.xy - start):.1f} m from the start {start.round(2).tolist()})")
    log(f"plan ({time.time() - t0:.2f} s): {plan.message}")

    from drone_lab.teaming.plots import plot_plan

    src_label = ("terrain ground truth" if mode == "ground_truth" else "drone elevation map") + \
        f" ({grid.resolution:g} m cells)"
    if apron_cells:
        src_label += " + the site's flat apron around it"
    if a.sar_seed is not None:
        src_label += f", SAR site seed {a.sar_seed}"
    plot_plan(tmap, plan, run_dir / "plan.png", pins=pins, goal_pin=goal_pin, source=src_label, theme=a.theme,
              full_view=True, robot=a.robot_name)
    plan_json = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "map_mode": mode,
        "map_source": source,
        "grid": grid.to_dict(),
        "apron_cells": apron_cells,
        "start_requested_xy": [float(v) for v in requested_start],
        "start_xy": [float(v) for v in start],
        "person_xy": [float(goal_pin.x), float(goal_pin.y)],
        "person_pose": goal_pin.pose or a.person_pose,
        "sar_seed": a.sar_seed,
        "goal_pin": goal_pin.to_dict(),
        "pins": [p.to_dict() for p in pins],
        "traversability_cfg": tcfg.to_dict(),
        "traversability": tmap.summary(),
        "plan": plan.to_dict(),
        "path_xy": np.round(plan.path_xy, 4).tolist(),
        "path_z": np.round(plan.path_z, 4).tolist(),
        "raw_xy": np.round(plan.raw_xy, 4).tolist(),
    }
    (run_dir / "plan.json").write_text(json.dumps(plan_json, indent=1))
    np.savez_compressed(run_dir / "traversability.npz", elevation=tmap.elevation, cost=tmap.cost,
                        lethal=tmap.lethal, slope_deg=tmap.slope_deg, step=tmap.step, roughness=tmap.roughness,
                        clearance=tmap.clearance, grid=np.array([grid.x_min, grid.y_min, grid.resolution,
                                                                  grid.nx, grid.ny]))
    log(f"wrote {run_dir / 'plan.png'} and plan.json")
    return {"tmap": tmap, "plan": plan, "pins": pins, "goal_pin": goal_pin, "start": start, "mode": mode,
            "source": src_label}


# ------------------------------------------------------------------------------------------------------ walk
def configure_env(env_cfg, ctx, max_time: float):
    """Route, person, follower settings, episode length and video camera."""
    from drone_lab.tasks.rescue.rescue_env_cfg import person_asset_cfg

    plan: PlanResult = ctx["plan"]
    tmap = ctx["tmap"]
    goal = ctx["goal_pin"]
    cmd = env_cfg.commands.base_velocity
    cmd.path_xy = np.round(plan.path_xy, 4).tolist()
    cmd.path_z = None  # heights come from the simulated terrain itself (exact), not from the planning map
    cmd.person_xy = (float(goal.x), float(goal.y))
    cmd.cruise_speed = float(args_cli.speed)
    cmd.lookahead_m = float(args_cli.lookahead)
    cmd.respawn_on_fall = args_cli.retries > 0
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.num_envs > 1:
        env_cfg.events.reset_base.params.update({"xy_jitter": 0.15, "yaw_jitter": 0.15})
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = max_time + 30.0
    if args_cli.sar_seed is not None:
        # the dressed site of the drone mission: people (the goal among them), props, ground materials, sun
        from isaaclab.assets import AssetBaseCfg

        from drone_lab.sar.scene import SarSiteCfg, sun_light_cfg

        env_cfg.scene.sar_site = AssetBaseCfg(prim_path="/World/sar", spawn=SarSiteCfg(seed=args_cli.sar_seed))
        env_cfg.scene.sun_light = sun_light_cfg()
        env_cfg.scene.person = None
    else:
        # the person: on the terrain under the pin, across the final approach (the robot walks up to their side)
        end_dir = plan.path_xy[-1] - plan.path_xy[max(0, len(plan.path_xy) - 6)] if len(plan.path_xy) > 1 else [1, 0]
        yaw = math.atan2(end_dir[1], end_dir[0])
        pz = float(np.nan_to_num(tmap.height_at(np.array([[goal.x, goal.y]]))[0], nan=0.0))
        lying = (goal.pose or args_cli.person_pose) != "standing"
        env_cfg.scene.person = person_asset_cfg((goal.x, goal.y), pz, lying=lying, yaw=yaw)
    if args_cli.record_video:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        open_window = "kit" in (args_cli.visualizer or [])
        w, h = args_cli.video_size
        eye, lookat = overview_camera(plan)
        env_cfg.sim.visualizer_cfgs = [
            KitVisualizerCfg(headless=not open_window, window_width=w, window_height=h, eye=eye, lookat=lookat)
        ]
        args_cli.enable_cameras = True
        if not open_window:
            args_cli.headless = True


def overview_camera(plan: PlanResult):
    """High oblique camera (eye, lookat) that frames the whole route."""
    p = plan.path_xy
    lo, hi = p.min(axis=0), p.max(axis=0)
    c = 0.5 * (lo + hi)
    size = max(float(np.max(hi - lo)), 6.0)
    zc = float(np.mean(plan.path_z)) if len(plan.path_z) else 0.0
    eye = (float(c[0] - 0.35 * size), float(c[1] - 0.95 * size), zc + 0.75 * size)
    return eye, (float(c[0]), float(c[1]), zc)


def import_register_module(name: str):
    """Import a module that registers extra tasks (dotted module name or path to a .py file)."""
    import importlib
    import importlib.util

    path = Path(name).expanduser()
    if path.suffix == ".py" and path.is_file():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(name)
    log(f"registered tasks from {name}")
    return module


def load_policy(checkpoint: str, env, device: str):
    """Return ``(policy, kind)``; ``policy(obs) -> actions`` for the policy observation group."""
    import torch

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        policy = torch.jit.load(str(path), map_location=device)
        policy.eval()
        return policy, "torchscript"
    except Exception:  # not a TorchScript archive -> RSL-RL training checkpoint
        pass
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    agent_cfg.device = device
    runner = create_rsl_rl_runner(RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), agent_cfg)
    runner.load(str(path), map_location=device)
    runner.alg.eval_mode()
    policy = runner.alg.get_policy().as_jit().to(device)
    policy.eval()
    return policy, "rsl_rl"


def termination_causes(base, env_id: int) -> list[str]:
    """Names of the termination terms that ended ``env_id``'s last episode (e.g. ``base_contact``, ``flipped``)."""
    tm = base.termination_manager
    names = list(tm.active_terms)
    flags = getattr(tm, "_last_episode_dones", None)  # kept across the reset that follows the termination
    try:
        row = flags[env_id].tolist() if flags is not None else [bool(tm.get_term(n)[env_id]) for n in names]
    except Exception:
        return []
    return [n for n, v in zip(names, row, strict=False) if v]


class FollowCam:
    """Smoothed 3/4-rear chase camera for a Kit visualizer (world-frame eye/target set before every capture)."""

    def __init__(self, viz, back: float = 2.6, side: float = 1.1, up: float = 1.5, ahead: float = 1.4):
        self.viz = viz
        self.back, self.side, self.up, self.ahead = back, side, up, ahead
        self.pos = None
        self.yaw = None
        self.arrival_m = 3.5
        self.blend = 0.0

    def update(self, pos: np.ndarray, yaw: float, alpha_pos: float = 0.25, alpha_yaw: float = 0.08,
               person=None, to_go: float | None = None):
        """Chase the robot; over the last ``arrival_m`` metres pull back and frame the robot and the person."""
        if self.pos is None:
            self.pos, self.yaw = np.array(pos, dtype=float), float(yaw)
        else:
            self.pos = (1 - alpha_pos) * self.pos + alpha_pos * np.asarray(pos, dtype=float)
            dyaw = (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += alpha_yaw * dyaw
        f = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        left = np.array([-f[1], f[0], 0.0])
        target = self.pos + self.ahead * f + np.array([0.0, 0.0, -0.15])
        back, up = self.back, self.up
        if person is not None and to_go is not None:
            b = float(np.clip((self.arrival_m - to_go) / (0.6 * self.arrival_m), 0.0, 1.0))
            b = b * b * (3 - 2 * b)  # smoothstep
            self.blend = max(self.blend, b)  # never swing back once arriving
            mid = 0.5 * (self.pos + np.asarray(person, dtype=float)) + np.array([0.0, 0.0, 0.45])
            target = (1 - self.blend) * target + self.blend * mid
            back, up = back + 1.6 * self.blend, up + 0.9 * self.blend
        eye = self.pos - back * f + self.side * (1 + self.blend) * left + np.array([0.0, 0.0, up])
        self.apply(eye, target)

    def apply(self, eye, target):
        eye = tuple(float(v) for v in eye)
        target = tuple(float(v) for v in target)
        self.viz.set_camera_view(eye, target)
        try:
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

            set_kit_renderer_camera_view(eye=np.array(eye), target=np.array(target),
                                         camera_prim_path="/OmniverseKit_Persp")
        except Exception:
            pass


class Overlay:
    """HUD text and a minimap with the robot's trail, drawn onto the video frames (PIL)."""

    def __init__(self, ctx, size_px: int = 280, theme: str = "light"):
        from matplotlib import font_manager
        from PIL import ImageFont

        from drone_lab.teaming.plots import render_minimap

        self.mini, self.to_px = render_minimap(ctx["tmap"], ctx["plan"], pins=ctx["pins"], goal_pin=ctx["goal_pin"],
                                               size_px=size_px, theme=theme)
        self.caption = "terrain ground truth + route" if ctx["mode"] == "ground_truth" else "drone map + route"
        font_path = font_manager.findfont("DejaVu Sans")
        bold_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
        self.font = ImageFont.truetype(font_path, 19)
        self.font_small = ImageFont.truetype(font_path, 17)
        self.font_bold = ImageFont.truetype(bold_path, 21)
        self.trail: list[tuple[float, float]] = []

    def draw(self, frame: np.ndarray, *, t: float, to_go: float, speed: float, falls: int, status: str,
             robot_xy) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # HUD (top-left)
        title = f"Rescue finale: the {args_cli.robot_name} walks the drone's route"
        line = f"t {t:5.1f} s    to go {to_go:4.1f} m    speed {speed:4.2f} m/s    falls {falls}"
        w = 28 + max(d.textlength(title, font=self.font_bold), d.textlength(line, font=self.font))
        d.rounded_rectangle((14, 12, 14 + w, 12 + 74), radius=10, fill=(16, 16, 15, 160))
        d.text((28, 18), title, font=self.font_bold, fill=(240, 239, 236, 255))
        d.text((28, 50), line, font=self.font, fill=(225, 224, 217, 255))
        if status:
            color = (12, 163, 12, 230) if status.startswith("Reached") else (208, 59, 59, 230)
            icon = "✓ " if status.startswith("Reached") else "✗ "
            wb = 28 + d.textlength(icon + status, font=self.font_bold)
            d.rounded_rectangle((14, 94, 14 + wb, 94 + 40), radius=10, fill=color)
            d.text((28, 100), icon + status, font=self.font_bold, fill=(255, 255, 255, 255))
        img = Image.alpha_composite(img, layer)
        # minimap (bottom-right) with the trail and the robot
        mini = Image.fromarray(self.mini.copy())
        md = ImageDraw.Draw(mini)
        if robot_xy is not None:
            self.trail.append((float(robot_xy[0]), float(robot_xy[1])))
        if len(self.trail) > 1:
            cols, rows = self.to_px(np.array([p[0] for p in self.trail]), np.array([p[1] for p in self.trail]))
            md.line(list(zip(cols.tolist(), rows.tolist(), strict=False)), fill=(42, 120, 214), width=3)
        if robot_xy is not None:
            c, r = self.to_px(np.array([robot_xy[0]]), np.array([robot_xy[1]]))
            c, r = float(c[0]), float(r[0])
            md.ellipse((c - 7, r - 7, c + 7, r + 7), fill=(42, 120, 214), outline=(252, 252, 251), width=2)
        W, H = img.size
        m = mini.size[0]
        border = Image.new("RGBA", (m + 6, m + 6), (252, 252, 251, 255))
        img.paste(border, (W - m - 22, H - m - 22))
        img.paste(mini.convert("RGBA"), (W - m - 19, H - m - 19))
        ImageDraw.Draw(img).text((W - m - 19, H - m - 46), self.caption, font=self.font_small,
                                 fill=(240, 239, 236, 255), stroke_width=2, stroke_fill=(16, 16, 15, 255))
        return np.asarray(img.convert("RGB"))


SHARE_CRF = 26
"""x264 quality of the sharing copy (``rescue_share.mp4``): about 5x smaller than the master (imageio quality 7, which
the textured ground pushes to ~10 Mbit/s), same size and frame rate."""


def write_share_copy(master: Path, dst: Path, crf: int = SHARE_CRF, timeout_s: float = 300.0) -> Path | None:
    """Re-encode the video into a smaller H.264 copy for sharing (README, chat); None if ffmpeg is missing or fails."""
    import shutil
    import subprocess

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    if not exe:
        log("WARNING: no ffmpeg; no sharing copy of the video")
        return None
    cmd = [exe, "-nostdin", "-v", "error", "-y", "-i", str(master), "-c:v", "libx264", "-preset", "medium",
           "-crf", str(int(crf)), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception as exc:
        log(f"WARNING: sharing copy failed ({exc})")
        return None
    if r.returncode != 0 or not Path(dst).is_file():
        log(f"WARNING: sharing copy failed (ffmpeg exit {r.returncode}: {r.stderr.strip()[-300:]})")
        return None
    return Path(dst)


def card_frames(png: Path, size: tuple[int, int], seconds: float, fps: int):
    """A figure, letterboxed to the video size, repeated for ``seconds``."""
    from PIL import Image

    W, H = size
    im = Image.open(png).convert("RGB")
    scale = min(W / im.width, H / im.height)
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    frame = np.asarray(canvas)
    return [frame] * max(1, int(round(seconds * fps)))


def walk(ctx, run_dir: Path) -> int:
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import drone_lab.tasks  # noqa: F401  (registers the DroneLab gym ids)
    from drone_lab.video import capture_frame, find_kit_visualizer, open_mp4_writer, playback_speedup

    plan: PlanResult = ctx["plan"]
    tmap = ctx["tmap"]
    goal = ctx["goal_pin"]
    max_time = args_cli.max_time or (90.0 + 3.0 * plan.length_m / max(args_cli.speed, 0.1))
    if args_cli.register:
        import_register_module(args_cli.register)
    hydra_overrides.append(f"physics={args_cli.physics_backend}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            overrides=hydra_overrides)
    configure_env(env_cfg, ctx, max_time)
    log(f"task {args_cli.task}: physics {type(env_cfg.sim.physics).__name__} ({args_cli.physics_backend}), "
        f"{env_cfg.scene.num_envs} env(s), dt {env_cfg.sim.dt} x {env_cfg.decimation}, max time {max_time:.0f} s")

    exit_code = 0
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        base = env.unwrapped
        device = base.device
        policy, policy_kind = load_policy(args_cli.checkpoint, base, device)
        log(f"policy: {args_cli.checkpoint} ({policy_kind})")
        term = base.command_manager.get_term("base_velocity")
        robot = base.scene["robot"]
        n = base.num_envs
        step_dt = base.step_dt
        print(str(term), flush=True)

        writer = viz = cam = overlay = None
        fps = args_cli.video_fps
        stride = max(1, args_cli.video_stride)
        if args_cli.record_video:
            viz = find_kit_visualizer(base.sim)
            if viz is None:
                log("WARNING: no Kit visualizer is active; no video")
            else:
                writer = open_mp4_writer(run_dir / "rescue.mp4", fps)
                if args_cli.camera == "follow":
                    cam = FollowCam(viz)
                try:
                    overlay = Overlay(ctx, theme=args_cli.theme)
                except Exception as exc:  # the overlay is decoration only
                    log(f"WARNING: no HUD overlay ({exc})")
                if not args_cli.no_cards:
                    for fr in card_frames(run_dir / "plan.png", tuple(args_cli.video_size), 3.0, fps):
                        writer.append_data(fr)
                log(f"video: every {stride} steps at {fps} fps = {playback_speedup(fps, step_dt, stride):g}x real time")

        obs, _ = base.reset()
        # the person's ground point (camera framing on arrival)
        person_w = np.array([goal.x, goal.y, float(term.terrain_height(np.array([[goal.x, goal.y]]))[0])])
        max_steps = int(math.ceil(max_time / step_dt))
        rows = []
        falls: list[list[tuple[float, float, float]]] = [[] for _ in range(n)]
        fall_causes: list[list[str]] = [[] for _ in range(n)]
        reached_t = np.full(n, np.nan)
        failed = np.array([""] * n, dtype=object)
        finished = np.zeros(n, dtype=bool)
        best_progress = np.zeros(n)
        best_progress_t = np.zeros(n)
        prev_pos = robot.data.root_pos_w.torch.detach().cpu().numpy().copy()
        t_wall = time.time()
        steps_done = 0
        interrupted = None
        status_text = ""
        try:
            with torch.inference_mode():
                for step in range(max_steps):
                    actions = policy(obs["policy"])
                    obs, _, terminated, truncated, _ = base.step(actions)
                    steps_done = step + 1
                    t = steps_done * step_dt
                    pos = robot.data.root_pos_w.torch.detach().cpu().numpy()
                    yaw = robot.data.heading_w.torch.detach().cpu().numpy()
                    vel_b = robot.data.root_lin_vel_b.torch.detach().cpu().numpy()
                    wz = robot.data.root_ang_vel_b.torch.detach().cpu().numpy()[:, 2]
                    cmd = term.command.detach().cpu().numpy()
                    term_np = terminated.detach().cpu().numpy()
                    trunc_np = truncated.detach().cpu().numpy()
                    for e in range(n):
                        if finished[e]:
                            continue
                        if term_np[e]:
                            causes = termination_causes(base, e)
                            falls[e].append((t, float(prev_pos[e, 0]), float(prev_pos[e, 1])))
                            fall_causes[e].append("+".join(causes) or "?")
                            log(f"robot {e} fell at t={t:.1f} s at ({prev_pos[e, 0]:.1f}, {prev_pos[e, 1]:.1f}), "
                                f"progress {best_progress[e]:.1f} / {term.path_length:.1f} m, cause "
                                f"{'+'.join(causes) or '?'}"
                                + (" -> respawn" if len(falls[e]) <= args_cli.retries else " -> out of retries"))
                            if len(falls[e]) > args_cli.retries:
                                failed[e] = "fell"
                                finished[e] = True
                                continue
                        elif trunc_np[e] and np.isnan(reached_t[e]):
                            failed[e] = "timeout"
                            finished[e] = True
                            continue
                        if term.goal_reached[e] and np.isnan(reached_t[e]):
                            reached_t[e] = t
                            log(f"robot {e} reached the person at t={t:.1f} s "
                                f"({np.linalg.norm(pos[e, :2] - term.person_xy):.2f} m from them)")
                        if term.progress_s[e] > best_progress[e] + 0.25:
                            best_progress[e] = term.progress_s[e]
                            best_progress_t[e] = t
                        elif np.isnan(reached_t[e]) and t - best_progress_t[e] > 30.0:
                            failed[e] = "stuck"
                            finished[e] = True
                            log(f"robot {e} made no progress for 30 s at t={t:.1f} s: stuck")
                            continue
                        rows.append((t, e, *pos[e, :3], yaw[e], *vel_b[e, :2], wz[e], *cmd[e],
                                     term.progress_s[e], term.cross_track[e], term.distance_to_goal[e],
                                     float(np.linalg.norm(pos[e, :2] - term.person_xy)), len(falls[e]),
                                     term.modes[e]))
                        if not np.isnan(reached_t[e]) and t - reached_t[e] >= args_cli.hold:
                            finished[e] = True
                    prev_pos = pos.copy()
                    if not np.isnan(reached_t[0]):
                        status_text = f"Reached the person in {reached_t[0]:.1f} s"
                    elif failed[0]:
                        status_text = {"fell": "Fell (out of retries)", "timeout": "Out of time",
                                       "stuck": "Stuck"}.get(failed[0], failed[0])
                    if writer is not None and step % stride == 0:
                        try:
                            if cam is not None:
                                cam.update(pos[0, :3], float(yaw[0]), person=person_w,
                                           to_go=float(term.distance_to_goal[0]))
                            frame = capture_frame(base.sim, viz)
                            if frame is not None and frame.any():
                                if overlay is not None:
                                    frame = overlay.draw(frame, t=t, to_go=float(term.distance_to_goal[0]),
                                                         speed=float(np.hypot(*vel_b[0, :2])),
                                                         falls=len(falls[0]), status=status_text,
                                                         robot_xy=pos[0, :2])
                                writer.append_data(frame)
                        except Exception as exc:  # a video problem must not cost the walk
                            log(f"WARNING: video capture failed ({exc}); continuing without video")
                            writer.close()
                            writer = None
                    if step % 250 == 0:
                        log(f"t={t:6.1f} s  progress {term.progress_s[0]:5.1f}/{term.path_length:.1f} m  "
                            f"to go {term.distance_to_goal[0]:5.1f} m  cmd ({cmd[0, 0]:+.2f}, {cmd[0, 1]:+.2f}, "
                            f"{cmd[0, 2]:+.2f})  mode {term.modes[0]}  falls {len(falls[0])}  "
                            f"wall {time.time() - t_wall:.0f} s")
                    if finished.all():
                        break
        except KeyboardInterrupt:
            interrupted = "stopped with Ctrl+C"
            log("Ctrl+C: saving what was walked so far")
        except Exception as exc:
            interrupted = f"walk loop error: {type(exc).__name__}: {exc}"
            log(f"ERROR in the walk loop ({type(exc).__name__}: {exc}); saving what was walked so far")
            import traceback

            traceback.print_exc()
            exit_code = 1
        sim_time = steps_done * step_dt
        wall = time.time() - t_wall
        log(f"walk ended after {sim_time:.1f} s sim time ({wall:.0f} s wall, "
            f"{steps_done / max(wall, 1e-6):.0f} steps/s)")

        # ---------------------------------------------------------------------------------- results
        cols = ["t", "env", "x", "y", "z", "yaw", "vx_b", "vy_b", "wz_b", "cmd_vx", "cmd_vy", "cmd_wz", "progress_m",
                "cross_track_m", "dist_goal_m", "dist_person_m", "falls", "mode"]
        with open(run_dir / "walk.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(f"{v:.4f}" if isinstance(v, float | np.floating) else str(v) for v in r) + "\n")
        envs = []
        for e in range(n):
            er = [r for r in rows if r[1] == e]
            xy = np.array([[r[2], r[3]] for r in er]) if er else np.zeros((0, 2))
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(0)
            walked = float(np.sum(steps[steps < 0.5]))  # respawn jumps are not walked
            ct = np.abs(np.array([r[13] for r in er])) if er else np.zeros(0)
            spd = np.array([math.hypot(r[6], r[7]) for r in er]) if er else np.zeros(0)
            clear = tmap.sample(tmap.clearance, xy) if len(xy) else np.zeros(0)
            ok = not np.isnan(reached_t[e])
            outcome = "reached" if ok else (failed[e] or ("interrupted" if interrupted else "timeout"))
            envs.append({
                "env": e,
                "success": bool(ok),
                "outcome": outcome,
                "time_s": round(float(reached_t[e]), 2) if ok else round(sim_time, 2),
                "walked_length_m": round(walked, 2),
                "falls": len(falls[e]),
                "fall_events": [{"t": round(f[0], 2), "x": round(f[1], 2), "y": round(f[2], 2), "cause": c}
                                for f, c in zip(falls[e], fall_causes[e], strict=False)],
                "final_distance_to_person_m": round(float(er[-1][15]), 3) if er else None,
                "final_distance_to_goal_m": round(float(er[-1][14]), 3) if er else None,
                "progress_m": round(float(best_progress[e]), 2),
                "cross_track_mean_m": round(float(np.mean(ct)), 3) if len(ct) else None,
                "cross_track_max_m": round(float(np.max(ct)), 3) if len(ct) else None,
                "mean_speed_mps": round(float(np.mean(spd)), 3) if len(spd) else None,
                "min_clearance_m": round(float(np.nanmin(clear)), 3) if len(clear) else None,
            })
        e0 = envs[0]
        summary = {
            **e0,
            "planned_length_m": round(plan.length_m, 2),
            "robots": n,
            "success_rate": round(float(np.mean([e["success"] for e in envs])), 3),
            "envs": envs if n > 1 else None,
            "person_xy": [round(goal.x, 3), round(goal.y, 3)],
            "start_xy": [round(float(v), 3) for v in ctx["start"]],
            "map": ctx["source"],
            "task": args_cli.task,
            "physics": args_cli.physics_backend,
            "checkpoint": str(Path(args_cli.checkpoint).expanduser().resolve()),
            "policy_kind": policy_kind,
            "cruise_speed_mps": args_cli.speed,
            "lookahead_m": args_cli.lookahead,
            "retries": args_cli.retries,
            "sim_time_s": round(sim_time, 2),
            "wall_time_s": round(wall, 1),
            "interrupted": interrupted,
            "plan": plan.to_dict(),
            "created": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
        }
        if args_cli.record_video:
            summary["video"] = {"file": "rescue.mp4", "fps": fps, "stride": stride,
                                "speedup": playback_speedup(fps, step_dt, stride), "camera": args_cli.camera}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        text = format_summary(summary)
        (run_dir / "summary.txt").write_text(text + "\n")
        print(text, flush=True)
        # figure of the walk (robot 0)
        try:
            from drone_lab.teaming.plots import plot_walk

            er = [r for r in rows if r[1] == 0]
            walk_d = {
                "t": np.array([r[0] for r in er]), "x": np.array([r[2] for r in er]),
                "y": np.array([r[3] for r in er]), "speed": np.array([math.hypot(r[6], r[7]) for r in er]),
                "cmd_vx": np.array([r[9] for r in er]), "dist_goal": np.array([r[14] for r in er]),
                "falls": falls[0],
            }
            plot_walk(tmap, plan, walk_d, run_dir / "walk.png", pins=ctx["pins"], goal_pin=goal, summary=summary,
                      theme=args_cli.theme, robot=args_cli.robot_name)
            log(f"wrote {run_dir / 'walk.png'}")
        except Exception as exc:
            log(f"WARNING: walk figure failed ({exc})")
        if writer is not None:
            try:
                if not args_cli.no_cards and (run_dir / "walk.png").is_file():
                    for fr in card_frames(run_dir / "walk.png", tuple(args_cli.video_size), 4.0, fps):
                        writer.append_data(fr)
                writer.close()
                log(f"wrote {run_dir / 'rescue.mp4'} ({(run_dir / 'rescue.mp4').stat().st_size / 1e6:.1f} MB)")
                share = write_share_copy(run_dir / "rescue.mp4", run_dir / "rescue_share.mp4")
                if share is not None:
                    log(f"wrote {share} ({share.stat().st_size / 1e6:.1f} MB, x264 crf {SHARE_CRF}: for sharing)")
            except Exception as exc:
                log(f"WARNING: closing the video failed ({exc})")
        try:
            env.close()
        except Exception as exc:
            log(f"WARNING: env.close() failed ({exc})")
        # still inside launch_simulation: closing Kit ends the process (fast shutdown); nothing may follow
        log(f"done: {run_dir}")
        if interrupted and interrupted.startswith("stopped"):
            raise KeyboardInterrupt
    return exit_code


def format_summary(s: dict) -> str:
    lines = [
        "Rescue finale",
        f"  outcome          : {s['outcome']}{' (success)' if s['success'] else ''}",
        f"  time             : {s['time_s']:.1f} s" + ("" if s["success"] else " (not arrived)"),
        f"  path length      : planned {s['planned_length_m']:.1f} m, walked {s['walked_length_m']:.1f} m",
        f"  falls            : {s['falls']}" + (f" at {[(f['x'], f['y']) for f in s['fall_events']]}"
                                                 if s["falls"] else ""),
        f"  stopped          : {s['final_distance_to_person_m']} m from the person"
        if s.get("final_distance_to_person_m") is not None else "  stopped          : -",
        f"  cross-track      : mean {s['cross_track_mean_m']} m, max {s['cross_track_max_m']} m",
        f"  mean speed       : {s['mean_speed_mps']} m/s (cruise {s['cruise_speed_mps']} m/s)",
        f"  min clearance    : {s['min_clearance_m']} m to the nearest no-go cell (planning map)",
        f"  physics / policy : {s['physics']} / {Path(s['checkpoint']).name} ({s['policy_kind']})",
        f"  sim / wall time  : {s['sim_time_s']:.1f} s / {s['wall_time_s']:.0f} s",
    ]
    if s.get("robots", 1) > 1:
        lines.append(f"  robots           : {s['robots']}, success rate {100 * s['success_rate']:.0f} %")
    if s.get("interrupted"):
        lines.append(f"  interrupted      : {s['interrupted']}")
    return "\n".join(lines)


def main() -> int:
    run_dir = resolve_run_dir(args_cli.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"run directory: {run_dir}")
    ctx = build_plan(run_dir)
    if not ctx["plan"].success:
        log(f"no route: {ctx['plan'].message}")
        return 2
    if args_cli.plan_only:
        return 0
    return walk(ctx, run_dir)


if __name__ == "__main__":
    sys.exit(main())
