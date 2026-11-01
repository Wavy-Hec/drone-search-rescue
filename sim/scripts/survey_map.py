"""Fly a survey with a trained ARL-Robot-1 tracking policy in Isaac Sim, map the area and score the map.

The policy is the one trained on the built-in task ``IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1``
(``isaaclab train`` / ``isaaclab play`` export ``exported/policy.pt``). It flies the survey task
``DroneLab-Survey-ARL-Robot-1-v0`` unchanged: the task feeds it the drone position relative to a carrot that
slides along the survey path (see ``drone_lab/carrot.py``).

During the flight the script logs the drone state, the carrot and the mapping ray hits; afterwards it builds a
2.5D elevation map, a ground-truth map from the terrain mesh, metrics, and plots. Outputs go to
``runs/<name>/`` (see ``drone_lab/survey_io.py``).

Examples (env_isaaclab, from the project root)::

    python scripts/survey_map.py --checkpoint logs/rsl_rl/arl_robot_1_track_position_state_based/<run>/exported/policy.pt
    python scripts/survey_map.py --checkpoint <run>/model_1499.pt --pattern spiral --speed 1.5 --video --out spiral
    python scripts/survey_map.py --checkpoint <policy.pt> --num_envs 3 --camera chase --video --out three_drones
    python scripts/survey_map.py --checkpoint <policy.pt> --print_config      # check the setup, no simulator

Trailing ``key=value`` tokens are forwarded as Hydra overrides to the task config, e.g.
``env.commands.survey_target.turn=square`` or ``env.scene.mapping_scanner.pattern_cfg.resolution=0.2``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from drone_lab.mapping import REDUCERS  # noqa: E402
from drone_lab.paths import PATTERNS  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="TorchScript policy (exported/policy.pt) or RSL-RL checkpoint (model_*.pt).")
parser.add_argument("--task", type=str, default="DroneLab-Survey-ARL-Robot-1-v0", help="Survey task id.")
# survey flags default to None: the task config (or a Hydra override such as
# env.commands.survey_target.speed=1.5) supplies the value unless the flag is given
parser.add_argument("--pattern", choices=PATTERNS, default=None, help="Survey path pattern (task default: lawnmower).")
parser.add_argument("--altitude", type=float, default=None,
                    help="Clearance above the highest terrain point [m] (task default: 4).")
parser.add_argument("--spacing", type=float, default=None, help="Lawnmower line / spiral ring spacing [m] (default 4).")
parser.add_argument("--speed", type=float, default=None, help="Carrot speed along the path [m/s] (default 1).")
parser.add_argument("--max_lead", type=float, default=None,
                    help="Max drone-carrot distance [m] (default 1; the policy was trained on starts within +-1 m).")
parser.add_argument("--waypoints", type=str, default=None,
                    help="CSV/JSON waypoints for --pattern waypoints: x,y (survey altitude) or x,y,z (absolute).")
parser.add_argument("--turn", choices=("round", "square"), default=None, help="Lawnmower turn style (default round).")
parser.add_argument("--direction", choices=("x", "y"), default=None, help="Lawnmower line direction (default x).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of drones (they split the area).")
parser.add_argument("--sensor", choices=("grid", "lidar"), default="grid",
                    help="Mapping rays: 5x5 m grid of vertical rays (0.25 m) or a downward lidar cone.")
parser.add_argument("--map_resolution", type=float, default=0.25, help="Elevation map cell size [m].")
parser.add_argument("--reducer", choices=REDUCERS, default="median", help="Per-cell statistic of the hit heights.")
parser.add_argument("--max_time", type=float, default=None, help="Stop after this much sim time [s].")
parser.add_argument("--physics", dest="physics_backend", choices=("isaacsim_physx", "newton_mjwarp"),
                    default="isaacsim_physx", help="Physics backend (the multirotor asset is PhysX-only).")
parser.add_argument("--allow_unsupported_physics", action="store_true",
                    help="Try --physics newton_mjwarp anyway (expected to fail: Multirotor is PhysX-only).")
parser.add_argument("--video", dest="record_video", action="store_true",
                    help="Record runs/<name>/survey.mp4 from a headless Kit camera (needs no moviepy).")
parser.add_argument("--camera", choices=("overview", "chase"), default="overview",
                    help="Video camera: high oblique view of the area, or a chase camera behind drone 0.")
parser.add_argument("--video_fps", type=int, default=30, help="Video frame rate (policy runs at 10 Hz).")
parser.add_argument("--video_stride", type=int, default=1, help="Capture every N-th policy step.")
parser.add_argument("--video_size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
parser.add_argument("--show_rays", action="store_true", help="Draw the mapping ray hits (debug markers).")
parser.add_argument("--hits_every", type=int, default=5, help="Log subsampled hits every N steps (hits.npy).")
parser.add_argument("--hits_stride", type=int, default=3, help="Keep every N-th ray in hits.npy.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed (thruster randomization).")
parser.add_argument("--out", type=str, default=None, help="Run name under runs/ (or a path).")
parser.add_argument("--no_plots", action="store_true", help="Skip the figures.")
parser.add_argument("--print_config", action="store_true",
                    help="Resolve and print the task config, then exit without launching Isaac Sim.")
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

"""Rest everything follows."""

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import drone_lab.tasks  # noqa: E402,F401  (registers the DroneLab gym ids)
from drone_lab.mapping import ElevationMapper, GridSpec, MeshHeightQuery, ground_truth_map  # noqa: E402
from drone_lab.paths import read_waypoint_rows  # noqa: E402
from drone_lab.survey_io import (  # noqa: E402
    TrajectoryRecorder,
    compute_summary,
    format_summary,
    quat_xyzw_to_euler,
    save_run,
)

UNSUPPORTED_PHYSICS_MSG = (
    "--physics {name} is not supported for this robot: isaaclab_contrib.assets.Multirotor subclasses the PhysX "
    "articulation (isaaclab_contrib/assets/multirotor/multirotor.py:17) and applies the rotor wrench through the "
    "PhysX view (multirotor.py:555), and the built-in task notes that ThrusterCfg has no Newton execution path "
    "(track_position_state_based_env_cfg.py:239). The RayCaster itself does have a Newton backend. Use "
    "--physics isaacsim_physx, or pass --allow_unsupported_physics to try anyway."
)


def resolve_run_dir(out: str | None, prefix: str) -> Path:
    if out and ("/" in out or out.startswith("~")):
        return Path(out).expanduser().resolve()
    return PROJECT_ROOT / "runs" / (out or f"{datetime.now():%Y%m%d_%H%M%S}_{prefix}")


def configure_env(env_cfg):
    """Apply the command-line survey settings to the resolved task config."""
    cmd = env_cfg.commands.survey_target
    for name in ("pattern", "spacing", "altitude", "speed", "turn", "direction"):
        value = getattr(args_cli, name)
        if value is not None:
            setattr(cmd, name, value)
    if args_cli.max_lead is not None:
        cmd.max_lead = args_cli.max_lead
        cmd.slow_lead = min(cmd.slow_lead, 0.7 * args_cli.max_lead)
    if args_cli.waypoints:
        cmd.waypoints = read_waypoint_rows(args_cli.waypoints)
        if args_cli.pattern is None:
            cmd.pattern = "waypoints"
    if cmd.pattern == "waypoints" and not cmd.waypoints:
        sys.exit("[survey_map] --pattern waypoints needs --waypoints <file.csv|file.json>")
    env_cfg.seed = args_cli.seed
    scanner = env_cfg.scene.mapping_scanner
    if args_cli.sensor == "lidar":
        from isaaclab.sensors import patterns

        # downward cone: 16 beams from nadir-ish to 50 deg below the horizon, 5 deg azimuth steps (1152 rays)
        scanner.pattern_cfg = patterns.LidarPatternCfg(
            channels=16, vertical_fov_range=(-88.0, -50.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0
        )
    if args_cli.show_rays:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkersCfg

        scanner.debug_vis = True
        scanner.visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/MappingRays",
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1))
                )
            },
        )
    if args_cli.record_video:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        from drone_lab.tasks.survey.config.arl_robot_1.survey_env_cfg import overview_camera

        open_window = "kit" in (args_cli.visualizer or [])
        w, h = args_cli.video_size
        if args_cli.camera == "chase":
            viz = KitVisualizerCfg(headless=not open_window, window_width=w, window_height=h, origin_type="asset",
                                   origin_track_path="robot", origin_env_index=0, eye=(-6.0, -8.0, 5.0),
                                   lookat=(0.0, 0.0, -1.0))
        else:
            eye, lookat = overview_camera(env_cfg.scene.terrain.terrain_generator)
            viz = KitVisualizerCfg(headless=not open_window, window_width=w, window_height=h, eye=eye, lookat=lookat)
        env_cfg.sim.visualizer_cfgs = [viz]
        # offscreen rendering for the camera; deliberately not the launcher's "video" flag (it needs moviepy)
        args_cli.enable_cameras = True
        # a Kit visualizer config makes the launcher start Kit with a window (kit_launcher.py:504-510);
        # record unattended unless a window was asked for with --viz kit
        if not open_window:
            args_cli.headless = True


def load_policy(checkpoint: str, env, device: str):
    """Return ``(policy, kind)``; ``policy(obs_tensor) -> actions`` for the 17-D policy observation."""
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


def find_kit_visualizer(sim):
    for viz in getattr(sim, "visualizers", []) or []:
        if getattr(viz.cfg, "visualizer_type", None) == "kit" and hasattr(viz, "render_rgb_array"):
            return viz
    return None


def capture_frame(sim, viz) -> np.ndarray | None:
    """Same capture path as isaaclab.envs.utils.video_recorder (VideoRecorder._frame_from_visualizer)."""
    if not sim.is_rendering:
        sim.forward()
        if viz.supports_markers():
            sim.vis_marker_registry.dispatch_callbacks()
    frame = viz.render_rgb_array()
    return None if frame is None else np.ascontiguousarray(frame[:, :, :3])


def main():
    import gymnasium as gym

    from isaaclab_tasks.utils import parse_env_cfg

    if args_cli.physics_backend != "isaacsim_physx":
        if not args_cli.allow_unsupported_physics:
            sys.exit("[survey_map] " + UNSUPPORTED_PHYSICS_MSG.format(name=args_cli.physics_backend))
        args_cli.physics = args_cli.physics_backend  # launcher-level override of every physics config
    else:
        hydra_overrides.append("physics=isaacsim_physx")  # the task's own PhysX preset

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            overrides=hydra_overrides)
    configure_env(env_cfg)
    if args_cli.print_config:
        cmd = env_cfg.commands.survey_target
        print(f"[survey_map] task {args_cli.task}: {env_cfg.scene.num_envs} env(s), physics "
              f"{type(env_cfg.sim.physics).__name__}, dt {env_cfg.sim.dt} x decimation {env_cfg.decimation}")
        print(f"[survey_map] survey: {cmd.pattern}, spacing {cmd.spacing} m, clearance {cmd.altitude} m, "
              f"speed {cmd.speed} m/s, max lead {cmd.max_lead} m, turn {cmd.turn}, waypoints {cmd.waypoints}")
        print(f"[survey_map] mapping sensor: {type(env_cfg.scene.mapping_scanner.pattern_cfg).__name__}, "
              f"alignment {env_cfg.scene.mapping_scanner.ray_alignment}")
        print(f"[survey_map] policy observation terms: "
              f"{[k for k, v in env_cfg.observations.policy.__dict__.items() if hasattr(v, 'func')]}")
        print(f"[survey_map] video: {args_cli.record_video} ({args_cli.camera}), visualizers "
              f"{[type(v).__name__ for v in env_cfg.sim.visualizer_cfgs]}")
        return

    cmd = env_cfg.commands.survey_target
    run_dir = resolve_run_dir(args_cli.out, cmd.pattern)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[survey_map] run directory: {run_dir}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        base = env.unwrapped
        device = base.device
        policy, policy_kind = load_policy(args_cli.checkpoint, base, device)
        print(f"[survey_map] policy: {args_cli.checkpoint} ({policy_kind})")

        term = base.command_manager.get_term("survey_target")
        scanner = base.scene.sensors["mapping_scanner"]
        robot = base.scene["robot"]
        num_envs = base.num_envs
        step_dt = base.step_dt
        print(str(term))

        grid = GridSpec.from_area(term.area, args_cli.map_resolution)
        mapper = ElevationMapper(grid)
        rec = TrajectoryRecorder()
        hits_log: list[np.ndarray] = []
        path_len = float(term.path_length.max())
        max_time = args_cli.max_time or (3.0 * path_len / max(cmd.speed, 1e-3) + 60.0)
        max_steps = int(math.ceil(max_time / step_dt))

        writer, viz = None, None
        if args_cli.record_video:
            import imageio.v2 as imageio

            viz = find_kit_visualizer(base.sim)
            if viz is None:
                print("[survey_map] WARNING: no Kit visualizer is active; no video will be recorded.")
            else:
                writer = imageio.get_writer(str(run_dir / "survey.mp4"), fps=args_cli.video_fps, codec="libx264",
                                            quality=7, macro_block_size=8)

        obs, _ = base.reset()
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        crashed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        end_points = term.path_points[torch.arange(num_envs, device=device), -1]
        t_wall = time.time()
        step = 0
        with torch.inference_mode():
            for step in range(max_steps):
                actions = policy(obs["policy"])
                progress_before = term.progress.clone()
                obs, _, terminated, truncated, _ = base.step(actions)
                # an env that terminated has already been reset inside step(): stop logging it
                newly_crashed = terminated & active
                if torch.any(newly_crashed):
                    for e in torch.nonzero(newly_crashed).flatten().tolist():
                        print(f"[survey_map] drone {e} crashed at t={(step + 1) * step_dt:.1f} s "
                              f"(progress {100 * float(progress_before[e]):.1f} %)")
                    crashed |= newly_crashed
                    active &= ~newly_crashed
                active &= ~truncated
                ids = torch.nonzero(active).flatten()
                if len(ids) > 0:
                    t = (step + 1) * step_dt
                    pos = robot.data.root_pos_w.torch[ids]
                    quat = robot.data.root_quat_w.torch[ids]
                    lin = robot.data.root_lin_vel_w.torch[ids]
                    ang = robot.data.root_ang_vel_b.torch[ids]
                    carrot = term.carrot_w[ids]
                    rows = {
                        "step": np.full(len(ids), step),
                        "time": np.full(len(ids), t),
                        "env": ids.cpu().numpy(),
                        "pos": pos.cpu().numpy(),
                        "quat": quat.cpu().numpy(),
                        "lin": lin.cpu().numpy(),
                        "ang": ang.cpu().numpy(),
                        "carrot": carrot.cpu().numpy(),
                        "s": term.s[ids].cpu().numpy(),
                        "progress": term.progress[ids].cpu().numpy(),
                        "seg": term.segment_index[ids].cpu().numpy(),
                        "obs": obs["policy"][ids, 0:3].cpu().numpy(),
                        "gate": term.gate[ids].cpu().numpy(),
                    }
                    roll, pitch, yaw = quat_xyzw_to_euler(rows["quat"])
                    rec.record(
                        step=rows["step"], time=rows["time"], env=rows["env"],
                        x=rows["pos"][:, 0], y=rows["pos"][:, 1], z=rows["pos"][:, 2],
                        qx=rows["quat"][:, 0], qy=rows["quat"][:, 1], qz=rows["quat"][:, 2], qw=rows["quat"][:, 3],
                        roll=roll, pitch=pitch, yaw=yaw,
                        vx=rows["lin"][:, 0], vy=rows["lin"][:, 1], vz=rows["lin"][:, 2],
                        wx=rows["ang"][:, 0], wy=rows["ang"][:, 1], wz=rows["ang"][:, 2],
                        speed=np.linalg.norm(rows["lin"], axis=1),
                        carrot_x=rows["carrot"][:, 0], carrot_y=rows["carrot"][:, 1], carrot_z=rows["carrot"][:, 2],
                        carrot_dist=np.linalg.norm(rows["pos"] - rows["carrot"], axis=1),
                        path_s=rows["s"], progress=rows["progress"], path_index=rows["seg"],
                        obs_dx=rows["obs"][:, 0], obs_dy=rows["obs"][:, 1], obs_dz=rows["obs"][:, 2],
                        gate=rows["gate"],
                    )
                    hits = scanner.data.ray_hits_w.torch[ids].cpu().numpy().reshape(-1, 3)
                    mapper.add(hits)
                    if args_cli.hits_every > 0 and step % args_cli.hits_every == 0:
                        env_col = np.repeat(rows["env"], scanner.num_rays).astype(np.float32)
                        sub = slice(None, None, max(1, args_cli.hits_stride))
                        h, e_col = hits[sub], env_col[sub]
                        ok = np.all(np.isfinite(h), axis=1)
                        hits_log.append(np.column_stack([np.full(ok.sum(), t), e_col[ok], h[ok]]).astype(np.float32))
                if writer is not None and step % max(1, args_cli.video_stride) == 0:
                    frame = capture_frame(base.sim, viz)
                    if frame is not None and frame.any():
                        writer.append_data(frame)
                # a drone is finished once its carrot reached the end and it is close to the last point
                close = torch.linalg.norm(robot.data.root_pos_w.torch - end_points, dim=-1) < 0.5
                finished |= active & term.done & close
                active &= ~finished
                if step % 200 == 0:
                    prog = ", ".join(f"{100 * p:.0f}%" for p in term.progress.tolist())
                    print(f"[survey_map] t={(step + 1) * step_dt:7.1f} s  progress [{prog}]  "
                          f"hits {mapper.num_points:,}  wall {time.time() - t_wall:.0f} s")
                if not torch.any(active):
                    break
        if writer is not None:
            writer.close()
        print(f"[survey_map] flight finished after {(step + 1) * step_dt:.1f} s sim time "
              f"({time.time() - t_wall:.0f} s wall); finished {finished.tolist()}, crashed {crashed.tolist()}")

        # ground truth from the exact terrain mesh in the stage (fallback: deterministic regeneration)
        from drone_lab.tasks.survey.mdp import terrain_mesh_from_stage

        mesh = terrain_mesh_from_stage(term.cfg.terrain_prim_path)
        gt_source = "usd_stage"
        if mesh is None:
            from drone_lab.terrain import generate_terrain_mesh

            mesh = generate_terrain_mesh(env_cfg.scene.terrain.terrain_generator)
            gt_source = "terrain_generator"
        terrain = MeshHeightQuery(*mesh)
        gt, gt_range = ground_truth_map(*mesh, grid, supersample=3, query=terrain)
        traj = rec.as_dict()
        traj["agl"] = traj["z"] - terrain(np.column_stack([traj["x"], traj["y"]]))
        elevation, count = mapper.finalize(args_cli.reducer)

        import isaaclab

        extra = {
            "source": "isaac_sim",
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).expanduser().resolve()),
            "policy_kind": policy_kind,
            "physics": args_cli.physics_backend,
            "pattern": cmd.pattern,
            "lawnmower_spacing_m": cmd.spacing,
            "altitude_clearance_m": cmd.altitude,
            "terrain_z_min_m": term.terrain_z_min,
            "terrain_z_max_m": term.terrain_z_max,
            "flight_z_m": term.flight_altitude,
            "max_lead_m": cmd.max_lead,
            "sensor": args_cli.sensor,
            "num_rays": int(scanner.num_rays),
            "map_reducer": args_cli.reducer,
            "ground_truth_source": gt_source,
            "step_dt_s": step_dt,
            "sim_time_s": (step + 1) * step_dt,
            "finished": finished.tolist(),
            "isaaclab_version": getattr(isaaclab, "__version__", "unknown"),
            "created": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
        }
        summary = compute_summary(
            traj, term.waypoints, grid=grid, area=term.area, elevation=elevation, count=count, ground_truth=gt,
            ground_truth_range=gt_range, commanded_speed=cmd.speed, crashed=crashed.tolist(), extra=extra,
        )
        hits_arr = np.concatenate(hits_log) if hits_log else np.zeros((0, 5), dtype=np.float32)
        meta = {"source": "isaac_sim", "map_reducer": args_cli.reducer, "sensor": args_cli.sensor,
                "num_rays": int(scanner.num_rays), "ground_truth_source": gt_source}
        save_run(run_dir, trajectory=traj, planned=term.waypoints, grid=grid, area=term.area, elevation=elevation,
                 count=count, ground_truth=gt, ground_truth_range=gt_range, hits=hits_arr, meta=meta,
                 summary=summary)
        print(format_summary(json.loads((run_dir / "summary.json").read_text())))
        env.close()

        # still inside launch_simulation: closing Kit uses fast shutdown, which ends the process
        # (isaacsim.simulation_app SimulationApp.close), so nothing after this block would run
        if not args_cli.no_plots:
            try:
                from drone_lab.plotting import make_all_plots

                for p in make_all_plots(run_dir):
                    print(f"[survey_map] wrote {p}")
            except Exception as exc:  # plotting must never lose the run data
                print(f"[survey_map] plotting failed ({exc}); run: python scripts/plot_survey.py {run_dir}")
        print(f"[survey_map] done: {run_dir}")


if __name__ == "__main__":
    main()
