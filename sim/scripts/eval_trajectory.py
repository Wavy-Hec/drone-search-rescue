"""Evaluate a trajectory-tracking policy on a fixed reference and plot reference vs flown trajectory.

Trains with::

    isaaclab train --rl_library rsl_rl --task DroneLab-TrackTrajectory-ARL-Robot-1-v0 --num_envs 4096

then (env_isaaclab, project root)::

    python scripts/eval_trajectory.py --checkpoint logs/rsl_rl/drone_lab_track_trajectory/<run>/model_1999.pt \
        --shape figure8 --size 2.0 --speed 1.0 --duration 30

Writes ``runs/<out>/trajectory_eval.csv``, ``eval_summary.json`` and ``trajectory_tracking.png`` (plus
``trajectory.mp4`` with ``--video``: a chase camera with the reference and look-ahead markers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from drone_lab.reference import SHAPES  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=str, required=True, help="TorchScript policy or RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="DroneLab-TrackTrajectory-ARL-Robot-1-v0")
parser.add_argument("--shape", choices=SHAPES, default="figure8")
parser.add_argument("--size", type=float, default=2.0, help="Radius / half-width [m].")
parser.add_argument("--speed", type=float, default=1.0, help="Peak reference speed [m/s].")
parser.add_argument("--z_amp", type=float, default=0.0, help="Vertical wave amplitude [m].")
parser.add_argument("--duration", type=float, default=30.0, help="Evaluation time [s].")
parser.add_argument("--video", dest="record_video", action="store_true", help="Record trajectory.mp4 (chase cam).")
parser.add_argument("--video_fps", type=int, default=30,
                    help="Playback frame rate. One frame per 0.1 s policy step (10 Hz), so the default 30 fps is a "
                         "3x time-lapse; 10 plays in real time.")
parser.add_argument("--out", type=str, default=None, help="Run name under runs/.")
parser.add_argument("--print_config", action="store_true", help="Resolve the config and exit (no simulator).")
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import drone_lab.tasks  # noqa: E402,F401
from drone_lab.video import capture_frame, find_kit_visualizer, open_mp4_writer, playback_speedup  # noqa: E402


def load_policy(checkpoint: str, env, device: str):
    path = Path(checkpoint).expanduser().resolve()
    try:
        policy = torch.jit.load(str(path), map_location=device)
        policy.eval()
        return policy, "torchscript"
    except Exception:
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


def main():
    import gymnasium as gym

    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                            overrides=[*hydra_overrides, "physics=isaacsim_physx"])
    cmd = env_cfg.commands.reference
    cmd.fixed_shape, cmd.fixed_size, cmd.fixed_speed, cmd.fixed_z_amp = (
        args_cli.shape, args_cli.size, args_cli.speed, args_cli.z_amp
    )
    env_cfg.events.reset_base.params.update(position_noise=0.0, attitude_noise=0.0, velocity_noise=0.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args_cli.duration + 5.0
    env_cfg.terminations.too_far.params["max_distance"] = 5.0
    if args_cli.record_video:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        open_window = "kit" in (args_cli.visualizer or [])
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=not open_window, origin_type="asset",
                                                        origin_track_path="robot", eye=(-4.0, -5.0, 3.0),
                                                        lookat=(0.0, 0.0, 0.0))]
        args_cli.enable_cameras = True
        if not open_window:
            args_cli.headless = True
    if args_cli.print_config:
        print(f"[eval_trajectory] {args_cli.task}: reference {cmd.fixed_shape}, size {cmd.fixed_size} m, "
              f"speed {cmd.fixed_speed} m/s, z_amp {cmd.fixed_z_amp} m, {args_cli.duration} s, physics "
              f"{type(env_cfg.sim.physics).__name__}")
        return
    run_dir = PROJECT_ROOT / "runs" / (args_cli.out or f"{datetime.now():%Y%m%d_%H%M%S}_track_{args_cli.shape}")
    run_dir.mkdir(parents=True, exist_ok=True)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        base = env.unwrapped
        policy, kind = load_policy(args_cli.checkpoint, base, base.device)
        term = base.command_manager.get_term("reference")
        robot = base.scene["robot"]
        writer = viz = None
        if args_cli.record_video:
            viz = find_kit_visualizer(base.sim)
            if viz is None:
                print("[eval_trajectory] WARNING: no Kit visualizer is active; no video will be recorded.")
            else:
                writer = open_mp4_writer(run_dir / "trajectory.mp4", args_cli.video_fps)
                print(f"[eval_trajectory] video at {args_cli.video_fps} fps = "
                      f"{playback_speedup(args_cli.video_fps, base.step_dt):g}x real time")
        obs, _ = base.reset()
        rows, terminated_at = [], None
        steps = int(math.ceil(args_cli.duration / base.step_dt))
        with torch.inference_mode():
            for k in range(steps):
                obs, _, terminated, truncated, _ = base.step(policy(obs["policy"]))
                if bool(terminated[0] | truncated[0]):
                    terminated_at = (k + 1) * base.step_dt
                    break
                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                vel = robot.data.root_lin_vel_w.torch[0].cpu().numpy()
                ref = term.ref_pos_w[0].cpu().numpy()
                ref_v = term.ref_vel_w[0].cpu().numpy()
                rows.append([float(term.t[0]), *pos, *ref, *vel, *ref_v])
                if writer is not None:
                    try:
                        # syncs the stage and dispatches the marker callbacks (reference / look-ahead spheres),
                        # which a headless visualizer never gets from env.step()
                        frame = capture_frame(base.sim, viz)
                        if frame is not None and frame.any():
                            writer.append_data(frame)
                    except Exception as exc:  # never lose the evaluation because of the video
                        print(f"[eval_trajectory] WARNING: video capture failed ({exc}); continuing without video.")
                        writer.close()
                        writer = None
        if writer is not None:
            writer.close()
        env.close()

        # stay inside launch_simulation: Kit's fast shutdown on exit ends the process

        data = np.asarray(rows, dtype=np.float64)
        if len(data) == 0:
            print(f"[eval_trajectory] the episode ended immediately (terminated at {terminated_at} s); nothing to plot")
            return
        header = ["t", "x", "y", "z", "ref_x", "ref_y", "ref_z", "vx", "vy", "vz", "ref_vx", "ref_vy", "ref_vz"]
        with open(run_dir / "trajectory_eval.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows([[f"{v:.5f}" for v in r] for r in data])
        err = np.linalg.norm(data[:, 1:4] - data[:, 4:7], axis=1)
        err_xy = np.linalg.norm(data[:, 1:3] - data[:, 4:6], axis=1)
        verr = np.linalg.norm(data[:, 7:10] - data[:, 10:13], axis=1)
        summary = {
            "task": args_cli.task, "checkpoint": args_cli.checkpoint, "policy_kind": kind,
            "shape": args_cli.shape, "size_m": args_cli.size, "speed_mps": args_cli.speed, "z_amp_m": args_cli.z_amp,
            "duration_s": float(data[-1, 0]) if len(data) else 0.0, "terminated_at_s": terminated_at,
            "position_rmse_m": float(np.sqrt(np.mean(err**2))), "position_max_m": float(err.max()),
            "horizontal_rmse_m": float(np.sqrt(np.mean(err_xy**2))), "velocity_rmse_mps": float(np.sqrt(np.mean(verr**2))),
        }
        (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
        from drone_lab.plotting import plot_trajectory_tracking

        png = plot_trajectory_tracking(
            data[:, 0], data[:, 1:4], data[:, 4:7], run_dir / "trajectory_tracking.png",
            title=f"Trajectory tracking: {args_cli.shape}, {args_cli.size:g} m, {args_cli.speed:g} m/s",
        )
        print(json.dumps(summary, indent=2))
        print(f"[eval_trajectory] wrote {png}")


if __name__ == "__main__":
    main()
