# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure a velocity-tracking policy on an Isaac Lab 3.0 manager-based task (episode metrics, no video).

Builds the environment exactly like ``isaaclab play --rl_library rsl_rl`` (``isaaclab_rl/entrypoints/backends/
play_rsl_rl.py``): preset/Hydra tokens from the command line, ``resolve_task_config(..., play_mode=True)``,
``launch_simulation``, ``gym.make``, ``RslRlVecEnvWrapper``, ``create_rsl_rl_runner`` + ``runner.load`` +
``get_inference_policy`` (deterministic mean action). Instead of the open-ended playback loop it runs every env for
``--episodes`` complete episodes and reports, over exactly ``num_envs * episodes`` episodes:

* mean episode length [s] and the fraction of episodes ending in ``base_contact`` vs ``time_out``;
* episodes in which the trunk tilted more than 60 deg (projected gravity z > -0.5), as a second fall indicator;
* mean velocity-tracking error over all alive steps: ``|cmd_xy - v_xy|`` [m/s] (body frame, as the
  ``track_lin_vel_xy_exp`` reward) and ``|cmd_wz - w_z|`` [rad/s], plus mean commanded / actual planar speed;
* mean episode reward per term in the Isaac Lab log convention (episode sum / max_episode_length_s, i.e. the
  ``Episode_Reward/<term>`` value TensorBoard shows) and the total;
* mean trunk height above the scanned terrain and the fraction of alive steps with ground contact on the
  ``*_foot`` and ``*_calf`` bodies (the feet-contact finding in PORT_NOTES.md).

Policies: ``--policy checkpoint`` (rsl-rl 5 checkpoint via ``--checkpoint``), ``zero`` (all-zero actions = hold the
default joint pose) or ``random`` (uniform [-1, 1]).

Run from the project root (headless, Kit + Isaac Sim PhysX), e.g. the converted 2.x policy::

    python scripts/eval_policy.py --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 128 --episodes 2 \\
        --policy checkpoint --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --seed 42 \\
        --out logs/eval/rough_s42.json

The summary line starts with ``[EVAL] summary:``; ``--out`` adds the per-episode records. The simulation is
deterministic for a fixed seed and ``num_envs``, so variants are compared with the same seeds, one change at a time
(e.g. the Isaac Lab 3.0 execution defaults:
``env.sim.use_newton_actuators=true env.sim.physics.enable_external_forces_every_iteration=true``).
``--native_refresh measure`` (with ``env.sim.use_newton_actuators=true``) measures how stale the joint state is that
the native actuator path reads on Isaac Sim PhysX.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=2, help="Complete episodes per env that are evaluated.")
    parser.add_argument("--policy", choices=("checkpoint", "zero", "random"), default="checkpoint")
    parser.add_argument("--checkpoint", default=None, help="rsl-rl 5 checkpoint (for --policy checkpoint).")
    parser.add_argument("--seed", type=int, default=42, help="Env/agent seed (play uses the agent cfg seed, 42).")
    parser.add_argument("--train_env_cfg", action="store_true", help="Skip play_mode() (training env cfg).")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None, help="JSON file for the summary and per-episode records.")
    parser.add_argument("--max_steps", type=int, default=None, help="Safety cap on env steps.")
    parser.add_argument(
        "--native_refresh",
        choices=("off", "measure", "fix"),
        default="off",
        help="Native actuator path only (env.sim.use_newton_actuators=true): patch "
        "PhysxActuatorControl.compute_native_actuators in this process. 'measure' records how stale the joint-state "
        "buffers bound to the native actuators are on each physics substep (then restores them, so the dynamics are "
        "unchanged); 'fix' refreshes them before every actuator step, like the non-identity-ordering branch.",
    )
    add_launcher_args(parser)
    return setup_preset_cli(parser, argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args_cli, hydra_args = _parse_args(argv)
    sys.argv = [sys.argv[0]] + hydra_args  # preset tokens + Hydra overrides, read by resolve_task_config

    import gymnasium as gym
    import torch

    from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    import quadruped_lab.tasks  # noqa: F401  (registers the project tasks)

    env_cfg, agent_cfg = resolve_task_config(
        args_cli.task, "rsl_rl_cfg_entry_point", play_mode=not args_cli.train_env_cfg
    )
    if args_cli.native_refresh != "off" and not env_cfg.sim.use_newton_actuators:
        raise SystemExit("--native_refresh patches the native actuator path: add env.sim.use_newton_actuators=true")
    t_start = time.time()
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        agent_cfg.seed = args_cli.seed
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        torch.manual_seed(args_cli.seed)

        physics = env_cfg.sim.physics
        resolved = {
            "task": args_cli.task,
            "play_mode": not args_cli.train_env_cfg,
            "num_envs": env_cfg.scene.num_envs,
            "seed": env_cfg.seed,
            "physics_cfg": type(physics).__name__,
            "enable_external_forces_every_iteration": getattr(physics, "enable_external_forces_every_iteration", None),
            "use_newton_actuators": getattr(env_cfg.sim, "use_newton_actuators", None),
            "robot_usd": getattr(env_cfg.scene.robot.spawn, "usd_path", None),
            "init_pos": list(env_cfg.scene.robot.init_state.pos),
            "episode_length_s": env_cfg.episode_length_s,
            "sim_dt": env_cfg.sim.dt,
            "decimation": env_cfg.decimation,
            "policy_corruption": getattr(env_cfg.observations.policy, "enable_corruption", None),
            "hydra_args": hydra_args,
        }
        print("[EVAL] resolved config: " + json.dumps(resolved, default=str))

        substep = [0]
        stale = {"n": [0] * 8, "pos_max": [0.0] * 8, "pos_sum": [0.0] * 8, "vel_max": [0.0] * 8, "vel_sum": [0.0] * 8}
        if args_cli.native_refresh != "off":
            import warp as wp

            from isaaclab_physx.assets.articulation.actuator_control import PhysxActuatorControl

            _orig_cna = PhysxActuatorControl.compute_native_actuators
            mode = args_cli.native_refresh
            first = [True]

            def _patched_cna(self, collection, dt):
                if not self._native_actuator_path_active or self._actuator_runtime is None:
                    return _orig_cna(self, collection, dt)
                data = self._articulation._data
                if not data.has_joint_ordering:
                    pos_t, vel_t = wp.to_torch(data._joint_pos.data), wp.to_torch(data._joint_vel.data)
                    old_pos, old_vel = pos_t.clone(), vel_t.clone()
                    data._refresh_joint_pos()
                    data._refresh_joint_vel()
                    new_pos, new_vel = wp.to_torch(data._joint_pos.data), wp.to_torch(data._joint_vel.data)
                    if first[0]:
                        wrapper = self._actuator_runtime.wrapper
                        print(
                            "[EVAL] native refresh: buffer ptrs before/after refresh "
                            f"pos {pos_t.data_ptr():#x}/{new_pos.data_ptr():#x}, "
                            f"wrapper joint_q {wrapper.joint_q.ptr:#x}; "
                            f"vel {vel_t.data_ptr():#x}/{new_vel.data_ptr():#x}, "
                            f"wrapper joint_qd {wrapper.joint_qd.ptr:#x}"
                        )
                        first[0] = False
                    k = min(substep[0], 7)
                    dp, dv = (new_pos - old_pos).abs(), (new_vel - old_vel).abs()
                    stale["n"][k] += 1
                    stale["pos_max"][k] = max(stale["pos_max"][k], float(dp.max()))
                    stale["pos_sum"][k] += float(dp.mean())
                    stale["vel_max"][k] = max(stale["vel_max"][k], float(dv.max()))
                    stale["vel_sum"][k] += float(dv.mean())
                    substep[0] += 1
                    if mode == "measure":  # put the stale values back: the actuator sees what it would have seen
                        new_pos.copy_(old_pos)
                        new_vel.copy_(old_vel)
                return _orig_cna(self, collection, dt)

            PhysxActuatorControl.compute_native_actuators = _patched_cna

        env = gym.make(args_cli.task, cfg=env_cfg)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        uenv = env.unwrapped
        device = uenv.device
        num_envs = uenv.num_envs
        dt = uenv.step_dt
        max_len = int(uenv.max_episode_length)

        if args_cli.policy == "checkpoint":
            if not args_cli.checkpoint:
                raise SystemExit("--policy checkpoint needs --checkpoint")
            runner = create_rsl_rl_runner(env, agent_cfg)
            runner.load(args_cli.checkpoint)
            inference = runner.get_inference_policy(device=device)
            print(f"[EVAL] loaded {args_cli.checkpoint} (iteration {runner.current_learning_iteration})")

            def act(obs):
                return inference(obs)

        elif args_cli.policy == "zero":
            zeros = torch.zeros(num_envs, env.num_actions, device=device)

            def act(obs):
                return zeros

        else:
            gen = torch.Generator(device=device).manual_seed(args_cli.seed)

            def act(obs):
                return 2.0 * torch.rand(num_envs, env.num_actions, device=device, generator=gen) - 1.0

        robot = uenv.scene["robot"]
        rew_mgr = uenv.reward_manager
        term_mgr = uenv.termination_manager
        term_names = list(rew_mgr.active_terms)
        contact = uenv.scene.sensors.get("contact_forces") if hasattr(uenv.scene.sensors, "get") else None
        scanner = uenv.scene.sensors.get("height_scanner") if hasattr(uenv.scene.sensors, "get") else None
        foot_ids = calf_ids = None
        if contact is not None:
            foot_ids, foot_names = contact.find_sensors(".*_foot")
            calf_ids, calf_names = contact.find_sensors(".*_calf")
            print(f"[EVAL] contact sensor bodies: {contact.body_names}")
            print(f"[EVAL] foot ids {foot_ids} {foot_names}; calf ids {calf_ids} {calf_names}")
        print(f"[EVAL] joint names: {robot.joint_names}")
        print(f"[EVAL] reward terms: {term_names}; termination terms: {list(term_mgr.active_terms)}")
        print(f"[EVAL] step_dt {dt}, max_episode_length {max_len} steps, scanner {'yes' if scanner else 'no'}")

        # attribute base_contact terminations to bodies: wrap the term function, which env.step calls before the reset
        bc_cfg = term_mgr.get_term_cfg("base_contact")
        bc_ids = list(bc_cfg.params["sensor_cfg"].body_ids)
        bc_names = [contact.body_names[i] for i in bc_ids]
        bc_trunk_col = [k for k, n in enumerate(bc_names) if n == "trunk"]
        bc_thigh_cols = [k for k, n in enumerate(bc_names) if n.endswith("_thigh")]
        bc_hit = torch.zeros(num_envs, len(bc_ids), dtype=torch.bool, device=device)
        _orig_bc = bc_cfg.func

        def _bc_attributed(env_, **params):
            out = _orig_bc(env_, **params)
            hist = contact.data.net_normal_forces_w_history.torch[:, :, bc_ids]
            bc_hit.copy_(torch.linalg.norm(hist, dim=-1).max(dim=1)[0] > params["threshold"])
            return out

        bc_cfg.func = _bc_attributed
        print(f"[EVAL] base_contact bodies: {bc_names}")

        # per-env running sums of the current episode
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)
        ep_rew = z(num_envs, len(term_names))
        ep_lin_err, ep_ang_err, ep_cmd_speed, ep_speed, ep_height, ep_err_steps = (z(num_envs) for _ in range(6))
        ep_foot, ep_calf = z(num_envs), z(num_envs)
        ep_jpos = z(num_envs, robot.num_joints)
        ep_act = z(num_envs)
        ep_hmin = torch.full((num_envs,), float("inf"), device=device)
        default_jpos = robot.data.default_joint_pos.torch[0].clone()
        groups = {g: [i for i, n in enumerate(robot.joint_names) if f"_{g}_" in n] for g in ("hip", "thigh", "calf")}
        ep_tilt = torch.zeros(num_envs, dtype=torch.bool, device=device)
        completed = torch.zeros(num_envs, dtype=torch.long, device=device)
        records: list[dict] = []
        target = args_cli.episodes
        max_steps = args_cli.max_steps or (target * max_len + 10)

        obs = env.get_observations()
        cmd_term = uenv.command_manager
        step = 0
        with torch.inference_mode():
            while step < max_steps and int(completed.min()) < target:
                cmd_prev = cmd_term.get_command("base_velocity").clone()
                actions = act(obs)
                substep[0] = 0
                obs, _, dones, _ = env.step(actions)
                step += 1
                done = dones.bool()
                # rewards of this step (computed before the reset inside env.step)
                ep_rew += rew_mgr._step_reward * dt
                ep_len += 1
                # state-based metrics: only for envs that were not reset in this step (post-reset state otherwise)
                alive = ~done
                v_b = robot.data.root_lin_vel_b.torch
                w_b = robot.data.root_ang_vel_b.torch
                g_b = robot.data.projected_gravity_b.torch
                lin_err = torch.linalg.norm(cmd_prev[:, :2] - v_b[:, :2], dim=1)
                ang_err = torch.abs(cmd_prev[:, 2] - w_b[:, 2])
                a = alive.float()
                ep_lin_err += lin_err * a
                ep_ang_err += ang_err * a
                ep_cmd_speed += torch.linalg.norm(cmd_prev[:, :2], dim=1) * a
                ep_speed += torch.linalg.norm(v_b[:, :2], dim=1) * a
                ep_err_steps += a
                ep_jpos += robot.data.joint_pos.torch * a.unsqueeze(1)
                ep_act += actions.abs().mean(dim=1) * a
                ep_tilt |= alive & (g_b[:, 2] > -0.5)
                if scanner is not None:
                    hits = scanner.data.ray_hits_w.torch[..., 2]
                    hits = torch.where(torch.isfinite(hits), hits, torch.nan)
                    ground = torch.nanmedian(hits, dim=1).values
                    height = robot.data.root_pos_w.torch[:, 2] - ground
                else:
                    height = robot.data.root_pos_w.torch[:, 2]
                ep_height += torch.nan_to_num(height) * a
                ep_hmin = torch.where(
                    alive, torch.minimum(ep_hmin, torch.nan_to_num(height, nan=float("inf"))), ep_hmin
                )
                if contact is not None:
                    f = torch.linalg.norm(contact.data.net_normal_forces_w.torch, dim=-1)
                    ep_foot += (f[:, foot_ids] > 1.0).float().mean(dim=1) * a
                    ep_calf += (f[:, calf_ids] > 1.0).float().mean(dim=1) * a

                if done.any():
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    base_contact = term_mgr.get_term("base_contact")[ids]
                    time_out = term_mgr.get_term("time_out")[ids]
                    steps = ep_err_steps[ids].clamp(min=1)
                    rows = torch.stack(
                        [
                            ep_len[ids].float(),
                            base_contact.float(),
                            time_out.float(),
                            ep_tilt[ids].float(),
                            ep_lin_err[ids] / steps,
                            ep_ang_err[ids] / steps,
                            ep_cmd_speed[ids] / steps,
                            ep_speed[ids] / steps,
                            ep_height[ids] / steps,
                            ep_foot[ids] / steps,
                            ep_calf[ids] / steps,
                            ep_err_steps[ids],
                        ],
                        dim=1,
                    ).cpu()
                    rew_rows = ep_rew[ids].cpu()
                    jpos_rows = (ep_jpos[ids] / steps.unsqueeze(1)).cpu()
                    act_rows = (ep_act[ids] / steps).cpu()
                    hmin_rows = ep_hmin[ids].cpu()
                    bc_trunk_rows = bc_hit[ids][:, bc_trunk_col].any(dim=1).cpu()
                    bc_thigh_rows = bc_hit[ids][:, bc_thigh_cols].any(dim=1).cpu()
                    for k, env_id in enumerate(ids.tolist()):
                        if int(completed[env_id]) < target:
                            r = rows[k].tolist()
                            records.append(
                                {
                                    "env": env_id,
                                    "episode": int(completed[env_id]),
                                    "length_steps": int(r[0]),
                                    "length_s": r[0] * dt,
                                    "base_contact": bool(r[1]),
                                    "time_out": bool(r[2]),
                                    "tilt_gt_60deg": bool(r[3]),
                                    "lin_err": r[4],
                                    "ang_err": r[5],
                                    "cmd_speed": r[6],
                                    "speed": r[7],
                                    "base_height": r[8],
                                    "foot_contact_frac": r[9],
                                    "calf_contact_frac": r[10],
                                    "alive_steps": r[11],
                                    "reward_sum": dict(zip(term_names, rew_rows[k].tolist(), strict=False)),
                                    "joint_pos_mean_deg": {
                                        g: math.degrees(float(jpos_rows[k, idx].mean())) for g, idx in groups.items()
                                    },
                                    "action_abs_mean": float(act_rows[k]),
                                    "min_base_height": float(hmin_rows[k]),
                                    "bc_trunk": bool(bc_trunk_rows[k]) and bool(r[1]),
                                    "bc_thigh": bool(bc_thigh_rows[k]) and bool(r[1]),
                                }
                            )
                    completed[ids] += 1
                    for buf in (
                        ep_lin_err,
                        ep_ang_err,
                        ep_cmd_speed,
                        ep_speed,
                        ep_height,
                        ep_err_steps,
                        ep_foot,
                        ep_calf,
                    ):
                        buf[ids] = 0.0
                    ep_rew[ids] = 0.0
                    ep_jpos[ids] = 0.0
                    ep_act[ids] = 0.0
                    ep_hmin[ids] = float("inf")
                    ep_len[ids] = 0
                    ep_tilt[ids] = False
                if step % 250 == 0:
                    print(
                        f"[EVAL] step {step}: episodes recorded {len(records)}, "
                        f"min completed per env {int(completed.min())}",
                        flush=True,
                    )

        summary = summarize(records, term_names, uenv.max_episode_length_s, dt)
        summary.update(
            label=args_cli.label,
            policy=args_cli.policy,
            checkpoint=args_cli.checkpoint,
            steps=step,
            wall_s=round(time.time() - t_start, 1),
            complete=int(completed.min()) >= target,
            resolved=resolved,
            default_joint_pos_deg={g: math.degrees(float(default_jpos[idx].mean())) for g, idx in groups.items()},
            native_refresh=args_cli.native_refresh,
        )
        if args_cli.native_refresh != "off":
            summary["native_state_staleness_per_substep"] = {
                f"substep{k + 1}": {
                    "calls": stale["n"][k],
                    "joint_pos_absdiff_mean_rad": stale["pos_sum"][k] / stale["n"][k],
                    "joint_pos_absdiff_max_rad": stale["pos_max"][k],
                    "joint_vel_absdiff_mean_radps": stale["vel_sum"][k] / stale["n"][k],
                    "joint_vel_absdiff_max_radps": stale["vel_max"][k],
                }
                for k in range(8)
                if stale["n"][k]
            }
        print("[EVAL] summary: " + json.dumps({k: v for k, v in summary.items() if k != "resolved"}, default=str))
        if args_cli.out:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
            with open(args_cli.out, "w") as f:
                json.dump({"summary": summary, "episodes": records}, f, indent=1, default=str)
            print(f"[EVAL] wrote {args_cli.out}")
    return 0


def summarize(records: list[dict], term_names: list[str], max_episode_length_s: float, dt: float) -> dict:
    n = len(records)
    if n == 0:
        return {"episodes": 0}

    def mean(key):
        return sum(r[key] for r in records) / n

    steps = sum(r["alive_steps"] for r in records) or 1.0

    def step_mean(key):  # mean over all alive steps (episodes weighted by their length)
        return sum(r[key] * r["alive_steps"] for r in records) / steps

    lengths = [r["length_s"] for r in records]
    mu = sum(lengths) / n
    rewards = {t: sum(r["reward_sum"][t] for r in records) / n / max_episode_length_s for t in term_names}
    return {
        "episodes": n,
        "mean_episode_length_s": mu,
        "std_episode_length_s": math.sqrt(sum((x - mu) ** 2 for x in lengths) / n),
        "min_episode_length_s": min(lengths),
        "frac_base_contact": mean("base_contact"),
        "frac_time_out": sum(1 for r in records if r["time_out"] and not r["base_contact"]) / n,
        "frac_tilt_gt_60deg": mean("tilt_gt_60deg"),
        "frac_short_lt_5s": sum(1 for x in lengths if x < 5.0) / n,
        "lin_vel_err_mps": step_mean("lin_err"),
        "ang_vel_err_radps": step_mean("ang_err"),
        "cmd_speed_mps": step_mean("cmd_speed"),
        "speed_mps": step_mean("speed"),
        "base_height_m": step_mean("base_height"),
        "foot_contact_frac": step_mean("foot_contact_frac"),
        "calf_contact_frac": step_mean("calf_contact_frac"),
        "joint_pos_mean_deg": {
            g: sum(r["joint_pos_mean_deg"][g] * r["alive_steps"] for r in records) / steps
            for g in ("hip", "thigh", "calf")
        },
        "action_abs_mean": step_mean("action_abs_mean"),
        "frac_bc_trunk": sum(1 for r in records if r.get("bc_trunk")) / n,
        "frac_bc_thigh": sum(1 for r in records if r.get("bc_thigh")) / n,
        "median_min_base_height_m": sorted(r.get("min_base_height", float("nan")) for r in records)[n // 2],
        "episode_reward_per_term": rewards,
        "episode_reward_total": sum(rewards.values()),
    }


if __name__ == "__main__":
    sys.exit(main())
