# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that the ported 3.0 rough task matches the 2.x interface of ``height_scanner_model.pt`` (CPU only, no Kit).

The 2.x side is read statically (Python AST) from the read-only sources it was built from:

* the Isaac Lab 2.3.2 velocity base config (``git show HEAD:...`` in the 2.3.2 clone, i.e. without the local
  depth-camera patch; the patch only appends a ``depth_scan`` term that the rough task sets to None),
* Hector's 2.x task files (``rough_env_cfg.py``, ``custom_quad.py``).

The 3.0 side is the instantiated :class:`CustomQuadRoughEnvCfg`. Dimensions come from the robot USD (revolute joints,
via ``pxr``), the ray-caster grid pattern (evaluated on the CPU) and the checkpoint's first actor layer.

Usage (from the project root, conda env ``env_isaaclab`` active)::

    PYTHONPATH=src python scripts/check_obs_parity.py [--markdown]
"""

from __future__ import annotations

import argparse
import ast
import math
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
DEFAULT_OLD_LAB = HOME / "IsaacLab"
DEFAULT_OLD_TASK = HOME / "quadruped_rough_terrain" / "custom_quadruped"
OLD_BASE_REL = "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT / "checkpoints_2x" / "height_scanner_model.pt"

# output size of the stock velocity observation terms that do not depend on the robot or a sensor
FIXED_DIMS = {"base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3, "generated_commands": 3}


def _literal(node: ast.AST | None, names: dict | None = None):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except ValueError:
        return eval(compile(ast.Expression(node), "<ast>", "eval"), {"__builtins__": {}}, names or {})  # noqa: S307


def old_policy_terms(base_src: str, rough_src: str) -> list[dict]:
    """Ordered policy terms of the 2.x rough task: 2.3.2 base PolicyCfg minus the terms the task sets to None."""
    tree = ast.parse(base_src)
    policy = next(
        n
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "ObservationsCfg"
        for n in c.body
        if isinstance(n, ast.ClassDef) and n.name == "PolicyCfg"
    )
    terms = []
    for node in policy.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        kw = {k.arg: k.value for k in node.value.keywords}
        noise = kw.get("noise")
        terms.append({
            "name": node.targets[0].id,
            "func": ast.unparse(kw["func"]).split(".")[-1],
            "noise": None
            if noise is None
            else {k.arg: _literal(k.value) for k in noise.keywords} | {"type": ast.unparse(noise.func)},
            "clip": _literal(kw.get("clip")),
            "scale": _literal(kw.get("scale")),
            "params": ast.unparse(kw["params"]) if "params" in kw else None,
        })
    removed = {
        t.attr
        for n in ast.walk(ast.parse(rough_src))
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
        and ast.unparse(t).startswith("self.observations.policy.")
        and isinstance(n.value, ast.Constant)
        and n.value.value is None
    }
    return [t for t in terms if t["name"] not in removed], sorted(removed)


def old_robot_cfg(custom_quad_src: str) -> dict:
    """Default joint positions and DC-motor parameters of the 2.x robot config."""
    tree = ast.parse(custom_quad_src)
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("DCMotorCfg"):
            out["actuator"] = {k.arg: _literal(k.value) for k in node.keywords if k.arg != "joint_names_expr"}
        if isinstance(node, ast.keyword) and node.arg == "joint_pos":
            out["joint_pos"] = _literal(node.value, {"pi": math.pi})
        if isinstance(node, ast.keyword) and node.arg == "pos" and "init_pos" not in out:
            out["init_pos"] = _literal(node.value)
    return out


def new_policy_terms(cfg) -> list[dict]:
    from isaaclab.managers import ObservationTermCfg

    group = cfg.observations.policy
    terms = []
    for name in group.__dataclass_fields__:
        term = getattr(group, name)
        if not isinstance(term, ObservationTermCfg):
            continue
        func = term.func
        noise = term.noise
        terms.append({
            "name": name,
            "func": getattr(func, "__name__", str(func).split(":")[-1].split(".")[-1]),
            "noise": None
            if noise is None
            else {"n_min": noise.n_min, "n_max": noise.n_max, "type": type(noise).__name__, "op": noise.operation},
            "clip": tuple(term.clip) if term.clip is not None else None,
            "scale": term.scale,
            "params": {k: (v.name if hasattr(v, "name") else v) for k, v in term.params.items()},
        })
    return terms


def usd_dof_joints(usd_path: str) -> list[str]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    return [
        p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old-lab", type=Path, default=DEFAULT_OLD_LAB, help="Isaac Lab 2.3.2 git clone (read only).")
    parser.add_argument("--old-task", type=Path, default=DEFAULT_OLD_TASK, help="Hector's 2.x task folder.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="2.x checkpoint to compare with.")
    parser.add_argument("--markdown", action="store_true", help="Print the layout as a Markdown table.")
    args = parser.parse_args(argv)

    import torch

    from isaaclab.sensors.ray_caster.patterns import grid_pattern

    from quadruped_lab.tasks.locomotion.config.custom_quad.rough_env_cfg import CustomQuadRoughEnvCfg

    base_src = subprocess.run(
        ["git", "-C", str(args.old_lab), "show", f"HEAD:{OLD_BASE_REL}"], check=True, capture_output=True, text=True
    ).stdout
    old_terms, removed = old_policy_terms(base_src, (args.old_task / "rough_env_cfg.py").read_text())
    old_robot = old_robot_cfg((args.old_task / "custom_quad.py").read_text())

    cfg = CustomQuadRoughEnvCfg()
    new_terms = new_policy_terms(cfg)
    joints = usd_dof_joints(cfg.scene.robot.spawn.usd_path)
    rays = grid_pattern(cfg.scene.height_scanner.pattern_cfg, "cpu")[0].shape[0]
    dims = FIXED_DIMS | {"joint_pos_rel": len(joints), "joint_vel_rel": len(joints), "last_action": len(joints)}
    dims["height_scan"] = rays
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt_in = int(ckpt["model_state_dict"]["actor.0.weight"].shape[1])
    ckpt_out = int(ckpt["model_state_dict"]["actor.6.weight"].shape[0])

    problems: list[str] = []
    if [t["name"] for t in old_terms] != [t["name"] for t in new_terms]:
        problems.append(f"term order differs: {[t['name'] for t in old_terms]} vs {[t['name'] for t in new_terms]}")
    rows, offset = [], 0
    for old, new in zip(old_terms, new_terms, strict=False):
        dim = dims[new["func"]]
        old_noise = None if old["noise"] is None else (old["noise"]["n_min"], old["noise"]["n_max"])
        new_noise = None if new["noise"] is None else (new["noise"]["n_min"], new["noise"]["n_max"])
        same = (
            old["func"] == new["func"]
            and old_noise == new_noise
            and old["clip"] == new["clip"]
            and old["scale"] == new["scale"]
            and (new["noise"] is None or new["noise"]["op"] == "add")
        )
        if not same:
            problems.append(f"term '{new['name']}' differs: 2.x={old} 3.0={new}")
        rows.append((new["name"], new["func"], dim, f"{offset}-{offset + dim - 1}", new_noise, new["clip"], same))
        offset += dim
    if offset != ckpt_in:
        problems.append(f"observation size {offset} != checkpoint input {ckpt_in}")
    if len(joints) != ckpt_out:
        problems.append(f"{len(joints)} actuated joints != checkpoint output {ckpt_out}")

    # non-observation interface: action, timing, robot defaults, actuator, events
    act = cfg.actions.joint_pos
    checks = {
        "action term": (("JointPositionAction", [".*"], 0.25, True), (
            type(act).__name__.replace("Cfg", ""), list(act.joint_names), act.scale, act.use_default_offset)),
        "sim dt / decimation / control dt": ((0.005, 4, 0.02), (cfg.sim.dt, cfg.decimation, cfg.sim.dt * cfg.decimation)),
        "episode length [s]": (20.0, cfg.episode_length_s),
        "init root pos": (old_robot["init_pos"], tuple(cfg.scene.robot.init_state.pos)),
        "default joint pos (regex -> rad)": (old_robot["joint_pos"], dict(cfg.scene.robot.init_state.joint_pos)),
    }
    act_cfg = cfg.scene.robot.actuators["base_legs"]
    o = old_robot["actuator"]
    checks["DC motor (effort, saturation, velocity, kp, kd, friction)"] = (
        (o["effort_limit"], o["saturation_effort"], o["velocity_limit"], o["stiffness"], o["damping"], o["friction"]),
        (act_cfg.actuator_effort_limit, act_cfg.saturation_effort, act_cfg.actuator_velocity_limit,
         act_cfg.stiffness, act_cfg.damping, act_cfg.friction),
    )
    mass = cfg.events.add_base_mass.params
    checks["add_base_mass (range, op, distribution, body)"] = (
        ((-1.0, 3.0), "add", "uniform", "trunk"),
        (tuple(mass["mass_distribution_params"]), mass["operation"], mass["distribution"], mass["asset_cfg"].body_names),
    )
    checks["physics default"] = ("PhysxCfg", type(cfg.sim.physics.default).__name__)
    for name, (expected, actual) in checks.items():
        if expected != actual:
            problems.append(f"{name}: 2.x={expected} 3.0={actual}")

    print(f"2.x terms removed by the rough task: {removed}")
    print(f"USD actuated joints ({len(joints)}, USD prim order): {joints}")
    if args.markdown:
        print("\n| # | term | function | dims | index range | noise U(min, max) | clip | 2.x == 3.0 |")
        print("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(rows):
            print(f"| {i} | `{r[0]}` | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {'yes' if r[6] else 'NO'} |")
        print(f"| | **total** | | **{offset}** | 0-{offset - 1} | | | checkpoint input {ckpt_in} |")
    else:
        for i, r in enumerate(rows):
            print(f"{i:2d} {r[0]:18s} {r[1]:18s} dim={r[2]:3d} [{r[3]:>7s}] noise={r[4]} clip={r[5]} same={r[6]}")
        print(f"total obs = {offset}  (checkpoint actor input = {ckpt_in}, actions = {ckpt_out})")
    for name, (expected, actual) in checks.items():
        print(f"{'OK ' if expected == actual else 'DIFF'} {name}: {actual}")
    if problems:
        print("\nPARITY PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nparity OK: same terms, order, dims, noise, clipping, scaling, action, timing, defaults and actuator")
    return 0


if __name__ == "__main__":
    sys.exit(main())
