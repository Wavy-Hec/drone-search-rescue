# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the joint order of an Isaac Lab 3.0 run against the order the 2.x height-scan checkpoint was trained with.

The converted policy maps the i-th ``joint_pos`` / ``joint_vel`` / ``actions`` entry to the i-th articulation joint
(``joint_ordering=None``, backend-native order). It only works if Isaac Sim 6.2 PhysX orders the 12 joints like
Isaac Sim 5.1 PhysX did. Isaac Lab 3.0 documents that order: links breadth-first, the children of each link in authored
joint-prim order (``docs/source/how-to/transfer_policies_between_physx_and_newton.rst:353-356``). The expected order is
derived from ``robot.usd`` with that rule (``FL/FR/RL/RR`` hips, then thighs, then calves).

The run's order is read from its log. Two INFO records carry it; ``--info`` makes sure they reach the console:

* the action term, ``Resolved joint names for the action term JointPositionAction: [...]``
  (``isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:70-73``), which is the order the policy actually drives;
* the articulation table after ``Simulation parameters for joints in <prim path>:``
  (``isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:4317``; Newton:
  ``isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py:3718``).

Usage (GPU phase, from the project root)::

    mkdir -p logs
    isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 200 --viz none --info \\
        physics=isaacsim_physx 2>&1 | tee logs/zero_agent_rough.log
    python scripts/check_joint_order.py logs/zero_agent_rough.log

Exit code 0: every order found in the log matches; 1: a mismatch; 2: no joint order in the log.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import deque
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_USD = PROJECT / "src" / "quadruped_lab" / "assets" / "data" / "robots" / "custom_quad" / "robot.usd"

_ACTION_RE = re.compile(r"Resolved joint names for the action term (\w+): (\[[^\]]*\])")
_TABLE_HEADER = "Simulation parameters for joints in "
_TABLE_ROW_RE = re.compile(r"^\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|")


def expected_joint_order(usd_path: str | Path) -> list[str]:
    """Isaac Sim PhysX DOF order of the asset: breadth-first over links, children in authored joint-prim order."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    edges = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not body0 or not body1:
            continue
        has_dof = prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        edges.append((prim.GetName(), body0[0].name, body1[0].name, has_dof))
    roots = [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise ValueError(f"expected one articulation root in {usd_path}, found {roots}")
    order, queue = [], deque(roots)
    while queue:
        link = queue.popleft()
        for name, body0, body1, has_dof in edges:
            if body0 == link:
                if has_dof:
                    order.append(name)
                queue.append(body1)
    return order


def parse_log(text: str) -> dict[str, list[str]]:
    """Joint orders found in a run log, keyed by source (``action term <name>`` / ``articulation <prim path>``)."""
    found: dict[str, list[str]] = {}
    for match in _ACTION_RE.finditer(text):
        found.setdefault(f"action term {match.group(1)}", list(ast.literal_eval(match.group(2))))
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _TABLE_HEADER not in line:
            continue
        prim_path = line.split(_TABLE_HEADER, 1)[1].rstrip().rstrip(":")
        rows: list[tuple[int, str]] = []
        for row in lines[i + 1 :]:
            stripped = row.strip()
            if not stripped.startswith(("|", "+")):
                break
            if m := _TABLE_ROW_RE.match(row):
                rows.append((int(m.group(1)), m.group(2)))
        if rows:
            found.setdefault(f"articulation {prim_path}", [name for _, name in sorted(rows)])
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path, help="Console log of an isaaclab zero_agent/play/train run (with --info).")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD, help="Robot USD the expected order is derived from.")
    parser.add_argument("--expected", default=None, help="Comma-separated expected joint order (overrides --usd).")
    args = parser.parse_args(argv)

    expected = args.expected.split(",") if args.expected else expected_joint_order(args.usd)
    found = parse_log(args.log.read_text(errors="replace"))
    print(f"expected ({len(expected)}): {expected}")
    if not found:
        print(
            "no joint order found in the log: run with --info and let the run reach environment creation",
            file=sys.stderr,
        )
        return 2
    ok = True
    for source, order in found.items():
        same = order == expected
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {source} ({len(order)}): {order}")
    if not ok:
        print("joint order differs: the converted 2.x policy would drive the wrong joints", file=sys.stderr)
        return 1
    print("joint order OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
