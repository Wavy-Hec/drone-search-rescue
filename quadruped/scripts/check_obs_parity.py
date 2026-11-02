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

Checked:

* the policy observation terms (order, function, size, noise, clip, scale) against the checkpoint input;
* every value the 2.x configs set explicitly (2.3.2 base class bodies and ``__post_init__``, then Hector's
  ``__post_init__``, with ``ROUGH_TERRAINS_CFG`` from the 2.3.2 ``terrains/config/rough.py``): actions, commands,
  rewards, terminations, events, curriculum, height scanner, contact sensor, terrain importer and generator, timing,
  number of envs and the PhysX scene setting; term parameters a config leaves out are compared through the term
  functions' default values (2.3.2 source vs. 3.0 signature);
* the robot's default pose and DC-motor parameters (``custom_quad.py``);
* how Isaac Sim PhysX executes the actions: the TGS ``enable_external_forces_every_iteration`` flag against the 2.3.2
  ``PhysxCfg`` default, and the actuator path (Isaac Lab 2.x had no native actuator path).

Not checked here: defaults of the config classes themselves that neither side sets (for example
``ContactSensorCfg.force_threshold``), rendering-only settings (sky light, terrain visual material) and the function
bodies; PORT_NOTES.md lists the manual diffs for those.

Usage (from the project root, conda env ``env_isaaclab`` active)::

    PYTHONPATH=src python scripts/check_obs_parity.py [--markdown]
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import inspect
import math
import operator
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quadruped_lab.tasks.locomotion.config.custom_quad.rough_env_cfg import CustomQuadRoughEnvCfg

HOME = Path.home()
DEFAULT_OLD_LAB = HOME / "IsaacLab"
DEFAULT_OLD_TASK = HOME / "quadruped_rough_terrain" / "custom_quadruped"
OLD_BASE_REL = "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
OLD_ROUGH_TERRAINS_REL = "source/isaaclab/isaaclab/terrains/config/rough.py"
OLD_SIM_CFG_REL = "source/isaaclab/isaaclab/sim/simulation_cfg.py"
OLD_MDP_RELS = (  # where the 2.x velocity mdp functions are defined (velocity package first)
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py",
    "source/isaaclab/isaaclab/envs/mdp/events.py",
    "source/isaaclab/isaaclab/envs/mdp/rewards.py",
    "source/isaaclab/isaaclab/envs/mdp/terminations.py",
    "source/isaaclab/isaaclab/envs/mdp/curriculums.py",
)
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
        terms.append(
            {
                "name": node.targets[0].id,
                "func": ast.unparse(kw["func"]).split(".")[-1],
                "noise": None
                if noise is None
                else {k.arg: _literal(k.value) for k in noise.keywords} | {"type": ast.unparse(noise.func)},
                "clip": _literal(kw.get("clip")),
                "scale": _literal(kw.get("scale")),
                "params": ast.unparse(kw["params"]) if "params" in kw else None,
            }
        )
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
        terms.append(
            {
                "name": name,
                "func": getattr(func, "__name__", str(func).split(":")[-1].split(".")[-1]),
                "noise": None
                if noise is None
                else {"n_min": noise.n_min, "n_max": noise.n_max, "type": type(noise).__name__, "op": noise.operation},
                "clip": tuple(term.clip) if term.clip is not None else None,
                "scale": term.scale,
                "params": {k: (v.name if hasattr(v, "name") else v) for k, v in term.params.items()},
            }
        )
    return terms


def usd_dof_joints(usd_path: str) -> list[str]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    return [
        p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Static evaluation of the 2.x configuration (no 2.x import: the 2.3.2 packages are not installed in this env)
# ---------------------------------------------------------------------------------------------------------------------


class Ref(str):
    """Reference to a function, class or symbol by its last dotted name (``mdp.feet_air_time`` -> ``feet_air_time``)."""


@dataclasses.dataclass
class Call:
    """A constructor call found in the source: callee name (import aliases resolved) and its arguments."""

    name: str
    args: list = dataclasses.field(default_factory=list)
    kwargs: dict = dataclasses.field(default_factory=dict)


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
# 2.x -> 3.0 renames of config classes that do not change behaviour
CLASS_RENAMES = {"AdditiveUniformNoiseCfg": "UniformNoiseCfg", "EventCfg": "EventsCfg"}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := _dotted(node.value)) is not None:
        return f"{base}.{node.attr}"
    return None


class StaticConfig:
    """Evaluate configclass bodies and ``__post_init__`` assignments of a module from its AST.

    Supports what the velocity configs use: literals, arithmetic, ``math``, constructor calls, module-local config
    classes (expanded into their fields), ``self.<path>`` reads, attribute/item assignments and ``if ... is (not)
    None`` tests. Anything else is recorded in :attr:`skipped`.
    """

    def __init__(self, source: str, symbols: dict | None = None):
        self.tree = ast.parse(source)
        self.aliases = {
            a.asname: a.name
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Import | ast.ImportFrom)
            for a in n.names
            if a.asname
        }
        self.classes = {n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)}
        self.symbols = symbols or {}
        self.state: Call | None = None
        self.skipped: list[str] = []

    def module_value(self, name: str):
        """Value of a module-level assignment ``name = ...``."""
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return self.expr(node.value)
        raise KeyError(name)

    def expand_class(self, cls: ast.ClassDef, scope: dict | None = None, kwargs: dict | None = None) -> Call:
        """Fields of a module-local config class (class-body assignments, nested classes resolved), plus kwargs."""
        scope = dict(scope or {})
        fields: dict = {}
        for node in cls.body:
            if isinstance(node, ast.ClassDef):
                scope[node.name] = node
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                fields[node.target.id] = self.expr(node.value, scope)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    fields[target.id] = self.expr(node.value, scope)
        fields.update(kwargs or {})
        return Call(cls.name, [], fields)

    def expr(self, node: ast.AST, scope: dict | None = None):  # noqa: C901 - a small expression interpreter
        scope = scope or {}
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple | ast.List):
            items = [self.expr(e, scope) for e in node.elts]
            return tuple(items) if isinstance(node, ast.Tuple) else items
        if isinstance(node, ast.Dict):
            return {self.expr(k, scope): self.expr(v, scope) for k, v in zip(node.keys, node.values, strict=False)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = self.expr(node.operand, scope)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](self.expr(node.left, scope), self.expr(node.right, scope))
        if isinstance(node, ast.JoinedStr):
            return "".join(
                v.value if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}" for v in node.values
            )
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Is | ast.IsNot):
            left, right = self.expr(node.left, scope), self.expr(node.comparators[0], scope)
            return (left is right) if isinstance(node.ops[0], ast.Is) else (left is not right)
        if isinstance(node, ast.Subscript):
            return self._item(self.expr(node.value, scope), self.expr(node.slice, scope))
        if isinstance(node, ast.Call):
            return self._call(node, scope)
        dotted = _dotted(node)
        if dotted is not None:
            head, *rest = dotted.split(".")
            if head == "self":
                return self._get(self.state, rest)
            if head == "math" and rest:
                return getattr(math, rest[0])
            if not rest and dotted in self.symbols:
                return copy.deepcopy(self.symbols[dotted])
            if not rest and dotted == "MISSING":
                return Ref("MISSING")
            return Ref(dotted.split(".")[-1])
        self.skipped.append(f"expression {ast.unparse(node)}")
        return Ref(ast.unparse(node))

    def _call(self, node: ast.Call, scope: dict):
        callee = _dotted(node.func) or ast.unparse(node.func)
        args = [self.expr(a, scope) for a in node.args]
        kwargs = {k.arg: self.expr(k.value, scope) for k in node.keywords if k.arg is not None}
        if callee.startswith("math."):
            return getattr(math, callee.split(".", 1)[1])(*args)
        if callee == "getattr":
            return self._attr(args[0], args[1], args[2] if len(args) > 2 else None)
        name = callee.split(".")[-1]
        name = self.aliases.get(name, name)
        local = scope.get(name) or self.classes.get(name)
        if isinstance(local, ast.ClassDef) and not args:
            return self.expand_class(local, scope, kwargs)
        return Call(name, args, kwargs)

    @staticmethod
    def _attr(obj, name, default=None):
        if isinstance(obj, Call):
            return obj.kwargs.get(name, default)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    @staticmethod
    def _item(obj, key):
        return obj.kwargs[key] if isinstance(obj, Call) else obj[key]

    def _get(self, obj, path: list[str]):
        for name in path:
            obj = self._attr(obj, name)
        return obj

    def _assign(self, target: ast.AST, value) -> None:
        steps = []
        node = target
        while isinstance(node, ast.Attribute | ast.Subscript):
            if isinstance(node, ast.Attribute):
                steps.append(("attr", node.attr))
            else:
                steps.append(("item", self.expr(node.slice)))
            node = node.value
        if not (isinstance(node, ast.Name) and node.id == "self"):
            self.skipped.append(f"assignment to {ast.unparse(target)}")
            return
        steps.reverse()
        obj = self.state
        for _, key in steps[:-1]:
            container = obj.kwargs if isinstance(obj, Call) else obj
            if container.get(key) is None or isinstance(container.get(key), Ref):
                container[key] = Call("<inherited>")  # e.g. ``sim`` / ``sim.physx`` come from the parent class
            obj = container[key]
        (obj.kwargs if isinstance(obj, Call) else obj)[steps[-1][1]] = value

    def run(self, body: list[ast.stmt]) -> None:
        """Execute the assignments of a ``__post_init__`` body on :attr:`state`."""
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                value = self.expr(stmt.value)
                for target in stmt.targets:
                    self._assign(target, value)
            elif isinstance(stmt, ast.If):
                self.run(stmt.body if self.expr(stmt.test) else stmt.orelse)
            elif isinstance(stmt, ast.Expr):
                continue  # docstrings, super().__post_init__()
            else:
                self.skipped.append(f"statement {ast.unparse(stmt).splitlines()[0]}")

    def post_init(self, class_name: str) -> list[ast.stmt]:
        cls = self.classes[class_name]
        return next(n.body for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")


def old_env_cfg(base_src: str, rough_src: str, rough_terrains_src: str) -> tuple[Call, list[str]]:
    """The 2.x rough env cfg as the sources define it: base class body, base ``__post_init__``, Hector's overrides."""
    terrains = StaticConfig(rough_terrains_src).module_value("ROUGH_TERRAINS_CFG")
    base = StaticConfig(base_src, symbols={"ROUGH_TERRAINS_CFG": terrains})
    base.state = base.expand_class(base.classes["LocomotionVelocityRoughEnvCfg"])
    base.run(base.post_init("LocomotionVelocityRoughEnvCfg"))
    task = StaticConfig(rough_src)
    task.state = base.state
    task.run(task.post_init("CustomQuadRoughEnvCfg"))
    return base.state, base.skipped + task.skipped


def old_function_defaults(sources: list[str]) -> dict[str, dict]:
    """Keyword defaults of every function / term class (``__call__``) defined in the given 2.x mdp sources."""
    out: dict[str, dict] = {}
    for src in sources:
        evaluator = StaticConfig(src)
        for node in evaluator.tree.body:
            func = node
            if isinstance(node, ast.ClassDef):
                func = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"), None)
            if not isinstance(func, ast.FunctionDef) or node.name in out:
                continue
            args = func.args.posonlyargs + func.args.args
            with_default = [a.arg for a in args[len(args) - len(func.args.defaults) :]]
            defaults = dict(zip(with_default, func.args.defaults, strict=True))
            kw_defaults = zip(func.args.kwonlyargs, func.args.kw_defaults, strict=True)
            defaults.update({a.arg: d for a, d in kw_defaults if d is not None})
            names = [a.arg for a in args + func.args.kwonlyargs if a.arg not in ("self", "env", "env_ids")]
            out[node.name] = {n: (evaluator.expr(defaults[n]) if n in defaults else REQUIRED) for n in names}
    return out


REQUIRED = Ref("<required>")
ABSENT = Ref("<absent>")


def new_function_defaults(func) -> dict:
    """Keyword defaults of a 3.0 term function or term class ``__call__``."""
    if isinstance(func, str):
        from isaaclab.utils.string import string_to_callable

        func = string_to_callable(func)
    signature = inspect.signature(func.__call__ if inspect.isclass(func) else func)
    return {
        n: (REQUIRED if p.default is inspect.Parameter.empty else p.default)
        for n, p in signature.parameters.items()
        if n not in ("self", "env", "env_ids") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


class CfgComparer:
    """Compare a statically evaluated 2.x value with the corresponding value of the instantiated 3.0 config."""

    TERM_CLASSES = ("EventTermCfg", "RewardTermCfg", "TerminationTermCfg", "CurriculumTermCfg")

    def __init__(self, old_defaults: dict[str, dict], allowed: dict[str, str], excluded: tuple[str, ...] = ()):
        self.old_defaults = old_defaults
        self.allowed = allowed  # path -> reason of a known, documented difference
        self.excluded = excluded  # paths compared elsewhere or irrelevant (rendering)
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.compared: list[str] = []  # leaf paths that were compared

    @property
    def checked(self) -> int:
        return len(self.compared)

    def differ(self, path: str, old, new) -> None:
        if path in self.allowed:
            self.notes.append(f"{path}: 2.x {old!r} -> 3.0 {new!r} (expected: {self.allowed[path]})")
        else:
            self.problems.append(f"{path}: 2.x {old!r} != 3.0 {new!r}")

    def compare(self, path: str, old, new) -> None:  # noqa: C901
        if (isinstance(old, Ref) and old == "MISSING") or path in self.excluded:
            return
        if isinstance(old, Call):
            self._compare_call(path, old, new)
        elif isinstance(old, dict):
            if not isinstance(new, dict):
                return self.differ(path, old, new)
            if set(old) != set(new):
                return self.differ(path + " (keys)", sorted(old), sorted(new))
            for key in old:
                self.compare(f"{path}[{key!r}]", old[key], new[key])
        elif isinstance(old, Ref):
            self.compared.append(path)
            name = getattr(new, "__name__", None) or (
                new.split(":")[-1].split(".")[-1] if isinstance(new, str) else new
            )
            if name != old:
                self.differ(path, str(old), name)
        elif isinstance(old, list | tuple):
            if not isinstance(new, list | tuple) or len(old) != len(new):
                return self.differ(path, old, new)
            for i, (o, n) in enumerate(zip(old, new, strict=False)):
                self.compare(f"{path}[{i}]", o, n)
        else:
            self.compared.append(path)
            same = (
                math.isclose(old, new, rel_tol=1e-9, abs_tol=1e-12)
                if isinstance(old, float) and isinstance(new, int | float)
                else old == new
            )
            if not same:
                self.differ(path, old, new)

    def _compare_call(self, path: str, old: Call, new) -> None:
        if new is None:
            return self.differ(path, old.name, None)
        expected = CLASS_RENAMES.get(old.name, old.name)
        if old.name != "<inherited>" and type(new).__name__ != expected:
            self.differ(path + " (type)", old.name, type(new).__name__)
        kwargs = dict(old.kwargs)
        if old.args and dataclasses.is_dataclass(new):  # positional arguments map to the leading dataclass fields
            kwargs.update(zip([f.name for f in dataclasses.fields(new)], old.args, strict=False))
        is_term = expected in self.TERM_CLASSES and "func" in kwargs
        for key, value in kwargs.items():
            if is_term and key == "params":
                continue
            if not hasattr(new, key):
                if value is None:  # e.g. the 2.x patch's depth_camera = None: absent in 3.0 means the same
                    continue
                self.differ(f"{path}.{key}", value, "<no such field in 3.0>")
                continue
            self.compare(f"{path}.{key}", value, getattr(new, key))
        if is_term:
            self._compare_params(path, str(kwargs["func"]), kwargs.get("params", {}), new)

    def _compare_params(self, path: str, func_name: str, old_params: dict, new) -> None:
        old_defaults = self.old_defaults.get(func_name)
        new_defaults = new_function_defaults(new.func)
        if old_defaults is None:
            self.problems.append(f"{path}: 2.x function {func_name} not found in the 2.3.2 mdp sources")
            old_defaults = {}
        new_params = new.params or {}
        for key in dict.fromkeys([*old_params, *new_params, *old_defaults, *new_defaults]):
            old_value = old_params.get(key, old_defaults.get(key, ABSENT))
            new_value = new_params.get(key, new_defaults.get(key, ABSENT))
            where = f"{path}.params[{key!r}]"
            if old_value is ABSENT:
                self.notes.append(f"{where}: new parameter in 3.0 = {new_value!r}")
            elif new_value is ABSENT:
                self.problems.append(f"{where}: parameter removed in 3.0 (2.x {old_value!r})")
            elif not (old_value is REQUIRED and new_value is REQUIRED):
                self.compare(where, old_value, new_value)


# 2.x paths that legitimately differ in 3.0 (each documented in PORT_NOTES.md)
ALLOWED_DIFFERENCES = {
    "scene.contact_forces.prim_path": (
        "3.0 base uses Robot/[^/]*; the PhysX sensor applies the last segment as a name filter to every "
        "contact-reporting body below Robot (isaaclab_physx contact_sensor.py:367-391), and all 17 bodies are direct "
        "children of the robot root anyway"
    ),
}
# 2.x paths that live elsewhere in 3.0
MOVED_PATHS = {"sim.physx": ("sim", "physics", "default")}  # PhysX scene settings moved into the physics preset
# compared elsewhere (observations, robot) or rendering only
EXCLUDED_ROOTS = ("observations", "scene.robot", "scene.sky_light", "scene.terrain.visual_material")


def compare_with_2x(old_state: Call, cfg, old_defaults: dict[str, dict]) -> CfgComparer:
    """Compare every 2.x explicit setting (outside :data:`EXCLUDED_ROOTS`) with the instantiated 3.0 cfg."""
    comparer = CfgComparer(old_defaults, ALLOWED_DIFFERENCES, EXCLUDED_ROOTS)

    def walk(path: str, old, new) -> None:
        is_container = isinstance(old, Call) and old.name in (
            "<inherited>",
            "LocomotionVelocityRoughEnvCfg",
            "MySceneCfg",
        )
        if not is_container:
            return comparer.compare(path, old, new)
        for key, value in old.kwargs.items():
            child = f"{path}.{key}" if path else key
            if child in EXCLUDED_ROOTS:
                continue
            if child in MOVED_PATHS:
                target = cfg
                for name in MOVED_PATHS[child]:
                    target = getattr(target, name)
                comparer.notes.append(f"{child} -> {'.'.join(MOVED_PATHS[child])} (moved in 3.0)")
                comparer.compare(child, value, target)
            elif not hasattr(new, key):
                if value is not None:  # the 2.x patch attributes set to None are simply absent in 3.0
                    comparer.differ(child, value, "<no such field in 3.0>")
            else:
                walk(child, value, getattr(new, key))

    walk("", old_state, cfg)
    return comparer


def old_class_fields(source: str, class_name: str) -> dict[str, object]:
    """Annotated fields of a 2.x config class with their literal defaults (``None`` where not a literal)."""
    cls = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef) and n.name == class_name)
    fields: dict[str, object] = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                fields[node.target.id] = None if node.value is None else ast.literal_eval(node.value)
            except ValueError:
                fields[node.target.id] = None
    return fields


def execution_checks(old: dict, cfg: CustomQuadRoughEnvCfg) -> dict[str, tuple[object, object]]:
    """How Isaac Sim PhysX executes the policy's actions: ``{check: (2.x value, 3.0 default)}``.

    * TGS external forces: the 2.3.2 ``PhysxCfg`` default vs the task's Isaac Sim PhysX default.
    * Native actuator path: Isaac Lab 2.x ran every actuator on Isaac Lab's actuator path (its ``SimulationCfg`` has
      no ``use_newton_actuators`` field); the task's per-backend preset must keep Isaac Sim PhysX on that path.
    """
    physx_2x = old_class_fields(old["sim_cfg_src"], "PhysxCfg")
    sim_2x = old_class_fields(old["sim_cfg_src"], "SimulationCfg")
    native = cfg.sim.use_newton_actuators
    native = getattr(native, "default", native)  # unresolved cfg: per-backend preset whose default is PhysX
    return {
        "TGS external forces every iteration": (
            physx_2x["enable_external_forces_every_iteration"],
            cfg.sim.physics.default.enable_external_forces_every_iteration,
        ),
        "native actuator path": (sim_2x.get("use_newton_actuators", False), native),
    }


def load_old_sources(old_lab: Path, old_task: Path) -> dict:
    """Read the 2.x sources (2.3.2 files from git HEAD, Hector's task files) and evaluate the rough env cfg."""

    def old_file(rel: str) -> str:
        return subprocess.run(
            ["git", "-C", str(old_lab), "show", f"HEAD:{rel}"], check=True, capture_output=True, text=True
        ).stdout

    out = {
        "base_src": old_file(OLD_BASE_REL),
        "sim_cfg_src": old_file(OLD_SIM_CFG_REL),
        "rough_src": (old_task / "rough_env_cfg.py").read_text(),
        "custom_quad_src": (old_task / "custom_quad.py").read_text(),
    }
    out["state"], out["skipped"] = old_env_cfg(out["base_src"], out["rough_src"], old_file(OLD_ROUGH_TERRAINS_REL))
    out["defaults"] = old_function_defaults([old_file(rel) for rel in OLD_MDP_RELS])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old-lab", type=Path, default=DEFAULT_OLD_LAB, help="Isaac Lab 2.3.2 git clone (read only).")
    parser.add_argument("--old-task", type=Path, default=DEFAULT_OLD_TASK, help="Hector's 2.x task folder.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="2.x checkpoint to compare with.")
    parser.add_argument("--markdown", action="store_true", help="Print the layout as a Markdown table.")
    parser.add_argument("--verbose", action="store_true", help="List every compared 2.x setting.")
    args = parser.parse_args(argv)

    import torch

    from isaaclab.sensors.ray_caster.patterns import grid_pattern

    from quadruped_lab.tasks.locomotion.config.custom_quad.rough_env_cfg import CustomQuadRoughEnvCfg

    old = load_old_sources(args.old_lab, args.old_task)
    old_terms, removed = old_policy_terms(old["base_src"], old["rough_src"])
    old_robot = old_robot_cfg(old["custom_quad_src"])
    old_state, skipped, old_defaults = old["state"], old["skipped"], old["defaults"]

    cfg = CustomQuadRoughEnvCfg()
    # scene.robot is the CustomQuadRobotCfg preset; its default is the PhysX robot (robot.usd) the checkpoint used
    robot = cfg.scene.robot.default
    new_terms = new_policy_terms(cfg)
    joints = usd_dof_joints(robot.spawn.usd_path)
    rays = grid_pattern(cfg.scene.height_scanner.pattern_cfg, "cpu")[0].shape[0]
    dims = FIXED_DIMS | {"joint_pos_rel": len(joints), "joint_vel_rel": len(joints), "last_action": len(joints)}
    dims["height_scan"] = rays
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt_in = int(ckpt["model_state_dict"]["actor.0.weight"].shape[1])
    ckpt_out = int(ckpt["model_state_dict"]["actor.6.weight"].shape[0])

    problems: list[str] = [f"2.x source not evaluated: {item}" for item in skipped]
    if [t["name"] for t in old_terms] != [t["name"] for t in new_terms]:
        problems.append(f"term order differs: {[t['name'] for t in old_terms]} vs {[t['name'] for t in new_terms]}")
    rows, offset = [], 0
    for old_term, new_term in zip(old_terms, new_terms, strict=False):
        dim = dims[new_term["func"]]
        old_noise = None if old_term["noise"] is None else (old_term["noise"]["n_min"], old_term["noise"]["n_max"])
        new_noise = None if new_term["noise"] is None else (new_term["noise"]["n_min"], new_term["noise"]["n_max"])
        same = (
            old_term["func"] == new_term["func"]
            and old_noise == new_noise
            and old_term["clip"] == new_term["clip"]
            and old_term["scale"] == new_term["scale"]
            and (new_term["noise"] is None or new_term["noise"]["op"] == "add")
        )
        if not same:
            problems.append(f"term '{new_term['name']}' differs: 2.x={old_term} 3.0={new_term}")
        row = (new_term["name"], new_term["func"], dim, f"{offset}-{offset + dim - 1}", new_noise, new_term["clip"])
        rows.append((*row, same))
        offset += dim
    if offset != ckpt_in:
        problems.append(f"observation size {offset} != checkpoint input {ckpt_in}")
    if len(joints) != ckpt_out:
        problems.append(f"{len(joints)} actuated joints != checkpoint output {ckpt_out}")

    # interface summary; the 2.x values are read from the evaluated 2.x sources, not typed in here
    old_act = old_state.kwargs["actions"].kwargs["joint_pos"]
    old_sim = old_state.kwargs["sim"].kwargs
    act = cfg.actions.joint_pos
    checks = {
        "action term (class, joints, scale, default offset)": (
            (
                old_act.name,
                old_act.kwargs["joint_names"],
                old_act.kwargs["scale"],
                old_act.kwargs["use_default_offset"],
            ),
            (type(act).__name__, list(act.joint_names), act.scale, act.use_default_offset),
        ),
        "sim dt / decimation / render interval": (
            (old_sim["dt"], old_state.kwargs["decimation"], old_sim["render_interval"]),
            (cfg.sim.dt, cfg.decimation, cfg.sim.render_interval),
        ),
        "episode length [s]": (old_state.kwargs["episode_length_s"], cfg.episode_length_s),
        "init root pos": (old_robot["init_pos"], tuple(robot.init_state.pos)),
        "default joint pos (regex -> rad)": (old_robot["joint_pos"], dict(robot.init_state.joint_pos)),
    }
    act_cfg = robot.actuators["base_legs"]
    o = old_robot["actuator"]
    checks["DC motor (effort, saturation, velocity, kp, kd, friction)"] = (
        (o["effort_limit"], o["saturation_effort"], o["velocity_limit"], o["stiffness"], o["damping"], o["friction"]),
        (
            act_cfg.actuator_effort_limit,
            act_cfg.saturation_effort,
            act_cfg.actuator_velocity_limit,
            act_cfg.stiffness,
            act_cfg.damping,
            act_cfg.friction,
        ),
    )
    checks.update(execution_checks(old, cfg))
    for name, (expected, actual) in checks.items():
        if expected != actual:
            problems.append(f"{name}: 2.x={expected} 3.0={actual}")
    # port decision, not a 2.x value: Isaac Sim PhysX is the default backend (the 3.0 velocity base defaults to Newton)
    if type(cfg.sim.physics.default).__name__ != "PhysxCfg":
        problems.append(f"physics default is {type(cfg.sim.physics.default).__name__}, expected PhysxCfg")

    # every other explicit 2.x setting: commands, rewards, terminations, events, curriculum, sensors, terrain, ...
    mdp = compare_with_2x(old_state, cfg, old_defaults)
    problems += mdp.problems

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
    print(f"OK  physics default: {type(cfg.sim.physics.default).__name__}")
    if args.verbose:
        print("\ncompared 2.x settings:\n  " + "\n  ".join(mdp.compared))
    print(f"\n2.x explicit settings compared with the 3.0 cfg: {mdp.checked} values; known differences / notes:")
    for note in mdp.notes:
        print(f"  - {note}")
    if problems:
        print("\nPARITY PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print(
        "\nparity OK: same observation terms, order, dims, noise, clipping and scaling; same action, timing, robot "
        "defaults and actuator; every other explicit 2.x setting equal or a listed known difference"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
