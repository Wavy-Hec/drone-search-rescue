"""Natural poses for the Isaac People characters: one retargeted animation frame, baked into a static mesh.

The characters of ``Isaac/People/Characters`` are skinned (UsdSkel) Reallusion rigs in their rest T-pose. The clips of
``Isaac/People/Animations`` drive another rig (``Characters/Biped_Setup.usd``: Y up, +Z forward, joint values in
metres) through ``omni.anim.graph`` at runtime, which Isaac Lab's apps do not load. This module poses a character on
the CPU (pxr + numpy, no simulator):

1. evaluate the clip at one frame (plus layered joints, e.g. the head of ``LookAround`` over an idle body);
2. retarget: per mapped joint, take the world-space rotation from the Biped's T-pose to that frame and apply it to
   the character's own T-pose (``controlRig:retargetTransforms``); bone lengths stay the character's;
3. edit: :class:`Turn` rotates a joint (and everything below it) about a body axis, :class:`Ground` searches the
   angle at which a limb touches the ground plane of the body (a heel, a hand, a knee);
4. skin the meshes with their own weights (linear blend skinning, as authored) and lay the body down
   (:attr:`PoseSpec.lay`);
5. write a static copy of the character: the same meshes, UVs and materials with the posed points and normals,
   no skeleton (:func:`write_posed_usd`).

Static meshes look the same in every renderer and sensor (RGB, depth, segmentation), cost nothing per frame and
have exact CPU geometry. ``people_poses.json`` (next to this module) holds the box and the under-side height map of
every character in every pose; :mod:`drone_lab.sar.scenario` places people with it, without the assets. Regenerate
it after changing :data:`POSES`::

    python -m drone_lab.sar.people --write-data [--preview runs/people_poses]

Frame of a posed asset: metres, Z up, origin at the bottom centre of the posed box (the lowest point is at z = 0).
Standing and sitting people face -Y like the source characters; lying people have their head towards +Y.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

PEOPLE_STYLES = ("tpose", "natural")
"""``tpose``: the characters as shipped (rest T-pose); ``natural``: the posed victims and responders below."""

ANIMATIONS_DIR = "Isaac/People/Animations"
"""Clips below the asset root (``<name>.skelanim.usd``)."""

BIPED_RIG = "Isaac/People/Characters/Biped_Setup.usd"
"""Rig the clips were authored for (its rest pose is a T-pose, like the characters')."""

DATA_FILE = Path(__file__).with_name("people_poses.json")
DATA_FORMAT = "drone_lab.sar.people_poses/v1"
CELL = 0.05
"""Cell size of the under-side height maps [m]."""

Axis = tuple[float, float, float]
X: Axis = (1.0, 0.0, 0.0)
"""Body axes of a standing character: +X is its left, -Y its front, +Z up."""
Y: Axis = (0.0, 1.0, 0.0)
Z: Axis = (0.0, 0.0, 1.0)

SEAT = ("Hip", "Pelvis", "L_ThighTwist01", "R_ThighTwist01", "L_ThighTwist02", "R_ThighTwist02")
"""Joints whose vertices carry a seated body (buttocks and the back of the thighs)."""

# Biped (source) joint -> character (Reallusion) joint, by leaf name. Twist and helper joints of the character
# follow their parents. The Biped's spine has four joints (Spine1-3, Chest), the character's three.
JOINT_MAP: dict[str, str] = {
    "Pelvis": "Hip",
    "Spine1": "Waist",
    "Spine2": "Spine01",
    "Chest": "Spine02",
    "Neck1": "NeckTwist01",
    "Neck2": "NeckTwist02",
    "Head": "Head",
}
for _s in ("L", "R"):
    JOINT_MAP.update(
        {
            f"{_s}_UpLeg": f"{_s}_Thigh",
            f"{_s}_LoLeg": f"{_s}_Calf",
            f"{_s}_Ankle": f"{_s}_Foot",
            f"{_s}_Ball": f"{_s}_ToeBase",
            f"{_s}_Clavicle": f"{_s}_Clavicle",
            f"{_s}_UpArm": f"{_s}_Upperarm",
            f"{_s}_LoArm": f"{_s}_Forearm",
            f"{_s}_Wrist": f"{_s}_Hand",
        }
    )
    for _k in (1, 2, 3):
        JOINT_MAP[f"{_s}_ThumbFinger{_k}"] = f"{_s}_Thumb{_k}"
        # the Biped's Finger1 is the metacarpal: its Finger2-4 are the character's phalanges 1-3
        for _src, _dst in (("Index", "Index"), ("Middle", "Mid"), ("Ring", "Ring"), ("Pinky", "Pinky")):
            JOINT_MAP[f"{_s}_{_src}Finger{_k + 1}"] = f"{_s}_{_dst}{_k}"


@dataclass(frozen=True)
class Turn:
    """Rotate ``joint`` and everything below it by ``degrees`` about a body axis through the joint."""

    joint: str
    axis: Axis
    degrees: float


@dataclass(frozen=True)
class Ground:
    """Rotate ``joint`` about ``axis`` (within ``limits`` degrees) until the lowest point of the part below
    ``measure`` is ``offset`` metres above the ground plane of the body, after the body is laid down. The ground
    plane is the lowest point of the ``reference`` joints' vertices, or of everything ``joint`` does not move.
    Puts a heel, a hand or a knee on the ground whatever the character's proportions."""

    joint: str
    axis: Axis
    limits: tuple[float, float]
    measure: str
    reference: tuple[str, ...] = ()
    offset: float = 0.0


@dataclass(frozen=True)
class Layer:
    """Joints taken from another clip (e.g. the head turn of ``LookAround`` over an idle body)."""

    clip: str
    frame: float
    joints: tuple[str, ...]


@dataclass(frozen=True)
class PoseSpec:
    """One natural pose: a clip frame, optional layers and edits, and the rotation that lays the body down."""

    key: str
    role: str
    """``victim`` or ``responder``."""
    clip: str
    frame: float
    layers: tuple[Layer, ...] = ()
    edits: tuple[Turn | Ground, ...] = ()
    lay: tuple[tuple[Axis, float], ...] = ()
    """Rotations (axis, degrees) applied in order to the posed body; empty for upright poses."""
    description: str = ""

    def digest(self) -> str:
        """Short hash of the definition (``people_poses.json`` and the baked files are keyed by it)."""
        blob = json.dumps({"type": "PoseSpec", **asdict(self)}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


POSES: dict[str, PoseSpec] = {
    p.key: p
    for p in (
        PoseSpec(
            "lying_back",
            "victim",
            "stand_idle_loop",
            0,
            edits=(
                Turn("Head", Z, 28.0),
                Ground("NeckTwist01", X, (-35.0, 15.0), "Head"),
                Ground("R_Thigh", X, (-10.0, 20.0), "R_Foot"),
                Turn("L_Thigh", X, -18.0),
                Ground("L_Calf", X, (0.0, 60.0), "L_Foot"),
                Turn("L_Upperarm", Y, -10.0),
                Turn("R_Upperarm", Y, 14.0),
                Ground("L_Upperarm", X, (-10.0, 40.0), "L_Forearm"),
                Ground("R_Upperarm", X, (-10.0, 40.0), "R_Forearm"),
            ),
            lay=((X, -90.0),),
            description="on the back, arms by the sides, one knee bent, head turned",
        ),
        PoseSpec(
            "lying_side",
            "victim",
            "stand_idle_loop",
            0,
            edits=(
                Turn("R_Upperarm", X, -85.0),
                Turn("R_Forearm", X, -30.0),
                Turn("L_Upperarm", X, -75.0),
                Turn("L_Forearm", X, -40.0),
                Ground("L_Upperarm", Z, (-80.0, 10.0), "L_Hand"),
                Turn("L_Thigh", X, -70.0),
                Turn("L_Calf", X, 85.0),
                Ground("L_Thigh", Z, (-60.0, 10.0), "L_Calf"),
                Turn("R_Thigh", X, -15.0),
                Turn("R_Calf", X, 20.0),
                Ground("R_Thigh", Y, (-10.0, 25.0), "R_Calf"),
                Ground("NeckTwist01", Y, (-40.0, 5.0), "Head", offset=0.04),
            ),
            lay=((X, -90.0), (Y, -90.0)),
            description="recovery position on the right side: top knee and hand on the ground",
        ),
        PoseSpec(
            "sitting",
            "victim",
            "Sit",
            30,
            edits=(
                Turn("R_Calf", X, -80.0),
                Ground("R_Thigh", X, (-25.0, 20.0), "R_Calf", SEAT, 0.01),
                Turn("L_Thigh", X, -35.0),
                Ground("L_Calf", X, (-70.0, 40.0), "L_Foot", SEAT, 0.0),
                Turn("L_Upperarm", X, -28.0),
            ),
            description="sitting on the ground, one leg out, the other knee up",
        ),
        PoseSpec(
            "standing_waving",
            "victim",
            "stand_idle_wave_loop",
            36,
            description="waving one arm above the head",
        ),
        PoseSpec(
            "standing_idle",
            "responder",
            "stand_idle_loop",
            60,
            description="standing at ease",
        ),
        PoseSpec(
            "standing_looking",
            "responder",
            "stand_idle_loop",
            0,
            layers=(Layer("LookAround", 108, ("Neck1", "Neck2", "Head")),),
            description="standing, looking up and to the side",
        ),
    )
}
"""Natural poses by key. Victims lie, sit or wave; responders stand."""

VICTIM_POSES = tuple(k for k, p in POSES.items() if p.role == "victim")
RESPONDER_POSES = tuple(k for k, p in POSES.items() if p.role == "responder")


# ---------------------------------------------------------------------------------------------------------------
# geometry data (numpy only)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseGeometry:
    """Box and under-side height map of one character in one pose (asset frame, unit scale)."""

    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    origin: tuple[float, float]
    """Lower corner (x, y) of the height map [m]."""
    underside: np.ndarray = field(repr=False)
    """(nx, ny) lowest point of the body above each :data:`CELL` square [m] (0 where it touches the ground,
    NaN where there is no body)."""

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(float(b - a) for a, b in zip(self.bbox_min, self.bbox_max, strict=True))  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _load_data(path: str = str(DATA_FILE)) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != DATA_FORMAT:
        raise ValueError(f"{path}: not a {DATA_FORMAT} file")
    if abs(float(data["cell_m"]) - CELL) > 1e-9:
        raise ValueError(f"{path}: cell size {data['cell_m']} != {CELL}")
    return data


@lru_cache(maxsize=256)
def pose_geometry(asset_key: str, pose: str) -> PoseGeometry:
    """Geometry of ``asset_key`` (a person of :data:`drone_lab.sar.assets.ASSETS`) in ``pose`` (a key of
    :data:`POSES`), from ``people_poses.json``."""
    try:
        rec = _load_data()["people"][asset_key][pose]
    except KeyError:
        raise KeyError(f"no geometry for {asset_key!r} in pose {pose!r}: run python -m drone_lab.sar.people") from None
    mm = np.asarray(rec["underside_mm"], dtype=np.float64)
    under = np.where(mm < 0, np.nan, mm / 1000.0)
    under.setflags(write=False)
    return PoseGeometry(tuple(rec["bbox_min"]), tuple(rec["bbox_max"]), tuple(rec["origin"]), under)


def data_digests() -> dict[str, str]:
    """Pose digests ``people_poses.json`` was generated with (compare with :meth:`PoseSpec.digest`)."""
    return dict(_load_data()["poses"])


def underside_grid(points: np.ndarray, cell: float = CELL) -> tuple[tuple[float, float], np.ndarray]:
    """Lowest z of ``points`` (N, 3) per ``cell`` square: ``(grid origin xy, (nx, ny) heights, NaN = empty)``."""
    p = np.asarray(points, dtype=np.float64)
    x0 = math.floor(p[:, 0].min() / cell) * cell
    y0 = math.floor(p[:, 1].min() / cell) * cell
    ix = np.floor((p[:, 0] - x0) / cell).astype(np.int64)
    iy = np.floor((p[:, 1] - y0) / cell).astype(np.int64)
    grid = np.full((ix.max() + 1, iy.max() + 1), np.inf)
    np.minimum.at(grid, (ix, iy), p[:, 2])
    grid[~np.isfinite(grid)] = np.nan
    return (x0, y0), grid


# ---------------------------------------------------------------------------------------------------------------
# retargeting and skinning (pxr)
# ---------------------------------------------------------------------------------------------------------------

# Biped world (Y up, +Z forward) -> character world (Z up, -Y forward): +90 deg about X, as a row-vector matrix
_TO_Z_UP = np.eye(4)
_TO_Z_UP[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T


def _rotation(axis: Axis, degrees: float) -> np.ndarray:
    """Column-vector rotation matrix about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    t = math.radians(degrees)
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def _matrices(values) -> np.ndarray:
    return np.array([np.array(m) for m in values], dtype=np.float64)


def _leaf(joint: str) -> str:
    return joint.rsplit("/", 1)[-1]


def _forward_kinematics(local: np.ndarray, parents: list[int]) -> np.ndarray:
    """Row-vector world transforms (J, 4, 4) from local ones (parents come before children)."""
    world = np.empty_like(local)
    for i, p in enumerate(parents):
        world[i] = local[i] if p < 0 else local[i] @ world[p]
    return world


def _skeleton(stage):
    from pxr import UsdSkel

    prim = next((p for p in stage.Traverse() if p.GetTypeName() == "Skeleton"), None)
    if prim is None:
        raise ValueError(f"no Skeleton in {stage.GetRootLayer().identifier}")
    skel = UsdSkel.Skeleton(prim)
    joints = list(skel.GetJointsAttr().Get())
    parents = list(UsdSkel.Topology(skel.GetJointsAttr().Get()).GetParentIndices())
    if any(p >= i for i, p in enumerate(parents)):
        raise ValueError("joints are not ordered parents first")
    return prim, skel, joints, parents


class SourceRig:
    """The Biped rig of the clips: rest pose and clip evaluation, in the characters' Z-up frame."""

    def __init__(self, rig_path: str, clip_paths: dict[str, str]):
        from pxr import Usd

        self.path = rig_path
        stage = Usd.Stage.Open(rig_path)
        _, skel, self.joints, self.parents = _skeleton(stage)
        self.rest_local = _matrices(skel.GetRestTransformsAttr().Get())
        self.index = {_leaf(j): i for i, j in enumerate(self.joints)}
        self.rest_world = _forward_kinematics(self.rest_local, self.parents) @ _TO_Z_UP
        self.clip_paths = dict(clip_paths)

    def local(self, clip: str, frame: float) -> np.ndarray:
        """Local joint transforms of ``clip`` at ``frame`` (components the clip leaves out come from the rest)."""
        from pxr import Gf, Usd, UsdSkel, Vt

        stage = Usd.Stage.Open(self.clip_paths[clip])
        prim = next(p for p in stage.Traverse() if p.GetTypeName() == "SkelAnimation")
        anim = UsdSkel.Animation(prim)
        if list(anim.GetJointsAttr().Get()) != self.joints:
            raise ValueError(f"{clip}: joints differ from the Biped rig")
        rest = UsdSkel.DecomposeTransforms(Vt.Matrix4dArray([Gf.Matrix4d(m.tolist()) for m in self.rest_local]))
        t = Usd.TimeCode(frame)
        parts = [a.Get(t) for a in (anim.GetTranslationsAttr(), anim.GetRotationsAttr(), anim.GetScalesAttr())]
        parts = [v if v is not None and len(v) else r for v, r in zip(parts, rest, strict=True)]
        return _matrices(UsdSkel.MakeTransforms(*parts))

    def world(self, clip: str, frame: float, layers: tuple[Layer, ...] = ()) -> np.ndarray:
        """World transforms (row-vector, Z up) of the rig at ``frame`` of ``clip``, with ``layers`` applied."""
        local = self.local(clip, frame)
        for layer in layers:
            other = self.local(layer.clip, layer.frame)
            for name in layer.joints:
                local[self.index[name]] = other[self.index[name]]
        return _forward_kinematics(local, self.parents) @ _TO_Z_UP


@dataclass
class PosedMesh:
    """One skinned mesh of a posed character."""

    path: str
    """Prim path of the mesh in the character's layer."""
    points: np.ndarray
    normals: np.ndarray


@dataclass
class PosedCharacter:
    """Skinned meshes of a posed character in the asset frame (see the module docstring)."""

    meshes: list[PosedMesh]
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    def points(self) -> np.ndarray:
        """All mesh points (N, 3)."""
        return np.concatenate([m.points for m in self.meshes])


class CharacterRig:
    """A skinned character: skeleton, reference T-pose, meshes and skin weights."""

    def __init__(self, usd_path: str):
        from pxr import Usd, UsdGeom, UsdSkel

        self.path = usd_path
        stage = Usd.Stage.Open(usd_path)
        if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-6 or UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError(f"{usd_path}: expected a metre, Z-up character")
        prim, skel, self.joints, self.parents = _skeleton(stage)
        self.index = {_leaf(j): i for i, j in enumerate(self.joints)}
        ref = prim.GetAttribute("controlRig:retargetTransforms").Get()
        self.ref_local = _matrices(ref if ref else skel.GetRestTransformsAttr().Get())
        self.ref_world = _forward_kinematics(self.ref_local, self.parents)
        self.bind = _matrices(skel.GetBindTransformsAttr().Get())
        self.children: list[list[int]] = [[] for _ in self.joints]
        for i, p in enumerate(self.parents):
            if p >= 0:
                self.children[p].append(i)
        self.meshes = []
        for p in stage.Traverse():
            if not p.IsA(UsdGeom.Mesh):
                continue
            binding = UsdSkel.BindingAPI(p)
            indices = binding.GetJointIndicesPrimvar()
            if not indices or binding.GetJointsAttr().HasAuthoredValue():
                raise ValueError(f"{p.GetPath()}: expected skeleton-ordered joint influences")
            n = indices.GetElementSize()
            mesh = UsdGeom.Mesh(p)
            geom_bind = binding.GetGeomBindTransformAttr().Get()
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if geom_bind is not None:
                points = np.c_[points, np.ones(len(points))] @ np.array(geom_bind)
                points = points[:, :3]
            jw = np.asarray(binding.GetJointWeightsPrimvar().Get(), dtype=np.float64).reshape(-1, n)
            ji = np.asarray(indices.Get(), dtype=np.int64).reshape(-1, n)
            self.meshes.append(
                {
                    "path": str(p.GetPath()),
                    "points": points,
                    "normals": np.asarray(mesh.GetNormalsAttr().Get(), dtype=np.float64),
                    "ji": ji,
                    "jw": jw / np.maximum(jw.sum(axis=1, keepdims=True), 1e-12),
                    "dominant": ji[np.arange(len(ji)), jw.argmax(axis=1)],
                }
            )

    def subtree(self, joint: int) -> list[int]:
        out, stack = [], [joint]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(self.children[j])
        return out

    def retarget(self, rig: SourceRig, source_world: np.ndarray) -> np.ndarray:
        """World transforms (row-vector, character frame) of this character in the source pose."""
        delta = {}
        for src, dst in JOINT_MAP.items():
            if src in rig.index and dst in self.index:
                i = rig.index[src]
                delta[self.index[dst]] = np.linalg.inv(rig.rest_world[i][:3, :3]) @ source_world[i][:3, :3]
        hip, pelvis = self.index["Hip"], rig.index["Pelvis"]
        scale = self.ref_world[hip][3, 2] / rig.rest_world[pelvis][3, 2]
        world = np.zeros_like(self.ref_local)
        for j, p in enumerate(self.parents):
            local = self.ref_local[j]
            if p < 0:
                world[j] = local
                continue
            if j == hip:  # the root motion, scaled to this character's leg length
                pos = self.ref_world[hip][3, :3] + scale * (source_world[pelvis][3, :3] - rig.rest_world[pelvis][3, :3])
            else:
                pos = local[3, :3] @ world[p][:3, :3] + world[p][3, :3]
            rot = self.ref_world[j][:3, :3] @ delta[j] if j in delta else local[:3, :3] @ world[p][:3, :3]
            world[j][:3, :3], world[j][3, :3], world[j][3, 3] = rot, pos, 1.0
        return world

    def turn(self, world: np.ndarray, joint: str, axis: Axis, degrees: float) -> None:
        """Rotate the subtree of ``joint`` about ``axis`` through the joint (in place)."""
        j = self.index[joint]
        r = _rotation(axis, degrees).T
        pivot = world[j][3, :3].copy()
        for k in self.subtree(j):
            world[k][:3, :3] = world[k][:3, :3] @ r
            world[k][3, :3] = (world[k][3, :3] - pivot) @ r + pivot

    def skin(self, world: np.ndarray, select: Callable[[dict], np.ndarray] | None = None) -> list[tuple]:
        """Linear blend skinning: ``[(points, normals)]`` per mesh (only the ``select``-ed vertices if given)."""
        skinning = np.linalg.inv(self.bind) @ world
        out = []
        for m in self.meshes:
            sel = select(m) if select is not None else slice(None)
            pts = np.c_[m["points"][sel], np.ones(len(m["points"][sel]))]
            nrm, ji, jw = m["normals"][sel], m["ji"][sel], m["jw"][sel]
            p_out, n_out = np.zeros((len(pts), 3)), np.zeros((len(pts), 3))
            for k in range(ji.shape[1]):
                s = skinning[ji[:, k]]
                p_out += jw[:, k : k + 1] * np.einsum("ni,nij->nj", pts, s)[:, :3]
                n_out += jw[:, k : k + 1] * np.einsum("ni,nij->nj", nrm, s[:, :3, :3])
            n_out /= np.maximum(np.linalg.norm(n_out, axis=1, keepdims=True), 1e-12)
            out.append((p_out, n_out))
        return out

    def _lowest(self, world: np.ndarray, joints: list[int], lay: np.ndarray) -> float:
        wanted = np.asarray(joints)
        parts = self.skin(world, select=lambda m: np.isin(m["dominant"], wanted))
        z = [(p @ lay)[:, 2] for p, _ in parts if len(p)]
        return float(np.concatenate(z).min())

    def ground(self, world: np.ndarray, edit: Ground, lay: np.ndarray) -> float:
        """Apply a :class:`Ground` edit (in place); returns the angle found [deg]."""
        moving = self.subtree(self.index[edit.measure])
        if edit.reference:
            reference = [self.index[n] for n in edit.reference if n in self.index]
        else:
            turned = set(self.subtree(self.index[edit.joint]))
            reference = [j for j in range(len(self.joints)) if j not in turned]
        target = self._lowest(world, reference, lay) + edit.offset

        def gap(deg: float) -> float:
            trial = world.copy()
            self.turn(trial, edit.joint, edit.axis, deg)
            return self._lowest(trial, moving, lay) - target

        lo, hi = edit.limits
        g_lo, g_hi = gap(lo), gap(hi)
        if g_lo * g_hi > 0:  # no contact inside the limits: take the closer end
            best = lo if abs(g_lo) < abs(g_hi) else hi
        else:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                g_mid = gap(mid)
                if g_mid * g_lo > 0:
                    lo, g_lo = mid, g_mid
                else:
                    hi = mid
            best = 0.5 * (lo + hi)
        self.turn(world, edit.joint, edit.axis, best)
        return best

    def pose(self, rig: SourceRig, spec: PoseSpec, log: list | None = None) -> PosedCharacter:
        """Pose, skin and lay down; points are re-centred to the asset frame of the module docstring.

        ``log`` (optional) receives ``(edit, angle)`` for every :class:`Ground` edit.
        """
        lay = np.eye(3)
        for axis, deg in spec.lay:
            lay = _rotation(axis, deg) @ lay
        lay_row = lay.T
        world = self.retarget(rig, rig.world(spec.clip, spec.frame, spec.layers))
        for edit in spec.edits:
            if isinstance(edit, Turn):
                self.turn(world, edit.joint, edit.axis, edit.degrees)
            else:
                angle = self.ground(world, edit, lay_row)
                if log is not None:
                    log.append((edit, round(angle, 2)))
        skinned = self.skin(world)
        pts = np.concatenate([p for p, _ in skinned]) @ lay_row
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        shift = -np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
        meshes = [
            PosedMesh(m["path"], p @ lay_row + shift, n @ lay_row)
            for m, (p, n) in zip(self.meshes, skinned, strict=True)
        ]
        return PosedCharacter(meshes, lo + shift, hi + shift)


# ---------------------------------------------------------------------------------------------------------------
# static USD
# ---------------------------------------------------------------------------------------------------------------

_SKEL_PROPERTIES = (
    "skel:skinningMethod",
    "primvars:skel:jointIndices",
    "primvars:skel:jointWeights",
    "primvars:skel:geomBindTransform",
    "primvars:skel:skinningMethod",
    "skel:skeleton",
    "skel:joints",
    "skel:animationSource",
    "skel:blendShapes",
    "skel:blendShapeTargets",
)


def write_posed_usd(character_usd: str, posed: PosedCharacter, out_path: str, metadata: dict | None = None) -> str:
    """Static copy of a character in a pose: its meshes and materials with the posed points and normals.

    The skeleton and the skinning data are removed and the ``SkelRoot`` becomes an ``Xform``, so nothing deforms
    the meshes at runtime. Relative asset paths (textures) are made absolute, so the file works from any folder.
    The file is written atomically (a concurrent writer of the same pose writes the same content).
    """
    from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdUtils, Vt

    source = Sdf.Layer.FindOrOpen(character_usd)
    layer = Sdf.Layer.CreateAnonymous(".usd")
    if source.subLayerPaths or source.externalReferences:
        layer.TransferContent(Usd.Stage.Open(source).Flatten())
    else:  # one self-contained layer: copy it as is (keeps Kit's own metadata fields)
        layer.TransferContent(source)
    stage = Usd.Stage.Open(layer)
    anchor = Ar.ResolvedPath(os.path.abspath(character_usd))
    resolver = Ar.GetResolver()

    def absolute(path: str) -> str:
        if not path or "://" in path or os.path.isabs(path) or (path.endswith(".mdl") and "/" not in path):
            return path  # remote, absolute, or an MDL module from the search paths (OmniPBR.mdl)
        return str(resolver.Resolve(resolver.CreateIdentifier(path, anchor))) or path

    UsdUtils.ModifyAssetPaths(layer, absolute)
    by_path = {m.path: m for m in posed.meshes}
    for prim in list(stage.Traverse()):
        if prim.GetTypeName() == "Skeleton":
            stage.RemovePrim(prim.GetPath())
            continue
        if prim.GetTypeName() == "SkelRoot":
            prim.SetTypeName("Xform")
        for name in _SKEL_PROPERTIES:
            if prim.HasProperty(name):
                prim.RemoveProperty(name)
        if "SkelBindingAPI" in prim.GetAppliedSchemas():
            prim.RemoveAPI(Usd.SchemaRegistry.GetTypeFromSchemaTypeName("SkelBindingAPI"))
        mesh = by_path.get(str(prim.GetPath()))
        if mesh is not None:
            m = UsdGeom.Mesh(prim)
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.points.astype(np.float32)))
            m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.normals.astype(np.float32)))
            lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
            m.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
    missing = set(by_path) - {str(p.GetPath()) for p in stage.Traverse()}
    if missing:
        raise RuntimeError(f"posed meshes not found in {character_usd}: {sorted(missing)}")
    if stage.GetPrimAtPath("/Render"):
        stage.RemovePrim("/Render")
    layer.customLayerData = {"drone_lab": dict(metadata or {})}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.{os.getpid()}.tmp.usd"
    if not layer.Export(tmp):
        raise RuntimeError(f"could not write {tmp}")
    os.replace(tmp, out_path)
    return out_path


# ---------------------------------------------------------------------------------------------------------------
# baking with downloads (spawner and CLI)
# ---------------------------------------------------------------------------------------------------------------


def _files_digest(paths: list[str]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]


class PoseBaker:
    """Downloads (through ``fetch``) and poses characters; keeps the rig and clips loaded across calls."""

    def __init__(self, fetch: Callable[[str], str], root_url: str, out_dir: str | None = None):
        self.fetch = fetch
        self.root_url = root_url.rstrip("/")
        self.out_dir = out_dir or os.path.join(tempfile.gettempdir(), "drone_lab_people")
        self._rig: SourceRig | None = None
        self._characters: dict[str, CharacterRig] = {}

    def _url(self, rel: str) -> str:
        return f"{self.root_url}/{rel}"

    @property
    def rig(self) -> SourceRig:
        if self._rig is None:
            clips = sorted({p.clip for p in POSES.values()} | {la.clip for p in POSES.values() for la in p.layers})
            paths = {c: self.fetch(self._url(f"{ANIMATIONS_DIR}/{c}.skelanim.usd")) for c in clips}
            self._rig = SourceRig(self.fetch(self._url(BIPED_RIG)), paths)
        return self._rig

    def character(self, character_path: str) -> CharacterRig:
        if character_path not in self._characters:
            self._characters[character_path] = CharacterRig(self.fetch(self._url(character_path)))
        return self._characters[character_path]

    def pose(self, character_path: str, pose: str) -> PosedCharacter:
        return self.character(character_path).pose(self.rig, POSES[pose])

    def usd(self, asset_key: str, character_path: str, pose: str) -> str:
        """Path of the static posed USD (baked on first use; later calls reuse the file)."""
        spec = POSES[pose]
        rig = self.rig
        char = self.character(character_path)
        key = _files_digest([char.path, rig.path, *(rig.clip_paths[c] for c in sorted(rig.clip_paths))])
        out = os.path.join(self.out_dir, asset_key, f"{pose}-{spec.digest()}-{key}.usd")
        if not os.path.isfile(out):
            posed = char.pose(rig, spec)
            meta = {"asset": asset_key, "pose": pose, "pose_digest": spec.digest(), "source_digest": key}
            write_posed_usd(char.path, posed, out, meta)
        return out


def geometry_record(posed: PosedCharacter, cell: float = CELL) -> dict:
    """``people_poses.json`` entry of a posed character (box, under-side height map in mm, -1 = empty)."""
    origin, grid = underside_grid(posed.points(), cell)
    mm = np.where(np.isnan(grid), -1, np.round(np.nan_to_num(grid) * 1000.0)).astype(int)
    return {
        "bbox_min": [round(float(v), 4) for v in posed.bbox_min],
        "bbox_max": [round(float(v), 4) for v in posed.bbox_max],
        "origin": [round(origin[0], 4), round(origin[1], 4)],
        "underside_mm": mm.tolist(),
    }


def write_data(records: dict[str, dict[str, dict]], path: Path = DATA_FILE) -> Path:
    """Write ``people_poses.json`` (one line per character and pose, so diffs stay readable)."""
    lines = [
        "{",
        f' "format": "{DATA_FORMAT}",',
        f' "cell_m": {CELL},',
        f' "poses": {json.dumps({k: p.digest() for k, p in POSES.items()})},',
        ' "people": {',
    ]
    keys = sorted(records)
    for i, key in enumerate(keys):
        poses = sorted(records[key])
        lines.append(f'  "{key}": {{')
        for j, pose in enumerate(poses):
            sep = "," if j < len(poses) - 1 else ""
            lines.append(f'   "{pose}": {json.dumps(records[key][pose], separators=(",", ":"))}{sep}')
        lines.append("  }" + ("," if i < len(keys) - 1 else ""))
    lines += [" }", "}"]
    path.write_text("\n".join(lines) + "\n")
    _load_data.cache_clear()
    pose_geometry.cache_clear()
    return path


def preview_sheet(baker: PoseBaker, character_path: str, out_png: str, poses=None) -> str:
    """Contact sheet of one character in every pose (matplotlib, front and side views)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from pxr import Usd, UsdGeom

    poses = list(poses or POSES)
    stage = Usd.Stage.Open(baker.character(character_path).path)
    faces = {}
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(p)
            counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
            tris, off = [], 0
            for c in counts:
                tris += [(idx[off], idx[off + k], idx[off + k + 1]) for k in range(1, c - 1)]
                off += c
            faces[str(p.GetPath())] = np.asarray(tris)
    fig = plt.figure(figsize=(4.2 * len(poses), 8.4))
    light = np.array([0.35, -0.6, 0.72]) / np.linalg.norm([0.35, -0.6, 0.72])
    for col, pose in enumerate(poses):
        posed = baker.pose(character_path, pose)
        tri = np.concatenate([m.points[faces[m.path]] for m in posed.meshes])
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = 0.3 + 0.7 * np.abs(n @ light)
        lo, hi = posed.bbox_min, posed.bbox_max
        ctr, r = 0.5 * (lo + hi), 0.5 * float(max(hi - lo)) + 0.05
        for row, (elev, azim) in enumerate(((12, -70), (8, 20))):
            ax = fig.add_subplot(2, len(poses), row * len(poses) + col + 1, projection="3d")
            ground = [[ctr[0] - r, ctr[1] - r, 0], [ctr[0] + r, ctr[1] - r, 0], [ctr[0] + r, ctr[1] + r, 0]]
            ax.add_collection3d(
                Poly3DCollection([[*ground, [ctr[0] - r, ctr[1] + r, 0]]], facecolors=(0.55, 0.7, 0.5, 0.4))
            )
            ax.add_collection3d(
                Poly3DCollection(tri, facecolors=np.c_[shade, shade * 0.97, shade * 0.93], edgecolors="none")
            )
            ax.set(xlim=(ctr[0] - r, ctr[0] + r), ylim=(ctr[1] - r, ctr[1] + r), zlim=(0, 2 * r))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"{pose}\n{np.round(hi - lo, 2).tolist()} m", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=72)
    plt.close(fig)
    return out_png


def main(argv: list[str] | None = None) -> None:
    from isaaclab.utils.assets import retrieve_file_path

    from .assets import asset_root, assets_of
    from .scene import asset_cache_dir

    parser = argparse.ArgumentParser(description="Bake the natural poses and write people_poses.json.")
    parser.add_argument("--write-data", action="store_true", help=f"Rewrite {DATA_FILE.name}.")
    parser.add_argument("--preview", type=str, default=None, help="Folder for contact sheets (one per character).")
    parser.add_argument("--people", nargs="*", default=None, help="Asset keys (default: every person).")
    args = parser.parse_args(argv)
    cache = asset_cache_dir()
    baker = PoseBaker(lambda url: retrieve_file_path(url, download_dir=cache), asset_root())
    people = [a for a in assets_of("person") if args.people is None or a.key in args.people]
    records: dict[str, dict[str, dict]] = {}
    for spec in people:
        records[spec.key] = {pose: geometry_record(baker.pose(spec.path, pose)) for pose in POSES}
        sizes = {
            k: [round(b - a, 2) for a, b in zip(r["bbox_min"], r["bbox_max"], strict=True)]
            for k, r in records[spec.key].items()
        }
        print(f"{spec.key}: {sizes}", flush=True)
        if args.preview:
            print(preview_sheet(baker, spec.path, os.path.join(args.preview, f"{spec.key}.png")), flush=True)
    if args.write_data:
        if args.people is not None:
            raise SystemExit("--write-data needs every person (drop --people)")
        print(write_data(records))


if __name__ == "__main__":
    main()
