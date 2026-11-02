# Copyright (c) 2026 Hector Lugo
# SPDX-License-Identifier: MIT

"""What the depth camera of the experimental depth task sees: self-occlusion and ground coverage (CPU, no Kit).

The robot is posed at its default joint positions (forward kinematics from the joint frames in ``robot.usd``) and
stands level on flat ground; its standing height is the lowest point of its collision meshes. Every pixel ray of the
pinhole camera is intersected with the robot's own meshes (visual and collision geometry, separately) and with the
ground plane. Reported per mount:

* ``robot %``: pixels whose first hit is the robot itself (self-occlusion);
* nearest visible ground ahead of the trunk origin on the centre line (``|y| < 0.1 m``) and in the foot tracks
  (``0.1 <= |y| < 0.3 m``), and the farthest visible ground point;
* the range of ray lengths to the ground and the share of pixels beyond the far clip and beyond 1.0 m (the 2.x clip).

The configured camera of ``CustomQuadRoughDepthEnvCfg`` is always studied, together with the 2.x mount for reference.
More mounts can be compared with ``--mount NAME:X,Y,Z:PITCH_DEG`` (forward-looking, pitched down, trunk frame).

Usage (from the project root, conda env ``env_isaaclab`` active)::

    PYTHONPATH=src python scripts/depth_camera_study.py [--markdown] [--mount front60:0.47,0,0.08:60]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.sensors import CameraCfg

MOUNT_2X = ((-0.25, 0.0, 0.7), (0.24184, -0.24184, -0.66446, 0.66446))
"""Camera offset (pos, rot xyzw, OpenGL convention) of the 2.x depth-camera patch, for reference."""

GEOMETRY_KINDS = ("visuals", "collisions")


def quat_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion given as (x, y, z, w)."""
    x, y, z, w = quat_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pitch_down_quat(pitch_deg: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) of an OpenGL camera looking along the parent's +x, pitched down by ``pitch_deg``."""
    t = math.radians(pitch_deg)
    # columns: camera +x (image right) = parent -y, camera +y (image up), camera +z (= -viewing direction)
    m = np.array([[0.0, math.sin(t), -math.cos(t)], [-1.0, 0.0, 0.0], [0.0, math.cos(t), math.sin(t)]])
    w = math.sqrt(max(0.0, 1 + np.trace(m))) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return (x, y, z, w)


def _usd_matrix(m: Sequence[Sequence[float]]) -> np.ndarray:
    """USD (row-vector) 4x4 matrix -> column-vector numpy matrix."""
    return np.array(m, dtype=float).T


def _joint_frame(pos: Sequence[float], rot: object) -> np.ndarray:
    """4x4 transform of a joint frame from its local position and ``Gf.Quatf`` rotation."""
    frame = np.eye(4)
    imag = rot.GetImaginary()
    frame[:3, :3] = quat_to_matrix((imag[0], imag[1], imag[2], rot.GetReal()))
    frame[:3, 3] = np.array(pos, dtype=float)
    return frame


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    """4x4 rotation by ``angle`` [rad] about the ``X``, ``Y`` or ``Z`` axis."""
    c, s = math.cos(angle), math.sin(angle)
    rot = {"X": [[1, 0, 0], [0, c, -s], [0, s, c]], "Y": [[c, 0, s], [0, 1, 0], [-s, 0, c]]}
    rot["Z"] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    out = np.eye(4)
    out[:3, :3] = rot[axis]
    return out


def posed_robot_triangles(
    usd_path: str, joint_pos: dict[str, float]
) -> tuple[dict[tuple[str, str], np.ndarray], float]:
    """Triangles of every link (per geometry kind) in the articulation-root frame, posed at ``joint_pos``.

    Args:
        usd_path: Robot USD; its rigid bodies must be direct children of the default prim.
        joint_pos: Joint positions [rad] keyed by regular expressions (full match), as in ``InitialStateCfg``.

    Returns:
        ``{(link, kind): (N, 3, 3) float32 triangles}`` and the standing height of the root-link origin above flat
        ground (minus the lowest collision vertex).
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    links = {p.GetName(): p for p in stage.GetDefaultPrim().GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    rest = {name: _usd_matrix(xform_cache.GetLocalToWorldTransform(prim)) for name, prim in links.items()}
    root = next(p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if body0 and body1:  # joints to the world do not move links relative to the root
                joints.append((prim, body0[0].name, body1[0].name))
    # child pose = parent pose * joint frame on the parent * joint motion * inverse(joint frame on the child)
    pose = {root: np.eye(4)}
    pending = list(joints)
    while pending:  # parents before children, whatever the authored order
        ready = [item for item in pending if item[1] in pose]
        if not ready:
            raise ValueError(f"joints not connected to the articulation root: {[p.GetName() for p, _, _ in pending]}")
        for item in ready:
            prim, body0, body1 = item
            joint = UsdPhysics.Joint(prim)
            frame0 = _joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            frame1 = _joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            # the joint frames must reproduce the authored link poses at zero joint position
            error = np.abs((rest[body0] @ frame0 @ np.linalg.inv(frame1))[:3, 3] - rest[body1][:3, 3]).max()
            if error > 1e-4:
                raise ValueError(f"joint {prim.GetName()} does not match the authored rest pose ({error:.2e} m)")
            motion = np.eye(4)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                angle = next((v for k, v in joint_pos.items() if re.fullmatch(k, prim.GetName())), 0.0)
                motion = _axis_rotation(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get(), angle)
            pose[body1] = pose[body0] @ frame0 @ motion @ np.linalg.inv(frame1)
            pending.remove(item)

    triangles: dict[tuple[str, str], np.ndarray] = {}
    for name, link in links.items():
        link_from_world = np.linalg.inv(rest[name])
        for kind in GEOMETRY_KINDS:
            parts = []
            for gprim in Usd.PrimRange(link.GetChild(kind), Usd.TraverseInstanceProxies()):
                if not gprim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(gprim)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                if np.all(counts == 3):
                    faces = indices.reshape(-1, 3)
                else:  # fan triangulation of polygons
                    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
                    faces = np.array(
                        [
                            (indices[s], indices[s + k], indices[s + k + 1])
                            for s, c in zip(starts, counts, strict=False)
                            for k in range(1, c - 1)
                        ]
                    )
                to_root = pose[name] @ link_from_world @ _usd_matrix(xform_cache.GetLocalToWorldTransform(gprim))
                posed = (np.c_[points, np.ones(len(points))] @ to_root.T)[:, :3]
                parts.append(posed[faces])
            if parts:
                triangles[(name, kind)] = np.concatenate(parts).astype(np.float32)
    lowest = min(float(t[..., 2].min()) for (_, kind), t in triangles.items() if kind == "collisions")
    return triangles, -lowest


def camera_rays(
    pos: Sequence[float],
    rot_xyzw: Sequence[float],
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Origin and unit directions (row-major pixels) of a pinhole camera mounted with the OpenGL convention."""
    vertical_aperture = vertical_aperture or horizontal_aperture * height / width  # square pixels
    fx, fy = width * focal_length / horizontal_aperture, height * focal_length / vertical_aperture
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    dirs = np.stack([(u - width / 2) / fx, -(v - height / 2) / fy, -np.ones_like(u)], -1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.asarray(pos, dtype=float), dirs @ quat_to_matrix(rot_xyzw).T


def _ray_aabb(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Which rays hit the axis-aligned box ``[lo, hi]`` (slab test)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - origin) / dirs, (hi - origin) / dirs
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    return t_far >= np.maximum(t_near, 0.0)


def _ray_triangles(origin: np.ndarray, dirs: np.ndarray, tris: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance to the nearest triangle hit per ray (Moller-Trumbore), inf for a miss."""
    best = np.full(len(dirs), np.inf)
    d = dirs.astype(np.float32)
    o = origin.astype(np.float32)
    for s in range(0, len(tris), chunk):
        v0 = tris[s : s + chunk, 0]
        e1, e2 = tris[s : s + chunk, 1] - v0, tris[s : s + chunk, 2] - v0
        p = np.cross(d[:, None, :], e2[None])
        det = np.einsum("rtk,tk->rt", p, e1)
        valid = np.abs(det) > 1e-12
        inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
        tvec = o - v0
        bary_u = np.einsum("rtk,tk->rt", p, tvec) * inv
        q = np.cross(tvec, e1)
        bary_v = (d @ q.T) * inv
        dist = (q * e2).sum(1)[None] * inv
        hit = valid & (bary_u >= 0) & (bary_v >= 0) & (bary_u + bary_v <= 1) & (dist > 1e-6)
        best = np.minimum(best, np.where(hit, dist, np.inf).min(1))
    return best


def study_mount(
    origin: np.ndarray, dirs: np.ndarray, triangles: dict, kind: str, standing_height: float, far_clip: float
) -> dict:
    """Self-occlusion and ground-coverage metrics of one camera mount for one geometry kind."""
    t_robot = np.full(len(dirs), np.inf)
    for (_, k), tris in triangles.items():
        if k != kind:
            continue
        pts = tris.reshape(-1, 3)
        sel = _ray_aabb(origin, dirs, pts.min(0) - 1e-4, pts.max(0) + 1e-4)
        if sel.any():
            t_robot[sel] = np.minimum(t_robot[sel], _ray_triangles(origin, dirs[sel], tris))
    with np.errstate(divide="ignore"):
        t_ground = np.where(dirs[:, 2] < 0, (-standing_height - origin[2]) / dirs[:, 2], np.inf)
    robot_first = t_robot < np.minimum(t_ground, far_clip)
    ground_first = ~robot_first & (t_ground <= far_clip)
    hits = origin + dirs * t_ground[:, None]
    gx, gy = hits[ground_first, 0], np.abs(hits[ground_first, 1])

    def nearest(lo: float, hi: float) -> float:
        sel = (gy >= lo) & (gy < hi)
        return float(gx[sel].min()) if sel.any() else math.nan

    return {
        "robot_pct": 100.0 * robot_first.mean(),
        "nearest_ground_centre": nearest(0.0, 0.1),
        "nearest_ground_feet": nearest(0.1, 0.3),
        "farthest_ground": float(gx.max()) if len(gx) else math.nan,
        "ground_ray_min": float(t_ground.min()),
        "ground_ray_max": float(t_ground[np.isfinite(t_ground)].max()) if np.isfinite(t_ground).any() else math.inf,
        "beyond_far_clip_pct": 100.0 * (t_ground > far_clip).mean(),
        "beyond_1m_pct": 100.0 * (t_ground > 1.0).mean(),
    }


def configured_setup() -> tuple[CameraCfg, str, dict[str, float]]:
    """Camera cfg, robot USD path and default joint positions of the depth task, from the project configs (no Kit)."""
    from quadruped_lab.tasks.locomotion.config.custom_quad.rough_env_depth_cfg import CustomQuadRoughDepthEnvCfg

    cfg = CustomQuadRoughDepthEnvCfg()
    cam = cfg.scene.depth_camera
    if cam.offset.convention != "opengl":
        raise NotImplementedError(f"only the 'opengl' offset convention is supported, got {cam.offset.convention!r}")
    robot = cfg.scene.robot.default  # CustomQuadRobotCfg preset: the PhysX robot (robot.usd), the task default
    return cam, robot.spawn.usd_path, dict(robot.init_state.joint_pos)


def main(argv: list[str] | None = None) -> int:
    """Print the study for the configured mount, the 2.x mount and any ``--mount``; exit 1 above ``--max-robot-pct``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mount", action="append", default=[], help="Extra mount NAME:X,Y,Z:PITCH_DEG (trunk frame).")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table.")
    parser.add_argument("--max-robot-pct", type=float, default=None, help="Fail if the configured mount sees more.")
    args = parser.parse_args(argv)

    cam, usd_path, joint_pos = configured_setup()
    spawn = cam.spawn
    far_clip = spawn.clipping_range[1]
    triangles, standing_height = posed_robot_triangles(usd_path, joint_pos)
    mounts = [("configured", tuple(cam.offset.pos), tuple(cam.offset.rot)), ("2.x patch", *MOUNT_2X)]
    for spec in args.mount:
        name, pos, pitch = spec.split(":")
        mounts.append((name, tuple(float(v) for v in pos.split(",")), pitch_down_quat(float(pitch))))

    print(
        f"camera {cam.width}x{cam.height}, focal {spawn.focal_length} mm, aperture {spawn.horizontal_aperture} mm, "
        f"far clip {far_clip} m; robot posed at its default joint positions, trunk origin {standing_height:.3f} m"
        " above flat ground"
    )
    header = (
        "| mount | pos (trunk frame) | geometry | robot % | nearest ground x, centre / feet [m] "
        "| farthest ground x [m] | ground ray length [m] | > far clip % | > 1.0 m % |"
    )
    if args.markdown:
        print(header)
        print("|" + "---|" * header.count(" |"))
    worst = 0.0
    for name, pos, rot in mounts:
        origin, dirs = camera_rays(
            pos, rot, cam.width, cam.height, spawn.focal_length, spawn.horizontal_aperture, spawn.vertical_aperture
        )
        for kind in GEOMETRY_KINDS:
            r = study_mount(origin, dirs, triangles, kind, standing_height, far_clip)
            if name == "configured":
                worst = max(worst, r["robot_pct"])
            cells = [
                name,
                str(pos),
                kind,
                f"{r['robot_pct']:.1f}",
                f"{r['nearest_ground_centre']:.2f} / {r['nearest_ground_feet']:.2f}",
                f"{r['farthest_ground']:.2f}",
                f"{r['ground_ray_min']:.2f}-{r['ground_ray_max']:.2f}",
                f"{r['beyond_far_clip_pct']:.1f}",
                f"{r['beyond_1m_pct']:.1f}",
            ]
            print("| " + " | ".join(cells) + " |" if args.markdown else "  ".join(cells))
    if args.max_robot_pct is not None and worst > args.max_robot_pct:
        print(
            f"configured mount sees the robot in {worst:.1f} % of the pixels (> {args.max_robot_pct} %)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
