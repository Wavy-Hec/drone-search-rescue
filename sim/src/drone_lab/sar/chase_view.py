"""What the Isaac Sim window and the mission video show: a follow camera, ground to the horizon, the mapping rays.

The extras live in their own RTX scene partition (:data:`CHASE_PARTITION`). Only the viewer camera (the window's
viewport and the video recorder, which render the same camera prim) is assigned to it; the drone's camera gets
a partition of its own. RTX renders a partition only for the cameras assigned to it, so the drone camera, and with
it the detector and the photo map, sees exactly the site it sees without the chase view.

* :class:`FollowCamera` -- smooth third-person camera that frames the drone and the ground under it, its eye kept
  above the site (no views from beyond the terrain edge) and turning with the drone's east-west position, so it
  looks back over the site at the ends of the survey lines.
* :func:`spawn_ground_skirt` -- flat ground from the site's edge to the horizon, with the site's own grass.
* :class:`ScanRays` -- the mapping scanner's ray hits as glowing dots on the ground.

Numpy at module level (the camera and the skirt geometry are testable without the simulator); USD and Isaac Lab
are imported inside the functions that need the running app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CHASE_PARTITION = "chase_view"
"""RTX scene partition of the window / video extras."""
DRONE_CAMERA_PARTITION = "drone_camera"
"""Partition of the drone's camera: it renders the unpartitioned site only."""
CHASE_ROOT = "/World/chase_view"
"""Parent prim of the extras (outside ``/World/envs`` and the site)."""
SCAN_RAY_COLOR = (0.15, 0.95, 1.0)
"""Cyan of the scan dots (also emissive, so they read in full sun)."""


@dataclass
class FollowCamera:
    """Third-person camera behind and above the drone that keeps the drone and the ground under it in view.

    It looks at a point below the drone (``look_below`` of the way down to the ground), with a wide lens
    (``focal_length``), so the drone sits in the upper part of the frame and its scan on the ground in the lower
    part. The look-at point follows the drone with first-order smoothing (time constants ``tau_xy`` / ``tau_z``),
    so the view glides through the turns and over steps in the terrain. ``swing_x`` turns the eye around the
    drone as it crosses the site from west to east, so at both ends of the lines the camera looks back over the
    site instead of out over the border. ``bounds`` (x_min, x_max, y_min, y_max) keeps the eye above the site:
    near the edges the camera moves in closer instead of looking in from outside.
    """

    offset: tuple[float, float, float] = (-4.5, -6.5, 6.0)
    """Eye position relative to the look-at point [m] (at the west end of ``swing_x``)."""
    look_below: float = 0.3
    """Height of the look-at point between the drone (0) and the ground under it (1)."""
    focal_length: float = 15.0
    """Lens of the viewer camera [mm] (horizontal aperture 20.955 mm: 70 deg field of view)."""
    bounds: tuple[float, float, float, float] | None = None
    swing_x: tuple[float, float] | None = None
    """West and east x [m] between which the eye turns from ``offset`` to its mirror image east of the look-at
    point (same distance and height); None keeps ``offset`` everywhere."""
    tau_xy: float = 0.4
    tau_z: float = 0.5
    _target: np.ndarray | None = field(default=None, init=False, repr=False)

    def eye_offset(self, x: float) -> np.ndarray:
        """Eye position relative to the look-at point when that point is at ``x`` [m]."""
        off = np.asarray(self.offset, dtype=np.float64)
        if self.swing_x is None:
            return off
        x0, x1 = self.swing_x
        s = min(max((x - x0) / max(x1 - x0, 1e-9), 0.0), 1.0)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep: the turn starts and ends without a jerk
        a0, a1 = math.atan2(off[1], off[0]), math.atan2(off[1], -off[0])
        a, r = a0 + s * (a1 - a0), math.hypot(off[0], off[1])
        return np.array([r * math.cos(a), r * math.sin(a), off[2]])

    def update(self, drone_pos, ground_z: float | None, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance by ``dt`` seconds; returns ``(eye, target)`` in world coordinates.

        ``ground_z`` is the ground height under the drone (e.g. the median scan hit); None looks at the drone.
        """
        p = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        goal = p.copy()
        if ground_z is not None and math.isfinite(ground_z):
            goal[2] -= self.look_below * max(p[2] - float(ground_z), 0.0)
        if self._target is None:
            self._target = goal
        else:
            a_xy = 1.0 - math.exp(-dt / max(self.tau_xy, 1e-6))
            a_z = 1.0 - math.exp(-dt / max(self.tau_z, 1e-6))
            self._target[:2] += a_xy * (goal[:2] - self._target[:2])
            self._target[2] += a_z * (goal[2] - self._target[2])
        target = self._target.copy()
        eye = target + self.eye_offset(target[0])
        if self.bounds is not None:
            x0, x1, y0, y1 = self.bounds
            eye[0] = min(max(eye[0], x0), x1)
            eye[1] = min(max(eye[1], y0), y1)
        return eye, target


def skirt_mesh(
    bounds: tuple[float, float, float, float], extent: float, z: float
) -> tuple[np.ndarray, list[int], list[int]]:
    """Flat square frame from the site's edge out to ``extent`` metres around it, at height ``z``.

    The hole is exactly the site, so the skirt never covers terrain that lies below ``z`` (pits, valleys).
    Returns ``(points (8, 3), face_vertex_counts, face_vertex_indices)`` of four quads facing up.
    """
    x0, x1, y0, y1 = bounds
    inner = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    outer = [
        (x0 - extent, y0 - extent),
        (x1 + extent, y0 - extent),
        (x1 + extent, y1 + extent),
        (x0 - extent, y1 + extent),
    ]
    points = np.array([(x, y, z) for x, y in outer + inner], dtype=np.float64)
    indices: list[int] = []
    for i in range(4):
        j = (i + 1) % 4
        indices += [i, j, 4 + j, 4 + i]  # counter-clockwise seen from above
    return points, [4] * 4, indices


def rim_points(bounds: tuple[float, float, float, float], inset: float = 0.1, n: int = 50) -> np.ndarray:
    """``(4 n, 2)`` points along the edge of the site, ``inset`` metres inside it (where the skirt meets it)."""
    x0, x1, y0, y1 = bounds
    s = np.linspace(0.0, 1.0, n)
    xs = x0 + inset + (x1 - x0 - 2.0 * inset) * s
    ys = y0 + inset + (y1 - y0 - 2.0 * inset) * s
    sides = [(xs, np.full(n, y0 + inset)), (xs, np.full(n, y1 - inset)), (np.full(n, x0 + inset), ys)]
    sides.append((np.full(n, x1 - inset), ys))
    return np.concatenate([np.column_stack(side) for side in sides])


def set_focal_length(stage, camera_path: str, focal_length: float) -> None:
    """Change the lens of a camera prim (in the session layer, like Kit's own viewport camera)."""
    from pxr import Usd, UsdGeom

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Camera(stage.GetPrimAtPath(camera_path)).GetFocalLengthAttr().Set(float(focal_length))


def isolate_chase_view(stage, viewer_camera_path: str, drone_camera_paths: list[str]) -> None:
    """Assign the viewer camera to :data:`CHASE_PARTITION` and the drone's cameras to a partition of their own."""
    from pxr import Sdf, Usd

    tokens = {viewer_camera_path: CHASE_PARTITION} | dict.fromkeys(drone_camera_paths, DRONE_CAMERA_PARTITION)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path, token in tokens.items():
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"no camera prim at {path}")
            attr = prim.GetAttribute("omni:scenePartition") or prim.CreateAttribute(
                "omni:scenePartition", Sdf.ValueTypeNames.Token
            )
            attr.Set(token)


def _chase_root(stage):
    """The partitioned parent prim of the extras (created on first use)."""
    from pxr import Sdf, UsdGeom

    root = stage.GetPrimAtPath(CHASE_ROOT)
    if not root.IsValid():
        root = UsdGeom.Xform.Define(stage, CHASE_ROOT).GetPrim()
        UsdGeom.PrimvarsAPI(root).CreatePrimvar("omni:scenePartition", Sdf.ValueTypeNames.Token).Set(CHASE_PARTITION)
    return root


def spawn_ground_skirt(
    stage, bounds: tuple[float, float, float, float], z: float, material_path: str | None, extent: float = 2000.0
) -> str:
    """Ground from the site's edge to the horizon, for the viewer camera only; returns its prim path.

    ``material_path`` is bound when it exists (the site's grass, projected in world space, so the skirt continues
    the border's texture without a seam); otherwise the skirt is a plain green-brown.
    """
    from pxr import Gf, UsdGeom, UsdShade, Vt

    _chase_root(stage)
    path = f"{CHASE_ROOT}/ground_skirt"
    points, counts, indices = skirt_mesh(bounds, extent, z)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * len(indices))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    material = stage.GetPrimAtPath(material_path) if material_path else None
    if material is not None and material.IsValid():
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(material))
    else:
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.36, 0.18)])
    return path


class ScanRays:
    """The mapping scanner's ray hits as small glowing dots on the ground, for the viewer camera only.

    Every ray keeps its marker from step to step (a ray that hits nothing shrinks to nothing), so the per-instance
    partition tokens stay valid.
    """

    def __init__(self, stage, num_rays: int, radius: float = 0.04):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        _chase_root(stage)
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=SCAN_RAY_COLOR, emissive_color=SCAN_RAY_COLOR)
        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"{CHASE_ROOT}/scan_rays",
                markers={"hit": sim_utils.SphereCfg(radius=radius, visual_material=material)},
            )
        )
        self._stage = stage
        self._count = num_rays
        self._partitioned = False

    def update(self, hits_w: np.ndarray) -> None:
        """Show the hits ``(N, 3)`` (inf where a ray missed)."""
        hits = np.asarray(hits_w, dtype=np.float64)
        ok = np.all(np.isfinite(hits), axis=1)
        self._markers.visualize(
            translations=np.where(ok[:, None], hits, 0.0).astype(np.float32),
            scales=np.repeat(ok[:, None], 3, axis=1).astype(np.float32),
        )
        if not self._partitioned:
            self._assign_partition()

    def _assign_partition(self) -> None:
        """One partition token per instance: RTX reads the partition of a point instancer per instance."""
        from pxr import Sdf, UsdGeom, Vt

        prim = self._stage.GetPrimAtPath(self._markers.prim_path)
        if not prim.IsValid():
            return
        primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "omni:scenePartition", Sdf.ValueTypeNames.TokenArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(Vt.TokenArray([CHASE_PARTITION] * self._count))
        self._partitioned = True


def ground_below(hits_w: np.ndarray) -> float | None:
    """Median height of the finite scan hits (the ground under the drone), or None."""
    hits = np.asarray(hits_w, dtype=np.float64)
    z = hits[np.all(np.isfinite(hits), axis=1), 2]
    return float(np.median(z)) if z.size else None


__all__ = [
    "CHASE_PARTITION",
    "CHASE_ROOT",
    "DRONE_CAMERA_PARTITION",
    "FollowCamera",
    "ScanRays",
    "ground_below",
    "isolate_chase_view",
    "rim_points",
    "set_focal_length",
    "skirt_mesh",
    "spawn_ground_skirt",
]
