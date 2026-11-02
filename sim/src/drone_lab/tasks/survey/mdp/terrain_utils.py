"""Read the imported terrain mesh back from the USD stage (needs ``pxr``; works once the scene exists)."""

from __future__ import annotations

import numpy as np


def terrain_mesh_from_stage(prim_path: str = "/World/ground") -> tuple[np.ndarray, np.ndarray] | None:
    """World-frame vertices and triangle faces of the first ``Mesh`` prim under ``prim_path``.

    Mirrors how the ray caster reads its target mesh
    (``isaaclab/sensors/ray_caster/base_ray_caster.py``, ``_initialize_warp_meshes``), so the result is exactly
    the geometry the mapping rays hit.

    Returns:
        ``(vertices (V, 3) float64, faces (F, 3) int64)`` or None when no mesh prim exists.
    """
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    prim = sim_utils.get_first_matching_child_prim(prim_path, lambda p: p.GetTypeName() == "Mesh")
    if prim is None or not prim.IsValid():
        return None
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    points = points @ world[:3, :3].T + world[:3, 3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if np.all(counts == 3):
        faces = indices.reshape(-1, 3)
    else:  # fan-triangulate polygons
        faces_list = []
        start = 0
        for c in counts:
            poly = indices[start : start + c]
            for k in range(1, c - 1):
                faces_list.append((poly[0], poly[k], poly[k + 1]))
            start += c
        faces = np.asarray(faces_list, dtype=np.int64).reshape(-1, 3)
    return points, faces
