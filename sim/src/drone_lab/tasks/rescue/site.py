"""The drone mission's dressed SAR site, spawned for a robot that walks on it.

The site textures the terrain by binding MDL materials to face subsets (``UsdGeom.Subset``, family
``materialBind``) of the terrain mesh. On Isaac Sim PhysX a collision mesh with such subsets stops colliding with
the robot, which then falls through the ground (binding the terrain's physics material to every subset does not
help; the drone never touches the terrain, so the mission is not affected). :func:`spawn_rescue_site` spawns the
site unchanged and then moves the textures to a render-only copy of the terrain mesh
(:func:`move_ground_materials_to_visual_mesh`): the collision mesh keeps its geometry, physics material and
collider, loses the subsets and is hidden; the copy has the same geometry, the subsets and no collider.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from isaaclab.utils import configclass

from drone_lab.sar.scene import SarSiteCfg

logger = logging.getLogger(__name__)

VISUAL_GROUND_SUBPATH = "ground_visual"
"""Render-only copy of the terrain mesh, under the site root."""


@configclass
class RescueSiteCfg(SarSiteCfg):
    """:class:`~drone_lab.sar.scene.SarSiteCfg` (same fields, same site) with a walkable textured terrain."""

    func: Callable | str = "drone_lab.tasks.rescue.site:spawn_rescue_site"


def move_ground_materials_to_visual_mesh(terrain_prim_path: str, visual_path: str, stage=None) -> int:
    """Move the material face subsets of the terrain mesh onto a render-only copy of the mesh at ``visual_path``.

    The copy gets the mesh's authored geometry and display attributes (no physics, no transform ops) and the
    mesh's world transform; each subset is re-created on it with its directly bound material. The terrain mesh
    loses its subsets and is made invisible, so the collider is unchanged and only the copy is rendered.

    Args:
        terrain_prim_path: Terrain prim; its first ``Mesh`` descendant is the collision mesh.
        visual_path: Path of the render-only copy.
        stage: USD stage (default: the current one).

    Returns:
        The number of subsets moved (0: nothing to do, no copy is made).
    """
    from pxr import Usd, UsdGeom, UsdShade

    import isaaclab.sim as sim_utils

    stage = stage if stage is not None else sim_utils.get_current_stage()
    mesh_prim = sim_utils.get_first_matching_child_prim(
        terrain_prim_path, lambda p: p.GetTypeName() == "Mesh", stage=stage
    )
    if mesh_prim is None or not mesh_prim.IsValid():
        return 0
    subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
    if not subsets:
        return 0
    visual = UsdGeom.Mesh.Define(stage, visual_path)
    skip = ("physics:", "physx", "material:binding", "xformOp", "visibility")
    for attr in mesh_prim.GetAuthoredAttributes():
        name = attr.GetName()
        if name.startswith(skip) or attr.Get() is None:
            continue
        visual.GetPrim().CreateAttribute(name, attr.GetTypeName(), custom=attr.IsCustom()).Set(attr.Get())
    world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    visual.AddTransformOp().Set(world)
    visual_binding = UsdShade.MaterialBindingAPI.Apply(visual.GetPrim())
    for subset in subsets:
        material = UsdShade.MaterialBindingAPI(subset.GetPrim()).GetDirectBinding().GetMaterial()
        copy = UsdGeom.Subset.CreateGeomSubset(
            visual,
            subset.GetPrim().GetName(),
            subset.GetElementTypeAttr().Get(),
            subset.GetIndicesAttr().Get(),
            subset.GetFamilyNameAttr().Get(),
        )
        if material:
            UsdShade.MaterialBindingAPI.Apply(copy.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        stage.RemovePrim(subset.GetPrim().GetPath())
    visual_binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)
    UsdGeom.Imageable(mesh_prim).MakeInvisible()
    return len(subsets)


def spawn_rescue_site(prim_path: str, cfg: RescueSiteCfg, translation=None, orientation=None, **kwargs):
    """Spawn the SAR site (:func:`drone_lab.sar.scene.spawn_sar_site`), then make its textured terrain walkable."""
    from drone_lab.sar.scene import spawn_sar_site

    root = spawn_sar_site(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    moved = move_ground_materials_to_visual_mesh(cfg.terrain_prim_path, f"{prim_path}/{VISUAL_GROUND_SUBPATH}")
    logger.info("Moved %d ground-material face subsets to a render-only copy of the terrain mesh.", moved)
    return root
