"""Physics presets shared by the drone tasks."""

from isaaclab_physx.physics import PhysxCfg

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class DronePhysicsCfg(PresetCfg):
    """Physics presets. The multirotor asset is PhysX-only in Isaac Lab 3.0, so Isaac Sim PhysX is the only one.

    ``isaaclab_contrib.assets.Multirotor`` subclasses ``isaaclab_physx.assets.articulation.Articulation`` and
    applies its rotor wrench through the PhysX view, and the built-in drone task notes that ``ThrusterCfg`` has no
    Newton execution path. Declaring the preset makes ``physics=isaacsim_physx`` resolve (the built-in task has no
    presets, so any ``physics=`` selector errors there); ``physics=newton_mjwarp`` fails early with a clear message.
    """

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    default = isaacsim_physx
