"""Rescue finale with Hector's custom quadruped (``quadruped_lab``) on the drone mission's disaster site.

The base is ``quadruped_lab``'s ``CustomQuadRoughEnvCfg`` (task ``Isaac-Velocity-Rough-Custom-Quad-v0``): same robot
(``robot.usd``, DC motors), actions (joint position, scale 0.25), policy observations (base velocities, projected
gravity, velocity command, joint state, last action, 187-ray height scan from the trunk = 235 values), command ranges,
decimation 4 x 5 ms. :func:`~drone_lab.tasks.rescue.rescue_env_cfg.apply_rescue_task` swaps in the survey terrain
dressed as the SAR site (seed 0, the mission's demo site), the path-following command and deterministic events.

Physics is Isaac Sim PhysX only, the backend of the drone mission, with the two execution settings under which the
converted 2.x checkpoint walks as it did in Isaac Lab 2.3.2 (``quadruped_lab/PORT_NOTES.md``, section 14.2):

* ``sim.use_newton_actuators = False``: the legacy actuator path feeds the DC-motor PD fresh joint positions;
* ``enable_external_forces_every_iteration = False``: the 2.3.2 TGS setting.
"""

from __future__ import annotations

from quadruped_lab.tasks.locomotion.config.custom_quad.rough_env_cfg import CustomQuadRoughEnvCfg

from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from drone_lab.tasks.rescue.rescue_env_cfg import apply_rescue_task

SITE_SEED = 0
"""SAR site seed of the drone mission's demo site (``env.scene.sar_site.spawn.seed=<n>`` picks another one)."""

PARITY_PHYSX_CFG = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15, enable_external_forces_every_iteration=False)
"""Isaac Sim PhysX as in the velocity base task, with the TGS setting of Isaac Lab 2.3.2."""


@configclass
class RescuePhysicsCfg(PresetCfg):
    """Isaac Sim PhysX is the only physics preset: ``physics=isaacsim_physx`` resolves, Newton or ``physx`` do not."""

    isaacsim_physx = PARITY_PHYSX_CFG
    default = PARITY_PHYSX_CFG


@configclass
class RescueCustomQuadEnvCfg(CustomQuadRoughEnvCfg):
    """The custom quadruped walks the planned route to a person on the SAR site; see the module docstring."""

    def __post_init__(self):
        super().__post_init__()
        apply_rescue_task(self, num_envs=1, episode_length_s=300.0, site_seed=SITE_SEED)
        self.sim.physics = RescuePhysicsCfg()
        self.sim.use_newton_actuators = False
        # the training resets drop the robot from its init height (0.719 m) onto the terrain: respawn the same way
        self.commands.base_velocity.spawn_height_offset = 0.0

    def play_mode(self) -> None:
        """``isaaclab play``: only the shared defaults (no corruption, <= 50 envs).

        The velocity base's ``play_mode`` would shrink the terrain grid to 5 x 5 tiles; the survey area must stay
        exactly as the drone mapped it, so it is skipped on purpose.
        """
        from isaaclab.envs import ManagerBasedRLEnvCfg

        ManagerBasedRLEnvCfg.play_mode(self)
