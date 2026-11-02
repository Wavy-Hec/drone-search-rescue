# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026 Hector Lugo
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain velocity tracking for the custom quadruped (height-scan policy).

Ported from the Isaac Lab 2.3.2 task ``custom_quadruped/rough_env_cfg.py``. The policy observation interface (term
order, dims, noise, clipping) is unchanged, so the converted 2.x checkpoint
``checkpoints_2x/height_scanner_model_rslrl5.pt`` (235 obs -> 12 actions) runs with this config.

Physics defaults to Isaac Sim PhysX with the execution settings the checkpoint was trained with in Isaac Lab 2.x:
Isaac Lab's actuator path (``sim.use_newton_actuators=False``) and the 2.3.2 TGS setting
(``enable_external_forces_every_iteration=False``). Newton keeps the Isaac Lab 3.0 defaults. The 3.0 execution
defaults on Isaac Sim PhysX are one override away:
``env.sim.use_newton_actuators=true env.sim.physics.enable_external_forces_every_iteration=true``.

The 2.x ``CustomQuadRoughEnvCfg_PLAY`` class is gone: ``isaaclab play`` applies
:meth:`~isaaclab_tasks.core.velocity.velocity_env_cfg.LocomotionVelocityRoughEnvCfg.play_mode`, which makes the same
changes (<=50 envs, 5x5 terrain grid without curriculum, random spawn level, no corruption, no push/force events).
"""

from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.utils import PresetCfg

from quadruped_lab.assets.custom_quad import CUSTOM_QUAD_CFG, CUSTOM_QUAD_NEWTON_CFG


@configclass
class CustomQuadRobotCfg(PresetCfg):
    """The robot per physics backend, picked by the ``physics=`` selector (like 3.0's ``ShadowHandRobotCfg``).

    Isaac Sim PhysX (the default, also ``physx`` / ``ovphysx``) keeps the canonical ``robot.usd`` the 2.x checkpoints
    were trained on, with backend-native ordering. The Newton presets use ``robot_newton.usd``, whose colliders Newton
    can import, and present joints and bodies in the Isaac Sim PhysX order (Newton's own order is per leg), so one
    policy runs on both backends. Only the asset's collision layout and the ordering differ; everything else is shared.
    """

    default = replace(CUSTOM_QUAD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    newton_mjwarp = replace(CUSTOM_QUAD_NEWTON_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    newton_kamino = newton_mjwarp


@configclass
class UseNewtonActuatorsCfg(PresetCfg):
    """``sim.use_newton_actuators`` per physics backend, picked by the ``physics=`` selector.

    Isaac Sim PhysX (the default, also ``isaacsim_physx`` and the ``physx`` alias) runs the DC motors on Isaac Lab's
    actuator path, as Isaac Lab 2.x did. On Isaac Sim PhysX, Isaac Lab 3.0's native actuator path feeds the DC-motor
    PD joint positions that are one to three physics substeps old, and the converted 2.x policy tracks its commands
    measurably worse with it (PORT_NOTES.md). Newton and OvPhysX keep the 3.0 default, native actuators.
    """

    default: bool = False
    ovphysx: bool = True
    newton_mjwarp: bool = True
    newton_kamino: bool = True


@configclass
class CustomQuadRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Custom quadruped on the stock rough-terrain generator with a 187-ray height scan."""

    def __post_init__(self) -> None:
        """Apply the custom quadruped's robot, terrain, MDP and physics settings to the velocity base task."""
        super().__post_init__()

        # physics: the 3.0 velocity base defaults to Newton MJWarp; default to Isaac Sim PhysX, the backend of the 2.x
        # checkpoints. Newton stays selectable on the command line with ``physics=newton_mjwarp``.
        physics = self.sim.physics
        # TGS applies external forces once per step, as in Isaac Lab 2.3.2 (the 3.0 default applies them on every
        # position iteration, which makes the 2.x policy track its commands worse)
        physics.isaacsim_physx.enable_external_forces_every_iteration = False
        physics.default = physics.isaacsim_physx
        # ``physics=physx`` is an auto preset that picks OvPhysX for kit-less runs and Isaac Sim PhysX for Kit runs,
        # i.e. different backends for headless train and ``--viz kit`` play. Alias it to Isaac Sim PhysX so a stray
        # ``physics=physx`` cannot switch backends.
        physics.physx = physics.isaacsim_physx
        # actuators: Isaac Lab's actuator path on Isaac Sim PhysX, native actuators on Newton (UseNewtonActuatorsCfg)
        self.sim.use_newton_actuators = UseNewtonActuatorsCfg()

        # scene: robot.usd on PhysX, robot_newton.usd (+ PhysX joint/body order) on Newton, see CustomQuadRobotCfg
        self.scene.robot = CustomQuadRobotCfg()
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # events
        self.events.push_robot = None
        # 3.0 changed this term's defaults to a multiplicative log-uniform +-25 %; with only the range overridden it
        # raises "log-uniform bounds must be > 0". Keep the 2.x semantics: add U(-1, 3) kg to the trunk.
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["operation"] = "add"
        self.events.add_base_mass.params["distribution"] = "uniform"
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.base_com.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["trunk", ".*_thigh"]
