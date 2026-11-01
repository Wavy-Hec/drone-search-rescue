# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for Hector's custom 12-DOF quadruped.

Ported from the Isaac Lab 2.3.2 task ``custom_quadruped/custom_quad.py``. Changes for Isaac Lab 3.0:

* The USD path is package-relative (``QUADRUPED_LAB_ASSETS_DIR``) instead of a hard-coded ``~/IsaacLab`` path.
* The deprecated ``sim_utils.RigidBodyPropertiesCfg`` / ``ArticulationRootPropertiesCfg`` (removal planned for 3.2)
  are replaced by schema fragments that author the same USD attributes.
* ``effort_limit`` / ``velocity_limit`` are renamed to ``actuator_effort_limit`` / ``actuator_velocity_limit``.
  ``joint_effort_limit`` is set explicitly: 3.0 no longer widens the solver effort limit of explicit actuators to
  1e9, so it is pinned to the actuator limit (the asset authors 500 N*m, so it never clips below 45 N*m).

New in 3.0: :data:`CUSTOM_QUAD_NEWTON_CFG`, the same robot for the Newton backend (``robot_newton.usd``, PhysX
joint/body order). The tasks select it with ``physics=newton_mjwarp`` / ``physics=newton_kamino``.
"""

from math import pi

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.utils import replace
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from . import QUADRUPED_LAB_ASSETS_DIR

CUSTOM_QUAD_USD_DIR = QUADRUPED_LAB_ASSETS_DIR / "robots" / "custom_quad"
"""Folder holding ``robot.usd`` and the ``configuration/`` layers it references (relative paths)."""

CUSTOM_QUAD_USD_PATH = CUSTOM_QUAD_USD_DIR / "robot.usd"
"""Canonical asset: the 2026-01-15 copy from ``~/quadruped_rough_terrain`` (predates the checkpoints)."""

CUSTOM_QUAD_FEB23_USD_PATH = CUSTOM_QUAD_USD_DIR / "robot_feb23.usd"
"""Unused alternative: the 2026-02-23 re-save from ``~/IsaacLab``. Its root layer differs from the canonical one only
in the saved viewport camera (``customLayerData``); the composed physics is identical. Kept for an A/B test via
``env.scene.robot.spawn.usd_path=<path>``."""

CUSTOM_QUAD_NEWTON_USD_PATH = CUSTOM_QUAD_USD_DIR / "robot_newton.usd"
"""Newton variant of :data:`CUSTOM_QUAD_USD_PATH`, written by ``scripts/make_newton_asset.py``.

``robot.usd`` authors ``PhysicsCollisionAPI`` + ``PhysicsMeshCollisionAPI`` on the Xform above each collision mesh.
Isaac Sim PhysX accepts that layout; Newton's USD importer only builds shapes from Gprims, skips all 17 colliders and
the scene does not build (the contact sensor finds no shapes). ``robot_newton.usd`` sublayers ``robot.usd`` and
selects a ``Physics = "Newton"`` variant whose payload moves the two APIs onto the Mesh prims. Bodies, masses,
joints, drives, limits and collision meshes are composed from the original layers, which stay untouched."""

CUSTOM_QUAD_PHYSX_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)
"""Isaac Sim PhysX joint order of the asset (breadth-first; measured on the GPU). The converted 2.x checkpoint's
``joint_pos`` / ``joint_vel`` / ``actions`` entries follow it. Newton lists the joints per leg (FL hip, thigh, calf,
FR hip, ...)."""

CUSTOM_QUAD_PHYSX_BODY_ORDER = (
    "trunk",
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)
"""Isaac Sim PhysX body order of the asset (breadth-first; measured on the GPU). Newton lists the bodies per leg."""

CUSTOM_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(CUSTOM_QUAD_USD_PATH),
        activate_contact_sensors=True,
        # same attributes as the 2.x RigidBodyPropertiesCfg, applied to every rigid body of the asset
        rigid_props=[
            UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ],
        # same attributes as the 2.x ArticulationRootPropertiesCfg, plus the Newton self-collision switch
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.719),
        joint_pos={
            ".*L_hip_joint": pi / 45,
            ".*R_hip_joint": pi / 45,
            "F[L,R]_thigh_joint": -5 * pi / 18,
            "R[L,R]_thigh_joint": -5 * pi / 18,
            ".*_calf_joint": 5 * pi / 9,
        },
        joint_vel={".*": 0.0},
    ),
    # The soft joint position limits are scaled by this factor to specify a safety region within the simulated
    # joint position limits. They are not used by the simulation, only by learning terms (e.g. termination/rewards).
    soft_joint_pos_limit_factor=0.9,
    # joint_ordering / body_ordering stay None (backend-native order). The 2.x checkpoints were trained with
    # Isaac Sim PhysX's native order; the "physx" alias means OvPhysX tensor-view order and may differ.
    # CUSTOM_QUAD_NEWTON_CFG pins both to that Isaac Sim PhysX order as explicit name lists.
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            actuator_effort_limit=45.0,  # 2.x: effort_limit=45 (A1 default 33.5)
            joint_effort_limit=45.0,  # 2.x: implicit 1e9 solver limit for explicit actuators
            saturation_effort=45.0,  # A1 default 33.5
            actuator_velocity_limit=21.0,  # 2.x: velocity_limit=21.0 (DC-motor torque-speed curve only)
            stiffness=60.0,  # A1 default 25.0
            damping=1.5,  # A1 default 0.5
            friction=0.0,
        ),
    },
)
"""Configuration of the custom quadruped with a DC-motor actuator model on all 12 leg joints."""

CUSTOM_QUAD_NEWTON_CFG = replace(
    CUSTOM_QUAD_CFG,
    spawn=replace(CUSTOM_QUAD_CFG.spawn, usd_path=str(CUSTOM_QUAD_NEWTON_USD_PATH)),
    joint_ordering=list(CUSTOM_QUAD_PHYSX_JOINT_ORDER),
    body_ordering=list(CUSTOM_QUAD_PHYSX_BODY_ORDER),
)
"""Newton (MJWarp / Kamino) variant of :data:`CUSTOM_QUAD_CFG`: :data:`CUSTOM_QUAD_NEWTON_USD_PATH`, with joints and
bodies presented in the Isaac Sim PhysX order. Observations and actions then use the same layout on both backends: the
converted 2.x checkpoint replays on Newton, and a policy trained on either backend has the layout the other expects.
Everything else equals :data:`CUSTOM_QUAD_CFG`."""
