"""Turn an Isaac Lab velocity-tracking locomotion config into the rescue task (robot-agnostic).

:func:`apply_rescue_task` takes any config built on
``isaaclab_tasks.core.velocity.velocity_env_cfg.LocomotionVelocityRoughEnvCfg`` (the robot, its actions and its
policy observations stay exactly as trained) and changes only what the finale needs:

* **Terrain:** the drone's survey area (:data:`drone_lab.terrain.SURVEY_TERRAIN_CFG`, same seed, same mesh as in
  the drone mission), without terrain origins or curriculum.
* **Site:** the dressed disaster site of the drone mission (:func:`add_sar_site`, same seed: the same people,
  props and ground materials the drone flew over), or the bare terrain with one person at the goal
  (:func:`use_plain_terrain`).
* **Command:** the ``base_velocity`` term becomes :class:`~drone_lab.tasks.rescue.mdp.PathFollowCommandCfg`
  (pure pursuit on the planned route, clipped to the replaced command's ranges). The policy observation
  ``velocity_commands`` reads it unchanged, so a checkpoint of the base task drives the robot as is.
* **Observations:** same terms, order, scales and clipping; noise off (as in ``isaaclab play``).
* **Events:** deterministic: no mass / centre-of-mass randomisation, no pushes, joints at their defaults, and the
  robot starts on the route (after a fall it respawns where it fell, see ``respawn_on_fall``).
* **Terminations:** the base task's (time-out, base contact) plus a flip check and ``goal_reached`` (a time-out
  a few seconds after arriving, so ``isaaclab play`` loops the walk).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.materials import MdlFileCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR

from drone_lab.tasks.rescue import mdp as rescue_mdp
from drone_lab.terrain import SURVEY_TERRAIN_CFG

COMMAND_NAME = "base_velocity"
"""The replaced command keeps the velocity task's term name (the policy observation reads it by name)."""

SITE_PRIM_PATH = "/World/sar"
"""Root prim of the dressed SAR site (the same path as in the drone mission)."""

PERSON_USD = f"{ISAAC_NUCLEUS_DIR}/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd"
"""Isaac People character used for the person the robot walks to (metres, Z-up, origin between the feet, facing -Y,
T-pose, no colliders)."""

GROUND_MDL = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA/Materials/Base/Natural/Soil_Rocky.mdl"
"""Ground material of the plain survey terrain (the dressed SAR site brings its own per-tile materials)."""

DEFAULT_START_XY = (-23.0, -18.0)
"""Default start: the landing pad on the flat border (the survey starts at the area's south-west corner)."""

DEFAULT_GOAL_XY = (-4.5, -12.5)
"""Default person on the plain terrain: in a gap between collapsed buildings of the south blocks tile."""


@configclass
class WorldProjectedMdlCfg(MdlFileCfg):
    """OmniPBR MDL with its texture projected in world space, so neighbouring terrain faces line up."""

    world_or_object: bool | None = True


def _quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def person_asset_cfg(
    xy: tuple[float, float],
    z: float = 0.0,
    *,
    lying: bool = False,
    yaw: float = 0.0,
    usd_path: str = PERSON_USD,
) -> AssetBaseCfg:
    """A static, collision-free person at ``(x, y)`` standing on height ``z`` (or lying on their back, body
    centred on ``(x, y)``)."""
    if lying:
        # the characters face -Y in a T-pose: -90 deg about X lays them on their back (front up, arms flat, head
        # along +Y from the feet), then the yaw. The feet (the asset origin) sit 0.9 m from the body centre, and
        # the origin rises by half the 0.35-0.43 m front-to-back depth so the back rests on the ground.
        quat = _quat_xyzw_from_rpy(-math.pi / 2, 0.0, yaw)
        head = (-math.sin(yaw), math.cos(yaw))
        pos = (xy[0] - 0.9 * head[0], xy[1] - 0.9 * head[1], z + 0.19)
    else:
        quat = _quat_xyzw_from_rpy(0.0, 0.0, yaw)
        pos = (xy[0], xy[1], z)
    return AssetBaseCfg(
        prim_path="/World/Person",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=quat),
    )


def add_sar_site(cfg, seed: int = 0) -> None:
    """Dress the terrain of a rescue config as the drone mission's disaster site (in place).

    Adds ``scene.sar_site`` (:class:`~drone_lab.tasks.rescue.site.RescueSiteCfg`: people, props with their
    colliders, landing pad, ground materials; placed on the scene's own terrain generator, so the same seed gives
    the same site as in the mission), the mission's sky and sun lights, and removes the single goal person (the site
    has its people).
    """
    from drone_lab.sar.scene import sky_light_cfg, sun_light_cfg
    from drone_lab.tasks.rescue.site import RescueSiteCfg

    terrain = cfg.scene.terrain
    # the site textures the terrain per tile (in the mission the terrain has no material of its own either)
    terrain.visual_material = None
    site = RescueSiteCfg(
        seed=int(seed), terrain_generator=terrain.terrain_generator, terrain_prim_path=terrain.prim_path
    )
    cfg.scene.sar_site = AssetBaseCfg(prim_path=SITE_PRIM_PATH, spawn=site)
    # the same lights as the mission scene (the sun is aligned with the HDRI's own sun: one shadow per object)
    cfg.scene.sky_light = sky_light_cfg(intensity=800.0)
    cfg.scene.sun_light = sun_light_cfg(intensity=2000.0, elevation_deg=43.0, azimuth_deg=136.0)
    cfg.scene.person = None


def use_plain_terrain(cfg, person: AssetBaseCfg | None = None) -> None:
    """The bare survey terrain in rocky soil, with ``person`` (if given) at the goal; removes a SAR site (in place)."""
    for name in ("sar_site", "sun_light"):
        if getattr(cfg.scene, name, None) is not None:
            setattr(cfg.scene, name, None)
    cfg.scene.terrain.visual_material = WorldProjectedMdlCfg(
        mdl_path=GROUND_MDL, project_uvw=True, texture_scale=(0.25, 0.25)
    )
    if person is not None or getattr(cfg.scene, "person", None) is not None:
        cfg.scene.person = person


def apply_rescue_task(
    cfg,
    *,
    terrain_generator=SURVEY_TERRAIN_CFG,
    num_envs: int = 1,
    episode_length_s: float = 300.0,
    person: AssetBaseCfg | None = None,
    site_seed: int | None = None,
    flip_limit_rad: float = 1.2,
    goal_hold_s: float = 3.0,
):
    """Modify a velocity-task config in place into the rescue task (see the module docstring); returns it.

    ``site_seed`` dresses the terrain as the SAR site of that seed (:func:`add_sar_site`, ``person`` is then
    ignored); None keeps the plain survey terrain with ``person`` (if given) at the goal.
    """
    import isaaclab.envs.mdp as base_mdp
    from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

    # ---------------------------------------------------------------------------------------------- scene
    cfg.scene.num_envs = num_envs
    terrain = cfg.scene.terrain
    terrain.terrain_type = "generator"
    terrain.terrain_generator = replace(terrain_generator)
    terrain.use_terrain_origins = False
    terrain.max_init_terrain_level = None
    if site_seed is not None:
        add_sar_site(cfg, site_seed)
    else:
        use_plain_terrain(cfg, person)
    # --------------------------------------------------------------------------------------------- command
    old = cfg.commands.base_velocity
    ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=tuple(old.ranges.lin_vel_x),
        lin_vel_y=tuple(old.ranges.lin_vel_y),
        ang_vel_z=tuple(old.ranges.ang_vel_z),
    )
    cfg.commands.base_velocity = rescue_mdp.PathFollowCommandCfg(
        asset_name=old.asset_name, ranges=ranges, debug_vis=True
    )
    # -------------------------------------------------------------------------------------------- observations
    for group_name in getattr(cfg.observations, "__dataclass_fields__", {}):
        group = getattr(cfg.observations, group_name)
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    # -------------------------------------------------------------------------------------------- events
    ev = cfg.events
    for name in ("add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    ev.reset_base = EventTerm(func=rescue_mdp.reset_root_on_route, mode="reset", params={"command_name": COMMAND_NAME})
    if getattr(ev, "reset_robot_joints", None) is not None:
        ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        if "velocity_range" in ev.reset_robot_joints.params:
            ev.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    # ------------------------------------------------------------------------------------------ terminations
    cfg.terminations.flipped = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": flip_limit_rad})
    cfg.terminations.goal_reached = DoneTerm(
        func=rescue_mdp.route_complete, params={"command_name": COMMAND_NAME, "hold_s": goal_hold_s}, time_out=True
    )
    # ------------------------------------------------------------------------------------------ curriculum
    if getattr(cfg, "curriculum", None) is not None and hasattr(cfg.curriculum, "terrain_levels"):
        cfg.curriculum.terrain_levels = None
    if terrain.terrain_generator is not None:
        terrain.terrain_generator.curriculum = False
    cfg.episode_length_s = episode_length_s
    return cfg
