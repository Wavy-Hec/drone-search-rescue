"""Survey-mapping mission for ARL-Robot-1, flown by the built-in position-tracking policy.

``SurveyEnvCfg`` subclasses the configuration the policy was trained with
(``isaaclab_tasks.contrib.drone_arl.track_position_state_based.config.arl_robot_1.no_obstacle_env_cfg:NoObstacleEnvCfg``)
and changes only what the mission needs:

* Policy observations keep the same five terms, order, sizes and (absence of) scaling:
  ``base_link_position`` (3), ``base_orientation`` (4, quaternion x,y,z,w), ``base_lin_vel`` (3, body frame),
  ``base_ang_vel`` (3, body frame), ``last_action`` (4) -> 17 values. Only ``base_link_position`` changes: it is
  the drone position relative to a moving carrot (clamped to +-1 m) instead of relative to the env origin, which
  was the training goal. Observation noise stays off (training also ran with ``enable_corruption=False``).
* Actions (``ThrustActionCfg``: scale 3, offset 3, clip 0..6 per rotor) and the thruster model are unchanged.
* Scene: a procedural 40 m x 40 m survey area (:data:`drone_lab.terrain.SURVEY_TERRAIN_CFG`) replaces the empty
  world, plus a downward mapping ray caster (5 m x 5 m grid at 0.25 m, 441 rays, not observed by the policy)
  and a contact sensor for crash detection.
* Command: a carrot that moves along a lawnmower (or spiral / figure-8 / custom) path.
* Events: deterministic reset at the first waypoint (at rest, level, yaw 0) instead of random starts.
* Terminations: no time-out; crash = collision, flip or falling below the terrain; a time-out 5 s after the
  survey is complete so ``isaaclab play`` loops the mission.
* One environment by default (several fly side-by-side strips), one-hour episodes.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.drone_arl.track_position_state_based.config.arl_robot_1.no_obstacle_env_cfg import (
    NoObstacleEnvCfg,
)
from isaaclab_tasks.contrib.drone_arl.track_position_state_based.config.arl_robot_1.track_position_state_based_env_cfg import (  # noqa: E501
    ArlTrackPositionStateBasedSceneCfg,
    ObservationsCfg,
    RewardsCfg,
)

from drone_lab.tasks.physics import DronePhysicsCfg
from drone_lab.tasks.survey import mdp as survey_mdp
from drone_lab.terrain import SURVEY_TERRAIN_CFG, terrain_area

COMMAND_NAME = "survey_target"
"""Name of the carrot command term."""

MAPPING_GRID = patterns.GridPatternCfg(resolution=0.25, size=(5.0, 5.0))
"""Default mapping footprint: 21 x 21 = 441 parallel downward rays."""


##
# Scene
##


@configclass
class SurveySceneCfg(ArlTrackPositionStateBasedSceneCfg):
    """Survey area, drone (set by the parent config), mapping ray caster and contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SURVEY_TERRAIN_CFG,
        use_terrain_origins=False,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # None keeps the generator's height colouring (vertex colours) visible
        visual_material=None,
        debug_vis=False,
    )

    mapping_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.1)),
        # "world": a gimballed scanner; the footprint stays axis-aligned and the rays vertical when the drone tilts
        ray_alignment="world",
        pattern_cfg=MAPPING_GRID,
        mesh_prim_paths=["/World/ground"],
        max_distance=60.0,
        global_world_only=True,
        debug_vis=False,
    )

    # direct child bodies only, as in the built-in drone navigation task (navigation_env_cfg.py:99-104)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, debug_vis=False)


##
# MDP
##


@configclass
class SurveyCommandsCfg:
    """Moving target along the survey path."""

    survey_target = survey_mdp.SurveyCarrotCommandCfg(asset_name="robot", debug_vis=True)


@configclass
class SurveyObservationsCfg:
    """Same policy group as the training task, with the position measured from the carrot."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Inherits the term order of the training config; overriding a field keeps its position."""

        base_link_position = ObsTerm(
            func=survey_mdp.position_relative_to_target,
            params={"command_name": COMMAND_NAME, "max_offset": 1.0, "per_axis_limit": (1.0, 1.0, 1.0)},
        )

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = True
            for name in ("base_link_position", "base_orientation", "base_lin_vel", "base_ang_vel", "last_action"):
                getattr(self, name).noise = None

    policy: PolicyCfg = PolicyCfg()


@configclass
class SurveyEventCfg:
    """Deterministic start at the first waypoint."""

    reset_base = EventTerm(
        func=survey_mdp.reset_root_to_path_start,
        mode="reset",
        params={"command_name": COMMAND_NAME, "position_offset": (0.0, 0.0, 0.0), "yaw": 0.0},
    )


@configclass
class SurveyRewardsCfg(RewardsCfg):
    """Training rewards with the goal term measured to the carrot (rewards are only logged here)."""

    distance_to_goal_exp = RewTerm(
        func=survey_mdp.target_distance_exp,
        weight=25.0,
        params={"command_name": COMMAND_NAME, "std": 1.5},
    )


@configclass
class SurveyTerminationsCfg:
    """Crash terminations only; the survey is never cut by a time limit."""

    time_out = None
    crash = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": -5.0})
    collision = DoneTerm(
        func=base_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
    )
    flipped = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 1.75})
    survey_complete = DoneTerm(
        func=survey_mdp.survey_complete, params={"command_name": COMMAND_NAME, "hold_s": 5.0}, time_out=True
    )


##
# Environment
##


def overview_camera(area_cfg=SURVEY_TERRAIN_CFG) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """High oblique camera (eye, lookat) that frames the whole survey area."""
    area = terrain_area(area_cfg)
    cx, cy = area.center
    size = max(area.width, area.height)
    return (cx - 0.55 * size, cy - 0.85 * size, 0.7 * size), (cx, cy + 0.05 * size, 0.0)


@configclass
class SurveyEnvCfg(NoObstacleEnvCfg):
    """Survey mission; see the module docstring."""

    scene: SurveySceneCfg = SurveySceneCfg(num_envs=1, env_spacing=2.0)
    observations: SurveyObservationsCfg = SurveyObservationsCfg()
    commands: SurveyCommandsCfg = SurveyCommandsCfg()
    events: SurveyEventCfg = SurveyEventCfg()
    rewards: SurveyRewardsCfg = SurveyRewardsCfg()
    terminations: SurveyTerminationsCfg = SurveyTerminationsCfg()

    def __post_init__(self):
        # parent: decimation 10, sim dt 0.01 s (policy at 10 Hz), PhysX, ARL-Robot-1 with thruster dt = sim dt
        super().__post_init__()
        self.episode_length_s = 3600.0
        self.sim.physics = DronePhysicsCfg()
        # spawn high above the terrain; the reset event moves the drone to its path start
        self.scene.robot.init_state = replace(self.scene.robot.init_state, pos=(0.0, 0.0, 20.0))
        self.scene.contact_forces.update_period = self.sim.dt
        # camera hint for --viz kit and for video capture
        eye, lookat = overview_camera()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=eye, lookat=lookat)
