"""Trajectory tracking for ARL-Robot-1: follow a moving reference (circle, figure-eight, Lissajous, random curve).

Built on the built-in position-tracking config (same robot, thruster actions, 100 Hz physics / 10 Hz policy,
empty world), with:

* Command: a reference trajectory around every env origin with randomized shape, size (0.8-2.5 m), peak speed
  (0.3-1.5 m/s), height and vertical wave, plus two look-ahead points (0.5 s and 1.0 s ahead).
* Policy observations (26): position error to the reference (3), velocity error (3), look-ahead points relative
  to the drone (6), orientation quaternion x,y,z,w (4), body linear velocity (3), body angular velocity (3),
  last action (4). Small uniform noise on the errors and body rates during training.
* Rewards: fine and coarse position tracking, velocity tracking, upright attitude, zero yaw, low body rates,
  smooth and small actions, termination penalty.
* Reset: new reference, drone near its start (+-0.3 m, +-15 deg, velocity of the reference +-0.2 m/s).
* Terminations: 12 s time-out, more than 2.5 m from the reference, falling, flipping.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.contrib.drone_arl.mdp.rewards import ang_vel_xyz_exp, yaw_aligned
from isaaclab_tasks.contrib.drone_arl.track_position_state_based.config.arl_robot_1.no_obstacle_env_cfg import (
    NoObstacleEnvCfg,
)
from isaaclab_tasks.contrib.drone_arl.track_position_state_based.config.arl_robot_1.track_position_state_based_env_cfg import (  # noqa: E501
    ArlTrackPositionStateBasedSceneCfg,
)

from drone_lab.tasks.physics import DronePhysicsCfg
from drone_lab.tasks.track_trajectory import mdp

COMMAND_NAME = "reference"


@configclass
class TrajectoryCommandsCfg:
    """Moving reference trajectory with two look-ahead points."""

    reference = mdp.TrajectoryReferenceCommandCfg(asset_name="robot", debug_vis=True)


@configclass
class TrajectoryObservationsCfg:
    """Errors to the reference and its look-ahead points, plus the drone's own state."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group (order preserved)."""

        reference_error = ObsTerm(
            func=mdp.reference_position_error,
            params={"command_name": COMMAND_NAME, "clip": 2.0},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        velocity_error = ObsTerm(
            func=mdp.reference_velocity_error,
            params={"command_name": COMMAND_NAME, "clip": 3.0},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        lookahead = ObsTerm(func=mdp.reference_lookahead, params={"command_name": COMMAND_NAME, "clip": 3.0})
        base_orientation = ObsTerm(func=base_mdp.root_quat_w, params={"make_quat_unique": True})
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TrajectoryEventCfg:
    """Reset near the start of a freshly sampled reference."""

    reset_base = EventTerm(
        func=mdp.reset_near_reference,
        mode="reset",
        params={"command_name": COMMAND_NAME, "position_noise": 0.3, "attitude_noise": 0.26, "velocity_noise": 0.2},
    )


@configclass
class TrajectoryRewardsCfg:
    """Position and velocity tracking, attitude and smoothness terms."""

    tracking_fine = RewTerm(
        func=mdp.reference_tracking_exp, weight=15.0, params={"command_name": COMMAND_NAME, "std": 0.4}
    )
    tracking_coarse = RewTerm(
        func=mdp.reference_tracking_exp, weight=5.0, params={"command_name": COMMAND_NAME, "std": 1.5}
    )
    velocity_tracking = RewTerm(
        func=mdp.reference_velocity_exp, weight=5.0, params={"command_name": COMMAND_NAME, "std": 1.0}
    )
    # a penalty (negative weight); the built-in tracking task uses +1.0 on this term
    flat_orientation_l2 = RewTerm(func=base_mdp.flat_orientation_l2, weight=-1.0)
    yaw_aligned = RewTerm(func=yaw_aligned, weight=2.0, params={"std": 1.0})
    ang_vel_xyz_exp = RewTerm(func=ang_vel_xyz_exp, weight=2.0, params={"std": 5.0})
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.05)
    action_magnitude_l2 = RewTerm(func=base_mdp.action_l2, weight=-0.05)
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-20.0)


@configclass
class TrajectoryTerminationsCfg:
    """Time-out, too far from the reference, crash or flip."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    too_far = DoneTerm(func=mdp.reference_too_far, params={"command_name": COMMAND_NAME, "max_distance": 2.5})
    crash = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": -3.0})
    flipped = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 1.75})


@configclass
class TrackTrajectoryEnvCfg(NoObstacleEnvCfg):
    """Trajectory tracking; see the module docstring."""

    scene: ArlTrackPositionStateBasedSceneCfg = ArlTrackPositionStateBasedSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: TrajectoryObservationsCfg = TrajectoryObservationsCfg()
    commands: TrajectoryCommandsCfg = TrajectoryCommandsCfg()
    events: TrajectoryEventCfg = TrajectoryEventCfg()
    rewards: TrajectoryRewardsCfg = TrajectoryRewardsCfg()
    terminations: TrajectoryTerminationsCfg = TrajectoryTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.sim.physics = DronePhysicsCfg()

    def play_mode(self) -> None:
        """``isaaclab play``: the parent's play settings with at most 16 drones."""
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
