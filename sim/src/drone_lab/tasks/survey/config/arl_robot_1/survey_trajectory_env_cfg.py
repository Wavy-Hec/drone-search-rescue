"""Survey mission for ARL-Robot-1, flown by a trajectory-tracking policy instead of the built-in hover policy.

``SurveyTrajectoryEnvCfg`` is :class:`~.survey_env_cfg.SurveyEnvCfg` with the policy observations of the task the
pilot was trained on, ``DroneLab-TrackTrajectory-ARL-Robot-1-v0``
(:mod:`drone_lab.tasks.track_trajectory.config.arl_robot_1.track_trajectory_env_cfg`):

* Policy observations (26): position error to the reference (3), velocity error (3), two look-ahead points
  relative to the drone (6), orientation quaternion x,y,z,w (4), body linear velocity (3), body angular velocity
  (3), last action (4). Same terms, order and clips as in training; the reference is the survey carrot, whose
  command term provides ``ref_pos_w``, ``ref_vel_w`` and ``lookahead_w``. Observation noise is off.
* Actions, thruster model and timing are the survey task's, which are the training task's (both configs subclass
  the built-in ``NoObstacleEnvCfg``).
* Carrot: it speeds up by at most 1 m/s^2, also after waiting for the drone, so the reference velocity does not
  jump up along a line (it slows down faster only when the drone falls behind), and it may lead the drone by up
  to 1.5 m (training ends an episode at 2.5 m and clips the observed error at 2 m).

Scene, events, rewards and terminations are the survey task's.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from drone_lab.tasks.survey.mdp import SurveyCarrotCommandCfg
from drone_lab.tasks.track_trajectory.config.arl_robot_1.track_trajectory_env_cfg import TrajectoryObservationsCfg

from .survey_env_cfg import COMMAND_NAME, SurveyEnvCfg


@configclass
class SurveyTrajectoryObservationsCfg:
    """The policy group of the trajectory-tracking task, with the survey carrot as the reference."""

    @configclass
    class PolicyCfg(TrajectoryObservationsCfg.PolicyCfg):
        """Inherits the terms, their order and their clips from the training config."""

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            for name in ("reference_error", "velocity_error", "lookahead"):
                getattr(self, name).params["command_name"] = COMMAND_NAME
            for name in (
                "reference_error",
                "velocity_error",
                "lookahead",
                "base_orientation",
                "base_lin_vel",
                "base_ang_vel",
                "last_action",
            ):
                getattr(self, name).noise = None

    policy: PolicyCfg = PolicyCfg()


def configure_trajectory_carrot(carrot: SurveyCarrotCommandCfg) -> None:
    """Carrot settings for a trajectory-tracking pilot.

    The policy trained with position errors of up to 2.5 m before termination and an observation clip of 2 m, so
    the carrot may lead by 1.5 m (it starts slowing down at 1.0 m), and it speeds up by at most 1 m/s^2.
    """
    carrot.max_accel = 1.0
    carrot.max_lead = 1.5
    carrot.slow_lead = 1.0


@configclass
class SurveyTrajectoryEnvCfg(SurveyEnvCfg):
    """Survey mission flown by a trajectory-tracking policy; see the module docstring."""

    observations: SurveyTrajectoryObservationsCfg = SurveyTrajectoryObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_trajectory_carrot(self.commands.survey_target)
