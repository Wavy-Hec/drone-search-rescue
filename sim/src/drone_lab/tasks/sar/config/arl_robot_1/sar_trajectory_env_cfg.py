"""Search-and-rescue mission for ARL-Robot-1, flown by a trajectory-tracking policy.

``SarTrajectoryEnvCfg`` is :class:`~.sar_env_cfg.SarEnvCfg` (the dressed site and the camera) with the policy
observations and the carrot settings of
:class:`~drone_lab.tasks.survey.config.arl_robot_1.survey_trajectory_env_cfg.SurveyTrajectoryEnvCfg`: the pilot is
a policy trained on ``DroneLab-TrackTrajectory-ARL-Robot-1-v0`` (26 observations), not the built-in hover policy.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from drone_lab.tasks.survey.config.arl_robot_1.survey_trajectory_env_cfg import (
    SurveyTrajectoryObservationsCfg,
    configure_trajectory_carrot,
)

from .sar_env_cfg import SarEnvCfg


@configclass
class SarTrajectoryEnvCfg(SarEnvCfg):
    """SAR mission flown by a trajectory-tracking policy; see the module docstring."""

    observations: SurveyTrajectoryObservationsCfg = SurveyTrajectoryObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_trajectory_carrot(self.commands.survey_target)
