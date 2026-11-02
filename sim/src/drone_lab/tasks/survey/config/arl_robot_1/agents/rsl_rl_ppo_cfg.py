"""RSL-RL runner config of the survey task: the built-in tracking task's network, so its checkpoints load as-is."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.drone_arl.track_position_state_based.config.arl_robot_1.agents.rsl_rl_ppo_cfg import (
    TrackPositionNoObstaclesEnvPPORunnerCfg,
)


@configclass
class SurveyPPORunnerCfg(TrackPositionNoObstaclesEnvPPORunnerCfg):
    """Actor/critic MLP (256, 128, 64), ELU, no observation normalization, same experiment name.

    Use it to *play* a checkpoint trained on ``IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1``; the survey
    task is not meant for training.
    """
