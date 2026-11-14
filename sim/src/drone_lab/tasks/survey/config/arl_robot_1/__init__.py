"""Survey mission for the ARL-Robot-1 quadrotor."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="DroneLab-Survey-ARL-Robot-1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.survey_env_cfg:SurveyEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SurveyPPORunnerCfg",
    },
)

gym.register(
    id="DroneLab-Survey-Trajectory-ARL-Robot-1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.survey_trajectory_env_cfg:SurveyTrajectoryEnvCfg",
        "default_agent": "rsl_rl",
        # the pilot is a policy trained on the trajectory-tracking task, so its checkpoints load with that runner
        "rsl_rl_cfg_entry_point": (
            "drone_lab.tasks.track_trajectory.config.arl_robot_1.agents.rsl_rl_ppo_cfg:TrackTrajectoryPPORunnerCfg"
        ),
    },
)
