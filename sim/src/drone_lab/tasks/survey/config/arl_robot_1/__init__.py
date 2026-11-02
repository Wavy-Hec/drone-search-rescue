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
