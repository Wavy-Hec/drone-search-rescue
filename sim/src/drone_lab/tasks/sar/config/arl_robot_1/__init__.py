"""SAR survey mission for the ARL-Robot-1 quadrotor."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="DroneLab-SAR-ARL-Robot-1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sar_env_cfg:SarEnvCfg",
        "default_agent": "rsl_rl",
        # the policy is the built-in tracking policy, exactly as in the survey task
        "rsl_rl_cfg_entry_point": "drone_lab.tasks.survey.config.arl_robot_1.agents.rsl_rl_ppo_cfg:SurveyPPORunnerCfg",
    },
)
