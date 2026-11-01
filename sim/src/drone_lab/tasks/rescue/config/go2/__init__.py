"""Rescue finale with the Unitree Go2 (Isaac Lab's rough-terrain velocity policy)."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="DroneLab-Rescue-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:RescueGo2EnvCfg",
        # the policy is the stock Go2 rough-terrain one: same runner config, so `isaaclab play --checkpoint` works
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
