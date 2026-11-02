"""Rescue finale with Hector's custom quadruped (``quadruped_lab``, converted 2.x height-scan policy, PhysX)."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="DroneLab-Rescue-CustomQuad-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:RescueCustomQuadEnvCfg",
        # the runner of Isaac-Velocity-Rough-Custom-Quad-v0, so `isaaclab play --checkpoint <its checkpoint>` works
        "rsl_rl_cfg_entry_point": (
            "quadruped_lab.tasks.locomotion.config.custom_quad.agents.rsl_rl_ppo_cfg:CustomQuadRoughPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
