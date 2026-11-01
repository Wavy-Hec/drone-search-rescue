# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom quadruped velocity-tracking environments.

The 2.x task IDs are kept. The ``-Play-v0`` variants are not registered any more: in Isaac Lab 3.0,
``isaaclab play`` applies the env config's ``play_mode()`` automatically (``--train_env_cfg`` skips it), and the
velocity base ``play_mode()`` already performs every change the 2.x ``*_PLAY`` classes made.

The project's ``isaaclab.tasks`` entry point imports ``quadruped_lab.tasks``, whose ``import_packages`` call imports
this package and runs the registrations below.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Custom-Quad-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CustomQuadFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomQuadFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Custom-Quad-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:CustomQuadRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomQuadRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "default_agent": "rsl_rl",
    },
)

# EXPERIMENTAL: depth-camera redesign. Only rsl_rl is registered: the 2.x skrl yaml describes an MLP on a flat
# observation vector and cannot consume the (N, 1, H, W) depth group.
gym.register(
    id="Isaac-Velocity-Rough-Depth-Custom-Quad-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_depth_cfg:CustomQuadRoughDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomQuadRoughDepthPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
