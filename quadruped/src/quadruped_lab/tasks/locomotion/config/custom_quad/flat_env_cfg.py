# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026 Hector Lugo
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-terrain velocity tracking for the custom quadruped (proprioception only, 48 obs).

Ported from the Isaac Lab 2.3.2 task ``custom_quadruped/flat_env_cfg.py``. The 2.x ``depth_camera`` /
``depth_scan`` lines are dropped (those attributes only existed in the locally patched 2.x base config), and the 2.x
``CustomQuadFlatEnvCfg_PLAY`` class is replaced by the base ``play_mode()`` that ``isaaclab play`` applies.
"""

from isaaclab.utils import configclass

from .rough_env_cfg import CustomQuadRoughEnvCfg


@configclass
class CustomQuadFlatEnvCfg(CustomQuadRoughEnvCfg):
    """Custom quadruped on a flat plane without height scan or terrain curriculum."""

    def __post_init__(self) -> None:
        """Switch the rough task to a flat plane and drop the height scan and the terrain curriculum."""
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
