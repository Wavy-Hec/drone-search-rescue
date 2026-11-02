# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import pytest

pytest.importorskip("isaaclab_tasks", reason="needs Isaac Lab 3.0")
gym = pytest.importorskip("gymnasium")

import quadruped_lab.tasks  # noqa: F401  (runs the gym.register calls)

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

_CFG = "quadruped_lab.tasks.locomotion.config.custom_quad"
_AGENTS = f"{_CFG}.agents"


def test_task_registrations() -> None:
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "Isaac-Velocity-Flat-Custom-Quad-v0": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": f"{_CFG}.flat_env_cfg:CustomQuadFlatEnvCfg",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:CustomQuadFlatPPORunnerCfg",
            "skrl_cfg_entry_point": f"{_AGENTS}:skrl_flat_ppo_cfg.yaml",
            "default_agent": "rsl_rl",
        },
        "Isaac-Velocity-Rough-Custom-Quad-v0": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": f"{_CFG}.rough_env_cfg:CustomQuadRoughEnvCfg",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:CustomQuadRoughPPORunnerCfg",
            "skrl_cfg_entry_point": f"{_AGENTS}:skrl_rough_ppo_cfg.yaml",
            "default_agent": "rsl_rl",
        },
        "Isaac-Velocity-Rough-Depth-Custom-Quad-v0": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": f"{_CFG}.rough_env_depth_cfg:CustomQuadRoughDepthEnvCfg",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:CustomQuadRoughDepthPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    }

    registered = {
        spec.id
        for spec in gym.registry.values()
        if str(spec.kwargs.get("env_cfg_entry_point", "")).startswith("quadruped_lab.tasks")
    }
    # the 2.x -Play-v0 variants are intentionally gone: `isaaclab play` applies play_mode() itself
    assert registered == set(expected)

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        for key, value in expected_values.items():
            if key != "entry_point":
                assert spec.kwargs[key] == value, (task_id, key)
    # the depth task has no skrl config (the MLP yaml cannot consume the image group)
    assert "skrl_cfg_entry_point" not in gym.spec("Isaac-Velocity-Rough-Depth-Custom-Quad-v0").kwargs
