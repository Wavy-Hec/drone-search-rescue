"""Tests for the project's task registrations (needs env_isaaclab; no simulator)."""

from __future__ import annotations

from _runner import main_for, require

EXPECTED = {
    "DroneLab-Rescue-Go2-v0": {
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "env_cfg_entry_point": "drone_lab.tasks.rescue.config.go2.rescue_env_cfg:RescueGo2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
    "DroneLab-SAR-ARL-Robot-1-v0": {
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "env_cfg_entry_point": "drone_lab.tasks.sar.config.arl_robot_1.sar_env_cfg:SarEnvCfg",
        # flown by the survey task's policy, so it shares the survey runner config
        "rsl_rl_cfg_entry_point": "drone_lab.tasks.survey.config.arl_robot_1.agents.rsl_rl_ppo_cfg:SurveyPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
    "DroneLab-Survey-ARL-Robot-1-v0": {
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "env_cfg_entry_point": "drone_lab.tasks.survey.config.arl_robot_1.survey_env_cfg:SurveyEnvCfg",
        "rsl_rl_cfg_entry_point": "drone_lab.tasks.survey.config.arl_robot_1.agents.rsl_rl_ppo_cfg:SurveyPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
    "DroneLab-TrackTrajectory-ARL-Robot-1-v0": {
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "env_cfg_entry_point": (
            "drone_lab.tasks.track_trajectory.config.arl_robot_1.track_trajectory_env_cfg:TrackTrajectoryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "drone_lab.tasks.track_trajectory.config.arl_robot_1.agents.rsl_rl_ppo_cfg:TrackTrajectoryPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
}


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    require("isaaclab_tasks")
    import gymnasium as gym

    import drone_lab.tasks  # noqa: F401

    ours = {k for k, s in gym.registry.items() if str(s.kwargs.get("env_cfg_entry_point", "")).startswith("drone_lab")}
    assert ours == set(EXPECTED), ours
    for task_id, expected in EXPECTED.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected["entry_point"]
        for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point", "default_agent"):
            assert spec.kwargs[key] == expected[key], (task_id, key)


def test_external_callback_registers_tasks():
    """``--external_callback drone_lab.register.register_tasks`` works without installing the project."""
    require("isaaclab_tasks")
    import gymnasium as gym

    from drone_lab.register import register_tasks

    assert register_tasks() is None  # None = "no arguments consumed" for the isaaclab CLI
    assert set(EXPECTED) <= set(gym.registry)


if __name__ == "__main__":
    main_for(globals(), "test_registration")
