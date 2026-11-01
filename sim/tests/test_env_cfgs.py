"""Task configs resolve on the CPU and stay compatible with the built-in tracking policy (env_isaaclab, no Kit)."""

from __future__ import annotations

from _runner import main_for, require

SURVEY = "DroneLab-Survey-ARL-Robot-1-v0"
BUILTIN = "IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1"
_GROUP_SETTINGS = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length",
                   "flatten_history_dim", "history_order"}


def _parse(task, overrides=()):
    require("isaaclab_tasks")
    import drone_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    return parse_env_cfg(task, device="cpu", overrides=list(overrides))


def _terms(group):
    return [k for k, v in group.__dict__.items() if k not in _GROUP_SETTINGS and v is not None]


def test_survey_policy_observations_match_training():
    cfg, base = _parse(SURVEY), _parse(BUILTIN)
    pol, bpol = cfg.observations.policy, base.observations.policy
    assert _terms(pol) == _terms(bpol) == [
        "base_link_position", "base_orientation", "base_lin_vel", "base_ang_vel", "last_action"
    ]
    for name in _terms(pol)[1:]:
        t, b = getattr(pol, name), getattr(bpol, name)
        assert t.func is b.func and t.params == b.params and t.scale == b.scale and t.clip == b.clip, name
    from drone_lab.tasks.survey.mdp import position_relative_to_target

    assert pol.base_link_position.func is position_relative_to_target
    assert pol.base_link_position.params["max_offset"] <= 1.0
    assert pol.enable_corruption is False and bpol.enable_corruption is False
    assert all(getattr(pol, n).noise is None for n in _terms(pol))
    assert pol.concatenate_terms and bpol.concatenate_terms
    # only the observation groups the policy uses (no extra group that could reach the actor/critic)
    assert [k for k, v in cfg.observations.__dict__.items() if v is not None] == ["policy"]


def test_survey_actions_and_timing_match_training():
    cfg, base = _parse(SURVEY), _parse(BUILTIN)
    a, b = cfg.actions.thrust_command, base.actions.thrust_command
    assert (a.scale, a.offset, a.clip) == (b.scale, b.offset, b.clip)
    assert (cfg.sim.dt, cfg.decimation) == (base.sim.dt, base.decimation) == (0.01, 10)
    assert cfg.scene.robot.spawn.usd_path == base.scene.robot.spawn.usd_path
    assert cfg.scene.robot.actuators["thrusters"].dt == cfg.sim.dt


def test_survey_mission_settings():
    cfg = _parse(SURVEY)
    assert cfg.scene.num_envs == 1 and cfg.episode_length_s >= 600.0
    assert cfg.terminations.time_out is None
    assert {"crash", "collision", "flipped"} <= {k for k, v in cfg.terminations.__dict__.items() if v is not None}
    assert cfg.terminations.survey_complete.time_out is True
    assert cfg.events.reset_base.mode == "reset"
    assert cfg.scene.terrain.terrain_type == "generator" and cfg.scene.terrain.terrain_generator.num_rows == 4
    scanner = cfg.scene.mapping_scanner
    assert scanner.mesh_prim_paths == ["/World/ground"] and scanner.prim_path.endswith("/Robot/base_link")
    assert type(scanner.pattern_cfg).__name__ == "GridPatternCfg" and scanner.pattern_cfg.resolution == 0.25
    cmd = cfg.commands.survey_target
    assert cmd.pattern == "lawnmower" and cmd.max_lead <= 1.0 and cmd.resampling_time_range[0] >= 1e6


def test_physics_presets_and_overrides():
    cfg = _parse(SURVEY, ["physics=isaacsim_physx"])
    assert type(cfg.sim.physics).__name__ == "PhysxCfg" and cfg.sim.physics.gpu_max_rigid_patch_count == 10 * 2**15
    try:
        _parse(SURVEY, ["physics=newton_mjwarp"])
    except ValueError as exc:
        assert "newton_mjwarp" in str(exc)
    else:
        raise AssertionError("physics=newton_mjwarp must be rejected (Multirotor is PhysX-only)")
    # the built-in task has no physics presets at all: a typed selector is an error there
    try:
        _parse(BUILTIN, ["physics=isaacsim_physx"])
    except ValueError as exc:
        assert "did not match a selectable preset" in str(exc)
    else:
        raise AssertionError("expected the built-in task to reject physics=...")
    cfg = _parse(SURVEY, ["env.commands.survey_target.speed=1.7", "env.commands.survey_target.pattern=spiral"])
    assert cfg.commands.survey_target.speed == 1.7 and cfg.commands.survey_target.pattern == "spiral"


def test_agent_cfg_loads_builtin_checkpoints():
    require("isaaclab_tasks")
    import drone_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    ours = load_cfg_from_registry(SURVEY, "rsl_rl_cfg_entry_point")
    base = load_cfg_from_registry(BUILTIN, "rsl_rl_cfg_entry_point")
    assert ours.experiment_name == base.experiment_name
    for model in ("actor", "critic"):
        a, b = getattr(ours, model), getattr(base, model)
        assert (a.hidden_dims, a.activation, a.obs_normalization) == (b.hidden_dims, b.activation, b.obs_normalization)


def test_track_trajectory_cfg_if_present():
    require("isaaclab_tasks")
    import gymnasium as gym

    import drone_lab.tasks  # noqa: F401

    task = "DroneLab-TrackTrajectory-ARL-Robot-1-v0"
    if task not in gym.registry:
        import unittest

        raise unittest.SkipTest("trajectory task not registered")
    cfg = _parse(task)
    names = _terms(cfg.observations.policy)
    assert names[0] == "reference_error" and "last_action" in names
    assert cfg.terminations.time_out is not None
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    assert agent.experiment_name == "drone_lab_track_trajectory"


if __name__ == "__main__":
    main_for(globals(), "test_env_cfgs")
