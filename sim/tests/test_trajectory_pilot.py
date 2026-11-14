"""Mission flown by a trajectory-tracking policy: the carrot as a moving reference, the two tasks, the pilot
choice and the terrain seed (CPU only; torch for the reference math, env_isaaclab for the configs, no Kit).

Run: ``CUDA_VISIBLE_DEVICES= python tests/test_trajectory_pilot.py`` (or with pytest).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
from _runner import main_for, require

SURVEY = "DroneLab-Survey-ARL-Robot-1-v0"
SAR = "DroneLab-SAR-ARL-Robot-1-v0"
SURVEY_TRAJECTORY = "DroneLab-Survey-Trajectory-ARL-Robot-1-v0"
SAR_TRAJECTORY = "DroneLab-SAR-Trajectory-ARL-Robot-1-v0"
TRAINING = "DroneLab-TrackTrajectory-ARL-Robot-1-v0"
TASKS = "drone_lab.tasks"

REFERENCE_TERMS = ["reference_error", "velocity_error", "lookahead"]
POLICY_TERMS = [*REFERENCE_TERMS, "base_orientation", "base_lin_vel", "base_ang_vel", "last_action"]
_GROUP_SETTINGS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
    "history_order",
}

DIAGONAL = np.array([[0.0, 0.0, 5.0], [30.0, 40.0, 5.0]])
"""50 m long, direction (0.6, 0.8, 0)."""
CLIMB = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
"""10 m straight up (the shorter path, so it is padded)."""


def _carrot():
    require("torch")
    import drone_lab.carrot as carrot

    return carrot


def _parse(task, overrides=()):
    require("isaaclab_tasks")
    from isaaclab_tasks.utils import parse_env_cfg

    import drone_lab.tasks  # noqa: F401

    return parse_env_cfg(task, device="cpu", overrides=list(overrides))


def _terms(group):
    return [k for k, v in group.__dict__.items() if k not in _GROUP_SETTINGS and v is not None]


##
# path_reference
##


def test_reference_on_straight_paths_is_speed_times_direction():
    c = _carrot()
    torch = require("torch")

    pts, arc, _ = c.pad_paths([DIAGONAL, CLIMB], 0.25)
    # one 0.1 s step at 1.5 m/s along the diagonal and at 1.0 m/s up the climb
    vel, ahead = c.path_reference(pts, arc, torch.tensor([10.0, 2.0]), torch.tensor([10.15, 2.1]), 0.1, (0.5, 1.0))
    assert vel.shape == (2, 3) and ahead.shape == (2, 2, 3)
    assert torch.allclose(vel, torch.tensor([[0.9, 1.2, 0.0], [0.0, 0.0, 1.0]]), atol=1e-3)
    # s + v t: 10.9 m and 11.65 m along the diagonal, 2.6 m and 3.1 m up the climb
    expected = torch.tensor([[[6.54, 8.72, 5.0], [6.99, 9.32, 5.0]], [[0.0, 0.0, 2.6], [0.0, 0.0, 3.1]]])
    assert torch.allclose(ahead, expected, atol=1e-3)


def test_reference_follows_the_path_round_a_corner():
    c = _carrot()
    torch = require("torch")

    corner = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 10.0, 5.0]])
    pts, arc, _ = c.pad_paths([corner], 0.25)
    # the step crosses the corner: 0.1 m along x, then 0.1 m along y, in 0.1 s (2 m/s along the path)
    vel, ahead = c.path_reference(pts, arc, torch.tensor([9.9]), torch.tensor([10.1]), 0.1, (0.5, 1.0))
    assert torch.allclose(vel, torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-3)  # displacement / dt, not speed x tangent
    # 1 m and 2 m further along the path, i.e. 1.1 m and 2.1 m past the corner
    assert torch.allclose(ahead, torch.tensor([[[10.0, 1.1, 5.0], [10.0, 2.1, 5.0]]]), atol=1e-3)


def test_lookahead_stops_at_the_end_of_the_path():
    c = _carrot()
    torch = require("torch")

    pts, arc, _ = c.pad_paths([DIAGONAL, CLIMB], 0.25)
    # diagonal: 2 m/s, 0.4 m before the end, so both points (1 m and 2 m ahead) are past it
    # climb: 0.5 m/s, 0.4 m before the end, so 0.25 m ahead is on the path and 0.5 m ahead is past it
    vel, ahead = c.path_reference(pts, arc, torch.tensor([49.4, 9.55]), torch.tensor([49.6, 9.6]), 0.1, (0.5, 1.0))
    assert torch.allclose(vel, torch.tensor([[1.2, 1.6, 0.0], [0.0, 0.0, 0.5]]), atol=1e-3)
    expected = torch.tensor([[[30.0, 40.0, 5.0], [30.0, 40.0, 5.0]], [[0.0, 0.0, 9.85], [0.0, 0.0, 10.0]]])
    assert torch.allclose(ahead, expected, atol=1e-3)


def test_lookahead_sits_on_a_waiting_carrot():
    c = _carrot()
    torch = require("torch")

    pts, arc, _ = c.pad_paths([DIAGONAL, CLIMB], 0.25)
    s = torch.tensor([20.0, 4.0])
    vel, ahead = c.path_reference(pts, arc, s, s, 0.1, (0.5, 1.0))
    assert torch.equal(vel, torch.zeros(2, 3))
    carrot = torch.tensor([[12.0, 16.0, 5.0], [0.0, 0.0, 4.0]])
    assert torch.allclose(ahead[:, 0], carrot, atol=1e-4) and torch.allclose(ahead[:, 1], carrot, atol=1e-4)


##
# The two tasks
##


def test_policy_observations_are_the_training_tasks():
    train = _parse(TRAINING).observations.policy
    assert _terms(train) == POLICY_TERMS
    for task in (SURVEY_TRAJECTORY, SAR_TRAJECTORY):
        cfg = _parse(task)
        pol = cfg.observations.policy
        assert _terms(pol) == POLICY_TERMS, task
        for name in POLICY_TERMS:
            ours, theirs = getattr(pol, name), getattr(train, name)
            assert ours.func is theirs.func and ours.scale == theirs.scale and ours.clip == theirs.clip, (task, name)
            # same parameters (the clips are among them); only the reference comes from the survey carrot
            expected = dict(theirs.params)
            if name in REFERENCE_TERMS:
                expected["command_name"] = "survey_target"
            assert ours.params == expected, (task, name)
            assert ours.noise is None, (task, name)
        assert pol.enable_corruption is False, task
        for key in _GROUP_SETTINGS - {"enable_corruption"}:
            assert getattr(pol, key, None) == getattr(train, key, None), (task, key)
        # only the observation group the policy uses, and the command the reference terms name
        assert [k for k, v in cfg.observations.__dict__.items() if v is not None] == ["policy"], task
        assert type(cfg.commands.survey_target).__name__ == "SurveyCarrotCommandCfg", task
    # the clips the policy trained with, and the training config itself is untouched
    clips = [getattr(train, name).params["clip"] for name in REFERENCE_TERMS]
    assert clips == [2.0, 3.0, 3.0]
    assert all(getattr(train, name).params["command_name"] == "reference" for name in REFERENCE_TERMS)
    assert train.enable_corruption is True and train.reference_error.noise is not None


def test_actions_and_timing_are_the_training_tasks():
    train = _parse(TRAINING)
    for task in (SURVEY_TRAJECTORY, SAR_TRAJECTORY):
        cfg = _parse(task)
        a, b = cfg.actions.thrust_command, train.actions.thrust_command
        assert (a.scale, a.offset, a.clip) == (b.scale, b.offset, b.clip), task
        assert (cfg.sim.dt, cfg.decimation) == (train.sim.dt, train.decimation) == (0.01, 10), task
        assert cfg.scene.robot.spawn.usd_path == train.scene.robot.spawn.usd_path, task
        assert cfg.scene.robot.actuators["thrusters"].to_dict() == train.scene.robot.actuators["thrusters"].to_dict()


def test_agent_config_is_the_trajectory_tasks():
    require("isaaclab_tasks")
    import gymnasium as gym

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import drone_lab.tasks  # noqa: F401

    entry = f"{TASKS}.track_trajectory.config.arl_robot_1.agents.rsl_rl_ppo_cfg:TrackTrajectoryPPORunnerCfg"
    assert gym.spec(TRAINING).kwargs["rsl_rl_cfg_entry_point"] == entry
    for task in (SURVEY_TRAJECTORY, SAR_TRAJECTORY):
        spec = gym.spec(task)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv", task
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == entry and spec.kwargs["default_agent"] == "rsl_rl", task
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        assert agent.experiment_name == "drone_lab_track_trajectory", task
        assert agent.actor.hidden_dims == [256, 128, 64] and agent.actor.obs_normalization is True, task


def test_trajectory_tasks_are_the_default_tasks_with_a_wider_smoother_carrot():
    require("isaaclab_tasks")
    from drone_lab.tasks.sar.config.arl_robot_1.sar_env_cfg import SarEnvCfg
    from drone_lab.tasks.survey.config.arl_robot_1.survey_env_cfg import SurveyEnvCfg

    for task, base_task, base_class in ((SURVEY_TRAJECTORY, SURVEY, SurveyEnvCfg), (SAR_TRAJECTORY, SAR, SarEnvCfg)):
        cfg, base = _parse(task), _parse(base_task)
        assert isinstance(cfg, base_class), task
        carrot = cfg.commands.survey_target
        assert (carrot.max_accel, carrot.max_lead, carrot.slow_lead) == (1.0, 1.5, 1.0), task
        # nothing else of the mission changes
        expected = base.commands.survey_target.to_dict() | {"max_accel": 1.0, "max_lead": 1.5, "slow_lead": 1.0}
        assert carrot.to_dict() == expected, task
        assert cfg.events.to_dict() == base.events.to_dict(), task
        assert cfg.terminations.to_dict() == base.terminations.to_dict(), task
        assert cfg.scene.to_dict() == base.scene.to_dict(), task
    assert not isinstance(_parse(SURVEY_TRAJECTORY), SarEnvCfg)


def test_default_tasks_keep_their_carrot_and_their_observations():
    for task in (SURVEY, SAR):
        cfg = _parse(task)
        carrot = cfg.commands.survey_target
        assert carrot.max_accel is None, task
        assert (carrot.speed, carrot.max_lead, carrot.slow_lead) == (1.0, 1.0, 0.7), task
        assert tuple(carrot.lookahead_times) == (0.5, 1.0), task
        expected = ["base_link_position", "base_orientation", "base_lin_vel", "base_ang_vel", "last_action"]
        assert _terms(cfg.observations.policy) == expected, task


##
# The carrot command (the real term, in a stand-in env whose drone the test moves)
##


def _command(num_envs: int = 1, **settings):
    """``SurveyCarrotCommand`` on a 100 m line along +x at z = 5 m. Returns ``(term, env, robot)``."""
    torch = require("torch")
    require("isaaclab")
    from drone_lab.tasks.survey.mdp import SurveyCarrotCommandCfg, commands

    data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 5.0]]).repeat(num_envs, 1)),
        root_lin_vel_w=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
    )
    robot = SimpleNamespace(data=data)
    markers = SimpleNamespace(clear_debug_vis_callback=lambda term: None)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.1,
        scene={"robot": robot},
        sim=SimpleNamespace(vis_marker_registry=markers),
    )
    cfg = SurveyCarrotCommandCfg(
        pattern="waypoints",
        waypoints=[[0.0, 0.0, 5.0], [100.0, 0.0, 5.0]],
        multi_env_mode="copy",
        area=(-10.0, 10.0, -10.0, 10.0),
        **settings,
    )
    # no USD stage here: the term then assumes flat ground at z = 0
    with mock.patch.object(commands, "terrain_mesh_from_stage", return_value=None):
        term = commands.SurveyCarrotCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term={"survey_target": term}.__getitem__)
    return term, env, robot


def test_default_carrot_moves_exactly_as_advance_carrot_does():
    c = _carrot()
    torch = require("torch")

    term, env, robot = _command(speed=1.2)
    assert term.cfg.max_accel is None
    s = torch.zeros(1)
    drone = torch.tensor([[0.0, 0.0, 5.0]])
    for step in range(40):
        if step < 25:  # the drone trails 0.3 m behind the carrot, then stops, so the carrot has to wait
            drone = term.carrot_w - torch.tensor([[0.3, 0.0, 0.0]])
        robot.data.root_pos_w.torch = drone
        s, carrot, gate = c.advance_carrot(s, term.path_points, term.path_arclength, drone, 1.2, 0.1, 1.0, 0.7)
        term.compute(env.step_dt)
        assert torch.equal(term.s, s) and torch.equal(term.carrot_w, carrot) and torch.equal(term.gate, gate), step
        if step == 24:  # 25 steps at 1.2 m/s
            assert torch.allclose(term.s, torch.tensor([3.0]), atol=1e-4)
            assert term.ref_pos_w is term.carrot_w
            assert torch.allclose(term.ref_vel_w, torch.tensor([[1.2, 0.0, 0.0]]), atol=1e-3)
            ahead = term.carrot_w.unsqueeze(1) + torch.tensor([[[0.6, 0.0, 0.0], [1.2, 0.0, 0.0]]])
            assert torch.allclose(term.lookahead_w, ahead, atol=1e-3)
    # the carrot waits 1 m ahead of the stopped drone: no reference velocity, look-ahead points on the carrot
    assert float(term.gate) < 0.05 and float(torch.linalg.norm(term.carrot_w - drone)) <= 1.0 + 1e-5
    assert float(term.ref_vel_w.abs().max()) < 0.05
    assert torch.allclose(term.lookahead_w, term.carrot_w.unsqueeze(1).expand(-1, 2, -1), atol=0.05)


def test_max_accel_ramps_the_carrot_speed():
    torch = require("torch")

    term, env, robot = _command(speed=1.5, max_accel=1.0)

    def step():
        robot.data.root_pos_w.torch = term.carrot_w.clone()  # the drone keeps up
        term.compute(env.step_dt)

    assert float(term.speed) == 0.0
    for k in range(1, 21):  # 1 m/s^2 for 1.5 s, then the configured 1.5 m/s
        step()
        speed = min(0.1 * k, 1.5)
        assert abs(float(term.speed) - speed) < 1e-5, k
        assert torch.allclose(term.ref_vel_w, torch.tensor([[speed, 0.0, 0.0]]), atol=1e-3), k
    # 0.1 s x (0.1 + 0.2 + ... + 1.5) m/s + 5 steps at 1.5 m/s
    assert abs(float(term.s) - (1.2 + 0.75)) < 1e-4
    term.cfg.speed = 0.3  # the mission slows the carrot down: it brakes at the same rate and stays at 0.3 m/s
    for k in range(1, 21):
        step()
        assert abs(float(term.speed) - max(1.5 - 0.1 * k, 0.3)) < 1e-5, k
    # 0.1 s x (1.4 + 1.3 + ... + 0.3) m/s + 8 steps at 0.3 m/s
    assert abs(float(term.s) - (1.95 + 1.02 + 0.24)) < 1e-4


def test_reset_restarts_the_speed_and_the_reference_of_that_env_only():
    torch = require("torch")

    term, env, robot = _command(num_envs=2, speed=1.5, max_accel=1.0)
    for _ in range(20):
        robot.data.root_pos_w.torch = term.carrot_w.clone()
        term.compute(env.step_dt)
    assert torch.allclose(term.speed, torch.tensor([1.5, 1.5]))
    term.reset(torch.tensor([0]))
    start = torch.tensor([0.0, 0.0, 5.0])
    assert float(term.s[0]) == 0.0 and float(term.speed[0]) == 0.0
    assert torch.equal(term.ref_pos_w[0], start) and torch.equal(term.ref_vel_w[0], torch.zeros(3))
    assert torch.equal(term.lookahead_w[0], start.expand(2, 3))
    # the other env flies on
    assert abs(float(term.s[1]) - 1.95) < 1e-4 and abs(float(term.speed[1]) - 1.5) < 1e-6
    assert torch.allclose(term.ref_vel_w[1], torch.tensor([1.5, 0.0, 0.0]), atol=1e-3)
    robot.data.root_pos_w.torch = term.carrot_w.clone()
    term.compute(env.step_dt)
    assert torch.allclose(term.speed, torch.tensor([0.1, 1.5]), atol=1e-6)  # env 0 starts its ramp again


def test_trajectory_observations_read_the_survey_carrot():
    torch = require("torch")

    pol = _parse(SURVEY_TRAJECTORY).observations.policy
    term, env, robot = _command(speed=1.0)
    for _ in range(10):
        robot.data.root_pos_w.torch = term.carrot_w - torch.tensor([[0.3, 0.0, 0.0]])
        term.compute(env.step_dt)
    # the carrot flies 1 m/s along +x; the drone is 0.3 m behind and 0.2 m below it, at 0.8 m/s and climbing 0.1 m/s
    robot.data.root_pos_w.torch = term.carrot_w - torch.tensor([[0.3, 0.0, 0.2]])
    robot.data.root_lin_vel_w.torch = torch.tensor([[0.8, 0.0, 0.1]])
    obs = [getattr(pol, name).func(env, **getattr(pol, name).params) for name in REFERENCE_TERMS]
    assert torch.allclose(obs[0], torch.tensor([[-0.3, 0.0, -0.2]]), atol=1e-3)  # drone - reference
    assert torch.allclose(obs[1], torch.tensor([[-0.2, 0.0, 0.1]]), atol=1e-3)  # drone - reference velocity
    # the reference 0.5 s and 1 s ahead (0.5 m and 1 m further), seen from the drone
    assert torch.allclose(obs[2], torch.tensor([[0.8, 0.0, 0.2, 1.3, 0.0, 0.2]]), atol=1e-3)


##
# Pilot choice and terrain seed of the mission
##


def test_pilot_tasks_are_registered():
    require("isaaclab_tasks")
    import gymnasium as gym

    import drone_lab.tasks  # noqa: F401
    from drone_lab.sar.mission import PILOT_TASKS

    assert PILOT_TASKS == {"hover": SAR, "trajectory": SAR_TRAJECTORY}
    entries = {
        SAR: f"{TASKS}.sar.config.arl_robot_1.sar_env_cfg:SarEnvCfg",
        SAR_TRAJECTORY: f"{TASKS}.sar.config.arl_robot_1.sar_trajectory_env_cfg:SarTrajectoryEnvCfg",
        SURVEY_TRAJECTORY: f"{TASKS}.survey.config.arl_robot_1.survey_trajectory_env_cfg:SurveyTrajectoryEnvCfg",
    }
    for task, entry in entries.items():
        assert gym.spec(task).kwargs["env_cfg_entry_point"] == entry, task


def test_terrain_seed_builds_and_dresses_another_site():
    """As ``scripts/sar_mission.py`` does: parse, set the terrain seed, regenerate the site, plan the flight."""
    require("isaaclab_tasks")
    from isaaclab.utils import clone

    from drone_lab.sar.mission import PILOT_TASKS, plan_mission, set_terrain_seed
    from drone_lab.sar.scenario import TerrainSite
    from drone_lab.sar.scene import scenario_for
    from drone_lab.terrain import SURVEY_TERRAIN_CFG

    spacing, altitude = 5.0, 4.0
    grid = np.arange(-19.5, 20.0, 1.0)
    grid = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    heights, flight_z = {}, {}
    for seed in (1, 3):
        cfg = _parse(PILOT_TASKS["trajectory"], ["physics=isaacsim_physx", "env.scene.sar_site.spawn.seed=0"])
        assert cfg.scene.terrain.terrain_generator.seed == 17  # the standard terrain
        # a site that holds its own copy of the generator config must end up on the scene's one as well
        cfg.scene.sar_site.spawn.terrain_generator = clone(cfg.scene.terrain.terrain_generator)
        set_terrain_seed(cfg, seed)
        assert cfg.scene.terrain.terrain_generator.seed == seed
        assert cfg.scene.sar_site.spawn.terrain_generator is cfg.scene.terrain.terrain_generator
        assert SURVEY_TERRAIN_CFG.seed == 17  # the shared default is not touched

        built = TerrainSite(cfg.scene.terrain.terrain_generator)  # the terrain the simulator imports
        scenario = scenario_for(cfg.scene.sar_site.spawn)  # the objects the site spawner places
        in_area = [o for o in scenario.objects if o.in_area]
        assert len(in_area) >= 20, seed
        for o in in_area:  # every object rests on the terrain that is built
            ground = built.height(o.center[0], o.center[1])
            assert abs(o.ground_z - ground) <= o.ground_spread + 0.01, (seed, o.name, o.ground_z, ground)

        pad = next(o for o in scenario.objects if o.category == "landing_pad")
        home = (float(pad.center[0]), float(pad.center[1]), float(pad.aabb_max[2]))
        terrain_z = built.terrain_z_range(inset=-(0.5 * spacing + 0.5))
        plan = plan_mission(scenario.area, home, terrain_z[1] + altitude, spacing)
        first, last = plan.survey_waypoints
        survey = plan.waypoints[first : last + 1]
        assert float(np.min(survey[:, 2] - built.heights(survey[:, :2]))) >= altitude - 1e-6, seed
        np.testing.assert_allclose(plan.waypoints[0], [home[0], home[1], home[2] + 0.5])
        heights[seed], flight_z[seed] = built.heights(grid), plan.flight_z

    # two seeds, two terrains: the heights differ by about 1 m on average, and so do the flight heights
    assert float(np.mean(np.abs(heights[1] - heights[3]))) > 0.3
    assert abs(flight_z[1] - flight_z[3]) > 0.05


if __name__ == "__main__":
    main_for(globals(), "test_trajectory_pilot")
