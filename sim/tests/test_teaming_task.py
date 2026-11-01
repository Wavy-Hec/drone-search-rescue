"""The rescue task config: parity with the Go2 velocity task the checkpoint was trained on, the survey terrain,
the path-following command, deterministic events; the default plan; the script's CPU plan mode.

Needs env_isaaclab (Isaac Lab configs, no simulator): ``CUDA_VISIBLE_DEVICES= python tests/test_teaming_task.py``.
Skipped elsewhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _runner import main_for, require

RESCUE = "DroneLab-Rescue-Go2-v0"
BASE = "Isaac-Velocity-Rough-UnitreeGo2"
_GROUP_SETTINGS = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length",
                   "flatten_history_dim", "history_order"}
PROJECT = Path(__file__).resolve().parents[1]


def _parse(task, overrides=()):
    require("isaaclab_tasks")
    from isaaclab_tasks.utils import parse_env_cfg

    import drone_lab.tasks  # noqa: F401

    return parse_env_cfg(task, device="cpu", overrides=list(overrides))


def _terms(group):
    return [k for k, v in group.__dict__.items() if k not in _GROUP_SETTINGS and v is not None]


def test_rescue_task_registration():
    require("isaaclab_tasks")
    import gymnasium as gym

    import drone_lab.tasks  # noqa: F401

    spec = gym.spec(RESCUE)
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "drone_lab.tasks.rescue.config.go2.rescue_env_cfg:RescueGo2EnvCfg"
    # the stock Go2 runner config, so `isaaclab play --task DroneLab-Rescue-Go2-v0 --checkpoint <Go2 ckpt>` works
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == gym.spec(BASE).kwargs["rsl_rl_cfg_entry_point"]


def test_policy_interface_matches_the_go2_velocity_task():
    cfg, base = _parse(RESCUE), _parse(BASE)
    pol, bpol = cfg.observations.policy, base.observations.policy
    assert _terms(pol) == _terms(bpol) == ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                                            "joint_pos", "joint_vel", "actions", "height_scan"]
    for name in _terms(pol):
        t, b = getattr(pol, name), getattr(bpol, name)
        assert t.func is b.func and t.params == b.params and t.scale == b.scale and t.clip == b.clip, name
        assert t.history_length == b.history_length, name
    assert pol.concatenate_terms and pol.enable_corruption is False  # play semantics: no noise
    assert [k for k, v in cfg.observations.__dict__.items() if v is not None] == \
        [k for k, v in base.observations.__dict__.items() if v is not None]
    a, b = cfg.actions.joint_pos, base.actions.joint_pos
    assert (a.scale, a.use_default_offset, a.joint_names) == (b.scale, b.use_default_offset, b.joint_names) == \
        (0.25, True, [".*"])
    assert (cfg.sim.dt, cfg.decimation) == (base.sim.dt, base.decimation) == (0.005, 4)
    assert cfg.sim.use_newton_actuators == base.sim.use_newton_actuators
    assert cfg.scene.robot.spawn.usd_path == base.scene.robot.spawn.usd_path
    hs, bhs = cfg.scene.height_scanner, base.scene.height_scanner
    assert (hs.prim_path, hs.offset.pos, hs.ray_alignment) == (bhs.prim_path, bhs.offset.pos, bhs.ray_alignment)
    assert (hs.pattern_cfg.resolution, list(hs.pattern_cfg.size)) == (bhs.pattern_cfg.resolution,
                                                                     list(bhs.pattern_cfg.size))
    assert hs.mesh_prim_paths == ["/World/ground"]
    # physics: the backend the checkpoint was trained on by default, PhysX selectable
    assert type(cfg.sim.physics).__name__ == "NewtonCfg"
    assert type(_parse(RESCUE, ["physics=isaacsim_physx"]).sim.physics).__name__ == "PhysxCfg"


def test_rescue_scene_command_events_and_terminations():
    require("isaaclab_tasks")
    from drone_lab.tasks.rescue.mdp import PathFollowCommandCfg, reset_root_on_route, route_complete
    from drone_lab.terrain import SURVEY_TERRAIN_CFG

    cfg, base = _parse(RESCUE), _parse(BASE)
    ter = cfg.scene.terrain
    assert ter.terrain_type == "generator" and ter.use_terrain_origins is False
    assert ter.visual_material.mdl_path.endswith("Soil_Rocky.mdl") and ter.visual_material.world_or_object is True
    gen = ter.terrain_generator
    assert gen.seed == SURVEY_TERRAIN_CFG.seed and gen.size == SURVEY_TERRAIN_CFG.size
    assert (gen.num_rows, gen.num_cols, gen.border_width) == (4, 4, 6.0) and gen.curriculum is False
    assert sorted(gen.sub_terrains) == sorted(SURVEY_TERRAIN_CFG.sub_terrains)
    assert [k for k, v in cfg.curriculum.__dict__.items() if v is not None] == []
    cmd, bcmd = cfg.commands.base_velocity, base.commands.base_velocity
    assert isinstance(cmd, PathFollowCommandCfg) and cmd.asset_name == "robot"
    for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        assert tuple(getattr(cmd.ranges, k)) == tuple(getattr(bcmd.ranges, k)), k
    assert 0.0 < cmd.cruise_speed <= cmd.ranges.lin_vel_x[1] and cmd.resampling_time_range[0] >= 1e6
    ev = {k: v for k, v in cfg.events.__dict__.items() if v is not None}
    assert set(ev) == {"physics_material", "reset_base", "reset_robot_joints"}
    assert ev["reset_base"].func is reset_root_on_route and ev["reset_robot_joints"].params["position_range"] == (1, 1)
    terms = {k: v for k, v in cfg.terminations.__dict__.items() if v is not None}
    assert set(terms) == {"time_out", "base_contact", "flipped", "goal_reached"}
    assert terms["goal_reached"].func is route_complete and terms["goal_reached"].time_out is True
    assert terms["base_contact"].params["sensor_cfg"].body_names == "base"
    assert cfg.scene.num_envs == 1 and cfg.episode_length_s >= 120.0
    assert cfg.scene.person.spawn.collision_props.collision_enabled is False
    # play mode keeps the survey area intact (the velocity base would shrink it to 5 x 5 tiles)
    cfg.play_mode()
    assert cfg.scene.terrain.terrain_generator.num_rows == 4 and cfg.observations.policy.enable_corruption is False


def test_apply_rescue_task_is_robot_agnostic():
    require("isaaclab_tasks")
    from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

    from drone_lab.tasks.rescue.rescue_env_cfg import apply_rescue_task, person_asset_cfg

    cfg = UnitreeGo2RoughEnvCfg()
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.8)
    person = person_asset_cfg((1.0, 2.0), 0.3, lying=True, yaw=0.0)
    apply_rescue_task(cfg, num_envs=3, episode_length_s=99.0, person=person)
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-0.5, 0.8)
    assert cfg.scene.num_envs == 3 and cfg.episode_length_s == 99.0 and cfg.scene.person is person
    # lying on the back: -90 deg about X (the characters face -Y), head along +Y at yaw 0, feet 0.9 m from the centre
    assert np.allclose(person.init_state.pos, (1.0, 1.1, 0.49))
    qx, qy, qz, qw = person.init_state.rot
    assert np.allclose([qy, qz], 0.0, atol=1e-12) and np.isclose(qx, -qw) and np.isclose(qw, np.sqrt(0.5))
    # same rotation as the SAR scenario's lying_face_up (so both scenes put people down the same way)
    try:
        from drone_lab.sar.scenario import pose_quaternion
    except Exception:  # the SAR module is optional here
        pass
    else:
        assert np.allclose(pose_quaternion("lying_face_up"), person.init_state.rot)
    standing = person_asset_cfg((1.0, 2.0), 0.3)
    assert standing.init_state.pos == (1.0, 2.0, 0.3) and np.allclose(standing.init_state.rot, (0, 0, 0, 1))


def test_default_route_on_the_ground_truth():
    require("isaaclab")
    require("trimesh")
    from drone_lab.tasks.rescue.rescue_env_cfg import DEFAULT_GOAL_XY, DEFAULT_START_XY
    from drone_lab.teaming.inputs import ground_truth_elevation
    from drone_lab.teaming.planner import plan_path
    from drone_lab.teaming.traversability import Footprint, compute_traversability

    z, grid, _ = ground_truth_elevation(0.25)
    person = Footprint(*DEFAULT_GOAL_XY, 1.0, "person")
    t = compute_traversability(z, grid, footprints=[person])
    plan = plan_path(t, DEFAULT_START_XY, DEFAULT_GOAL_XY)
    straight = float(np.hypot(*(np.subtract(DEFAULT_GOAL_XY, DEFAULT_START_XY))))
    assert plan.success, plan.message
    assert 15.0 <= straight <= 25.0 and straight < plan.length_m < 35.0
    assert np.all(np.isfinite(t.sample(t.cost, plan.path_xy)))
    # mixed terrain: the route leaves the flat ground (crater rim) but stays within the robot's limits
    slope = t.sample(t.slope_deg, plan.path_xy)
    assert slope.max() > 5.0 and slope.max() <= t.cfg.max_slope_deg
    assert t.sample(t.step, plan.path_xy).max() <= t.cfg.max_step_m


def test_rescue_finale_plan_only_cli():
    require("isaaclab")
    require("trimesh")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "plan_only"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, str(PROJECT / "scripts" / "rescue_finale.py"), "--plan_only", "--out",
                            str(out), "--goal", "-16", "5", "--person_pose", "standing"],
                           capture_output=True, text=True, timeout=300, env=env, cwd=str(PROJECT))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        plan = json.loads((out / "plan.json").read_text())
        assert plan["plan"]["success"] and plan["person_xy"] == [-16.0, 5.0] and plan["map_mode"] == "ground_truth"
        assert len(plan["path_xy"]) == len(plan["path_z"]) > 100
        assert (out / "plan.png").stat().st_size > 50_000 and (out / "traversability.npz").is_file()
        assert not (out / "summary.json").exists()  # no walk in plan-only mode


def _perfect_survey_run(tmp: Path, hole_xy=None) -> Path:
    """A drone_lab survey run directory whose map is the terrain ground truth on the survey grid (40 m area)."""
    from drone_lab.mapping import GridSpec, ground_truth_map
    from drone_lab.paths import Area
    from drone_lab.terrain import SURVEY_TERRAIN_CFG, generate_terrain_mesh

    grid = GridSpec.for_survey(Area(-20.0, 20.0, -20.0, 20.0), 0.25)
    z, _ = ground_truth_map(*generate_terrain_mesh(SURVEY_TERRAIN_CFG), grid, supersample=3)
    if hole_xy is not None:  # an area the drone did not see
        xs, ys = grid.cell_centers()
        (x0, x1), (y0, y1) = hole_xy
        z[np.ix_((ys > y0) & (ys < y1), (xs > x0) & (xs < x1))] = np.nan
    run = tmp / "survey"
    run.mkdir()
    np.save(run / "elevation_map.npy", np.asarray(z, dtype=np.float32))
    (run / "map_meta.json").write_text(json.dumps({"source": "test", "grid": grid.to_dict()}))
    return run


def test_survey_site_apron_matches_the_terrain_ground_truth():
    """The apron added around a survey map is exactly the simulated terrain's flat border."""
    require("isaaclab")
    require("trimesh")
    from drone_lab.mapping import ground_truth_map
    from drone_lab.teaming.inputs import extend_with_apron, load_elevation, survey_site_bounds
    from drone_lab.terrain import SURVEY_TERRAIN_CFG, generate_terrain_mesh

    terrain, site = survey_site_bounds()
    assert terrain == (-20.0, 20.0, -20.0, 20.0) and site == (-26.0, 26.0, -26.0, 26.0)
    with tempfile.TemporaryDirectory() as tmp:
        z, grid, _ = load_elevation(_perfect_survey_run(Path(tmp)))
    z2, g2, apron = extend_with_apron(z, grid, terrain, site)
    truth, _ = ground_truth_map(*generate_terrain_mesh(SURVEY_TERRAIN_CFG), g2, supersample=3)
    assert apron.sum() == 16928 and np.isfinite(truth[apron]).all()
    assert np.abs(z2[apron] - truth[apron]).max() < 1e-6


def test_rescue_finale_cli_starts_on_the_landing_pad_of_a_survey_map():
    """A drone_lab survey map does not reach the landing pad on the border: the apron is added and the robot starts
    on the pad; --no_apron moves the start onto the map edge (logged); a start with no free cell nearby is reported
    once, not once per person."""
    require("isaaclab")
    require("trimesh")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    script = str(PROJECT / "scripts" / "rescue_finale.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = _perfect_survey_run(tmp, hole_xy=((8.0, 14.0), (8.0, 14.0)))
        pins = tmp / "pins.json"
        pins.write_text(json.dumps({"pins": [
            {"class": "landing_pad", "x": -23.0, "y": -18.0},
            {"class": "person", "x": -4.5, "y": -12.5, "pose": "lying"},
            {"class": "person", "x": -16.0, "y": 5.0, "pose": "standing"},
        ]}))

        def cli(name, *extra):
            out = tmp / name
            r = subprocess.run([sys.executable, script, "--plan_only", "--run", str(run), "--pins", str(pins),
                                "--out", str(out), *extra], capture_output=True, text=True, timeout=300, env=env,
                               cwd=str(PROJECT))
            plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").is_file() else None
            return r, plan

        r, plan = cli("pad")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert "added the site's flat apron" in r.stdout and plan["map_mode"] == "drone_map"
        assert plan["start_xy"] == plan["start_requested_xy"] == [-23.0, -18.0] and plan["apron_cells"] == 16928
        assert plan["grid"] == {"x_min": -26.125, "y_min": -26.125, "resolution": 0.25, "nx": 209, "ny": 209}
        assert plan["plan"]["success"] and plan["person_xy"] == [-4.5, -12.5]
        assert np.linalg.norm(np.subtract(plan["path_xy"][0], [-23.0, -18.0])) < 0.2

        r, plan = cli("no_apron", "--no_apron")
        assert "WARNING: the start [-23.0, -18.0] is outside the map" in r.stdout and "onto the map edge" in r.stdout
        assert plan["start_xy"] == [-19.625, -18.0] and plan["apron_cells"] == 0 and plan["grid"]["nx"] == 161

        r, plan = cli("blocked", "--start", "11", "11")  # inside the unseen hole: no free cell within 2 m
        assert r.returncode == 2 and plan["plan"]["failure"] == "start_blocked"
        assert r.stdout.count("no route from the start") == 1 and "no route to person" not in r.stdout


if __name__ == "__main__":
    main_for(globals(), "test_teaming_task")
