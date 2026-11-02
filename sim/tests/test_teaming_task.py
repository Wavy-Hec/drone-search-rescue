"""The rescue tasks and the finale script (no simulator).

* both rescue tasks keep the policy interface of the velocity task their checkpoint was trained on (custom quadruped:
  ``Isaac-Velocity-Rough-Custom-Quad-v0``; Go2: ``Isaac-Velocity-Rough-UnitreeGo2``), on the survey terrain with the
  path-following command and deterministic events; the custom quadruped walks the dressed SAR site on Isaac Sim
  PhysX with the 2.x execution settings;
* the site's textured terrain stays a plain collider (the textures move to a render-only copy of the mesh);
* the robot profiles; the default plan; the script's CPU plan mode on the ground truth, a drone_lab survey map and a
  drone_gcs mission report.

Needs env_isaaclab (Isaac Lab configs, no simulator): ``CUDA_VISIBLE_DEVICES= python tests/test_teaming_task.py``.
Skipped elsewhere.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _runner import main_for, require

RESCUE_GO2 = "DroneLab-Rescue-Go2-v0"
BASE_GO2 = "Isaac-Velocity-Rough-UnitreeGo2"
RESCUE_QUAD = "DroneLab-Rescue-CustomQuad-v0"
BASE_QUAD = "Isaac-Velocity-Rough-Custom-Quad-v0"
POLICY_TERMS = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
]
_GROUP_SETTINGS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
    "history_order",
}
PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = str(PROJECT / "scripts" / "rescue_finale.py")


def _parse(task, overrides=()):
    require("isaaclab_tasks")
    from isaaclab_tasks.utils import parse_env_cfg

    import drone_lab.tasks  # noqa: F401

    if task in (RESCUE_QUAD, BASE_QUAD):
        require("quadruped_lab")
        import quadruped_lab.tasks  # noqa: F401  (registers the custom quadruped's velocity tasks)
    return parse_env_cfg(task, device="cpu", overrides=list(overrides))


def _terms(group):
    return [k for k, v in group.__dict__.items() if k not in _GROUP_SETTINGS and v is not None]


def _check_policy_interface(cfg, base):
    """The checkpoint of ``base`` sees exactly the same inputs and drives the same robot in ``cfg``."""
    pol, bpol = cfg.observations.policy, base.observations.policy
    assert _terms(pol) == _terms(bpol) == POLICY_TERMS
    for name in POLICY_TERMS:
        t, b = getattr(pol, name), getattr(bpol, name)
        assert t.func is b.func and t.params == b.params and t.scale == b.scale and t.clip == b.clip, name
        assert t.history_length == b.history_length, name
    assert pol.concatenate_terms and pol.enable_corruption is False  # play semantics: no noise
    active = [k for k, v in cfg.observations.__dict__.items() if v is not None]
    assert active == [k for k, v in base.observations.__dict__.items() if v is not None]
    a, b = cfg.actions.joint_pos, base.actions.joint_pos
    assert (a.scale, a.use_default_offset, a.joint_names) == (b.scale, b.use_default_offset, b.joint_names)
    assert (cfg.sim.dt, cfg.decimation) == (base.sim.dt, base.decimation) == (0.005, 4)
    robot, brobot = cfg.scene.robot, base.scene.robot
    assert robot.spawn.usd_path == brobot.spawn.usd_path
    assert robot.init_state == brobot.init_state and robot.actuators == brobot.actuators
    hs, bhs = cfg.scene.height_scanner, base.scene.height_scanner
    assert (hs.prim_path, hs.offset.pos, hs.ray_alignment) == (bhs.prim_path, bhs.offset.pos, bhs.ray_alignment)
    assert (hs.pattern_cfg.resolution, list(hs.pattern_cfg.size)) == (
        bhs.pattern_cfg.resolution,
        list(bhs.pattern_cfg.size),
    )
    assert hs.mesh_prim_paths == ["/World/ground"]
    cmd, bcmd = cfg.commands.base_velocity, base.commands.base_velocity
    for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        assert tuple(getattr(cmd.ranges, k)) == tuple(getattr(bcmd.ranges, k)), k


def _check_rescue_mdp(cfg, base):
    """Survey terrain, path-following command, deterministic events, rescue terminations, intact play mode."""
    from drone_lab.tasks.rescue.mdp import PathFollowCommandCfg, reset_root_on_route, route_complete
    from drone_lab.terrain import SURVEY_TERRAIN_CFG

    ter = cfg.scene.terrain
    assert ter.terrain_type == "generator" and ter.use_terrain_origins is False
    gen = ter.terrain_generator
    assert gen.seed == SURVEY_TERRAIN_CFG.seed and gen.size == SURVEY_TERRAIN_CFG.size
    assert (gen.num_rows, gen.num_cols, gen.border_width) == (4, 4, 6.0) and gen.curriculum is False
    assert sorted(gen.sub_terrains) == sorted(SURVEY_TERRAIN_CFG.sub_terrains)
    assert [k for k, v in cfg.curriculum.__dict__.items() if v is not None] == []
    cmd = cfg.commands.base_velocity
    assert isinstance(cmd, PathFollowCommandCfg) and cmd.asset_name == "robot"
    assert 0.0 < cmd.cruise_speed <= cmd.ranges.lin_vel_x[1] and cmd.resampling_time_range[0] >= 1e6
    ev = {k: v for k, v in cfg.events.__dict__.items() if v is not None}
    assert set(ev) == {"physics_material", "reset_base", "reset_robot_joints"}
    assert ev["reset_base"].func is reset_root_on_route and ev["reset_robot_joints"].params["position_range"] == (1, 1)
    terms = {k: v for k, v in cfg.terminations.__dict__.items() if v is not None}
    assert set(terms) == {"time_out", "base_contact", "flipped", "goal_reached"}
    assert terms["goal_reached"].func is route_complete and terms["goal_reached"].time_out is True
    # a fall means what it meant in training
    bc = base.terminations.base_contact.params["sensor_cfg"].body_names
    assert terms["base_contact"].params["sensor_cfg"].body_names == bc
    assert cfg.scene.num_envs == 1 and cfg.episode_length_s >= 120.0
    # play mode keeps the survey area intact (the velocity base would shrink it to 5 x 5 tiles)
    cfg.play_mode()
    assert cfg.scene.terrain.terrain_generator.num_rows == 4 and cfg.observations.policy.enable_corruption is False


def test_rescue_task_registrations():
    require("isaaclab_tasks")
    import gymnasium as gym

    import drone_lab.tasks  # noqa: F401

    for rescue, entry, base_agent in (
        (
            RESCUE_GO2,
            "drone_lab.tasks.rescue.config.go2.rescue_env_cfg:RescueGo2EnvCfg",
            "isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        ),
        (
            RESCUE_QUAD,
            "drone_lab.tasks.rescue.config.custom_quad.rescue_env_cfg:RescueCustomQuadEnvCfg",
            "quadruped_lab.tasks.locomotion.config.custom_quad.agents.rsl_rl_ppo_cfg:CustomQuadRoughPPORunnerCfg",
        ),
    ):
        spec = gym.spec(rescue)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"] == entry
        # the base task's runner, so `isaaclab play --task <rescue> --checkpoint <its checkpoint>` works
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == base_agent


def test_go2_rescue_task_keeps_the_go2_policy_interface():
    cfg, base = _parse(RESCUE_GO2), _parse(BASE_GO2)
    _check_policy_interface(cfg, base)
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.sim.use_newton_actuators == base.sim.use_newton_actuators
    # physics: the backend the checkpoint was trained on by default, PhysX selectable
    assert type(cfg.sim.physics).__name__ == "NewtonCfg"
    assert type(_parse(RESCUE_GO2, ["physics=isaacsim_physx"]).sim.physics).__name__ == "PhysxCfg"
    _check_rescue_mdp(cfg, base)
    # the plain survey terrain in rocky soil with a collision-free person at the goal
    assert cfg.scene.terrain.visual_material.mdl_path.endswith("Soil_Rocky.mdl")
    assert cfg.scene.terrain.visual_material.world_or_object is True
    assert cfg.scene.person.spawn.collision_props.collision_enabled is False
    assert getattr(cfg.scene, "sar_site", None) is None


def test_custom_quad_rescue_task_keeps_its_policy_interface_on_physx():
    cfg, base = _parse(RESCUE_QUAD), _parse(BASE_QUAD)
    from drone_lab.tasks.rescue.site import RescueSiteCfg

    _check_policy_interface(cfg, base)
    assert cfg.actions.joint_pos.scale == 0.25 and cfg.scene.height_scanner.prim_path == "{ENV_REGEX_NS}/Robot/trunk"
    assert cfg.scene.robot.spawn.usd_path.endswith("custom_quad/robot.usd")
    # Isaac Sim PhysX only, with the execution settings under which the converted 2.x checkpoint walks as in 2.3.2
    for overrides in ((), ("physics=isaacsim_physx",)):
        c = _parse(RESCUE_QUAD, overrides)
        assert type(c.sim.physics).__name__ == "PhysxCfg" and type(base.sim.physics).__name__ == "PhysxCfg"
        assert c.sim.physics.enable_external_forces_every_iteration is False and c.sim.use_newton_actuators is False
        assert c.sim.physics.gpu_max_rigid_patch_count == base.sim.physics.gpu_max_rigid_patch_count
    for backend in ("newton_mjwarp", "physx"):
        try:
            _parse(RESCUE_QUAD, [f"physics={backend}"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"physics={backend} must not resolve")
    _check_rescue_mdp(cfg, base)
    assert cfg.terminations.base_contact.params["sensor_cfg"].body_names == ["trunk", ".*_thigh"]
    # respawn like the training resets: from the init height (0.719 m) above the ground
    assert cfg.commands.base_velocity.spawn_height_offset == 0.0
    # the drone mission's dressed site (seed 0) on the scene's own terrain, its lights, no extra person
    site = cfg.scene.sar_site
    assert site.prim_path == "/World/sar" and isinstance(site.spawn, RescueSiteCfg) and site.spawn.seed == 0
    assert site.spawn.func == "drone_lab.tasks.rescue.site:spawn_rescue_site"
    assert site.spawn.terrain_generator.seed == cfg.scene.terrain.terrain_generator.seed
    assert cfg.scene.terrain.visual_material is None and cfg.scene.sun_light is not None
    assert cfg.scene.person is None
    assert _parse(RESCUE_QUAD, ["env.scene.sar_site.spawn.seed=3"]).scene.sar_site.spawn.seed == 3


def test_apply_rescue_task_is_robot_agnostic():
    require("isaaclab_tasks")
    from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

    from drone_lab.tasks.rescue.rescue_env_cfg import apply_rescue_task, person_asset_cfg, use_plain_terrain

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
    from drone_lab.sar.scenario import pose_quaternion

    assert np.allclose(pose_quaternion("lying_face_up"), person.init_state.rot)
    standing = person_asset_cfg((1.0, 2.0), 0.3)
    assert standing.init_state.pos == (1.0, 2.0, 0.3) and np.allclose(standing.init_state.rot, (0, 0, 0, 1))
    # the dressed site instead of one person, and back
    site_cfg = apply_rescue_task(UnitreeGo2RoughEnvCfg(), site_seed=4, person=person)
    assert site_cfg.scene.sar_site.spawn.seed == 4 and site_cfg.scene.person is None
    assert site_cfg.scene.sar_site.spawn.terrain_generator == site_cfg.scene.terrain.terrain_generator
    assert site_cfg.scene.terrain.visual_material is None
    use_plain_terrain(site_cfg, person)
    assert site_cfg.scene.sar_site is None and site_cfg.scene.sun_light is None and site_cfg.scene.person is person
    assert site_cfg.scene.terrain.visual_material.mdl_path.endswith("Soil_Rocky.mdl")


def test_textured_site_terrain_stays_a_plain_collider():
    """The ground-material subsets move to a render-only copy; the collider keeps its geometry and physics material."""
    require("isaaclab")
    require("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from drone_lab.tasks.rescue.site import move_ground_materials_to_visual_mesh

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/ground/terrain").AddTranslateOp().Set((1.0, 2.0, 0.0))
    mesh = UsdGeom.Mesh.Define(stage, "/World/ground/terrain/mesh")
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0.2), (0, 1, 0.1)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    physics_material = UsdShade.Material.Define(stage, "/World/ground/terrain/physicsMaterial")
    UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        physics_material, UsdShade.Tokens.strongerThanDescendants, "physics"
    )
    binding = UsdShade.MaterialBindingAPI(mesh.GetPrim())
    looks = {name: UsdShade.Material.Define(stage, f"/World/sar/Looks/ground_{name}") for name in ("grass", "soil")}
    for i, (name, material) in enumerate(looks.items()):
        subset = binding.CreateMaterialBindSubset(f"ground_{name}", [i], UsdGeom.Tokens.face)
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)

    assert move_ground_materials_to_visual_mesh("/World/ground", "/World/sar/ground_visual", stage=stage) == 2
    # the collider: same geometry and physics material, no subsets, hidden
    prim = mesh.GetPrim()
    assert prim.HasAPI(UsdPhysics.CollisionAPI) and not UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim))
    assert UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0].GetPath() == Sdf.Path(
        "/World/ground/terrain/physicsMaterial"
    )
    assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
    # the copy: same faces in world coordinates, the subsets with their materials, no collider
    visual = stage.GetPrimAtPath("/World/sar/ground_visual")
    assert visual.GetTypeName() == "Mesh" and not visual.HasAPI(UsdPhysics.CollisionAPI)
    assert list(UsdGeom.Mesh(visual).GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 0, 2, 3]
    world = UsdGeom.Xformable(visual).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert np.allclose(world.ExtractTranslation(), (1.0, 2.0, 0.0))
    subsets = {s.GetPrim().GetName(): s for s in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(visual))}
    assert sorted(subsets) == ["ground_grass", "ground_soil"]
    for i, name in enumerate(("grass", "soil")):
        sub = subsets[f"ground_{name}"]
        assert list(sub.GetIndicesAttr().Get()) == [i] and sub.GetFamilyNameAttr().Get() == "materialBind"
        bound = UsdShade.MaterialBindingAPI(sub.GetPrim()).GetDirectBinding().GetMaterialPath()
        assert bound == Sdf.Path(f"/World/sar/Looks/ground_{name}")
    assert UsdGeom.Imageable(visual).ComputeVisibility() == UsdGeom.Tokens.inherited
    # nothing left to move: no second copy
    assert move_ground_materials_to_visual_mesh("/World/ground", "/World/sar/other", stage=stage) == 0
    assert not stage.GetPrimAtPath("/World/sar/other").IsValid()


def test_robot_profiles():
    from drone_lab.teaming.robots import ROBOTS, quadruped_lab_dir

    assert list(ROBOTS) == ["custom", "go2"]  # the first one is the script's default
    custom, go2 = ROBOTS["custom"], ROBOTS["go2"]
    assert (custom.task, custom.physics) == (RESCUE_QUAD, "isaacsim_physx")
    assert (go2.task, go2.physics) == (RESCUE_GO2, "newton_mjwarp")
    assert Path(custom.checkpoint) == quadruped_lab_dir() / "checkpoints_2x" / "height_scanner_model_rslrl5.pt"
    if importlib.util.find_spec("quadruped_lab") is not None:
        assert Path(custom.checkpoint).is_file()
    # the custom quadruped is larger than a Go2 (0.89 m x 0.47 m): it needs more room and fits its plane to its body
    tc, tg = custom.traversability, go2.traversability
    assert tc.robot_radius_m > tg.robot_radius_m and tc.footprint_radius_m > tg.footprint_radius_m
    assert tc.max_slope_deg >= tg.max_slope_deg and tc.max_step_m >= tg.max_step_m
    assert custom.camera_scale > go2.camera_scale == 1.0


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


def _cli(out: Path, *args: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, SCRIPT, "--plan_only", "--out", str(out), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(PROJECT),
    )
    plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").is_file() else None
    return r, plan


def test_rescue_finale_plan_only_cli():
    """Default: the custom quadruped on the ground truth of SAR site 0 goes to a lying person 15-25 m from the landing
    pad, down into the crater; the Go2 on the plain terrain still plans to a standing person on the hill."""
    require("isaaclab")
    require("trimesh")
    from drone_lab.teaming.robots import ROBOTS

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "default"
        r, plan = _cli(out)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert plan["robot"] == "custom" and plan["map_mode"] == "ground_truth" and plan["sar_seed"] == 0
        assert plan["traversability_cfg"] == ROBOTS["custom"].traversability.to_dict()
        assert plan["start_xy"] == [-23.0, -18.0]  # the landing pad
        goal = plan["goal_pin"]
        assert goal["class"] == "person" and goal["pose"].startswith("lying")
        assert 15.0 <= np.hypot(goal["x"] + 23.0, goal["y"] + 18.0) <= 25.0
        assert plan["plan"]["success"] and plan["goal_check"] is None  # the pins are the truth here
        assert min(plan["path_z"]) < -1.0 and plan["plan"]["goal_snap_m"] < 2.0  # down on the crater floor
        assert len(plan["path_xy"]) == len(plan["path_z"]) > 100
        assert (out / "plan.png").stat().st_size > 50_000 and (out / "traversability.npz").is_file()
        assert not (out / "summary.json").exists()  # no walk in plan-only mode

        out = Path(tmp) / "go2"
        r, plan = _cli(out, "--robot", "go2", "--plain_terrain", "--goal", "-16", "5", "--person_pose", "standing")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert plan["robot"] == "go2" and plan["sar_seed"] is None and plan["pins_source"] == "none"
        assert plan["plan"]["success"] and plan["person_xy"] == [-16.0, 5.0]
        assert plan["traversability_cfg"] == ROBOTS["go2"].traversability.to_dict()


def _perfect_survey_map():
    """The terrain ground truth on the survey grid (40 m area), as a drone survey would map it perfectly."""
    from drone_lab.mapping import GridSpec, ground_truth_map
    from drone_lab.paths import Area
    from drone_lab.terrain import SURVEY_TERRAIN_CFG, generate_terrain_mesh

    grid = GridSpec.for_survey(Area(-20.0, 20.0, -20.0, 20.0), 0.25)
    z, _ = ground_truth_map(*generate_terrain_mesh(SURVEY_TERRAIN_CFG), grid, supersample=3)
    return z, grid


def _perfect_survey_run(tmp: Path, hole_xy=None) -> Path:
    """A drone_lab survey run directory whose map is the terrain ground truth on the survey grid."""
    z, grid = _perfect_survey_map()
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
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = _perfect_survey_run(tmp, hole_xy=((8.0, 14.0), (8.0, 14.0)))
        pins = tmp / "pins.json"
        pins.write_text(
            json.dumps(
                {
                    "pins": [
                        {"class": "landing_pad", "x": -23.0, "y": -18.0},
                        {"class": "person", "x": -4.5, "y": -12.5, "pose": "lying"},
                        {"class": "person", "x": -16.0, "y": 5.0, "pose": "standing"},
                    ]
                }
            )
        )
        common = ("--robot", "go2", "--plain_terrain", "--run", str(run), "--pins", str(pins))

        r, plan = _cli(tmp / "pad", *common)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert "added the site's flat apron" in r.stdout and plan["map_mode"] == "drone_map"
        assert plan["start_xy"] == plan["start_requested_xy"] == [-23.0, -18.0] and plan["apron_cells"] == 16928
        assert plan["grid"] == {"x_min": -26.125, "y_min": -26.125, "resolution": 0.25, "nx": 209, "ny": 209}
        assert plan["plan"]["success"] and plan["person_xy"] == [-4.5, -12.5]
        assert np.linalg.norm(np.subtract(plan["path_xy"][0], [-23.0, -18.0])) < 0.2

        r, plan = _cli(tmp / "no_apron", *common, "--no_apron")
        assert "WARNING: the start [-23.0, -18.0] is outside the map" in r.stdout and "onto the map edge" in r.stdout
        assert plan["start_xy"] == [-19.625, -18.0] and plan["apron_cells"] == 0 and plan["grid"]["nx"] == 161

        r, plan = _cli(tmp / "blocked", *common, "--start", "11", "11")  # inside the unseen hole
        assert r.returncode == 2 and plan["plan"]["failure"] == "start_blocked"
        assert r.stdout.count("no route from the start") == 1 and "no route to person" not in r.stdout


def test_rescue_finale_cli_plans_on_a_drone_gcs_mission_report():
    """A drone_gcs report (maps/ + detections.json): the robot plans on the drone's map with the drone's pins only
    (a false alarm, a missed crate), starts on the landing pad and is told how far the pin is from the real person."""
    require("isaaclab")
    require("trimesh")
    from drone_lab.sar.scene import SarSiteCfg, scenario_for

    z, grid = _perfect_survey_map()
    person = next(o for o in scenario_for(SarSiteCfg(seed=0)).ground_truth()["objects"] if o["name"] == "person_08")
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "mission"
        (report / "maps").mkdir(parents=True)
        np.save(report / "maps" / "elevation_mean.npy", np.asarray(z, dtype=np.float32))
        (report / "maps" / "grid.json").write_text(json.dumps({"elevation_grid": grid.to_dict(), "area": [-20, 20]}))
        (report / "detections.json").write_text(
            json.dumps(
                {
                    "frame": "world (local ENU: x east, y north, z up) [m]",
                    "objects": [
                        {"id": 1, "class": "landing_pad", "x": -22.8, "y": -18.1, "confirmed": True},
                        # person_08 of site 0 (lying in the crater), pinned 0.3 m off
                        {"id": 2, "class": "person", "x": person["x"] + 0.3, "y": person["y"], "confirmed": True},
                        {"id": 3, "class": "person", "x": 5.0, "y": 5.0, "confidence": 0.4, "confirmed": False},
                    ],
                }
            )
        )
        r, plan = _cli(Path(tmp) / "out", "--report", str(report))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert plan["map_mode"] == "drone_map" and plan["pins_source"].endswith("detections.json")
        assert [p["id"] for p in plan["pins"]] == ["1", "2"]  # confirmed pins only, no ground truth
        assert plan["apron_cells"] > 0 and plan["start_xy"] == [-23.0, -18.0]  # the site's landing pad
        assert plan["plan"]["success"] and plan["goal_pin"]["id"] == "2"
        check = plan["goal_check"]
        assert (check["true_person"], check["pose"]) == ("person_08", "lying_side")
        assert abs(check["pin_error_m"] - 0.3) < 1e-3
        assert "check: the pin marks person_08" in r.stdout


if __name__ == "__main__":
    main_for(globals(), "test_teaming_task")
