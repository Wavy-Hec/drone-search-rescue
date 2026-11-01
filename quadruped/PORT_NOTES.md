# Port notes: custom quadruped, Isaac Lab 2.3.2 (Isaac Sim 5.1) -> Isaac Lab 3.0 (Isaac Sim 6.2)

Date: 2026-11-01. Ported on CPU only (another job was using the GPU): no Isaac Sim launch, no simulation, no training,
no pip installs. The simulator checks planned in [section 9](#9-gpu-validation-plan) were run on the GPU the same day:
results, verified commands and open issues are in [section 14](#14-gpu-validation-2026-11-01).

Evidence paths below are relative to `~/isaac_ws/IsaacLab/source/` (the Isaac Lab 3.0 clone, release/3.0.0 @ 44044b096)
unless they start with `~`, `site-packages/` (= `~/anaconda3/envs/env_isaaclab/lib/python3.12/site-packages/`) or
`2.3.2:` (= `~/IsaacLab/source/`, the old clone).

## 1. Status

| Item | State |
|---|---|
| Project | `~/isaac_ws/projects/quadruped_lab`, generated with `isaaclab --new --non_interactive --task_type external --project_path ~/isaac_ws/projects --name quadruped_lab --author "Hector" --initial_content blank` (own git repo, Git LFS hooks installed with `git lfs install --local`) |
| Rough / Flat tasks | ported, all 3.0 API changes applied, config resolution + play mode verified on CPU |
| Height-scan checkpoint | converted to rsl-rl 5.5 (`checkpoints_2x/height_scanner_model_rslrl5.pt`), CPU outputs bit-identical (max abs diff 0.0) |
| Observation interface | identical to 2.x (term order, dims 235 = 48 + 187, noise, clip, scale), checked by `scripts/check_obs_parity.py` |
| Depth task | **experimental** redesign (camera on the trunk's front face, separate image group, CNN actor with a fixed learning rate, privileged critic); needs retraining. GPU check found and fixed a frozen camera (Isaac Sim PhysX + Newton Warp), section 14.6 |
| Tests | 22 kit-less tests pass (pytest 9.2.0, installed into `env_isaaclab` on 2026-11-01; the CPU port used pytest 9.0.2 from the `lab` env through `PYTHONPATH`) |
| Corrections | ten issues found after the first version, each checked against the sources and fixed ([section 12](#12-corrections-after-the-first-version)) |
| Simulation | **GPU-validated on 2026-11-01** ([section 14](#14-gpu-validation-2026-11-01)): joint order as trained; the converted 2.x checkpoint walks and, with two 2.x parity switches, matches its 2.x replay; training, resume, Newton and depth checked |
| Newton | opt-in: `physics=newton_mjwarp` loads `robot_newton.usd` in the PhysX joint/body order; GPU-validated on 2026-11-01 ([section 13](#13-newton-asset-and-robot-preset)) |

## 2. What was ported (old -> new)

Originals were only read (`~/quadruped_rough_terrain`, `~/IsaacLab`, `~/isaac_ws/IsaacLab`); all copies used `cp -a`.

| 2.x source (`~/quadruped_rough_terrain/custom_quadruped/...`) | 3.0 project file (`src/quadruped_lab/...`) | Change |
|---|---|---|
| `robot/robot.usd` (4716 B, 2026-01-15, md5 4c40f5a3...) + `robot/configuration/*.usd` | `assets/data/robots/custom_quad/robot.usd` + `configuration/` | copied unchanged (canonical asset) |
| `~/IsaacLab/.../custom_quadruped/robot/robot.usd` (5558 B, 2026-02-23, md5 5325c9aa...) | `assets/data/robots/custom_quad/robot_feb23.usd` | copied unchanged, unused alternative |
| `custom_quad.py` | `assets/custom_quad.py` | package-relative USD path, schema fragments, renamed actuator limits, explicit `joint_effort_limit` |
| `rough_env_cfg.py` | `tasks/locomotion/config/custom_quad/rough_env_cfg.py` | 3.0 base import, PhysX default (+ `physx` alias), mass-event operation/distribution, `_PLAY` class removed |
| `flat_env_cfg.py` | `tasks/locomotion/config/custom_quad/flat_env_cfg.py` | depth-patch lines removed, `_PLAY` class removed |
| `rough_env_depth_cfg.py` | `tasks/locomotion/config/custom_quad/rough_env_depth_cfg.py` | **redesigned** (section 8) |
| `__init__.py` | `tasks/locomotion/config/custom_quad/__init__.py` | same 3 IDs, `-Play-v0` IDs dropped, `default_agent`, depth task rsl_rl only |
| `agents/rsl_rl_ppo_cfg.py` | `.../agents/rsl_rl_ppo_cfg.py` | `RslRlMLPModelCfg` actor/critic + `obs_groups`; new CNN runner for depth |
| `agents/skrl_{rough,flat}_ppo_cfg.yaml` | `.../agents/skrl_{rough,flat}_ppo_cfg.yaml` | `input: STATES` -> `OBSERVATIONS`; rough directory `unitree_a1_rough` -> `custom_quad_rough` |
| `agents/__init__.py` | `.../agents/__init__.py` | copied unchanged |
| `~/quadruped_rough_terrain/logs/*.pt` | `checkpoints_2x/{height_scanner_model,depth_camera_model}.pt` | copied unchanged (md5 a6f84f57..., ac605bb2...) |
| (new) | `checkpoints_2x/height_scanner_model_rslrl5.pt` | converted checkpoint |
| `commands` (2.x `train.py`/`play.py` lines) | `README.md`, section 9 | `isaaclab train/play`, `physics=isaacsim_physx`, no `--headless`/`--enable_cameras` |
| `git -C ~/IsaacLab diff` (depth-camera patch of the 2.3.2 base config) | not ported as a patch | camera lives in the depth task's own scene cfg; Rough/Flat never needed it |
| (new) | `scripts/convert_rslrl_checkpoint.py`, `scripts/check_obs_parity.py`, `tests/test_port.py`, `tests/test_registration.py` | conversion, parity check, kit-less tests |
| (new) | `scripts/check_joint_order.py`, `scripts/depth_camera_study.py` | joint-order check of a run log; depth-camera self-occlusion study |

Template files touched: `pyproject.toml` (description; dependency `isaaclab-dev[rsl-rl,skrl]` as the template writes
for these RL libraries), `.gitattributes` (+ `*.onnx` through LFS), `.gitignore` (+ `checkpoints_2x/exported/`),
`README.md` (rewritten for the conda workflow), `.vscode/tasks.json` (`isaaclab --editor` inside `env_isaaclab`
instead of `uv run`, which would build a separate .venv), `.vscode/tools/launch.template.json` (train/play entries).

Task registration follows the template: `pyproject.toml` declares
`[project.entry-points."isaaclab.tasks"] quadruped_lab = "quadruped_lab.tasks"`
(`../tools/template/templates/external/pyproject.toml.jinja:26-27`); the CLI loads that group
(`isaaclab/isaaclab/cli/__init__.py:48-55`); `tasks/__init__.py` calls `import_packages`
(`isaaclab_tasks/isaaclab_tasks/utils/importer.py:15`), which imports every sub-package and so runs the
`gym.register` calls in `config/custom_quad/__init__.py`. `isaaclab list_envs` filters by entry-point module, not by ID
prefix (`isaaclab/isaaclab/cli/commands/list_envs.py:91-110`), so the 2.x IDs are listed from the project root.

## 3. 3.0 API changes applied (with evidence)

| Change | Where in the port | 3.0 evidence |
|---|---|---|
| Velocity base moved: `isaaclab_tasks.manager_based.locomotion.velocity` -> `isaaclab_tasks.core.velocity` | rough cfg import | `isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:335` (`LocomotionVelocityRoughEnvCfg`) |
| Velocity family defaults to Newton MJWarp | `self.sim.physics.default = self.sim.physics.isaacsim_physx` | `velocity_env_cfg.py:72` (`default = newton_mjwarp`), `:54` (`isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10*2**15)`); `isaaclab_tasks/docs/CHANGELOG.rst:1018` |
| `physics=physx` is an auto preset (OvPhysX for kit-less runs) | aliased: `self.sim.physics.physx = self.sim.physics.isaacsim_physx` | `velocity_env_cfg.py:56`; `isaaclab/isaaclab/physics/physics_manager_cfg.py:68-78`; `isaaclab/isaaclab/app/sim_launcher.py:435` |
| `add_base_mass` defaults to `operation="scale"`, `distribution="log_uniform"`; `(-1, 3)` alone raises | set `operation="add"`, `distribution="uniform"` | `velocity_env_cfg.py:209-219`; `isaaclab/isaaclab/envs/mdp/events.py:1933` ("log-uniform bounds must be > 0"), `:281-291` (signature) |
| Contact sensor path `Robot/.*` -> `Robot/[^/]*` | inherited unchanged | `velocity_env_cfg.py:117`; the asset's bodies are direct children of the root (section 6), and the PhysX sensor treats the last segment as a name filter over contact-reporting descendants (verify notes) |
| `effort_limit`/`velocity_limit` deprecated -> `actuator_effort_limit`/`actuator_velocity_limit` | `assets/custom_quad.py` | `isaaclab/isaaclab/actuators/actuator_base_cfg.py:39,52,162-177`; `../docs/source/isaaclab/migrating.rst:1045-1080` |
| Explicit actuators keep the solver effort limit (2.x widened it to 1e9) | `joint_effort_limit=45.0` (>= actuator limit, as the docs advise) | `../docs/source/isaaclab/migrating.rst:1085-1091`; 2.3.2: `isaaclab/isaaclab/actuators/actuator_base.py:106,174-175`; asset authors `maxForce=500` |
| DC motors run through the native actuator path by default | explicit `self.sim.use_newton_actuators = True` | `isaaclab/isaaclab/sim/simulation_cfg.py:100-112`; same 4-quadrant clamp as 2.x: `site-packages/newton/_src/actuators/clamping/clamping_dc_motor.py:64-93` vs 2.3.2: `isaaclab/isaaclab/actuators/actuator_pd.py:295-306` |
| `RigidBodyPropertiesCfg`/`ArticulationRootPropertiesCfg` deprecated (removal 3.2) | `[UsdPhysicsRigidBodyCfg, PhysxRigidBodyCfg]`, `[PhysxArticulationCfg, NewtonArticulationCfg]` | `isaaclab/isaaclab/sim/schemas/schemas_cfg.py:168`; `isaaclab_physx/isaaclab_physx/sim/schemas/schemas_cfg.py:318,377,653`; `isaaclab_newton/isaaclab_newton/sim/schemas/schemas_cfg.py:748`; bare fragment lists target all bodies carrying the API in the subtree: `isaaclab/isaaclab/sim/spawners/from_files/from_files.py:340-361` |
| `cfg.replace(...)` -> `isaaclab.utils.replace(cfg, ...)` | rough cfg | `isaaclab/isaaclab/utils/configclass.py:142` |
| `-Play` registrations and `_PLAY` classes removed; `isaaclab play` applies `play_mode()` | `-Play-v0` IDs and `_PLAY` classes dropped | `isaaclab_tasks/docs/CHANGELOG.rst:1779-1783`; `isaaclab_rl/isaaclab_rl/entrypoints/backends/play_rsl_rl.py:107`; `isaaclab_tasks/isaaclab_tasks/utils/hydra.py:682-683`; `velocity_env_cfg.py:377-389`; `isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py:101-125` |
| rsl-rl 5.5 model cfgs replace `policy=RslRlPpoActorCriticCfg` / `empirical_normalization` | agents cfg | `isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:22,48,117,165,265,340`; legacy path warns and is unsupported from 3.1: `isaaclab_rl/isaaclab_rl/rsl_rl/utils.py:122-131` |
| skrl >= 2.1: `STATES` means the state space | yamls use `OBSERVATIONS` | `isaaclab_rl/isaaclab_rl/skrl.py:39`; `site-packages/skrl/utils/model_instantiators/torch/common.py:258-259`; 3.0 A1 yaml: `isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/a1/agents/skrl_rough_ppo_cfg.yaml:22,31` |
| `mdp.image` removed -> `mdp.image_depth` | depth task | `isaaclab/docs/CHANGELOG.rst:709`; `isaaclab/isaaclab/envs/mdp/observations.py:470-519`; re-exported by the velocity mdp: `isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/__init__.pyi:30` |
| `AdditiveUniformNoiseCfg` alias gone -> `UniformNoiseCfg` (same semantics) | depth task | `isaaclab/isaaclab/utils/noise/noise_cfg.py:41`; 2.3.2: `isaaclab/isaaclab/utils/noise/__init__.py:34` |
| `TiledCameraCfg` -> `CameraCfg`; offset quaternion `(w,x,y,z)` -> `(x,y,z,w)` | depth task | `isaaclab/isaaclab/sensors/camera/tiled_camera_cfg.py:17`; `isaaclab/isaaclab/sensors/camera/camera_cfg.py:35,46,100,199` |
| Hard-coded `~/IsaacLab` USD path | `QUADRUPED_LAB_ASSETS_DIR / "robots/custom_quad/robot.usd"` | template: `../tools/template/templates/external/__init__assets:10` |

All 40 symbols the project imports resolve on the CPU (checked with an import resolver that records each definition;
e.g. `RslRlMLPModelCfg` `isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:21`, `CameraCfg` `isaaclab/isaaclab/sensors/camera/camera_cfg.py:34`).

## 4. Decisions

| Topic | Choice | Evidence / reason |
|---|---|---|
| Canonical robot.usd | Jan-15 copy as `robot.usd`; Feb-23 kept as `robot_feb23.usd` | `height_scanner_model.pt` (2026-01-29) postdates Jan 15 and predates the Feb-23 re-save. A `pxr` comparison shows the two root layers differ **only** in the saved viewport camera (`customLayerData` camera position/target); joints, bodies, masses, drives and limits of the composed stages are identical, so the A/B test is low priority. |
| Physics default | Isaac Sim PhysX for all three tasks; Newton via `physics=newton_mjwarp` | parity with the 2.x checkpoints; the 3.0 velocity base defaults to Newton (`velocity_env_cfg.py:72`) |
| `physics=physx` | aliased to Isaac Sim PhysX inside the cfg | `PhysxAutoCfg` picks OvPhysX for kit-less runs (`physics_manager_cfg.py:68-78`); OvPhysX is not installed and would give train/play backend mismatch |
| Actuator execution | 3.0 native path (`use_newton_actuators=True`) | non-deprecated path; the Newton DC-motor clamp is the same formula as 2.x (citations in section 3). A/B switch: `env.sim.use_newton_actuators=false`. **GPU (section 14.2): on Isaac Sim PhysX this path feeds the clamp joint positions that are 1-3 substeps old (3.0 defect); pass the switch to replay or fine-tune the 2.x checkpoint** |
| Solver effort limit | `joint_effort_limit=45.0` | 3.0 docs recommend `>= actuator_effort_limit`; asset authors 500 N*m so even `None` would not clip; behaviour equals 2.x's 1e9 because the DC model already clips to 45 |
| TGS external forces every iteration | 3.0 default (`True`) kept | default flipped in 3.0 and `False` is deprecated (`isaaclab_physx/docs/CHANGELOG.rst:781-783`, `isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py:166`); 2.3.2 had `False` (`2.3.2: isaaclab/isaaclab/sim/simulation_cfg.py:123`). A/B: `env.sim.physics.enable_external_forces_every_iteration=false`. **GPU (section 14.2): `false` cuts the 2.x policy's tracking error by a quarter to a third; pass it for the 2.x checkpoint** |
| Joint/body ordering | `None` (backend-native) | the checkpoint was trained with Isaac Sim PhysX's native order; the `"physx"` alias means OvPhysX order, which can differ (`../docs/source/how-to/transfer_policies_between_physx_and_newton.rst:353-368`) |
| Task IDs | keep `Isaac-Velocity-{Flat,Rough,Rough-Depth}-Custom-Quad-v0` | as decided; gymnasium accepts `-v0`, no 3.0 code treats it specially |
| `-Play-v0` IDs | dropped | no real need: Hector's `commands` file played with the training ID, and `isaaclab play` applies `play_mode()`. If wanted, 3.0 supports deprecated aliases via a `"deprecated": {"alias": ...}` kwarg (`isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py:71-84`, example `isaaclab_tasks/isaaclab_tasks/contrib/dr_legs/__init__.py:24-38`) |
| `default_agent` | `rsl_rl` on all tasks | like 3.0 Go2 (`isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/__init__.py:23`); `--rl_library` becomes optional (`isaaclab_rl/isaaclab_rl/entrypoints/dispatch.py:145-163`) |
| Robot cfg location | `quadruped_lab/assets/custom_quad.py` | template README: keep asset configuration in the `<name>.assets` package |
| Depth task critic | asymmetric: MLP on proprio + privileged height scan | 3.0 camera tasks use a state critic with a camera actor (`isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py:100-108`); far cheaper than a second CNN |
| Depth renderer | Newton Warp default, `renderer=isaacsim_rtx` available | section 8 |
| Depth camera mount | front face of the trunk, pos (0.47, 0, 0.08), the 2.x orientation (50 deg down) | the 2.x mast mount (0.7 m above, 0.25 m behind the trunk origin) puts the robot in 33 % of the pixels; the redesign is retrained anyway, so nothing ties it to the 2.x pose (section 8) |
| Depth camera update period | 0.0 (render when the observation reads it) | lazy sensor updates render once per policy step either way; a positive period adds a host sync per read and stale poses late in long episodes (section 8) |
| Depth PPO schedule | fixed learning rate 7e-5, 8 mini-batches; NaN check on until the first smoke run | the 3.0 camera-actor setting (section 8) |
| skrl hyperparameters | unchanged | the rough yaml still carries the A1 values (entropy 0.01, value 1.0, 36000 timesteps) unlike the rsl_rl cfg; not changed silently |

## 5. Observation layout and interface parity (height-scan checkpoint)

Output of `PYTHONPATH=src python scripts/check_obs_parity.py --markdown`: the 2.x side is parsed from the 2.3.2 base
(`git show HEAD:` of `velocity_env_cfg.py`) plus Hector's `rough_env_cfg.py`; the 3.0 side is the instantiated
`CustomQuadRoughEnvCfg`; dims come from the USD (12 revolute joints), the ray pattern evaluated on the CPU and the
checkpoint's `actor.0.weight` (512 x 235).

| # | term | function | dims | index range | noise U(min, max) | clip | 2.x == 3.0 |
|---|---|---|---|---|---|---|---|
| 0 | `base_lin_vel` | `base_lin_vel` | 3 | 0-2 | (-0.1, 0.1) | None | yes |
| 1 | `base_ang_vel` | `base_ang_vel` | 3 | 3-5 | (-0.2, 0.2) | None | yes |
| 2 | `projected_gravity` | `projected_gravity` | 3 | 6-8 | (-0.05, 0.05) | None | yes |
| 3 | `velocity_commands` | `generated_commands` | 3 | 9-11 | None | None | yes |
| 4 | `joint_pos` | `joint_pos_rel` | 12 | 12-23 | (-0.01, 0.01) | None | yes |
| 5 | `joint_vel` | `joint_vel_rel` | 12 | 24-35 | (-1.5, 1.5) | None | yes |
| 6 | `actions` | `last_action` | 12 | 36-47 | None | None | yes |
| 7 | `height_scan` | `height_scan` | 187 | 48-234 | (-0.1, 0.1) | (-1.0, 1.0) | yes |
| | **total** | | **235** | 0-234 | | | checkpoint input 235 |

No term has a scale; corruption is on during training and off in play mode (both versions). The 2.x patch's
`depth_scan` term came after `height_scan` and the rough task set it to None, so it never affected the layout. The
term functions differ only by the 3.0 `.torch` accessor (`base_lin_vel` is the root COM velocity in the base frame in
both: 2.3.2 `isaaclab/isaaclab/assets/articulation/articulation_data.py:1041-1048`, 3.0
`isaaclab/isaaclab/assets/articulation/base_articulation_data.py:1394-1402`). Noise -> clip -> scale order is the same
(3.0 `isaaclab/isaaclab/managers/observation_manager.py:446-461`). The height-scan grid code is identical
(`GridPatternCfg(resolution=0.1, size=[1.6, 1.0])` -> 17 x 11 = 187 rays, `velocity_env_cfg.py:108-116`).

Other interface items (same script):

| Item | 2.x | 3.0 port |
|---|---|---|
| Action | `JointPositionAction`, joints `[".*"]`, scale 0.25, default offset | same |
| Joint order | Isaac Sim PhysX native order | same (ordering `None`): breadth-first in authored joint order, FL/FR/RL/RR hip, then thigh, then calf (docs above). **Verified on the GPU** (section 14.1) |
| Sim dt / decimation / control | 0.005 s / 4 / 50 Hz | same |
| Episode length | 20 s | same |
| Init root pos / default joint pos | z 0.719; hip +pi/45, thigh -5pi/18, calf 5pi/9 | same |
| DC motor | effort 45, saturation 45, velocity 21 (torque-speed curve), kp 60, kd 1.5, friction 0 | same values; solver effort 45 (2.x 1e9, asset 500), solver velocity limit stays the asset's 10 rad/s in both |
| Mass randomization | trunk + U(-1, 3) kg, inertia rescaled | same (explicit `operation`/`distribution`) |
| Commands, rewards, terminations, events, curriculum, sensors, terrain generator | as configured | every explicit 2.x value equal (214 values, section 5.1); the term functions that changed keep their semantics (section 5.2) |
| PhysX scene | TGS, `gpu_max_rigid_patch_count=327680` | same, except `enable_external_forces_every_iteration` True (see decisions) |

Deliberate differences: physics preset handling (`physx` alias), native actuator path, the TGS flag above, and the
rsl-rl 5 std clamp `(1e-6, 1e6)` (the checkpoint's std 0.249-0.313 is far inside, and play uses the mean).

### 5.1 What `check_obs_parity.py` compares

* The policy observation terms above, against the checkpoint's input size.
* Action term, sim dt / decimation / render interval, episode length, init pose, default joint positions and DC-motor
  parameters. The 2.x values are parsed from the 2.x sources, none is typed into the script.
* Every value the 2.x configs set explicitly. The script evaluates the class bodies and `__post_init__` assignments of
  the 2.3.2 base (`git show HEAD:`), then Hector's `rough_env_cfg.py` overrides, with `ROUGH_TERRAINS_CFG` taken from
  the 2.3.2 `terrains/config/rough.py`. It compares the result with the instantiated 3.0 config: 214 values covering
  actions, commands, rewards, terminations, events, curriculum, the height scanner (offset, ray alignment, grid
  pattern, mesh paths, update period), the contact sensor, the terrain importer, generator and sub-terrains, timing,
  number of envs and `gpu_max_rigid_patch_count`. A term parameter that a config leaves out is compared through the
  default of its function (2.3.2 source vs. 3.0 signature), so `add_base_mass`'s implicit 2.x `distribution="uniform"`
  is checked as well. `--verbose` lists every compared path. Two known differences are reported, not failed: the
  contact-sensor path `Robot/.*` -> `Robot/[^/]*`, and `sim.physx.*` moving into the physics preset
  (`sim.physics.default`).
* A test perturbs three 3.0 values (a reward weight, a defaulted event parameter and a sub-terrain range) and checks
  that the script reports all three (`tests/test_port.py::test_parity_with_2x_sources`).

Not covered by the script: defaults of the config classes that neither side sets, rendering-only settings (sky light,
terrain visual material) and the bodies of the term functions. Manual diffs cover them. The relevant
ones: `ContactSensorCfg.force_threshold=None` resolves to 1.0 on PhysX
(`isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py:133-135`, as in 2.x), the 3.0 height scanner
adds the Newton-only `global_world_only=True` (`isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:115`),
and the functions in 5.2.

### 5.2 Term functions that changed in 3.0 (semantics preserved)

| Term (task name) | 3.0 change | Why the behaviour is the same |
|---|---|---|
| observation terms | `.torch` accessors; `torch.linalg.norm` instead of `torch.norm` | same quantities (section 5 above) |
| `illegal_contact` (`base_contact`) | reads `net_normal_forces_w_history` (`isaaclab/isaaclab/envs/mdp/terminations.py:175-178`) | 3.0 renamed the PhysX output: the contact view only reports normal forces, and `net_forces_w` now returns `net_normal_forces_w` with a warning (`isaaclab_physx/docs/CHANGELOG.rst:421-426`) |
| `undesired_contacts` | same rename (`isaaclab/isaaclab/envs/mdp/rewards.py:303-307`) | set to None by the task |
| `joint_torques_l2` (`dof_torques_l2`) | reads `asset.actuators.applied_effort` instead of `data.applied_torque` (`rewards.py:168-176`) | the native actuator path writes the clipped DC-motor effort into that buffer every step (`isaaclab/isaaclab/actuators/newton/physx_runtime.py:143`, `sync_torque_telemetry`) |
| `reset_root_state_uniform` (`reset_base`) | now a `ManagerTermBase` class (`isaaclab/isaaclab/envs/mdp/events.py:1244`) | same parameters (its `__call__` signature is compared by the script) and the same uniform sampling |
| `apply_external_force_torque` (`base_external_force_torque`) | returns early when both ranges are zero (`events.py:1198-1200`) | the task's ranges are zero, so 2.x applied a zero wrench: no force either way |

## 6. Robot asset facts (read with `pxr`, no Kit)

* Default prim `/quadruped_robot`, Z-up, meters; articulation root on `trunk`; 17 rigid bodies, all direct children:
  trunk 9.70 kg, per leg hip 1.16 / thigh 1.24 / calf 0.66 / foot 0.015 kg (about 22 kg total).
* 12 revolute DOFs `{FL,FR,RL,RR}_{hip,thigh,calf}_joint` + 4 fixed foot joints. Limits: hip +-35 deg,
  thigh -70..0 deg, calf 20..140 deg. Authored drive `maxForce` 500 N*m, `physxJoint:maxJointVelocity`
  572.96 deg/s (= 10 rad/s), joint friction 0.1 (the actuator cfg overrides it to 0, as in 2.x).
* `robot.usd` references `configuration/robot_{base,physics,robot,sensor}.usd` with **relative** paths; the stage
  composes without errors from the project location.
* Body names used by the task exist: `trunk`, `.*_thigh`, `.*_foot`.

## 7. Checkpoints and conversion

| File | Format | Notes |
|---|---|---|
| `checkpoints_2x/height_scanner_model.pt` | rsl-rl 3.x `model_state_dict` | copy of `~/quadruped_rough_terrain/logs/height_scanner_model.pt`; iter 13900; 235 -> [512, 256, 128] -> 12, ELU; scalar std 0.249-0.313; **no** observation normalizer tensors (`empirical_normalization=False`); Adam (1 group, 17 params, lr 1.139e-4) |
| `checkpoints_2x/height_scanner_model_rslrl5.pt` | rsl-rl 5.5 | `actor_state_dict {distribution.std_param, mlp.{0,2,4,6}.*}`, `critic_state_dict {mlp.{0,2,4,6}.*}`, optimizer copied, iter 13900 |
| `checkpoints_2x/depth_camera_model.pt` | rsl-rl 3.x | kept, **not converted**: 240 inputs (48 + 192) match no recorded camera config, the redesigned depth task uses a CNN |

Mapping: `std` -> `distribution.std_param`, `actor.N.*` -> actor `mlp.N.*`, `critic.N.*` -> critic `mlp.N.*`,
`{actor,critic}_obs_normalizer.{_mean,_var,_std,count}` -> `obs_normalizer.*` (identical buffers and formula in
3.1.2 and 5.5.1; none present in this checkpoint). The optimizer state is index-based and reusable because rsl-rl
3.1.2 optimized `ActorCritic.parameters()` (std first, `~/anaconda3/envs/lab/.../rsl_rl/modules/actor_critic.py:93`,
`ppo.py:104`) and rsl-rl 5.5.1 optimizes `chain(actor.parameters(), critic.parameters())` with the distribution
registered before the MLP (`site-packages/rsl_rl/algorithms/ppo.py:101`, `models/mlp_model.py:64-76`,
`modules/distribution.py:166`); the converter checks every Adam moment shape against the new order.

Rerun (CPU, from the project root):

```bash
PYTHONPATH=src python scripts/convert_rslrl_checkpoint.py checkpoints_2x/height_scanner_model.pt \
    checkpoints_2x/height_scanner_model_rslrl5.pt --force --verify
```

`--verify` builds the runner exactly like `isaaclab play` (`handle_deprecated_rsl_rl_cfg` -> `create_rsl_rl_runner` ->
`runner.load()` with strict state-dict loading, optimizer and iteration; `isaaclab_rl/isaaclab_rl/entrypoints/backends/play_rsl_rl.py:113-143`)
from `CustomQuadRoughPPORunnerCfg`, then compares against a hand-written Linear+ELU forward pass of the **old** weights
on 4096 random 235-dim observations (half of them scaled x3). Result (2026-11-01):

| Check | Max abs diff |
|---|---|
| actor mean actions (new `MLPModel` vs old weights) | 0.0 |
| critic values | 0.0 |
| std | 0.0 |
| TorchScript export (`exported/policy.pt` path used by play) | 0.0 |
| loaded iteration / optimizer lr | 13900 / 1.139e-4 |

The converter never writes in place (refuses `OUTPUT == INPUT`, needs `--force` to overwrite) and loads with
`weights_only=True`.

## 8. Depth task (EXPERIMENTAL redesign)

Why a redesign: the 2.x depth task depended on the local `~/IsaacLab` patch, whose 320x240 `TiledCameraCfg` +
`mdp.image` term was concatenated into the 1-D policy group, which cannot be built (a (240, 320, 1) image cannot be
concatenated with 1-D terms; same logic in 3.0 `isaaclab/isaaclab/managers/observation_manager.py:91-115`), and the
old checkpoint (240 inputs) matches no recorded config.

* **Camera** (`CameraCfg`, prim `{ENV_REGEX_NS}/Robot/trunk/depth_camera`): on the **front face of the trunk**, pos
  (0.47, 0, 0.08) (`DEPTH_CAMERA_POS`; the trunk's visual mesh ends at x = 0.459 m), with the 2.x orientation,
  `convention="opengl"`, rot (w,x,y,z) (0.66446, 0.24184, -0.24184, -0.66446) -> (x,y,z,w)
  (0.24184, -0.24184, -0.66446, 0.66446). It looks forward, pitched 50.0 deg down; pinhole f = 18.14756 mm, aperture
  20.955 mm -> 60.0 x 46.8 deg FOV. `update_latest_camera_pose=True` (body-mounted camera). The 2.x mast mount
  (-0.25, 0, 0.7) is not kept (section 12): from there the robot's own meshes fill a third of the image (table
  below).
* **Standing height**: at the default joint positions the trunk origin is 0.451 m above flat ground, not the spawn
  height 0.719 m, which is the straight-leg (zero-joint) height. This comes from forward kinematics of `robot.usd`
  (joint frames, checked against the authored rest pose) and Newton's USD importer + `eval_fk` on the CPU gives the
  same link positions. The front-face camera is therefore 0.53 m above the ground.
* **Far clip 1.0 -> 4.0 m** (`DEPTH_MAX_RANGE`). From the new mount the pixel rays reach flat ground after 0.56-1.32 m,
  so the 2.x 1.0 m clip would blank the top sixth of the image (16.6 % of the pixels); 4 m leaves headroom for downhill
  terrain and a pitched-up trunk. With the 2.x mount every ground ray was longer than 1.0 m (1.20-2.86 m).
* **Render rate**: `update_period=0.0`. With `InteractiveSceneCfg.lazy_sensor_update=True`
  (`isaaclab/isaaclab/scene/interactive_scene_cfg.py:96`) a sensor only advances its timestamp on each physics step
  (`isaaclab/isaaclab/sensors/sensor_base.py:197-212`) and the camera renders when the observation term reads
  `camera.data` (`isaaclab/isaaclab/sensors/camera/camera.py:345-349, 714-731`), i.e. once per policy step. The
  2.x patch's `update_period=0.0` therefore cost one frame per policy step too: 2.3.2 also renders only every
  `render_interval = decimation` physics steps (`2.3.2: isaaclab/isaaclab/envs/manager_based_rl_env.py:194`) and also
  updates sensors lazily. The port's earlier claim of "4x the work" was wrong (section 12). A positive period
  would not save renders. It would add a device-to-host sync on every read (`camera.py:1029-1032`, `env_mask.numpy()`
  unless the period is <= 0), and with float32 sensor timestamps (`sensor_base.py:270`) and a fixed `1e-6` tolerance
  (`isaaclab/isaaclab/sensors/kernels.py:29`) a 0.02 s period misses every other refresh once an episode passes
  ~16 s: 100 of the 1000 policy steps of a 20 s episode, measured with a float32 replica of the kernel. Only the
  envs that are due get their pose updated before the shared render (`camera.py:706-712`), so those frames would
  come from a 20 ms-old camera pose.
* **Observations**: `policy` (48 proprio, no height scan); `depth` = `mdp.image_depth(data_type="distance_to_camera",
  normalize=True, max_depth=4.0, invalid_value=4.0, channel_first=True)` -> (N, 1, 48, 64) in [0, 1], noise
  U(+-0.005) (= the 2.x +-2 cm in metric units), clip (0, 1); `privileged` = noise-free height scan for the critic only.
  Both renderers are set to `depth_clipping_behavior="max"` so a ray without a hit reads 1.0 (far), not Newton's 0.0
  (`isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer_cfg.py:58-69`) or RTX `inf`.
* **Model**: `RslRlCNNModelCfg` actor, conv [16, 32, 32] / kernels [5, 3, 3] / stride 2 -> 768 features + 48 proprio ->
  MLP [512, 256, 128] -> 12 (Gaussian, init std 1.0); MLP critic on 48 + 187. `obs_groups = {"actor": ["policy",
  "depth"], "critic": ["policy", "privileged"]}`. rsl-rl's `CNNModel` takes (B, C, H, W) groups
  (`site-packages/rsl_rl/models/cnn_model.py:140`).
* **PPO** (section 12): the rough settings with `schedule="fixed"`, `learning_rate=7e-5`, `num_mini_batches=8`
  (`DEPTH_PPO_ALGORITHM_CFG`). Isaac Lab 3.0 trains its camera actors with a fixed rate because "the adaptive KL
  schedule varies it across the range where encoder features change faster than the policy head can track, and does
  not converge" (`isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py:80-82`, the same
  three values). The other 3.0 CNN-actor configs use a fixed 1.5e-4
  (`contrib/ur10_particle_push/agents/rsl_rl_ppo_cfg.py:113-115`), a fixed 3e-4
  (`core/cartpole/agents/rsl_rl_ppo_cfg.py:116-118`) and an adaptive schedule starting at 1e-4 with 8 mini-batches
  (`core/lift/config/franka_soft/agents/rsl_rl_ppo_cfg.py:21-22,92`), none of them the MLP's adaptive 1e-3, so 7e-5 is a
  conservative, untuned start. `check_for_nan` stays on for now. The 3.0 camera runners turn it off
  (`kuka_allegro/agents/rsl_rl_ppo_cfg.py:107-108`) because rsl-rl scans every observation group with
  `torch.isnan(...).any()` and syncs with the host on every env step
  (`site-packages/rsl_rl/runners/on_policy_runner.py:89`, `rsl_rl/utils/utils.py:290-305`). `image_depth` already
  replaces NaN/inf pixels (`isaaclab/isaaclab/utils/images.py:145`, `nan_to_num`), so after a NaN-free smoke run pass
  `agent.check_for_nan=false`. That override was checked on the CPU.
* **Renderer**: Newton Warp by default. It is the kit-less Warp ray tracer "for training workflows where throughput
  matters", produces `distance_to_camera`, and is documented to pair with Isaac Sim PhysX
  (`../docs/source/concepts/renderers.rst:86-93`); for a 64x48 depth-only image the RTX path tracer is wasted work.
  `renderer=isaacsim_rtx` switches to the legacy Isaac RTX renderer (Kit) for an A/B test. OVRTX is excluded: it cannot
  share a process with Kit, which Isaac Sim PhysX requires, and its runtime is not installed (`renderers.rst:95-103`).
  A Newton Warp camera is not a "Kit camera", so no RTX camera rendering is enabled for it
  (`isaaclab/isaaclab/app/sim_launcher.py:144-159`).
* **skrl**: not registered for the depth task (the MLP yaml cannot consume a 2-D group).

What the camera sees (`PYTHONPATH=src python scripts/depth_camera_study.py --markdown`). Each of the 3072 pixel rays
is intersected with the robot's own visual and collision meshes at the default pose (362,138 and 104,672 triangles)
and with flat ground. Both renderers draw the visual meshes: Newton's USD importer brings in only the 34 visual shapes
of this asset (risk 4), and RTX does not draw the `guide`-purpose collision meshes. x is measured forward from the trunk
origin; the front feet stand at x = 0.22 m.

| Mount | Geometry | Robot pixels | Nearest ground x, centre / foot tracks | Farthest ground x | Ground ray length | > 1.0 m |
|---|---|---|---|---|---|---|
| port: front face (0.47, 0, 0.08), 50 deg | visual / collision | **0.0 % / 0.0 %** | 0.63 / 0.63 m | 1.51 m | 0.56-1.32 m | 16.6 % |
| 2.x mast (-0.25, 0, 0.7), 50 deg | visual | 33.1 % | 1.26 / 0.89 m | 2.01 m | 1.20-2.86 m | 100 % |
| 2.x mast (-0.25, 0, 0.7), 50 deg | collision | 29.4 % | 1.21 / 0.20 m | 2.01 m | 1.20-2.86 m | 100 % |
| front face, 55 deg (option) | visual / collision | 0.0 % / 0.0 % | 0.58 / 0.58 m | 1.32 m | 0.54-1.13 m | 5.4 % |
| front face, 60 deg (option) | visual / collision | 0.0 % / 0.0 % | 0.54 / 0.54 m | 1.18 m | 0.54-1.00 m | 0.0 % |

An earlier estimate for the 2.x mount, with box approximations of the trunk at the spawn height of 0.719 m, was
19-33 % robot pixels. Even from the front face the ground closer than ~0.6 m ahead of the trunk origin stays out of
view, and the camera sees nothing to the sides or behind. A memoryless CNN actor therefore cannot see where its feet are
now (risk 5; frame stacking or a recurrent actor in section 11). A steeper pitch brings the near edge closer at the
cost of look-ahead (rows 3-4). Change it in `DEPTH_CAMERA_ROT`; `pitch_down_quat()` in `scripts/depth_camera_study.py`
gives the (x, y, z, w) value for a pitch.

Memory budget on the 16 GB RTX 5080 (float32; rollout storage keeps every observation group,
`site-packages/rsl_rl/storage/rollout_storage.py:141-148`). The depth runner uses 8 mini-batches:

| Resolution | Envs | Rollout per env-step | Rollout (24 steps) | PPO mini-batch (1/8) | CNN activations per mini-batch |
|---|---|---|---|---|---|
| 64x48 (port default) | 1024 | 13,393 B | 0.31 GiB | 3072 samples | 0.37 GiB (x~2 with gradients) |
| 64x48 | 2048 | 13,393 B | 0.61 GiB | 6144 | 0.74 GiB |
| 64x48 | 4096 | 13,393 B | 1.23 GiB | 12288 | 1.48 GiB |
| 320x240 (2.x patch) | 1024 | 308,305 B | 7.06 GiB | 3072 | 11.7 GiB |

Per env-step: obs (48 + 64*48 + 187) * 4 B = 13,228 B plus actions/rewards/values/log-probs/returns/advantages/
mean/std = 165 B. CNN activations: 32,384 floats per sample (conv + ELU outputs, measured with the rsl-rl `CNN`).
Camera output 64*48*4 B * 1024 = 12 MiB. Adding rough estimates for Kit (~2-3 GB), Isaac Sim PhysX for 1024 robots
(~1-2 GB), the Newton Warp scene/BVH (~0.5 GB, the rough terrain is ~2.6 M triangles) and the CUDA context (~0.5 GB),
1024 envs should need roughly 6-8 GB, leaving headroom; 2048 envs is likely to fit (~10-11 GB); 4096 is not
recommended. The 2.x 320x240 config would need more than 30 GB. **These are estimates; measure in the GPU phase.**
**Measured on 2026-11-01** (training, 1024 envs, peak of the process): 5.5 GB with Newton Warp, 7.3 GB with RTX
(section 14.6).

## 9. GPU validation plan

The plan written on the CPU. It was run on 2026-11-01; the results, the commands as verified and the open issues are in
[section 14](#14-gpu-validation-2026-11-01). Where the two differ, section 14 wins: replay metrics come from
`scripts/eval_policy.py` (not from resumed training), and the 2.x checkpoint needs the two parity switches of 14.2.

Run from the project root with the env active. The first Isaac Sim 6.2 launch can take more than 10 minutes.

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES                       # only if the EULA has not been accepted yet
cd ~/isaac_ws/projects/quadruped_lab

# 1. install the project (isaaclab-dev is not on any index -> --no-deps; pip fetches uv_build for the build)
python -m pip install --no-deps -e .
python -c "import importlib.metadata as m; print([e for e in m.entry_points(group='isaaclab.tasks') if e.name == 'quadruped_lab'])"
isaaclab list_envs                                    # expect the 3 task IDs
python -m pip install pytest && python -m pytest tests -q   # 18 kit-less tests

# 2. zero-action smoke tests (bounded, headless; zero_agent opens a newton_gl window unless --viz none)
mkdir -p logs
isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 200 --viz none --info \
    physics=isaacsim_physx 2>&1 | tee logs/zero_agent_rough.log
python scripts/check_joint_order.py logs/zero_agent_rough.log
#    PASS = exit 0 and "joint order OK": the action term and the articulation table both list
#    FL_hip FR_hip RL_hip RR_hip FL_thigh FR_thigh RL_thigh RR_thigh FL_calf FR_calf RL_calf RR_calf (*_joint).
#    FAIL (exit 1) = do not judge the replay; either set the expected order explicitly (quote the names: the override
#    is parsed with ast.literal_eval, unquoted names stay one string),
#    "env.scene.robot.joint_ordering=['FL_hip_joint','FR_hip_joint',...]" (articulation_cfg.py:71-83), or retrain.
#    Exit 2 = no joint order in the log (forgot --info, or the run died before the env was built).
#    Also check: robot spawns from the package path, no "No rigid-body targets matched" warning, no contact-sensor error.
isaaclab zero_agent --task Isaac-Velocity-Flat-Custom-Quad-v0 --num_envs 16 --max_steps 300 --viz none physics=isaacsim_physx

# 3. replay the converted 2.x checkpoint, recorded to video (stops after the clip)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --video --video_length 500 physics=isaacsim_physx
#    video: checkpoints_2x/videos/play/ ; exports: checkpoints_2x/exported/policy.{pt,onnx} (both git-ignored)
#    interactive viewer: replace --video ... with --viz kit (runs until the window is closed)
#    if the gait is off, A/B one at a time:
#      env.sim.use_newton_actuators=false
#      env.sim.physics.enable_external_forces_every_iteration=false
#      env.scene.robot.spawn.usd_path=$PWD/src/quadruped_lab/assets/data/robots/custom_quad/robot_feb23.usd
#    judge each variant by numbers, not by eye: `isaaclab play` prints no metrics, so resume-train briefly with a fixed
#    seed and compare "Mean episode length" (1000 steps = no falls), "Episode_Termination/base_contact" and
#    "Episode_Reward/track_lin_vel_xy_exp" in the console / TensorBoard (logs/rsl_rl/custom_quad_rough/<run>):
#      isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 60 \
#          --seed 42 --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" physics=isaacsim_physx [override]
#    (the policy keeps learning during those 60 iterations, so compare variants with the same seed and length)

# 4. short training runs
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 5 physics=isaacsim_physx
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 20 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" physics=isaacsim_physx   # resume incl. Adam state
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-Custom-Quad-v0 --num_envs 1024 --max_iterations 5 physics=isaacsim_physx
isaaclab train --rl_library skrl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 256 --max_iterations 5 physics=isaacsim_physx

# 5. depth smoke test (EXPERIMENTAL)
isaaclab zero_agent --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 16 --max_steps 200 --viz none physics=isaacsim_physx
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 64 --max_iterations 3 physics=isaacsim_physx \
    --capture_env_sensors 2 --capture_env_sensors_length 20 --capture_env_sensors_interval 50 --capture_env_sensors_format file
#    frames: logs/rsl_rl/custom_quad_rough_depth/<run>/sensor_frames/train -> ground fills the image from the bottom row
#    up, no fixed robot blob, flat ground ~0.14-0.33 (0.56-1.32 m / 4 m), misses = 1.0; no NaN error from the runner
#    -> from then on add agent.check_for_nan=false
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 1024 --max_iterations 10 physics=isaacsim_physx
#    meanwhile in another shell: nvidia-smi --query-gpu=memory.used --format=csv -l 5   (budget in section 8)
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 64 --max_iterations 3 physics=isaacsim_physx renderer=isaacsim_rtx

# 6. optional: Newton MJWarp (physics=newton_mjwarp loads robot_newton.usd in the PhysX joint/body order, section 13)
isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 300 --viz none \
    physics=newton_mjwarp 2>&1 | tee logs/zero_agent_rough_newton.log
#    kit-less runs print no INFO records even with --info (no log handler; section 13), so check_joint_order.py has
#    nothing to read here
```

Never use `physics=physx` on the command line out of habit; it is aliased to Isaac Sim PhysX in these configs, but other
tasks would resolve it to OvPhysX.

## 10. Known risks and open items

1. ~~**Nothing has run in the simulator yet.**~~ **Resolved on 2026-11-01** (section 14): USD loading, the schema
   fragments, contact-sensor matching and both actuator paths work under Isaac Sim 6.2 PhysX. The native actuator
   path has an execution defect on PhysX (stale joint positions, section 14.2).
2. **Joint order**: the policy depends on Isaac Sim 6.2 PhysX producing the same DOF order as 5.1. Expected
   breadth-first FL/FR/RL/RR per level; `scripts/check_joint_order.py` checks the step-2 log. A mismatch makes the
   replayed policy fall within a few dozen steps. **Verified on 2026-11-01**: "joint order OK" for the action term and
   the articulation table; the replay times out in 78-82 % of episodes (zero / random actions: 0 %).
3. **Dynamics drift and config provenance**: the native actuator path and the TGS external-force flag differ in
   execution from 2.x (same model math), and the 2.x checkpoint came from an unrecorded January setup. On top of that,
   `rough_env_cfg.py` was edited 4 minutes **after** the checkpoint file was written: `height_scanner_model.pt` mtime
   2026-01-29 00:06:30, `rough_env_cfg.py` 00:10:31 (`flat_env_cfg.py` 07:34:23 and `commands` 07:30:29 are later
   still; the `~/IsaacLab` copies carry the same mtimes). The only bytecode, the 2026-02-23 `__pycache__` in
   `~/IsaacLab`, was compiled from the edited file (pyc header: source mtime 00:10:31, size 3576), so the pre-edit
   rewards/events cannot be recovered. The observation interface still matches the weights (`actor.0.weight` 512 x
   235), but bit parity of the training config is not provable. Judge the replay by behaviour and metrics (step 3), and
   when resumed training regresses, consider config drift (events, rewards) alongside backend drift (section 12).
   **Measured on 2026-11-01**: the original checkpoint in the 2.x stack (Isaac Lab 2.3.2 / Isaac Sim 5.1, play config,
   128 envs, seed 42) falls in 18.8 % ± 3.4 of first episodes, with lin / yaw tracking error 0.187 / 0.103. The 3.0
   replay with `env.sim.use_newton_actuators=false env.sim.physics.enable_external_forces_every_iteration=false`
   gives 15.6-21.1 % and 0.185-0.212 / 0.096-0.107 (seeds 42-44). The port default tracks worse: 0.243-0.298 /
   0.150-0.168. Resuming 20 PPO iterations with those two switches keeps the replay level (play-mode reward/ep 1.497 vs
   1.471, 3 seeds x 256 episodes). Evidence: `runs/2026-11-01_quadruped_gpu/training/results/first_episode_2x_vs_30.md`,
   `eval_resumed_vs_original.md`.
4. **Newton**: **resolved on 2026-11-01 with an opt-in copy, see [section 13](#13-newton-asset-and-robot-preset)**
   (`robot_newton.usd`, selected by `physics=newton_mjwarp`, PhysX joint/body order). Original finding: Newton's USD
   importer (`newton.ModelBuilder.add_usd`, run
   on the CPU) imports the 34 visual shapes of `robot.usd` and **no collision shape**: all 17 collision prims are
   skipped with "CollisionAPI applied to an unknown UsdGeomGPrim type", because this importer layout puts
   `PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI` (convexDecomposition) on the `node_STL_BINARY_` Xform above each
   mesh, not on the mesh. Isaac Sim PhysX accepts that layout (the 2.x training ran on it); on Newton the robot would
   have no contacts. Fix it in a copy of the asset (move the two APIs onto the Mesh prims; see
   `../docs/source/how-to/prepare_asset_for_newton.rst`), never in place. Newton's model also lists the joints in
   authored prim order, i.e. per leg (FL hip, thigh, calf, then FR ...), not breadth-first like Isaac Sim PhysX.
   Replaying the PhysX checkpoint there needs an explicit `joint_ordering` list (transfer guide `:264-269`; the expected
   list is what `scripts/check_joint_order.py` prints).
5. **Depth task**: Newton Warp + Isaac Sim PhysX in one process is a documented pairing but unexercised here. Self-
   occlusion is gone at the default pose (0 % robot pixels, section 8); a leg swung far forward may still show at the
   bottom edge. The camera cannot see the ground closer than ~0.6 m ahead of the trunk origin, beside the robot or
   behind it, so a memoryless CNN actor has to infer what is under its feet; lateral and backward velocity commands
   are blind. The hyperparameters are untuned. **GPU check 2026-11-01** (section 14.6): the pairing runs, but the
   camera stayed at its spawn pose until `body_mounted_image_depth` was added; memory measured (5.5 GB at 1024 envs).
   No depth policy has been trained beyond 10 iterations yet.
6. **Packaging**: `pip install --no-deps -e .` builds with `uv_build`, which pip downloads into an isolated build env
   (network needed). The `[tool.uv.sources]` paths are relative to `../../IsaacLab`.
7. **Git LFS**: binaries are committed as LFS objects in this local repo; a remote must support LFS before pushing.
8. Upstream deprecation warnings (e.g. `RigidBodyMaterialCfg` in the 3.0 velocity base terrain) are not from this
   project.
9. `max_iterations=80000` is inherited from 2.x (the converted policy has 13900); override with `--max_iterations`.
10. **Feet probably never touch the ground** (2.x and 3.0 alike; found while computing the camera geometry). Each calf's
    collision mesh contains a ~3.7 cm ball around the foot centre: calf vertices extend 2.1-6.0 cm from the foot centre
    in all 26 sampled directions. The `*_foot` link is only a 1.5 cm sphere, so it sits fully inside the calf ball, and
    at the default pose the calf reaches 2.2 cm lower than the foot sphere. Ground contact therefore goes through the
    `*_calf` bodies. The `feet_air_time` reward (contact sensor on `.*_foot`) would never see a first contact and stays
    0. That matters most for Flat (weight 0.25). Parity is unaffected (same asset in 2.x). Check it in the step-4 training
    runs: if `Episode_Reward/feet_air_time` stays exactly 0 while the robot walks, the feet never register a contact.
    **Confirmed on 2026-11-01, in 3.0 and in 2.x** (`runs/2026-11-01_quadruped_gpu/training`):
    * Geometry: all 27,072 foot collision vertices of each leg lie inside the convex hull of the calf collision mesh,
      at least 6.1 mm from its boundary. The hull reaches 2.11 cm or more from the foot centre in every direction (foot
      radius 1.51 cm). The foot is welded to the calf, so the calf meets the ground or a stair edge first in every pose.
    * 3.0: `Episode_Reward/feet_air_time` is 0.0000 in every iteration of every training run (rough and flat from
      scratch, resume from the 2.x checkpoint). Evaluations show foot contact 0.00 and calves 0.55, on Newton as well.
    * 2.x: in Isaac Lab 2.3.2 / Isaac Sim 5.1 with the original checkpoint (128 envs, 1100 steps, run from a copy of
      `~/IsaacLab/source`), the largest foot contact force was 0.0 N, with 0 foot first-contacts and 0.0 paid by
      feet_air_time. A calf touched the ground in 98.9 % of env-steps.
    * For new training (breaks checkpoint parity): point `feet_air_time` at `.*_calf`, or shrink the calf collider in a
      copy of the asset so that the foot sphere protrudes.
11. **Reset drop**: `init_state.pos` z = 0.719 m is the straight-leg height, but the default joint positions bend the
    legs to a 0.451 m stance, so every reset drops the robot ~0.27 m. Kept for parity with the checkpoint.
12. **Height-scan refresh late in episodes** (upstream, 2.x and 3.0 alike): the scanner's `update_period=0.02` with
    float32 timestamps skips every other refresh after ~16 s of an episode. 2.3.2 has the same logic in torch
    (`2.3.2: isaaclab/isaaclab/sensors/sensor_base.py:187,216`), so the checkpoint was trained with it. Kept for parity.

## 11. Possible improvements

* **Depth via ray casting instead of rendering**: a `RayCasterCameraCfg` / `MultiMeshRayCasterCameraCfg` against
  `/World/ground` gives terrain-only depth with Warp ray casts (like the height scanner), no renderer, no
  self-occlusion, works on PhysX and Newton, and is far cheaper per env. `mdp.image_depth` accepts ray-caster cameras
  (`isaaclab/isaaclab/envs/mdp/observations.py:470-473`); 3.0 ships such presets for kuka_allegro
  (`isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/camera_cfg.py:82-107`).
* **Depth policy memory**: the forward camera cannot see the ground under the robot (section 8), so give the actor
  memory: `frame_stack=2` or more on `image_depth` (`isaaclab/isaaclab/envs/mdp/observations.py:470-519`), or an
  `RslRlRNNModelCfg` actor. Mixed precision (`RslRlPpoAlgorithmCfg.use_mixed_precision`, `rl_cfg.py:220`) for the
  CNN update; `agent.check_for_nan=false` once validated (saves the host syncs of the NaN scan).
* **Depth camera tuning**: pitch 55-60 deg brings the near edge to 0.54-0.58 m (section 8 table); a 2 m far clip would
  double the dynamic range of the normalized image on flat ground (then use noise +-0.01 to keep +-2 cm).
* **Feet contact** (risk 10, confirmed on the GPU in 3.0 and 2.x): point `feet_air_time` and any foot-contact term
  at `.*_calf` for new training, or fix a copy of the asset so the foot link carries the contact ball.
* **For new training only** (each breaks parity with the 2.x checkpoint): spawn at ~0.47 m instead of 0.719 m (risk 11);
  `height_scanner.update_period=0.0` (risk 12, no extra ray casts under lazy updates); raise the solver velocity limit
  from the asset's 10 rad/s to the DC motor's 21 rad/s with `joint_velocity_limit=21.0` on the `base_legs` actuator
  (`isaaclab/isaaclab/actuators/actuator_base_cfg.py:79`). Today the 10 rad/s clamp, not the motor curve, bounds joint
  speed. The curve's torque limit at 10 rad/s is still 45 * (1 - 10/21) = 23.6 N*m.
* **A/B by numbers**: done. `scripts/eval_policy.py` (written during the GPU validation, section 14) runs the
  deterministic policy for complete episodes and reports time-outs vs. `base_contact`, tracking errors and the
  per-term episode rewards.
* **Faster PPO updates**: `torch_compile_mode="max-autotune-no-cudagraphs"` on the runner (`rl_cfg.py:355`).
* **Throughput**: train on Newton MJWarp (the 3.0 default for the velocity family): the asset is fixed (section 13)
  and Newton trains 3.9x faster than Isaac Sim PhysX here. Then check sim-to-sim on PhysX with the transfer guide
  (`isaaclab-transferring-policies-sim-to-sim` skill).
* **Gait quality**: a `base_height_l2` reward like 3.0's Go2 (`isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rough_env_cfg.py:22-35,70-72`)
  prevents crouching gaits; consider shorter `max_iterations` defaults.
* **Tooling**: install the template's dev group (pytest, ruff, pre-commit, codespell) and run
  `python -m pytest tests -q` and `pre-commit run --all-files` before commits. Until then, `ruff check` / `ruff format
  --check` with the ruff 0.12.0 binary from the base conda env (`~/anaconda3/bin/ruff`) pass on the whole project.
  pre-commit pins 0.14.10.

## 12. Corrections after the first version

Issues found in the first version, each checked against the 2.3.2 / 3.0 sources before changing anything. This was done on
the CPU only: no Kit, no GPU, no installs.

| # | Issue | Verdict | Change |
|---|---|---|---|
| 1 (major) | 2.x camera mount: the robot's own trunk hides the near-field terrain | **confirmed** with a mesh ray cast instead of boxes: 33.1 % (visual) / 29.4 % (collision) robot pixels; nearest visible ground 1.26 m ahead of the trunk origin on the centre line, 0.89 m in the foot tracks. The first version also assumed the spawn height 0.719 m; the real stance is 0.451 m (section 8) | camera on the trunk's front face, `DEPTH_CAMERA_POS = (0.47, 0, 0.08)`, 2.x orientation kept: 0 % robot pixels, ground from 0.63 m. New `scripts/depth_camera_study.py` reruns the estimate; `test_depth_camera_mount_sees_no_robot` guards it. Section 8, risk 5. The ray-caster camera stays an option (section 11) |
| 2 (minor) | depth runner inherits the adaptive 1e-3 schedule and the per-step NaN scan | **confirmed** (`kuka_allegro/agents/rsl_rl_ppo_cfg.py:80-82,107-108`; no 3.0 CNN-actor config uses the MLP's adaptive 1e-3: fixed 1.5e-4 / 3e-4, or adaptive from 1e-4) | `DEPTH_PPO_ALGORITHM_CFG` = rough settings + `schedule="fixed"`, `learning_rate=7e-5`, `num_mini_batches=8`; `check_for_nan` stays on until a NaN-free smoke run, then `agent.check_for_nan=false` (override checked on the CPU). Rough/flat PPO unchanged (test) |
| 3 (minor) | the joint-order check of the GPU plan cannot be seen in the log | **fix applied, evidence corrected**. In Kit runs (Isaac Sim PhysX) the Kit launcher already adds a handler that prints `isaaclab*` INFO records at the default WARNING level (`isaaclab_physx/isaaclab_physx/app/kit_launcher.py:116-132, 805-812`), so the table probably does print there. `SimulationCfg.logging_level` is never applied in 3.0 (`configure_logging` has no caller). Kit-less runs (Newton) drop INFO records unless `--info` is given (`isaaclab/isaaclab/app/sim_launcher.py:256-263, 641`). The table needs 18 lines after its header, so a `grep -A16` would cut off the last joint (`RR_calf_joint`) | GPU step 2 and the README use `--info ... \| tee logs/...log` plus the new `scripts/check_joint_order.py`. It parses the action-term line and the articulation table, derives the expected order from `robot.usd` with the documented rule (`transfer_policies_between_physx_and_newton.rst:353-356`), and exits 0 / 1 / 2 (pass / mismatch / not found). It is tested on logs produced with the same `PrettyTable` and logging format. The remedy for a mismatch is an explicit `joint_ordering` list |
| 4 (minor) | `rough_env_cfg.py` edited 4 min after the checkpoint was saved | **confirmed** (00:06:30 vs 00:10:31; the pyc header shows the same source mtime and size, 3576 B) | risk 3 rewritten: config parity is not provable; judge the replay by behaviour and metrics; consider config drift when resumed training regresses |
| 5 (nit) | "the 2.x camera rendered every physics step, 4x the work" is wrong | **confirmed** (2.3.2 renders every `render_interval` steps, `manager_based_rl_env.py:194`; both versions update sensors lazily) | **different fix**: instead of keeping `update_period = 0.02` and rewording, the camera now uses `update_period=0.0`, because a positive period adds a host sync per read (`camera.py:1029-1032`) and, with float32 timestamps, skips every other pose refresh after ~16 s of an episode (section 8). Comment, docstring and section 8 corrected; test updated |
| 6 (nit) | README: "headless unless --viz" is false for zero_agent/random_agent | **confirmed** (`isaaclab_rl/isaaclab_rl/entrypoints/simple_agents.py:265`) | README states the newton_gl default and the `--viz none` rule |
| 7 (nit) | README runs `git lfs pull` in a repo without a remote | **confirmed** (`git remote` empty; working-tree binaries md5-identical to the originals) | README runs `git lfs fsck` and explains `git lfs pull` for a fresh clone from an LFS-capable remote |
| 8 (nit) | "functions identical apart from .torch" is imprecise | **confirmed** (citations in 5.2) | section 5.2 lists the renamed fields and the class-based term, with why each keeps its semantics |
| 9 (nit) | parity script hard-codes the 2.x action/timing/episode values and covers only the observation terms | **confirmed** | the 2.x values are now parsed from the 2.x sources; a static evaluator of the 2.3.2 base + Hector's overrides compares 214 explicit settings (MDP terms with function defaults, sensors, terrain, timing). Known differences are listed. A negative-control test confirms that changed 3.0 values are caught. The scope and its limits are in section 5.1 |
| 10 (nit) | log-std checkpoints need `std_type='log'` and fail strict loading with no hint | **confirmed** (`rsl_rl/modules/distribution.py:165-168`; default `std_type="scalar"`, `isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:57`) | the converter prints the required agent cfg and `--verify` refuses a mismatched `std_type` up front with that hint. Tested with a synthetic log-std checkpoint (strict load into a log-std `MLPModel`, same mean and std). `height_scanner_model.pt` (scalar std) is unaffected |

Also found while verifying: the 0.451 m standing height (section 8, risk 11); the calf ball
that hides the foot spheres (risk 10); Newton's importer skipping every collision prim of the asset (risk 4); and the
float32 sensor-timestamp quirk (section 8, risk 12). None of them changes the Rough/Flat configs, which stay at
parity with the 2.x checkpoint.

Re-run after the fixes (CPU, 2026-11-01): `py_compile` on all 19 `.py` files; import resolution of all 42 imported
symbols (no Kit module loaded, CUDA not initialised); pytest 18/18; `check_obs_parity.py` OK (214 values);
`convert_rslrl_checkpoint.py --verify-only` max abs diff 0.0 (actions, values, std, TorchScript), iteration 13900;
`depth_camera_study.py` 0 % robot pixels for the configured mount; `ruff check` and `ruff format --check` (0.12.0)
clean. The checkpoints are unchanged (md5 identical before and after).


## 13. Newton asset and robot preset

Added on 2026-11-01 after the GPU validation; resolves risk 4 without changing `robot.usd` or the PhysX default.
Evidence: `~/isaac_ws/runs/2026-11-01_quadruped_gpu/training/` (job logs in `jobs/`, results in `results/`).

**Asset.** `robot_newton.usd` and `configuration/robot_physics_newton.usd` are small text layers written by
`scripts/make_newton_asset.py`. `robot_newton.usd` sublayers `robot.usd` and selects a new `Physics = "Newton"`
variant. Its payload, `robot_physics_newton.usd`, sublayers `robot_physics.usd`, deletes `PhysicsCollisionAPI` +
`PhysicsMeshCollisionAPI` from the 17 `node_STL_BINARY_` Xforms under `/colliders` and applies them to the Mesh prim
below each one. The instanceable `<body>/collisions` prims stay instanced, because their internal references resolve
in that layer stack. `--check` compares all 334 composed prims of both assets: only the collision APIs moved, plus the
approximation token. The original files are md5-identical before and after.

* **Approximation `convexHull`** (PhysX keeps `convexDecomposition` through `robot.usd`). Newton maps
  `convexDecomposition` to CoACD: 191 convex parts per robot (31 per hip, up to 4494 vertices per part). Rough task,
  1024 envs, zero actions: 5,156 env-steps/s with CoACD vs 21,166 with hulls (17 shapes, at most 64 vertices); env
  build 20.6 s vs 5.5 s. At the default pose, the hull moves the calf's lowest point by 2.6 mm.
* `robot.usd` on Newton: the env does not build:
  `RuntimeError: failed to initialize contact sensor ... No local entities but world_count=64`
  (CPU import: 0 collision shapes, 17 "CollisionAPI applied to an unknown UsdGeomGPrim type" warnings).
  `robot_newton.usd`: 17 collision shapes on 17 bodies.

**Preset.** `CustomQuadRobotCfg` in `rough_env_cfg.py` is keyed by the physics name, like 3.0's `ShadowHandRobotCfg`:
* `default` (also `isaacsim_physx`, `physx`): `CUSTOM_QUAD_CFG`, `robot.usd`, native order.
* `newton_mjwarp` / `newton_kamino`: `CUSTOM_QUAD_NEWTON_CFG` (`robot_newton.usd`), with `joint_ordering` and
  `body_ordering` set to explicit Isaac Sim PhysX lists (`CUSTOM_QUAD_PHYSX_JOINT_ORDER`, `CUSTOM_QUAD_PHYSX_BODY_ORDER`).

Resolved configs of all three tasks for `<default>` / `isaacsim_physx` / `physx`, in train and play mode, are
identical before and after the change. The Newton selections differ in exactly `spawn.usd_path`, `joint_ordering` and
`body_ordering`. Scripts that read the unresolved cfg (`check_obs_parity.py`, `depth_camera_study.py`) use
`cfg.scene.robot.default`. Tests: `test_robot_preset_per_backend`, `test_newton_asset_matches_canonical_asset`.

**Ordering.** Newton lists joints and bodies per leg: `FL_hip, FL_thigh, FL_calf, FR_hip, ...` and
`trunk, FL_hip, FL_thigh, FL_calf, FL_foot, ...`. Isaac Sim PhysX goes breadth-first: `FL_hip, FR_hip, RL_hip,
RR_hip, FL_thigh, ...` and `trunk, FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ...`. With the preset, Newton runs print
the PhysX order for `robot.joint_names`, `robot.body_names` and the action term, and Newton's order for
`backend_joint_names`.

The converted 2.x checkpoint, replayed on Newton (eval_policy.py, play mode, 128 envs x 2 episodes, seed 42):

| run | ep len | base_contact | time_out | lin / yaw err | reward/ep |
|---|---|---|---|---|---|
| PhysX best variant (reference) | 18.07 s | 18.0 % | 82.0 % | 0.187 / 0.101 | 1.491 |
| Newton, preset (PhysX order) | 18.46 s | 16.0 % | 84.0 % | 0.246 / 0.157 | 1.285 |
| Newton, native order (`joint_ordering=null body_ordering=null`) | 0.90 s | 99.6 % | 0.4 % | 0.884 / 1.288 | -0.099 |

The policy transfers to Newton zero-shot only through the remap.

**Zero actions** (flat task, 64 envs, 5 s, base_contact off): both backends land on the calves at about 0.3 s and sit
down. The lowest foot origin stays at +0.021 m or higher, with no NaN. Trunk height after 5 s: 0.32 m on PhysX, 0.32 to
0.34 m on Newton. On Newton 14 to 28 % of the robots end up on the trunk; on PhysX none do. Videos:
`videos/zero_agent_flat_{newton,physx}_rest.mp4`.

**Training benchmark** (rough task, 1024 envs, 30 iterations from scratch, seed 42; `steps/s` = median of
iterations 5-29; VRAM = the job's own processes). On an idle GPU: Newton preset 56,444 steps/s, 1,370 MiB, 14.5 s
training. Isaac Sim PhysX (best variant) 14,362 steps/s, 4,022 MiB, 48.3 s. Newton is 3.9x faster and needs 2.9x
less memory.

With other jobs on the GPU, Newton ran at 23,880-57,200 and PhysX at 10,208-13,494 steps/s. On the idle GPU,
removing the ordering remap (`joint_ordering=null body_ordering=null`: 56,819) or switching to the torch actuator path
(`env.sim.use_newton_actuators=false`: 55,808) changes nothing measurable. All four PhysX runs ended iteration 29 at
the same mean reward (-0.81, bit-reproducible); the Newton runs did not (-1.16 to -1.30). `newton_kamino` with the same
asset passes the zero-action stand check (calves at 0.3 s, no NaN); it was not trained.

**Logging.** Kit-less runs print no `INFO` records, even with `--info`. `apply_python_logging_level`
(`isaaclab/isaaclab/app/logging_utils.py:41-50`) only lowers the level of the root logger's existing handlers, and a
kit-less run has none, so `scripts/check_joint_order.py` finds nothing in a Newton log. With a
root handler installed (`logging.basicConfig(level=logging.INFO)` before `isaaclab.cli.cli()`), a Newton zero_agent
log passes `check_joint_order.py`: the action term and the articulation table list the PhysX order.

## 14. GPU validation (2026-11-01)

Run on the RTX 5080 (16 GB, driver 595.99.02) with Isaac Sim 6.2 + Isaac Lab 3.0 in `env_isaaclab`, Isaac Sim PhysX
unless a row says otherwise, while other jobs shared the GPU. Evidence (logs, JSON, scripts, videos, frames):
`~/isaac_ws/runs/2026-11-01_quadruped_gpu/{smoke,training,depth,finish}/`. Replay metrics come from
`scripts/eval_policy.py`: play-mode config, deterministic policy, every env run for complete episodes. The simulation is
deterministic for a fixed seed and number of envs: reruns, `robot_feb23.usd` and the repository copy of the script
reproduce earlier per-episode records bit for bit.

### 14.1 Verdicts

| Item | Verdict | Evidence |
|---|---|---|
| Install, entry point, `list_envs` | pass | `pip install --no-deps -e .`; the three task IDs are listed from the project root |
| Kit-less tests | 22 passed | `python -m pytest tests -q` |
| Joint order (Isaac Sim PhysX) | **as trained**: FL/FR/RL/RR hip, then thigh, then calf | `check_joint_order.py` on the `zero_agent --info` log: "OK action term JointPositionAction (12)", "OK articulation ... (12)", "joint order OK" |
| zero_agent rough / flat / depth | run clean | obs 235 / 48 / dict with a (16, 1, 48, 64) depth group; contact sensor and ray caster registered; no errors |
| Replay of the converted 2.x checkpoint | **walks**; with the two parity switches it matches its own 2.x replay | 14.2 |
| Training: rsl_rl rough and flat, skrl rough, resume with optimizer state | pass | 14.3 |
| `feet_air_time` | exactly 0 in every run, in 2.x as well (asset geometry) | risk 10, 14.5 |
| Newton | opt-in preset works; the checkpoint transfers through the joint-order remap; training is 3.9x faster | section 13, 14.4 |
| Depth task | the camera never moved on Isaac Sim PhysX + Newton Warp: **fixed** in the task cfg | 14.6 |
| Demo videos and the GUI watch command | recorded / verified | 14.8 |

Rerun on the final code, after all of the changes above were in place: pytest (22 passed), ruff, the parity check,
`make_newton_asset.py --check` (regenerating the two layers gives byte-identical files), the converter's
`--verify-only` (max abs diff 0.0), the rough `zero_agent --info` run + `check_joint_order.py` (joint order OK), the
depth `zero_agent` (pose refresh on, observation groups as above) and the A + B replay for seed 42 (bit-identical
per-episode records).

### 14.2 Replay of the converted checkpoint and the A/B test

Rough task, play mode, 128 envs x 2 complete episodes per seed, seeds 42-44 (768 episodes per row) unless noted.
"reward/ep" is the episode sum divided by 20 s (the `Episode_Reward` convention); tracking errors are means over alive
steps in the body frame.

| variant | ep len | base_contact | tilt > 60 deg | time_out | lin / yaw err | reward/ep |
|---|---|---|---|---|---|---|
| port default (native actuators, TGS flag on) | 17.82 s | 21.7 % | 5.9 % | 78.3 % | 0.258 / 0.157 | 1.117 |
| A: `env.sim.use_newton_actuators=false` | 18.26 s | 18.5 % | 4.8 % | 81.5 % | 0.226 / 0.134 | 1.312 |
| B: `env.sim.physics.enable_external_forces_every_iteration=false` | 17.88 s | 19.8 % | 3.8 % | 80.2 % | 0.193 / 0.105 | 1.435 |
| **A + B (2.x parity)** | 18.03 s | 18.9 % | 3.6 % | 81.1 % | 0.196 / 0.102 | **1.471** |
| native path + joint-state refresh (in-process patch, diagnostic) | 18.04 s | 18.1 % | 4.3 % | 81.9 % | 0.225 / 0.133 | 1.302 |
| A + B, policy resumed for 20 iterations (14.3) | 18.25 s | 18.1 % | 3.3 % | 81.9 % | 0.194 / 0.102 | 1.497 |
| `robot_feb23.usd` (seed 42) | bit-identical to the port default with seed 42 | | | | | |
| zero actions (seed 42) | 0.58 s | 100 % | 0 % | 0 % | 0.913 / 0.693 | -0.101 |
| random actions (seed 42) | 0.50 s | 100 % | 0 % | 0 % | 0.851 / 0.751 | -0.103 |

Per-seed base_contact for A + B: 18.0 / 23.8 / 14.8 % (binomial standard error about 2.5 % per seed); A is below the
port default in every seed. **2.x reference**: the original checkpoint in Isaac Lab 2.3.2 / Isaac Sim 5.1 (play config,
128 envs, seed 42, run from a copy of `~/IsaacLab/source`) falls in 18.8 % ± 3.4 of first episodes, with lin / yaw
error 0.187 / 0.103. First episodes in 3.0 with A + B, seeds 42-44: 15.6-21.1 % and 0.185-0.212 / 0.096-0.107. The port
default tracks worse: 0.243-0.298 / 0.150-0.168.

Why the two switches matter:

* **A: stale joint positions on the native actuator path** (an Isaac Lab 3.0 defect on PhysX). With identity joint
  ordering, `PhysxActuatorControl.compute_native_actuators`
  (`isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py:101-112`) refreshes the joint state only when
  `has_joint_ordering` is set, and the native wrapper reads the PhysX getter buffers directly (same pointer).
  `ArticulationData.update` re-reads the velocities every physics substep but not the positions. Measured on substeps
  2 / 3 / 4 of each 50 Hz step: the DC-motor PD sees positions that are off by 0.0095 / 0.019 / 0.028 rad on average,
  0.19 rad at most (about 11 N*m at kp = 60); velocities are fresh. Refreshing the buffers in-process makes the native
  path behave like the legacy one (row "refresh"). So "same 4-quadrant clamp as 2.x" (section 3) holds for the formula,
  not for the state fed to it. Reproduce: `scripts/eval_policy.py ... --native_refresh measure`.
* **B: TGS `enable_external_forces_every_iteration`**: 3.0 defaults to True, 2.3.2 used False. With False the 2.x
  policy's tracking error drops by a quarter to a third, and the `lin_vel_z_l2` / `ang_vel_xy_l2` penalties shrink 3-5x
  (-0.029 vs -0.091, -0.027 vs -0.140).
* The project defaults stay at the 3.0 values: both switches select deprecated APIs, and `use_newton_actuators` is a
  global sim setting that Newton runs share. Pass both switches to replay or fine-tune the 2.x checkpoint. For new
  Isaac Sim PhysX training, `env.sim.use_newton_actuators=false` avoids the stale positions until upstream fixes the
  refresh (open issue 1).

What the policy does: it trots in a tall stance on its calves (trunk 0.69 m above the ground vs 0.45 m at the default
pose; mean thigh -15 deg and calf 42 deg vs the default -50 / 100 deg; mean |action| 2.5). About 94 % of the falls
are thigh contacts, only a fifth to a quarter involve a tilt above 60 deg, and they are spread over the whole episode,
so the 0.27 m reset drop (risk 11) is not their main cause. Play mode puts robots on random tiles of all difficulties,
and the tall stance brings thighs and knees close to stair and box edges.

### 14.3 Training checks

All with the A + B switches, Isaac Sim PhysX, seed 42.

| run | result |
|---|---|
| rsl_rl rough, 1024 envs, 5 iterations from scratch | rc 0; `check_joint_order.py` on the training log: joint order OK |
| rsl_rl flat, 1024 envs, 5 iterations | rc 0 |
| skrl rough, 256 envs, 5 iterations | rc 0 (skrl 2.x warns about the 1.x yaml keys, open issue 6) |
| resume from the converted checkpoint, rough, 1024 envs, 20 iterations | continues at the replay level: optimizer state restored (Adam step counter 278020 -> 278040 -> 278420, +20 updates per iteration; learning rate 1.139e-4, then KL-adaptive), value loss 0.0157 -> 0.0068, action std 0.278 -> 0.275 |
| frozen control (learning rate 0, same seed and pipeline) | iteration 13900 identical; at 13919 mean reward 13.02 vs 13.24 (resumed), base_contact 5.6 % vs 5.5 % |
| play-mode evaluation of the resumed `model_13919` (3 seeds) | reward/ep 1.497 vs 1.471, base_contact 18.1 % vs 18.9 %, tracking unchanged (table 14.2) |

`Episode_Reward/feet_air_time` was 0.0000 in every iteration of every run.

### 14.4 Newton (details in section 13)

* `robot.usd` cannot build on Newton (0 collision shapes, `No local entities but world_count=64`); `robot_newton.usd`
  (17 convex-hull colliders) through the `CustomQuadRobotCfg` preset works with `physics=newton_mjwarp` and
  `newton_kamino`.
* Converted checkpoint on Newton MJWarp with the preset (PhysX joint and body order): 18.46 s episodes, 16.0 %
  base_contact, lin / yaw error 0.246 / 0.157, reward/ep 1.285 (PhysX A + B, seed 42: 1.491). In Newton's own joint
  order it falls within 0.9 s.
* Training benchmark (rough, 1024 envs, 30 iterations, idle GPU): Newton 56,444 env-steps/s and 1.37 GB, Isaac Sim
  PhysX (A + B) 14,362 env-steps/s and 4.02 GB.

### 14.5 Zero actions and feet

* Every zero-action episode ends in `base_contact` after about 0.58 s. The default crouch (thigh -50, calf 100 deg) does
  not hold itself up with kp 60 / kd 1.5: the knees fold to the 140 deg limit, the robot sits down on its rear knees and
  a thigh touches the ground. The same happens with a 0.47 m spawn (no reset drop), on flat ground, with both actuator
  paths and on Newton. `zero_agent` is a load and sensor smoke test only, not a stance test.
* Risk 10 is confirmed: foot contact 0.00 in every run and on every backend, calves in contact in about half of the
  alive steps; in 2.x the largest foot force was 0.0 N. `feet_air_time` never paid anything in 2.x or 3.0.

### 14.6 Depth task

* **Bug (fixed)**: with the default renderer (Newton Warp) on Isaac Sim PhysX the camera stayed at its spawn pose for
  the whole run. Probe with the 2.x policy walking: camera displacement exactly 0 over 150 steps while the trunk walked
  3.4 m away and turned 136 deg; the robot even walked into its own camera's view (up to 13 % of the pixels). Official
  training capture: 0 of 78 consecutive frame pairs changed. Cause: the 3.0 camera reads its pose from the camera
  prim's Fabric world matrix, and only the Kit RTX path keeps that matrix in step with the parent link
  (`update_latest_camera_pose` alone re-reads the stale value). Fix: the observation term `body_mounted_image_depth`
  (a subclass of `mdp.image_depth` in `rough_env_depth_cfg.py`) writes trunk pose x mount offset with
  `Camera.set_world_poses` before every read, only for Isaac Sim PhysX with a non-Kit renderer. The log says
  `pose refresh from 'trunk' on (physics PhysxCfg, renderer newton_warp)`.
* After the fix: camera pose error at most 2.4e-7 m; at least 99.998 % of the pixels agree with a terrain-only ray cast
  (99.996 % on reset frames); Warp images equal RTX images (median difference 0.00 mm, p99 0.03 mm); 0.0016 % own-robot
  pixels; no NaN or inf; 78 of 78 capture frame pairs change. RTX and Newton physics were already correct.
* RTX quirks (upstream): the first frame after `env.reset()` is wrong (22 % agreement, once per run), and the reported
  camera pose lags one step on reset frames while the image is right. `num_rerenders_on_reset` does not help in 3.0.
* 1024 envs: camera render 4.4-5.4 ms (Warp) vs 17.5 ms (RTX) of a 70-110 ms env step (PhysX 36-56 ms, pose refresh
  1.9-3.5 ms). Both renderers train at the same speed: 8.5-15.7k env-steps/s on a quiet GPU, 11.0-20.5k with
  `agent.check_for_nan=false` (no NaN in any of the 15 depth runs, so the override is safe). The speed falls during the
  10 iterations as episodes get shorter.

### 14.7 GPU memory (peak of the process, nvidia-smi)

| run | VRAM |
|---|---|
| `eval_policy.py`, rough, 128 envs, headless | 2.9 GB |
| `isaaclab play --video`, 32 envs (headless Kit viewport, 1280x720) | 5.3 GB |
| training rough, 1024 envs, Isaac Sim PhysX / Newton MJWarp | 4.0 / 1.4 GB |
| depth `zero_agent`, 16 envs | 3.3 GB |
| depth training, 1024 envs, Newton Warp / RTX | 5.5 / 7.3 GB |
| depth training, 64 envs with `--capture_env_sensors` (enables the Kit camera stack) | 5.7 GB (Warp), 5.2 GB (RTX) |

Isaac Sim PhysX and RTX preallocate at scene creation. With other jobs holding 12-15 GB, launches failed with
`PxgCudaDeviceMemoryAllocator failed to allocate memory 268435456 bytes`, `torch.OutOfMemoryError` or
`[omni.rtx] Out of GPU memory`; each succeeded after a retry. Check `nvidia-smi` first.

### 14.8 Demo videos and the GUI

The converted 2.x checkpoint with the A + B switches, 32 envs, seed 42 (`isaaclab play --video`, 1280x720, 50 fps;
the clips land in `checkpoints_2x/videos/play/`, git-ignored; copies in `runs/2026-11-01_quadruped_gpu/finish/videos/`):

* Follow-cam on env 30 (`demo_followcam_env30_best_s42_32envs_final.mp4`, 19.8 s; a 960x540 copy is
  `docs/media/quadruped_followcam.mp4`): the robot starts on random-rough terrain next to two robots of other envs that
  spawned on the same tile, crosses flat ground, turns with the command arrow and starts down a flight of stairs at the
  end, upright for the whole episode. `eval_policy.py` on the same seed and env count measures this episode as a
  time-out (no fall) with lin / yaw error 0.140 / 0.081; 28 of the 32 robots finish the 20 s episode.
* Follow-cam on env 0 (`take1_followcam_env0_best_s42_32envs.mp4`, 20 s): boxes terrain, then flat ground. The last
  frame shows the time-out reset (robot respawned in the air), so the final clip uses `--video_length 990`.
* Default view (`demo_defaultview_best_s42_32envs.mp4`, 10 s): fixed camera on the centre tile; every visible robot
  trots upright.
* GUI: the watch command in 14.9 opened the Kit window (screenshot of the app window:
  `finish/frames/gui_watch_kit_window.png`), played in real time and closed cleanly on SIGINT (Ctrl+C). The IsaacLab
  panel in that window has Viewer Settings (environment index, follow mode) for following one robot.

### 14.9 Verified commands

From the project root with `env_isaaclab` active. Every command ran on 2026-11-01 in this form (output paths and
`--run_name` / `--label` aside).

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd ~/isaac_ws/projects/quadruped_lab
python -m pip install --no-deps -e .
isaaclab list_envs                                        # the three task IDs
python -m pytest tests -q                                 # 22 passed

# smoke test + joint order (Isaac Sim PhysX)
mkdir -p logs
isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 500 --viz none --info \
    physics=isaacsim_physx 2>&1 | tee logs/zero_agent_rough.log
python scripts/check_joint_order.py logs/zero_agent_rough.log           # "joint order OK", exit 0

# replay metrics with the 2.x parity switches (about 1 min): base_contact 0.180, lin err 0.187, reward/ep 1.491
python scripts/eval_policy.py --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 128 --episodes 2 \
    --policy checkpoint --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --seed 42 \
    --out logs/eval/best_s42.json physics=isaacsim_physx \
    env.sim.use_newton_actuators=false env.sim.physics.enable_external_forces_every_iteration=false

# watch it in the GUI (real time; close the window or Ctrl+C to stop)
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority isaaclab play --rl_library rsl_rl \
    --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --viz kit --real-time physics=isaacsim_physx \
    env.sim.use_newton_actuators=false env.sim.physics.enable_external_forces_every_iteration=false

# demo video, follow-cam on env 30 (headless, stops after the clip; checkpoints_2x/videos/play/clip_<n>.mp4)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 --seed 42 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --video --video_length 990 \
    physics=isaacsim_physx env.sim.use_newton_actuators=false \
    env.sim.physics.enable_external_forces_every_iteration=false \
    env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=30 \
    'env.viewer.eye=(1.9,-1.9,1.0)' 'env.viewer.lookat=(0.0,0.0,-0.2)'
#   default view: drop the five env.viewer.* tokens (used with --video_length 500)

# resume training from the converted checkpoint (optimizer state included)
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 20 \
    --seed 42 --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" physics=isaacsim_physx \
    env.sim.use_newton_actuators=false env.sim.physics.enable_external_forces_every_iteration=false

# training from scratch: Isaac Sim PhysX (rsl_rl rough / flat, skrl rough) and Newton MJWarp
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 5 \
    --seed 42 physics=isaacsim_physx env.sim.use_newton_actuators=false \
    env.sim.physics.enable_external_forces_every_iteration=false
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-Custom-Quad-v0 --num_envs 1024 --max_iterations 5 \
    --seed 42 physics=isaacsim_physx env.sim.use_newton_actuators=false \
    env.sim.physics.enable_external_forces_every_iteration=false
isaaclab train --rl_library skrl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 256 --max_iterations 5 \
    --seed 42 physics=isaacsim_physx env.sim.use_newton_actuators=false \
    env.sim.physics.enable_external_forces_every_iteration=false
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 30 \
    --seed 42 physics=newton_mjwarp

# the converted checkpoint on Newton (newton_gl window; --video --video_length 400 records instead)
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority isaaclab play --rl_library rsl_rl \
    --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --viz newton_gl physics=newton_mjwarp

# depth task (EXPERIMENTAL): smoke, frame capture, 1024-env training (renderer=isaacsim_rtx for the RTX variant)
isaaclab zero_agent --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 16 --max_steps 200 --viz none --info \
    physics=isaacsim_physx                                 # log: "pose refresh from 'trunk' on (...)"
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 64 --max_iterations 3 \
    --capture_env_sensors 2 --capture_env_sensors_length 20 --capture_env_sensors_interval 50 \
    --capture_env_sensors_format file physics=isaacsim_physx
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 1024 \
    --max_iterations 10 physics=isaacsim_physx agent.check_for_nan=false
```

### 14.10 Open issues

1. **Native actuator path on Isaac Sim PhysX** (upstream, Isaac Lab 3.0): stale joint positions in the DC-motor PD
   (14.2). The port default (`use_newton_actuators=True`) is affected; the depth task's GPU checks ran with it. Upstream
   fix: refresh `_joint_pos` / `_joint_vel` unconditionally before `self._actuator_runtime.compute()` in
   `compute_native_actuators`. Until then pass `env.sim.use_newton_actuators=false` on PhysX. Whether to make A + B the
   project default (deprecated APIs, and the flag also switches Newton runs) is Hector's call; every number here
   documents both.
2. **Feet never touch** (risk 10): `feet_air_time` is dead weight in rough (0.01) and flat (0.25). For new training,
   point it at `.*_calf` or shrink the calf collider in a copy of the asset; both break parity with the 2.x checkpoint.
3. **Falls in play mode**: about 19 % of play-mode episodes end in `base_contact`, the same as the 2.x replay
   (18.8 %). The tall stance on random tiles catches thighs on edges. Improve by fine-tuning (A + B) or retraining, e.g.
   with a `base_height_l2` term (section 11).
4. **Soft default crouch**: zero actions sit the robot down in 0.6 s (14.5). A taller default pose, a 0.47 m spawn or
   stiffer gains would help new training and break parity.
5. **Newton**: the 2.x policy tracks worse on Newton (sim-to-sim gap: reward/ep 1.285 vs 1.491), on Newton 14-28 % of
   zero-action robots end up on the trunk, and Newton training is not bit-reproducible for a fixed seed (PhysX is).
   Kit-less runs print no INFO records even with `--info` (upstream `apply_python_logging_level` only adjusts existing
   handlers), so `check_joint_order.py` cannot read a Newton log.
6. **skrl yamls** still use skrl 1.x keys; skrl 2.x warns on every run (`lambda` -> `gae_lambda`,
   `clip_predicted_values` -> value clipping, `state_preprocessor` -> `observation_preprocessor`; `random_timesteps` and
   `learning_starts` are ignored for PPO).
7. **Depth task**: no policy trained yet (10 iterations at most). RTX first-frame and reset-pose quirks (upstream,
   14.6). `--capture_env_sensors` turns on the Kit camera stack even for Newton Warp, so capture runs are heavier and
   their speed is not representative. `check_for_nan` stays on by default; `agent.check_for_nan=false` is safe after
   the NaN-free runs.
8. **Shared GPU**: Isaac Sim launches fail when other jobs hold most of the 16 GB (14.7); the run scripts in
   `runs/2026-11-01_quadruped_gpu/` wait for free memory and retry.
