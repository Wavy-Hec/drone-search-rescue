# Port notes: custom quadruped, Isaac Lab 2.3.2 (Isaac Sim 5.1) -> Isaac Lab 3.0 (Isaac Sim 6.2)

Date: 2026-11-01. Ported on CPU only (another job was using the GPU): no Isaac Sim launch, no simulation, no training,
no pip installs. Everything that needs the simulator is listed in [GPU validation](#9-gpu-validation-next-phase).

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
| Depth task | **experimental** redesign (camera cfg, separate image group, CNN actor, privileged critic); needs retraining |
| Tests | 13 kit-less tests pass (run with a stub runner because pytest is not installed yet) |
| Simulation | **not run yet** (see section 9) |

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
| Actuator execution | 3.0 native path (`use_newton_actuators=True`) | non-deprecated path; the Newton DC-motor clamp is the same formula as 2.x (citations in section 3). A/B switch: `env.sim.use_newton_actuators=false` |
| Solver effort limit | `joint_effort_limit=45.0` | 3.0 docs recommend `>= actuator_effort_limit`; asset authors 500 N*m so even `None` would not clip; behaviour equals 2.x's 1e9 because the DC model already clips to 45 |
| TGS external forces every iteration | 3.0 default (`True`) kept | default flipped in 3.0 and `False` is deprecated (`isaaclab_physx/docs/CHANGELOG.rst:781-783`, `isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py:166`); 2.3.2 had `False` (`2.3.2: isaaclab/isaaclab/sim/simulation_cfg.py:123`). A/B: `env.sim.physics.enable_external_forces_every_iteration=false` |
| Joint/body ordering | `None` (backend-native) | the checkpoint was trained with Isaac Sim PhysX's native order; the `"physx"` alias means OvPhysX order, which can differ (`../docs/source/how-to/transfer_policies_between_physx_and_newton.rst:353-368`) |
| Task IDs | keep `Isaac-Velocity-{Flat,Rough,Rough-Depth}-Custom-Quad-v0` | as decided; gymnasium accepts `-v0`, no 3.0 code treats it specially |
| `-Play-v0` IDs | dropped | no real need: Hector's `commands` file played with the training ID, and `isaaclab play` applies `play_mode()`. If wanted, 3.0 supports deprecated aliases via a `"deprecated": {"alias": ...}` kwarg (`isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py:71-84`, example `isaaclab_tasks/isaaclab_tasks/contrib/dr_legs/__init__.py:24-38`) |
| `default_agent` | `rsl_rl` on all tasks | like 3.0 Go2 (`isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/__init__.py:23`); `--rl_library` becomes optional (`isaaclab_rl/isaaclab_rl/entrypoints/dispatch.py:145-163`) |
| Robot cfg location | `quadruped_lab/assets/custom_quad.py` | template README: keep asset configuration in the `<name>.assets` package |
| Depth task critic | asymmetric: MLP on proprio + privileged height scan | 3.0 camera tasks use a state critic with a camera actor (`isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py:100-108`); far cheaper than a second CNN |
| Depth renderer | Newton Warp default, `renderer=isaacsim_rtx` available | section 8 |
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
| Joint order | Isaac Sim PhysX native order | same (ordering `None`); expected breadth-first in authored joint order: FL/FR/RL/RR hip, then thigh, then calf (docs above). **Verify on GPU.** |
| Sim dt / decimation / control | 0.005 s / 4 / 50 Hz | same |
| Episode length | 20 s | same |
| Init root pos / default joint pos | z 0.719; hip +pi/45, thigh -5pi/18, calf 5pi/9 | same |
| DC motor | effort 45, saturation 45, velocity 21 (torque-speed curve), kp 60, kd 1.5, friction 0 | same values; solver effort 45 (2.x 1e9, asset 500), solver velocity limit stays the asset's 10 rad/s in both |
| Mass randomization | trunk + U(-1, 3) kg, inertia rescaled | same (explicit `operation`/`distribution`) |
| Commands, rewards, terminations, curriculum, terrain generator | as configured | functions/configs identical apart from `.torch` and `torch.linalg.norm`; `ROUGH_TERRAINS_CFG` identical |
| PhysX scene | TGS, `gpu_max_rigid_patch_count=327680` | same, except `enable_external_forces_every_iteration` True (see decisions) |

Deliberate differences: physics preset handling (`physx` alias), native actuator path, the TGS flag above, and the
rsl-rl 5 std clamp `(1e-6, 1e6)` (the checkpoint's std 0.249-0.313 is far inside, and play uses the mean).

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

* **Camera** (`CameraCfg`, prim `{ENV_REGEX_NS}/Robot/trunk/depth_camera`): 2.x mount pose kept, pos
  (-0.25, 0, 0.7) on the trunk, `convention="opengl"`, rot (w,x,y,z) (0.66446, 0.24184, -0.24184, -0.66446) ->
  (x,y,z,w) (0.24184, -0.24184, -0.66446, 0.66446). It looks forward, pitched 50.0 deg down; pinhole
  f = 18.14756 mm, aperture 20.955 mm -> 60.0 x 46.8 deg FOV. `update_latest_camera_pose=True` (body-mounted camera).
* **Far clip 1.0 -> 4.0 m** (`DEPTH_MAX_RANGE`). The camera is ~1.42 m above flat ground (trunk 0.72 m + 0.70 m), so
  the steepest ray (73.4 deg down) reaches the ground at ~1.48 m and the top image edge at ~3.2 m: with the 2.x far clip
  every ground pixel was out of range.
* **Render rate** once per policy step (`update_period = decimation * dt = 0.02 s`); the 2.x patch rendered every
  5 ms physics step (`update_period=0.0`), 4x the work for frames nobody read.
* **Observations**: `policy` (48 proprio, no height scan); `depth` = `mdp.image_depth(data_type="distance_to_camera",
  normalize=True, max_depth=4.0, invalid_value=4.0, channel_first=True)` -> (N, 1, 48, 64) in [0, 1], noise
  U(+-0.005) (= the 2.x +-2 cm in metric units), clip (0, 1); `privileged` = noise-free height scan for the critic only.
  Both renderers are set to `depth_clipping_behavior="max"` so a ray without a hit reads 1.0 (far), not Newton's 0.0
  (`isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer_cfg.py:58-69`) or RTX `inf`.
* **Model**: `RslRlCNNModelCfg` actor, conv [16, 32, 32] / kernels [5, 3, 3] / stride 2 -> 768 features + 48 proprio ->
  MLP [512, 256, 128] -> 12 (Gaussian, init std 1.0); MLP critic on 48 + 187. `obs_groups = {"actor": ["policy",
  "depth"], "critic": ["policy", "privileged"]}`. rsl-rl's `CNNModel` takes (B, C, H, W) groups
  (`site-packages/rsl_rl/models/cnn_model.py:140`). PPO hyperparameters inherited from the rough runner.
* **Renderer**: Newton Warp by default. It is the kit-less Warp ray tracer "for training workflows where throughput
  matters", produces `distance_to_camera`, and is documented to pair with Isaac Sim PhysX
  (`../docs/source/concepts/renderers.rst:86-93`); for a 64x48 depth-only image the RTX path tracer is wasted work.
  `renderer=isaacsim_rtx` switches to the legacy Isaac RTX renderer (Kit) for an A/B test. OVRTX is excluded: it cannot
  share a process with Kit, which Isaac Sim PhysX requires, and its runtime is not installed (`renderers.rst:95-103`).
  A Newton Warp camera is not a "Kit camera", so no RTX camera rendering is enabled for it
  (`isaaclab/isaaclab/app/sim_launcher.py:144-159`).
* **skrl**: not registered for the depth task (the MLP yaml cannot consume a 2-D group).

Memory budget on the 16 GB RTX 5080 (float32; rollout storage keeps every observation group,
`site-packages/rsl_rl/storage/rollout_storage.py:141-148`):

| Resolution | Envs | Rollout per env-step | Rollout (24 steps) | PPO mini-batch (1/4) | CNN activations per mini-batch |
|---|---|---|---|---|---|
| 64x48 (port default) | 1024 | 13,393 B | 0.31 GiB | 6144 samples | 0.76 GiB (x~2 with gradients) |
| 64x48 | 2048 | 13,393 B | 0.61 GiB | 12288 | 1.52 GiB |
| 64x48 | 4096 | 13,393 B | 1.23 GiB | 24576 | 3.04 GiB |
| 320x240 (2.x patch) | 1024 | 308,305 B | 7.06 GiB | 6144 | 23.4 GiB |

Per env-step: obs (48 + 64*48 + 187) * 4 B = 13,228 B plus actions/rewards/values/log-probs/returns/advantages/
mean/std = 165 B. CNN activations: 32,384 floats per sample (conv + ELU outputs, measured with the rsl-rl `CNN`).
Camera output 64*48*4 B * 1024 = 12 MiB. Adding rough estimates for Kit (~2-3 GB), Isaac Sim PhysX for 1024 robots
(~1-2 GB), the Newton Warp scene/BVH (~0.5 GB, the rough terrain is ~2.6 M triangles) and the CUDA context (~0.5 GB),
1024 envs should need roughly 6-8 GB, leaving headroom; 2048 envs is likely to fit (~10-11 GB); 4096 is not
recommended. The 2.x 320x240 config would need more than 30 GB. **These are estimates; measure in the GPU phase.**

## 9. GPU validation (next phase)

Run from the project root with the env active. The first Isaac Sim 6.2 launch can take more than 10 minutes.

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES                       # only if the EULA has not been accepted yet
cd ~/isaac_ws/projects/quadruped_lab

# 1. install the project (isaaclab-dev is not on any index -> --no-deps; pip fetches uv_build for the build)
python -m pip install --no-deps -e .
python -c "import importlib.metadata as m; print([e for e in m.entry_points(group='isaaclab.tasks') if e.name == 'quadruped_lab'])"
isaaclab list_envs                                    # expect the 3 task IDs
python -m pip install pytest && python -m pytest tests -q   # 13 kit-less tests

# 2. zero-action smoke tests (bounded, headless)
isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 500 --viz none physics=isaacsim_physx
isaaclab zero_agent --task Isaac-Velocity-Flat-Custom-Quad-v0 --num_envs 16 --max_steps 300 --viz none physics=isaacsim_physx
#    check the log: robot spawns from the package path, no "No rigid-body targets matched" warning,
#    contact sensor bodies include trunk/*_thigh/*_foot, and the articulation joint_names order is
#    FL_hip FR_hip RL_hip RR_hip FL_thigh FR_thigh RL_thigh RR_thigh FL_calf FR_calf RL_calf RR_calf

# 3. replay the converted 2.x checkpoint, recorded to video (stops after the clip)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --video --video_length 500 physics=isaacsim_physx
#    video: checkpoints_2x/videos/play/ ; exports: checkpoints_2x/exported/policy.{pt,onnx} (both git-ignored)
#    interactive viewer: replace --video ... with --viz kit (runs until the window is closed)
#    if the gait is off, A/B one at a time:
#      env.sim.use_newton_actuators=false
#      env.sim.physics.enable_external_forces_every_iteration=false
#      env.scene.robot.spawn.usd_path=$PWD/src/quadruped_lab/assets/data/robots/custom_quad/robot_feb23.usd

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
#    frames: logs/rsl_rl/custom_quad_rough_depth/<run>/sensor_frames/train -> ground visible, values in [0, 1], misses = 1.0
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 1024 --max_iterations 10 physics=isaacsim_physx
#    meanwhile in another shell: nvidia-smi --query-gpu=memory.used --format=csv -l 5   (budget in section 8)
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 64 --max_iterations 3 physics=isaacsim_physx renderer=isaacsim_rtx

# 6. optional: Newton MJWarp
isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 300 --viz none physics=newton_mjwarp
```

Never use `physics=physx` on the command line out of habit; it is aliased to Isaac Sim PhysX in these configs, but other
tasks would resolve it to OvPhysX.

## 10. Known risks and open items

1. **Nothing has run in the simulator yet.** USD loading under Isaac Sim 6.2 PhysX, the schema-fragment authoring on
   this asset, contact-sensor body matching and the native actuator path are verified only by source reading and CPU
   config resolution.
2. **Joint order**: the policy depends on Isaac Sim 6.2 PhysX producing the same DOF order as 5.1. Expected
   breadth-first FL/FR/RL/RR per level; print `joint_names` in step 2. A mismatch makes the replayed policy fall within
   a few dozen steps.
3. **Dynamics drift**: native actuator path and the TGS external-force flag differ in execution from 2.x (same model
   math); the 2.x checkpoint came from an unrecorded January setup. Use the A/B overrides in step 3 if needed.
4. **Newton**: untested. The asset may need the Newton audit (`../docs/source/how-to/prepare_asset_for_newton.rst`), MJWarp
   may order joints differently, and replaying the PhysX checkpoint there needs an explicit `joint_ordering` list
   recorded from a PhysX run (transfer guide `:264-269`).
5. **Depth task**: Newton Warp + Isaac Sim PhysX in one process is a documented pairing but unexercised here; the
   trunk may be visible at the bottom of the image (self-occlusion); hyperparameters are untuned; memory numbers are
   estimates.
6. **Packaging**: `pip install --no-deps -e .` builds with `uv_build`, which pip downloads into an isolated build env
   (network needed). The `[tool.uv.sources]` paths are relative to `../../IsaacLab`.
7. **Git LFS**: binaries are committed as LFS objects in this local repo; a remote must support LFS before pushing.
8. Upstream deprecation warnings (e.g. `RigidBodyMaterialCfg` in the 3.0 velocity base terrain) are not from this
   project.
9. `max_iterations=80000` is inherited from 2.x (the converted policy has 13900); override with `--max_iterations`.

## 11. Possible improvements

* **Depth via ray casting instead of rendering**: a `RayCasterCameraCfg` / `MultiMeshRayCasterCameraCfg` against
  `/World/ground` gives terrain-only depth with Warp ray casts (like the height scanner), no renderer, no
  self-occlusion, works on PhysX and Newton, and is far cheaper per env. `mdp.image_depth` accepts ray-caster cameras
  (`isaaclab/isaaclab/envs/mdp/observations.py:470-473`); 3.0 ships such presets for kuka_allegro
  (`isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/camera_cfg.py:82-107`).
* **Depth policy memory**: `frame_stack=2` (motion cue) or an `RslRlRNNModelCfg` actor; mixed precision
  (`RslRlPpoAlgorithmCfg.use_mixed_precision`, `rl_cfg.py:220`) for the CNN update; `check_for_nan=False` once
  validated (saves a host sync per step, `rl_cfg.py:294`).
* **Faster PPO updates**: `torch_compile_mode="max-autotune-no-cudagraphs"` on the runner (`rl_cfg.py:355`).
* **Throughput**: train on Newton MJWarp (the 3.0 default for the velocity family) after validating the asset, then
  sim-to-sim check on PhysX with the transfer guide.
* **Gait quality**: a `base_height_l2` reward like 3.0's Go2 (`isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rough_env_cfg.py:22-35,70-72`)
  prevents crouching gaits; consider shorter `max_iterations` defaults.
* **Tooling**: install the template's dev group (pytest, ruff, pre-commit, codespell) and run
  `python -m pytest tests -q` and `pre-commit run --all-files` before commits.
