# Porting the custom quadruped to Isaac Lab 3.0

This write-up covers how the custom quadruped's velocity-tracking tasks moved from Isaac Lab 2.3.2 (Isaac Sim 5.1) to
Isaac Lab 3.0 (Isaac Sim 6.2): what was ported, which Isaac Lab 3.0 differences change how the trained policy walks
and how each one was measured, two problems of the robot asset, the depth-camera fix, and the validation on the GPU.

| | Isaac Lab 2.x project | This project |
|---|---|---|
| Isaac Lab / Isaac Sim | 2.3.2 / 5.1 | 3.0 (`release/3.0.0` @ `44044b096`) / 6.2 |
| RL libraries | rsl-rl 3.1.2, skrl 1.4.3 | rsl-rl 5.5.1, skrl 2.1.1 |
| Physics | Isaac Sim PhysX | Isaac Sim PhysX (default), Newton MJWarp / Kamino (opt-in) |
| Validation | | RTX 5080 (16 GB), 2026-11-01 |

Source references (`file:line`) point into the Isaac Lab 3.0 repository at that commit (Python packages under
`source/`) unless marked `2.3.2:`.

**Contents:** [1. Summary](#1-summary) ·
[2. What was ported](#2-what-was-ported) ·
[3. Policy interface parity](#3-policy-interface-parity) ·
[4. Checkpoint conversion](#4-checkpoint-conversion) ·
[5. Execution differences in Isaac Lab 3.0](#5-execution-differences-in-isaac-lab-30) ·
[6. The feet never touch the ground](#6-the-feet-never-touch-the-ground) ·
[7. Newton backend](#7-newton-backend) ·
[8. Depth-camera task](#8-depth-camera-task) ·
[9. Validation](#9-validation) ·
[10. Reproducing the results](#10-reproducing-the-results) ·
[11. Known issues and next steps](#11-known-issues-and-next-steps)

## 1. Summary

* **Tasks.** `Isaac-Velocity-Rough-Custom-Quad-v0` (48 proprioceptive values + a 187-ray height scan),
  `Isaac-Velocity-Flat-Custom-Quad-v0` (proprioception only) and an experimental redesign of the depth-camera task,
  `Isaac-Velocity-Rough-Depth-Custom-Quad-v0`.
* **Policy.** The 2.x height-scan policy (iteration 13,900) is converted to the rsl-rl 5.5 checkpoint format. The
  converted actor, critic and TorchScript export reproduce the original weights exactly (max abs difference 0.0).
* **Walking parity.** In Isaac Lab 3.0 the converted policy walks at its 2.x level once two execution settings match
  Isaac Lab 2.x. Both are the project's defaults on Isaac Sim PhysX; Newton keeps the Isaac Lab 3.0 defaults
  ([section 5](#5-execution-differences-in-isaac-lab-30)):

  | Rough task, play mode, 768 episodes | Falls | Tilt > 60° | Tracking error lin / yaw | Reward / episode |
  |---|---|---|---|---|
  | Isaac Lab 3.0 execution defaults | 21.7 % | 5.9 % | 0.258 m/s / 0.157 rad/s | 1.117 |
  | **2.x execution settings (project default)** | **18.9 %** | **3.6 %** | **0.196 / 0.102** | **1.471** |

  The original checkpoint in Isaac Lab 2.3.2 falls in 18.8 % of its first episodes with tracking errors
  0.187 / 0.103.
* **Findings** on the way, each measured:
  * Isaac Lab 3.0's native actuator path feeds the DC-motor PD joint positions that are 1-3 physics substeps old on
    Isaac Sim PhysX ([5.3](#53-the-native-actuator-path-reads-stale-joint-positions)).
  * Isaac Lab 3.0 applies TGS external forces on every position iteration by default; 2.3.2 did not
    ([5.4](#54-tgs-external-forces)).
  * The robot's feet never touch the ground, in 2.x either: the calf collision mesh encloses the foot sphere, so the
    `feet_air_time` reward never paid anything ([section 6](#6-the-feet-never-touch-the-ground)).
  * Newton's USD importer skips all 17 colliders of the asset. A generated Newton layer fixes that, and the policy
    walks on Newton zero-shot through a joint-order remap ([section 7](#7-newton-backend)).
  * With the Newton Warp renderer on Isaac Sim PhysX, the depth camera rendered from its spawn pose for the whole run.
    An observation term now writes the camera pose from the trunk before every read
    ([8.4](#84-camera-pose-fix)).

## 2. What was ported

### 2.1 Files

| Isaac Lab 2.x task | Isaac Lab 3.0 project | Change |
|---|---|---|
| `robot/robot.usd` + `configuration/*.usd` (2026-01-15) | `assets/data/robots/custom_quad/` | unchanged; the canonical asset (it predates the checkpoint of 2026-01-29) |
| a re-save of `robot.usd` from 2026-02-23 | `assets/data/robots/custom_quad/robot_feb23.usd` | unchanged; its root layer differs only in the saved viewport camera, and the policy replays bit-identically on it |
| `custom_quad.py` | `assets/custom_quad.py` | package-relative USD path, schema fragments, renamed actuator limits, explicit solver effort limit, Newton variant |
| `rough_env_cfg.py` | `tasks/locomotion/config/custom_quad/rough_env_cfg.py` | 3.0 velocity base, Isaac Sim PhysX default with the 2.x execution settings, robot preset per backend, explicit mass-event semantics |
| `flat_env_cfg.py` | `.../flat_env_cfg.py` | lines of the 2.x depth-camera patch dropped |
| `rough_env_depth_cfg.py` | `.../rough_env_depth_cfg.py` | redesigned ([section 8](#8-depth-camera-task)) |
| `__init__.py` | `.../__init__.py` | same task IDs; the `-Play-v0` IDs are gone because `isaaclab play` applies `play_mode()` |
| `agents/rsl_rl_ppo_cfg.py` | `.../agents/rsl_rl_ppo_cfg.py` | rsl-rl 5.5 actor/critic model configs with explicit `obs_groups`; a CNN runner for the depth task |
| `agents/skrl_{rough,flat}_ppo_cfg.yaml` | `.../agents/skrl_{rough,flat}_ppo_cfg.yaml` | `input: STATES` becomes `OBSERVATIONS` (skrl ≥ 2.1) |
| `logs/*.pt` | `checkpoints_2x/` | copied; plus the converted `height_scanner_model_rslrl5.pt` |

Tools written for the port live in `scripts/`: the checkpoint converter, the 2.x parity check, the joint-order check,
the play-mode evaluator, the Newton asset generator and the depth-camera study.

### 2.2 Isaac Lab 3.0 API changes

| 3.0 change | Applied as | Reference |
|---|---|---|
| The velocity base moved to `isaaclab_tasks.core.velocity` and defaults to Newton MJWarp | `sim.physics.default = isaacsim_physx` | `core/velocity/velocity_env_cfg.py:72` |
| `physics=physx` is an auto preset that picks OvPhysX for kit-less runs | aliased to Isaac Sim PhysX, so headless training and windowed play use one backend | `isaaclab/physics/physics_manager_cfg.py:68-78` |
| `add_base_mass` defaults to a log-uniform scale; the 2.x range `(-1, 3)` alone raises | `operation="add"`, `distribution="uniform"` (the 2.x semantics) | `isaaclab/envs/mdp/events.py:1933` |
| `effort_limit` / `velocity_limit` are deprecated | `actuator_effort_limit` / `actuator_velocity_limit` | `isaaclab/actuators/actuator_base_cfg.py:162-177` |
| Explicit actuators keep the solver effort limit (2.x widened it to 1e9) | `joint_effort_limit=45.0`; the DC motor already clips at 45 N·m | `docs/source/isaaclab/migrating.rst:1085-1091` |
| `RigidBodyPropertiesCfg` / `ArticulationRootPropertiesCfg` are deprecated | `UsdPhysicsRigidBodyCfg` + `PhysxRigidBodyCfg`, `PhysxArticulationCfg` + `NewtonArticulationCfg` | `isaaclab/sim/schemas/schemas_cfg.py:168` |
| `cfg.replace()` became `isaaclab.utils.replace(cfg, ...)` | rough cfg | `isaaclab/utils/configclass.py:142` |
| `-Play` registrations are gone; `isaaclab play` applies `play_mode()` | `-Play-v0` IDs and `_PLAY` classes dropped | `isaaclab_tasks/docs/CHANGELOG.rst:1779-1783` |
| rsl-rl 5 model configs replace `RslRlPpoActorCriticCfg` and `empirical_normalization` | `RslRlMLPModelCfg` actor and critic | `isaaclab_rl/rsl_rl/rl_cfg.py` |
| skrl ≥ 2.1 reads `STATES` as the state space | yamls use `OBSERVATIONS` | `isaaclab_rl/skrl.py:39` |
| `mdp.image` became `mdp.image_depth`; `TiledCameraCfg` became `CameraCfg`, offsets in (x, y, z, w) | depth task | `isaaclab/envs/mdp/observations.py:470-519` |

### 2.3 Decisions

| Topic | Choice | Why |
|---|---|---|
| Physics | Isaac Sim PhysX by default; Newton with `physics=newton_mjwarp` | the backend the 2.x checkpoints were trained on |
| Execution settings | Isaac Lab's actuator path and the 2.3.2 TGS setting on Isaac Sim PhysX | the converted policy walks measurably better ([section 5](#5-execution-differences-in-isaac-lab-30)) |
| Joint and body order | backend-native on Isaac Sim PhysX; Newton remapped to it | the checkpoint was trained in PhysX order ([section 7](#7-newton-backend)) |
| Task IDs | the 2.x IDs; `default_agent = "rsl_rl"` | scripts and commands keep working; `--rl_library` becomes optional |
| skrl hyperparameters | unchanged | the rough yaml still carries the template's A1 values; not changed silently |
| Robot config location | `quadruped_lab.assets` | the Isaac Lab project template's convention |

### 2.4 Robot asset

* Default prim `/quadruped_robot`, Z-up, meters. The articulation root is on the `trunk`, and all 17 rigid bodies are
  direct children of the default prim: trunk 9.70 kg and, per leg, hip 1.16, thigh 1.24, calf 0.66 and foot
  0.015 kg (about 22 kg).
* 12 revolute joints `{FL,FR,RL,RR}_{hip,thigh,calf}_joint` and 4 fixed foot joints. Limits: hip ±35°, thigh
  −70…0°, calf 20…140°. The drives author 500 N·m and 10 rad/s.
* Default pose: hip 4°, thigh −50°, calf 100°. Forward kinematics from the joint frames in the USD put the trunk
  origin 0.452 m above flat ground in that pose. The spawn height of 0.719 m is the straight-leg height, so every
  reset drops the robot about 0.27 m; this is kept for parity with the checkpoint.
* DC motors on all joints: 45 N·m effort and saturation, 21 rad/s torque-speed limit, kp 60, kd 1.5.

## 3. Policy interface parity

The policy observation of the rough task, as the checkpoint's first layer (512 × 235) expects it:

| # | Term | Function | Dims | Index range | Noise U(min, max) | Clip |
|---|---|---|---|---|---|---|
| 0 | `base_lin_vel` | `base_lin_vel` | 3 | 0-2 | (−0.1, 0.1) | |
| 1 | `base_ang_vel` | `base_ang_vel` | 3 | 3-5 | (−0.2, 0.2) | |
| 2 | `projected_gravity` | `projected_gravity` | 3 | 6-8 | (−0.05, 0.05) | |
| 3 | `velocity_commands` | `generated_commands` | 3 | 9-11 | | |
| 4 | `joint_pos` | `joint_pos_rel` | 12 | 12-23 | (−0.01, 0.01) | |
| 5 | `joint_vel` | `joint_vel_rel` | 12 | 24-35 | (−1.5, 1.5) | |
| 6 | `actions` | `last_action` | 12 | 36-47 | | |
| 7 | `height_scan` | `height_scan` | 187 | 48-234 | (−0.1, 0.1) | (−1, 1) |

No term is scaled; noise is on in training and off in play mode, in both versions. The rest of the interface is the
same as well: joint-position action on all joints with scale 0.25 and the default offset, 5 ms physics step with
decimation 4 (50 Hz control), 20 s episodes, the initial pose and the DC-motor parameters.

`scripts/check_obs_parity.py` checks all of this without a simulator. It reads the 2.x configuration statically
(Python AST) from the Isaac Lab 2.3.2 sources and the 2.x task files, so no 2.x value is typed into the script, and
compares it with the instantiated 3.0 config:

* the policy terms against the checkpoint's 235 inputs;
* every value the 2.x configs set explicitly, 214 in total: actions, commands, rewards, terminations, events,
  curriculum, height scanner, contact sensor, terrain generator and sub-terrains, timing. A term parameter that a config
  leaves out is compared through the default of its function (2.3.2 source vs 3.0 signature);
* the robot's default pose and DC-motor parameters, and the two execution settings of
  [section 5](#5-execution-differences-in-isaac-lab-30).

Two known differences are reported, not failed: the contact-sensor path `Robot/.*` became `Robot/[^/]*` in the 3.0
base (all bodies are direct children of the robot root, so it matches the same bodies), and the PhysX scene settings
moved from `sim.physx` into the physics preset. A test perturbs three settings and both execution settings and
checks that each one is reported.

Term functions that changed in 3.0 keep their semantics:

| Term | 3.0 change | Same behaviour because |
|---|---|---|
| observation terms | `.torch` accessors, `torch.linalg.norm` | same quantities |
| `illegal_contact` (`base_contact`) | reads `net_normal_forces_w_history` | the PhysX contact view only ever reported normal forces (`isaaclab_physx/docs/CHANGELOG.rst:421-426`) |
| `joint_torques_l2` | reads `asset.actuators.applied_effort` | both actuator paths write the clipped DC-motor effort there |
| `reset_root_state_uniform` | now a class-based term | same parameters and sampling |
| `apply_external_force_torque` | returns early for zero ranges | the task's ranges are zero, so 2.x applied a zero wrench |

Deliberate differences: the physics preset handling above and rsl-rl 5's std clamp to (1e-6, 1e6), whose bounds
are far from the checkpoint's std of 0.249-0.313 (and play uses the mean anyway).

One provenance caveat: in the 2.x project, `rough_env_cfg.py` was saved 4 minutes after the checkpoint (checkpoint
00:06:30, config 00:10:31 on 2026-01-29), and the only bytecode was compiled from the edited file. The observation
interface provably matches the weights, but the exact reward and event settings of the training run cannot be
recovered. The replay is therefore evaluated by behaviour and metrics, against the same checkpoint run in Isaac Lab
2.3.2.

## 4. Checkpoint conversion

| File | Format | Contents |
|---|---|---|
| `height_scanner_model.pt` | rsl-rl 3.x `model_state_dict` | iteration 13,900; 235 → [512, 256, 128] → 12, ELU; scalar std 0.249-0.313; no observation normalizer; Adam state (lr 1.139e-4) |
| `height_scanner_model_rslrl5.pt` | rsl-rl 5.5 | `actor_state_dict` (`distribution.std_param`, `mlp.*`), `critic_state_dict`, the same optimizer state and iteration |
| `depth_camera_model.pt` | rsl-rl 3.x | kept for reference, not converted: its 240 inputs (48 + 192) match no recorded camera config |

Mapping: `std` → `distribution.std_param`, `actor.N.*` → actor `mlp.N.*`, `critic.N.*` → critic `mlp.N.*`, and
`{actor,critic}_obs_normalizer.*` → `obs_normalizer.*`. The Adam state is index-based and carries over unchanged:
rsl-rl 3.1 optimized `ActorCritic.parameters()` (std first) and rsl-rl 5.5 optimizes the actor's then the critic's
parameters with the distribution registered before the MLP, i.e. the same tensors in the same order. The converter
checks every Adam moment shape against the new order, loads with `weights_only=True` and never writes in place.
Log-std checkpoints map to `distribution.log_std_param`, and the converter prints the agent config they need.

`--verify` builds the runner exactly like `isaaclab play` (deprecated-config handling, `create_rsl_rl_runner`,
strict `runner.load()`) and compares it with a hand-written Linear + ELU forward pass of the old weights on 4096
random observations, half of them scaled ×3:

| Check | Result |
|---|---|
| actor mean actions, critic values, std | max abs difference 0.0 |
| TorchScript export (`exported/policy.pt`, used by play) | max abs difference 0.0 |
| loaded iteration / optimizer learning rate | 13,900 / 1.139e-4 |

## 5. Execution differences in Isaac Lab 3.0

### 5.1 Method

`isaaclab play` prints no metrics, so `scripts/eval_policy.py` builds the environment exactly as play does (play-mode
config, deterministic policy mean) and runs every env for complete episodes. It reports falls (`base_contact`
terminations) and time-outs, episodes with a trunk tilt above 60°, velocity-tracking errors in the body frame over
alive steps, and the episode reward per term in the `Episode_Reward` convention (episode sum / 20 s).

Each row below is the rough task with 128 envs × 2 episodes for seeds 42, 43 and 44 (768 episodes). The simulation
is deterministic for a fixed seed and number of envs (reruns reproduce every per-episode record), so the variants
differ only by the setting under test. The reference is the original checkpoint run in Isaac Lab 2.3.2 /
Isaac Sim 5.1 with its play config (128 envs, seed 42): 18.8 % ± 3.4 % (standard error) of the first episodes end
in a fall, with tracking errors 0.187 m/s (linear) and 0.103 rad/s (yaw).

### 5.2 Results

| Variant | Episode length | Falls | Tilt > 60° | Time-outs | Tracking error lin / yaw | Reward / episode |
|---|---|---|---|---|---|---|
| Isaac Lab 3.0 defaults (native actuators, TGS forces every iteration) | 17.82 s | 21.7 % | 5.9 % | 78.3 % | 0.258 / 0.157 | 1.117 |
| Isaac Lab's actuator path only | 18.26 s | 18.5 % | 4.8 % | 81.5 % | 0.226 / 0.134 | 1.312 |
| 2.3.2 TGS setting only | 17.88 s | 19.8 % | 3.8 % | 80.2 % | 0.193 / 0.105 | 1.435 |
| **Both: the 2.x execution settings (project default)** | 18.03 s | 18.9 % | 3.6 % | 81.1 % | 0.196 / 0.102 | **1.471** |
| Native actuators with a joint-state refresh (diagnostic patch) | 18.04 s | 18.1 % | 4.3 % | 81.9 % | 0.225 / 0.133 | 1.302 |
| Both, after fine-tuning for 20 PPO iterations | 18.25 s | 18.1 % | 3.3 % | 81.9 % | 0.194 / 0.102 | 1.497 |
| Zero actions (seed 42) | 0.58 s | 100 % | 0 % | 0 % | 0.913 / 0.693 | −0.101 |
| Random actions (seed 42) | 0.50 s | 100 % | 0 % | 0 % | 0.851 / 0.751 | −0.103 |

* Falls with both settings, per seed: 18.0 / 23.8 / 14.8 % (binomial standard error about 2.5 % per seed). The
  actuator setting alone falls less than the 3.0 defaults in every seed.
* First episodes only, to match the 2.x reference: both settings give 15.6-21.1 % falls and tracking errors
  0.185-0.212 / 0.096-0.107 over seeds 42-44, the 2.x range. The 3.0 defaults track worse: 0.243-0.298 /
  0.150-0.168.
* `robot_feb23.usd` replays bit-identically to `robot.usd`.

### 5.3 The native actuator path reads stale joint positions

Isaac Lab 3.0 runs explicit actuators such as `DCMotorCfg` through a native path by default
(`sim.use_newton_actuators=True`). On Isaac Sim PhysX with the identity joint order,
`PhysxActuatorControl.compute_native_actuators` (`isaaclab_physx/assets/articulation/actuator_control.py:101-112`)
refreshes the joint state only when a joint reordering is set, and the native wrapper reads the PhysX getter buffers
directly. `ArticulationData.update` re-reads the velocities on every physics substep, but not the positions.

Measured with an in-process probe on substeps 2, 3 and 4 of each 50 Hz control step, the DC-motor PD sees joint
positions that are off by 0.0095 / 0.019 / 0.028 rad on average and 0.19 rad at most (about 11 N·m at kp = 60);
the velocities are fresh. Refreshing the buffers in-process makes the native path behave like Isaac Lab's actuator
path (the diagnostic row above). The torque-speed clamp itself is the same formula as in 2.x; the state fed into it
is not. A fix in Isaac Lab is to refresh `_joint_pos` and `_joint_vel` unconditionally before
`self._actuator_runtime.compute()`. Reproduce the measurement with
`eval_policy.py ... --native_refresh measure env.sim.use_newton_actuators=true`.

### 5.4 TGS external forces

Isaac Lab 3.0 sets `PhysxCfg.enable_external_forces_every_iteration=True` by default and deprecates `False`; Isaac
Lab 2.3.2 used `False` (`2.3.2: isaaclab/sim/simulation_cfg.py:123`). With `False`, the converted policy's tracking
error drops by a quarter to a third, and the `lin_vel_z_l2` and `ang_vel_xy_l2` penalties shrink three- to five-fold
(`Episode_Reward` −0.029 vs −0.091 and −0.027 vs −0.140).

### 5.5 Project defaults

* `UseNewtonActuatorsCfg` (`rough_env_cfg.py`) sets `sim.use_newton_actuators` per physics backend through the
  `physics=` selector: `False` on Isaac Sim PhysX (the default, `isaacsim_physx`, and the `physx` alias), `True` on
  Newton MJWarp, Newton Kamino and OvPhysX.
* The Isaac Sim PhysX preset sets `enable_external_forces_every_iteration=False`.
* The Isaac Lab 3.0 defaults stay one override away:
  `env.sim.use_newton_actuators=true env.sim.physics.enable_external_forces_every_iteration=true`.
* Trade-off: both settings select APIs that Isaac Lab 3.0 marks as deprecated. Once the joint-state refresh is fixed
  upstream, the actuator setting can return to the 3.0 default.

Checked when the defaults changed: the resolved configs of all three tasks, in training and play mode, for every
physics selection, differ from the previous defaults in exactly these two values and only on Isaac Sim PhysX; Newton
and OvPhysX resolve identically, and the 3.0 override above reproduces the previous default config exactly. On the
GPU, a replay with the new defaults and no overrides reproduces the earlier replay that passed both settings
explicitly, record for record (seed 42, 128 envs × 2 episodes: 18.0 % falls, 0.187 m/s tracking error, reward 1.491).

### 5.6 How the policy walks

It trots in a tall stance on its calves: the trunk rides 0.69 m above the ground (0.45 m in the default pose), with
mean thigh and calf angles of −15° and 42° (default −50° and 100°) and a mean |action| of 2.5. About 94 % of the
falls are thigh contacts, and only a fifth to a quarter of them involve a tilt above 60°. They are spread over the
whole episode, so the 0.27 m reset drop is not their main cause. Play mode puts the robots on random tiles of all
difficulties, and the tall stance brings thighs and knees close to stair and box edges.

## 6. The feet never touch the ground

Each calf's collision mesh contains a ball of about 3.7 cm around the foot centre, while the `*_foot` link is only a
1.5 cm sphere welded to the calf. All 27,072 foot collision vertices of each leg lie inside the convex hull of the
calf collision mesh, at least 6.1 mm from its boundary, and the hull reaches at least 2.11 cm from the foot centre in
every direction (foot radius 1.51 cm). The calf therefore meets the ground or a stair edge first in every pose.

Consequences, measured in both versions:

* Isaac Lab 3.0: `Episode_Reward/feet_air_time` is 0.0000 in every iteration of every training run (rough and flat
  from scratch, fine-tuning from the 2.x checkpoint). Evaluations show foot contact on 0.00 and calf contact on 0.55
  of the alive steps, on Newton as well.
* Isaac Lab 2.3.2 with the original checkpoint (128 envs, 1100 steps): the largest foot contact force was 0.0 N, with
  no foot first-contact and nothing paid by `feet_air_time`; a calf touched the ground in 98.9 % of the env-steps.

So `feet_air_time` was dead weight in 2.x training as well (weight 0.01 rough, 0.25 flat), and parity is unaffected.
For new training, point the term at `.*_calf`, or shrink the calf collider in a copy of the asset so the foot sphere
protrudes; both break parity with the 2.x checkpoint.

## 7. Newton backend

**The asset.** `robot.usd` puts `PhysicsCollisionAPI` and `PhysicsMeshCollisionAPI` on the `node_STL_BINARY_` Xform
above each collision mesh. Isaac Sim PhysX accepts that layout; Newton's importer builds shapes only from Gprims, skips
all 17 colliders ("CollisionAPI applied to an unknown UsdGeomGPrim type") and the scene fails to build:
`RuntimeError: failed to initialize contact sensor ... No local entities but world_count=64`.

`scripts/make_newton_asset.py` writes `robot_newton.usd` and `configuration/robot_physics_newton.usd`, two small text
layers over the untouched `robot.usd` that move both APIs onto the 17 Mesh prims. `--check` compares all 334 composed
prims of both assets (only the collision APIs moved, plus the approximation token), `--newton` imports both with
Newton on the CPU (0 vs 17 collision shapes), and regenerating gives byte-identical files.

The Newton colliders are convex hulls. Newton maps PhysX's `convexDecomposition` to CoACD, which splits the 17 meshes
into 191 convex parts (31 per hip, up to 4494 vertices each): 5,156 env-steps/s with CoACD vs 21,166 with hulls
(rough task, 1024 envs, zero actions) and an env build of 20.6 s vs 5.5 s. The hull moves the calf's lowest point by
2.6 mm in the default pose.

**The robot preset.** `CustomQuadRobotCfg` (`rough_env_cfg.py`) is keyed by physics name: Isaac Sim PhysX keeps
`robot.usd` with native ordering; `newton_mjwarp` / `newton_kamino` use `robot_newton.usd` with `joint_ordering` and
`body_ordering` pinned to the PhysX order. Newton lists joints per leg (FL hip, thigh, calf, then FR, ...), Isaac Sim
PhysX breadth-first (all hips, then thighs, then calves), so with the preset observations and actions have one layout
on both backends.

**The converted checkpoint on Newton** (play mode, 128 envs × 2 episodes, seed 42):

| Run | Episode length | Falls | Tracking error lin / yaw | Reward / episode |
|---|---|---|---|---|
| Isaac Sim PhysX, project default (reference) | 18.07 s | 18.0 % | 0.187 / 0.101 | 1.491 |
| Newton MJWarp, preset (PhysX order) | 18.46 s | 16.0 % | 0.246 / 0.157 | 1.285 |
| Newton MJWarp, native order | 0.90 s | 99.6 % | 0.884 / 1.288 | −0.099 |

The policy transfers zero-shot only through the remap; the remaining gap is sim-to-sim.

**Speed** (rough task, 1024 envs, 30 PPO iterations from scratch, idle GPU): Newton 56,444 env-steps/s with
1,370 MiB of GPU memory, Isaac Sim PhysX 14,362 env-steps/s with 4,022 MiB, i.e. 3.9× faster with 2.9× less
memory. Removing the remap or switching Newton to Isaac Lab's actuator path changes nothing measurable (56,819 and
55,808 env-steps/s). Isaac Sim PhysX training is bit-reproducible for a fixed seed; Newton training is not.

**Zero actions** (flat task, 64 envs, 5 s, no termination): on both backends the robot lands on its calves after
about 0.3 s and sits down; the trunk ends at 0.32 m (PhysX) and 0.32-0.34 m (Newton). On Newton 14-28 % of the
robots end up on the trunk, on PhysX none. `newton_kamino` passes the same check; it was not trained.

**Logging.** Kit-less (Newton) runs print no `INFO` records, even with `--info`: Isaac Lab only lowers the level of
existing root-logger handlers, and a kit-less run has none. `scripts/check_joint_order.py` therefore finds nothing in
a Newton log; `robot.joint_names` and `robot.backend_joint_names` show the public and the Newton order.

## 8. Depth-camera task

`Isaac-Velocity-Rough-Depth-Custom-Quad-v0` is experimental and needs training from scratch. The 2.x depth task
depended on a local patch of the 2.3.2 velocity config that concatenated a 320 × 240 image into the 1-D policy
group, which cannot be built, and its checkpoint (240 inputs) matches no recorded camera config.

### 8.1 Camera

* `CameraCfg` on the front face of the trunk at (0.47, 0, 0.08) m, looking forward and pitched 50° down (the 2.x
  orientation), 60° × 46.8° field of view, 64 × 48 pixels, `distance_to_camera`, far clip 4 m.
* `scripts/depth_camera_study.py` ray-casts every pixel against the robot's own meshes in the default pose and against
  flat ground (no simulator):

  | Mount | Geometry | Robot pixels | Nearest ground ahead, centre / feet | Farthest ground | Ground ray length |
  |---|---|---|---|---|---|
  | front face (0.47, 0, 0.08), 50° | visual / collision | 0.0 % / 0.0 % | 0.63 / 0.63 m | 1.51 m | 0.56-1.32 m |
  | 2.x mast (−0.25, 0, 0.7), 50° | visual | 33.1 % | 1.26 / 0.89 m | 2.01 m | 1.20-2.86 m |
  | 2.x mast (−0.25, 0, 0.7), 50° | collision | 29.4 % | 1.21 / 0.20 m | 2.01 m | 1.20-2.86 m |
  | front face, 60° (option) | visual / collision | 0.0 % / 0.0 % | 0.54 / 0.54 m | 1.18 m | 0.54-1.00 m |

  The 2.x mast mount sees its own trunk and legs in a third of the image, so the camera moved to the front face.
  The 2.x far clip of 1.0 m would blank the top sixth of the new image; 4 m leaves room for downhill terrain.
* `update_period=0.0`: with lazy sensor updates the camera renders when the observation reads it, once per policy
  step, whatever the period. A positive period would only add a device-to-host sync per read and, with float32
  sensor timestamps, skip every other pose refresh after about 16 s of an episode.

### 8.2 Observations and model

* Actor: `policy` (48 proprioceptive values) and `depth`, a `(N, 1, 48, 64)` image in [0, 1] with ±0.005 noise
  (±2 cm, as in 2.x); rays without a hit read as far (1.0) on both renderers.
* Critic: `policy` plus a noise-free `privileged` height scan (asymmetric actor-critic).
* rsl-rl `CNNModel` actor: conv [16, 32, 32], kernels [5, 3, 3], stride 2 → 768 features, concatenated with the
  48 proprioceptive values → MLP [512, 256, 128] → 12. MLP critic.
* PPO: the rough settings with a fixed learning rate of 7e-5 and 8 mini-batches, the setting Isaac Lab 3.0 uses for
  its camera actors (`core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py:80-82`). rsl-rl's per-step NaN scan
  stays on by default; `agent.check_for_nan=false` is safe (no NaN in 15 GPU runs) and about 30 % faster.

### 8.3 Renderer

Newton Warp by default: the kit-less Warp ray tracer is documented to pair with Isaac Sim PhysX and is far lighter
than RTX for a 64 × 48 depth image. `renderer=isaacsim_rtx` selects the Isaac RTX renderer. OVRTX is not offered: it
cannot share a process with Kit, which Isaac Sim PhysX needs.

### 8.4 Camera pose fix

With Newton Warp on Isaac Sim PhysX, the camera rendered from its spawn pose for the whole run. With the 2.x policy
walking, the camera moved exactly 0 m over 150 steps while the trunk walked 3.4 m away and turned 136°, and the robot
walked into its own camera's view; a training capture changed 0 of 78 consecutive frame pairs.

Cause: the 3.0 camera reads its pose from the camera prim's Fabric world matrix, and only the Kit RTX path keeps that
matrix in step with the parent link. `update_latest_camera_pose` alone re-reads the stale value.

Fix: the observation term `body_mounted_image_depth` (a subclass of `mdp.image_depth` in `rough_env_depth_cfg.py`)
composes the trunk pose with the mount offset and writes it with `Camera.set_world_poses` before every read, only for
Isaac Sim PhysX with a non-Kit renderer. The log states it: `pose refresh from 'trunk' on (physics PhysxCfg,
renderer newton_warp)`.

Verified: camera pose error at most 2.4e-7 m; at least 99.998 % of the pixels agree with a terrain-only ray cast
(also on reset frames); Warp and RTX images agree (median difference 0.00 mm, 99th percentile 0.03 mm); 0.0016 %
own-robot pixels; no NaN or inf; 78 of 78 capture frame pairs change. RTX and Newton physics were already correct.
Two RTX quirks remain upstream: the first frame after `env.reset()` is wrong (once per run), and the reported camera
pose lags one step on reset frames while the image is right.

### 8.5 Cost

At 1024 envs a camera render takes 4.4-5.4 ms with Newton Warp and 17.5 ms with RTX, out of a 70-110 ms env step
(PhysX 36-56 ms, pose refresh 1.9-3.5 ms). Training runs at 8.5-15.7k env-steps/s, or 11.0-20.5k with
`agent.check_for_nan=false`, and needs 5.5 GB (Warp) or 7.3 GB (RTX) of GPU memory. These measurements predate the
execution defaults of section 5; with the new defaults the depth smoke test and a short training run pass as well.
No depth policy has been trained beyond 10 iterations yet.

## 9. Validation

On the RTX 5080 (16 GB) with Isaac Sim 6.2 and Isaac Lab 3.0, Isaac Sim PhysX unless noted.

| Check | Result |
|---|---|
| Install, entry point, `isaaclab list_envs` | the three task IDs are listed from the project root |
| Test suite | 31 passed with Isaac Lab installed; without it, 3 pure-Python tests pass and the rest skip |
| Joint order on Isaac Sim PhysX | as trained: hips, thighs, calves (FL, FR, RL, RR); `check_joint_order.py` reports "joint order OK" for the action term and the articulation |
| `zero_agent` on the rough, flat and depth tasks | run clean: 235 / 48 observations, the depth group as `(16, 1, 48, 64)`; contact sensor and ray caster registered |
| Replay of the converted checkpoint | walks; with the 2.x execution settings it matches its own 2.x replay ([section 5](#5-execution-differences-in-isaac-lab-30)) |
| Training | below |
| Newton | the preset builds and the checkpoint transfers through the remap; training is 3.9× faster ([section 7](#7-newton-backend)) |
| Depth task | camera pose fixed and verified ([section 8](#8-depth-camera-task)) |
| GUI | the watch command opens the Kit window, plays in real time and closes cleanly on Ctrl+C; the IsaacLab panel's Viewer Settings can follow one robot |

**Training** (2.x execution settings, seed 42):

| Run | Result |
|---|---|
| rsl-rl rough, 1024 envs, 5 iterations from scratch | completes; joint order OK in the training log |
| rsl-rl flat, 1024 envs, 5 iterations | completes |
| skrl rough, 256 envs, 5 iterations | completes (skrl 2.x warns about the 1.x yaml keys) |
| Fine-tuning from the converted checkpoint, 1024 envs, 20 iterations | continues at the replay level: the Adam state is restored (step counter +20 per iteration, learning rate 1.139e-4, then KL-adaptive), value loss 0.0157 → 0.0068, action std 0.278 → 0.275 |
| The same with learning rate 0 (control) | iteration 13,900 identical; at 13,919 mean reward 13.02 vs 13.24 fine-tuned |
| Play-mode evaluation of the fine-tuned policy (3 seeds) | reward/episode 1.497 vs 1.471, falls 18.1 % vs 18.9 %, tracking unchanged |

**Demo clip** (`docs/media/quadruped_followcam.mp4`, 960 × 540, from the 1280 × 720 recording): the converted policy
with the 2.x execution settings, 32 envs, seed 42, follow-cam on env 30. The robot starts on rough terrain next to two
robots of other envs that spawned on the same tile, crosses flat ground, turns with its command and starts down a
flight of stairs, upright until the episode times out. `eval_policy.py` on the same seed and env count measures this
episode as a time-out with tracking errors 0.140 / 0.081; 28 of the 32 robots finish the episode.

**GPU memory** (peak of the process):

| Run | GPU memory |
|---|---|
| `eval_policy.py`, rough, 128 envs | 2.9 GB |
| `isaaclab play --video`, 32 envs (1280 × 720 viewport) | 5.3 GB |
| Training, rough, 1024 envs: Isaac Sim PhysX / Newton MJWarp | 4.0 / 1.4 GB |
| Depth `zero_agent`, 16 envs | 3.3 GB |
| Depth training, 1024 envs: Newton Warp / RTX | 5.5 / 7.3 GB |

Isaac Sim PhysX and RTX allocate their GPU memory when the scene is created; when other jobs hold most of the 16 GB,
a launch fails with `PxgCudaDeviceMemoryAllocator failed to allocate memory` or an out-of-memory error.

## 10. Reproducing the results

From the project root, in the Python environment that has Isaac Lab 3.0. The commands ran on the RTX 5080 with
the current defaults (shorter `--max_iterations` for training), or before the default change with the two execution
settings passed explicitly, which resolves to the identical config. The 2.x parity check needs the 2.x sources
(defaults: `~/IsaacLab` at 2.3.2 and `~/quadruped_rough_terrain/custom_quadruped`). The CPU block needs no GPU and
takes about 15 s on that machine: 7-9 s for the tests (the first run compiles a few Warp kernels) and 1-3 s for each
script.

```bash
# CPU (about 15 s): tests, parity with the 2.x sources, conversion check, Newton asset, depth-camera study
python -m pytest
python scripts/check_obs_parity.py --markdown
python scripts/convert_rslrl_checkpoint.py checkpoints_2x/height_scanner_model.pt \
    checkpoints_2x/height_scanner_model_rslrl5.pt --verify-only
CUDA_VISIBLE_DEVICES= python scripts/make_newton_asset.py --check --newton
python scripts/depth_camera_study.py --markdown

# smoke test and joint order (Isaac Sim PhysX)
mkdir -p logs
isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 200 --viz none --info \
    2>&1 | tee logs/zero_agent_rough.log
python scripts/check_joint_order.py logs/zero_agent_rough.log     # "joint order OK"

# replay metrics (section 5): falls 0.180, tracking error 0.187 m/s, reward/episode 1.491
python scripts/eval_policy.py --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 128 --episodes 2 \
    --policy checkpoint --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --seed 42 \
    --out logs/eval/rough_s42.json
#   the Isaac Lab 3.0 execution defaults: append
#   env.sim.use_newton_actuators=true env.sim.physics.enable_external_forces_every_iteration=true

# demo clip, follow-cam on env 30 (headless; the clip lands in checkpoints_2x/videos/play/)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 --seed 42 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --video --video_length 990 \
    env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=30 \
    'env.viewer.eye=(1.9,-1.9,1.0)' 'env.viewer.lookat=(0.0,0.0,-0.2)'

# fine-tuning (resumes with the optimizer state) and training from scratch on both backends
isaaclab train --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 20 --seed 42 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt"
isaaclab train --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 30 --seed 42
isaaclab train --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 30 --seed 42 \
    physics=newton_mjwarp

# the converted checkpoint on Newton (newton_gl window)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --viz newton_gl physics=newton_mjwarp

# depth task: smoke test (the log shows "pose refresh from 'trunk' on") and training
isaaclab zero_agent --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 16 --max_steps 200 --viz none --info
isaaclab train --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 1024 --max_iterations 10 \
    agent.check_for_nan=false
```

## 11. Known issues and next steps

1. **Native actuator path on Isaac Sim PhysX** (Isaac Lab 3.0): stale joint positions in the DC-motor PD
   ([5.3](#53-the-native-actuator-path-reads-stale-joint-positions)). The project avoids it by default; return to the
   3.0 default once the joint-state refresh is fixed upstream.
2. **Feet contact** ([section 6](#6-the-feet-never-touch-the-ground)): `feet_air_time` pays nothing. For new
   training, point it at `.*_calf` or fix the foot collider in a copy of the asset.
3. **Falls in play mode**: about 19 % of play-mode episodes end in a fall, the same as the 2.x replay. Fine-tuning
   or retraining with a base-height reward (as in Isaac Lab 3.0's Go2 task) should reduce the tall-stance thigh
   contacts.
4. **Soft default stance**: with zero actions the robot sits down within 0.6 s (kp 60 / kd 1.5 do not hold the
   default crouch), and every reset drops it 0.27 m. A 0.47 m spawn height, a taller default pose or stiffer gains
   would help new training and break parity.
5. **Joint speed**: the asset's solver velocity limit of 10 rad/s, not the motor's 21 rad/s torque-speed curve, bounds
   the joint speed; `joint_velocity_limit=21.0` on the actuator lifts it for new training.
6. **Height-scan refresh** (Isaac Lab, 2.x and 3.0 alike): with `update_period=0.02` and float32 sensor timestamps,
   the scanner skips every other refresh after about 16 s of an episode. Kept for parity; `update_period=0.0` costs
   no extra ray casts with lazy sensor updates.
7. **Newton**: the 2.x policy tracks worse on Newton (reward 1.285 vs 1.491), Newton training is not
   bit-reproducible, and kit-less runs print no `INFO` records.
8. **skrl yamls** still use skrl 1.x keys, which skrl 2.x renames with a warning on every run (`lambda` →
   `gae_lambda`, `state_preprocessor` → `observation_preprocessor`, ...).
9. **Depth task**: not trained yet. Promising directions: a ray-cast depth camera against the terrain mesh
   (`MultiMeshRayCasterCameraCfg`, no renderer, works on both backends), memory for the actor (frame stacking or a
   recurrent actor, since the forward camera cannot see the ground under the robot), and a 55-60° pitch to bring the
   near edge of the image closer.
10. **Training length**: `max_iterations=80000` is inherited from the 2.x runner; pass `--max_iterations`.
