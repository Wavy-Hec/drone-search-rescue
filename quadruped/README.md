# quadruped_lab

Hector's custom 12-DOF quadruped velocity-tracking tasks, ported from Isaac Lab 2.3.2 (Isaac Sim 5.1) to
**Isaac Lab 3.0 (Isaac Sim 6.2)** as an external Isaac Lab project (template: `isaaclab --new`, external, blank).

The port details, the evidence behind each decision and the GPU validation plan are in [PORT_NOTES.md](PORT_NOTES.md).

| Task ID | Policy input | rsl_rl | skrl | Status |
|---|---|---|---|---|
| `Isaac-Velocity-Rough-Custom-Quad-v0` | 48 proprio + 187 height scan = 235 | MLP | yes | ported; 2.x checkpoint converted (`checkpoints_2x/height_scanner_model_rslrl5.pt`) |
| `Isaac-Velocity-Flat-Custom-Quad-v0` | 48 proprio | MLP | yes | ported (no 2.x checkpoint) |
| `Isaac-Velocity-Rough-Depth-Custom-Quad-v0` | 48 proprio + 64x48 depth image (camera on the trunk's front face) | CNN | no | **experimental** 3.0 redesign, retrain from scratch |

The 2.x `-Play-v0` IDs are gone: `isaaclab play` applies the env config's `play_mode()` automatically
(`--train_env_cfg` skips it).

## Physics backend

All three tasks default to **Isaac Sim PhysX** (`physics=isaacsim_physx`), the backend the 2.x checkpoints were trained
on. The Isaac Lab 3.0 velocity base would otherwise default to Newton MJWarp. Newton stays available with
`physics=newton_mjwarp`. `physics=physx` (an auto preset that would pick the uninstalled OvPhysX for headless runs) is
aliased to Isaac Sim PhysX in these configs; still prefer the explicit `physics=isaacsim_physx`.

## Setup (conda env `env_isaaclab`)

The generated `pyproject.toml` depends on `isaaclab-dev`, which exists only as the local source checkout
(`../../IsaacLab`), so install with `--no-deps` into the env that already has Isaac Lab 3.0:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd ~/isaac_ws/projects/quadruped_lab
git lfs fsck                              # USD assets and checkpoints are Git LFS objects; checks the local copies
python -m pip install --no-deps -e .      # registers the isaaclab.tasks entry point
isaaclab list_envs                        # run from the project root: lists only this project's tasks
```

This repository has no remote yet, so its working tree already holds the binaries. In a fresh clone from an
LFS-capable remote, run `git lfs install --local && git lfs pull` first.

## Run

`train` and `play` are headless unless `--viz` is given. `zero_agent` and `random_agent` open a `newton_gl` window by
default, so pass `--viz none` for headless smoke tests. `--headless` / `--enable_cameras` no longer exist in 3.0.

```bash
# smoke test (zero actions, bounded, no viewer) + joint-order check for the converted 2.x policy
mkdir -p logs
isaaclab zero_agent --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 16 --max_steps 200 --viz none --info \
    physics=isaacsim_physx 2>&1 | tee logs/zero_agent_rough.log
python scripts/check_joint_order.py logs/zero_agent_rough.log    # expects FL/FR/RL/RR hips, then thighs, then calves

# replay the converted 2.x height-scan policy (exports exported/policy.{pt,onnx} next to the checkpoint)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --viz kit physics=isaacsim_physx

# train (logs go to ./logs/rsl_rl/custom_quad_rough)
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 physics=isaacsim_physx

# experimental depth task (Newton Warp depth renderer by default; renderer=isaacsim_rtx for RTX)
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 1024 physics=isaacsim_physx
```

The depth runner uses a fixed learning rate (7e-5, 8 mini-batches), as the Isaac Lab 3.0 camera tasks do. After a
NaN-free smoke run, `agent.check_for_nan=false` removes the per-step NaN scan of the image.

Useful overrides: `--max_iterations N`, `--video --video_length 500`, `env.scene.robot.joint_ordering=...`,
`env.sim.use_newton_actuators=false` (deprecated 2.x actuator path, for A/B tests).

## Checkpoints

`checkpoints_2x/` holds copies of the 2.x checkpoints (originals in `~/quadruped_rough_terrain/logs` are untouched):

* `height_scanner_model.pt`: rsl-rl 3.x format, iteration 13900 (235 obs, 12 actions).
* `height_scanner_model_rslrl5.pt`: the same weights in rsl-rl 5.5 format, bit-identical outputs on CPU.
* `depth_camera_model.pt`: kept for reference only; its 240-dim input matches no recorded camera config.

Re-run the conversion and its CPU verification with:

```bash
PYTHONPATH=src python scripts/convert_rslrl_checkpoint.py checkpoints_2x/height_scanner_model.pt \
    checkpoints_2x/height_scanner_model_rslrl5.pt --force --verify
```

## Project layout

```
src/quadruped_lab/
  assets/custom_quad.py                      # ArticulationCfg (DC motors, package-relative USD path)
  assets/data/robots/custom_quad/            # robot.usd (canonical, Jan 15), robot_feb23.usd (alternative), configuration/
  tasks/locomotion/config/custom_quad/
    __init__.py                              # gym.register for the three task IDs
    rough_env_cfg.py, flat_env_cfg.py        # height-scan rough / flat tasks
    rough_env_depth_cfg.py                   # EXPERIMENTAL depth-camera task
    agents/rsl_rl_ppo_cfg.py                 # rsl-rl 5.5 runner cfgs (MLP, CNN for depth)
    agents/skrl_{rough,flat}_ppo_cfg.yaml    # skrl >= 2.1 configs
scripts/convert_rslrl_checkpoint.py          # rsl-rl 3.x -> 5.x checkpoint converter + CPU verification
scripts/check_obs_parity.py                  # 2.x vs 3.0 parity: observations, action, timing, MDP terms, terrain (CPU)
scripts/check_joint_order.py                 # joint order in a run log vs the order the 2.x policy expects
scripts/depth_camera_study.py                # depth-camera self-occlusion and ground coverage, ray-cast on the CPU
scripts/list_envs.py                         # wrapper for `isaaclab list_envs`
checkpoints_2x/                              # 2.x checkpoints and the converted one (Git LFS)
tests/                                       # kit-less tests (registration, configs, conversion)
```

## Development

```bash
python -m pip install pytest               # not installed by ./isaaclab.sh -i
PYTHONPATH=src python -m pytest tests -q   # 18 kit-less tests: run on the CPU without Isaac Sim
PYTHONPATH=src python scripts/check_obs_parity.py --markdown   # add --verbose to list all 214 compared 2.x settings
PYTHONPATH=src python scripts/depth_camera_study.py --markdown
```

VS Code: run the `setup_python_env` task (it calls `isaaclab --editor` inside `env_isaaclab`); the launch template has
Train/Play entries for the three tasks.

USD files, `*.pt` and `*.onnx` go through Git LFS (`.gitattributes`); `git lfs install --local` is already set up in
this clone.
