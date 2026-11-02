# quadruped_lab

Rough-terrain locomotion for a custom 12-DOF quadruped in **Isaac Lab 3.0 / Isaac Sim 6.2**, ported from Isaac Lab
2.3. The height-scan policy trained in Isaac Lab 2.x walks again at its original level: on Isaac Sim PhysX, and
through a generated Newton asset on Newton MJWarp.

<p align="center">
  <img src="docs/media/quadruped_followcam.jpg" width="720"
       alt="The custom quadruped starting down a flight of stairs in Isaac Lab 3.0">
</p>
<p align="center"><em>The converted 2.x policy in Isaac Lab 3.0: rough terrain, flat ground, then down a flight of
stairs (follow-cam on one of 32 robots).</em></p>

## Highlights

* **Three tasks** on the Isaac Lab 3.0 velocity base: rough terrain with a 187-ray height scan, flat ground, and an
  experimental depth-camera task with a CNN actor.
* **The 2.x policy, converted** from rsl-rl 3.x to the rsl-rl 5.5 checkpoint format with bit-identical actions,
  values and TorchScript export.
* **Walking parity, measured.** Two execution changes in Isaac Lab 3.0 cost the converted policy a quarter of its
  reward. Isaac Sim PhysX runs with the 2.x settings by default here; the 3.0 defaults are one override away.
* **Newton support.** A generated Newton-compatible asset and a joint-order remap let the same policy walk on Newton
  MJWarp, which trains 3.9x faster than Isaac Sim PhysX on this robot.
* **Tooling** for the port: a static 2.x/3.0 configuration parity check (214 settings), a checkpoint converter with
  verification, a joint-order check, a play-mode evaluator and a depth-camera self-occlusion study.

## Results

The converted 2.x checkpoint on the rough task, play mode, 768 episodes (seeds 42-44), Isaac Sim PhysX:

| Execution settings | Falls | Tilt > 60° | Tracking error lin / yaw | Reward / episode |
|---|---|---|---|---|
| Isaac Lab 3.0 defaults | 21.7 % | 5.9 % | 0.258 m/s / 0.157 rad/s | 1.117 |
| **Isaac Lab 2.x (default here)** | **18.9 %** | **3.6 %** | **0.196 / 0.102** | **1.471** |

In Isaac Lab 2.3.2 the same checkpoint falls in 18.8 % of its first episodes, with tracking errors 0.187 / 0.103.
On Newton MJWarp it walks zero-shot (reward 1.285 vs 1.491 on PhysX, seed 42), and training runs at 56k env-steps/s
with 1.4 GB of GPU memory vs 14k and 4.0 GB on Isaac Sim PhysX (1024 envs, RTX 5080). The method, the Isaac Lab 3.0
differences behind these numbers and the full validation are in [PORT_NOTES.md](PORT_NOTES.md).

## Tasks

| Task ID | Policy input | Agents | Status |
|---|---|---|---|
| `Isaac-Velocity-Rough-Custom-Quad-v0` | 48 proprioceptive + 187 height-scan values | rsl-rl (MLP), skrl | converted 2.x policy, validated |
| `Isaac-Velocity-Flat-Custom-Quad-v0` | 48 proprioceptive values | rsl-rl (MLP), skrl | trains; no 2.x checkpoint |
| `Isaac-Velocity-Rough-Depth-Custom-Quad-v0` | 48 proprioceptive values + 64x48 depth image | rsl-rl (CNN) | experimental, train from scratch |

`isaaclab play` applies each task's play mode itself, so there are no separate `-Play` task IDs.

## Quick start

Requirements: Isaac Lab 3.0 with Isaac Sim 6.2 (Python 3.12) and an NVIDIA RTX GPU. Validated on an RTX 5080 (16 GB).
Run everything from this folder.

```bash
git lfs pull                            # robot USD files and checkpoints are Git LFS objects
python -m pip install --no-deps -e .    # registers the tasks with the isaaclab CLI (Isaac Lab is already installed)
isaaclab list_envs                      # the three Custom-Quad tasks
```

If the checkpoints are distributed as release assets instead, place them in `checkpoints_2x/`.

```bash
# watch the converted policy walk (Kit window, real time; close the window or Ctrl+C to stop)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --viz kit --real-time

# record the demo clip headless, follow-cam on one robot (checkpoints_2x/videos/play/clip_<n>.mp4)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 32 --seed 42 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --video --video_length 990 \
    env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=30 \
    'env.viewer.eye=(1.9,-1.9,1.0)' 'env.viewer.lookat=(0.0,0.0,-0.2)'

# measure it: falls vs time-outs, tracking error, rewards per term (about a minute)
python scripts/eval_policy.py --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 128 --episodes 2 \
    --policy checkpoint --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt" --seed 42 \
    --out logs/eval/rough_s42.json
#   expect: base_contact 0.180, lin_vel_err 0.187 m/s, episode_reward_total 1.491 (deterministic)

# fine-tune the 2.x policy (resumes with its optimizer state), or train from scratch on either backend
isaaclab train --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 100 \
    --checkpoint "$PWD/checkpoints_2x/height_scanner_model_rslrl5.pt"
isaaclab train --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 1500
isaaclab train --task Isaac-Velocity-Rough-Custom-Quad-v0 --num_envs 1024 --max_iterations 1500 physics=newton_mjwarp

# experimental depth task (Newton Warp depth renderer; renderer=isaacsim_rtx for RTX)
isaaclab train --task Isaac-Velocity-Rough-Depth-Custom-Quad-v0 --num_envs 1024 --max_iterations 1500 \
    agent.check_for_nan=false
```

Logs and checkpoints of training runs go to `logs/rsl_rl/<experiment>/<run>/`. `train` and `play` run headless
unless `--viz` is given; `zero_agent` and `random_agent` open a window unless `--viz none`. Windowed runs need a
display. Training with 1024 envs needs about 4 GB of GPU memory on Isaac Sim PhysX and 1.4 GB on Newton.

### Physics backends and execution settings

* **Isaac Sim PhysX** is the default (the backend of the 2.x checkpoints). It runs with Isaac Lab's actuator path and
  the 2.3.2 TGS setting, under which the converted policy walks as it did in 2.x. For the Isaac Lab 3.0 execution
  defaults, add `env.sim.use_newton_actuators=true env.sim.physics.enable_external_forces_every_iteration=true`.
* **Newton** (`physics=newton_mjwarp` or `physics=newton_kamino`) keeps the Isaac Lab 3.0 defaults and loads
  `robot_newton.usd` with joints and bodies in the PhysX order, so a policy sees the same observation and action
  layout on both backends.
* `physics=physx` is aliased to Isaac Sim PhysX, so headless training and windowed play never switch backends.

## Project structure

```
src/quadruped_lab/
  assets/custom_quad.py                    robot config: DC motors, PhysX and Newton variants
  assets/data/robots/custom_quad/          robot.usd (+ configuration/), robot_newton.usd (generated), robot_feb23.usd
  tasks/locomotion/config/custom_quad/
    __init__.py                            task registration
    rough_env_cfg.py, flat_env_cfg.py      height-scan rough task and flat task; robot and actuator presets per backend
    rough_env_depth_cfg.py                 experimental depth-camera task
    agents/                                rsl-rl 5.5 runners (MLP; CNN for depth) and skrl configs
scripts/
  eval_policy.py                           play-mode metrics for a policy (GPU)
  convert_rslrl_checkpoint.py              rsl-rl 3.x -> 5.x checkpoint converter with CPU verification
  check_obs_parity.py                      2.x vs 3.0 configuration parity, read statically from the 2.x sources
  check_joint_order.py                     joint order of a run log vs the order the 2.x policy expects
  make_newton_asset.py                     writes and checks robot_newton.usd
  depth_camera_study.py                    depth-camera self-occlusion and ground coverage (CPU ray cast)
checkpoints_2x/                            the 2.x checkpoints and the converted one
docs/media/                                README media
tests/                                     CPU tests (no Isaac Sim launch)
PORT_NOTES.md                              engineering notes: port, measurements, findings, validation
```

## Checkpoints

| File | Contents |
|---|---|
| `checkpoints_2x/height_scanner_model_rslrl5.pt` | the rough-terrain policy in rsl-rl 5.5 format (use this one) |
| `checkpoints_2x/height_scanner_model.pt` | the original Isaac Lab 2.x checkpoint (rsl-rl 3.x, iteration 13,900) |
| `checkpoints_2x/depth_camera_model.pt` | the 2.x depth checkpoint, for reference only (its input matches no recorded camera config) |

Re-run the conversion and its verification on the CPU:

```bash
python scripts/convert_rslrl_checkpoint.py checkpoints_2x/height_scanner_model.pt \
    checkpoints_2x/height_scanner_model_rslrl5.pt --force --verify
```

## Development

```bash
python -m pip install pytest pre-commit
python -m pytest              # CPU only; without Isaac Lab installed the pure-Python tests run and the rest skip
pre-commit run --all-files    # ruff check + format, codespell, file hygiene
```

Other checks that run on the CPU: `python scripts/check_obs_parity.py` (needs the 2.x sources),
`python scripts/make_newton_asset.py --check --newton` and `python scripts/depth_camera_study.py --markdown`.

## Acknowledgements

Built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) and Isaac Sim, with
[rsl-rl](https://github.com/leggedrobotics/rsl_rl), [skrl](https://github.com/Toni-SM/skrl) and
[Newton](https://github.com/newton-physics/newton). The project layout comes from the Isaac Lab external-project
template.
