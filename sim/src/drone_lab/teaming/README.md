# Rescue finale: air-ground teaming

After the drone's survey, a ground robot (a Unitree Go2 running Isaac Lab's rough-terrain locomotion policy) gets
the drone's elevation map and the person pins. It plans a safe route over the rubble and walks to a person.

```
drone elevation map ─┐
   (or ground truth) │   traversability.py           planner.py                  follower.py               Isaac Lab
pins (people, props) ┴─► slope / step / roughness ─► cost-weighted 8-conn. A* ─► pure pursuit ─────────► Go2 policy
                         + object footprints          start/goal snapping         (vx, vy, yaw rate)        (unchanged)
                         + robot-radius inflation     shortcuts + smoothing       clipped to the policy's   walks on the
                         = cost map (inf = no-go)     = route                     command ranges            same terrain
```

## Run it

From `~/isaac_ws/projects/drone_lab`, in `env_isaaclab` (`source ~/anaconda3/etc/profile.d/conda.sh && conda
activate env_isaaclab`):

```bash
# CPU only: plan on the terrain ground truth and write runs/<out>/plan.png + plan.json
CUDA_VISIBLE_DEVICES= python scripts/rescue_finale.py --plan_only --out rescue_plan

# walk it with the Go2 (Newton physics, the backend the checkpoint was trained on) and record the follow-cam video
python scripts/rescue_finale.py --video --out rescue_gt

# the dressed disaster site the drone flies over (people, props, ground materials); the robot starts on the
# landing pad and walks to a person picked from the site's ground truth (about 20 m away, reachable)
python scripts/rescue_finale.py --sar_seed 0 --video --out rescue_sar0
python scripts/rescue_finale.py --sar_seed 0 --goal_pin person_08 --video --out crater   # a chosen person

# from a drone run: its elevation map and pins (drone_lab survey run or drone_gcs run directory)
python scripts/rescue_finale.py --run runs/<survey_run> --pins <detections.json> --video --out from_drone
# the drone's survey map of the plain terrain + the dressed site's people and props (exact footprints)
python scripts/rescue_finale.py --run runs/lawnmower_v1 --sar_seed 0 --video --out rescue_drone_sar0

# the drone mission's backend (Isaac Sim PhysX) resolves too, but today's Go2 checkpoint (trained on Newton) does
# not walk there: it flips within 1-2 s (see "Physics backend" below)
python scripts/rescue_finale.py --physics isaacsim_physx --video --out rescue_physx

# several robots on the same route at once (small spawn jitter): a success rate
python scripts/rescue_finale.py --num_envs 8 --out rescue_x8
```

Watch it live in Isaac Lab's player (plans its own route on the stage terrain at start-up, loops the walk):

```bash
PYTHONPATH=src isaaclab play --rl_library rsl_rl --external_callback drone_lab.register.register_tasks \
    --task DroneLab-Rescue-Go2-v0 --viz kit \
    --checkpoint ~/isaac_ws/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-11-01_15-38-28_demo_quadruped_cont/model_1498.pt
```

Outputs in `runs/<out>/`: `plan.png` (elevation + traversability cost with the route), `plan.json` (route, pins,
settings), `traversability.npz` (all layers), `walk.csv` (per step: pose, velocities, command, progress,
cross-track error, distances, falls), `walk.png` (walked vs planned, speed, distance to the goal),
`summary.json` / `summary.txt` (success, time, planned and walked length, falls, stop distance, clearance) and,
with `--video`, `rescue.mp4`: a title card with the plan, the follow-cam walk with a minimap and a HUD, and an end
card with the result. `rescue_share.mp4` is the same video about 5x smaller (x264 crf 26: 6-11 MB instead of
29-55 MB) for a README or a chat.

**A drone map and the landing pad.** A `drone_lab` survey map covers the surveyed 40 m x 40 m terrain, but the
landing pad (where the robot starts) sits on the flat 6 m border around it. When the start lies off the map, the
planner adds that border to the map: the drone's cells stay as measured, the added cells that lie entirely on the
border get its height (z = 0, the top of the generator's flat border; a test checks it against the terrain mesh),
and terrain the drone did not see stays unknown (no-go). The log says so. `--no_apron` plans on the drone map
alone; the start then moves onto the map edge, with a warning.

## Results (2026-11-01, RTX 5080, Newton physics)

| Run | Route | Outcome | Time | Planned / walked | Falls | Stopped from the person |
|---|---|---|---|---|---|---|
| `--video` (ground truth, default goal) | landing pad -> crater rim -> gap between collapsed buildings | reached | 33.0 s | 24.0 / 23.8 m | 0 | 1.56 m (body centre; lying) |
| `--sar_seed 0 --video` (dressed site) | landing pad -> crater's west rim -> person_05 on the hill | reached | 25.8 s | 18.7 / 18.5 m | 0 | 0.88 m |
| `--sar_seed 0 --goal_pin person_08 --max_slope 26 --video` | down into the crater (to z = -1.4 m) | reached | 21.6 s | 14.5 / 14.5 m | 0 | 1.89 m |
| `--run runs/lawnmower_v1 --sar_seed 0 --video` (the drone's map, Isaac Sim survey) | landing pad -> north along the border -> crater rim -> person_03 lying among the collapsed buildings | reached | 34.9 s | 25.2 / 25.0 m | 0 | 1.53 m |
| `--num_envs 8` x 2 routes x 2 checkpoints, no retries | default route; crater descent (`--goal -14.77 -4.85`) | 8/8 each | 32-35 s | 24.0 m; 24.1 m planned | 0 | - |
| stress: `--max_slope 30 --max_step 0.2 --max_roughness 0.08`, 8 robots x 2 checkpoints | hill (`--goal -16 5`, up to z = 2.2 m); crater (down to z = -1.4 m) | 8/8 each | 22-43 s | 30.4 m; 14.6 m planned | 0 | - |

Mean cross-track error 1.3-4.5 cm (max 6-17 cm). The policy is
`IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-11-01_15-38-28_demo_quadruped_cont/model_1498.pt`: today's 600-iteration
Go2 run resumed for 900 more iterations on Newton. On the training curriculum it falls 4x less often
(base-contact terminations 5.1 % vs 19.2 %, mean reward 26.9 vs 19.0); on the finale routes both checkpoints
reach the person every time, so the planner's limits leave the robot a safety margin.

**Physics backend.** The Go2 checkpoint was trained on Newton (`newton_mjwarp`), and the task uses Newton by
default; the survey terrain, the dressed SAR site (MDL ground materials, props, people) and the Kit follow camera all
work with it. On Isaac Sim PhysX (`--physics isaacsim_physx`) the same task and spawn fail: 4/4 robots flip within
0.9-2.1 s, with either actuator execution path (`env.sim.use_newton_actuators=false` too), so it is a sim-to-sim gap of
the checkpoint, not of the task. Sharing one PhysX scene with the drone would need a Go2 policy trained (or
fine-tuned) with `physics=isaacsim_physx`.

## How it works

**Traversability** (`traversability.py`). For every cell of the elevation grid:

* *slope*: a least-squares plane over a disk of `footprint_radius_m` (0.35 m) around the cell, in degrees;
* *roughness*: the RMS residual of that plane fit;
* *step*: the largest height jump to a cell within `step_radius_m` (0.25 m) after removing the local slope. The
  local slope is the *minmod* of the one-sided differences, so a smooth incline has no step while the flat side of
  a ledge sees the full ledge height.

A cell is no-go when the slope exceeds 22 deg, the step 0.12 m or the roughness 0.05 m (the top of the Go2's
training curriculum), when nothing is known there (never seen by the drone), or when it lies inside a known object
footprint (circles per class, or the exact outline from a ground-truth file). No-go cells are inflated by the robot
radius (0.35 m), and a soft band beyond that keeps the route off edges when there is room. Free cells cost
`1 + 6 (slope/22)^2 + 4 (step/0.12)^2 + 3 (roughness/0.05)^2 + band` per metre, so flat open ground costs 1 and
the planner pays for every degree and every centimetre.

**Planner** (`planner.py`). 8-connected A* where a move costs its length times the mean cost of its two cells, with
an octile heuristic scaled by the cheapest cell (admissible, so the path is optimal on the grid; a test checks it
against Dijkstra). Diagonal moves may not cut the corner of a no-go cell. The start snaps to the nearest free cell;
the goal (a person, whose own cells are no-go) snaps to the nearest free cell *reachable from the start*, so the
robot stops next to the person. The grid path is then shortcut by line of sight (never through a no-go cell, never
more expensive than the stretch it replaces), its corners are rounded where that stays safe, and it is resampled
every 0.1 m.

**Follower** (`follower.py`). Pure pursuit: the closest point on the route is searched only around the previous
progress (so a route that folds back is never skipped), the target is 0.8 m further along, and the robot steers on
the arc through it (`kappa = 2 sin(alpha) / L`). Speed drops on tight arcs, when the target is off to the side and
over the last metre; above 1 rad of heading error the robot turns on the spot. A small sideways velocity pulls it
back onto the route. Every command is clipped to the policy's training ranges (`vx, vy, yaw rate` in [-1, 1]); the
minimum driving speed is 0.3 m/s because the policy stands still for much smaller commands.

**Task** `DroneLab-Rescue-Go2-v0` (`drone_lab/tasks/rescue/`). The config is Isaac Lab's
`Isaac-Velocity-Rough-UnitreeGo2` with only what the finale needs changed (`apply_rescue_task`, robot-agnostic):
the survey terrain (`SURVEY_TERRAIN_CFG`, same seed: the same mesh the drone mapped), the `base_velocity` command
replaced by the path-following command under the same name (the policy observation `velocity_commands` reads it
unchanged), observation noise off, deterministic events (no mass / CoM randomisation, no pushes), the robot spawned
on the route, a flip check and a `goal_reached` time-out. The policy observations (235 values: base velocities,
gravity, command, joints, last action, 187-ray height scan) are identical to the training task, so the Go2
checkpoint drives it as is. After a fall the robot respawns where it fell (`--retries`, default 2); falls are
counted in the summary.

## Tests

```bash
~/anaconda3/bin/python -m pytest tests/test_teaming_*.py          # CPU (the task tests skip without Isaac Lab)
CUDA_VISIBLE_DEVICES= python tests/test_teaming_task.py              # env_isaaclab: task config + CLI plan mode
CUDA_VISIBLE_DEVICES= python tests/run_cpu_tests.py                  # env_isaaclab: the whole project, these included
```

Traversability on synthetic grids (planes, ledges, noise, holes, footprints, inflation), A* optimality against
Dijkstra and obstacle avoidance, snapping and smoothing, failure reasons (a start off the map or boxed in is
reported once, not once per person), follower limits / steering / goal / progress and a closed-loop run on an
S-curve, input formats and figures, the border added around a survey map (lattice kept, drone cells unchanged,
equal to the terrain mesh, start on the landing pad from the CLI), and the task config's parity with the Go2
training task.
