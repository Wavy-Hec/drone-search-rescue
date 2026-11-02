# Rescue finale: air-ground teaming

After the drone's survey, a ground robot gets the drone's elevation map and the pins of what it found, plans a safe
route over the rubble and walks to a person. The robot is Hector's own quadruped (`quadruped_lab`, the converted
Isaac Lab 2.x height-scan policy) on Isaac Sim PhysX, the same backend as the drone mission, on the same seeded
terrain and dressed disaster site. A Unitree Go2 on Newton is the fallback.

```
drone elevation map ─┐
   (or ground truth) │   traversability.py           planner.py                  follower.py               Isaac Lab
pins (people, props) ┴─► slope / step / roughness ─► cost-weighted 8-conn. A* ─► pure pursuit ─────────► locomotion
                         + object footprints          start/goal snapping         (vx, vy, yaw rate)        policy
                         + robot-radius inflation     shortcuts + smoothing       clipped to the policy's   (unchanged)
                         + cross-slope cost           = route                     command ranges
                         = cost map (inf = no-go)
```

## Run it

From `~/isaac_ws/projects/drone_lab`, in `env_isaaclab` (`source ~/anaconda3/etc/profile.d/conda.sh && conda
activate env_isaaclab`):

```bash
# CPU only: plan on the ground truth of the mission's site (seed 0) and write runs/<out>/plan.png + plan.json
CUDA_VISIBLE_DEVICES= python scripts/rescue_finale.py --plan_only --out rescue_plan

# walk it with the custom quadruped (Isaac Sim PhysX) and record the follow-cam video
python scripts/rescue_finale.py --video --out finale

# 8 robots on the same route at once (small spawn jitter): success rate, falls, times
python scripts/rescue_finale.py --num_envs 8 --out finale_x8

# from the drone mission: its map and pins (a drone_gcs report directory, or a drone_lab survey run)
python scripts/rescue_finale.py --report ~/isaac_ws/projects/drone_gcs/runs/<mission> --video --out from_drone
python scripts/rescue_finale.py --report runs/lawnmower_v1 --video --out from_survey   # map only: site pins

# another person, stricter limits, another site
python scripts/rescue_finale.py --goal_pin person_04 --video --out person04
python scripts/rescue_finale.py --max_slope 22 --video --out rim                    # stays on the crater rim
python scripts/rescue_finale.py --sar_seed 3 --video --out site3

# the Go2 fallback (Newton, the backend its checkpoint was trained on)
python scripts/rescue_finale.py --robot go2 --video --out finale_go2
```

Outputs in `runs/<out>/`: `plan.png` (elevation + traversability cost with the route), `plan.json` (route, pins,
settings, and for a drone report which real person the goal pin marks), `traversability.npz` (all layers), `walk.csv`
(per step: pose, velocities, command, progress, cross-track error, distances, falls), `walk.png` (walked vs planned,
speed, distance to the goal), `summary.json` / `summary.txt` (success, time, planned and walked length, falls,
stop distance, clearance; with several robots the success rate and every robot's result) and, with `--video`,
`rescue.mp4`: a title card with the plan, the follow-cam walk with a minimap and a HUD, and an end card with the
result. `rescue_share.mp4` is the same video about 5x smaller (x264 crf 26) for a README or a chat.

**Inputs.** `--report` takes a `drone_gcs` mission report (`maps/elevation_mean.npy` + `maps/grid.json`, pins from
`detections.json`, confirmed ones only) or a `drone_lab` survey run (`elevation_map.npy`; such a run has no pins, so
the site's ground truth gives them). Without a report the planner uses the ground truth: the exact terrain mesh the
simulator builds and the people and props of the dressed site (`--sar_seed`, default 0 = the mission's demo site),
with their exact footprints. The simulator always spawns that site (`--plain_terrain`: the bare terrain with one
person at the goal). With a report the goal pin is checked against the site's ground truth: the log, `plan.json` and
the summary say which real person it marks and how far off the pin is.

**The landing pad and the apron.** The robot starts on the drone's landing pad, which sits on the flat 6 m border
around the surveyed area. A `drone_lab` survey map does not cover that border, and a `drone_gcs` map knows only the
parts the drone happened to scan. The planner fills in the border: every cell that lies entirely on it and has no
measurement gets its height (z = 0, the top of the generator's flat border; a test checks it against the terrain
mesh), the drone's cells stay as measured, and terrain the drone did not see stays unknown (no-go). `--no_apron`
plans on the drone map alone (a start off the map then moves onto the map edge, with a warning).

## The robots (`robots.py`)

| `--robot` | Policy | Physics | Planner limits |
|---|---|---|---|
| `custom` (default) | `quadruped_lab/checkpoints_2x/height_scanner_model_rslrl5.pt` (task `Isaac-Velocity-Rough-Custom-Quad-v0`) | Isaac Sim PhysX | slope 23 deg, step 0.15 m, roughness 0.05 m, robot radius 0.45 m, slope fit over 0.45 m, cross-slope weight 6 |
| `go2` | `IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-11-01_15-38-28_demo_quadruped_cont/model_1498.pt` | Newton (MJWarp) | slope 22 deg, step 0.12 m, roughness 0.05 m, robot radius 0.35 m, slope fit over 0.35 m |

The custom quadruped trained on slopes up to 0.4 rise/run (21.8 deg), stairs up to 0.23 m, boxes up to 0.1 m and
random rough ground up to 0.06 m. It is 0.89 m long and 0.47 m wide (feet included) and walks with its trunk about
0.7 m up, so it gets a larger clearance than the Go2 and the slope is fitted over half its body length. Fitted that
way, the crater walls of the demo site reach about 24 deg. At 23 deg the routes into the crater run down the fall
line and end on the crater floor. At 24 deg the planner also takes the steepest parts of the wall: on the ground
truth the robots still walked straight down them (8 of 8), but on the drone's map the last two metres then ran
across the wall, and while slowing down for the goal the robots drifted 0.4 m downhill and ended beside the person's
legs (still 8 of 8 reached). With `--max_slope 22` the route stays on the crater rim and stops on the slope 1.8-2.2 m
from the person. The `--max_*`, `--robot_radius` and `--cross_slope_weight` flags override any limit or weight;
`--task`, `--checkpoint`, `--physics`, `--speed` and `--lookahead` override the robot's walk settings.

**Steep ground along the fall line.** Turning on a steep descent is what makes this policy run away downhill. In the
walk logs on this site (about 8,700 policy steps on ground of 17 deg and steeper), yaw-rate commands below 0.4 rad/s
almost never lead to a speed above 1.2 m/s. Between 0.4 and 0.6 rad/s they do in 5 % of the cases at cruise speed
and in 44 % while the robot slows down for the goal; above 0.6 rad/s in 43-74 %. On the live mission's map the old
route came down the crater's gentler south-west corner and turned east on the wall in its last metre, into the only
free lane next to the person: the recorded robot ran away there at 2.2 m/s and tumbled into the person half a second
after arriving (with 32 robots on that route none fell, but one reached 1.62 m/s and one stopped 1.01 m from the
pin). The custom quadruped's planner therefore also pays `6 (cross_slope / max_slope)^2` per metre for the slope
across its direction of travel. The route on that map now turns on the flat border and comes straight down the fall
line: 32 of 32 robots, top speed 1.33 m/s, closest stop 1.63 m from the pin, and the recorded walk arrives cleanly.
On the rim route (`--max_slope 22`), which ends on the slope above the person, the weight also stops the run-aways
while slowing down there: no robot of 32 above 1.2 m/s instead of 8, closest stop 2.02 m from the person instead of
0.94 m. On the ground truth the route comes down the crater's corner, where the fall line itself runs diagonally,
and stays as it was. The Go2 keeps the plain cost (weight 0).

## Results (2026-11-01, RTX 5080)

**Custom quadruped, Isaac Sim PhysX, dressed site seed 0**, robots starting on the landing pad, 32 at once (small
spawn jitter), no second tries:

| Map and pins | Route | Robots reached | Time | Planned / walked | Falls |
|---|---|---|---|---|---|
| the recorded live mission's report (`drone_gcs/runs/sar_mission_hero`: 10 of 10 people found), `--goal_pin 32` | north along the border, then straight down the crater's west wall to the drone's pin of person_08 (0.05 m from the true centre) | 32/32 | 23.9-24.7 s | 16.5 / 16.3-16.6 m | 0 |
| an earlier full mission's report (`sar_demo_seed0`), T-pose people | the same way, to that mission's pin of person_08 | 32/32 | 24.9-25.4 s | 17.1 / 16.9-17.3 m | 0 |
| a partial mission report (`e2e_v2_gt`: the first 90 s of the flight, 5 people found), T-pose people | border, past the crates to the drone's pin of person_02 (0.15 m off) | 32/32 | 19.8-21.5 s | 12.8 / 12.4-12.8 m | 0 |
| the drone's survey map `runs/lawnmower_v1` (no detections: site pins) | along the crater's north rim, then down its north wall to person_08 | 32/32 | 32.4-33.1 s | 22.8 / 22.4-23.1 m | 1 |
| ground truth (default) | down the crater's south-west corner to person_08, lying in the recovery position on the crater floor 1.4 m below the rim | 32/32 | 23.5-24.1 s | 16.1 / 15.5-15.8 m | 0 |
| ground truth, `--goal_pin person_04` | past the collapsed buildings to person_04, lying face up | 32/32 | 28.3-29.3 s | 17.5 / 17.3-17.6 m | 0 |
| ground truth, `--max_slope 22` | along the crater's north rim, stops on the slope above person_08 | 32/32 | 34.7-36.8 s | 23.7 / 23.0-23.4 m | 0 |
| ground truth, `--sar_seed 3` (another site) | to that site's person_04, lying face up | 32/32 | 33.5-38.3 s | 20.7 / 20.4-20.8 m | 0 |

The two rows marked T-pose people were walked before the site's people got their natural poses; the others are on
the site as it is now. On the plain survey map the route now ends on the crater's north wall, and one robot of the 32
tipped over there 0.1 s after arriving. Mean cross-track error 2-10 cm, at most 23 cm (on the rim route up to 0.7 m:
turning on the 22 deg slope above the person, a few robots drift downhill, with the cross-slope weight or without),
walking speed 0.51-0.68 m/s at a cruise command of 0.7 m/s. The robots stop 0.8-2.2 m from the person's centre (the
person's outline or, for a drone pin, a 1.2 m circle, plus the robot's clearance); from the live mission's report
1.58-1.72 m from the real person's centre. With `--video` (one robot) on the live mission's report: 24.3 s, 16.5 m
walked, no falls, stopped 1.74 m from the pin. In the earlier `sar_demo_seed0` report the detector also pinned a "barrel" 4.3 m above the landing pad (one of
its three false alarms); the planner keeps clear of every pin, so that robot starts 0.75 m west of the pad's centre.
The live mission's report has no such pin.

**Go2 fallback, Newton**, the same dressed site: `--robot go2 --num_envs 4` reached person_08 4/4 along the crater
rim (its limits keep it off the steeper wall), 32.5-32.7 s, 23.7 m, 0 falls.

## PhysX and the textured site

The site textures the terrain by binding MDL ground materials to face subsets (`UsdGeom.Subset`, family
`materialBind`) of the terrain mesh. On Isaac Sim PhysX a collision mesh with such subsets stops colliding with the
robot: off the 4 cm landing-pad disk (a separate collider) the custom quadruped fell straight through the ground,
and on the pad it stamped in place (0 m of progress in 7 s). Binding the terrain's physics material to every
subset does not change that; without the subsets the robot walks normally. The drone never touches the terrain, so
the mission is not affected. The rescue tasks therefore spawn the site with `tasks/rescue/site.py`: the site is
unchanged, then the textured subsets move to a render-only copy of the terrain mesh (`/World/sar/ground_visual`) and
the collision mesh, which keeps its geometry, collider and physics material, is hidden. The robot's height scanner
reads that same collision mesh.

## How it works

**Traversability** (`traversability.py`). For every cell of the elevation grid:

* *slope*: a least-squares plane over a disk of `footprint_radius_m` around the cell, in degrees;
* *roughness*: the RMS residual of that plane fit;
* *step*: the largest height jump to a cell within `step_radius_m` (0.25 m) after removing the local slope. The
  local slope is the *minmod* of the one-sided differences, so a smooth incline has no step while the flat side of
  a ledge sees the full ledge height.

A cell is no-go when the slope, the step or the roughness exceeds the robot's limit, when nothing is known there
(never seen by the drone), or when it lies inside a known object footprint (circles per class, or the exact outline
from a ground-truth file). No-go cells are inflated by the robot radius, and a soft band beyond that keeps the route
off edges when there is room. Free cells cost `1 + 6 (slope/max)^2 + 4 (step/max)^2 + 3 (roughness/max)^2 + band`
per metre, so flat open ground costs 1 and the planner pays for every degree and every centimetre. With a cross-slope
weight `w` (`w_cross_slope`; the custom quadruped: 6) a move also pays `w (cross_slope/max)^2` per metre, where
`cross_slope` is the slope across its direction, from the gradient of the same plane fit: steep ground is then
crossed along the fall line, and turns move to flatter ground.

**Planner** (`planner.py`). 8-connected A* where a move costs its length times the mean cost of its two cells (plus
their cross-slope cost in the move's direction), with an octile heuristic scaled by the cheapest cell (admissible,
so the path is optimal on the grid; tests check it against Dijkstra, with and without direction costs). Diagonal
moves may not cut the corner of a no-go cell. The start snaps to the nearest free cell; the goal (a person, whose own
cells are no-go) snaps to the nearest free cell *reachable from the start*, so the robot stops next to the person.
The grid path is then shortcut by line of sight (never through a no-go cell, never more expensive than the stretch
it replaces, by the same direction-dependent cost), its corners are rounded where that stays safe, and it is resampled
every 0.1 m. Without `--goal` / `--goal_pin` the script tries the people 15-25 m from the start first, lying ones
first, nearest to 20 m first, and takes the first one it can reach.

**Follower** (`follower.py`). Pure pursuit: the closest point on the route is searched only around the previous
progress (so a route that folds back is never skipped), the target is 0.8 m further along, and the robot steers on
the arc through it (`kappa = 2 sin(alpha) / L`). Speed drops on tight arcs, when the target is off to the side and
over the last metre; above 1 rad of heading error the robot turns on the spot. A small sideways velocity pulls it
back onto the route. Every command is clipped to the policy's training ranges (`vx, vy, yaw rate` in [-1, 1]); the
minimum driving speed is 0.3 m/s because the policies stand still for much smaller commands.

**Tasks** (`drone_lab/tasks/rescue/`). `apply_rescue_task` (robot-agnostic) turns a velocity-tracking config into the
rescue task: the survey terrain (`SURVEY_TERRAIN_CFG`, same seed: the same mesh the drone mapped), optionally dressed
as the SAR site of a seed, the `base_velocity` command replaced by the path-following command under the same name
(the policy observation `velocity_commands` reads it unchanged), observation noise off, deterministic events (no
mass / centre-of-mass randomisation, no pushes), the robot spawned on the route, a flip check and a `goal_reached`
time-out. The policy observations (235 values: base velocities, gravity, command, joints, last action, 187-ray
height scan) are those of the training task, so the checkpoints drive the robots as they are.

* `DroneLab-Rescue-CustomQuad-v0`: `quadruped_lab`'s `CustomQuadRoughEnvCfg` (same robot, actuators, actions,
  observations, command ranges, falls = trunk or thigh contact as in training) on the dressed site of seed 0
  (`env.scene.sar_site.spawn.seed=<n>`). Isaac Sim PhysX only, with the two execution settings under which the
  converted 2.x checkpoint walks as it did in Isaac Lab 2.3.2 (`quadruped_lab/PORT_NOTES.md`, section 14.2):
  `sim.use_newton_actuators = False` and `enable_external_forces_every_iteration = False`. The robot respawns
  0.719 m above the ground, its training reset height.
* `DroneLab-Rescue-Go2-v0`: Isaac Lab's `Isaac-Velocity-Rough-UnitreeGo2` on Newton by default (its checkpoint flips
  within about 2 s on PhysX, a sim-to-sim gap of that checkpoint), on the plain terrain with one person unless the
  script adds the site.

After a fall the robot respawns where it fell, a little behind (`--retries`, default 2); falls are counted.

## Tests

```bash
~/anaconda3/bin/python -m pytest tests/test_teaming_*.py          # CPU (the task tests skip without Isaac Lab)
CUDA_VISIBLE_DEVICES= python tests/test_teaming_task.py              # env_isaaclab: task configs + CLI plan mode
CUDA_VISIBLE_DEVICES= python tests/run_cpu_tests.py                  # env_isaaclab: the whole project, these included
```

Traversability on synthetic grids (planes, ledges, noise, holes, footprints, inflation), the cross-slope cost (zero
along the fall line, exact across a plane; A* with direction costs against Dijkstra; the path cost the shortcuts use),
A* optimality against Dijkstra and obstacle avoidance, snapping and smoothing, failure reasons (a start off the map or
boxed in is reported once, not once per person), follower limits / steering / goal / progress and a closed-loop run on
an S-curve, input formats and figures, the apron (added around a survey map, filled in on a site-sized map; drone
cells unchanged, equal to the terrain mesh), both rescue tasks' parity with the velocity task their checkpoint was
trained on (observations, actions, robot, sensors, command ranges, physics settings), the textured site terrain moved
to a render-only mesh (in-memory USD stage), the robot profiles, and the CLI on the ground truth, a survey map and a
drone_gcs mission report (start on the landing pad, drone pins only, the goal pin checked against the truth).
