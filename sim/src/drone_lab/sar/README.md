# SAR scene: the dressed disaster site and the drone camera

`drone_lab.sar` turns the 40 m x 40 m survey terrain into a search-and-rescue site and gives the drone a
downward RGB-D camera. Task: **`DroneLab-SAR-ARL-Robot-1-v0`** (the survey task plus the site, the camera and
sun/sky lighting; the policy sees exactly the same 17 observations, so the built-in tracking policy flies it).

What is in the default scene (seed 0): 9 people (6 lying; 6 right next to "collapsed buildings"), a forklift, 9
crates (cardboard, plastic, a 1 m wooden crate), 6 barrels, 5 pallets, 8 traffic cones (4 around the helipad),
4 trees, and a landing pad with an "H" on an asphalt apron next to the start of the lawnmower path. The ground
has grass (hills, waves, border), soil (pyramids, pit), gravel (rough tile, rubble between blocks) and concrete
(block tops and walls, stairs). Every object except the trees carries a `class` semantic label: `person`,
`vehicle`, `crate`, `barrel`, `pallet`, `cone`, `landing_pad` (class ids 0-6 in that order).

## Run it

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd ~/isaac_ws/projects/drone_lab

python scripts/sar_preview.py --no_sim                 # CPU, ~3 s: runs/sar_preview_seed0/{ground_truth.json,object_map.png}
python scripts/sar_preview.py                          # GPU: renders + drone frames + checks (~1 min, 1st run longer)
python scripts/sar_preview.py --seed 3 --out sar_seed3 # another site
python scripts/sar_preview.py --bench_only             # render time of the drone camera alone + VRAM
```

In another script or the mission, pick the site with the Hydra token `env.scene.sar_site.spawn.seed=<n>` and read
the spawned scenario with `drone_lab.sar.scene.get_spawned_scenario()` (or the JSON stored on `/World/sar` as
custom data `drone_lab:sar_ground_truth`). Without installing the project, use `PYTHONPATH=src` or
`--external_callback drone_lab.register.register_tasks` as for the other tasks. Run with `physics=isaacsim_physx`.

## Files

| Module | What it does |
|---|---|
| `assets.py` | verified asset URLs (Isaac Sim 6.2 bucket), measured sizes, unit fixes (trees and DigitalTwin assets are in cm), classes, ground materials |
| `camera.py` | camera intrinsics and mount, and the pixel-budget math (below) |
| `scenario.py` | seeded placement on a CPU copy of the terrain (height queries, flatness, spacing) and the ground-truth JSON |
| `people.py` | natural poses: animation frames retargeted onto the characters and baked into static meshes, each also mirrored; `people_poses.json` holds their boxes, under-side height maps and face directions |
| `scene.py` | Isaac Lab spawner of the whole site (`SarSiteCfg`), ground materials on face subsets, landing pad, camera / light configs, persistent asset cache |
| `labels.py` | instance segmentation -> object names, classes, 2D boxes (numpy only; for auto-labelling) |
| `capture.py` | drone views, lights and ground looks of the detector dataset (`scripts/gen_yolo_dataset.py`; numpy only) |
| `topdown.py` | top-down object map figure |

## People: T-pose or natural

`people_style` (`SarSiteCfg`, `ScenarioCfg`; default `tpose`) picks how the people look:

* `tpose`: Isaac Sim's characters as shipped, standing or lying flat in their rest T-pose. The detector dataset
  and the missions so far use this; its sites and ground truth are byte for byte what they were before the
  option existed (`tests/test_sar_people.py`).
* `natural`: the same layout, re-posed. Victims lie on their back or in the recovery position, sit on the ground or
  wave; a victim next to a step slides up to it (a sitting one with its back to the block), at most 1 m from its
  T-pose place. Two or three responders in uniform are by the forklift and around the helipad, each doing
  something else: looking around, pointing ahead or walking (`group` `responders` in the ground truth). Every
  other object stays exactly where it was.

| Pose (`pose` in the ground truth) | Who | Clip frame and edits |
|---|---|---|
| `lying_back` | victim | idle; on the back, one knee bent, arms by the sides, head turned and resting on the ground |
| `lying_side` | victim | idle; recovery position: top knee and hand on the ground, face forward |
| `sitting` | victim | `Sit`; one leg out, the other knee up, a hand on the knee |
| `standing_waving`, `standing_waving_wide` | victim | `stand_idle_wave_loop`: the hand above the head, or the arm out to the side |
| `standing_looking` | responder | idle with the head of `LookAround`; arms loose, one knee relaxed |
| `standing_pointing` | responder | `push_button`: pointing ahead |
| `walking` | responder | `stand_walk_loop`: mid-stride |

Every pose also comes as its mirror image, `<pose>_mirrored`: the other arm waves, the other knee is up, the victim
lies on the other side. Each person draws its variant (from a random stream of its own, so the pose drawn for a
victim does not depend on it), and three waving people in one site are not copies of each other.

Switch it with `env.scene.sar_site.spawn.people_style=natural` (any SAR task or script) or
`scripts/sar_preview.py --people_style natural`; `--closeups` adds a low oblique close-up of one person per pose
(`--closeup_people person_03 ...` picks them), taken from the side the face looks to.

How the poses are made (`people.py`): the clips of `Isaac/People/Animations` drive the Biped rig of
`Characters/Biped_Setup.usd`, not the characters' own (Reallusion) skeletons, and Isaac Lab's apps do not load
`omni.anim.graph` that retargets them at runtime. So one clip frame is retargeted on the CPU (world-space joint
rotations from the Biped's T-pose applied to the character's T-pose; for a mirror image, the reflected rotation of
the joint on the other side), adjusted with a few joint turns (bent knees, loose arms, the recovery position) and
ground searches that put a heel, a hand or a knee on the floor, skinned with the character's own weights and
written as a static copy of the character: same meshes, UVs and materials, no skeleton, nothing mirrored on the
character itself (badges and lettering stay readable). The copies are baked on first use into
`<asset cache>/posed_people` (about 1 s each). `people_poses.json` holds every character's box, under-side height
map and face direction in every pose, so placement needs no assets; regenerate it after changing the poses with
`python -m drone_lab.sar.people --write-data`.

Placement of a posed person: its under-side height map (5 cm squares) is laid over the terrain; it rests on the
highest terrain point under it, so nothing pokes into the body, and it must rest on the parts that carry it (back,
seat, soles) with the ground under them at most 5-6 cm lower (on a slope). The spawned meshes are checked point by
point against the terrain by `scripts/sar_preview.py` (`people_on_terrain` in `checks.json`).

## Placement rules (all checked by `tests/test_sar_scenario.py`)

* The terrain is regenerated on the CPU with Isaac Lab's generator; it is identical to the imported one (the GPU
  preview measured a max vertex difference of 1e-6 m).
* An object's footprint (oriented rectangle of its posed box, plus 5-25 cm clearance) must be flat within
  8-15 cm; it rests on the highest terrain point under it, so no block edge, stair step or wall pokes into it.
  People touch the ground along a cross (body + T-pose arms), standing people only with their feet; nothing may
  reach their outstretched arms.
* Objects keep 0.6 m free space between their footprint circles (0.12 m inside a crate / barrel pile), labelled
  objects stay 0.75 m inside the survey area, trees stay 2 m below the survey flight height.
* "Lying" = rotated 90 deg about the body's left-right axis (face up or down), flat on the ground.
* 40 % of the people must be right next to a step at least 0.6 m high (within ~1.3 m of their footprint):
  `near_step`. `near_block` marks those whose step belongs to a collapsed building (a blocks or stairs tile); the
  others stand by a pyramid terrace, the crater wall or a hill. Seed 0: 9 of 9 next to a step, 6 next to a building.
* Same seed, same site, down to the millimetre. Seed 0 is the demo site and other scripts name its people
  (`--goal_pin person_08`), so `test_seed0_layout_is_stable` fails if a change moves them.

## Ground truth (`ground_truth.json`, format `drone_lab.sar.ground_truth/v1`)

One record per object: `name`, `class` (None for trees), `class_id`, `x`/`y`/`z` (= `center`, the middle of
the object's box: use it for pins and pin errors), `position` (the asset origin, e.g. the feet of a person),
`size`, `yaw_deg`, `quat_xyzw`, `pose`, `aabb_min/max`, `footprint_xy`, `in_area`, `near_block` (next to a
collapsed building), `near_step` (next to any step >= 0.6 m), `prim_path` (`/World/sar/objects/<name>`). Plus
`home` (landing pad), `survey.flight_z`, `camera` (intrinsics, mount), counts.

## Camera and the pixel budget

640 x 480 px, f = 480 px (67.4 deg x 53.1 deg), nadir, 0.1 m under `base_link`, image top = drone nose. Data:
`rgb`, `distance_to_image_plane`, `instance_segmentation` (raw ids; `camera.data.info["instance_segmentation"]`
maps ids to prim paths and `{"class": ...}`). `scene.read_camera(cam)` returns one frame as numpy (images,
labels, pose, K); `camera.pixel_to_world(uv, depth, pos_w, quat_w_ros, K)` turns a detection into a map pin and
`camera.world_to_pixel` does the reverse (the preview's GT pins land on the centres of the segmentation boxes).

Rendering: anti-aliasing is FXAA. The kit's default DLSS upscales from a lower internal resolution: blurrier
ground and screen-aligned smears around high-contrast details (it drew a dark band through the helipad). FXAA is
sharp and clean; switch back with `env.scene.sar_camera.renderer_cfg.global_settings.antialiasing_mode=DLSS`.
RTX global settings are process-wide and Isaac Lab refuses cameras with different ones: give any extra camera
`renderer_cfg.global_settings = sar_camera.renderer_cfg.global_settings.copy()` (as `scripts/sar_preview.py` does).
Lighting: the sky HDRI has its own sun (azimuth ~136 deg, elevation ~43 deg); the distant "sun" light is aligned
with it, otherwise everything casts two shadows.

Cost (RTX 5080, 1 env, one `env.step` = 10 PhysX steps + render + readback): 13 ms without the camera, 48 ms
median / 65 ms mean with it (15 Hz; p95 117 ms, rare stalls up to ~0.8 s); the camera adds ~2.4 GB of VRAM
(4.9-5.0 GB for the whole process). The trained tracking policy flew 45 s of the survey in this task at 2.6x real
time without a crash (cross-track error 0.04 m).

The survey flies at z = 3.515 m (highest terrain incl. U-turn band) + 4 m = 7.515 m, so the camera is 7.4 m
above the flat ground (GSD = 7.4 / 480 = 1.54 cm/px) and 8.85 m above the pit floor (1.84 cm/px). A lying adult
(1.61-1.80 m long, shoulders ~0.40-0.45 m) is 104-117 px long and 26-29 px wide over flat ground, and still
22-24 px wide over the pit floor, so every lying person is >= 20 px across. The swath is 9.9 m wide over flat
ground and 5.3 m at the lowest height (4 m above the highest hill), wider than the 4 m line spacing.

## Asset cache

Remote assets go to `~/.cache/drone_lab/isaac_assets` (persistent; `DRONE_LAB_ASSET_CACHE=<dir>` to move it,
`DRONE_LAB_ASSET_CACHE=tmp` for Isaac Lab's default `/tmp`). This uses Isaac Lab's own
`retrieve_file_path(url, download_dir=...)`, which checks every cached file against the server's revision before
use (offline it falls back to the cached copy with a warning). Every spawn also writes rewritten layer copies
(`isaaclab_usd_*`, ~30 MB); the next spawn prunes those older than 6 hours, except the ones a running process
still uses: each process holds a shared `flock` on its copies until it exits, so a long demo or training run
keeps its layers. The first spawn downloads ~0.5 GB (about 50 s); later ones reuse it.
MDL ground materials and the sky HDRI are loaded by Kit itself.

## Notes

* The people are Isaac Sim's characters in their rest T-pose unless `people_style=natural` (see above).
* Trees are sized for the survey height at spawn time (`survey_clearance`, copied from
  `commands.survey_target.altitude` when the config is built). If a script lowers the altitude later, also set
  `env.scene.sar_site.spawn.survey_clearance` (trees have no colliders, but the drone would fly through foliage).
* Props keep their USD mesh colliders (static); the landing pad's base disk has a collider, so a landing touches
  it like the ground (the survey task's `collision` termination fires on any contact).
* The skinned characters' USD extents are padded by ~0.15 m; their real size comes from the mesh points (the
  preview's stage check uses the points for people and the USD bounds for everything else).
* `instance_id_segmentation_fast` is not used: in Isaac Sim 6.2 it is the renderer's per-mesh id (terrain
  included) without semantics; `instance_segmentation` gives one id per labelled object with its class.
