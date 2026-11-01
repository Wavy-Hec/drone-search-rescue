# drone_gcs: ground control station for the simulated drone survey

The simulator flies the drone. **drone_gcs** is the ground-control station and companion computer for it.
It receives the drone's telemetry, mapping ray hits and camera frames over a local socket, and **builds the map
live while the drone uncovers the area**. Unknown ground stays as dark-grey fog until a sensor has seen it. It
also runs object detection (YOLO) on the camera feed, places each detection on the map in world coordinates,
merges repeated sightings into an object list, shows everything in a ground-control style dashboard
(recorded to mp4) and writes a mission report.

![dashboard](docs/images/dashboard_live.png)

*Mid-mission (fake drone): camera feed with detection boxes and their map coordinates. The map shows the
elevation layer: grey = not yet scanned, flown trail in orange, planned path dashed, camera footprint in cyan,
labelled objects. The HUD is along the bottom.*

## Architecture

```
  SIMULATOR (env_isaaclab: Isaac Sim 6.2 / Isaac Lab 3.0,              GCS (conda env 'vision': numpy, OpenCV,
  project drone_lab; or scripts/fake_drone.py)                          Ultralytics) - scripts/gcs.py
 +-----------------------------------------+                         +--------------------------------------------+
 | RL flight policy -> multirotor physics   |                         | Receiver (reader threads, decode frames)    |
 | downward RayCaster (mapping grid)        |  drone_gcs.link.Sender  |   |                                         |
 | RGB(+depth) camera                       |  ---------------------> |   v                                         |
 |                                          |  UNIX socket            | GCSState --> ElevationMap (2.5D, running    |
 | Sender: send_mission / send_telemetry /  |  /run/user/1000/        |   |          stats, fog-of-war mask)         |
 |   send_scan / send_camera / send_end     |  drone_gcs.sock         |   |      --> Orthomosaic (pixels -> world)    |
 |   (stdlib + numpy only; non-blocking:    |  (or tcp://host:47800)  |   |      --> DetectorWorker (YOLO, thread)    |
 |   bounded queue, drops oldest frames,    |                         |   |           -> Geolocator -> ObjectTracker  |
 |   auto-reconnect, never raises)          |  length-prefixed frames |   v                                         |
 +-----------------------------------------+  JSON header + raw LE   | Dashboard (cv2 window 12 FPS + mp4)         |
                                              buffers (PROTOCOL.md)  | Report: maps, detections, trajectory, summary|
                                                                     +--------------------------------------------+
```

* **Link** (`drone_gcs/link.py`, `drone_gcs/protocol.py`): MISSION, TELEMETRY, SCAN, CAMERA, GT_OBJECTS
  and END messages. No pickle: dtype and shape travel in a JSON header, so the two ends can run different
  numpy versions (env_isaaclab has 2.3, vision has 2.5; a test checks this).
  Frames, units and conventions are in [PROTOCOL.md](PROTOCOL.md).
* **Mapping** (`drone_gcs/mapping.py`, numpy only):
  * elevation map: per-cell count, mean, min, max, std and coverage mask
  * orthomosaic: camera pixels back-projected with depth, or with the elevation map when there is no depth
  * detection geolocation: box → pixel → ray → depth or map intersection
  * tracker: same class within 2 m means the same object, with confidence, hits, first/last seen
  * evaluation against ground truth
* **Detector** (`drone_gcs/detector.py`):
  * Ultralytics YOLO, default `~/isaac_ws/assets/models/yolo26n.pt` (COCO)
  * a colour-blob stand-in for the fake world
  * a worker thread that always runs the newest frame, so the UI never waits on the detector
* **Dashboard** (`drone_gcs/dashboard.py`): OpenCV canvas 1600×900 (Tokyo Night colours), window plus mp4.
  `--headless` only records.
* **Report** (`drone_gcs/report.py`): written at END or on quit.

## Run the demo (fake drone, no Isaac Sim)

Two terminals, both `conda activate vision`, in `~/isaac_ws/projects/drone_gcs`:

```bash
python scripts/gcs.py --detector blob --name my_demo   # 1) the GCS window (waits for the sim)
python scripts/fake_drone.py --speedup 4                # 2) the stand-in simulator (3 min survey at 1x)
```

The fake drone flies a lawnmower survey over a synthetic 40 × 40 m world: hills, buildings, trees, a dirt
road, and 10 objects (car = red, tent = blue, person = magenta, tarp = yellow). It sends 20 Hz telemetry,
10 Hz scans from a 5 × 5 m grid of downward rays at 0.25 m spacing (like Isaac Lab's `RayCaster` +
`GridPatternCfg`), 5 Hz camera frames (640×480, 75° FOV, RGB + depth, rendered from the height field) and
the ground-truth objects. COCO YOLO does not know these coloured shapes, so use `--detector blob` with the
fake drone. With the default `--detector yolo` you get false positives such as "sports ball" for round trees.

* Headless, record only, exits after END: `python scripts/gcs.py --headless --detector blob`
* Other fake drone options: `--no-depth` (the GCS then intersects rays with its own map), `--jpeg`,
  `--mount fixed` (camera tilts with the drone), `--area 60 40`, `--spacing 6`, `--speed 5`, `--seed 1`.
  See `--help`.
* Either side can start first. The link reconnects, and MISSION (and GT_OBJECTS) are re-sent on every
  reconnect. A GCS restarted **mid-mission** starts a new, empty map for that mission: earlier data is not
  replayed (only what was still queued on the sim side while no GCS was listening).

### Keys

| key | action |
|---|---|
| `m` | map layer: **elevation** → **photo** (orthomosaic) → **coverage** (ray hits per cell / photo-only) |
| `e` | elevation statistic: mean ↔ max |
| `g` | show the ground-truth objects (evaluation only; hollow squares) |
| `c` | camera of the next drone (with several drones) |
| `s` | screenshot of the canvas → `runs/<name>/screenshots/` |
| `q` / `Esc` | quit (writes the report). Closing the window does the same |

Arrow keys and other non-character keys are ignored. (With OpenCV's Qt backend they arrive as X key codes
whose low byte is a letter: Left used to read as `q` and quit, Right as `s`.)

HUD (bottom bar):

| Cell | Shows |
|---|---|
| Mission time | sim time since the mission started |
| Altitude AGL | height above the mapped ground under the drone |
| Speed | horizontal and vertical speed |
| Heading | 0° = north = +Y |
| Waypoint | waypoint i/N |
| Coverage | scan % and photo %, inside the survey area |
| Objects | confirmed objects and raw detections |
| Link | message rate, MB/s, inbound frames dropped |
| Detector | detector ms and fps |

## Connecting the Isaac Sim survey

The sender only needs the standard library and numpy, so it imports inside Isaac Sim's Python:

```python
import sys
sys.path.insert(0, "~/isaac_ws/projects/drone_gcs")   # or: pip install --no-deps -e <this repo>
from drone_gcs.link import Sender
from drone_gcs.isaac_bridge import IsaacLabStreamer               # duck-typed Isaac Lab 3.0 adapter

stream = IsaacLabStreamer(Sender(name="isaac-survey"), env_id=0)   # default unix:/run/user/1000/drone_gcs.sock
stream.mission(area=(-20, 20, -20, 20), resolution=0.25, planned_path=path_xyz, survey_altitude=8.0,
               name="survey_lawnmower")                         # local frame; world=True subtracts origin
stream.gt_objects([{"class": "car", "pos": [x, y, z]}, ...])    # optional, evaluation only (after mission)
for step in range(n_steps):
    obs, *_ = env.step(policy(obs))
    t = step * env.unwrapped.step_dt
    stream.telemetry(t, robot, wp_index=k, carrot=carrot_xyz)   # robot = env.unwrapped.scene["robot"]
    stream.scan(t, scanner)                                     # scene["mapping_scanner"] (RayCaster)
    stream.camera(t, camera)        # scene["camera"]: rgb + distance_to_image_plane; sends new captures only
stream.end("complete")
```

`stream.camera()` sends a frame only when Isaac Lab's capture counter (`camera.frame`) has advanced, so it
can be called every step whatever the camera's `update_period` (it returns `False` otherwise). Under OVRTX
`async_rendering` it takes the image's pose from `camera.data.info["rgb"]["capture"]`.
`IsaacLabStreamer(origin=env.scene.env_origins[i])` subtracts the env origin from everything it reads from
Isaac Lab; positions you pass to `mission()` / `gt_objects()` are local unless you add `world=True`.

Or call the `Sender` methods directly with numpy arrays, CPU or GPU torch tensors, or Isaac Lab `ProxyArray`s:
`send_telemetry(t, pos, quat_wxyz, vel, wp_index=, carrot=)`, `send_scan(t, points_Nx3, origin=)`,
`send_camera(t, rgb, K, pos, quat_wxyz_ros, depth=, jpeg=False)`, `send_gt_objects([...])`, `send_end()`.

* Every `send_*` call copies its data (arrays and extra keyword fields) and returns at once. It **never
  raises**: it returns `False` and logs instead. A background thread does the I/O, in call order. When the GCS
  is slow or absent, the oldest camera frames are dropped.
* Each data message and END carries the current `mission_id`; the GCS ignores ones from another mission. With
  several drones, use one `mission_id` for all of them (END from any drone ends the mission).
* **Quaternions are (w, x, y, z) on the wire. Isaac Lab 3.0 gives (x, y, z, w).**
  `drone_gcs.geometry.quat_xyzw_to_wxyz` converts. The camera orientation is the **ROS optical** frame
  (`camera.data.quat_w_ros`). PROTOCOL.md has the conversion from `quat_w_world` and `quat_w_opengl` too.
* Ray-caster misses (`inf`) can be sent as they are.
* TCP instead of the UNIX socket (for example from a container): `Sender("tcp://127.0.0.1:47800")` with
  `python scripts/gcs.py --addr tcp://0.0.0.0:47800`, or set `DRONE_GCS_ADDR` on both sides.
* Import check (it passes, see the tests):
  `conda activate env_isaaclab && python -c "import sys; sys.path.insert(0, '~/isaac_ws/projects/drone_gcs'); from drone_gcs.link import Sender"`.
  `pip install --no-deps -e .` also works (base dependency: numpy only). It was **not** done, to leave
  env_isaaclab untouched.

## Outputs: `runs/<name>/`

| file | content |
|---|---|
| `dashboard.mp4` | the exact dashboard canvas at the dashboard rate. OpenCV writes it (mp4v), then system ffmpeg re-encodes it to H.264 so it plays in browsers and Obsidian (`--no-transcode` keeps mp4v) |
| `final_map_elevation.png`, `final_map_photo.png`, `coverage.png` | final maps, each with the flown trajectory, planned path, detections and (if known) ground truth |
| `maps/` | raw arrays: `elevation_{mean,max,min}.npy`, `hit_count.npy`, `photo_rgb.png`, `photo_filled.npy`, `grid.json`. Arrays are (ny, nx), row 0 = south |
| `detections.csv` / `.json` | objects: class, x, y, z (world / local ENU, m), confidence (mean, max), hits, first/last seen, spread, confirmed. Also the matched ground-truth id and error when GT_OBJECTS was sent |
| `trajectory.csv` | t, position, quaternion (wxyz), velocity, waypoint index, carrot, AGL |
| `coverage_timeline.csv` | scan and photo coverage vs sim time |
| `summary.json` | area, coverage %, path length, flight time, speeds, detection counts, precision / recall / mean localization error vs ground truth, link and detector statistics |
| `screenshots/` | `s` key or `--screenshot-at` |

`runs/` is git-ignored. Final maps of the 40 × 40 m fake survey (`runs/demo_headless`, copied to `docs/images/`):

| elevation | orthomosaic | coverage |
|---|---|---|
| ![elevation](docs/images/final_map_elevation.png) | ![photo](docs/images/final_map_photo.png) | ![coverage](docs/images/coverage.png) |

## Detection and fine-tuning on simulator images

* Default: COCO nano weights (`--weights` to change), `--conf 0.25 --iou 0.45 --imgsz 640 --device cpu`.
  YOLO runs on the **CPU** by default (about 19 ms per 640×480 frame, plenty for a 5–10 Hz camera): the GPU
  belongs to Isaac Sim and the RL jobs. `--device cuda:0` opts in; `--device auto` uses the GPU only when
  `nvidia-smi` shows at least 2 GiB free on it.
* `--classes person,car` keeps only those classes; a name matches the model's class name or the name after
  `--class-map` (`--classes car --class-map truck=car` keeps cars and trucks). A name the model does not
  have is an error if nothing would be left, else a warning. `--class-map truck=car,bench=` renames or drops
  classes.
* The GCS process sets `YOLO_OFFLINE=1` (no update checks or analytics from the GCS). The env-local
  `YOLO_CONFIG_DIR` of 'vision' is left as it is.
* With several drones every camera gets detection: the worker keeps the newest frame per drone and serves
  the drones in turn.
* Boxes that touch the image border are partial views. They are shown, but not put on the map
  (`--edge-margin 0` changes that).
* Fine-tuning workflow (nothing has been trained yet):
  1. Capture RGB + `instance_id_segmentation_fast` (+ its `info` JSON) from an Isaac Lab camera. The
     snippet is in the docstring of `scripts/seg_to_yolo.py`.
  2. `python scripts/seg_to_yolo.py --src <capture> --out ~/isaac_ws/datasets/drone_yolo --classes car,tent,person,tarp`
     writes the YOLO labels (normalized cx cy w h) and `data.yaml`. Objects cut off by the image border keep
     their visible box (`--max-truncation 0.6` would drop those with under 40 % visible). Tiny instances
     (and any dropped object of a wanted class) are painted over with the image's mean colour, so the
     detector is not taught that a visible object is background (`--no-mask-dropped` keeps the pixels).
     Train/val are split in blocks of consecutive frames (no near-duplicate leakage); small captures use
     shorter blocks, so val stays near 15 % and train is never empty (e.g. 40 frames → 34 / 6).
  3. `python scripts/train_detector.py --data ~/isaac_ws/datasets/drone_yolo/data.yaml --dry-run` checks it.
     Without `--dry-run` it trains from the nano weights. Defaults for about 2000 images on a 16 GB GPU:
     120 epochs, patience 30, imgsz 640, batch 32, RAM cache, flipud aug. It copies `best.pt` to
     `models/detector.pt` (Git LFS) and writes `models/detector_metrics.json`.
  4. `python scripts/gcs.py --weights models/detector.pt`

## Tests

```bash
./run_tests.sh                      # all 77 tests, ~25 s, CPU only (activates 'vision' itself)
DRONE_GCS_SKIP_E2E=1 ./run_tests.sh # without the end-to-end run
```

| Area | What the tests check |
|---|---|
| Wire format | Round trips of all dtypes and shapes, JPEG, rejection of bad frames, ProxyArray-like objects converted without touching their implicit-torch bridge |
| Link | Exact call order with MISSION / GT_OBJECTS, mission_id on data. A mission change while the GCS is absent arrives as MISSION A, A's data, END A, MISSION B. A GCS restarted mid-mission gets the MISSION of the queued data. Extra fields are copied at call time. No gap between taking and writing an item (flush). A sender with no GCS never blocks or raises. A GCS that stops reading makes the sender drop camera frames without blocking the sim loop. TCP. Stale socket cleanup |
| Geometry | Isaac Lab xyzw → wxyz, and the world/opengl → ROS camera conversion checked against Isaac Lab 3.0's fixed constants and a re-implementation of its matrix formula (no Isaac Lab import). Projection round trips |
| Mapping | Running statistics vs brute force. Cell centres on the area lattice for any area size and padding. Grid cap for huge missions. Orthomosaic from depth and from the map, self-occlusion depth ignored. Geolocation by depth, map and plane. Tracker merging. Precision and recall |
| Isaac adapter | Only one env row leaves the GPU. Each capture is sent once (`camera.frame`). Async-rendering capture pose. `world=True` origin handling |
| App | Arrow keys neither quit nor take screenshots. Data / END of another mission ignored; a new mission clears an old END. Rollover keeps ground truth and drops the old mission's late detections. Every drone gets detection. Video keeps real time through stalls |
| Fake world | Rays and depth lie on the height field |
| Detectors | The blob detector finds every object. The worker keeps the newest frame. YOLO wrapper: 2 CPU inferences on Ultralytics' `bus.jpg`; class filters validated (typos raise) and matched after `--class-map` |
| Dataset tools | `seg_to_yolo` on synthetic segmentation gives exact labels, keeps cut-off boxes, paints over tiny ones. Block split gives train data and ~15 % val from 8 to 2000 frames. `train_detector --dry-run` |
| Sender import | The sender imports without cv2, torch or PIL. **A sender process in env_isaaclab** sends to a receiver in vision, and arrays arrive bit-exact |
| Dashboard and report | All three layers render. All report files are written |
| End to end | Fake drone process → socket → headless GCS process. Coverage grows monotonically to over 98 %. Elevation error is under 1 cm RMSE on smooth cells (measured 2.6 mm) and under 5 mm median. All synthetic objects are found with no false positives and a mean horizontal error under 25 cm |

Measured on the 40 × 40 m fake survey (`runs/demo_headless`; repeated after the fixes with the same results):

| Metric | Value |
|---|---|
| Scan coverage | 99.9 % |
| Photo coverage | 100 % |
| Objects | 10 / 10 found, precision 1.0, recall 1.0 |
| Mean horizontal error | 4–5 cm (max 6 cm) |
| Elevation, smooth cells | 2.5 mm RMSE (median error 0.7 mm) |
| Elevation, all cells | 8 cm RMSE |
| GCS loop | 14–30 ms per frame (fake drone at 8× speed; depends on the CPU load from other jobs) |
| Fake camera render | about 20 ms per frame |
| YOLO26n inference | about 19 ms on CPU |

Elevation RMSE over all cells is higher because cells that straddle a wall are sampled by the 0.25 m ray
lattice on one side only.

## Notes and limitations

* The elevation map is 2.5D (one height per cell: no overhangs). The mean is the right statistic for terrain.
  Use `e` (max) to see roofs and canopy edges.
* The orthomosaic is a back-projection with a "most top-down view wins" rule, not a true orthophoto. With
  depth it is geometrically correct. Without depth it uses the elevation map, so cells not yet scanned use
  a ground plane.
* Geolocation uses the visible surface at the box centre (nadir view) or the box bottom centre (oblique
  view). For tall objects, z is the top of the object. Horizontal error is the primary metric.
* The fake world is a stand-in. Walls are rendered from the height field (streaky textures on vertical faces).
  Its job is to exercise the geometry and the pipeline, not to look photoreal.
* Several drones are supported on the map (`drone_id`), and every drone's camera gets detection. The camera
  panel shows one drone (`c` cycles). All drones share one mission: one `mission_id`, and END from any
  drone ends it. No run so far used more than one drone.
* Depth closer than 0.5 m (`--depth-min`) is treated as the drone itself (legs, propellers) and not mapped.
* Very large missions: grids above 4 M elevation cells get a coarser resolution (logged), and a large map
  layer is redrawn at most every few hundred ms instead of on every frame (250 m at 0.25 m: about 18 ms per
  dashboard frame instead of 60 ms; a 40 m map still updates every frame).

## License note: Ultralytics is AGPL-3.0

The detector uses [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and the YOLO26 weights
(AGPL-3.0, see `~/isaac_ws/assets/models/yolo26n.pt.provenance.txt`). Models fine-tuned from them are under the
same terms. That is fine for this private, local project. If you ever distribute this software, or offer it
as a network service, the AGPL obligations apply (release the source under the AGPL). Closed or commercial
use needs an Ultralytics Enterprise License. Everything except `drone_gcs/detector.py`'s YOLO path and
`scripts/train_detector.py` runs without Ultralytics (`--detector blob|none`).

## Layout

```
drone_gcs/        protocol.py link.py geometry.py isaac_bridge.py   <- sender side: stdlib + numpy only
                  mapping.py state.py detector.py dashboard.py text.py report.py app.py dataset.py
                  sim/fake_world.py sim/flight.py                   <- stand-in simulator
scripts/          gcs.py  fake_drone.py  seg_to_yolo.py  train_detector.py
tests/            unittest suite (./run_tests.sh)
PROTOCOL.md       message types, fields, units, frames, Isaac Lab conversions
models/           fine-tuned detector goes here (Git LFS)
docs/images/      README images (Git LFS)
```
