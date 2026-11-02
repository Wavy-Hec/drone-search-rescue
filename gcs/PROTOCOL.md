# drone_gcs link protocol (v1)

Simulator (`drone_lab/scripts/sar_mission.py` in Isaac Sim, or `scripts/fake_drone.py`) → GCS (`scripts/gcs.py`).
One direction only: the simulator sends, the GCS listens. Implementation: `drone_gcs/protocol.py`
(framing) and `drone_gcs/link.py` (`Sender`, `Receiver`); both need only the Python standard library and numpy.

## Transport

| | |
|---|---|
| Default | UNIX domain stream socket `unix:$XDG_RUNTIME_DIR/drone_gcs.sock` (here `/run/user/1000/drone_gcs.sock`); fallback `/tmp/drone_gcs-<uid>.sock` |
| TCP | `tcp://127.0.0.1:47800` (or `host:port`, `:port`). Use it when the simulator runs in a container or on another machine |
| Override | environment variable `DRONE_GCS_ADDR`, or `--addr` / `Sender(address=...)` |
| Roles | the GCS binds and listens and accepts several senders. The simulator connects and reconnects automatically with backoff (0.2 s up to 2 s) |

## Frame format

Every message is one length-prefixed frame:

```
offset  size         field
0       4            magic  b"DGC1"
4       4            header_len   uint32 little-endian
8       8            payload_len  uint64 little-endian
16      header_len   header: UTF-8 JSON object
16+H    payload_len  payload: raw buffers, each starting at its own 8-byte-aligned offset
```

Header:

```json
{"proto": "drone_gcs", "v": 1, "type": "CAMERA", "seq": 812, "drone_id": 0, "sender": "isaac-survey",
 "wall_time": 1793541234.52,
 "fields": {"t": 81.2, "K": [[...]], "pos": [...], "quat": [...]},
 "buffers": [{"name": "rgb",   "dtype": "|u1", "shape": [480, 640, 3], "offset": 0,      "nbytes": 921600,  "encoding": "raw"},
             {"name": "depth", "dtype": "<f4", "shape": [480, 640],    "offset": 921600, "nbytes": 1228800, "encoding": "raw"}]}
```

* `dtype` is numpy's explicit array-protocol string (`<f4`, `<f8`, `|u1`, `|b1`, `<i4`, ...). Raw buffers are the
  little-endian bytes of a C-contiguous array, so either end can run any numpy version. Nothing is pickled.
  Allowed kinds are bool, int, uint and float.
* `encoding` is `"raw"` or `"jpeg"`. For JPEG, `shape` is the decoded shape and `nbytes` is the compressed size.
* `fields` holds plain JSON. Non-finite floats are sent as `null`. Arrays bigger than a few numbers go in
  buffers. Unknown extra fields are allowed and ignored by the GCS.
* `seq` counts up per `Sender` object (it is not reset on reconnect, so gaps show lost messages). `wall_time`
  is the sender's `time.time()`.

## Common conventions

| Item | Convention |
|---|---|
| Units | metres, seconds, radians (heading displays use degrees) |
| `t` | simulation time [s] since the simulation started (float64). It is not wall time |
| World frame | Isaac Sim stage/world frame: right-handed, **Z up**. The GCS treats it as a local ENU tangent plane: **+X = east, +Y = north**, so maps are drawn north-up with +Y up. No geodetic coordinates. A MISSION may carry `geo_origin` for later use |
| Quaternions | **(w, x, y, z)**, Hamilton, unit norm. `q` rotates frame-local vectors into the world: `v_world = R(q) · v_local` |
| Body frame | FLU (x forward, y left, z up), as in Isaac Lab and ROS REP-103 |
| Camera frame | **ROS optical**: x right, y down, z forward (into the scene). `quat` = camera-optical → world |
| Pixels | continuous image coordinates. Pixel index `(u, v)` covers `[u, u+1) × [v, v+1)`, so its centre is `(u+0.5, v+0.5)`. Isaac Lab's K has `cx = W/2, cy = H/2` (`pixel_center = 0.5`, the default). Send `pixel_center = 0` for an OpenCV-style K whose pixel centres are integers |
| Images | `rgb` is uint8 (H, W, 3) in **RGB** order (not BGR). Row 0 is the top of the image |
| Heading (HUD) | compass degrees: 0 = north (+Y), 90 = east (+X), clockwise. `heading = (90° − yaw) mod 360`, where yaw is CCW from +X |

### Converting from Isaac Lab 3.0

Isaac Lab 3.0 stores quaternions as **(x, y, z, w)** (warp `quatf`). Reorder them before sending:
`q_wxyz = q_xyzw[[3, 0, 1, 2]]` (`drone_gcs.geometry.quat_xyzw_to_wxyz`).

| Isaac Lab quantity | Send as |
|---|---|
| `robot.data.root_pos_w[i]`, `root_lin_vel_w[i]` | TELEMETRY `pos`, `vel` |
| `robot.data.root_quat_w[i]` (x,y,z,w) | TELEMETRY `quat` = reordered to (w,x,y,z) |
| `ray_caster.data.ray_hits_w[i]` (B,3), `inf` = miss | SCAN `points` (misses may stay `inf`) |
| `camera.data.pos_w[i]` | CAMERA `pos` |
| `camera.data.quat_w_ros[i]` (x,y,z,w) | CAMERA `quat` = reordered to (w,x,y,z). It is already the ROS optical convention |
| `camera.data.quat_w_world[i]` (x,y,z,w) | reorder, then `q_ros = q_world ⊗ (0.5, −0.5, 0.5, −0.5)` (w,x,y,z), i.e. `R_ros = R_world · C` with `C = [[0,0,1],[−1,0,0],[0,−1,0]]` (columns: x_right = −y, y_down = −z, z_fwd = +x of the "world" camera convention, which has x forward and z up). `drone_gcs.geometry.camera_quat_to_ros(q, "world")` does this |
| `camera.data.quat_w_opengl[i]` (x,y,z,w) | reorder, then `q_ros = q_gl ⊗ (0, 1, 0, 0)` (180° about x). `camera_quat_to_ros(q, "opengl")` does this |
| `camera.data.intrinsic_matrices[i]` | CAMERA `K` (cx = W/2, so keep `pixel_center = 0.5`) |
| `camera.data.output["rgb"][i]` (H,W,3) uint8 | CAMERA `rgb` (an RGBA input has its alpha channel dropped) |
| `camera.data.output["distance_to_image_plane"][i]` (H,W,1) | CAMERA `depth` (H,W) float32. This is z-depth along the optical axis, **not** the ray length (`distance_to_camera`) |

`drone_gcs.isaac_bridge.IsaacLabStreamer` does all of the above. Two tests in `tests/test_geometry.py` check
the conversions without importing Isaac Lab: `test_matches_isaaclab_3_warp_constants` against the fixed
constants of Isaac Lab 3.0 (`isaaclab/utils/warp/warp_math.py`, applied as `q_out = q_in * q_const`, xyzw:
world → ros `(−0.5, 0.5, −0.5, 0.5)`, opengl → ros `(1, 0, 0, 0)`), and
`test_isaac_world_convention_to_ros_matches_isaaclab_formula` against a re-implementation of the matrix
formula of `isaaclab.utils.math.convert_camera_frame_orientation_convention`.

`IsaacLabStreamer.camera()` sends each capture once: it reads Isaac Lab's capture counter `camera.frame` and
skips calls where it has not advanced (a `CameraCfg.update_period` longer than the env step would otherwise
repeat the same image, and the GCS would count every copy as a new sighting). It sends that counter as
`frame_id`. With the OVRTX renderer's `async_rendering` the image is one capture old; the streamer then
takes the pose and intrinsics from `camera.data.info["rgb"]["capture"]` (`pos_w`, `quat_w_world` converted
to ROS optical, `intrinsic_matrices`, `frame`), which match the pixels, instead of the live `pos_w`.

If several environments share one stage, subtract the environment origin (`env.scene.env_origins[i]`)
from all positions. `IsaacLabStreamer(origin=...)` subtracts it from everything it reads from Isaac Lab.
Positions you pass yourself (`mission()` area, planned path and altitudes, `gt_objects()`) are taken as
local, or as world coordinates with `world=True`.

## Messages

Required fields are **bold**. Message types not in this table are ignored by the GCS.

### HELLO
Sent automatically by `Sender` as the first frame of every connection.

| field | type | meaning |
|---|---|---|
| `name` | str | sender name |
| `protocol` | int | protocol version (1) |
| `pid`, `host`, `python`, `numpy` | | sender process info |
| `started` | float | sender start time (unix seconds) |

### MISSION (latched)
Announces the survey and sets up the GCS map grid. It is delivered in call order like everything else, and
`Sender` re-sends it after every reconnect (see Delivery), so the GCS can be started or restarted at any
time. A MISSION with a **different** `mission_id` makes the GCS archive the current mission
(`runs/<name>/mission<k>_<name>/`) and start a new map. Ground truth (GT_OBJECTS) stays in force for the new
mission until the simulator sends a new one.

Several drones share one mission: send one `mission_id` for all of them (a second MISSION with the same id
is ignored; a different id starts a new mission). END from any drone ends the mission for the GCS.

| field | type / unit | meaning |
|---|---|---|
| **`area`** | `{x_min, x_max, y_min, y_max}` m, world | survey rectangle. Coverage % is computed inside it |
| **`resolution`** | m | elevation-map cell size, e.g. 0.25 |
| **`mission_id`** | str | unique per mission (Sender generates one if not given) |
| `name` | str | display name |
| `planned_path` (buffer) | float64 (N, 3) m, world | planned waypoints / path polyline. `wp_index` refers to these rows |
| `survey_altitude` | m, world z | planned survey height (display only) |
| `ground_z` | m | nominal ground height, used before the map knows the ground (AGL, rays without depth) |
| `z_range` | [zmin, zmax] m | fixed colour scale for the elevation layer. If absent, the GCS uses a robust automatic range |
| `padding` | m | map margin around `area` (default max(4 m, 15 % of the larger side)), rounded up to whole cells. Cell centres sit on `x_min + k·resolution` (same for y), so a ray grid of the same spacing hits cell centres, not cell edges, when the area bounds lie on that lattice (e.g. multiples of the resolution). Grids above 4 M cells get a coarser resolution (a whole multiple of the requested one) |
| `photo_resolution` | m | orthomosaic cell size (default `resolution / 2`) |
| `geo_origin` | `{lat, lon, alt}` | optional geodetic anchor of the world origin (carried through only) |
| `home` | [x, y, z] m, world | landing pad (top surface); drawn as "H" on the map |
| others | | e.g. `pattern`, `spacing`, `speed`, `phases`, kept in the report |

### Mission scope of data messages
`Sender` adds the `mission_id` of the last MISSION it was given to every TELEMETRY, SCAN, CAMERA and END
(field `mission_id`, unless the caller set one). The GCS ignores data and END whose `mission_id` differs from
its current mission (counted in `summary.json` → `messages.ignored_other_mission`), so a late message of an
earlier mission can neither land on the new map nor end the new mission. Messages without `mission_id` are
accepted.

### TELEMETRY
Vehicle state, typically 10 to 50 Hz.

| field | type / unit | meaning |
|---|---|---|
| **`t`** | s | simulation time |
| **`pos`** | [x, y, z] m, world | vehicle (body) position |
| **`quat`** | [w, x, y, z] | body (FLU) → world orientation |
| `vel` | [vx, vy, vz] m/s, world | linear velocity |
| `ang_vel` | [wx, wy, wz] rad/s | angular velocity (as given by the sim) |
| `wp_index` | int | index into `planned_path` of the waypoint currently targeted |
| `carrot` | [x, y, z] m, world | current tracking target (carrot) |
| `phase` | str | mission phase, e.g. `takeoff`, `transit`, `survey`, `return`, `descent`, `touchdown` (shown in the HUD); `landed` once the drone sits on the ground with its motors off (the camera panel then shows a LANDED card) |

### SCAN
Mapping ray-caster hits, typically 5 to 20 Hz. Each scan updates the elevation map: per-cell count, sum,
sum of squares, min, max. `count == 0` is the unknown (fog-of-war) state.

| field | type / unit | meaning |
|---|---|---|
| **`t`** | s | simulation time |
| **`points`** (buffer) | float32 (N, 3) m, world | hit points. `inf`/`NaN` = no hit |
| `valid` (buffer) | bool (N,) | optional mask, combined with the finiteness test |
| `origin` | [x, y, z] m, world | sensor origin |

### CAMERA
Camera frame, typically 2 to 10 Hz. Each frame updates the orthomosaic and, if a detector runs, is sent to it.
The GCS keeps only the newest frames when it falls behind, as does the Sender (2 queued by default).

| field | type / unit | meaning |
|---|---|---|
| **`t`** | s | simulation time of the exposure |
| **`rgb`** (buffer) | uint8 (H, W, 3) RGB | image. `encoding: "jpeg"` if `send_camera(..., jpeg=True)` (needs cv2 or PIL on the sender, else raw) |
| `depth` (buffer) | float32 (H, W) m | distance to the image plane. Invalid = 0, negative, NaN or inf. Depth under 0.5 m (`gcs.py --depth-min`) is taken to be the vehicle itself (landing gear, propellers) and ignored. Without depth, the GCS intersects pixel rays with its elevation map |
| **`K`** | 3×3 (row-major nested list) px | intrinsic matrix |
| **`pos`** | [x, y, z] m, world | optical centre |
| **`quat`** | [w, x, y, z] | **ROS optical** camera frame → world |
| `width`, `height` | px | image size (default: from `rgb`) |
| `pixel_center` | 0.5 or 0 | see Pixels above (default 0.5) |
| `convention` | `"ros_optical"` | written by `Sender` for clarity |
| `camera` | str | camera name (default `cam0`) |
| `frame_id` | int | frame counter (`IsaacLabStreamer`: Isaac Lab's capture index). Send each capture once: repeats count as extra sightings |
| `gt_boxes` | list | evaluation only: the labelled objects in this frame from the simulator's instance segmentation, `[{"class", "box": [x1, y1, x2, y2], "name"?, "pixels"?}]` (continuous pixel coordinates, like detections). `gcs.py --detector gt` maps these instead of running a model, which checks the whole geometry chain with perfect boxes |

### GT_OBJECTS (latched, evaluation only)
Ground truth for the report's precision / recall / localization error. The GCS never uses it for mapping or
detection. The map shows it only when you press `g`.

| field | type / unit | meaning |
|---|---|---|
| **`objects`** | list of objects | `{"id", "class", "pos": [x, y, z] m world, "size"?: [sx, sy, h], "yaw"?}`. `pos` = footprint centre at ground level |

Send it after the MISSION it belongs to. It stays in force (also across a mission change) until the next
GT_OBJECTS; an empty list clears it.

Detections are matched to ground truth by class and **horizontal** distance ≤ 2 m, greedily from the
nearest pair. The report gives horizontal error (primary) and z error separately: the GCS geolocates the
visible top surface of an object, while GT `pos` is at the ground.

### GT_TERRAIN (latched, evaluation only)
The true terrain height, for the report's map height error. The GCS never uses it for mapping.

| field | type / unit | meaning |
|---|---|---|
| **`heights`** (buffer) | float32 (ny, nx) m, world z | true height per cell, row 0 = south (y_min); NaN = unknown |
| **`x_min`**, **`y_min`** | m, world | lower edges of cell (0, 0); cell `(iy, ix)` spans `[x_min + ix·res, x_min + (ix+1)·res)` |
| **`resolution`** | m | cell size |

The report samples it at the centres of its own map cells (nearest cell), so a raster whose cell centres sit on
the same lattice as the map (e.g. multiples of 0.25 m) compares like with like. Like GT_OBJECTS it stays in force
across a mission change until the next GT_TERRAIN.

### END
End of the mission. A headless GCS (or one started with `--exit-on-end`) writes its report and exits about
`--linger` seconds later. `Sender.send_end()` waits up to `flush_timeout` (default 2 s) for the queue to drain.

| field | type | meaning |
|---|---|---|
| **`reason`** | str | `complete`, `landed`, `crashed`, `aborted`, `timeout`, ... |
| `t` | s | simulation time |
| `stats` | dict | the drone's own mission statistics, kept in the report. `stats.closest_approach.closest_m` (closest distance to terrain or objects away from the pad [m]) goes into the scorecard |

## Delivery guarantees

* `send_*` validates its input, copies the arrays and converts the fields (extra keyword fields included) to
  plain JSON values before it returns, so the simulator may reuse its buffers at once. Encoding (including
  JPEG) and socket writes happen in a background daemon thread. On bad input, a full queue or no GCS, it
  returns `False` or drops data. It never raises into the simulation loop and uses `MSG_NOSIGNAL`, so no SIGPIPE.
* Queue limits per type, with the oldest dropped first: CAMERA 2, SCAN 256, TELEMETRY 512, others 256, plus a 256 MB byte
  budget that drops CAMERA first. HELLO, MISSION, GT_OBJECTS, GT_TERRAIN and END are never dropped.
* Order: messages are written in the order of the `send_*` calls, MISSION and GT_OBJECTS included (only
  dropped data is missing). A mission change while the GCS is away or behind therefore arrives as
  MISSION A, A's data, END A, MISSION B, B's data.
* After every (re)connect the Sender writes HELLO, then the MISSION / GT_OBJECTS / GT_TERRAIN in force when the next
  queued message was sent (not necessarily the newest ones), then the queue. A GCS restarted mid-mission
  therefore starts a **new, empty map** for that mission: earlier data is not replayed, apart from what was
  still queued (sent while no GCS was listening, within the queue limits). The message being written when
  the link broke is lost if it is data; MISSION, GT_OBJECTS, GT_TERRAIN and END are retried.
* If the GCS stops reading, the background thread blocks for at most `send_timeout` (5 s) per write call
  (one frame takes 3 or more calls: prefix, header, each buffer), so up to about 3 × 5 s per frame. Then the
  connection is reset and re-established. Meanwhile the queue keeps dropping old frames.
* `flush()` (and `send_end(flush_timeout=...)`) return once everything queued, including the message being
  written, has gone out, or after the timeout.
* The Receiver keeps the newest 3 CAMERA frames if the GCS loop falls behind (`dropped` in the HUD link cell).
