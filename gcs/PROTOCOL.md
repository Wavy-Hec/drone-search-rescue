# drone_gcs link protocol (v1)

Simulator (Isaac Sim / Isaac Lab survey, or `scripts/fake_drone.py`) → GCS (`scripts/gcs.py`).
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
* `seq` counts up per sender connection. `wall_time` is the sender's `time.time()`.

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

`drone_gcs.isaac_bridge.IsaacLabStreamer` does all of the above. The test
`tests/test_geometry.py::test_isaac_world_convention_to_ros_matches_isaaclab_formula` checks the constant `C`
against Isaac Lab's own formula in `isaaclab.utils.math.convert_camera_frame_orientation_convention`.
If several environments share one stage, subtract the environment origin
(`env.scene.env_origins[i]`) from all positions; `IsaacLabStreamer(origin=...)` can do that.

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
Announces the survey and sets up the GCS map grid. `Sender` keeps the latest MISSION and re-sends it after
every reconnect, so the GCS can be started or restarted at any time. A MISSION with a **different**
`mission_id` makes the GCS archive the current mission (`runs/<name>/mission<k>_<name>/`) and start a new map.

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
| `padding` | m | map margin around `area` (default max(4 m, 15 % of the larger side)). Cell centres sit on `x_min − padding + k·resolution`, so a ray grid of the same spacing on round coordinates hits cell centres, not cell edges |
| `photo_resolution` | m | orthomosaic cell size (default `resolution / 2`) |
| `geo_origin` | `{lat, lon, alt}` | optional geodetic anchor of the world origin (carried through only) |
| others | | e.g. `pattern`, `spacing`, `speed`, kept in the report |

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
| `depth` (buffer) | float32 (H, W) m | distance to the image plane. Invalid = 0, negative, NaN or inf. Without depth, the GCS intersects pixel rays with its elevation map |
| **`K`** | 3×3 (row-major nested list) px | intrinsic matrix |
| **`pos`** | [x, y, z] m, world | optical centre |
| **`quat`** | [w, x, y, z] | **ROS optical** camera frame → world |
| `width`, `height` | px | image size (default: from `rgb`) |
| `pixel_center` | 0.5 or 0 | see Pixels above (default 0.5) |
| `convention` | `"ros_optical"` | written by `Sender` for clarity |
| `camera` | str | camera name (default `cam0`) |
| `frame_id` | int | frame counter |

### GT_OBJECTS (latched, evaluation only)
Ground truth for the report's precision / recall / localization error. The GCS never uses it for mapping or
detection. The map shows it only when you press `g`.

| field | type / unit | meaning |
|---|---|---|
| **`objects`** | list of objects | `{"id", "class", "pos": [x, y, z] m world, "size"?: [sx, sy, h], "yaw"?}`. `pos` = footprint centre at ground level |

Detections are matched to ground truth by class and **horizontal** distance ≤ 2 m, greedily from the
nearest pair. The report gives horizontal error (primary) and z error separately: the GCS geolocates the
visible top surface of an object, while GT `pos` is at the ground.

### END
End of the mission. A headless GCS (or one started with `--exit-on-end`) writes its report and exits about
`--linger` seconds later. `Sender.send_end()` waits up to `flush_timeout` (default 2 s) for the queue to drain.

| field | type | meaning |
|---|---|---|
| **`reason`** | str | `complete`, `aborted`, `timeout`, ... |
| `t` | s | simulation time |

## Delivery guarantees

* `send_*` validates its input, copies the arrays (the simulator may reuse its buffers), queues them and
  returns at once. Encoding (including JPEG) and socket writes happen in a background daemon thread. On bad
  input, a full queue or no GCS, it returns `False` or drops data. It never raises into the simulation loop and
  uses `MSG_NOSIGNAL`, so no SIGPIPE.
* Queue limits per type, with the oldest dropped first: CAMERA 2, SCAN 256, TELEMETRY 512, others 256, plus a 256 MB byte
  budget that drops CAMERA first. HELLO, MISSION, GT_OBJECTS and END are never dropped.
* If the GCS stops reading, a write blocks the background thread for at most `send_timeout` (5 s). Then the
  connection is reset and re-established. Meanwhile the queue keeps dropping old frames.
* Order is preserved within the stream, except that a newly set MISSION or GT_OBJECTS jumps ahead of queued data.
* The Receiver keeps the newest 3 CAMERA frames if the GCS loop falls behind (`dropped` in the HUD link cell).
