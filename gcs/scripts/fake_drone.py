#!/usr/bin/env python3
"""Fake drone: a stand-in simulator that flies a lawnmower survey over a synthetic world and streams
MISSION / GT_OBJECTS / TELEMETRY / SCAN / CAMERA / END through the real link (drone_gcs.link.Sender).

Synthetic world: hills, buildings, trees, a dirt road and coloured objects (car = red, tent = blue,
person = magenta, tarp = yellow) with known positions. Sensors: a downward grid ray-caster (like Isaac Lab's
RayCaster + GridPatternCfg, 5 x 5 m at 0.25 m) and a nadir RGB camera with depth (ray-marched height field).

Examples (conda env 'vision', start scripts/gcs.py first or after - the link reconnects)::

    python scripts/fake_drone.py                       # real time (~3 min survey of 40 x 40 m)
    python scripts/fake_drone.py --speedup 8           # 8x faster than real time
    python scripts/fake_drone.py --no-depth --jpeg     # GCS must intersect rays with its elevation map
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_gcs.geometry import C_BODY_TO_NADIR_CAM, PinholeCamera, intrinsics_from_fov, quat_to_matrix  # noqa: E402
from drone_gcs.link import Sender, default_address  # noqa: E402
from drone_gcs.sim.fake_world import FakeWorld  # noqa: E402
from drone_gcs.sim.flight import CarrotFlight, lawnmower  # noqa: E402


def camera_pose(
    flight: CarrotFlight, mount: str, offset: tuple[float, float, float] = (0.05, 0.0, -0.08)
) -> tuple[np.ndarray, np.ndarray]:
    """(R_wc ROS optical -> world, camera position) for a nadir camera on a gimbal or fixed to the body."""
    R_wb = quat_to_matrix(flight.quat)
    pos = flight.p + R_wb @ np.asarray(offset)
    if mount == "gimbal":  # stabilised: only yaw follows the body
        c, s = math.cos(flight.yaw), math.sin(flight.yaw)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return R_yaw @ C_BODY_TO_NADIR_CAM, pos
    return R_wb @ C_BODY_TO_NADIR_CAM, pos


def build(
    args: argparse.Namespace,
) -> tuple[tuple[float, float, float, float], FakeWorld, float, np.ndarray, CarrotFlight]:
    """Survey area, world, flight height, lawnmower path and flight of the command-line settings."""
    if len(args.area) == 2:
        w, h = args.area
        area = (-w / 2, w / 2, -h / 2, h / 2)
    else:
        area = tuple(args.area)
    world = FakeWorld(area, seed=args.seed)
    z = args.altitude if args.altitude is not None else world.hmax + args.clearance
    path = lawnmower(area, args.spacing, z, direction=args.direction)
    flight = CarrotFlight(path, speed=args.speed)
    return area, world, z, path, flight


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--addr", default=None, help=f"GCS link address (default {default_address()})")
    p.add_argument("--name", default="fake_survey", help="mission name")
    p.add_argument("--area", type=float, nargs="+", default=[40.0, 40.0], help="W H (centred) or x0 x1 y0 y1 [m]")
    p.add_argument("--seed", type=int, default=0, help="world seed")
    p.add_argument("--resolution", type=float, default=0.25, help="map cell size announced in MISSION [m]")
    p.add_argument("--spacing", type=float, default=4.0, help="lawnmower line spacing [m]")
    p.add_argument("--direction", choices=("x", "y"), default="x", help="lawnmower line direction")
    p.add_argument("--speed", type=float, default=3.0, help="cruise speed [m/s]")
    p.add_argument("--clearance", type=float, default=4.0, help="survey height above the highest terrain point [m]")
    p.add_argument("--altitude", type=float, default=None, help="absolute survey z [m] (overrides --clearance)")
    p.add_argument("--scan-size", type=float, nargs=2, default=(5.0, 5.0), help="ray grid size [m]")
    p.add_argument("--scan-res", type=float, default=0.25, help="ray grid spacing [m]")
    p.add_argument("--scan-noise", type=float, default=0.0, help="std of ray-hit height noise [m]")
    p.add_argument("--telemetry-hz", type=float, default=20.0)
    p.add_argument("--scan-hz", type=float, default=10.0)
    p.add_argument("--camera-hz", type=float, default=5.0)
    p.add_argument("--camera-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    p.add_argument("--hfov", type=float, default=75.0, help="camera horizontal field of view [deg]")
    p.add_argument("--mount", choices=("gimbal", "fixed"), default="gimbal", help="nadir camera mount")
    p.add_argument("--no-depth", action="store_true", help="send RGB without depth")
    p.add_argument("--jpeg", action="store_true", help="JPEG-compress the RGB frames")
    p.add_argument("--no-gt", action="store_true", help="do not send GT_OBJECTS")
    p.add_argument("--speedup", type=float, default=1.0, help="sim time / wall time (default 1 = real time)")
    p.add_argument("--sim-dt", type=float, default=0.02, help="flight model step [s]")
    p.add_argument("--max-sim-time", type=float, default=1800.0)
    p.add_argument("--wait-gcs", type=float, default=15.0, help="wait up to N s for the GCS before flying (0 = no)")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if a.quiet else logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t_build = time.monotonic()
    area, world, z, path, flight = build(a)
    W, H = a.camera_size
    K = intrinsics_from_fov(W, H, a.hfov)
    say = (lambda *x: None) if a.quiet else print
    say(
        f"[fake_drone] world {world.grid.nx}x{world.grid.ny} @ {world.grid.resolution} m, "
        f"{len(world.objects)} objects, terrain {world.hmin:.1f}..{world.hmax:.1f} m, survey z {z:.1f} m, "
        f"path {flight.length:.0f} m ({len(path)} waypoints), built in {time.monotonic() - t_build:.1f} s"
    )

    sender = Sender(a.addr, name="fake_drone")
    if a.wait_gcs > 0:
        say(f"[fake_drone] waiting up to {a.wait_gcs:.0f} s for the GCS at {sender.address_str} ...")
        if not sender.wait_connected(a.wait_gcs):
            say("[fake_drone] no GCS yet - flying anyway (data is dropped until it connects)")
    sender.send_mission(
        area, a.resolution, path, z, name=a.name, ground_z=float(np.median(world.ground)),
        z_range=(world.hmin, world.hmax), pattern="lawnmower", spacing=a.spacing, speed=a.speed,
    )  # fmt: skip
    if not a.no_gt:
        sender.send_gt_objects(world.gt_objects())

    rng = np.random.default_rng(a.seed + 1)
    dt = a.sim_dt
    t = 0.0
    next_tel = next_scan = next_cam = 0.0
    frame_id = 0
    wall0 = time.monotonic()
    next_print = 10.0
    render_ms = 0.0
    aborted = False
    try:
        while not flight.done and t < a.max_sim_time:
            flight.step(dt)
            t += dt
            if t >= next_tel:
                next_tel += 1.0 / a.telemetry_hz
                carrot, _ = flight.carrot()
                sender.send_telemetry(t, flight.p, flight.quat, flight.v, wp_index=flight.wp_index, carrot=carrot)
            if t >= next_scan:
                next_scan += 1.0 / a.scan_hz
                origin = flight.p + np.array([0.0, 0.0, -0.1])
                pts, valid = world.scan_grid(origin, flight.yaw, a.scan_size, a.scan_res, a.scan_noise, rng)
                sender.send_scan(t, pts, valid, origin)
            if t >= next_cam:
                next_cam += 1.0 / a.camera_hz
                R_wc, cpos = camera_pose(flight, a.mount)
                cam = PinholeCamera(K, W, H, R_wc, cpos)
                t0 = time.perf_counter()
                rgb, depth = world.render(cam)
                render_ms = (
                    0.8 * render_ms + 0.2 * (time.perf_counter() - t0) * 1000
                    if frame_id
                    else (time.perf_counter() - t0) * 1000
                )
                sender.send_camera(
                    t, rgb, K, cpos, cam.quat, None if a.no_depth else depth, jpeg=a.jpeg, frame_id=frame_id
                )
                frame_id += 1
            if t >= next_print:
                next_print += 10.0
                s = sender.stats()
                px, py, pz = flight.p
                link = "up" if s["connected"] else "DOWN"
                say(
                    f"[fake_drone] t={t:6.1f} s  wp {flight.wp_index + 1}/{len(path)}  "
                    f"pos ({px:+6.1f}, {py:+6.1f}, {pz:4.1f})  sent {s.get('sent', 0)}  dropped {s['dropped']}  "
                    f"link {link}  render {render_ms:.0f} ms  x{t / max(1e-6, time.monotonic() - wall0):.1f}"
                )
            sleep = wall0 + t / a.speedup - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        aborted = True
    sender.send_end("aborted" if aborted else "complete", t, flush_timeout=10.0, path_length=flight.length)
    s = sender.stats()
    sender.close(flush_timeout=2.0)
    world.close()
    wall = time.monotonic() - wall0
    say(
        f"[fake_drone] {'aborted' if aborted else 'done'}: sim {t:.1f} s in {wall:.1f} s wall "
        f"(x{t / max(wall, 1e-6):.1f}), {frame_id} camera frames, "
        f"sent {s.get('sent', 0)} msgs / {s.get('sent_bytes', 0) / 1e6:.1f} MB, "
        f"dropped {s['dropped']} {s['dropped_by_type'] or ''}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
