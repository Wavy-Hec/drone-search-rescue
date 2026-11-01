#!/usr/bin/env python3
"""drone_gcs ground-control station: live map + detection dashboard for a simulated drone survey.

Run it in the conda env ``vision``, then start the simulator (Isaac Sim survey or scripts/fake_drone.py).
Keys in the window: m layer (elevation / photo / coverage), e elevation statistic (mean / max),
g show ground truth, c next drone camera, s screenshot, q / Esc quit.

Examples::

    python scripts/gcs.py                                  # window + mp4, YOLO (COCO nano weights) on the CPU
    python scripts/gcs.py --detector blob --name demo      # colour-blob stand-in for the fake world
    python scripts/gcs.py --headless --detector blob       # record only, exit after END
    python scripts/gcs.py --addr tcp://127.0.0.1:47800     # TCP instead of the default UNIX socket
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_gcs.app import AppConfig, GCSApp, parse_size  # noqa: E402
from drone_gcs.link import default_address  # noqa: E402
from drone_gcs.state import GCSConfig  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--addr", default=None, help=f"link address (default {default_address()}; env DRONE_GCS_ADDR)")
    p.add_argument("--name", default=None, help="run name under runs/ (default: timestamp)")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs"), help="parent directory of the run folders")
    p.add_argument("--headless", action="store_true", help="no window: record the dashboard to mp4 only")
    p.add_argument("--fps", type=float, default=12.0, help="dashboard refresh / video frame rate (default 12)")
    p.add_argument("--size", type=parse_size, default=(1600, 900), help="canvas size WxH (default 1600x900)")
    p.add_argument("--detector", choices=("yolo", "blob", "none"), default="yolo", help="detector (default yolo)")
    p.add_argument("--weights", default=None, help="YOLO weights (default ~/isaac_ws/assets/models/yolo26n.pt)")
    p.add_argument(
        "--device", default="cpu",
        help="YOLO device: cpu (default: the GPU belongs to the sim and the RL jobs) | cuda:0 | "
             "auto (cuda:0 only if nvidia-smi shows >= 2 GiB free)",
    )  # fmt: skip
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    p.add_argument("--classes", default=None, help="comma-separated class names to keep (YOLO)")
    p.add_argument("--class-map", default=None, help="rename/drop classes, e.g. 'truck=car,bench='")
    p.add_argument("--layer", choices=("elevation", "photo", "coverage"), default="elevation", help="initial map layer")
    p.add_argument("--show-gt", action="store_true", help="show ground-truth objects on the map (key g)")
    p.add_argument("--photo-mode", choices=("nadir", "latest", "blend"), default="nadir", help="orthomosaic cell update rule")
    p.add_argument("--depth-elevation", action="store_true", help="also add camera depth points to the elevation map")
    p.add_argument("--merge-radius", type=float, default=2.0, help="tracker merge radius [m]")
    p.add_argument("--min-hits", type=int, default=2, help="detections needed to confirm an object")
    p.add_argument("--det-ref", choices=("auto", "center", "bottom"), default="auto", help="geolocation reference pixel")
    p.add_argument("--edge-margin", type=float, default=3.0, help="boxes within N px of the image border are not mapped (0 = map all)")
    p.add_argument(
        "--depth-min", type=float, default=0.5, help="ignore depth closer than this (the drone's own legs/props) [m]"
    )
    exit_g = p.add_mutually_exclusive_group()
    exit_g.add_argument("--exit-on-end", dest="exit_on_end", action="store_true", default=None, help="quit after END (default when headless)")
    exit_g.add_argument("--stay", dest="exit_on_end", action="store_false", help="keep running after END")
    p.add_argument("--linger", type=float, default=2.0, help="seconds to keep showing/recording after END")
    p.add_argument("--max-time", type=float, default=None, help="quit after this many wall-clock seconds")
    p.add_argument("--no-video", action="store_true", help="do not record dashboard.mp4")
    p.add_argument("--no-transcode", action="store_true", help="keep OpenCV's mp4v video (skip the H.264 re-encode)")
    p.add_argument("--screenshot-at", default="", help="comma-separated wall seconds for automatic canvas screenshots")
    p.add_argument("--auto-layer-cycle", type=float, default=None, help="demo: switch map layer every N seconds")
    p.add_argument("--window-pos", type=int, nargs=2, default=None, metavar=("X", "Y"), help="window position")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = AppConfig(
        addr=a.addr,
        name=a.name,
        out_dir=Path(a.out_dir),
        headless=a.headless,
        fps=a.fps,
        size=a.size,
        detector=a.detector,
        weights=a.weights,
        device=a.device,
        conf=a.conf,
        iou=a.iou,
        imgsz=a.imgsz,
        classes=a.classes,
        class_map=a.class_map,
        layer=a.layer,
        show_gt=a.show_gt,
        exit_on_end=a.exit_on_end,
        linger=a.linger,
        max_time=a.max_time,
        video=not a.no_video,
        transcode=not a.no_transcode,
        screenshot_at=[float(s) for s in a.screenshot_at.split(",") if s.strip()],
        auto_layer_cycle=a.auto_layer_cycle,
        window_pos=tuple(a.window_pos) if a.window_pos else None,
        gcs=GCSConfig(
            photo_mode=a.photo_mode,
            depth_elevation=a.depth_elevation,
            merge_radius=a.merge_radius,
            min_hits=a.min_hits,
            det_ref=a.det_ref,
            edge_margin=a.edge_margin,
            depth_min_range=a.depth_min,
        ),
    )
    return GCSApp(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
