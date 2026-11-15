#!/usr/bin/env python3
"""Score the detector on labelled real aerial images, for the people only.

The detector was trained and tested on simulator images. This script scores it on any labelled set of real aerial
images in YOLO format, e.g. a public drone dataset, so the gap between simulator and reality can be measured for
the class that matters. For every image of ``--images``:

1. The labels are read from the file with the same stem in ``--labels`` (lines ``<class id> <cx> <cy> <w> <h>``,
   normalised to the image). Only the class ids of ``--person-ids`` are kept, every other label is ignored: a
   dataset may have more than one class of people (VisDrone as Ultralytics converts it: 0 pedestrian,
   1 people). An image without a label file counts as having no people.
2. The image is resized by ``--scale``, so that people are about as large as in the mission camera (640 x 480 px,
   1.54 cm of ground per pixel: a lying person is about 110 px long and 28 px across the shoulders). Choose
   ``--scale`` as the dataset's ground resolution in cm per pixel divided by 1.54, e.g. 3.08 / 1.54 = 2 for
   images at 3.08 cm per pixel.
3. The resized image is cut into 640 x 480 tiles that overlap by ``--overlap``; the detector runs on every tile
   and a box that two tiles found twice is kept once. With ``--no-tile`` the whole image goes to the detector,
   which resizes it to 640 px itself.
4. The person boxes are matched to the labels, one to one, the most confident box first; a label is found by a
   box that overlaps it with an IoU of at least ``--iou``. A box without a label is a false alarm.

Writes ``external_eval.json`` and ``external_eval.md`` into ``--out`` (default: a time-stamped folder under
``runs/``) and prints the Markdown: images, people labelled, found, missed, false alarms, precision and recall,
and the recall per size of the label (the longer side of its box, in pixels of the resized image).

Example (conda env 'vision')::

    python scripts/eval_external.py --images ~/data/aerial/images/val --labels ~/data/aerial/labels/val \\
        --person-ids 0,1 --weights models/drone_sar_yolo26n.pt --scale 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_gcs.detector import Detector, YoloDetector  # noqa: E402
from drone_gcs.external_eval import MISSION_TILE, PERSON, evaluate_people, read_yolo_labels  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def yolo_detector(a: argparse.Namespace) -> YoloDetector:
    """The ground station's detector with the weights, device and confidence of the command line; it reports
    people only."""
    return YoloDetector(Path(a.weights).expanduser(), device=a.device, conf=a.conf, classes=PERSON)


def load_items(images: list[Path], labels_dir: Path, person_ids: set[int]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """``(rgb, boxes)`` of every image, read one at a time: the RGB image (H, W, 3) and the boxes (N, 4)
    ``(x1, y1, x2, y2)`` [px] of its people. Prints the progress every 50 images."""
    for k, path in enumerate(images, 1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"cannot read {path}")
        height, width = bgr.shape[:2]
        boxes = read_yolo_labels(labels_dir / f"{path.stem}.txt", width, height, person_ids)
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), boxes  # the station works in RGB
        if k % 50 == 0:
            print(f"[{k}/{len(images)}] images", flush=True)


def markdown(report: dict) -> str:
    """The report as Markdown: the settings, the counts with precision and recall, and the recall per label size."""

    def ratio(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    if report["tile"] is None:
        cut = "given whole to the detector"
    else:
        cut = f"cut into {report['tile'][0]} x {report['tile'][1]} px tiles that overlap by {report['overlap']}"
    counts = [report[key] for key in ("images", "people_labelled", "found", "missed", "false_alarms")]
    lines = [
        "# People on external images",
        "",
        f"Images: {report['images_dir']}. Labels: {report['labels_dir']}, class ids "
        f"{', '.join(map(str, report['person_ids']))}. Detector: {report['detector']}.",
        f"Every image is resized by {report['scale']}, then {cut}. A label is found when a person box overlaps "
        f"it with an IoU of at least {report['match_iou']}.",
        "",
        "| Images | People labelled | Found | Missed | False alarms | Precision | Recall |",
        "|---|---|---|---|---|---|---|",
        f"| {' | '.join(map(str, counts))} | {ratio(report['precision'])} | {ratio(report['recall'])} |",
        "",
        "| Longer side of the label [px] | People labelled | Found | Recall |",
        "|---|---|---|---|",
    ]
    for name, size in report["by_size"].items():
        lines.append(f"| {name} | {size['labelled']} | {size['found']} | {ratio(size['recall'])} |")
    lines += ["", "The size of a label is measured in the resized image."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, make_detector: Callable[[argparse.Namespace], Detector] = yolo_detector) -> int:
    """Run the evaluation. ``make_detector`` builds the detector from the parsed arguments (tests pass their own)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, metavar="DIR", help="folder of the images (.png, .jpg, .jpeg)")
    p.add_argument("--labels", required=True, metavar="DIR", help="folder of the YOLO label files (<image stem>.txt)")
    p.add_argument("--person-ids", required=True, metavar="IDS", help="class ids of the people in the labels, e.g. 0,1")
    p.add_argument("--weights", required=True, help="detector weights, e.g. models/drone_sar_yolo26n.pt")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize every image by this factor first, so that people are as large as in the mission camera: "
        "the dataset's ground resolution in cm per pixel divided by the mission's 1.54 (default 1)",
    )
    p.add_argument("--no-tile", action="store_true", help="give the whole image to the detector, which resizes it")
    p.add_argument("--overlap", type=float, default=0.2, help="share of a tile that neighbours overlap (default 0.2)")
    p.add_argument("--iou", type=float, default=0.45, help="a box finds a label at this IoU or more (default 0.45)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold of the detector (default 0.25)")
    p.add_argument("--device", default="cpu", help="'cpu' or a GPU index such as '0'")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="only the first N images, by file name")
    p.add_argument("--out", default=None, metavar="DIR", help="output folder (default: runs/<time>_external_eval)")
    a = p.parse_args(argv)
    images_dir, labels_dir = Path(a.images).expanduser(), Path(a.labels).expanduser()
    for flag, folder in (("--images", images_dir), ("--labels", labels_dir)):
        if not folder.is_dir():
            p.error(f"{flag}: no such folder: {folder}")
    try:
        person_ids = {int(s) for s in a.person_ids.split(",") if s.strip()}
    except ValueError:
        p.error("--person-ids must be whole numbers, e.g. 0,1")
    if not person_ids:
        p.error("--person-ids needs at least one class id")
    if a.scale <= 0:
        p.error("--scale must be above 0")
    if not 0 <= a.overlap < 1:
        p.error("--overlap must be at least 0 and below 1")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be at least 1")
    images = sorted(q for q in images_dir.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES)[: a.limit]
    if not images:
        p.error(f"--images: no image in {images_dir}")
    if not any((labels_dir / f"{q.stem}.txt").is_file() for q in images):  # most likely the wrong folder
        p.error(f"--labels: no label file in {labels_dir} has the name of an image")

    detector = make_detector(a)
    tile = None if a.no_tile else MISSION_TILE
    result = evaluate_people(detector, load_items(images, labels_dir, person_ids), a.scale, tile, a.overlap, a.iou)
    report = {
        "created": f"{datetime.now():%Y-%m-%dT%H:%M:%S}",
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "person_ids": sorted(person_ids),
        "weights": str(a.weights),
        "detector": detector.describe(),
        "conf": a.conf,
        "scale": a.scale,
        "tile": None if tile is None else list(tile),
        "overlap": None if tile is None else a.overlap,
        "match_iou": a.iou,
        **result,
    }
    out_dir = (
        Path(a.out).expanduser() if a.out else PROJECT_ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_external_eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "external_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    text = markdown(report)
    (out_dir / "external_eval.md").write_text(text)
    print(text)
    print(f"wrote {out_dir}/external_eval.md and external_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
