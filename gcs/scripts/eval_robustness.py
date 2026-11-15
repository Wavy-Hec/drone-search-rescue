#!/usr/bin/env python3
"""Score the ground station's detector on one split and on corrupted copies of it.

The fine-tuned detector finds nearly every object of the simulator's frames (mAP50 0.995 on the test frames),
which leaves nothing to compare. This script writes corrupted copies of the split with
``scripts/degrade_dataset.py`` (sensor noise, motion blur, low light, JPEG, haze, occlusion, a higher flight; see
:data:`drone_gcs.corruptions.PRESETS`) into ``--out`` and runs the ground station's own detector
(:class:`drone_gcs.detector.YoloDetector`, confidence ``--conf``, its default NMS IoU and image size) on the clean
split and on every copy. The objects found, missed and the false alarms per class are counted as in
``scripts/eval_detector.py``: a label is found when a box of its class overlaps it with an IoU above 0.45, one to
one, highest confidence first.

Writes into ``--out`` (default: a time-stamped folder under ``runs/``), next to the corrupted copies:
``robustness.json`` (everything, with the counts per class), ``robustness.csv`` and ``robustness.md`` (one row per
preset: people found / labelled, people false alarms, all objects found / labelled, all false alarms), and prints
that table. Class 0 is person.

Example (conda env 'vision')::

    python scripts/eval_robustness.py --data ../datasets/drone_sar_yolo_v2/yolo/data.yaml \\
        --weights models/drone_sar_yolo26n.pt --split val --presets dark,blur,occlude30,altitude2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_gcs.corruptions import PRESETS  # noqa: E402
from drone_gcs.detector import Detector, YoloDetector  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("eval_detector")
dd = _load("degrade_dataset")

CLEAN = "clean"
"""Name of the row of the split as it is."""

COLUMNS = (
    "preset",
    "frames",
    "people_found",
    "people_labelled",
    "people_false_alarms",
    "found",
    "labelled",
    "false_alarms",
)
"""Columns of ``robustness.csv``, in order."""


def score(detector: Detector, datasets: dict[str, Path], split: str) -> list[dict]:
    """One row per ``{name: data.yaml}`` (in order): the detector's counts on ``split`` of that dataset.

    ``detector`` is anything with ``detect(rgb)``; its own threshold applies. A row holds the columns of
    :data:`COLUMNS` (class 0 is person; "found" etc. without a prefix are over all classes) and ``per_class``:
    ``{class: {"found", "missed", "false_alarms"}}``. A label of a class that is detected as another class is missed
    and the detection a false alarm.
    """
    rows = []
    for name, data in datasets.items():
        root, image_dirs, names = ev.read_names(data)
        image_dir = root / image_dirs[split]
        per_class = ev.counts(ev.station_matrix(detector, root, image_dirs[split], names).matrix, names)
        person = per_class[names[0]]
        rows.append(
            {
                "preset": name,
                "frames": sum(p.suffix.lower() in ev.IMAGE_SUFFIXES for p in image_dir.iterdir()),
                "people_found": person["found"],
                "people_labelled": person["found"] + person["missed"],
                "people_false_alarms": person["false_alarms"],
                "found": sum(c["found"] for c in per_class.values()),
                "labelled": sum(c["found"] + c["missed"] for c in per_class.values()),
                "false_alarms": sum(c["false_alarms"] for c in per_class.values()),
                "per_class": per_class,
            }
        )
    return rows


def markdown(rows: list[dict], setup: str) -> str:
    """The rows as a Markdown table, with what every preset does; ``setup`` is a line about the model and data."""
    head = [
        "Preset",
        "Corruption",
        "Frames",
        "People found / labelled",
        "People false alarms",
        "All found / labelled",
        "All false alarms",
    ]
    lines = ["# Detector robustness", "", setup, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        about = "none: the split as it is" if r["preset"] == CLEAN else PRESETS[r["preset"]].about
        cells = [r["preset"], about, r["frames"], f"{r['people_found']} / {r['people_labelled']}"]
        cells += [r["people_false_alarms"], f"{r['found']} / {r['labelled']}", r["false_alarms"]]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml from scripts/seg_to_yolo.py")
    p.add_argument("--weights", required=True, help="detector weights")
    p.add_argument("--split", default="val", help="the split to score and corrupt (default val)")
    p.add_argument("--presets", default=None, help=f"comma-separated, default all: {','.join(PRESETS)}")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold of the ground station")
    p.add_argument("--device", default="cpu", help="'cpu' or a GPU such as 'cuda:0'")
    p.add_argument("--out", default=None, metavar="DIR", help="output folder (default: runs/<time>_robustness)")
    a = p.parse_args(argv)

    os.environ.setdefault("YOLO_OFFLINE", "1")
    data, weights = Path(a.data).expanduser().resolve(), Path(a.weights).expanduser().resolve()
    out = Path(a.out).expanduser() if a.out else PROJECT_ROOT / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_robustness"
    try:
        presets = dd.parse_presets(a.presets)
        copies = dd.degrade(data, a.split, out, presets)
    except (ValueError, FileExistsError) as e:
        p.error(str(e))
    detector = YoloDetector(weights, device=a.device, conf=a.conf)
    datasets = {CLEAN: data, **{name: out / name / "data.yaml" for name in presets}}
    rows = score(detector, datasets, a.split)
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights": str(weights),
        "data": str(data),
        "split": a.split,
        "station_detector": detector.describe(),
        "match_iou": ev.MATCH_IOU,
        "frames_left_out": {name: s["dropped_frames"] for name, s in copies.items() if s["dropped_frames"]},
        "rows": rows,
    }
    (out / "robustness.json").write_text(json.dumps(report, indent=2))
    with open(out / "robustness.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    setup = f"{weights.name} ({detector.describe()}) on the {a.split} split of {data}."
    text = markdown(rows, setup)
    (out / "robustness.md").write_text(text)
    print(text)
    print(f"wrote {out}/robustness.md, robustness.csv and robustness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
