#!/usr/bin/env python3
"""Write YOLO label files for the VisDrone detection set, for ``scripts/eval_external.py``.

VisDrone (http://aiskyeye.com, real drone photos; for research use) labels every image with one line per object in
``annotations/<stem>.txt``: ``<left>,<top>,<width>,<height>,<score>,<category>,<truncation>,<occlusion>`` [px].
This writes ``<out>/<stem>.txt`` with ``<class> <cx> <cy> <w> <h>`` normalised to the image, by the rule Ultralytics
uses for its VisDrone dataset: lines with score 0 (the "ignored regions") are left out and ``class = category - 1``,
so 0 is pedestrian and 1 is people (2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus,
9 motor). The images stay where they are.

Example (Ultralytics hosts the validation set as VisDrone2019-DET-val.zip in its assets releases)::

    python scripts/visdrone_to_yolo.py --src ~/data/VisDrone2019-DET-val --out ~/data/VisDrone2019-DET-val/labels
    python scripts/eval_external.py --images ~/data/VisDrone2019-DET-val/images \\
        --labels ~/data/VisDrone2019-DET-val/labels --person-ids 0,1 --weights ../downloads/drone_sar_yolo26n.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def convert_line(line: str, width: int, height: int) -> str | None:
    """One VisDrone annotation line -> one YOLO label line, or None for an ignored region (score 0)."""
    fields = line.strip().split(",")
    if len(fields) < 6 or fields[4] == "0":
        return None
    left, top, w, h = (float(v) for v in fields[:4])
    cls = int(fields[5]) - 1
    cx, cy = (left + w / 2) / width, (top + h / 2) / height
    return f"{cls} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def convert(src: Path, out: Path) -> tuple[int, int]:
    """Label files for every image of ``src/images`` into ``out``; returns (images, labels written)."""
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n_labels = 0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"cannot read {path}")
        height, width = img.shape[:2]
        ann = src / "annotations" / f"{path.stem}.txt"
        lines = ann.read_text().splitlines() if ann.is_file() else []
        labels = [y for y in (convert_line(line, width, height) for line in lines if line.strip()) if y is not None]
        (out / f"{path.stem}.txt").write_text("".join(label + "\n" for label in labels))
        n_labels += len(labels)
    return len(images), n_labels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, metavar="DIR", help="VisDrone folder with images/ and annotations/")
    p.add_argument("--out", required=True, metavar="DIR", help="folder for the YOLO label files")
    a = p.parse_args(argv)
    src = Path(a.src).expanduser()
    if not (src / "images").is_dir() or not (src / "annotations").is_dir():
        p.error(f"--src: {src} needs images/ and annotations/ folders")
    n_images, n_labels = convert(src, Path(a.out).expanduser())
    print(f"{n_images} images, {n_labels} labels -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
