#!/usr/bin/env python3
"""Score a fine-tuned detector on its dataset splits, and against the stock COCO model on 'person'.

For every evaluated split of data.yaml (default: val and test):

* precision, recall, mAP50 and mAP50-95 per class and the precision-recall curve of the fine-tuned model;
* at the ground station's confidence threshold (``--conf``, default 0.25): objects found, missed and false
  alarms per class, and the confusion matrix (counts; the validator's own matrix at confidence 0.001 also
  counts every low-score box);
* the person scores of both models on the same images (``classes=[0]`` keeps only the person predictions and
  labels; COCO's class 0 is person too). The COCO model cannot know the other classes.

A side-by-side figure of a few test frames with a person in full view shows both models' detections.

Writes next to the weights (``--out-dir`` to change): ``<name>_eval.json``, ``<name>_confusion_matrix_<split>.png``,
``<name>_pr_curve_<split>.png`` and ``<name>_vs_coco.jpg``. Ultralytics' own validation folders go to
``runs/eval/`` (git-ignored).

Example (conda env 'vision')::

    python scripts/eval_detector.py --data ~/isaac_ws/datasets/drone_sar_yolo/yolo/data.yaml \\
        --weights ~/isaac_ws/assets/models/drone_sar_yolo26n.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_MODEL = os.path.expanduser("~/isaac_ws/assets/models/yolo26n.pt")
HEADER_PX = 28


def read_names(data: Path) -> tuple[Path, dict[str, str], dict[int, str]]:
    """Dataset root, split -> image folder, and class names of a data.yaml written by seg_to_yolo."""
    from ultralytics.utils import YAML

    d = YAML.load(data)
    root = Path(d.get("path") or data.parent)
    splits = {k: d[k] for k in ("train", "val", "test") if d.get(k)}
    names = d["names"] if isinstance(d["names"], dict) else dict(enumerate(d["names"]))
    return root, splits, {int(k): v for k, v in names.items()}


def read_labels(root: Path, image_dir: str) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """``{frame: [(class, cx, cy, w, h), ...]}`` of one split (normalised YOLO boxes)."""
    labels = root / image_dir.replace("images", "labels", 1)
    out = {}
    for p in sorted(labels.glob("*.txt")):
        rows = [line.split() for line in p.read_text().splitlines() if line.strip()]
        out[p.stem] = [(int(r[0]), *(float(v) for v in r[1:5])) for r in rows]
    return out


def scores(metrics, names: dict[int, str]) -> tuple[dict, dict]:
    """Overall and per-class (P, R, mAP50, mAP50-95) of an Ultralytics DetMetrics."""
    box = metrics.box
    overall = {"precision": box.mp, "recall": box.mr, "map50": box.map50, "map50_95": box.map}
    per_class = {
        names.get(int(c), str(c)): {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "map50": float(box.ap50[i]),
            "map50_95": float(box.ap[i]),
        }
        for i, c in enumerate(metrics.ap_class_index)
    }
    return {k: round(float(v), 4) for k, v in overall.items()}, {
        k: {m: round(v, 4) for m, v in d.items()} for k, d in per_class.items()
    }


def counts(metrics, names: dict[int, str], classes: list[int] | None = None) -> dict:
    """Found / missed / false alarms per class from the confusion matrix (rows: predicted, columns: true)."""
    m = metrics.confusion_matrix.matrix
    out = {}
    for c in classes if classes is not None else sorted(names):
        found = int(m[c, c])
        out[names[c]] = {"found": found, "missed": int(m[:, c].sum()) - found, "false_alarms": int(m[c].sum()) - found}
    return out


def validate(model, data: Path, split: str, args, name: str, **kw):
    return model.val(
        data=str(data),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).expanduser().resolve()),
        name=name,
        exist_ok=True,
        verbose=False,
        **kw,
    )


def tile(model, path: Path, title: str, conf: float, imgsz: int, device: str, classes: list[int] | None):
    """One model's detections on one frame, with a title bar above the image."""
    import cv2

    res = model.predict(str(path), conf=conf, imgsz=imgsz, device=device, classes=classes, verbose=False)[0]
    image = res.plot(line_width=2, font_size=11)
    header = np.full((HEADER_PX, image.shape[1], 3), 24, np.uint8)
    n = len(res.boxes)
    text = f"{title}: {n} detection{'s' if n != 1 else ''}"
    cv2.putText(header, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def side_by_side(fine, coco, images: list[Path], out: Path, conf: float, imgsz: int, device: str) -> None:
    """One row per frame: fine-tuned detections left, stock COCO (person only) right."""
    import cv2

    rows = []
    for path in images:
        left = tile(fine, path, f"fine-tuned ({path.stem})", conf, imgsz, device, None)
        right = tile(coco, path, "stock COCO, person", conf, imgsz, device, [0])
        rows.append(np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right]))
    gap = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, gap, row])
    cv2.imwrite(str(out), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def pick_frames(root: Path, image_dir: str, n: int) -> list[Path]:
    """``n`` frames spread over the split, each with a person in full view; the busiest of each stretch."""

    def person_in_view(box) -> bool:
        c, cx, cy, w, h = box
        return c == 0 and min(cx - w / 2, cy - h / 2) > 0.01 and max(cx + w / 2, cy + h / 2) < 0.99

    labels = read_labels(root, image_dir)
    candidates = [k for k, boxes in labels.items() if any(person_in_view(b) for b in boxes)]
    picks = [max(chunk, key=lambda k: len(labels[k])) for chunk in np.array_split(candidates, n) if len(chunk)]
    images = root / image_dir
    return [next(images.glob(f"{k}.*")) for k in picks]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml from scripts/seg_to_yolo.py")
    p.add_argument("--weights", required=True, help="fine-tuned weights")
    p.add_argument("--coco", default=COCO_MODEL, help="stock COCO weights for the person comparison")
    p.add_argument("--splits", default="val,test", help="splits to score (those missing from data.yaml are skipped)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu", help="'cpu' or a GPU index such as '0'")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold of the ground station")
    p.add_argument("--samples", type=int, default=3, help="test frames in the side-by-side figure")
    p.add_argument("--out-dir", default=None, help="where the results go (default: next to the weights)")
    p.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "eval"), help="Ultralytics validation folders")
    a = p.parse_args(argv)

    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO

    data, weights = Path(a.data).expanduser().resolve(), Path(a.weights).expanduser().resolve()
    out_dir = Path(a.out_dir).expanduser() if a.out_dir else weights.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    root, image_dirs, names = read_names(data)
    fine, coco = YOLO(str(weights)), YOLO(str(Path(a.coco).expanduser()))
    stem = weights.stem
    report: dict = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights": str(weights),
        "coco_weights": str(Path(a.coco).expanduser()),
        "data": str(data),
        "imgsz": a.imgsz,
        "conf": a.conf,
        "splits": {},
    }
    for split in [s.strip() for s in a.splits.split(",") if s.strip() in image_dirs]:
        t0 = time.time()
        labels = read_labels(root, image_dirs[split])
        res = validate(fine, data, split, a, f"{stem}_{split}", plots=True)
        overall, per_class = scores(res, names)
        at_conf = validate(fine, data, split, a, f"{stem}_{split}_conf", plots=True, conf=a.conf)
        coco_at_conf = validate(coco, data, split, a, f"coco_{split}_conf", plots=True, conf=a.conf, classes=[0])
        fine_person, _ = scores(validate(fine, data, split, a, f"{stem}_{split}_person", classes=[0]), names)
        coco_person, _ = scores(validate(coco, data, split, a, f"coco_{split}_person", classes=[0]), names)
        report["splits"][split] = {
            "images": len(labels),
            "instances": dict(Counter(names[b[0]] for boxes in labels.values() for b in boxes)),
            "all": overall,
            "per_class": per_class,
            "at_conf": counts(at_conf, names),
            "person": {
                "fine_tuned": {**fine_person, **counts(at_conf, names, [0])["person"]},
                "coco": {**coco_person, **counts(coco_at_conf, coco.names, [0])["person"]},
            },
            "seconds": round(time.time() - t0, 1),
        }
        for run, src, dst in (
            (res, "BoxPR_curve.png", "pr_curve"),
            (at_conf, "confusion_matrix.png", "confusion_matrix"),
        ):
            if (Path(run.save_dir) / src).exists():
                shutil.copy2(Path(run.save_dir) / src, out_dir / f"{stem}_{dst}_{split}.png")
        person = report["splits"][split]["person"]
        print(f"{split}: all {overall}; person fine-tuned {person['fine_tuned']} vs COCO {person['coco']}", flush=True)
    figure_split = "test" if "test" in image_dirs else "val"
    frames = pick_frames(root, image_dirs[figure_split], a.samples)
    if frames:
        side_by_side(fine, coco, frames, out_dir / f"{stem}_vs_coco.jpg", a.conf, a.imgsz, a.device)
        report["figure"] = {"split": figure_split, "frames": [f.stem for f in frames], "conf": a.conf}
    path = out_dir / f"{stem}_eval.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"results -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
