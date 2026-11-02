#!/usr/bin/env python3
"""Fine-tune the YOLO detector on simulator images (dataset from scripts/seg_to_yolo.py).

Starts from the COCO nano weights, trains with defaults sized for ~2000 images on a 16 GB GPU, copies the best
checkpoint to ``--out`` (default ``models/detector.pt``, tracked with Git LFS) and writes the training summary
next to it as ``<out name>_metrics.json`` (default ``models/detector_metrics.json``), with the training curves
(losses and val scores per epoch) as ``<out name>_training.png`` and ``.csv``.
The GCS then uses it with ``python scripts/gcs.py --weights models/detector.pt``.

Defaults and why:
  epochs 120 + patience 30   ~2000 images x 120 epochs; early stop when val mAP stalls for 30 epochs
  imgsz 640                  matches the camera width; use 960 if the objects are < ~15 px in the images
  batch 32                   YOLO26n at 640 uses ~5-6 GB at batch 32 (fits next to other jobs on 16 GB)
  cache ram                  2000 x 640x480 images ~ 1.8 GB decoded: no disk I/O per epoch
  flipud 0.5                 nadir imagery has no 'up', vertical flips are valid augmentations
  close_mosaic 15, cos_lr    standard fine-tuning schedule
Ultralytics (AGPL-3.0): fine-tuned weights fall under the same licence, see the README.

Examples::

    python scripts/train_detector.py --data ~/isaac_ws/datasets/drone_yolo/data.yaml --dry-run
    python scripts/train_detector.py --data ~/isaac_ws/datasets/drone_yolo/data.yaml --epochs 120
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.path.expanduser("~/isaac_ws/assets/models/yolo26n.pt")


def read_data_yaml(path: Path) -> dict:
    """Minimal reader for the data.yaml written by seg_to_yolo (path / train / val / names)."""
    d: dict = {"names": {}}
    in_names = False
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if in_names and line.startswith((" ", "\t")) and ":" in line:
            k, v = line.strip().split(":", 1)
            d["names"][int(k)] = v.strip()
            continue
        in_names = False
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "names" and not v:
                in_names = True
            else:
                d[k] = v
    return d


def count_split(root: Path, split_dir: str) -> tuple[int, int]:
    img_dir = root / split_dir
    lab_dir = root / split_dir.replace("images", "labels", 1)
    imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")] if img_dir.is_dir() else []
    boxes = 0
    for p in lab_dir.glob("*.txt") if lab_dir.is_dir() else []:
        boxes += sum(1 for ln in p.read_text().splitlines() if ln.strip())
    return len(imgs), boxes


def build_train_args(a) -> dict:
    return {
        "data": str(Path(a.data).expanduser().resolve()),
        "epochs": a.epochs,
        "patience": a.patience,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "device": a.device,
        "workers": a.workers,
        "cache": a.cache if a.cache != "none" else False,
        "project": str(Path(a.project).expanduser().resolve()),
        "name": a.name,
        "exist_ok": False,
        "seed": a.seed,
        "deterministic": True,
        "cos_lr": True,
        "close_mosaic": 15,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "scale": 0.5,
        "degrees": 0.0,
        "pretrained": True,
        "plots": True,
        "verbose": True,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml from scripts/seg_to_yolo.py")
    p.add_argument("--model", default=DEFAULT_MODEL, help="starting weights (default: COCO yolo26n.pt)")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0", help="'0' (first GPU) or 'cpu'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", choices=("ram", "disk", "none"), default="ram")
    p.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"), help="Ultralytics project dir")
    p.add_argument("--name", default=f"detector_{time.strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(PROJECT_ROOT / "models" / "detector.pt"), help="where best.pt is copied")
    p.add_argument("--offline", action="store_true", help="set YOLO_OFFLINE=1 (no update checks or analytics)")
    p.add_argument("--dry-run", action="store_true", help="check the dataset, print the training config and stop")
    a = p.parse_args(argv)

    data = Path(a.data).expanduser()
    if not data.exists():
        print(f"error: {data} not found", file=sys.stderr)
        return 2
    d = read_data_yaml(data)
    root = Path(d.get("path") or data.parent)
    n_train, b_train = count_split(root, d.get("train", "images/train"))
    n_val, b_val = count_split(root, d.get("val", "images/val"))
    train_args = build_train_args(a)
    report = {
        "dataset": {"yaml": str(data.resolve()), "classes": d["names"], "train_images": n_train, "train_boxes": b_train,
                    "val_images": n_val, "val_boxes": b_val},
        "model": a.model,
        "train_args": train_args,
        "output": a.out,
    }  # fmt: skip
    problems = []
    if not d["names"]:
        problems.append("data.yaml has no class names")
    if n_train == 0:
        problems.append("no training images")
    if n_val == 0:
        problems.append("no validation images")
    if not Path(a.model).expanduser().exists():
        problems.append(f"model weights {a.model} not found")
    report["problems"] = problems
    print(json.dumps(report, indent=2))
    if a.dry_run:
        return 1 if problems else 0
    if problems:
        print("refusing to train: " + "; ".join(problems), file=sys.stderr)
        return 2

    if a.offline:
        os.environ["YOLO_OFFLINE"] = "1"
    from ultralytics import YOLO  # noqa: PLC0415

    t0 = time.time()
    model = YOLO(str(Path(a.model).expanduser()))
    results = model.train(**train_args)
    trainer = model.trainer
    best = Path(trainer.best)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    for ext in ("png", "csv"):  # Ultralytics' training curves
        curves = Path(trainer.save_dir) / f"results.{ext}"
        if curves.exists():
            shutil.copy2(curves, out.with_name(f"{out.stem}_training.{ext}"))
    metrics = {k: float(v) for k, v in getattr(results, "results_dict", {}).items()}
    per_class = {}
    try:
        box = results.box
        for i, c in enumerate(results.ap_class_index):
            per_class[d["names"].get(int(c), str(c))] = {"ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i])}
    except Exception:  # noqa: BLE001  (metrics layout differs between versions)
        pass
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights": str(out),
        "source_best": str(best),
        "start_model": a.model,
        "train_minutes": round((time.time() - t0) / 60, 1),
        "epochs_run": int(getattr(trainer, "epoch", -1)) + 1,
        "metrics": metrics,
        "per_class": per_class,
        **report,
    }
    metrics_path = out.with_name(f"{out.stem}_metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"best weights -> {out}\nmetrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
