#!/usr/bin/env python3
"""Convert Isaac Lab camera captures (RGB + instance-id segmentation + id -> class mapping) into a YOLO
detection dataset: images/{train,val[,test]}, labels/{train,val[,test]} (``cls cx cy w h`` normalised),
data.yaml and dataset_stats.json.

Input layout (see drone_gcs/dataset.py): ``rgb_<k>.png|jpg|npy`` + ``instance_<k>.npy|png`` [+ ``instance_<k>.json``]
in one folder, or ``rgb/<k>.*`` + ``instance/<k>.*`` [+ ``instance/<k>.json``], or ``train/``, ``val/`` [and
``test/``] folders that each hold such a capture (they become the splits). The JSON is the annotator's ``info``
dict (``idToSemantics`` / ``idToLabels``, optionally ``bboxUnclipped``) or ``{id: class}``; ``--mapping`` gives
one for all frames.

The search-and-rescue site of drone_lab writes split captures directly (``scripts/gen_yolo_dataset.py``)::

    python scripts/seg_to_yolo.py --src ~/isaac_ws/datasets/drone_sar_yolo/capture \\
        --out ~/isaac_ws/datasets/drone_sar_yolo/yolo --classes person,vehicle,crate,barrel,pallet,cone,landing_pad \\
        --min-area 100 --min-side 6 --max-truncation 0.6

Saving captures from another Isaac Lab camera (data_types=["rgb", "instance_id_segmentation_fast"] and
colorize_instance_id_segmentation=False)::

    out = camera.data.output; info = camera.data.info
    cv2.imwrite(f"cap/rgb_{k:06d}.png", out["rgb"].torch[0].cpu().numpy()[..., ::-1])
    np.save(f"cap/instance_{k:06d}.npy", out["instance_id_segmentation_fast"].torch[0, ..., 0].cpu().numpy())
    json.dump(info["instance_id_segmentation_fast"], open(f"cap/instance_{k:06d}.json", "w"))

    python scripts/seg_to_yolo.py --src ~/isaac_ws/datasets/drone_capture --out ~/isaac_ws/datasets/drone_yolo \\
        --classes car,tent,person,tarp --prim-rules "Car=car,Tent=tent,Person=person,Tarp=tarp"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_gcs.dataset import ConvertConfig, convert_dataset, parse_prim_rules  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="capture directory")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--classes", default=None, help="class names in YOLO id order (default: all, sorted)")
    p.add_argument("--mapping", default=None, help="global id -> class JSON (when there are no per-frame JSON files)")
    p.add_argument("--prim-rules", default=None, help="prim-path substring -> class, e.g. 'Car=car,Tent=tent'")
    p.add_argument("--min-area", type=int, default=20, help="min visible pixels per instance")
    p.add_argument("--min-side", type=int, default=4, help="min box width/height [px]")
    p.add_argument(
        "--max-truncation",
        type=float,
        default=1.0,
        help="drop border boxes truncated by more than this fraction (default 1.0: label every cut-off object by "
        "its visible box)",
    )
    p.add_argument(
        "--no-mask-dropped",
        action="store_true",
        help="keep the pixels of dropped (tiny / too truncated) objects; by default they are painted with the "
        "image mean so the detector is not trained to call them background",
    )
    p.add_argument("--val-frac", type=float, default=0.15, help="val share (captures without split folders)")
    p.add_argument("--split-mode", choices=("block", "random"), default="block", help="block: contiguous runs to val")
    p.add_argument("--block-size", type=int, default=50, help="frames per block in block mode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-format", choices=("copy", "png", "jpg"), default="copy")
    p.add_argument("--skip-empty", action="store_true", help="drop frames without any label (default: keep them)")
    a = p.parse_args(argv)
    cfg = ConvertConfig(
        classes=[c.strip() for c in a.classes.split(",")] if a.classes else None,
        min_area=a.min_area,
        min_side=a.min_side,
        max_truncation=a.max_truncation,
        mask_dropped=not a.no_mask_dropped,
        val_frac=a.val_frac,
        split_mode=a.split_mode,
        block_size=a.block_size,
        seed=a.seed,
        image_format=a.image_format,
        skip_empty=a.skip_empty,
        prim_rules=parse_prim_rules(a.prim_rules),
    )
    mapping = json.loads(Path(a.mapping).read_text()) if a.mapping else None
    stats = convert_dataset(a.src, a.out, cfg, mapping)
    keys = ("classes", "frames", "images", "boxes", "per_class", "dropped", "painted_instances", "split", "warnings")
    print(json.dumps({k: stats[k] for k in keys}, indent=2))
    print(f"data.yaml: {Path(a.out) / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
