#!/usr/bin/env python3
"""Convert Isaac Lab camera captures (RGB + instance-id segmentation + id -> class mapping) into a YOLO
detection dataset: images/{train,val}, labels/{train,val} (``cls cx cy w h`` normalised), data.yaml and
dataset_stats.json.

Input layout (see drone_gcs/dataset.py): ``rgb_<k>.png|jpg|npy`` + ``instance_<k>.npy|png`` [+ ``instance_<k>.json``]
in one folder, or ``rgb/<k>.*`` + ``instance/<k>.*`` [+ ``instance/<k>.json``]. The JSON is the annotator's
``info`` dict (``idToSemantics`` / ``idToLabels``) or ``{id: class}``; ``--mapping`` gives one for all frames.

Saving such captures from Isaac Lab (camera with data_types=["rgb", "instance_id_segmentation_fast"] and
colorize_instance_id_segmentation=False)::

    out = camera.data.output; info = camera.data.info
    cv2.imwrite(f"cap/rgb_{k:06d}.png", out["rgb"].torch[0].cpu().numpy()[..., ::-1])
    np.save(f"cap/instance_{k:06d}.npy", out["instance_id_segmentation_fast"].torch[0, ..., 0].cpu().numpy())
    json.dump(info["instance_id_segmentation_fast"], open(f"cap/instance_{k:06d}.json", "w"))

Example::

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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="capture directory")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--classes", default=None, help="comma-separated class names in YOLO id order (default: all found, sorted)")
    p.add_argument("--mapping", default=None, help="global id -> class JSON (when there are no per-frame JSON files)")
    p.add_argument("--prim-rules", default=None, help="prim-path substring -> class, e.g. 'Car=car,Tent=tent'")
    p.add_argument("--min-area", type=int, default=20, help="min visible pixels per instance")
    p.add_argument("--min-side", type=int, default=4, help="min box width/height [px]")
    p.add_argument("--max-truncation", type=float, default=0.6, help="drop border boxes truncated by more than this")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--split-mode", choices=("block", "random"), default="block", help="block = contiguous chunks to val")
    p.add_argument("--block-size", type=int, default=50, help="frames per block in block mode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-format", choices=("copy", "png", "jpg"), default="copy")
    p.add_argument("--skip-empty", action="store_true", help="drop frames without any label (default: keep as background)")
    a = p.parse_args(argv)
    cfg = ConvertConfig(
        classes=[c.strip() for c in a.classes.split(",")] if a.classes else None,
        min_area=a.min_area,
        min_side=a.min_side,
        max_truncation=a.max_truncation,
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
    print(json.dumps({k: stats[k] for k in ("classes", "frames", "images", "boxes", "per_class", "dropped")}, indent=2))
    print(f"data.yaml: {Path(a.out) / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
