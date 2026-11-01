import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_gcs.dataset import ConvertConfig, convert_dataset, frame_labels, instance_boxes, parse_id_mapping, split_keys

from ._util import ROOT, rmtree, short_tmpdir

W, H = 100, 50
CLASSES = ["car", "tent", "person"]


def synthetic_seg() -> np.ndarray:
    seg = np.zeros((H, W), np.int32)
    seg[45:50, 0:100] = 1  # UNLABELLED strip
    seg[5:15, 10:30] = 5  # car 20 x 10 px
    seg[20:22, 30:40] = 7  # tent, L-shaped mask: box x 30..39, y 20..29
    seg[22:30, 30:32] = 7
    seg[40:42, 50:52] = 9  # person, 2 x 2 px -> tiny
    seg[30:40, 0:6] = 11  # car cut by the left border: 6 of ~20 px visible -> truncated
    seg[35:45, 93:100] = 12  # tent at the right border: 7 of 10 px visible -> kept
    seg[2:12, 60:70] = 13  # a class that is not in the class list
    return seg


INFO = {
    "idToLabels": {"0": "BACKGROUND", "1": "UNLABELLED", "5": "/World/envs/env_0/Car_01", "7": "/World/Tent_2",
                   "9": "/World/Person", "11": "/World/Car_02", "12": "/World/Tent_3", "13": "/World/Bird"},
    "idToSemantics": {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}, "5": {"class": "car"},
                      "7": {"class": "tent"}, "9": {"class": "person"}, "11": {"class": "car"},
                      "12": {"class": "tent"}, "13": {"class": "bird"}},
}  # fmt: skip

EXPECTED = [
    "0 0.200000 0.200000 0.200000 0.200000",  # car: x 10..29, y 5..14
    "1 0.350000 0.500000 0.100000 0.200000",  # tent (L-shape): x 30..39, y 20..29
    "1 0.965000 0.800000 0.070000 0.200000",  # tent at the border: x 93..99, y 35..44
]


class TestSegToYolo(unittest.TestCase):
    def test_mapping_formats(self):
        self.assertEqual(parse_id_mapping(INFO), {5: "car", 7: "tent", 9: "person", 11: "car", 12: "tent", 13: "bird"})
        prim_only = {"idToLabels": INFO["idToLabels"]}
        m = parse_id_mapping(prim_only, {"Car": "car", "Tent": "tent"})
        self.assertEqual(m, {5: "car", 7: "tent", 11: "car", 12: "tent"})
        self.assertEqual(parse_id_mapping({"3": "car", "4": {"class": "tent"}, "x": "y"}), {3: "car", 4: "tent"})

    def test_boxes_and_exact_labels(self):
        seg = synthetic_seg()
        boxes, dropped = instance_boxes(seg, parse_id_mapping(INFO), min_area=20, min_side=4)
        self.assertEqual(dropped["tiny_area"], 1)
        self.assertEqual([(b.inst, b.x0, b.y0, b.x1, b.y1) for b in boxes][:2], [(5, 10, 5, 29, 14), (7, 30, 20, 39, 29)])
        cfg = ConvertConfig(classes=CLASSES)
        class_ids = {c: i for i, c in enumerate(CLASSES)}
        sizes = {"car": (20.0, 10.0), "tent": (10.0, 10.0)}
        lines, dropped = frame_labels(seg, parse_id_mapping(INFO), class_ids, cfg, sizes)
        self.assertEqual(lines, EXPECTED)
        self.assertEqual(dict(dropped), {"tiny_area": 1, "truncated": 1, "unknown_class": 1})
        # with an unclipped (loose) box for instance 11 showing it is 75 % visible, it is kept
        lines2, _ = frame_labels(seg, parse_id_mapping(INFO), class_ids, cfg, sizes, loose_boxes={11: (-2, 30, 5, 39)})
        self.assertIn("0 0.030000 0.700000 0.060000 0.200000", lines2)

    def test_convert_dataset_end_to_end(self):
        tmp = Path(short_tmpdir("dgcs_ds"))
        try:
            src, out = tmp / "cap", tmp / "yolo"
            src.mkdir()
            rng = np.random.default_rng(0)
            for k in range(6):
                cv2.imwrite(str(src / f"rgb_{k:04d}.png"), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
                np.save(src / f"instance_{k:04d}.npy", synthetic_seg()[..., None])  # (H, W, 1) like Isaac Lab
                (src / f"instance_{k:04d}.json").write_text(json.dumps(INFO))
            cfg = ConvertConfig(classes=CLASSES, val_frac=0.34, split_mode="block", block_size=2)
            stats = convert_dataset(src, out, cfg)
            self.assertEqual(stats["frames"], 6)
            self.assertEqual(stats["images"], {"train": 4, "val": 2})
            self.assertEqual(stats["dropped"], {"tiny_area": 6, "truncated": 6, "unknown_class": 6})
            labels = sorted((out / "labels").rglob("*.txt"))
            self.assertEqual(len(labels), 6)
            for p in labels:
                self.assertEqual(p.read_text().splitlines(), EXPECTED)
            self.assertEqual(len(list((out / "images" / "val").glob("*.png"))), 2)
            y = (out / "data.yaml").read_text()
            self.assertIn("train: images/train", y)
            self.assertIn("  0: car\n  1: tent\n  2: person", y)
            # block split keeps consecutive frames together
            val = sorted(p.stem for p in (out / "labels" / "val").glob("*.txt"))
            self.assertEqual(int(val[1]) - int(val[0]), 1)
            # the training script accepts it (dry run: no training, no GPU)
            spec = importlib.util.spec_from_file_location("train_detector", ROOT / "scripts" / "train_detector.py")
            td = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(td)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = td.main(["--data", str(out / "data.yaml"), "--dry-run"])
            self.assertEqual(rc, 0, buf.getvalue())
            rep = json.loads(buf.getvalue())
            self.assertEqual(rep["dataset"]["train_images"], 4)
            self.assertEqual(rep["dataset"]["val_boxes"], 6)
            self.assertEqual(rep["dataset"]["classes"], {"0": "car", "1": "tent", "2": "person"})
            self.assertEqual(rep["train_args"]["imgsz"], 640)
        finally:
            rmtree(tmp)

    def test_split_random(self):
        keys = [f"{i:03d}" for i in range(100)]
        s = split_keys(keys, ConvertConfig(val_frac=0.2, split_mode="random"))
        self.assertEqual(sum(v == "val" for v in s.values()), 20)


if __name__ == "__main__":
    unittest.main()
