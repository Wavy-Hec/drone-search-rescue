import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_gcs.dataset import (
    ConvertConfig,
    convert_dataset,
    frame_annotations,
    frame_labels,
    instance_boxes,
    parse_id_mapping,
    split_keys,
)
from drone_gcs.detector import Detection

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
CUT_CAR = "0 0.030000 0.700000 0.060000 0.200000"  # car cut by the left border: visible box x 0..5, y 30..39
EXPECTED_DEFAULT = [EXPECTED[0], EXPECTED[1], CUT_CAR, EXPECTED[2]]  # default: cut-off objects keep their visible box


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
        self.assertEqual(
            [(b.inst, b.x0, b.y0, b.x1, b.y1) for b in boxes][:2], [(5, 10, 5, 29, 14), (7, 30, 20, 39, 29)]
        )
        class_ids = {c: i for i, c in enumerate(CLASSES)}
        sizes = {"car": (20.0, 10.0), "tent": (10.0, 10.0)}
        # default: the cut-off car is labelled with its visible box; the tiny person is dropped and painted over
        cfg = ConvertConfig(classes=CLASSES)
        lines, dropped, paint = frame_annotations(seg, parse_id_mapping(INFO), class_ids, cfg, sizes)
        self.assertEqual(lines, EXPECTED_DEFAULT)
        self.assertEqual(dict(dropped), {"tiny_area": 1, "unknown_class": 1})
        self.assertEqual(paint, [9])  # the 'bird' (not a wanted class) is background and is not painted
        # opt-in: drop objects with less than 40 % visible (and paint them over too)
        cfg = ConvertConfig(classes=CLASSES, max_truncation=0.6)
        lines, dropped, paint = frame_annotations(seg, parse_id_mapping(INFO), class_ids, cfg, sizes)
        self.assertEqual(lines, EXPECTED)
        self.assertEqual(dict(dropped), {"tiny_area": 1, "truncated": 1, "unknown_class": 1})
        self.assertEqual(sorted(paint), [9, 11])
        self.assertEqual(frame_labels(seg, parse_id_mapping(INFO), class_ids, cfg, sizes)[0], EXPECTED)
        # with an unclipped (loose) box for instance 11 showing it is 75 % visible, it is kept
        lines2, _ = frame_labels(seg, parse_id_mapping(INFO), class_ids, cfg, sizes, loose_boxes={11: (-2, 30, 5, 39)})
        self.assertIn(CUT_CAR, lines2)

    def test_convert_dataset_end_to_end(self):
        tmp = Path(short_tmpdir("dgcs_ds"))
        try:
            src, out = tmp / "cap", tmp / "yolo"
            src.mkdir()
            rng = np.random.default_rng(0)
            images = {}
            for k in range(6):
                images[k] = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
                cv2.imwrite(str(src / f"rgb_{k:04d}.png"), images[k])
                np.save(src / f"instance_{k:04d}.npy", synthetic_seg()[..., None])  # (H, W, 1) like Isaac Lab
                (src / f"instance_{k:04d}.json").write_text(json.dumps(INFO))
            cfg = ConvertConfig(classes=CLASSES, val_frac=0.34, split_mode="block", block_size=2)
            stats = convert_dataset(src, out, cfg)
            self.assertEqual(stats["frames"], 6)
            self.assertEqual(stats["images"], {"train": 4, "val": 2})
            self.assertEqual(stats["dropped"], {"tiny_area": 6, "unknown_class": 6})
            self.assertEqual(stats["painted_instances"], 6)  # the tiny person, once per frame
            self.assertEqual(stats["warnings"], [])
            labels = sorted((out / "labels").rglob("*.txt"))
            self.assertEqual(len(labels), 6)
            for p in labels:
                self.assertEqual(p.read_text().splitlines(), EXPECTED_DEFAULT)
            self.assertEqual(len(list((out / "images" / "val").glob("*.png"))), 2)
            # the tiny person's pixels are painted with the image mean, everything else is the original image
            img_path = next((out / "images").rglob("0000.png"))
            got, orig = cv2.imread(str(img_path)).astype(int), images[0].astype(int)
            person = synthetic_seg() == 9
            np.testing.assert_array_equal(got[~person], orig[~person])
            mean = np.broadcast_to(orig[~person].mean(axis=0), got[person].shape)
            np.testing.assert_allclose(got[person], mean, atol=1)
            y = (out / "data.yaml").read_text()
            self.assertIn("train: images/train", y)
            self.assertIn("  0: car\n  1: tent\n  2: person", y)
            # block split keeps consecutive frames together
            val = sorted(p.stem for p in (out / "labels" / "val").glob("*.txt"))
            self.assertEqual(int(val[1]) - int(val[0]), 1)
            # the training script accepts it (dry run: no training, no GPU; the starting weights only have to exist)
            start = tmp / "yolo26n.pt"
            start.write_bytes(b"")
            spec = importlib.util.spec_from_file_location("train_detector", ROOT / "scripts" / "train_detector.py")
            td = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(td)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = td.main(["--data", str(out / "data.yaml"), "--model", str(start), "--dry-run"])
            self.assertEqual(rc, 0, buf.getvalue())
            rep = json.loads(buf.getvalue())
            self.assertEqual(rep["dataset"]["train_images"], 4)
            self.assertEqual(rep["dataset"]["val_boxes"], 8)
            self.assertEqual(rep["dataset"]["classes"], {"0": "car", "1": "tent", "2": "person"})
            self.assertEqual(rep["train_args"]["imgsz"], 640)
        finally:
            rmtree(tmp)

    def test_split_capture_and_unclipped_boxes(self):
        # train/ val/ test/ folders are the splits; bboxUnclipped (full box of a cut-off object) sets its visible share
        tmp = Path(short_tmpdir("dgcs_split"))
        try:
            src, out = tmp / "cap", tmp / "yolo"
            counts = {"train": 3, "val": 2, "test": 1}
            for split, n in counts.items():
                (src / split).mkdir(parents=True)
                # the cut-off car (id 11, 6 px visible): 6 of 8 px in train / test, 6 of 20 px in val
                full_box = [-14, 30, 5, 39] if split == "val" else [-2, 30, 5, 39]
                for k in range(n):
                    key = f"{split}_{k}"
                    cv2.imwrite(str(src / split / f"rgb_{key}.png"), np.full((H, W, 3), 90, np.uint8))
                    cv2.imwrite(str(src / split / f"instance_{key}.png"), synthetic_seg().astype(np.uint16))
                    info = {**INFO, "bboxUnclipped": {"11": full_box}}
                    (src / split / f"instance_{key}.json").write_text(json.dumps(info))
            stats = convert_dataset(src, out, ConvertConfig(classes=CLASSES, max_truncation=0.6))
            self.assertEqual(stats["images"], counts)
            self.assertEqual(stats["split"]["mode"], "folders")
            self.assertEqual(stats["warnings"], [])
            self.assertEqual(stats["dropped"], {"tiny_area": 6, "unknown_class": 6, "truncated": 2})
            self.assertEqual(stats["painted_instances"], 8)
            self.assertIn("val: images/val\ntest: images/test\n", (out / "data.yaml").read_text())
            for split in counts:
                labels = sorted((out / "labels" / split).glob("*.txt"))
                self.assertEqual(len(labels), counts[split])
                for p in labels:
                    self.assertEqual(p.read_text().splitlines(), EXPECTED if split == "val" else EXPECTED_DEFAULT)
        finally:
            rmtree(tmp)

    def test_split_random(self):
        keys = [f"{i:03d}" for i in range(100)]
        s = split_keys(keys, ConvertConfig(val_frac=0.2, split_mode="random"))
        self.assertEqual(sum(v == "val" for v in s.values()), 20)

    def test_block_split_small_and_large_captures(self):
        # defaults (blocks of up to 50 frames, 15 % val): small captures must still get train data and ~15 % val
        for n in (8, 20, 40, 49, 60, 120, 333, 2000):
            for seed in range(5):
                keys = [f"{i:06d}" for i in range(n)]
                s = split_keys(keys, ConvertConfig(seed=seed))
                n_val = sum(v == "val" for v in s.values())
                self.assertGreater(n - n_val, 0, (n, seed))
                self.assertGreater(n_val, 0, (n, seed))
                self.assertLessEqual(abs(n_val / n - 0.15), 0.03, (n, seed, n_val))
                # val frames come in runs of consecutive frames
                runs = sum(1 for i in range(n) if s[keys[i]] == "val" and (i == 0 or s[keys[i - 1]] != "val"))
                self.assertLessEqual(runs, max(1, n_val // 2), (n, seed))
        s = split_keys([f"{i:06d}" for i in range(2000)], ConvertConfig())
        self.assertEqual(sum(v == "val" for v in s.values()), 300)  # large captures: 6 blocks of 50


class StubDetector:
    """Returns the same detections for every frame."""

    def __init__(self, detections: list[Detection]):
        self.detections = detections

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        return self.detections


class TestEvalDetector(unittest.TestCase):
    def test_counts_of_the_station_detector(self):
        # found / missed / false alarms come from the station's boxes, matched like the validator's confusion matrix
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.skipTest("ultralytics not installed")
        spec = importlib.util.spec_from_file_location("eval_detector", ROOT / "scripts" / "eval_detector.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        names = {0: "person", 1: "crate", 2: "cone"}
        tmp = Path(short_tmpdir("dgcs_eval"))
        try:
            for sub in ("images", "labels"):
                (tmp / sub / "val").mkdir(parents=True)
            cv2.imwrite(str(tmp / "images" / "val" / "f0.png"), np.zeros((H, W, 3), np.uint8))
            labels = [
                "0 0.200000 0.500000 0.200000 0.600000",  # person: x 10..30, y 10..40
                "1 0.600000 0.400000 0.200000 0.400000",  # crate: x 50..70, y 10..30
                "2 0.850000 0.700000 0.100000 0.200000",  # cone: x 80..90, y 30..40
            ]
            (tmp / "labels" / "val" / "f0.txt").write_text("\n".join(labels) + "\n")
            station = StubDetector(
                [
                    Detection(0, "person", 0.9, (11.0, 10.0, 30.0, 41.0)),  # the person
                    Detection(2, "cone", 0.8, (50.0, 10.0, 70.0, 30.0)),  # the crate, called a cone
                    Detection(0, "person", 0.4, (35.0, 5.0, 45.0, 20.0)),  # nothing there; the cone is missed
                ]
            )
            people: dict = {}
            matrix = ev.station_matrix(station, tmp, "images/val", names, people=people).matrix
            self.assertEqual(people, {tmp / "images" / "val" / "f0.png": (1, 1)})  # per frame: found, labelled
            self.assertEqual(
                ev.counts(matrix, names),
                {
                    "person": {"found": 1, "missed": 0, "false_alarms": 1},
                    "crate": {"found": 0, "missed": 1, "false_alarms": 0},
                    "cone": {"found": 0, "missed": 1, "false_alarms": 1},
                },
            )
            person = ev.station_matrix(station, tmp, "images/val", names, keep=[0]).matrix  # people only
            self.assertEqual(ev.counts(person, names, [0]), {"person": {"found": 1, "missed": 0, "false_alarms": 1}})
        finally:
            rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
