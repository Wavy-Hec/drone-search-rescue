import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_gcs.detector import ColorBlobDetector, Detection
from drone_gcs.external_eval import (
    SIZE_BUCKETS,
    box_ios,
    box_iou,
    detect_tiled,
    evaluate_people,
    match_boxes,
    merge_tiles,
    read_yolo_labels,
    scale_item,
    size_buckets,
    tile_origins,
)
from drone_gcs.sim.fake_world import OBJECT_CLASSES

from ._util import ROOT, rmtree, short_tmpdir

GREY = (90, 90, 90)
MAGENTA = OBJECT_CLASSES["person"]["color"]  # what the colour-blob detector calls a person
RED = OBJECT_CLASSES["car"]["color"]


def load_script(name: str = "eval_external"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plain(width: int, height: int) -> np.ndarray:
    """A grey RGB image."""
    return np.full((height, width, 3), GREY, np.uint8)


def disc(rgb: np.ndarray, cx: int, cy: int, r: int, color: tuple = MAGENTA) -> tuple[int, int, int, int]:
    """Draw a filled disc; returns its box (x1, y1, x2, y2) [px]: pixels cx - r .. cx + r, cy - r .. cy + r."""
    cv2.circle(rgb, (cx, cy), r, color, -1)
    return (cx - r, cy - r, cx + r + 1, cy + r + 1)


def yolo_line(cls_id: int, box: tuple, width: int, height: int) -> str:
    """The YOLO label line of a pixel box."""
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


class Recorder:
    """The colour-blob detector, keeping every image it was given."""

    def __init__(self):
        self.blob = ColorBlobDetector()
        self.seen: list[np.ndarray] = []

    def describe(self) -> str:
        return "stand-in"

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        self.seen.append(rgb)
        return self.blob.detect(rgb)


class TestLabels(unittest.TestCase):
    def test_reads_only_the_kept_classes_in_pixels(self):
        tmp = Path(short_tmpdir("dgcs_ext"))
        try:
            path = tmp / "a.txt"
            path.write_text(
                "0 0.500000 0.500000 0.200000 0.400000\n"  # x 80..120, y 30..70 in a 200 x 100 image
                "3 0.100000 0.100000 0.100000 0.100000\n"  # another class
                "\n"
                "1 0.250000 0.200000 0.100000 0.200000 0.9 0.8\n"  # x 40..60, y 10..30; extra columns
            )
            np.testing.assert_allclose(read_yolo_labels(path, 200, 100, {0, 1}), [[80, 30, 120, 70], [40, 10, 60, 30]])
            np.testing.assert_allclose(read_yolo_labels(path, 200, 100, {1}), [[40, 10, 60, 30]])
            self.assertEqual(read_yolo_labels(path, 200, 100, {7}).shape, (0, 4))
            self.assertEqual(read_yolo_labels(tmp / "missing.txt", 200, 100, {0}).shape, (0, 4))  # no label file
        finally:
            rmtree(tmp)


class TestTiling(unittest.TestCase):
    def test_hand_computed_origins(self):
        # 640 px tiles every 512 px, 480 px tiles every 384 px; the last one ends at the border
        self.assertEqual(tile_origins(640, 480), [(0, 0)])
        self.assertEqual(tile_origins(1000, 480), [(0, 0), (360, 0)])
        self.assertEqual(tile_origins(1152, 480), [(0, 0), (512, 0)])  # exactly two steps: no tile twice
        self.assertEqual(tile_origins(641, 481), [(0, 0), (1, 0), (0, 1), (1, 1)])
        self.assertEqual(
            tile_origins(1500, 1000),
            [(x, y) for y in (0, 384, 520) for x in (0, 512, 860)],
        )
        self.assertEqual(tile_origins(300, 200), [(0, 0)])  # smaller than a tile
        self.assertEqual(tile_origins(300, 1000), [(0, 0), (0, 384), (0, 520)])  # narrower, but taller
        grid = [(x, y) for y in (0, 30, 60, 70) for x in (0, 40, 60)]
        self.assertEqual(tile_origins(100, 100, tile=(40, 30), overlap=0.0), grid)
        self.assertEqual(tile_origins(100, 30, tile=(40, 30), overlap=0.5), [(0, 0), (20, 0), (40, 0), (60, 0)])
        for bad in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                tile_origins(1000, 480, overlap=bad)

    def test_tiles_cover_every_pixel_with_the_overlap(self):
        cases = [
            (640, 480, (640, 480), 0.2),
            (1920, 1080, (640, 480), 0.2),
            (1000, 777, (640, 480), 0.2),  # not multiples of the step
            (300, 200, (640, 480), 0.2),  # smaller than a tile
            (700, 100, (640, 480), 0.2),  # smaller along one axis only
            (97, 61, (20, 16), 0.0),
            (97, 61, (20, 16), 0.5),
            (97, 61, (20, 16), 0.9),
            (53, 47, (10, 10), 0.33),
        ]
        for width, height, tile, overlap in cases:
            with self.subTest(width=width, height=height, tile=tile, overlap=overlap):
                origins = tile_origins(width, height, tile, overlap)
                self.assertEqual(len(set(origins)), len(origins))  # no tile twice
                covered = np.zeros((height, width), int)
                for x, y in origins:
                    self.assertTrue(x >= 0 and y >= 0)
                    self.assertTrue(x + tile[0] <= width or x == 0)  # inside the image, unless the image is smaller
                    self.assertTrue(y + tile[1] <= height or y == 0)
                    covered[y : y + tile[1], x : x + tile[0]] += 1
                self.assertGreaterEqual(covered.min(), 1)
                for axis, size, starts in ((0, width, {x for x, _ in origins}), (1, height, {y for _, y in origins})):
                    starts = np.array(sorted(starts))
                    shared = starts[:-1] + tile[axis] - starts[1:]  # pixels a tile shares with the next one
                    wanted = round(tile[axis] * overlap)
                    self.assertTrue((shared[:-1] == wanted).all())  # the requested overlap ...
                    self.assertTrue((shared >= wanted).all())  # ... and at least as much for the shifted last tile
                    self.assertEqual(starts[-1], max(0, size - tile[axis]))  # the last tile ends at the border


class TestMerging(unittest.TestCase):
    def test_iou_by_hand(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[5, 5, 15, 15], [0, 0, 10, 5], [20, 20, 30, 30], [3, 3, 3, 3]])
        # 25 / 175; 50 / 100; apart; a box without area
        np.testing.assert_allclose(box_iou(a, b), [[1 / 7, 0.5, 0.0, 0.0]])
        self.assertEqual(box_iou(np.zeros((0, 4)), b).shape, (0, 4))
        self.assertEqual(box_iou(a, np.zeros((0, 4))).shape, (1, 0))

    def test_ios_by_hand(self):
        self.assertEqual(box_ios((0, 0, 10, 10), (5, 5, 15, 15)), 0.25)  # 25 of the 100 px of either box
        self.assertEqual(box_ios((0, 0, 10, 10), (2, 2, 4, 8)), 1.0)  # inside: the whole of the smaller one
        self.assertEqual(box_ios((0, 0, 10, 10), (8, 0, 12, 10)), 0.5)  # half of the 40 px sliver
        self.assertEqual(box_ios((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(box_ios((0, 0, 10, 10), (3, 3, 3, 3)), 0.0)  # a box without area

    def test_an_object_two_tiles_found_is_kept_once_and_whole(self):
        whole = Detection(0, "person", 0.9, (100.0, 50.0, 131.0, 81.0))
        cut = Detection(0, "person", 0.6, (100.0, 50.0, 110.0, 81.0))  # the same person, cut by a tile's border
        merged = merge_tiles([(1, cut), (0, whole)])
        self.assertEqual([(d.conf, d.box) for d in merged], [(0.9, whole.box)])
        # cut by the borders of both tiles: the two parts make the person whole, with the higher confidence
        top = Detection(0, "person", 0.5, (200.0, 372.0, 230.0, 480.0))
        bottom = Detection(0, "person", 0.7, (200.0, 384.0, 230.0, 487.0))
        merged = merge_tiles([(0, top), (2, bottom)])
        self.assertEqual([(d.conf, d.box) for d in merged], [(0.7, (200.0, 372.0, 230.0, 487.0))])

    def test_what_one_tile_kept_apart_stays_apart(self):
        a = Detection(0, "person", 0.9, (0.0, 0.0, 10.0, 10.0))
        b = Detection(0, "person", 0.8, (2.0, 2.0, 12.0, 12.0))  # two people side by side in one tile
        self.assertEqual(len(merge_tiles([(0, a), (0, b)])), 2)
        # ... and both still pair up with their own copies from the next tile, not with each other
        a2 = Detection(0, "person", 0.7, (0.0, 0.0, 10.0, 10.0))
        b2 = Detection(0, "person", 0.6, (2.0, 2.0, 12.0, 12.0))
        merged = merge_tiles([(0, a), (0, b), (1, a2), (1, b2)])
        self.assertEqual(sorted((d.conf, d.box) for d in merged), [(0.8, b.box), (0.9, a.box)])

    def test_merge_needs_the_same_class_and_enough_overlap(self):
        person = Detection(0, "person", 0.9, (0.0, 0.0, 10.0, 10.0))
        car = Detection(1, "car", 0.8, (0.0, 0.0, 10.0, 10.0))
        self.assertEqual(len(merge_tiles([(0, person), (1, car)])), 2)
        half = Detection(0, "person", 0.5, (8.0, 0.0, 12.0, 10.0))  # half of it lies in the person's box
        self.assertEqual(len(merge_tiles([(0, person), (1, half)])), 1)  # exactly at the threshold: merged
        self.assertEqual(len(merge_tiles([(0, person), (1, half)], ios=0.51)), 2)
        self.assertEqual(merge_tiles([]), [])

    def test_a_person_in_two_tiles_is_found_once(self):
        rgb = plain(1000, 480)  # tiles start at x = 0 and x = 360: columns 360..639 are in both
        boxes = [
            disc(rgb, 100, 100, 20),  # only in the left tile
            disc(rgb, 500, 240, 20),  # whole in both
            disc(rgb, 640, 400, 15),  # whole in the right tile, cut by the left tile's border at x = 640
            disc(rgb, 900, 300, 20),  # only in the right tile
        ]
        detector = Recorder()
        dets = detect_tiled(detector, rgb, (640, 480), 0.2)
        self.assertEqual([im.shape for im in detector.seen], [(480, 640, 3), (480, 640, 3)])
        self.assertEqual({d.cls for d in dets}, {"person"})
        # in whole-image pixels, each person once: what the detector finds when it gets the image whole
        whole = sorted(d.box for d in ColorBlobDetector().detect(rgb))
        self.assertEqual(sorted(d.box for d in dets), whole)
        # (the stand-in's 3 px opening trims the one-pixel tips of a drawn disc: its boxes are 1 px smaller)
        np.testing.assert_allclose(whole, boxes, atol=1.0)
        np.testing.assert_array_equal(detector.seen[1], rgb[:, 360:])
        # the left tile saw a sliver of the cut person: it was merged into that person, not counted on its own
        sliver = [d for d in ColorBlobDetector().detect(detector.seen[0]) if d.box[2] >= 639]
        self.assertEqual(len(sliver), 1)
        # without overlap the tiles are 0..639 and 360..999 all the same (the last one is shifted back)
        self.assertEqual(len(detect_tiled(Recorder(), rgb, (640, 480), 0.0)), 4)
        # a smaller image is one tile: the detector gets it whole
        small = Recorder()
        self.assertEqual(len(detect_tiled(small, rgb[:300, :400], (640, 480), 0.2)), 1)
        self.assertEqual([im.shape for im in small.seen], [(300, 400, 3)])


class TestMatching(unittest.TestCase):
    def test_one_to_one_highest_confidence_first(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0]])
        det = np.array([[0.0, 0.0, 10.0, 9.0], [0.0, 0.0, 10.0, 6.0]])  # IoU 0.9 and 0.6 with the label
        matched, false_alarm = match_boxes(det, np.array([0.6, 0.8]), gt, 0.45)
        self.assertEqual(matched.tolist(), [1])  # the more confident box gets the label, not the better fitting one
        self.assertEqual(false_alarm.tolist(), [True, False])
        matched, false_alarm = match_boxes(det, np.array([0.8, 0.6]), gt, 0.45)
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([0], [False, True]))

    def test_a_box_finds_one_label_only(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 2.0, 10.0, 12.0]])
        det = np.array([[0.0, 1.0, 10.0, 10.0]])  # IoU 0.9 with the first label, 8 / 11 with the second
        matched, false_alarm = match_boxes(det, np.array([0.7]), gt, 0.45)
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([0, -1], [False]))

    def test_the_second_box_takes_a_free_label(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0]])
        # box 0 fits label 0 (IoU 1; 6 / 14 with label 1) and box 1 lies between the labels: IoU 7 / 13 with
        # label 0 and 9 / 11 with label 1. Box 1 is more confident and takes label 1, box 0 then takes label 0.
        det = np.array([[0.0, 0.0, 10.0, 10.0], [3.0, 0.0, 13.0, 10.0]])
        matched, false_alarm = match_boxes(det, np.array([0.5, 0.9]), gt, 0.45)
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([0, 1], [False, False]))
        # label 1 alone: the more confident box 1 takes it; box 0 does not reach it (IoU 6 / 14) -> false alarm
        matched, false_alarm = match_boxes(det, np.array([0.5, 0.9]), gt[1:], 0.45)
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([1], [True, False]))
        # label 0 alone: box 1 goes first and takes it with IoU 7 / 13, so the perfect but less confident box 0 is left
        matched, false_alarm = match_boxes(det, np.array([0.5, 0.9]), gt[:1], 0.45)
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([1], [True, False]))

    def test_threshold_and_empty_inputs(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0]])
        det = np.array([[0.0, 0.0, 10.0, 5.0]])  # IoU 0.5
        self.assertEqual(match_boxes(det, np.array([0.9]), gt, 0.5)[0].tolist(), [0])  # at least the threshold
        matched, false_alarm = match_boxes(det, np.array([0.9]), gt, 0.51)
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([-1], [True]))
        matched, false_alarm = match_boxes(np.zeros((0, 4)), np.zeros(0), gt, 0.45)  # nothing detected
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([-1], []))
        matched, false_alarm = match_boxes(det, np.array([0.9]), np.zeros((0, 4)), 0.45)  # nothing labelled
        self.assertEqual((matched.tolist(), false_alarm.tolist()), ([], [True]))


class TestSizes(unittest.TestCase):
    def test_buckets_by_the_longer_side(self):
        self.assertEqual(SIZE_BUCKETS, ("<16", "16-32", "32-64", "64-128", ">=128"))
        sides = [(15.9, 3), (16, 16), (5, 31.9), (32, 1), (63.9, 40), (2, 64), (127.9, 127.9), (128, 5), (20, 900)]
        boxes = np.array([[10.0, 20.0, 10.0 + w, 20.0 + h] for w, h in sides])
        self.assertEqual(size_buckets(boxes).tolist(), [0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual(size_buckets(np.zeros((0, 4))).tolist(), [])


def site() -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Two synthetic images with their people labels, and where each disc is.

    Image 0 (900 x 600 px; at full size the tiles start at x = 0, 260 and y = 0, 120):
    * a labelled person of 41 px in all four tiles;
    * a labelled person of 9 px (about 50 px of area: too small for the blob detector at half size);
    * a person without a label -> a false alarm;
    * a label of 100 x 30 px on empty ground -> missed;
    * a red disc, which the detector calls a car -> not counted.
    Image 1 (320 x 240 px, smaller than a tile): empty ground without labels.
    """
    rgb = plain(900, 600)
    where = {
        "big": disc(rgb, 450, 300, 20),
        "small": disc(rgb, 100, 80, 4),
        "unlabelled": disc(rgb, 800, 500, 15),
        "empty": (700, 100, 800, 130),
        "car": disc(rgb, 150, 400, 25, RED),
    }
    labels = np.array([where["big"], where["small"], where["empty"]], float)
    return [(rgb, labels), (plain(320, 240), np.zeros((0, 4)))], where


class TestEvaluatePeople(unittest.TestCase):
    def test_counts_at_full_size(self):
        items, _ = site()
        result = evaluate_people(ColorBlobDetector(), items)
        self.assertEqual(
            result,
            {
                "images": 2,
                "people_labelled": 3,
                "found": 2,
                "missed": 1,
                "false_alarms": 1,
                "precision": 0.6667,
                "recall": 0.6667,
                "by_size": {
                    "<16": {"labelled": 1, "found": 1, "recall": 1.0},  # the 9 px person
                    "16-32": {"labelled": 0, "found": 0, "recall": None},
                    "32-64": {"labelled": 1, "found": 1, "recall": 1.0},  # the 41 px person
                    "64-128": {"labelled": 1, "found": 0, "recall": 0.0},  # the 100 px label on empty ground
                    ">=128": {"labelled": 0, "found": 0, "recall": None},
                },
            },
        )
        self.assertEqual(evaluate_people(ColorBlobDetector(), items, tile=None), result)  # whole images: the same

    def test_counts_at_half_size(self):
        items, _ = site()
        result = evaluate_people(ColorBlobDetector(), items, scale=0.5)
        self.assertEqual(
            result,
            {
                "images": 2,
                "people_labelled": 3,
                "found": 1,
                "missed": 2,
                "false_alarms": 1,
                "precision": 0.5,
                "recall": 0.3333,
                "by_size": {
                    "<16": {"labelled": 1, "found": 0, "recall": 0.0},  # 4.5 px: too small to be found
                    "16-32": {"labelled": 1, "found": 1, "recall": 1.0},  # the person, now 20.5 px
                    "32-64": {"labelled": 1, "found": 0, "recall": 0.0},  # the label on empty ground, now 50 px
                    "64-128": {"labelled": 0, "found": 0, "recall": None},
                    ">=128": {"labelled": 0, "found": 0, "recall": None},
                },
            },
        )

    def test_scaling_moves_the_labels_with_the_image(self):
        items, where = site()
        rgb, boxes = scale_item(items[0][0], items[0][1], 2.0)
        self.assertEqual(rgb.shape, (1200, 1800, 3))
        np.testing.assert_allclose(boxes[0], 2.0 * np.array(where["big"]))
        x1, y1, x2, y2 = boxes[0].astype(int)
        self.assertEqual(tuple(rgb[(y1 + y2) // 2, (x1 + x2) // 2]), MAGENTA)  # the disc is under its label
        self.assertEqual(tuple(rgb[y1 - 4, x1 - 4]), GREY)
        # a size that does not divide evenly: the boxes follow the true size of the resized image
        rgb, boxes = scale_item(plain(101, 50), np.array([[0.0, 0.0, 101.0, 50.0]]), 0.5)
        self.assertEqual(rgb.shape, (25, 50, 3))
        np.testing.assert_allclose(boxes, [[0.0, 0.0, 50.0, 25.0]])
        same, same_boxes = scale_item(items[0][0], items[0][1], 1.0)
        self.assertIs(same, items[0][0])
        self.assertIs(same_boxes, items[0][1])

    def test_tiles_are_used_and_only_people_count(self):
        items, where = site()
        detector = Recorder()
        evaluate_people(detector, items[:1])
        self.assertEqual([im.shape for im in detector.seen], [(480, 640, 3)] * 4)  # 900 x 600 px: 2 x 2 tiles
        # had the red disc been labelled as a person, it would still be missed: the detector calls it a car
        with_car = [(items[0][0], np.array([where["car"]], float))]
        result = evaluate_people(ColorBlobDetector(), with_car)
        self.assertEqual((result["found"], result["missed"], result["false_alarms"]), (0, 1, 3))
        self.assertEqual(evaluate_people(ColorBlobDetector(), [])["images"], 0)  # no image: every ratio is None
        self.assertIsNone(evaluate_people(ColorBlobDetector(), [])["precision"])


class TestScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()
        cls.tmp = Path(short_tmpdir("dgcs_ext"))
        (cls.tmp / "images").mkdir()
        (cls.tmp / "labels").mkdir()
        items, where = site()
        rgb = items[0][0]
        cv2.imwrite(str(cls.tmp / "images" / "a.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lines = [
            yolo_line(0, where["big"], 900, 600),
            yolo_line(2, where["small"], 900, 600),  # people have two class ids in this dataset
            yolo_line(0, where["empty"], 900, 600),
            yolo_line(5, where["car"], 900, 600),  # another class: ignored
            yolo_line(5, where["unlabelled"], 900, 600),  # a person labelled as something else: a false alarm
        ]
        (cls.tmp / "labels" / "a.txt").write_text("\n".join(lines) + "\n")
        lonely = plain(320, 240)  # no label file: no people, so its disc is a false alarm
        disc(lonely, 160, 120, 12)
        cv2.imwrite(str(cls.tmp / "images" / "b.png"), cv2.cvtColor(lonely, cv2.COLOR_RGB2BGR))
        (cls.tmp / "images" / "notes.txt").write_text("not an image\n")
        cls.where = where

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp)

    def run_script(self, *extra: str, detector: Recorder | None = None) -> tuple[int, str, Path]:
        """Run main() with the stand-in detector; returns its exit code, what it printed and the output folder."""
        out = self.tmp / f"out{len(list(self.tmp.glob('out*')))}"
        argv = ["--images", str(self.tmp / "images"), "--labels", str(self.tmp / "labels"), "--weights", "none.pt"]
        argv += ["--out", str(out), *extra]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            code = self.script.main(argv, make_detector=lambda args: detector or Recorder())
        return code, printed.getvalue(), out

    def test_writes_the_json_and_the_markdown(self):
        detector = Recorder()
        code, printed, out = self.run_script("--person-ids", "0,2", detector=detector)
        self.assertEqual(code, 0)
        report = json.loads((out / "external_eval.json").read_text())
        counts = {k: report[k] for k in ("images", "people_labelled", "found", "missed", "false_alarms")}
        self.assertEqual(counts, {"images": 2, "people_labelled": 3, "found": 2, "missed": 1, "false_alarms": 2})
        self.assertEqual((report["precision"], report["recall"]), (0.5, 0.6667))
        self.assertEqual(report["by_size"]["<16"], {"labelled": 1, "found": 1, "recall": 1.0})
        self.assertEqual(report["by_size"]["64-128"], {"labelled": 1, "found": 0, "recall": 0.0})
        self.assertEqual((report["person_ids"], report["scale"], report["tile"]), ([0, 2], 1.0, [640, 480]))
        self.assertEqual((report["overlap"], report["match_iou"], report["detector"]), (0.2, 0.45, "stand-in"))
        text = (out / "external_eval.md").read_text()
        self.assertIn("| Images | People labelled | Found | Missed | False alarms | Precision | Recall |", text)
        self.assertIn("| 2 | 3 | 2 | 1 | 2 | 0.500 | 0.667 |", text)
        self.assertIn("| <16 | 1 | 1 | 1.000 |", text)
        self.assertIn("| 16-32 | 0 | 0 | - |", text)
        self.assertIn("| 64-128 | 1 | 0 | 0.000 |", text)
        self.assertIn(text, printed)
        # the detector gets RGB tiles: the disc's centre has the person colour, red-green-blue
        x1, y1, x2, y2 = self.where["big"]
        self.assertEqual(tuple(detector.seen[0][(y1 + y2) // 2, (x1 + x2) // 2]), MAGENTA)
        self.assertEqual(len(detector.seen), 5)  # four tiles of a.png, then b.png whole; notes.txt is not an image

    def test_person_ids_scale_and_limit(self):
        # class 0 only: the 9 px person (class 2) is not a label any more, so its box becomes a false alarm
        _, _, out = self.run_script("--person-ids", "0")
        report = json.loads((out / "external_eval.json").read_text())
        counts = [report[k] for k in ("images", "people_labelled", "found", "missed", "false_alarms")]
        self.assertEqual(counts, [2, 2, 1, 1, 3])
        # half size, first image only: the 9 px person is too small to be found; the unlabelled one still is
        _, _, out = self.run_script("--person-ids", "0,2", "--scale", "0.5", "--limit", "1")
        report = json.loads((out / "external_eval.json").read_text())
        counts = [report[k] for k in ("images", "people_labelled", "found", "missed", "false_alarms")]
        self.assertEqual(counts, [1, 3, 1, 2, 1])
        self.assertEqual(report["by_size"]["16-32"], {"labelled": 1, "found": 1, "recall": 1.0})
        self.assertIn("resized by 0.5", (out / "external_eval.md").read_text())

    def test_no_tile_gives_the_whole_image_to_the_detector(self):
        detector = Recorder()
        _, _, out = self.run_script("--person-ids", "0,2", "--no-tile", detector=detector)
        self.assertEqual([im.shape for im in detector.seen], [(600, 900, 3), (240, 320, 3)])
        report = json.loads((out / "external_eval.json").read_text())
        self.assertEqual((report["tile"], report["overlap"], report["found"]), (None, None, 2))
        self.assertIn("given whole to the detector", (out / "external_eval.md").read_text())

    def test_bad_arguments_stop_before_the_detector_is_built(self):
        empty = self.tmp / "empty"
        empty.mkdir(exist_ok=True)
        base = ["--weights", "none.pt", "--person-ids", "0"]
        folders = ["--images", str(self.tmp / "images"), "--labels", str(self.tmp / "labels")]
        cases = [
            [*base, "--images", str(self.tmp / "nowhere"), "--labels", str(self.tmp / "labels")],
            [*base, "--images", str(empty), "--labels", str(self.tmp / "labels")],  # no image
            [*base, "--images", str(self.tmp / "images"), "--labels", str(empty)],  # no label file fits an image
            ["--weights", "none.pt", "--person-ids", "0,people", *folders],
            [*base, *folders, "--scale", "0"],
            [*base, *folders, "--overlap", "1"],
            [*base, *folders, "--limit", "0"],
        ]

        def never(args):
            raise AssertionError("the detector must not be built")

        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    self.script.main(argv, make_detector=never)
                self.assertEqual(cm.exception.code, 2)


class TestVisDroneLabels(unittest.TestCase):
    """scripts/visdrone_to_yolo.py: VisDrone annotations -> YOLO labels that read_yolo_labels reads back."""

    def setUp(self):
        self.tmp = Path(short_tmpdir("visdrone"))
        self.addCleanup(rmtree, self.tmp)
        self.script = load_script("visdrone_to_yolo")
        src = self.tmp / "VisDrone2019-DET-val"
        (src / "images").mkdir(parents=True)
        (src / "annotations").mkdir()
        cv2.imwrite(str(src / "images" / "a.jpg"), np.full((100, 200, 3), 90, np.uint8))  # 200 x 100 px
        cv2.imwrite(str(src / "images" / "b.jpg"), np.full((50, 80, 3), 90, np.uint8))  # no annotation file
        (src / "annotations" / "a.txt").write_text(
            "10,20,30,40,1,1,0,0\n"  # pedestrian -> class 0
            "50,10,20,20,0,0,0,0\n"  # ignored region (score 0) -> left out
            "100,50,40,20,1,4,0,1\n"  # car -> class 3
            "0,0,10,10,1,2,0,0\n"  # people -> class 1
        )
        self.src = src

    def test_the_labels_by_hand_and_back_to_pixels(self):
        out = self.tmp / "labels"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.script.main(["--src", str(self.src), "--out", str(out)]), 0)
        lines = (out / "a.txt").read_text().splitlines()
        self.assertEqual(
            lines,
            [
                "0 0.125000 0.400000 0.150000 0.400000",  # centre (25, 40) of 200 x 100, size 30 x 40
                "3 0.600000 0.600000 0.200000 0.200000",
                "1 0.025000 0.050000 0.050000 0.100000",
            ],
        )
        self.assertEqual((out / "b.txt").read_text(), "")  # an image without annotations has no labels
        people = read_yolo_labels(out / "a.txt", 200, 100, {0, 1})  # the people only, as eval_external reads them
        np.testing.assert_allclose(people, [[10, 20, 40, 60], [0, 0, 10, 10]], atol=1e-6)

    def test_a_folder_without_annotations_is_refused(self):
        (self.src / "annotations" / "a.txt").unlink()
        (self.src / "annotations").rmdir()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            self.script.main(["--src", str(self.src), "--out", str(self.tmp / "labels")])
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
