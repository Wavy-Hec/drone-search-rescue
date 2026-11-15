"""Image corruptions (``drone_gcs/corruptions.py``), ``scripts/degrade_dataset.py`` and the scoring of
``scripts/eval_robustness.py`` (that part needs torch and Ultralytics). ~2 s."""

import contextlib
import importlib.util
import io
import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_gcs.corruptions import (
    HAZE_GREY,
    PRESETS,
    gaussian_noise,
    haze,
    jpeg,
    low_light,
    motion_blur,
    occlude,
    shrink_mosaic,
)
from drone_gcs.detector import Detection

from ._util import ROOT, rmtree, short_tmpdir

W, H = 64, 48
NAMES = {0: "person", 1: "crate"}
SPLITS = {"train": "images/val", "val": "images/val"}
"""Splits of a corrupted copy of val: the Ultralytics validator needs a train key."""


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_ultralytics() -> bool:
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def texture(seed: int = 0, low: int = 0, high: int = 256, shape: tuple = (H, W)) -> np.ndarray:
    """A random RGB image, every pixel and channel uniform in [low, high)."""
    return np.random.default_rng(seed).integers(low, high, (*shape, 3), dtype=np.uint8)


def write_dataset(root: Path, frames: list[tuple[np.ndarray, list[str]]]) -> Path:
    """A val split of ``frames`` (image, label lines) named f0, f1, ... with a data.yaml like seg_to_yolo's."""
    for sub in ("images", "labels"):
        (root / sub / "val").mkdir(parents=True)
    for i, (rgb, lines) in enumerate(frames):
        cv2.imwrite(str(root / "images" / "val" / f"f{i}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (root / "labels" / "val" / f"f{i}.txt").write_text("".join(line + "\n" for line in lines))
    names = "".join(f"  {i}: {c}\n" for i, c in NAMES.items())
    data = root / "data.yaml"
    data.write_text(f"# written by drone_gcs seg_to_yolo\npath: {root.resolve()}\nval: images/val\nnames:\n{names}")
    return data


class TestPhotometric(unittest.TestCase):
    def test_every_frame_preset_keeps_shape_and_dtype_and_is_deterministic(self):
        names = ["noise", "blur", "dark", "jpeg", "haze", "occlude30", "occlude60", "altitude2", "altitude3"]
        self.assertEqual(list(PRESETS), names)
        self.assertEqual([PRESETS[n].k for n in ("altitude2", "altitude3")], [2, 3])
        rgb = texture()
        labels = [(0, 0.5, 0.5, 0.25, 0.25)]
        for name, preset in PRESETS.items():
            if preset.frame is None:
                continue
            with self.subTest(preset=name):
                a = preset.frame(rgb, labels, np.random.default_rng(3))
                b = preset.frame(rgb, labels, np.random.default_rng(3))
                self.assertEqual((a.shape, a.dtype), (rgb.shape, np.uint8))
                np.testing.assert_array_equal(a, b)
                self.assertFalse(np.array_equal(a, rgb))

    def test_noise_has_the_requested_std(self):
        grey = np.full((200, 200, 3), 128, np.uint8)
        noisy = gaussian_noise(grey, 15.0, np.random.default_rng(0))
        self.assertAlmostEqual(float(noisy.std()), 15.0, delta=0.3)
        self.assertAlmostEqual(float(noisy.mean()), 128.0, delta=0.3)
        self.assertFalse(np.array_equal(noisy, gaussian_noise(grey, 15.0, np.random.default_rng(1))))

    def test_blur_spreads_a_point_along_the_direction(self):
        point = np.zeros((31, 31, 3), np.uint8)
        point[15, 15] = 220
        for angle, cols, rows in ((0.0, 11, 1), (90.0, 1, 11)):
            with self.subTest(angle=angle):
                ys, xs = np.nonzero(motion_blur(point, 11, angle)[..., 0])
                self.assertEqual((np.ptp(xs) + 1, np.ptp(ys) + 1, len(xs)), (cols, rows, 11))
                self.assertEqual(int(motion_blur(point, 11, angle)[15, 15, 0]), 20)  # 220 / 11 on each pixel
        ys, xs = np.nonzero(motion_blur(point, 11, 45.0)[..., 0])  # top left to bottom right (y points down)
        np.testing.assert_array_equal(xs - 15, ys - 15)
        self.assertEqual(len(xs), 9)  # the 11 px kernel holds the diagonal from -4 to +4
        with self.assertRaises(ValueError):
            motion_blur(point, 10, 0.0)

    def test_low_light_scales_the_brightness(self):
        rgb = texture(low=100, high=200)  # scaled to 25..50: the noise is not clipped at 0
        dark = low_light(rgb, 0.25, 6.0, np.random.default_rng(0))
        self.assertAlmostEqual(float(dark.mean()) / float(rgb.mean()), 0.25, delta=0.005)
        self.assertAlmostEqual(float((dark - rgb * 0.25).std()), 6.0, delta=0.3)

    def test_jpeg_loses_more_at_lower_quality(self):
        rgb = cv2.GaussianBlur(texture(), (5, 5), 0)
        err = {q: np.abs(jpeg(rgb, q).astype(int) - rgb).mean() for q in (15, 95)}
        self.assertLess(err[95], err[15])
        self.assertGreater(err[15], 1.0)

    def test_haze_lowers_the_contrast(self):
        rgb = texture()
        hazy = haze(rgb, 0.5)
        self.assertAlmostEqual(float(hazy.std()) / float(rgb.std()), 0.5, delta=0.01)
        self.assertAlmostEqual(float(hazy.mean()), 0.5 * float(rgb.mean()) + 0.5 * HAZE_GREY, delta=0.3)


class TestOcclude(unittest.TestCase):
    BOXES = [(10.0, 8.0, 30.0, 28.0), (40.2, 20.0, 55.6, 36.0)]  # 20 x 20 px and 16 x 16 px (x 40..55)
    INSIDE = [(slice(8, 28), slice(10, 30)), (slice(20, 36), slice(40, 56))]

    def test_hides_the_fraction_with_ground_from_outside_the_boxes(self):
        rgb = texture()
        outside = np.ones((H, W), bool)
        for rows, cols in self.INSIDE:
            outside[rows, cols] = False
        for fraction, hidden in ((0.3, (120, 80)), (0.6, (240, 160))):  # 6 or 12 of 20 rows, 5 or 10 of 16 rows
            for seed in range(4):
                with self.subTest(fraction=fraction, seed=seed):
                    out = occlude(rgb, self.BOXES, fraction, np.random.default_rng(seed))
                    np.testing.assert_array_equal(out, occlude(rgb, self.BOXES, fraction, np.random.default_rng(seed)))
                    changed = (out != rgb).any(axis=2)
                    self.assertFalse(changed[outside].any())  # the ground the strips come from stays
                    self.assertEqual(tuple(int(changed[r, c].sum()) for r, c in self.INSIDE), hidden)
                    for rows, cols in self.INSIDE:  # each strip is a copy of a place outside the boxes
                        ys, xs = np.nonzero(changed[rows, cols])
                        strip = out[rows, cols][ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                        windows = np.lib.stride_tricks.sliding_window_view(rgb, strip.shape)[..., 0, :, :, :]
                        same = np.argwhere((windows == strip).all(axis=(2, 3, 4)))
                        self.assertEqual(len(same), 1)
                        (sr, sc), (sh, sw) = same[0], strip.shape[:2]
                        self.assertTrue(outside[sr : sr + sh, sc : sc + sw].all())

    def test_a_box_without_ground_elsewhere_is_left(self):
        rgb = texture()
        np.testing.assert_array_equal(occlude(rgb, [(0, 0, W, H)], 0.3, np.random.default_rng(0)), rgb)
        # the ground left of a 60 px wide box is 4 px wide: too narrow for a 60 px strip across the box, wide enough
        # for a 3 px strip down it
        outs = [occlude(rgb, [(4, 0, W, H)], 0.05, np.random.default_rng(seed)) for seed in range(8)]
        hidden = {int((out != rgb).any(axis=2).sum()) for out in outs}
        self.assertEqual(hidden, {0, 3 * H})


class TestShrinkMosaic(unittest.TestCase):
    def test_label_lands_where_its_object_does(self):
        for k, frame, label, expected in (
            (2, 3, (1, 0.5, 0.25, 1 / 3, 0.25), (1, 0.75, 0.625, 1 / 6, 0.125)),  # frame 3: row 1, column 1
            (3, 5, (0, 0.5, 0.5, 0.25, 0.5), (0, 2.5 / 3, 0.5, 0.25 / 3, 0.5 / 3)),  # frame 5: row 1, column 2
        ):
            with self.subTest(k=k):
                # 72 x 48 px frames; the object's pixel box is a multiple of k, so it shrinks to whole pixels
                images = [np.full((H, 72, 3), 10 * i, np.uint8) for i in range(k * k)]
                _, cx, cy, w, h = label
                x0, y0 = round((cx - w / 2) * 72), round((cy - h / 2) * H)
                images[frame][y0 : y0 + round(h * H), x0 : x0 + round(w * 72)] = 255  # the object
                labels = [[] for _ in range(k * k)]
                labels[frame] = [label]
                tiled, tiled_labels = shrink_mosaic(images, labels, k)
                self.assertEqual((tiled.shape, tiled.dtype), (images[0].shape, np.uint8))
                self.assertEqual(len(tiled_labels), 1)
                np.testing.assert_allclose(tiled_labels[0], expected)
                ys, xs = np.nonzero(tiled[..., 0] == 255)
                _, cx, cy, w, h = expected
                box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                want = ((cx - w / 2) * 72, (cy - h / 2) * H, (cx + w / 2) * 72, (cy + h / 2) * H)
                np.testing.assert_allclose(box, want)
                self.assertEqual(int(tiled[0, 0, 0]), 0)  # frame 0 in the top left tile
                self.assertEqual(int(tiled[-1, 0, 0]), 10 * k * (k - 1))  # the first frame of the last row
        with self.assertRaises(ValueError):
            shrink_mosaic([np.zeros((H, W, 3), np.uint8)] * 3, [[]] * 3, 2)


class TestDegradeDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(short_tmpdir("dgcs_degrade"))
        self.dd = load_script("degrade_dataset")
        labels = ["0 0.500000 0.500000 0.250000 0.250000", "1 0.200000 0.300000 0.125000 0.250000"]
        self.data = write_dataset(self.tmp / "src", [(texture(i), labels) for i in range(5)])

    def tearDown(self):
        rmtree(self.tmp)

    def test_writes_a_dataset_per_preset(self):
        out = self.tmp / "out"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            argv = ["--data", str(self.data), "--out", str(out), "--presets", "noise,occlude30,altitude2"]
            self.assertEqual(self.dd.main(argv), 0)  # the split is val by default
        printed = buf.getvalue()
        self.assertIn("noise: 5 frames, 10 labels", printed)
        self.assertIn("altitude2: 1 frames, 8 labels", printed)
        self.assertIn("1 last frames left out", printed)
        source = sorted(p.read_text() for p in (self.tmp / "src" / "labels" / "val").glob("*.txt"))
        for name in ("noise", "occlude30"):
            folder = out / name
            self.assertEqual(
                sorted(p.name for p in (folder / "images" / "val").iterdir()), [f"f{i}.png" for i in range(5)]
            )
            self.assertEqual(sorted(p.read_text() for p in (folder / "labels" / "val").glob("*.txt")), source)
            self.assertEqual(self.dd.read_data_yaml(folder / "data.yaml"), (folder.resolve(), SPLITS, NAMES))
            image = cv2.imread(str(folder / "images" / "val" / "f2.png"))[..., ::-1]
            self.assertEqual(image.shape, (H, W, 3))
            self.assertFalse(np.array_equal(image, texture(2)))
        tiled = out / "altitude2"
        self.assertEqual([p.name for p in (tiled / "images" / "val").iterdir()], ["f0_to_f3.png"])
        rows = (tiled / "labels" / "val" / "f0_to_f3.txt").read_text().splitlines()
        self.assertEqual(rows[:2], ["0 0.250000 0.250000 0.125000 0.125000", "1 0.100000 0.150000 0.062500 0.125000"])
        self.assertEqual(rows[-1], "1 0.600000 0.650000 0.062500 0.125000")  # frame 3: the bottom right tile
        # the same seed gives the same files, another seed other noise; an existing output folder is refused
        again, other = self.tmp / "again", self.tmp / "other"
        self.dd.degrade(self.data, "val", again, ["noise", "occlude30"], seed=0)
        for name in ("noise", "occlude30"):
            for p in (out / name / "images" / "val").iterdir():
                self.assertEqual(p.read_bytes(), (again / name / "images" / "val" / p.name).read_bytes())
        self.dd.degrade(self.data, "val", other, ["noise"], seed=1)
        f0 = Path("noise", "images", "val", "f0.png")
        self.assertNotEqual((out / f0).read_bytes(), (other / f0).read_bytes())
        with self.assertRaises(FileExistsError):
            self.dd.degrade(self.data, "val", again, ["noise"])
        with self.assertRaises(ValueError):
            self.dd.parse_presets("noise,fog")

    def test_a_copy_of_another_split_names_it_train_and_val_too(self):
        # the Ultralytics validator needs both keys, whatever the split
        test = self.tmp / "with_test"
        shutil.copytree(self.tmp / "src", test)
        for sub in ("images", "labels"):
            (test / sub / "val").rename(test / sub / "test")
        data = test / "data.yaml"
        data.write_text(data.read_text().replace(f"path: {(self.tmp / 'src').resolve()}", f"path: {test.resolve()}")
                        .replace("val: images/val", "test: images/test"))  # fmt: skip
        self.dd.degrade(data, "test", self.tmp / "out_test", ["blur"])
        folder = (self.tmp / "out_test" / "blur").resolve()
        expected = {"train": "images/test", "val": "images/test", "test": "images/test"}
        self.assertEqual(self.dd.read_data_yaml(folder / "data.yaml"), (folder, expected, NAMES))
        if has_ultralytics():
            from ultralytics.data.utils import check_det_dataset

            self.assertEqual(Path(check_det_dataset(str(folder / "data.yaml"))["test"]), folder / "images" / "test")

    def test_data_yaml_is_read_by_eval_detector(self):
        if not has_ultralytics():
            self.skipTest("torch or ultralytics not installed")
        self.dd.degrade(self.data, "val", self.tmp / "out", ["blur"])
        folder = (self.tmp / "out" / "blur").resolve()
        self.assertEqual(self.dd.ev.read_names(folder / "data.yaml"), (folder, SPLITS, NAMES))


class StubDetector:
    """Finds the person and the crate, plus a person where there is none, in a bright frame; nothing in a dark one."""

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        if rgb.mean() < 100:
            return []
        return [
            Detection(0, "person", 0.9, (24.0, 18.0, 40.0, 30.0)),
            Detection(1, "crate", 0.8, (8.0, 8.0, 17.0, 20.0)),
            Detection(0, "person", 0.5, (48.0, 30.0, 60.0, 44.0)),
        ]


class TestEvalRobustness(unittest.TestCase):
    def test_counts_on_the_clean_split_and_a_copy(self):
        if not has_ultralytics():
            self.skipTest("torch or ultralytics not installed")
        er = load_script("eval_robustness")
        tmp = Path(short_tmpdir("dgcs_robust"))
        try:
            labels = ["0 0.500000 0.500000 0.250000 0.250000", "1 0.200000 0.300000 0.125000 0.250000"]
            # person x 24..40, y 18..30 [px]; crate x 8.8..16.8, y 8.4..20.4
            data = write_dataset(tmp / "src", [(np.full((H, W, 3), 180, np.uint8), labels)])
            er.dd.degrade(data, "val", tmp / "out", ["dark"])
            rows = er.score(StubDetector(), {er.CLEAN: data, "dark": tmp / "out" / "dark" / "data.yaml"}, "val")
            clean, dark = ({k: r[k] for k in er.COLUMNS} for r in rows)
            self.assertEqual(clean, dict(zip(er.COLUMNS, ("clean", 1, 1, 1, 1, 2, 2, 1), strict=True)))
            self.assertEqual(dark, dict(zip(er.COLUMNS, ("dark", 1, 0, 1, 0, 0, 2, 0), strict=True)))
            self.assertEqual(rows[0]["per_class"]["crate"], {"found": 1, "missed": 0, "false_alarms": 0})
            table = er.markdown(rows, "setup")
            self.assertIn("| clean | none: the split as it is | 1 | 1 / 1 | 1 | 2 / 2 | 1 |", table)
            self.assertIn(f"| dark | {PRESETS['dark'].about} | 1 | 0 / 1 | 0 | 0 / 2 | 0 |", table)
        finally:
            rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
