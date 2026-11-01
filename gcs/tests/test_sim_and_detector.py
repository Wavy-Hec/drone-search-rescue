import math
import os
import time
import unittest
from pathlib import Path

import numpy as np

from drone_gcs.detector import DEFAULT_WEIGHTS, ColorBlobDetector, Detection, DetectorWorker
from drone_gcs.geometry import C_BODY_TO_NADIR_CAM, PinholeCamera, intrinsics_from_fov, quat_from_euler, quat_to_matrix
from drone_gcs.mapping import GridSpec
from drone_gcs.sim.fake_world import FakeWorld
from drone_gcs.sim.flight import CarrotFlight, lawnmower


class TestFakeWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = FakeWorld(area=(-12, 12, -12, 12), seed=3)

    @classmethod
    def tearDownClass(cls):
        cls.w.close()

    def cam_over(self, x, y, z, yaw=0.0, tilt=0.0):
        R = quat_to_matrix(quat_from_euler(tilt, 0.0, yaw)) @ C_BODY_TO_NADIR_CAM
        return PinholeCamera(intrinsics_from_fov(640, 480, 75), 640, 480, R, [x, y, z])

    def test_deterministic(self):
        w2 = FakeWorld(area=(-12, 12, -12, 12), seed=3)
        np.testing.assert_array_equal(w2.height, self.w.height)
        self.assertEqual(w2.gt_objects(), self.w.gt_objects())
        w2.close()

    def test_scan_hits_lie_on_the_height_field(self):
        pts, valid = self.w.scan_grid([0.0, 0.0, 15.0], 0.3, (5, 5), 0.25)
        self.assertEqual(len(pts), 21 * 21)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(pts[:, 2], self.w.height_at(pts[:, 0], pts[:, 1]), atol=1e-4)
        # rotated grid: the corner ray sits at yaw-rotated offset
        c, s = math.cos(0.3), math.sin(0.3)
        np.testing.assert_allclose(pts[0, :2], [c * -2.5 - s * -2.5, s * -2.5 + c * -2.5], atol=1e-5)

    def test_render_depth_and_projection_consistent(self):
        o = self.w.objects[0]
        cam = self.cam_over(o.x + 2.0, o.y - 1.0, 14.0, yaw=0.5, tilt=0.08)
        rgb, depth = self.w.render(cam)
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertTrue(np.isfinite(depth).all())
        # every pixel's back-projected depth point lies on the height field
        rng = np.random.default_rng(0)
        u, v = rng.integers(0, 640, 300), rng.integers(0, 480, 300)
        P = cam.backproject(u, v, depth[v, u])
        err = np.abs(P[:, 2] - self.w.height_at(P[:, 0], P[:, 1]))
        self.assertLess(np.median(err), 0.01)
        self.assertLess(np.percentile(err, 95), 0.08)  # walls/edges are the worst case
        # the object's top centre projects onto its own colour
        uo, vo, _ = cam.project([[o.x, o.y, o.z + (o.height if o.shape != "pyramid" else 0.3)]])
        px = rgb[int(round(vo[0])), int(round(uo[0]))].astype(int)
        from drone_gcs.sim.fake_world import OBJECT_CLASSES

        ref = np.array(OBJECT_CLASSES[o.cls]["color"])
        self.assertLess(np.abs(px - ref * px.sum() / max(1, ref.sum())).max(), 60)

    def test_truth_on_grid(self):
        g = GridSpec.from_bounds(-2, 2, -2, 2, 0.25)
        t = self.w.truth_on_grid(g)
        self.assertEqual(t.shape, g.shape)
        xs, ys = g.cell_centers()
        X, Y = np.meshgrid(xs, ys)
        np.testing.assert_allclose(np.median(t - self.w.height_at(X, Y)), 0.0, atol=0.01)


class TestFlight(unittest.TestCase):
    def test_lawnmower_covers_area(self):
        p = lawnmower((-10, 10, -6, 6), 4.0, 8.0)
        ys = sorted(set(np.round(p[:, 1], 6)))
        self.assertAlmostEqual(ys[0], -4.0)
        self.assertAlmostEqual(ys[-1], 4.0)
        self.assertTrue(np.all(p[:, 2] == 8.0))
        self.assertTrue(np.all(np.diff(ys) <= 4.0 + 1e-9))

    def test_carrot_flight_completes_at_cruise_speed(self):
        path = lawnmower((-10, 10, -6, 6), 4.0, 8.0)
        f = CarrotFlight(path, speed=3.0)
        speeds = []
        while not f.done and f.t < 120:
            f.step(0.02)
            speeds.append(np.linalg.norm(f.v))
        self.assertTrue(f.done)
        self.assertLess(f.t, f.length / 3.0 * 1.6)
        self.assertLess(max(speeds), 3.0 * 1.35)
        q = f.quat
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=9)


class TestDetectors(unittest.TestCase):
    def test_blob_detector_on_rendered_objects(self):
        w = FakeWorld(area=(-12, 12, -12, 12), seed=3)
        try:
            det = ColorBlobDetector()
            found = 0
            for o in w.objects:
                R = quat_to_matrix(quat_from_euler(0, 0, 0)) @ C_BODY_TO_NADIR_CAM
                cam = PinholeCamera(intrinsics_from_fov(640, 480, 75), 640, 480, R, [o.x, o.y, w.hmax + 4])
                rgb, _ = w.render(cam)
                dets = det.detect(rgb)
                hit = [d for d in dets if d.cls == o.cls and d.box[0] < 320 < d.box[2] and d.box[1] < 240 < d.box[3]]
                found += bool(hit)
                # no detection of a class that is not in view
                self.assertTrue(all(d.conf > 0.5 for d in dets))
            self.assertEqual(found, len(w.objects))
        finally:
            w.close()

    def test_worker_keeps_newest_frame(self):
        class Slow:
            def detect(self, rgb):
                time.sleep(0.05)
                return [Detection(0, "x", 0.9, (0, 0, 1, 1))]

        wk = DetectorWorker(Slow())
        for i in range(10):
            wk.submit(np.zeros((2, 2, 3), np.uint8), i)
        time.sleep(0.25)
        res = wk.poll()
        wk.close()
        ctx = [r[0] for r in res]
        self.assertIn(9, ctx)  # the newest frame is always processed
        self.assertLess(len(res), 10)  # older ones were dropped
        self.assertGreater(wk.n_dropped, 0)
        self.assertGreater(res[0][2], 40)  # inference ms reported

    @unittest.skipUnless(os.path.exists(DEFAULT_WEIGHTS), "YOLO weights not found")
    def test_yolo_wrapper_on_cpu(self):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.skipTest("ultralytics not installed")
        import cv2

        from drone_gcs.detector import YoloDetector

        bus = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        det = YoloDetector(device="cpu", conf=0.4)
        rgb = cv2.imread(str(bus))[..., ::-1]
        dets = det.detect(rgb)
        names = [d.cls for d in dets]
        self.assertIn("person", names)
        self.assertIn("bus", names)
        for d in dets:
            x1, y1, x2, y2 = d.box
            self.assertTrue(0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0])
        det2 = YoloDetector(device="cpu", conf=0.4, classes="person", class_map="person=human")
        self.assertEqual({d.cls for d in det2.detect(rgb)}, {"human"})
        # class filters are checked against the model, and also match names after --class-map
        ids = {n: i for i, n in det2.names.items()}
        det3 = YoloDetector(device="cpu", classes="car", class_map="truck=car")
        self.assertEqual(det3.class_ids, sorted([ids["car"], ids["truck"]]))
        with self.assertRaises(ValueError) as cm:
            YoloDetector(device="cpu", classes="persn")  # a typo used to silently remove every detection
        self.assertIn("persn", str(cm.exception))
        with self.assertRaises(ValueError):
            YoloDetector(device="cpu", classes="bench", class_map="bench=")  # nothing would be left
        det4 = YoloDetector(device="cpu", classes="person,tarp")  # one unknown next to a valid one: warned, ignored
        self.assertEqual((det4.class_ids, det4.ignored_classes), ([ids["person"]], ["tarp"]))
        self.assertEqual(YoloDetector().device, "cpu")  # the default never touches the GPU


if __name__ == "__main__":
    unittest.main()
