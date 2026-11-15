"""Tracker statistics: the spread of a pin against brute force (with and without merges), the count of the frames
that had a pin in view, the detection rate and the ``min_rate`` confirmation rule. ~1 s."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from drone_gcs.dashboard import PAL, Dashboard, render_report_map
from drone_gcs.detector import Detection
from drone_gcs.geometry import C_BODY_TO_NADIR_CAM, PinholeCamera
from drone_gcs.mapping import ObjectTracker
from drone_gcs.report import write_report
from drone_gcs.state import CameraFrame, GCSConfig, GCSState

from ._util import ROOT, rmtree, short_tmpdir

# A 320 x 240 camera with a focal length of 100 px. Looking straight down from 10 m, one metre on the ground is
# 10 px; the image's right is world -y and its bottom is world -x (C_BODY_TO_NADIR_CAM).
W, H = 320, 240
K = np.array([[100.0, 0.0, 160.0], [0.0, 100.0, 120.0], [0.0, 0.0, 1.0]])


def frame_at(x, y, z=10.0, depth=10.0):
    """A camera frame looking straight down from (x, y, z) [m]. ``depth``: None, one value for every pixel [m]
    or an (H, W) array."""
    cam = PinholeCamera(K, W, H, C_BODY_TO_NADIR_CAM, [x, y, z])
    if depth is not None and np.ndim(depth) == 0:
        depth = np.full((H, W), depth, np.float32)
    return CameraFrame(0, 0.0, 0, np.zeros((H, W, 3), np.uint8), depth, cam)


def box_of(frame, x=0.0, y=0.0, size=20.0):
    """A ``size`` px square box around the ground point (x, y, 0) in a frame taken from 10 m up."""
    u = 160.0 + 10.0 * (frame.cam.t[1] - y)
    v = 120.0 + 10.0 * (frame.cam.t[0] - x)
    return Detection(0, "person", 0.9, (u - size / 2, v - size / 2, u + size / 2, v + size / 2))


def new_state(**cfg):
    st = GCSState(GCSConfig(**cfg))
    st.setup_grid((-20.0, 20.0, -20.0, 20.0), 0.5)
    return st


def det(x, y=0.0, conf=0.5, z=0.0):
    return {"cls": "person", "xyz": [x, y, z], "conf": conf}


def brute_force(xyz, conf):
    """Confidence-weighted mean (3,) and RMS horizontal distance from it of detections (N, 3) [m]."""
    xyz, conf = np.asarray(xyz, dtype=np.float64), np.asarray(conf, dtype=np.float64)
    mean = (conf[:, None] * xyz).sum(axis=0) / conf.sum()
    d2 = ((xyz[:, :2] - mean[:2]) ** 2).sum(axis=1)
    return mean, float(np.sqrt((conf * d2).sum() / conf.sum()))


class TestSpread(unittest.TestCase):
    def test_two_and_three_sightings_by_hand(self):
        tr = ObjectTracker(merge_radius=2.0)
        tr.update([det(0.0)])
        self.assertEqual(tr.objects[0].spread, 0.0)
        tr.update([det(1.0)])
        # mean 0.5, both 0.5 m from it
        self.assertAlmostEqual(tr.objects[0].spread, 0.5, places=12)
        tr.update([det(0.5, 1.5, conf=1.0)])
        # weights 0.5, 0.5, 1: mean (0.5, 0.75); squared distances 0.8125, 0.8125, 0.5625 -> sqrt(1.375 / 2)
        np.testing.assert_allclose(tr.objects[0].pos, [0.5, 0.75, 0.0], atol=1e-12)
        self.assertAlmostEqual(tr.objects[0].spread, float(np.sqrt(0.6875)), places=12)

    def test_height_does_not_count(self):
        tr = ObjectTracker(merge_radius=2.0)
        tr.update([det(0.0, z=0.0)])
        tr.update([det(0.0, z=3.0)])
        self.assertEqual(tr.objects[0].spread, 0.0)

    def test_random_sequences_without_merges(self):
        centres = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [0.0, 10.0, -1.0]])  # far apart: one object each
        for seed in range(5):
            rng = np.random.default_rng(seed)
            tr = ObjectTracker(merge_radius=2.0)
            seen = [[] for _ in centres]
            for _ in range(20):
                batch = []
                for k in rng.permutation(len(centres))[: rng.integers(1, 4)]:
                    xyz, conf = centres[k] + rng.normal(0.0, 0.25, 3), float(rng.uniform(0.2, 1.0))
                    seen[k].append((xyz, conf))
                    batch.append({"cls": "person", "xyz": xyz, "conf": conf})
                tr.update(batch)
            self.assertEqual(len(tr.objects), len(centres))
            for ob in tr.objects:
                k = int(np.argmin(np.linalg.norm(centres - ob.pos, axis=1)))
                mean, spread = brute_force([s[0] for s in seen[k]], [s[1] for s in seen[k]])
                self.assertEqual(ob.hits, len(seen[k]))
                np.testing.assert_allclose(ob.pos, mean, atol=1e-9)
                self.assertAlmostEqual(ob.spread, spread, places=9, msg=f"seed {seed}")
                self.assertGreater(spread, 0.1)

    def test_random_sequences_with_merges(self):
        # two sightings 2.2 m apart start two objects (radius 2 m); the sightings that follow fall between them
        # and pull both together until they merge. From then on a second box in a frame starts a new object
        # that is merged at once. In the end one object holds every sighting.
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            tr = ObjectTracker(merge_radius=2.0)
            start = [np.array([-1.1, 0.0, 0.2]), np.array([1.1, 0.0, -0.2])]
            xyz, conf = list(start), [0.6, 0.4]
            tr.update([{"cls": "person", "xyz": start[0], "conf": conf[0]}])
            tr.update([{"cls": "person", "xyz": start[1], "conf": conf[1]}])
            self.assertEqual(len(tr.objects), 2)
            merges = 0
            for _ in range(30):
                batch = []
                for _ in range(rng.integers(1, 4)):
                    xyz.append(rng.normal(0.0, 0.25, 3))
                    conf.append(float(rng.uniform(0.2, 1.0)))
                    batch.append({"cls": "person", "xyz": xyz[-1], "conf": conf[-1]})
                before = len(tr.objects)
                tr.update(batch)
                # one object before and after: every box but the first started an object that was merged
                if before == 1 and len(tr.objects) == 1:
                    merges += len(batch) - 1
            self.assertEqual(len(tr.objects), 1, f"seed {seed}")
            self.assertGreater(merges, 5, f"seed {seed}")
            ob = tr.objects[0]
            mean, spread = brute_force(xyz, conf)
            self.assertEqual(ob.hits, len(xyz))
            np.testing.assert_allclose(ob.pos, mean, atol=1e-9)
            self.assertAlmostEqual(ob.spread, spread, places=9, msg=f"seed {seed}")
            self.assertGreater(spread, 0.3)


class TestTrackerViews(unittest.TestCase):
    def test_count_views_and_detection_rate(self):
        tr = ObjectTracker(merge_radius=2.0)
        tr.update([det(0.0), det(10.0)])
        a, b = tr.objects
        self.assertEqual((a.views, a.detection_rate), (0, 1.0))  # nothing counted yet: no evidence against it
        tr.count_views([a.id, b.id])
        tr.count_views([a.id])
        tr.count_views(iter([a.id, a.id, 99]))  # one frame is one view; an unknown id is ignored
        self.assertEqual((a.hits, a.views, b.hits, b.views), (1, 3, 1, 1))
        self.assertAlmostEqual(a.detection_rate, 1 / 3)
        self.assertEqual(b.detection_rate, 1.0)
        d = a.to_dict()
        self.assertEqual((d["views"], d["detection_rate"]), (3, 0.333))

    def test_merge_keeps_the_larger_view_count(self):
        tr = ObjectTracker(merge_radius=2.0)
        tr.update([det(0.0, conf=0.05)])
        tr.update([det(2.1)])
        a, b = tr.objects
        a.views, b.views = 5, 3
        touched = tr.update([det(1.04, conf=1.0)])  # nearer to a (1.04 m) than to b (1.06 m): a moves to 0.99
        self.assertEqual(len(tr.objects), 2)
        self.assertIs(touched[0], a)
        tr.update([det(1.5, conf=1.0)])  # a again (0.51 m against 0.6 m): a moves to 1.24, within 1 m of b
        self.assertEqual(tr.objects, [a])
        self.assertEqual((a.hits, a.views), (4, 5))

    def test_merged_away_object_is_reported_as_the_one_that_took_it_in(self):
        tr = ObjectTracker(merge_radius=2.0)
        tr.update([det(0.0)])
        tr.update([det(2.1, conf=0.05)])
        a, b = tr.objects
        a.views, b.views = 2, 6
        self.assertIs(tr.update([det(1.06, conf=1.0)])[0], b)  # nearer to b: b moves to 1.11
        touched = tr.update([det(0.6, conf=1.0)])  # b again (0.51 m against 0.6 m): b moves to 0.86 and merges
        self.assertEqual(tr.objects, [a])
        self.assertEqual(len(touched), 1)
        self.assertIs(touched[0], a)
        self.assertEqual((a.hits, a.views), (4, 6))

    def test_views_never_fall_below_the_hits(self):
        # two boxes of one frame on the same object: the second starts an object that is merged at once
        tr = ObjectTracker(merge_radius=2.0)
        tr.count_views([ob.id for ob in tr.update([det(0.0)])])
        touched = tr.update([det(0.1), det(0.3)])
        self.assertEqual(len(tr.objects), 1)
        ob = tr.objects[0]
        self.assertEqual({t.id for t in touched}, {ob.id})
        self.assertEqual(ob.detection_rate, 1.0)  # also before the frame is counted
        tr.count_views([t.id for t in touched])
        self.assertEqual((ob.hits, ob.views, ob.detection_rate), (3, 3, 1.0))


class TestMinRate(unittest.TestCase):
    def tracker(self, **kw):
        """Three objects in view of 5 frames: detected in 1, 2 and 3 of them; a fourth in 2 of 4."""
        tr = ObjectTracker(merge_radius=2.0, min_hits=2, **kw)
        for k in range(5):
            batch = [det(20.0)] if k < 3 else []
            batch += [det(10.0)] if k < 2 else []
            batch += [det(0.0)] if k < 1 else []
            batch += [det(30.0)] if k in (1, 2) else []
            tr.update(batch)
            tr.count_views([1, 2, 3] + ([4] if k >= 1 else []))
        return tr

    def test_counts(self):
        tr = self.tracker()
        self.assertEqual(
            [(round(o.pos[0]), o.hits, o.views) for o in tr.objects], [(20, 3, 5), (10, 2, 5), (0, 1, 5), (30, 2, 4)]
        )

    def test_default_is_hits_only(self):
        tr = self.tracker()
        self.assertEqual(tr.min_rate, 0.0)
        self.assertEqual(tr.confirmed(), [o for o in tr.objects if o.hits >= 2])
        self.assertEqual([round(o.pos[0]) for o in tr.confirmed()], [20, 10, 30])

    def test_min_rate_drops_the_rarely_detected(self):
        tr = self.tracker(min_rate=0.5)
        # 3 of 5 = 0.6 passes, 2 of 5 = 0.4 fails, 2 of 4 = 0.5 is just enough, 1 of 5 has too few hits anyway
        self.assertEqual([round(o.pos[0]) for o in tr.confirmed()], [20, 30])
        self.assertEqual([tr.is_confirmed(o) for o in tr.objects], [True, False, False, True])
        tr.min_rate = 0.1  # the hits still count
        self.assertEqual([round(o.pos[0]) for o in tr.confirmed()], [20, 10, 30])

    def test_config_reaches_the_tracker(self):
        self.assertEqual(GCSState().tracker.min_rate, 0.0)
        self.assertEqual(GCSState(GCSConfig(min_rate=0.6)).tracker.min_rate, 0.6)


class TestViewsOfAFrame(unittest.TestCase):
    """GCSState.apply_detections: which frames count as a view of a tracked object (a person at the origin)."""

    def setUp(self):
        self.st = new_state()
        frame = frame_at(0.0, 0.0)
        self.st.apply_detections(frame, [box_of(frame)], 0.0)
        self.ob = self.st.tracker.objects[0]

    def look(self, frame, detections=()):
        self.st.apply_detections(frame, list(detections), 0.0)
        return self.ob.hits, self.ob.views

    def test_first_sighting_is_a_view(self):
        np.testing.assert_allclose(self.ob.pos, [0.0, 0.0, 0.0], atol=1e-6)
        self.assertEqual((self.ob.hits, self.ob.views), (1, 1))

    def test_in_view_and_missed(self):
        self.assertEqual(self.look(frame_at(0.0, 0.0)), (1, 2))
        self.assertEqual(self.look(frame_at(3.0, -5.0)), (1, 3))
        self.assertEqual(self.look(frame_at(0.0, 0.0, depth=None)), (1, 4))  # a frame without depth
        self.assertEqual(self.ob.detection_rate, 0.25)

    def test_in_view_and_detected(self):
        frame = frame_at(2.0, 1.0)
        self.assertEqual(self.look(frame, [box_of(frame)]), (2, 2))  # one view, not two

    def test_outside_the_image(self):
        self.assertEqual(self.look(frame_at(0.0, 17.0)), (1, 1))  # 170 px right of the centre: past the edge
        self.assertEqual(self.look(frame_at(-13.0, 0.0)), (1, 1))  # 130 px above the centre
        self.assertEqual(self.st.objects_in_view(frame_at(0.0, 17.0)), set())

    def test_behind_the_camera(self):
        self.assertEqual(self.look(frame_at(0.0, 0.0, z=-5.0, depth=None)), (1, 1))

    def test_at_the_border(self):
        # the last box is 20 px wide and the margin 3 px: the centre may come within 13 px of the border
        self.assertEqual(self.look(frame_at(0.0, 14.0)), (1, 2))  # centre at u = 300: box up to 310 <= 317
        self.assertEqual(self.look(frame_at(0.0, 15.0)), (1, 2))  # centre at u = 310, in the image: box up to 320
        self.assertEqual(self.look(frame_at(0.0, -15.0)), (1, 2))  # left border
        self.assertEqual(self.look(frame_at(11.0, 0.0)), (1, 2))  # centre at v = 230: box down to 240 > 237
        self.assertEqual(self.look(frame_at(-11.0, 0.0)), (1, 2))  # top border
        self.assertEqual(self.look(frame_at(10.0, 0.0)), (1, 3))  # centre at v = 220: box down to 230 <= 237

    def test_border_margin_is_the_configured_one(self):
        frame = frame_at(0.0, 14.9)  # box up to u = 319
        self.assertEqual(self.st.objects_in_view(frame), set())
        self.st.cfg.edge_margin = 0.0
        self.assertEqual(self.st.objects_in_view(frame), {self.ob.id})

    def test_hidden_behind_nearer_depth(self):
        self.assertEqual(self.look(frame_at(0.0, 0.0, depth=6.0)), (1, 1))  # a tree crown 4 m above it
        self.assertEqual(self.look(frame_at(0.0, 0.0, depth=8.9)), (1, 1))  # 1.1 m above it
        self.assertEqual(self.look(frame_at(0.0, 0.0, depth=9.1)), (1, 2))  # 0.9 m: the object itself
        self.assertEqual(self.look(frame_at(0.0, 0.0, depth=12.0)), (1, 3))  # farther than expected hides nothing

    def test_only_the_depth_at_its_pixel_counts(self):
        depth = np.full((H, W), 10.0, np.float32)
        depth[:60, :80] = 6.0  # a crown in the corner of the image
        self.assertEqual(self.look(frame_at(0.0, 0.0, depth=depth)), (1, 2))
        depth = np.full((H, W), 10.0, np.float32)
        depth[110:130, 150:170] = 6.0  # a crown over the object (pixel 160, 120)
        self.assertEqual(self.look(frame_at(0.0, 0.0, depth=depth)), (1, 2))
        self.assertEqual(self.look(frame_at(2.0, 2.0, depth=depth)), (1, 3))  # from there it is at pixel 180, 140

    def test_invalid_depth_hides_nothing(self):
        for k, bad in enumerate((0.0, -1.0, np.inf, np.nan)):
            self.assertEqual(self.look(frame_at(0.0, 0.0, depth=bad)), (1, 2 + k), bad)

    def test_a_hit_counts_even_when_the_object_was_not_expected_in_view(self):
        # hidden by the depth at its pixel, yet detected (the box geolocates through the pixels next to it)
        depth = np.full((H, W), 10.0, np.float32)
        depth[119:122, 159:162] = 6.0
        frame = frame_at(0.0, 0.0, depth=depth)
        self.assertEqual(self.st.objects_in_view(frame), set())
        self.assertEqual(self.look(frame, [box_of(frame)]), (2, 2))
        # expected box across the border (the last box was 20 px), detected with a smaller box inside the margin
        frame = frame_at(0.0, 15.0)
        self.assertEqual(self.st.objects_in_view(frame), set())
        self.assertEqual(self.look(frame, [box_of(frame, size=8.0)]), (3, 3))

    def test_a_box_at_the_border_is_neither_a_hit_nor_a_view(self):
        frame = frame_at(0.0, 15.0)
        self.assertEqual(self.look(frame, [box_of(frame)]), (1, 1))
        self.assertTrue(frame.detections[0][1]["truncated"])

    def test_two_boxes_on_one_object_in_one_frame(self):
        frame = frame_at(0.0, 0.0)
        self.look(frame, [box_of(frame), box_of(frame, 0.2, 0.0)])
        self.assertEqual(len(self.st.tracker.objects), 1)
        self.assertEqual((self.ob.hits, self.ob.views, self.ob.detection_rate), (3, 3, 1.0))

    def test_every_object_is_judged_on_its_own(self):
        frame = frame_at(0.0, 0.0)
        self.look(frame, [box_of(frame, 5.0, 5.0)])  # a second person, both in view
        other = self.st.tracker.objects[1]
        self.assertEqual((self.ob.hits, self.ob.views, other.hits, other.views), (1, 2, 1, 1))
        self.look(frame_at(5.0, 17.0))  # only the second in view (u = 280; the first is at u = 330)
        self.assertEqual((self.ob.views, other.views), (2, 2))


class TestReportAndDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Two people in view of 5 frames: object 1 detected in 2 of them, object 2 in 3."""
        st = new_state(min_rate=0.5)
        for k in range(5):
            frame = frame_at(0.0, 0.0)
            dets = [box_of(frame, -4.0, 0.0)] if k < 2 else []
            dets += [box_of(frame, 4.0, 0.0)] if k < 3 else []
            st.apply_detections(frame, dets, 0.0)
        cls.state = st

    def tearDown(self):
        self.state.tracker.min_rate = 0.5

    def test_report_lists_views_rate_and_min_rate(self):
        tmp = Path(short_tmpdir("dgcs_views"))
        try:
            summary = write_report(self.state, tmp / "run", map_size=300)
            with open(tmp / "run" / "detections.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
            got = [(r["x"], r["hits"], r["views"], r["detection_rate"], r["confirmed"]) for r in rows]
            self.assertEqual(got, [("-4.0", "2", "5", "0.4", "False"), ("4.0", "3", "5", "0.6", "True")])
            objects = json.loads((tmp / "run" / "detections.json").read_text())["objects"]
            self.assertEqual(
                [(o["views"], o["detection_rate"], o["confirmed"]) for o in objects], [(5, 0.4, False), (5, 0.6, True)]
            )
            d = summary["detections"]
            self.assertEqual((d["confirmed"], d["tentative"], d["min_hits"], d["min_rate"]), (1, 1, 2, 0.5))
            self.assertEqual(summary["scorecard"]["people_found"], 1)
            self.state.tracker.min_rate = 0.0
            d = write_report(self.state, tmp / "run0", map_size=300)["detections"]
            self.assertEqual((d["confirmed"], d["tentative"], d["min_rate"]), (2, 0, 0.0))
        finally:
            rmtree(tmp)

    def test_map_draws_a_pin_below_the_rate_as_tentative(self):
        strict = render_report_map(self.state, "elevation", 300)
        self.state.tracker.min_rate = 0.0
        loose = render_report_map(self.state, "elevation", 300)
        self.state.tracker.min_rate = 0.4
        same = render_report_map(self.state, "elevation", 300)
        self.assertTrue(np.any(strict != loose))
        np.testing.assert_array_equal(loose, same)  # 2 of 5 = 0.4 is enough for a minimum of 0.4

    def object_list(self, dash):
        """(id, text colour) of the rows of the dashboard's object list, top to bottom."""
        ui = {"layer": "elevation", "stat": "mean", "active_drone": 0, "detector_on": True, "show_gt": False}
        id_column = dash.r_obj[0] + 10
        with mock.patch.object(dash.text, "put", wraps=dash.text.put) as put:
            dash.render(self.state, ui)
        drawn = [c.args for c in put.call_args_list]  # (image, text, x, y, size, colour, ...)
        return [(a[1], a[5]) for a in drawn if a[2] == id_column and a[1].isdigit()]

    def test_object_list_shows_a_pin_below_the_rate_as_tentative(self):
        dash = Dashboard((1600, 900))
        # confirmed first, in the bright colour; the pin below the rate after it, dimmed
        self.assertEqual(self.object_list(dash), [("2", PAL["fg"]), ("1", PAL["dim"])])
        self.state.tracker.min_rate = 0.0
        self.assertEqual(self.object_list(dash), [("1", PAL["fg"]), ("2", PAL["fg"])])


class TestCommandLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("gcs_script", ROOT / "scripts" / "gcs.py")
        cls.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script)

    def config(self, *args):
        """The GCSConfig that ``gcs.py`` builds from ``args`` (no station is started)."""
        with mock.patch.object(self.script, "GCSApp") as app, mock.patch.object(self.script.logging, "basicConfig"):
            self.script.main(list(args))
        return app.call_args.args[0].gcs

    def test_defaults(self):
        cfg = self.config()
        self.assertEqual((cfg.min_rate, cfg.surface_map), (0.0, True))

    def test_min_rate_and_no_surface_map(self):
        cfg = self.config("--min-rate", "0.4", "--no-surface-map")
        self.assertEqual((cfg.min_rate, cfg.surface_map), (0.4, False))

    def test_min_rate_outside_0_to_1_is_refused(self):
        for bad in ("1.5", "-0.1"):
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as stop:
                self.config("--min-rate", bad)
            self.assertEqual(stop.exception.code, 2)
            self.assertIn("--min-rate", err.getvalue())


if __name__ == "__main__":
    unittest.main()
