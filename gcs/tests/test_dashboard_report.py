import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from drone_gcs import protocol as P
from drone_gcs.dashboard import LAYERS, PAL, Dashboard, layout_map_labels, overlap_area, pick_spot, render_report_map
from drone_gcs.detector import ColorBlobDetector
from drone_gcs.geometry import C_BODY_TO_NADIR_CAM, PinholeCamera, intrinsics_from_fov
from drone_gcs.mapping import GridSpec
from drone_gcs.report import write_report
from drone_gcs.sim.fake_world import FakeWorld
from drone_gcs.sim.flight import CarrotFlight, lawnmower
from drone_gcs.state import GCSState

from ._util import rmtree, short_tmpdir


def msg(mtype, fields, arrays=None, drone_id=0):
    """Encode + decode through the real wire format."""
    return P.decode_bytes(P.encode_bytes(mtype, fields, arrays, drone_id=drone_id))


class TestDashboardAndReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Fly half of a small survey through GCSState (no sockets) and keep the state."""
        area = (-8.0, 8.0, -8.0, 8.0)
        w = FakeWorld(area, seed=5)
        z = w.hmax + 4
        path = lawnmower(area, 4.0, z)
        st = GCSState()
        mission = {
            "area": dict(zip(("x_min", "x_max", "y_min", "y_max"), area, strict=True)),
            "resolution": 0.25,
            "mission_id": "t1",
            "name": "unit",
            "survey_altitude": z,
        }
        st.apply(msg(P.MISSION, mission, {"planned_path": path}))
        st.apply(msg(P.GT_OBJECTS, {"objects": w.gt_objects()}))
        truth_grid = GridSpec.from_bounds(-12.0, 12.0, -12.0, 12.0, 0.25)
        fields = {"x_min": truth_grid.x_min, "y_min": truth_grid.y_min, "resolution": truth_grid.resolution}
        st.apply(msg(P.GT_TERRAIN, fields, {"heights": w.truth_on_grid(truth_grid).astype(np.float32)}))
        f = CarrotFlight(path, speed=3.0)
        K = intrinsics_from_fov(320, 240, 75)
        det = ColorBlobDetector()
        cls.coverage_seen = []
        k = 0
        while f.s < 0.5 * f.length:
            f.step(0.05)
            k += 1
            tel = {
                "t": f.t,
                "pos": f.p.tolist(),
                "quat": f.quat.tolist(),
                "vel": f.v.tolist(),
                "wp_index": f.wp_index,
                "carrot": f.carrot()[0].tolist(),
                "phase": "survey",
            }
            st.apply(msg(P.TELEMETRY, tel))
            if k % 2 == 0:
                pts, valid = w.scan_grid(f.p, f.yaw)
                st.apply(msg(P.SCAN, {"t": f.t, "origin": f.p.tolist()}, {"points": pts, "valid": valid}))
                cls.coverage_seen.append(st.coverage()[0])
            if k % 4 == 0:
                c, s_ = np.cos(f.yaw), np.sin(f.yaw)
                R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ C_BODY_TO_NADIR_CAM
                cam = PinholeCamera(K, 320, 240, R, f.p)
                rgb, depth = w.render(cam)
                fields = {
                    "t": f.t,
                    "K": K.tolist(),
                    "width": 320,
                    "height": 240,
                    "pos": f.p.tolist(),
                    "quat": cam.quat.tolist(),
                    "frame_id": k,
                }
                frame = st.apply(msg(P.CAMERA, fields, {"rgb": rgb, "depth": depth}))
                st.apply_detections(frame, det.detect(frame.rgb), 1.0)
        w.close()
        cls.state = st

    def test_coverage_grows_monotonically(self):
        cov = self.coverage_seen
        self.assertTrue(all(b >= a for a, b in itertools.pairwise(cov)))
        self.assertGreater(cov[-1], 0.3)
        self.assertLess(cov[-1], 0.9)  # half the survey: part of the map is still fog

    def test_dashboard_renders_all_layers(self):
        dash = Dashboard((1600, 900))
        for layer in LAYERS:
            ui = {
                "layer": layer,
                "stat": "mean",
                "active_drone": 0,
                "detector_on": True,
                "show_gt": True,
                "link": {"connected": True, "msg_rate": 50, "byte_rate": 1e6, "address": "unix:/x"},
                "detector": {"on": True, "ms": 3.0, "fps": 5.0, "name": "blob"},
                "recording": True,
                "rec_seconds": 3.0,
            }
            img = dash.render(self.state, ui)
            self.assertEqual(img.shape, (900, 1600, 3))
            x, y, w, h = dash.r_map
            roi = img[y + 30 : y + h - 4, x + 4 : x + w - 180]
            fog = np.all(roi == np.array(PAL["fog"], np.uint8), axis=-1).mean()
            self.assertGreater(fog, 0.03, layer)  # unknown cells shown as fog
            self.assertLess(fog, 0.9, layer)
        # camera panel shows the image (not the NO VIDEO placeholder)
        x, y, w, h = dash.r_cam
        self.assertGreater(img[y + 30 : y + h - 2, x + 2 : x + w - 2].std(), 10)

    def test_landed_card_replaces_the_camera_image(self):
        dash = Dashboard((1600, 900))
        ui = {"layer": "elevation", "active_drone": 0, "detector_on": True, "link": {"connected": True}}
        x, y, w, h = dash.r_cam
        mx, my, mw, mh = dash.r_map
        d = self.state.drones[0]
        live = dash.render(self.state, ui)
        flying = d.phase, d.landed_t
        d.phase, d.landed_t = "landed", d.t
        try:
            landed = dash.render(self.state, ui)
        finally:
            d.phase, d.landed_t = flying
        card = landed[y + 30 : y + h - 2, x + 2 : x + w - 2]
        background = np.all(card == (24, 20, 20), axis=-1).mean()
        green = np.all(np.abs(card.astype(int) - PAL["green"]) < 8, axis=-1).sum()
        self.assertGreater(background, 0.85)  # the card, not the camera image
        self.assertLess(np.all(live[y + 30 : y + h - 2, x + 2 : x + w - 2] == (24, 20, 20), axis=-1).mean(), 0.5)
        self.assertGreater(green, 500)  # LANDED and the pad sign

        def waypoint_pixels(img):  # the current waypoint's yellow ring and label on the map (legend left out)
            return np.all(img[my + 27 : my + mh, mx : mx + mw - 180] == PAL["yellow"], axis=-1).sum()

        self.assertGreater(waypoint_pixels(live), 20)
        self.assertEqual(waypoint_pixels(landed), 0)

    def test_people_outside_the_survey_area_are_counted_apart(self):
        st = self.state
        saved = st.gt_objects
        outsider = {"id": "walker", "class": "person", "pos": [st.area[1] + 3.0, st.area[2] - 3.0, 0.0]}
        st.gt_objects = [*saved, outsider]
        tmp = Path(short_tmpdir("dgcs_out"))
        try:
            card = write_report(st, tmp / "run")["scorecard"]
            people = [g for g in saved if g["class"] == "person"]
            self.assertEqual(card["people_total"], len(people))
            self.assertEqual(card["people_outside_area"], {"found": 0, "total": 1})
            report = (tmp / "run" / "report.md").read_text()
            self.assertIn("in the survey area; outside it 0 / 1 |", report)
            self.assertIn("Outside it: 0 of 1 found (not found: walker).", report)
        finally:
            st.gt_objects = saved
            rmtree(tmp)

    def test_report_files(self):
        tmp = Path(short_tmpdir("dgcs_rep"))
        try:
            s = write_report(self.state, tmp / "run")
            for f in (
                "final_map_elevation.png",
                "final_map_photo.png",
                "coverage.png",
                "detections.csv",
                "detections.json",
                "trajectory.csv",
                "summary.json",
                "coverage_timeline.csv",
                "maps/elevation_mean.npy",
                "report.md",
            ):
                self.assertTrue((tmp / "run" / f).exists(), f)
            summary = json.loads((tmp / "run" / "summary.json").read_text())
            self.assertEqual(summary["mission"]["name"], "unit")
            self.assertGreater(summary["flight"]["path_length_m"], 10)
            self.assertIsNotNone(summary["evaluation"])
            card = summary["scorecard"]
            people = [g for g in self.state.gt_objects if g["class"] == "person"]
            self.assertEqual(card["people_total"], len(people))
            self.assertLessEqual(card["people_found"], card["people_total"])
            self.assertEqual(card["false_alarms"], summary["evaluation"]["fp"])
            # the elevation map against the true terrain: the rays sample the height field exactly
            self.assertLess(card["map_height_error_smooth_m"], 0.05)
            self.assertGreater(summary["terrain_error"]["all"]["cells"], 500)
            report = (tmp / "run" / "report.md").read_text()
            self.assertIn(f"| People found | **{card['people_found']} / {card['people_total']}** |", report)
            self.assertIn("![Photo map", report)
            self.assertEqual(s["coverage"]["elevation_pct"], round(100 * self.state.coverage()[0], 2))
            img = render_report_map(self.state, "photo", 600)
            self.assertEqual(img.shape, (648, 800, 3))
            rows = (tmp / "run" / "trajectory.csv").read_text().splitlines()
            self.assertEqual(rows[0].split(",")[:5], ["drone_id", "t", "x", "y", "z"])
            self.assertGreater(len(rows), 50)
        finally:
            rmtree(tmp)


class TestLabelPlacement(unittest.TestCase):
    def test_first_option_inside_the_image_and_clear(self):
        taken = [(0, 0, 20, 20)]
        options = [[(10, 10, 30, 30)], [(-5, 30, 15, 40)], [(30, 30, 50, 40)], [(60, 0, 80, 10)]]
        self.assertEqual(pick_spot(options, taken, (100, 100)), 2)  # 0 overlaps, 1 leaves the image

    def test_every_box_of_an_option_must_be_clear(self):
        taken = [(0, 20, 40, 30)]
        options = [[(0, 0, 40, 10), (0, 20, 40, 30)], [(0, 40, 40, 50), (0, 60, 40, 70)]]
        self.assertEqual(pick_spot(options, taken, (100, 100)), 1)

    def test_when_all_are_blocked_the_least_hidden_wins(self):
        taken = [(0, 0, 100, 10)]
        options = [[(0, 0, 20, 10)], [(30, 5, 50, 15)], [(-15, 50, 5, 60)]]
        # hidden: 200 px under the taken box, 100 px under it, 150 px outside the image
        self.assertEqual(pick_spot(options, taken, (100, 100)), 1)

    def test_map_labels_keep_clear_of_a_landed_drone_and_of_each_other(self):
        drone = (78, 78, 122, 122)  # the drone's ring, landed on the pad's pin at (100, 100)
        pins = [(100, 100), (104, 96), (96, 104), (108, 100)]  # a pile of finds around it
        markers = [(x - 9, y - 9, x + 9, y + 9) for x, y in pins]
        spots = layout_map_labels(pins, [(60, 12)] * len(pins), [drone, *markers], (300, 300))
        boxes = [s[3] for s in spots]
        for i, b in enumerate(boxes):
            self.assertEqual(overlap_area(b, drone), 0)
            self.assertTrue(b[0] >= 0 and b[1] >= 0 and b[2] <= 300 and b[3] <= 300, b)
            for other in boxes[i + 1 :]:
                self.assertEqual(overlap_area(b, other), 0, (b, other))

    def test_map_labels_stay_inside_the_map(self):
        right, left = layout_map_labels([(290, 50), (5, 50)], [(60, 12)] * 2, [], (300, 100))
        self.assertEqual(right[2], "rm")  # next to the right edge the label goes to the left of its pin
        self.assertLessEqual(right[3][2], 300)
        self.assertEqual(left[2], "lm")
        self.assertGreaterEqual(left[3][0], 0)


if __name__ == "__main__":
    unittest.main()
