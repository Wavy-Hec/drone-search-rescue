import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from drone_gcs import protocol as P
from drone_gcs.dashboard import LAYERS, PAL, Dashboard, render_report_map
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


if __name__ == "__main__":
    unittest.main()
