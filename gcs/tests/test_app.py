"""GCS application logic without a window or sockets: keys, mission changes, detector routing, video timing."""

import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_gcs import protocol as P
from drone_gcs.app import AppConfig, GCSApp, VideoRecorder, decode_key
from drone_gcs.detector import Detection, DetectorWorker
from drone_gcs.state import GCSState

from ._util import rmtree, short_tmpdir

AREA = {"x_min": -10.0, "x_max": 10.0, "y_min": -10.0, "y_max": 10.0}


def msg(mtype, fields, arrays=None, drone_id=0):
    return P.decode_bytes(P.encode_bytes(mtype, fields, arrays, drone_id=drone_id))


def mission(mid):
    return msg(P.MISSION, {"area": AREA, "resolution": 0.5, "mission_id": mid, "name": mid})


def tel(t, mid=None, drone_id=0):
    f = {"t": t, "pos": [0.0, 0.0, 5.0], "quat": [1, 0, 0, 0]}
    if mid is not None:
        f["mission_id"] = mid
    return msg(P.TELEMETRY, f, drone_id=drone_id)


class TestKeys(unittest.TestCase):
    def test_decode_key(self):
        # Qt build (this env): arrows / Home arrive as X keysyms whose low byte is a letter
        for code in (0xFF51, 0xFF52, 0xFF53, 0xFF54, 0xFF50, 0xFFBE, 0xFFFF, 13, -1):
            self.assertIsNone(decode_key(code), hex(code))
        self.assertEqual(decode_key(0x6D), "m")
        self.assertEqual(decode_key(0x4D), "m")  # Shift / Caps Lock
        self.assertEqual(decode_key(27), "esc")
        # GTK builds put the modifier state in bits 16+ (here Num Lock)
        self.assertEqual(decode_key(0x10006D), "m")
        self.assertEqual(decode_key(0x10001B), "esc")
        self.assertIsNone(decode_key(0x10FF51))

    def test_arrow_keys_do_not_quit_or_screenshot(self):
        tmp = short_tmpdir("dgcs_app")
        try:
            app = GCSApp(AppConfig(addr=f"unix:{tmp}/k.sock", out_dir=Path(tmp), name="keys", headless=True))
            canvas = np.zeros((10, 10, 3), np.uint8)
            for code in (0xFF51, 0xFF53, 0xFF52, 0xFF54):  # Left used to quit, Right to take a screenshot
                app.handle_key(code, canvas)
            self.assertIsNone(app.quit_reason)
            self.assertEqual(app.shots, 0)
            app.handle_key(ord("m"), canvas)
            self.assertEqual(app.ui["layer"], "photo")
            app.handle_key(27, canvas)
            self.assertEqual(app.quit_reason, "user (esc)")
        finally:
            rmtree(tmp)


class TestMissionScope(unittest.TestCase):
    def test_data_and_end_of_another_mission_are_ignored(self):
        st = GCSState()
        st.apply(mission("A"))
        st.apply(tel(1.0, "A"))
        st.apply(tel(1.1))  # unstamped (older sender / raw client): accepted
        self.assertEqual(st.counts[P.TELEMETRY], 2)
        st.apply(mission("B"))  # (the app archives A first when A has data; here the state is reused)
        st.apply(tel(2.0, "A"))
        st.apply(msg(P.END, {"reason": "complete", "mission_id": "A"}))
        self.assertEqual(st.counts[P.TELEMETRY], 2)
        self.assertFalse(st.ended)
        self.assertEqual(dict(st.stale), {"A": 2})
        st.apply(msg(P.END, {"reason": "complete", "mission_id": "B"}))
        self.assertTrue(st.ended)

    def test_new_mission_clears_the_previous_end(self):
        st = GCSState()
        st.apply(mission("A"))
        st.apply(msg(P.END, {"reason": "aborted", "t": 30.0}))  # mission A ended without any data
        self.assertTrue(st.ended)
        st.apply(mission("B"))
        self.assertFalse(st.ended)
        st.apply(tel(0.5))
        self.assertAlmostEqual(st.mission_time, 0.0)  # B's clock, not A's END time
        self.assertEqual(st.end_reason, "")


class FakeWorker:
    def __init__(self, results):
        self.results = results

    def poll(self):
        out, self.results = self.results, []
        return out


class TestRollover(unittest.TestCase):
    def test_rollover_keeps_ground_truth_and_drops_stale_detections(self):
        tmp = short_tmpdir("dgcs_app")
        try:
            app = GCSApp(AppConfig(addr=f"unix:{tmp}/r.sock", out_dir=Path(tmp), name="roll", headless=True))
            st = app.state
            st.apply(mission("A"))
            st.apply(msg(P.GT_OBJECTS, {"objects": [{"id": 0, "class": "car", "pos": [1.0, 2.0, 0.0]}]}))
            hello = msg(P.HELLO, {"name": "sim"})
            hello.conn_id = 1
            st.apply(hello)
            st.apply(tel(1.0, "A"))
            K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
            fields = {"t": 1.0, "K": K.tolist(), "pos": [0, 0, 5.0], "quat": [0, 1, 0, 0], "mission_id": "A"}
            cam = msg(P.CAMERA, fields, {"rgb": np.zeros((48, 64, 3), np.uint8)})
            frame = st.apply(cam)
            self.assertIsNotNone(frame)
            app._rollover()
            self.assertTrue((Path(tmp) / "roll" / "mission1_A" / "summary.json").exists())
            new = app.state
            self.assertIsNot(new, st)
            self.assertEqual(new.gt_objects, st.gt_objects)  # latched GT stays in force
            self.assertEqual(new.senders, st.senders)
            self.assertEqual(app.gen, 1)
            new.apply(mission("B"))
            new.apply(tel(0.1, "B"))
            # a detection of mission A's frame that finishes after the rollover is not put on B's map
            det = Detection(0, "car", 0.9, (20.0, 14.0, 44.0, 34.0))
            app.worker = FakeWorker([((0, frame), [det], 5.0)])
            app._apply_detector_results()
            self.assertEqual(new.tracker.n_detections, 0)
            app.worker = FakeWorker([((1, frame), [det], 5.0)])
            app._apply_detector_results()
            self.assertEqual(new.tracker.n_detections, 1)
            app.worker = None
            app.receiver.close()
        finally:
            rmtree(tmp)


class TestDetectorWorker(unittest.TestCase):
    def test_every_drone_gets_detection(self):
        class Slow:
            def detect(self, rgb):
                time.sleep(0.02)
                return []

        wk = DetectorWorker(Slow())
        img = np.zeros((2, 2, 3), np.uint8)
        done = []
        for i in range(40):  # two drones submitting faster than the detector runs
            wk.submit(img, (0, i), key=0)
            wk.submit(img, (1, i), key=1)
            time.sleep(0.01)
            done += [r[0] for r in wk.poll()]
        t0 = time.monotonic()
        while wk.busy and time.monotonic() - t0 < 2.0:
            time.sleep(0.01)
        time.sleep(0.05)
        done += [r[0] for r in wk.poll()]
        wk.close()
        per_drone = {d: sum(1 for k, _ in done if k == d) for d in (0, 1)}
        self.assertGreaterEqual(per_drone[0], 4)
        self.assertGreaterEqual(per_drone[1], 4)
        self.assertLessEqual(abs(per_drone[0] - per_drone[1]), 1)  # served in turn
        self.assertIn((0, 39), done)  # the newest frame of each drone is processed
        self.assertIn((1, 39), done)
        self.assertGreater(wk.n_dropped, 0)


class TestVideoRecorder(unittest.TestCase):
    def test_stall_keeps_real_time(self):
        tmp = Path(short_tmpdir("dgcs_vid"))
        try:
            rec = VideoRecorder(tmp / "v.mp4", 10.0, (64, 48), transcode=False)
            img = np.zeros((48, 64, 3), np.uint8)
            rec.write(img, 100.0)
            rec.write(img, 102.0)  # a 2 s stall (e.g. archiving a mission report): repeated, not cut
            self.assertEqual(rec.frames, 21)
            rec.write(img, 132.0)  # 30 s gap (machine suspended): at most 5 s are filled in
            self.assertEqual(rec.frames, 71)
            self.assertAlmostEqual(rec.lost_s, 25.0, places=6)
            rec.write(img, 132.1)
            self.assertEqual(rec.frames, 72)
            path = rec.close()
            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertEqual(n, 72)
        finally:
            rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
