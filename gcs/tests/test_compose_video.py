"""scripts/compose_video.py: frame matching by sim time and the composed video."""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import cv2
import numpy as np

from ._util import rmtree, short_tmpdir

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("compose_video", ROOT / "scripts" / "compose_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


class TestComposeVideo(unittest.TestCase):
    def test_match_takes_the_newest_dashboard_frame_not_ahead_of_the_sim(self):
        cv = load_script()
        dash_t = np.array([np.nan, np.nan, 0.1, 0.1, 0.3, 0.45, 0.45, 0.6])
        sim_t = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
        np.testing.assert_array_equal(cv.match_frames(sim_t, dash_t), [2, 3, 3, 4, 4, 6, 7])
        with self.assertRaises(ValueError):
            cv.match_frames(sim_t, np.full(3, np.nan))

    def test_composes_both_recordings_in_step(self):
        cv = load_script()
        tmp = Path(short_tmpdir("dgcs_compose"))
        try:
            sim, gcs = tmp / "sim", tmp / "gcs"
            sim.mkdir()
            gcs.mkdir()
            # simulator: 10 frames at 10 fps, t = 1.0 ... 1.9 s; brightness encodes the frame
            write_video(sim / "mission.mp4", [np.full((72, 128, 3), 20 * k, np.uint8) for k in range(10)], 10.0)
            (sim / "mission_frames.csv").write_text("frame,t\n" + "".join(f"{k},{1 + k / 10:.1f}\n" for k in range(10)))
            # dashboard: 30 frames at 12 fps; it shows sim time 1.0 + 0.05 j from frame 2 on
            write_video(gcs / "dashboard.mp4", [np.full((90, 160, 3), 8 * j, np.uint8) for j in range(30)], 12.0)
            stamps = ["", ""] + [f"{1.0 + 0.05 * j:.2f}" for j in range(28)]
            (gcs / "dashboard_frames.csv").write_text(
                "frame,sim_t\n" + "".join(f"{j},{t}\n" for j, t in enumerate(stamps))
            )
            out = tmp / "demo.mp4"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cv.main(["--gcs", str(gcs), "--sim", str(sim), "--out", str(out), "--tail", "0.5"])
            self.assertEqual(rc, 0)
            cap = cv2.VideoCapture(str(out))
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            self.assertEqual(len(frames), 10 + 5)
            self.assertEqual(frames[0].shape, (90, 160 + 160, 3))  # simulator scaled to the dashboard's height
            # frame 4: sim t = 1.4 s -> dashboard frame 10 (sim time 1.40); left half = sim frame 4
            self.assertAlmostEqual(float(frames[4][:, :160].mean()), 80.0, delta=6.0)
            self.assertAlmostEqual(float(frames[4][:, 160:].mean()), 80.0, delta=6.0)
        finally:
            rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
