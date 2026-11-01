"""Whole pipeline: fake drone process -> UNIX socket -> headless GCS process -> report.

Checks: coverage grows monotonically to ~100 %, elevation map vs synthetic truth, geolocation of every
synthetic object (colour-blob stand-in detector), video written. ~20-30 s. Skip with DRONE_GCS_SKIP_E2E=1.
"""

import csv
import itertools
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_gcs.mapping import GridSpec
from drone_gcs.sim.fake_world import FakeWorld

from ._util import ROOT, rmtree, short_tmpdir

AREA = 24.0
SEED = 3


@unittest.skipIf(os.environ.get("DRONE_GCS_SKIP_E2E"), "DRONE_GCS_SKIP_E2E set")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = short_tmpdir("dgcs_e2e")
        addr = f"unix:{cls.tmp}/gcs.sock"
        gcs_cmd = [sys.executable, str(ROOT / "scripts" / "gcs.py"), "--addr", addr, "--headless"]
        gcs_cmd += ["--detector", "blob", "--out-dir", cls.tmp, "--name", "e2e", "--fps", "10"]
        gcs_cmd += ["--max-time", "150", "--linger", "1"]
        drone_cmd = [sys.executable, str(ROOT / "scripts" / "fake_drone.py"), "--addr", addr]
        drone_cmd += ["--area", str(AREA), str(AREA), "--seed", str(SEED), "--speedup", "8", "--wait-gcs", "20"]
        with open(Path(cls.tmp) / "gcs.log", "w") as log:
            gcs = subprocess.Popen(gcs_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                cls.drone = subprocess.run(drone_cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
                gcs.wait(timeout=120)
            finally:
                if gcs.poll() is None:
                    gcs.kill()
        cls.gcs_rc = gcs.returncode
        cls.run_dir = Path(cls.tmp) / "e2e"
        cls.summary = (
            json.loads((cls.run_dir / "summary.json").read_text()) if (cls.run_dir / "summary.json").exists() else None
        )

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("DRONE_GCS_KEEP_E2E"):
            print(f"\n[e2e outputs kept in {cls.run_dir}]")
        else:
            rmtree(cls.tmp)

    def test_processes_and_files(self):
        self.assertEqual(self.drone.returncode, 0, self.drone.stderr[-1500:])
        self.assertEqual(self.gcs_rc, 0, (Path(self.tmp) / "gcs.log").read_text()[-2000:])
        self.assertIsNotNone(self.summary)
        self.assertEqual(self.summary["end_reason"], "complete")
        for f in ("final_map_elevation.png", "final_map_photo.png", "coverage.png", "detections.csv", "detections.json",
                  "trajectory.csv", "summary.json", "dashboard.mp4", "coverage_timeline.csv"):  # fmt: skip
            self.assertTrue((self.run_dir / f).exists(), f)
        cap = cv2.VideoCapture(str(self.run_dir / "dashboard.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (900, 1600, 3))
        self.assertGreater(n, 50)

    def test_coverage_monotonic_to_full(self):
        with open(self.run_dir / "coverage_timeline.csv") as fh:
            rows = list(csv.DictReader(fh))
        cov = [float(r["elevation_coverage"]) for r in rows]
        photo = [float(r["photo_coverage"]) for r in rows]
        self.assertGreater(len(cov), 20)
        self.assertTrue(all(b >= a for a, b in itertools.pairwise(cov)), "elevation coverage decreased")
        self.assertTrue(all(b >= a for a, b in itertools.pairwise(photo)), "photo coverage decreased")
        self.assertLess(cov[len(cov) // 4], 0.6)  # it really grows during the flight
        self.assertGreater(cov[-1], 0.98)
        self.assertGreater(photo[-1], 0.98)
        self.assertGreater(self.summary["coverage"]["elevation_pct"], 98.0)

    def test_elevation_matches_synthetic_truth(self):
        g = json.loads((self.run_dir / "maps" / "grid.json").read_text())
        grid = GridSpec.from_dict(g["elevation_grid"])
        w = FakeWorld((-AREA / 2, AREA / 2, -AREA / 2, AREA / 2), seed=SEED)
        truth = w.truth_on_grid(grid)
        # cells with a height step inside (walls, tree crowns): sampled by the ray lattice on one side only
        k = 5
        fine = GridSpec(grid.x_min, grid.y_min, grid.resolution / k, grid.nx * k, grid.ny * k)
        xs, ys = fine.cell_centers()
        X, Y = np.meshgrid(xs, ys)
        hf = w.height_at(X, Y).reshape(grid.ny, k, grid.nx, k)
        step = (hf.max(axis=(1, 3)) - hf.min(axis=(1, 3))) > 0.3
        w.close()
        est = np.load(self.run_dir / "maps" / "elevation_mean.npy")
        cnt = np.load(self.run_dir / "maps" / "hit_count.npy")
        mask = grid.area_mask(g["area"]) & (cnt > 0)
        err = (est - truth)[mask]
        smooth = (est - truth)[mask & ~step]
        rmse, rmse_smooth = float(np.sqrt(np.mean(err**2))), float(np.sqrt(np.mean(smooth**2)))
        print(f"\n[e2e] elevation RMSE all cells {rmse:.3f} m, smooth cells {rmse_smooth:.4f} m, "
              f"median |err| {np.median(np.abs(err)):.4f} m, step cells {step[mask].mean() * 100:.1f} %")  # fmt: skip
        self.assertLess(rmse_smooth, 0.01)
        self.assertLess(float(np.median(np.abs(err))), 0.005)
        self.assertLess(rmse, 0.2)

    def test_geolocation_of_synthetic_objects(self):
        e = self.summary["evaluation"]
        print(
            f"\n[e2e] objects: precision {e['precision']}, recall {e['recall']}, "
            f"mean xy error {e['mean_err_xy_m']:.3f} m, max {e['max_err_xy_m']:.3f} m ({e['ground_truth']} objects)"
        )
        self.assertGreaterEqual(e["ground_truth"], 5)
        self.assertEqual(e["recall"], 1.0)
        self.assertEqual(e["precision"], 1.0)
        self.assertLess(e["mean_err_xy_m"], 0.25)
        self.assertLess(e["max_err_xy_m"], 0.6)


if __name__ == "__main__":
    unittest.main()
