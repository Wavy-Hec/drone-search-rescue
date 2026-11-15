"""The surface map: a second height map from the camera depth, next to the elevation map from the mapping rays,
and its files in the report. Also the seeds of the mission in the summary. ~1 s."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from drone_gcs import protocol as P
from drone_gcs.geometry import C_BODY_TO_NADIR_CAM, PinholeCamera
from drone_gcs.report import write_report
from drone_gcs.state import GCSConfig, GCSState

from ._util import rmtree, short_tmpdir

AREA = (-6.0, 6.0, -6.0, 6.0)
BOX = (1.0, 3.0, -1.0, 1.0)  # x_min, x_max, y_min, y_max of a box standing on the ground [m]
BOX_TOP = 1.0  # [m]
W, H = 160, 120
K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])


def msg(mtype, fields, arrays=None):
    """Encode + decode through the real wire format."""
    return P.decode_bytes(P.encode_bytes(mtype, fields, arrays))


def camera_above_box(z, yaw):
    """A camera looking straight down from ``z`` [m] above the centre of the box, turned by ``yaw`` [rad]."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ C_BODY_TO_NADIR_CAM
    return PinholeCamera(K, W, H, R, [0.5 * (BOX[0] + BOX[1]), 0.5 * (BOX[2] + BOX[3]), z])


def render_depth(cam):
    """Depth image [m] of flat ground at z = 0 with the box on it. The camera is straight above the centre of the
    box, so no ray meets a side face: a ray either crosses the plane of the top inside the box or goes on to the
    ground."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = cam.rays(u.ravel(), v.ravel())
    d_top = (BOX_TOP - cam.t[2]) / rays[:, 2]
    top = cam.t + d_top[:, None] * rays
    on_box = (top[:, 0] >= BOX[0]) & (top[:, 0] <= BOX[1]) & (top[:, 1] >= BOX[2]) & (top[:, 1] <= BOX[3])
    d_ground = (0.0 - cam.t[2]) / rays[:, 2]
    return np.where(on_box, d_top, d_ground).reshape(H, W).astype(np.float32)


def fly(cfg=None, depth=True, **mission):
    """A state after the MISSION, one scan of the flat ground under the whole area and two camera frames."""
    st = GCSState(cfg)
    area = dict(zip(("x_min", "x_max", "y_min", "y_max"), AREA, strict=True))
    st.apply(msg(P.MISSION, {"area": area, "resolution": 0.25, "mission_id": "s1", "name": "unit", **mission}))
    xs = np.arange(AREA[0], AREA[1] + 1e-6, 0.25)
    X, Y = np.meshgrid(xs, xs)
    ground = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(np.float32)
    st.apply(msg(P.SCAN, {"t": 0.0}, {"points": ground}))
    for k, (z, yaw) in enumerate(((8.0, 0.0), (9.0, 0.6))):
        cam = camera_above_box(z, yaw)
        fields = {"t": 1.0 + k, "K": K.tolist(), "pos": cam.t.tolist(), "quat": cam.quat.tolist(), "frame_id": k}
        arrays = {"rgb": np.full((H, W, 3), 90, np.uint8)}
        if depth:
            arrays["depth"] = render_depth(cam)
        st.apply(msg(P.CAMERA, fields, arrays))
    assert not st.errors, st.last_error
    return st


def cells_by_distance_to_box(grid):
    """Masks of the cells that lie whole on the box and of those that lie whole beside it. The cells in between
    have their centre on the edge of the box."""
    xs, ys = np.meshgrid(*grid.cell_centers())
    dx = np.maximum(BOX[0] - xs, xs - BOX[1])  # distance of the cell centre outside the box, negative inside
    dy = np.maximum(BOX[2] - ys, ys - BOX[3])
    d = np.maximum(dx, dy)
    return d < -0.5 * grid.resolution, d > 0.5 * grid.resolution


class TestSurfaceMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = fly(seed=3, terrain_seed=7)

    def test_depth_image_of_the_test_scene(self):
        depth = render_depth(camera_above_box(8.0, 0.0))
        self.assertEqual((float(depth[60, 80]), float(depth[0, 0])), (7.0, 8.0))  # box top, ground

    def test_elevation_map_stays_at_the_ground(self):
        el = self.state.elevation
        self.assertEqual(el.n_points, 49 * 49)  # the scan only: the camera frames added nothing
        self.assertEqual(float(np.nanmax(el.max())), 0.0)
        on_box, _ = cells_by_distance_to_box(self.state.grid)
        np.testing.assert_array_equal(el.max()[on_box], 0.0)

    def test_surface_map_shows_the_box(self):
        st = self.state
        self.assertEqual(st.surface.grid, st.grid)
        on_box, off_box = cells_by_distance_to_box(st.grid)
        self.assertEqual(int(on_box.sum()), 7 * 7)  # 2 m box, 0.25 m cells with their centres on its edges
        top = st.surface.max()
        self.assertTrue(np.isfinite(top[on_box]).all())
        np.testing.assert_allclose(top[on_box], BOX_TOP, atol=0.02)
        np.testing.assert_allclose(st.surface.mean()[on_box], BOX_TOP, atol=0.02)
        seen_ground = off_box & st.surface.known
        # the frame from 8 m covers 12.8 m x 9.6 m of ground: about 51 x 38 cells
        self.assertGreater(int(seen_ground.sum()), 1500)
        np.testing.assert_allclose(top[seen_ground], 0.0, atol=0.02)
        # a cell with its centre on the edge of the box holds a strip of the top as well
        edge = ~on_box & ~off_box
        self.assertEqual(int(edge.sum()), 9 * 9 - 7 * 7)
        np.testing.assert_allclose(top[edge], BOX_TOP, atol=0.02)
        # the difference of the two maps is the box and nothing else
        raised = (top - st.elevation.mean()) > 0.2
        self.assertTrue(raised[on_box].all())
        self.assertFalse(raised[off_box].any())

    def test_frames_without_depth_add_nothing(self):
        st = fly(depth=False)
        self.assertEqual(st.surface.n_points, 0)
        self.assertGreater(st.mosaic.n_frames, 0)

    def test_switched_off(self):
        st = fly(GCSConfig(surface_map=False))
        self.assertIsNone(st.surface)
        self.assertEqual(float(np.nanmax(st.elevation.max())), 0.0)

    def test_depth_elevation_option_still_feeds_the_elevation_map(self):
        st = fly(GCSConfig(depth_elevation=True))
        on_box, _ = cells_by_distance_to_box(st.grid)
        np.testing.assert_allclose(st.elevation.max()[on_box], BOX_TOP, atol=0.02)
        np.testing.assert_allclose(st.surface.max()[on_box], BOX_TOP, atol=0.02)

    def test_a_new_mission_starts_an_empty_surface_map(self):
        st = fly()
        area = dict(zip(("x_min", "x_max", "y_min", "y_max"), AREA, strict=True))
        st.apply(msg(P.MISSION, {"area": area, "resolution": 0.5, "mission_id": "s2"}))
        self.assertEqual((st.surface.n_points, st.surface.grid), (0, st.grid))


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(short_tmpdir("dgcs_surf"))

    def tearDown(self):
        rmtree(self.tmp)

    def test_surface_files(self):
        st = fly(seed=3, terrain_seed=7)
        # one more depth point, half a metre up in a ground cell: there the highest point and the mean differ
        ix, iy, _ = st.grid.cell_index(-2.0, 3.0)
        n = int(st.surface.count[iy, ix])
        self.assertGreater(n, 0)
        st.surface.add_points([[-2.0, 3.0, 0.5]])
        summary = write_report(st, self.tmp / "run", map_size=300)
        maps = self.tmp / "run" / "maps"
        top, mean, count = (np.load(maps / f"surface_{name}.npy") for name in ("max", "mean", "count"))
        for a in (top, mean, count):
            self.assertEqual(a.shape, st.grid.shape)
            self.assertEqual(a.shape, np.load(maps / "elevation_mean.npy").shape)
        self.assertEqual((top.dtype, mean.dtype, count.dtype), (np.float32, np.float32, np.int32))
        np.testing.assert_array_equal(np.isnan(top), count == 0)
        np.testing.assert_array_equal(np.isnan(mean), count == 0)
        self.assertEqual(int(count.sum()), st.surface.n_points)
        # same orientation as the elevation arrays: row = y, column = x, row 0 = south
        on_box, off_box = cells_by_distance_to_box(st.grid)
        np.testing.assert_allclose(top[on_box], BOX_TOP, atol=0.02)
        beside = off_box & (count > 0)
        beside[iy, ix] = False
        np.testing.assert_allclose(top[beside], 0.0, atol=0.02)
        self.assertEqual(int(count[iy, ix]), n + 1)
        self.assertAlmostEqual(float(top[iy, ix]), 0.5, places=6)
        self.assertAlmostEqual(float(mean[iy, ix]), 0.5 / (n + 1), places=6)
        self.assertEqual(summary["coverage"]["surface_cells"], int((count > 0).sum()))
        self.assertGreater(summary["coverage"]["surface_cells"], 1500)
        for name in ("surface_max.npy", "surface_mean.npy", "surface_count.npy"):
            self.assertIn(f"maps/{name}", summary["files"])
        self.assertIn("surface_*.npy", json.loads((maps / "grid.json").read_text())["note"])

    def test_no_surface_files_without_depth_points(self):
        for k, st in enumerate((fly(GCSConfig(surface_map=False)), fly(depth=False))):
            run = self.tmp / f"run{k}"
            summary = write_report(st, run, map_size=300)
            self.assertEqual(sorted(p.name for p in (run / "maps").glob("surface*")), [])
            self.assertEqual(summary["coverage"]["surface_cells"], 0)
            self.assertNotIn("surface", json.loads((run / "maps" / "grid.json").read_text())["note"])
            self.assertTrue((run / "maps" / "elevation_mean.npy").exists())

    def test_seeds_of_the_mission(self):
        summary = write_report(fly(seed=3, terrain_seed=7), self.tmp / "a", map_size=300)
        self.assertEqual((summary["mission"]["seed"], summary["mission"]["terrain_seed"]), (3, 7))
        saved = json.loads((self.tmp / "a" / "summary.json").read_text())["mission"]
        self.assertEqual((saved["seed"], saved["terrain_seed"]), (3, 7))
        summary = write_report(fly(seed=4), self.tmp / "b", map_size=300)  # a mission on the standard terrain
        self.assertEqual((summary["mission"]["seed"], summary["mission"]["terrain_seed"]), (4, None))
        summary = write_report(fly(), self.tmp / "c", map_size=300)  # a sender that names no seeds
        self.assertEqual((summary["mission"]["seed"], summary["mission"]["terrain_seed"]), (None, None))
        summary = write_report(GCSState(), self.tmp / "d", map_size=300)  # no MISSION at all
        self.assertEqual((summary["mission"]["seed"], summary["mission"]["terrain_seed"]), (None, None))


if __name__ == "__main__":
    unittest.main()
