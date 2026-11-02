import math
import unittest

import numpy as np

from drone_gcs.geometry import C_BODY_TO_NADIR_CAM, PinholeCamera, intrinsics_from_fov, quat_from_euler, quat_to_matrix
from drone_gcs.mapping import (
    ElevationMap,
    Geolocator,
    GridSpec,
    ObjectTracker,
    Orthomosaic,
    evaluate_detections,
    intersect_rays_with_map,
    path_length,
    terrain_error,
)


def nadir_cam(x, y, z, yaw=0.0, roll=0.0, pitch=0.0, w=320, h=240, hfov=80.0):
    R = quat_to_matrix(quat_from_euler(roll, pitch, yaw)) @ C_BODY_TO_NADIR_CAM
    return PinholeCamera(intrinsics_from_fov(w, h, hfov), w, h, R, [x, y, z])


class TestGrid(unittest.TestCase):
    def test_indices_and_area_mask(self):
        g = GridSpec.from_bounds(-1, 1, 0, 2, 0.5)
        self.assertEqual(g.shape, (4, 4))
        ix, iy, ok = g.cell_index(np.array([-1.0, 0.99, 1.0]), np.array([0.0, 1.99, 0.5]))
        np.testing.assert_array_equal(ix, [0, 3, 4])
        np.testing.assert_array_equal(ok, [True, True, False])
        m = g.area_mask((-0.5, 0.5, 0.0, 1.0))
        self.assertEqual(int(m.sum()), 4)
        self.assertEqual(g.subdivided(2).shape, (8, 8))
        self.assertEqual(GridSpec.from_dict(g.to_dict()), g)


class TestMissionGrid(unittest.TestCase):
    def test_cell_centres_on_the_area_lattice_for_any_area_size(self):
        # the default padding (15 % of the larger side) is rounded up to whole cells
        from drone_gcs.state import GCSState

        for side in (33.0, 34.0, 36.0, 37.0, 40.0, 50.0):
            for res, padding in ((0.25, None), (0.25, 5.1), (0.5, None)):
                st = GCSState()
                st.setup_grid((-side / 2, side / 2, -side / 2 + 3, side / 2 + 3), res, padding)
                xs, ys = st.grid.cell_centers()
                for c, edge in ((xs, -side / 2), (ys, -side / 2 + 3)):
                    k = (c - edge) / res
                    np.testing.assert_allclose(k, np.round(k), atol=1e-6, err_msg=f"{side} m, {res} m, pad {padding}")
                self.assertGreaterEqual(-side / 2 - st.grid.x_min, (padding or 4.0) - 1e-9)

    def test_huge_mission_gets_a_coarser_grid(self):
        from drone_gcs.state import GCSConfig, GCSState

        st = GCSState(GCSConfig(max_grid_cells=1_000_000))
        st.setup_grid((0, 1000, 0, 1000), 0.25)  # ~5000^2 cells at 0.25 m
        self.assertLessEqual(st.grid.size, 1_000_000)
        k = st.grid.resolution / 0.25
        self.assertAlmostEqual(k, round(k))  # a whole multiple: the requested lattice is kept
        self.assertTrue(any("too large" in e[1] for e in st.events))
        st2 = GCSState()
        st2.setup_grid((-20, 20, -20, 20), 0.25)
        self.assertEqual(st2.grid.resolution, 0.25)


class TestElevation(unittest.TestCase):
    def test_running_stats_match_bruteforce(self):
        g = GridSpec.from_bounds(0, 4, 0, 4, 1.0)
        em = ElevationMap(g)
        rng = np.random.default_rng(0)
        pts_all = []
        for _ in range(5):
            pts = np.column_stack([rng.uniform(0, 4, 200), rng.uniform(0, 4, 200), rng.normal(1, 0.3, 200)])
            pts_all.append(pts)
            em.add_points(pts.astype(np.float32))
        P = np.concatenate(pts_all).astype(np.float32).astype(np.float64)
        ix, iy, _ = g.cell_index(P[:, 0], P[:, 1])
        for cy in range(4):
            for cx in range(4):
                z = P[(ix == cx) & (iy == cy), 2]
                self.assertEqual(em.count[cy, cx], len(z))
                self.assertAlmostEqual(float(em.mean()[cy, cx]), float(z.mean()), places=5)
                self.assertAlmostEqual(float(em.max()[cy, cx]), float(z.max()), places=5)
                self.assertAlmostEqual(float(em.min()[cy, cx]), float(z.min()), places=5)
                self.assertAlmostEqual(float(em.std()[cy, cx]), float(z.std()), places=4)

    def test_invalid_points_ignored_and_coverage(self):
        g = GridSpec.from_bounds(0, 2, 0, 2, 1.0)
        em = ElevationMap(g)
        pts = np.array(
            [[0.5, 0.5, 1], [np.inf, np.inf, np.inf], [np.nan, 0.5, 1], [5, 5, 1], [1.5, 0.5, 2]], np.float32
        )
        n = em.add_points(pts, valid=[True, True, True, True, False])
        self.assertEqual(n, 1)
        self.assertAlmostEqual(em.coverage(), 0.25)
        self.assertTrue(math.isnan(em.mean()[0, 1]))
        self.assertAlmostEqual(em.height_at(0.5, 0.5), 1.0)
        self.assertAlmostEqual(em.coverage(np.array([[True, False], [False, False]])), 1.0)


class TestMosaicAndGeolocation(unittest.TestCase):
    def setUp(self):
        self.grid = GridSpec.from_bounds(-10, 10, -10, 10, 0.1)

    def checker(self, cam, z_ground=0.0):
        """Render a flat checkerboard (1 m squares, red/blue) seen by ``cam`` (RGB + depth)."""
        u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
        rays = cam.rays(u.ravel(), v.ravel())
        t = (z_ground - cam.t[2]) / rays[:, 2]
        P = cam.t + t[:, None] * rays
        red = (np.floor(P[:, 0]) + np.floor(P[:, 1])) % 2 == 0
        rgb = np.where(red[:, None], [200, 20, 20], [20, 20, 200]).astype(np.uint8).reshape(cam.height, cam.width, 3)
        return rgb, t.reshape(cam.height, cam.width).astype(np.float32)

    def test_mosaic_from_depth_and_from_map(self):
        cam = nadir_cam(0.3, -0.2, 8.0, yaw=0.4, roll=0.05, pitch=-0.04)
        rgb, depth = self.checker(cam)
        for use_depth in (True, False):
            mos = Orthomosaic(self.grid, "nadir")
            em = ElevationMap(self.grid)
            n, P = mos.add_camera_frame(cam, rgb, depth if use_depth else None, em, ground_z=0.0)
            self.assertGreater(n, 1000)
            xs, ys = self.grid.cell_centers()
            X, Y = np.meshgrid(xs, ys)
            red_truth = (np.floor(X) + np.floor(Y)) % 2 == 0
            # ignore cells within 15 cm of a square border (sub-cell aliasing)
            far = (np.abs(X - np.round(X)) > 0.15) & (np.abs(Y - np.round(Y)) > 0.15) & mos.filled
            red_est = mos.rgb[..., 0] > 100
            agree = (red_est == red_truth)[far].mean()
            self.assertGreater(agree, 0.995, f"use_depth={use_depth}")
            np.testing.assert_allclose(P[:, 2], 0.0, atol=1e-6)

    def test_mosaic_modes(self):
        g = GridSpec.from_bounds(0, 2, 0, 1, 1.0)
        for mode in ("nadir", "latest", "blend"):
            m = Orthomosaic(g, mode)
            m.add_samples([0.5, 0.5], [0.5, 0.5], [[100, 0, 0], [200, 0, 0]], [0.9, 0.5])
            m.add_samples([0.5], [0.5], [[50, 0, 0]], [0.6])
            v = int(m.rgb[0, 0, 0])
            if mode == "nadir":
                self.assertEqual(v, 100)  # best (most top-down) sample kept
            elif mode == "latest":
                self.assertEqual(v, 50)
            else:
                self.assertTrue(50 < v < 200)
            self.assertTrue(m.filled[0, 0] and not m.filled[0, 1])

    def test_geolocation_depth_and_map(self):
        cam = nadir_cam(1.0, 2.0, 12.0, yaw=0.3, roll=0.03, pitch=0.06)
        obj = np.array([3.0, 0.5, 0.0])
        u, v, _ = cam.project([obj])
        box = (u[0] + 0.5 - 10, v[0] + 0.5 - 8, u[0] + 0.5 + 10, v[0] + 0.5 + 8)  # continuous coords, centred
        depth = np.full((cam.height, cam.width), np.nan, np.float32)
        uu, vv = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
        rays = cam.rays(uu.ravel(), vv.ravel())
        depth[:] = ((0.0 - cam.t[2]) / rays[:, 2]).reshape(cam.height, cam.width)
        geo = Geolocator(None, 0.0, "center")
        r = geo.locate(box, cam, depth)
        self.assertEqual(r["method"], "depth")
        np.testing.assert_allclose(r["xyz"], obj, atol=0.02)
        em = ElevationMap(self.grid)
        xs, ys = self.grid.cell_centers()
        X, Y = np.meshgrid(xs, ys)
        em.add_points(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]))
        r2 = Geolocator(em, 5.0, "center").locate(box, cam, None)  # wrong ground_z: the map must win
        self.assertEqual(r2["method"], "map")
        np.testing.assert_allclose(r2["xyz"], obj, atol=0.02)
        r3 = Geolocator(None, 0.0, "center").locate(box, cam, None)
        self.assertEqual(r3["method"], "plane")
        np.testing.assert_allclose(r3["xyz"], obj, atol=0.02)

    def test_depth_of_the_drone_itself_is_ignored(self):
        # a belly camera that sees its own landing gear (depth ~0.2 m) must not paint it on the ground
        cam = nadir_cam(0.0, 0.0, 8.0)
        rgb, depth = self.checker(cam)
        rgb[:40, :] = (0, 255, 0)  # "landing gear" across the top of the image, 0.2 m away
        depth[:40, :] = 0.2
        mos = Orthomosaic(self.grid, "nadir")
        n, P = mos.add_camera_frame(cam, rgb, depth, None, 0.0)
        self.assertGreater(n, 1000)
        green = (mos.rgb[..., 1] > 200) & (mos.rgb[..., 0] < 50)
        self.assertEqual(int(green.sum()), 0)
        np.testing.assert_allclose(P[:, 2], 0.0, atol=1e-6)  # only ground points come back
        # geolocation: a box inside the near patch falls back to the map / ground plane
        u, _, _ = cam.project([[0.0, 2.0, 0.0]])
        r = Geolocator(None, 0.0, "center").locate((u[0] - 5, 5, u[0] + 5, 15), cam, depth)
        self.assertEqual(r["method"], "plane")

    def test_ray_map_intersection_on_slope(self):
        g = GridSpec.from_bounds(-20, 20, -20, 20, 0.05)
        em = ElevationMap(g)
        xs, ys = g.cell_centers()
        X, Y = np.meshgrid(xs, ys)
        Z = 0.1 * X + 0.05 * Y
        em.add_points(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        cam = nadir_cam(0, 0, 10.0, pitch=0.2)
        P, ok = intersect_rays_with_map(cam.t, cam.rays([50, 160, 300], [30, 120, 200]), em, 0.0)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(P[:, 2], 0.1 * P[:, 0] + 0.05 * P[:, 1], atol=0.03)


class TestTracker(unittest.TestCase):
    def test_merge_same_class_within_radius(self):
        tr = ObjectTracker(merge_radius=1.5, min_hits=2)
        rng = np.random.default_rng(0)
        for k in range(10):
            dets = [
                {"cls": "car", "xyz": np.array([5, 5, 0]) + rng.normal(0, 0.2, 3), "conf": 0.8, "t": k},
                {"cls": "car", "xyz": np.array([5, 8, 0]) + rng.normal(0, 0.2, 3), "conf": 0.7, "t": k},  # 3 m away
                {"cls": "person", "xyz": np.array([5.3, 5.2, 0]), "conf": 0.9, "t": k},  # other class, same place
            ]
            tr.update(dets)
        self.assertEqual(len(tr.objects), 3)
        cars = sorted([o for o in tr.objects if o.cls == "car"], key=lambda o: o.pos[1])
        np.testing.assert_allclose(cars[0].pos[:2], [5, 5], atol=0.2)
        self.assertEqual(cars[0].hits, 10)
        self.assertEqual((cars[0].first_seen, cars[0].last_seen), (0, 9))
        self.assertEqual(len(tr.confirmed()), 3)

    def test_two_objects_in_one_frame_are_not_merged(self):
        tr = ObjectTracker(merge_radius=2.0)
        tr.update(
            [{"cls": "person", "xyz": [0, 0, 0], "conf": 0.9}, {"cls": "person", "xyz": [1.2, 0, 0], "conf": 0.9}]
        )
        tr.update(
            [{"cls": "person", "xyz": [0.1, 0, 0], "conf": 0.9}, {"cls": "person", "xyz": [1.1, 0, 0], "conf": 0.9}]
        )
        self.assertEqual(len(tr.objects), 2)


class TestEvaluation(unittest.TestCase):
    def test_precision_recall_error(self):
        preds = [
            {"class": "car", "x": 1.1, "y": 0, "z": 0},
            {"class": "car", "x": 9, "y": 9, "z": 0},  # false positive
            {"class": "tent", "x": 5, "y": 5.2, "z": 1},
        ]
        gt = [
            {"class": "car", "pos": [1, 0, 0]},
            {"class": "tent", "pos": [5, 5, 0]},
            {"class": "person", "pos": [0, 9, 0]},
        ]
        e = evaluate_detections(preds, gt, 2.0)
        self.assertEqual((e["tp"], e["fp"], e["fn"]), (2, 1, 1))
        self.assertAlmostEqual(e["precision"], 2 / 3)
        self.assertAlmostEqual(e["recall"], 2 / 3)
        self.assertAlmostEqual(e["mean_err_xy_m"], 0.15)
        self.assertEqual(path_length([[0, 0], [3, 4], [3, 5]]), 6.0)


class TestTerrainError(unittest.TestCase):
    def test_errors_and_steps(self):
        truth = np.zeros((20, 20))
        truth[:, 10:] = 2.0  # a 2 m wall between columns 9 and 10
        est = truth + 0.01
        est[:, 9] = 1.0  # the cells next to the wall straddle it
        est[0, 0] = np.nan  # not mapped
        mask = np.ones_like(truth, bool)
        mask[-1] = False  # outside the area
        e = terrain_error(est, truth, mask)
        self.assertEqual(e["all"]["cells"], 19 * 20 - 1)
        self.assertGreater(e["all"]["rmse_m"], 0.1)
        # away from the wall (columns 9 and 10 are next to the step) only the 1 cm offset is left
        self.assertAlmostEqual(e["smooth"]["rmse_m"], 0.01, places=6)
        self.assertAlmostEqual(e["smooth"]["bias_m"], 0.01, places=6)
        self.assertEqual(e["smooth"]["cells"], 19 * 18 - 1)


if __name__ == "__main__":
    unittest.main()
