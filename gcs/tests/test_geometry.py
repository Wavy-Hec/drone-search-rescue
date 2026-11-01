import math
import unittest

import numpy as np

from drone_gcs import geometry as G


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class TestRotations(unittest.TestCase):
    def test_quat_matrix_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = G.quat_normalize(rng.normal(size=4))
            q = -q if q[0] < 0 else q
            np.testing.assert_allclose(G.matrix_to_quat(G.quat_to_matrix(q)), q, atol=1e-9)

    def test_euler_and_heading(self):
        q = G.quat_from_euler(0.1, -0.2, 0.3)
        np.testing.assert_allclose(G.euler_from_quat(q), (0.1, -0.2, 0.3), atol=1e-9)
        self.assertAlmostEqual(G.compass_heading_deg(0.0), 90.0)  # facing +x = east
        self.assertAlmostEqual(G.compass_heading_deg(math.pi / 2), 0.0)  # facing +y = north
        self.assertAlmostEqual(G.compass_heading_deg(math.pi), 270.0)

    def test_xyzw_wxyz(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])  # Isaac Lab 3.0 order (x, y, z, w)
        np.testing.assert_allclose(G.quat_xyzw_to_wxyz(q), [0.9, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(G.quat_wxyz_to_xyzw(G.quat_xyzw_to_wxyz(q)), q)

    def test_matches_isaaclab_3_warp_constants(self):
        # isaaclab/utils/warp/warp_math.py (Isaac Lab 3.0): q_out = q_in * q_const, constants in (x, y, z, w):
        #   ("world", "ros"): (-0.5, 0.5, -0.5, 0.5)     ("opengl", "ros"): (1, 0, 0, 0)
        world_to_ros = G.quat_xyzw_to_wxyz(np.array([-0.5, 0.5, -0.5, 0.5]))
        opengl_to_ros = G.quat_xyzw_to_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
        rng = np.random.default_rng(3)
        for _ in range(10):
            q = G.quat_normalize(rng.normal(size=4))
            for conv, c in (("world", world_to_ros), ("opengl", opengl_to_ros)):
                expected = G.quat_to_matrix(G.quat_multiply(q, c))
                np.testing.assert_allclose(G.quat_to_matrix(G.camera_quat_to_ros(q, conv)), expected, atol=1e-12)

    def test_isaac_world_convention_to_ros_matches_isaaclab_formula(self):
        # Re-implementation (no Isaac Lab import) of the 2.x matrix formula behind
        # isaaclab.utils.math.convert_camera_frame_orientation_convention("world" -> "ros"):
        # R_gl = R_world @ R_X(pi/2) R_Y(-pi/2); R_ros = R_gl with columns 1 and 2 negated
        rng = np.random.default_rng(1)
        for _ in range(10):
            q_world = G.quat_normalize(rng.normal(size=4))
            Rw = G.quat_to_matrix(q_world)
            R_gl = Rw @ rx(math.pi / 2) @ ry(-math.pi / 2)
            R_ros = R_gl.copy()
            R_ros[:, 1:] *= -1
            np.testing.assert_allclose(G.quat_to_matrix(G.camera_quat_to_ros(q_world, "world")), R_ros, atol=1e-9)
            q_gl = G.matrix_to_quat(R_gl)
            np.testing.assert_allclose(G.quat_to_matrix(G.camera_quat_to_ros(q_gl, "opengl")), R_ros, atol=1e-9)

    def test_nadir_mount_looks_down_with_image_top_forward(self):
        R = G.quat_to_matrix(G.quat_from_euler(0, 0, 0.0)) @ G.C_BODY_TO_NADIR_CAM
        np.testing.assert_allclose(R[:, 2], [0, 0, -1])  # optical axis down
        np.testing.assert_allclose(-R[:, 1], [1, 0, 0])  # image "up" (-y_cam) = body forward
        self.assertAlmostEqual(np.linalg.det(R), 1.0)


class TestPinhole(unittest.TestCase):
    def setUp(self):
        self.K = G.intrinsics_from_fov(640, 480, 90.0)
        R = G.quat_to_matrix(G.quat_from_euler(0.05, -0.08, 0.7)) @ G.C_BODY_TO_NADIR_CAM
        self.cam = G.PinholeCamera(self.K, 640, 480, R, [3.0, -2.0, 10.0])

    def test_principal_point_is_isaac_style(self):
        self.assertAlmostEqual(self.K[0, 2], 320.0)
        self.assertAlmostEqual(self.K[0, 0], 320.0)  # 90 deg hfov

    def test_backproject_project_roundtrip(self):
        rng = np.random.default_rng(2)
        u, v = rng.uniform(0, 639, 100), rng.uniform(0, 479, 100)
        d = rng.uniform(2, 20, 100)
        P = self.cam.backproject(u, v, d)
        uu, vv, z = self.cam.project(P)
        np.testing.assert_allclose(uu, u, atol=1e-8)
        np.testing.assert_allclose(vv, v, atol=1e-8)
        np.testing.assert_allclose(z, d, atol=1e-8)

    def test_center_pixel_ray_is_optical_axis(self):
        # pixel-index coordinate (W/2 - 0.5) is the continuous image centre W/2 = cx
        r = self.cam.rays([319.5], [239.5])[0]
        np.testing.assert_allclose(r / np.linalg.norm(r), self.cam.optical_axis, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
