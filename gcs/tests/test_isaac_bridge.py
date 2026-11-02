"""IsaacLabStreamer with duck-typed stand-ins for Isaac Lab 3.0 objects (no isaaclab / torch import)."""

import unittest
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from drone_gcs import geometry as G
from drone_gcs.isaac_bridge import IsaacLabStreamer, row


class FakeTensor:
    """torch-like: indexing gives a view, .cpu() records how many elements leave the 'GPU'."""

    transferred: ClassVar[list] = []

    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, i):
        return FakeTensor(self.a[i])

    def detach(self):
        return self

    def cpu(self):
        FakeTensor.transferred.append(self.a.size)
        return self

    def numpy(self):
        return self.a


class FakeProxy:
    """isaaclab.utils.warp.ProxyArray look-alike (class-level .torch / .warp)."""

    def __init__(self, a):
        self._t = FakeTensor(a)

    @property
    def torch(self):
        return self._t

    @property
    def warp(self):
        return None


class RecordingSender:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("send_"):
            return lambda *a, **kw: self.calls.append((name, a, kw)) or True
        raise AttributeError(name)

    def _fail(self, what, e):
        self.calls.append(("_fail", (what, repr(e)), {}))
        return False


def xyzw(q_wxyz):
    return G.quat_wxyz_to_xyzw(np.asarray(q_wxyz, float))


def make_camera(n=2, h=6, w=8):
    rng = np.random.default_rng(0)
    q_ros = [G.quat_normalize(rng.normal(size=4)) for _ in range(n)]
    data = SimpleNamespace(
        output={
            "rgb": FakeProxy(rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8)),
            "distance_to_image_plane": FakeProxy(np.full((n, h, w, 1), 5.0, np.float32)),
        },
        intrinsic_matrices=FakeProxy(np.stack([np.eye(3)] * n)),
        pos_w=FakeProxy(np.arange(n * 3, dtype=np.float32).reshape(n, 3)),
        quat_w_ros=FakeProxy(np.stack([xyzw(q) for q in q_ros]).astype(np.float32)),
        info={"rgb": None},
    )
    cam = SimpleNamespace(data=data, frame=FakeProxy(np.array([1, 1], np.int64)))
    return cam, q_ros


class TestIsaacBridge(unittest.TestCase):
    def test_row_copies_only_one_env(self):
        FakeTensor.transferred.clear()
        x = FakeProxy(np.zeros((16, 48, 64, 3), np.uint8))
        r = row(x, 3)
        self.assertEqual(r.shape, (48, 64, 3))
        self.assertEqual(FakeTensor.transferred, [48 * 64 * 3])  # not 16x that
        np.testing.assert_array_equal(row(np.arange(6).reshape(2, 3), 1), [3, 4, 5])

    def test_camera_sends_each_capture_once(self):
        cam, q_ros = make_camera()
        tx = RecordingSender()
        s = IsaacLabStreamer(tx, env_id=1)
        self.assertTrue(s.camera(0.1, cam))
        self.assertFalse(s.camera(0.12, cam))  # same capture (update_period > step): not re-sent
        self.assertFalse(s.camera(0.14, cam))
        cam.frame = FakeProxy(np.array([2, 2], np.int64))
        self.assertTrue(s.camera(0.2, cam))
        self.assertTrue(s.camera(0.22, cam, force=True))
        sent = [c for c in tx.calls if c[0] == "send_camera"]
        self.assertEqual([c[2]["frame_id"] for c in sent], [1, 2, 2])
        _, (_, rgb, _, pos, quat), kw = sent[0]
        self.assertEqual(rgb.shape, (6, 8, 3))
        self.assertEqual(kw["depth"].shape, (6, 8, 1))
        np.testing.assert_allclose(pos, [3, 4, 5])
        np.testing.assert_allclose(G.quat_to_matrix(quat), G.quat_to_matrix(q_ros[1]), atol=1e-6)

    def test_camera_async_capture_pose(self):
        # OVRTX async rendering: the image is one capture old; its pose is in info[rgb]["capture"]
        cam, _ = make_camera()
        q_world = G.quat_normalize(np.array([0.9, 0.1, -0.2, 0.3]))
        cam.data.info = {
            "rgb": {
                "capture": {
                    "pos_w": FakeProxy(np.array([[0, 0, 0], [10, 20, 30]], np.float32)),
                    "quat_w_world": FakeProxy(np.stack([xyzw(q_world)] * 2).astype(np.float32)),
                    "intrinsic_matrices": FakeProxy(np.stack([np.eye(3) * 2] * 2)),
                    "frame": FakeProxy(np.array([7, 7], np.int64)),
                }
            }
        }
        tx = RecordingSender()
        s = IsaacLabStreamer(tx, env_id=1, origin=[1, 2, 3])
        self.assertTrue(s.camera(1.0, cam))
        _, (_, _, K, pos, quat), kw = tx.calls[-1]
        self.assertEqual(kw["frame_id"], 7)
        np.testing.assert_allclose(pos, [9, 18, 27])  # capture pose, minus the env origin
        np.testing.assert_allclose(K, np.eye(3) * 2)
        np.testing.assert_allclose(quat, G.camera_quat_to_ros(q_world, "world"), atol=1e-6)

    def test_camera_without_frame_counter_counts_calls(self):
        cam, _ = make_camera()
        del cam.frame
        tx = RecordingSender()
        s = IsaacLabStreamer(tx)
        self.assertTrue(s.camera(0.0, cam))
        self.assertTrue(s.camera(0.1, cam))
        self.assertEqual([c[2]["frame_id"] for c in tx.calls], [0, 1])

    def test_world_flag_subtracts_the_env_origin(self):
        tx = RecordingSender()
        s = IsaacLabStreamer(tx, env_id=2, origin=[100.0, -50.0, 1.0])
        s.mission((80, 120, -70, -30), 0.25, np.array([[80, -70], [120, -30]], float), 9.0, world=True, name="w")
        s.mission((-20, 20, -20, 20), 0.25, name="local")
        s.gt_objects([{"class": "car", "pos": [101, -49, 1]}, ("tent", (99, -51, 1))], world=True)
        s.gt_objects([{"class": "car", "pos": [1, 1, 0]}])
        (_, a1, k1), (_, a2, _), (_, g1, kg), (_, g2, _) = tx.calls
        self.assertEqual(a1[0], (-20.0, 20.0, -20.0, 20.0))
        np.testing.assert_allclose(a1[2], [[-20, -20], [20, 20]])
        self.assertEqual(a1[3], 8.0)
        self.assertEqual((k1["drone_id"], k1["name"]), (2, "w"))
        self.assertEqual(a2[0], (-20, 20, -20, 20))
        self.assertEqual(g1[0][0]["pos"], [1.0, 1.0, 0.0])
        self.assertEqual(g1[0][1], ("tent", [-1.0, -1.0, 0.0]))
        self.assertEqual(kg["drone_id"], 2)
        self.assertEqual(g2[0][0]["pos"], [1, 1, 0])
        s.gt_terrain(np.full((2, 3), 4.0), 74.0, -76.0, 0.5, world=True)
        (_, (h, x0, y0, res), kt) = tx.calls[-1]
        np.testing.assert_allclose(h, 3.0)
        self.assertEqual((x0, y0, res, kt["drone_id"]), (-26.0, -26.0, 0.5, 2))


if __name__ == "__main__":
    unittest.main()
