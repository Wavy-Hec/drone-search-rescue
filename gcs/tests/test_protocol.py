import json
import unittest

import numpy as np

from drone_gcs import protocol as P


class TestProtocol(unittest.TestCase):
    def roundtrip(self, mtype, fields=None, arrays=None, **kw):
        frame = P.encode_bytes(mtype, fields, arrays, **kw)
        return frame, P.decode_bytes(frame)

    def test_roundtrip_dtypes_shapes(self):
        arrays = {
            "f32": np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32),
            "u8": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            "b": np.array([True, False, True]),
            "i64": np.array([-5, 1 << 40], dtype=np.int64),
            "f64_big_endian": np.arange(4, dtype=">f8"),
            "non_contig": np.arange(20, dtype=np.int32).reshape(4, 5)[:, ::2],
            "empty": np.zeros((0, 3), np.float32),
            "scalar": np.array(3.5, dtype=np.float32),
        }
        _, m = self.roundtrip(P.SCAN, {"t": 1.5, "origin": [1, 2, 3]}, arrays, seq=7, drone_id=2, sender="x")
        self.assertEqual(m.type, P.SCAN)
        self.assertEqual((m.seq, m.drone_id, m.sender), (7, 2, "x"))
        self.assertEqual(m.t, 1.5)
        for k, a in arrays.items():
            b = m.arrays[k]
            self.assertEqual(b.shape, a.shape, k)
            np.testing.assert_array_equal(b, a, err_msg=k)
            self.assertTrue(b.dtype.isnative, k)

    def test_header_is_plain_json_with_explicit_dtypes(self):
        frame = P.encode_bytes(
            P.CAMERA, {"K": np.eye(3)}, {"rgb": np.zeros((2, 2, 3), np.uint8), "depth": np.ones((2, 2), np.float32)}
        )
        hlen, plen = P.parse_prefix(frame[: P.PREFIX_SIZE])
        header = json.loads(frame[P.PREFIX_SIZE : P.PREFIX_SIZE + hlen])
        self.assertEqual(header["proto"], "drone_gcs")
        self.assertEqual(header["v"], P.PROTOCOL_VERSION)
        self.assertEqual(header["fields"]["K"], np.eye(3).tolist())
        bufs = {b["name"]: b for b in header["buffers"]}
        self.assertEqual(bufs["rgb"]["dtype"], "|u1")
        self.assertEqual(bufs["depth"]["dtype"], "<f4")
        self.assertEqual(bufs["depth"]["offset"] % P.ALIGN, 0)
        self.assertEqual(len(frame), P.PREFIX_SIZE + hlen + plen)

    def test_jsonable_handles_numpy_and_nonfinite(self):
        out = P.jsonable({"a": np.float32(1.5), "b": np.int64(3), "c": [np.nan, np.inf, 1.0], "d": np.arange(3)})
        self.assertEqual(out, {"a": 1.5, "b": 3, "c": [None, None, 1.0], "d": [0, 1, 2]})

    def test_jpeg_roundtrip(self):
        rgb = np.zeros((48, 64, 3), np.uint8)
        rgb[:, :32] = (200, 30, 30)
        rgb[:, 32:] = (30, 30, 200)
        frame, m = self.roundtrip(P.CAMERA, {"t": 0}, {"rgb": rgb}, encodings={"rgb": "jpeg"}, jpeg_quality=95)
        self.assertLess(len(frame), rgb.nbytes)
        out = m.arrays["rgb"]
        self.assertEqual(out.shape, rgb.shape)
        self.assertLess(np.abs(out.astype(int) - rgb).mean(), 6)  # still RGB order, not BGR
        self.assertGreater(out[10, 5, 0], 150)

    def test_rejects_bad_frames(self):
        good = P.encode_bytes(P.END, {"reason": "x"})
        with self.assertRaises(P.ProtocolError):
            P.decode_bytes(b"XXXX" + good[4:])
        with self.assertRaises(P.ProtocolError):
            P.decode_bytes(good[:-1])
        with self.assertRaises(P.ProtocolError):
            P.encode_bytes(P.SCAN, {}, {"bad": np.array(["a", "b"])})

    def test_to_numpy_duck_typing(self):
        class FakeTensor:
            def __init__(self, a):
                self.a = a

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.a

        class FakeProxy:
            def __init__(self, a):
                self.torch = FakeTensor(a)

        a = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(P.to_numpy(FakeTensor(a)), a)
        np.testing.assert_array_equal(P.to_numpy(FakeProxy(a)), a)
        np.testing.assert_array_equal(P.to_numpy([[0, 1, 2], [3, 4, 5]]), a)

    def test_to_numpy_uses_proxyarray_torch_view_explicitly(self):
        # Like isaaclab.utils.warp.ProxyArray: .torch/.warp properties plus a __getattr__ deprecation bridge
        # that forwards everything else to .torch (and warns once per process). to_numpy must not touch it.
        class Tensor:
            def __init__(self, a):
                self.a = a

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.a

        implicit = []

        class ProxyLike:
            def __init__(self, a):
                self._t = Tensor(a)

            @property
            def torch(self):
                return self._t

            @property
            def warp(self):
                return None

            def __getattr__(self, name):
                implicit.append(name)
                return getattr(self._t, name)

        a = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(P.to_numpy(ProxyLike(a)), a)
        self.assertTrue(P.is_proxy_array(ProxyLike(a)))
        self.assertEqual(implicit, [])


if __name__ == "__main__":
    unittest.main()
