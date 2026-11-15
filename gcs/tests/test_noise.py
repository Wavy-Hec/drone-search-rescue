"""Sensor noise injection (``drone_gcs/noise.py``): the settings, the injector's errors and their statistics, its
use in the app, ``gcs.py --noise`` and ``scripts/noise_sweep.py``. ~15 s."""

import contextlib
import copy
import csv
import importlib.util
import io
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from drone_gcs import protocol as P
from drone_gcs.app import AppConfig
from drone_gcs.geometry import euler_from_quat, quat_from_euler, quat_to_matrix
from drone_gcs.noise import KEYS, NoiseCfg, NoiseInjector, parse_noise
from drone_gcs.recording import read_recording

from ._util import ROOT, rmtree, short_tmpdir
from .test_recording import make_small_recording, run_app


def load_sweep_script():
    spec = importlib.util.spec_from_file_location("noise_sweep", ROOT / "scripts" / "noise_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SWEEP = load_sweep_script()

EVERYTHING = "pos=0.5,att=1,depth=0.02,dropout=0.1,scan=0.02,lag=0.05,tau=30,pos_white=0.05,att_white=0.2,seed=1"
BODY_QUAT = quat_from_euler(0.05, -0.1, 0.7)  # body -> world of the test drone
CAM_QUAT = quat_from_euler(3.0, 0.2, -1.1)  # camera -> world: a different rotation than the body's
POINTS = np.array(
    [[1.0, 2.0, 0.5], [np.inf, np.inf, np.inf], [-3.0, 4.0, 1.5], [np.nan, 2.0, 0.0], [5.0, -6.0, 2.5]], np.float32
)
HITS = np.array([True, False, True, False, True])  # the finite rows of POINTS
RGB = np.random.default_rng(5).integers(0, 255, (30, 40, 3), dtype=np.uint8)
DEPTH = np.linspace(2.0, 40.0, 30 * 40, dtype=np.float32).reshape(30, 40)


def telemetry(t, pos=(1.0, 2.0, 30.0), quat=BODY_QUAT, drone_id=0, **fields):
    fields = {"t": t, "pos": list(pos), "quat": list(quat), "wp_index": 3, "carrot": [9.0, 8.0, 7.0], **fields}
    return P.Message(P.TELEMETRY, fields, drone_id=drone_id)


def scan(t, points=POINTS, drone_id=0):
    arrays = {"points": points, "valid": np.isfinite(points).all(axis=1)}
    return P.Message(P.SCAN, {"t": t, "origin": [1.0, 2.0, 29.9]}, arrays, drone_id=drone_id)


def camera(t, pos=(1.0, 2.0, 29.9), quat=CAM_QUAT, depth=DEPTH, drone_id=0):
    fields = {
        "t": t,
        "pos": list(pos),
        "quat": list(quat),
        "K": [[30.0, 0.0, 20.0], [0.0, 30.0, 15.0], [0.0, 0.0, 1.0]],
    }
    fields["gt_boxes"] = [{"class": "person", "box": [3.0, 4.0, 9.0, 12.0]}]
    return P.Message(P.CAMERA, fields, {"rgb": RGB, "depth": depth}, drone_id=drone_id)


def one_step(t, drone_id=0):
    """The three data messages of one time step of one drone."""
    return [telemetry(t, drone_id=drone_id), scan(t, drone_id=drone_id), camera(t, drone_id=drone_id)]


def pos_error(noisy, clean):
    """Position error [m] the injector added to a TELEMETRY or CAMERA message."""
    return np.array(noisy.fields["pos"]) - np.array(clean.fields["pos"])


def world_rotation(noisy, clean):
    """The rotation R with ``R(noisy quat) = R @ R(clean quat)``: the attitude error about the world axes."""
    return quat_to_matrix(noisy.fields["quat"]) @ quat_to_matrix(clean.fields["quat"]).T


def rot_x(a):
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def rot_z(a):
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def correlation(x, lag):
    """Correlation of the columns of ``x`` (N, 3) with themselves ``lag`` rows later (all columns pooled)."""
    return float(np.mean(x[:-lag] * x[lag:]) / np.mean(x * x))


def telemetry_errors(cfg, times):
    """Position errors [m] and attitude errors (roll, pitch, yaw) [deg] of one telemetry message per time."""
    inj = NoiseInjector(cfg)
    pos, att = [], []
    for t in times:
        clean = telemetry(float(t), quat=(1.0, 0.0, 0.0, 0.0))  # level flight: the output quaternion is the error
        noisy = inj.apply(clean)
        pos.append(pos_error(noisy, clean))
        att.append(np.degrees(euler_from_quat(noisy.fields["quat"])))
    return np.array(pos), np.array(att)


def assert_same_content(case, got, expected):
    """Same fields and the same arrays, bit for bit."""
    case.assertEqual(got.fields, expected.fields)
    case.assertEqual(list(got.arrays), list(expected.arrays))
    for name, arr in expected.arrays.items():
        case.assertEqual((got.arrays[name].dtype, got.arrays[name].shape), (arr.dtype, arr.shape), name)
        case.assertEqual(got.arrays[name].tobytes(), arr.tobytes(), name)


def assert_passed_on(case, inj, msg):
    """The injector hands ``msg`` on with the content it came with."""
    before = copy.deepcopy(msg)
    assert_same_content(case, inj.apply(msg), before)


class TestNoiseCfg(unittest.TestCase):
    def test_parse_noise_reads_every_key(self):
        expected = NoiseCfg(
            seed=1, pos_bias_m=0.5, pos_white_m=0.05, att_bias_deg=1.0, att_white_deg=0.2, tau_s=30.0,
            depth_rel=0.02, depth_dropout=0.1, scan_z_m=0.02, time_offset_s=0.05,
        )  # fmt: skip
        self.assertEqual(parse_noise(EVERYTHING), expected)
        self.assertEqual(parse_noise(" scan = 0.1 , tau=5,"), NoiseCfg(scan_z_m=0.1, tau_s=5.0))  # any subset
        self.assertEqual(parse_noise("pos=1,pos=2"), NoiseCfg(pos_bias_m=2.0))  # the last one wins
        self.assertEqual(parse_noise(""), NoiseCfg())
        self.assertIsInstance(parse_noise("seed=7").seed, int)

    def test_defaults_are_off(self):
        cfg = NoiseCfg()
        self.assertFalse(cfg.active)
        self.assertEqual(
            cfg.to_dict(),
            {
                "seed": 0, "pos_bias_m": 0.0, "pos_white_m": 0.0, "att_bias_deg": 0.0, "att_white_deg": 0.0,
                "tau_s": 30.0, "depth_rel": 0.0, "depth_dropout": 0.0, "scan_z_m": 0.0, "time_offset_s": 0.0,
            },
        )  # fmt: skip
        self.assertFalse(NoiseCfg(seed=4, tau_s=2.0).active)  # the seed and tau are not noise levels
        for key in KEYS:
            if key not in ("seed", "tau"):
                self.assertTrue(parse_noise(f"{key}=0.1").active, key)
        self.assertTrue(parse_noise("lag=-0.1").active)  # the camera paired with an earlier pose

    def test_parse_noise_errors(self):
        with self.assertRaises(ValueError) as caught:
            parse_noise("pos=1,gps=2")
        self.assertIn("'gps'", str(caught.exception))
        for key in ("pos", "pos_white", "att", "att_white", "tau", "depth", "dropout", "scan", "lag", "seed"):
            self.assertIn(key, str(caught.exception))  # the message lists the valid keys
        for bad in ("pos", "pos=", "pos=high", "seed=1.5", "pos=-1", "scan=-0.1", "dropout=1.5", "tau=0", "seed=-1"):
            with self.assertRaises(ValueError, msg=bad):
                parse_noise(bad)

    def test_spec_names_the_settings_that_are_on(self):
        self.assertEqual(NoiseCfg().spec(), "")
        self.assertEqual(NoiseCfg(seed=3, att_bias_deg=1.0, pos_bias_m=0.25).spec(), "pos=0.25,att=1,seed=3")
        self.assertEqual(parse_noise(parse_noise(EVERYTHING).spec()), parse_noise(EVERYTHING))


class TestNoiseInjector(unittest.TestCase):
    def test_zero_config_is_the_identity(self):
        inj = NoiseInjector(NoiseCfg(seed=3, tau_s=5.0))
        for t in (0.1, 0.2):
            for msg in one_step(t):
                assert_passed_on(self, inj, msg)
                self.assertIs(inj.apply(msg), msg)  # (nothing is copied either)

    def test_other_messages_and_the_ground_truth_stay_as_they_are(self):
        inj = NoiseInjector(parse_noise(EVERYTHING))
        others = [
            P.Message(P.HELLO, {"name": "sim"}),
            P.Message(P.MISSION, {"area": [0, 10, 0, 10], "home": [1.0, 2.0, 0.0]}, {"planned_path": np.ones((4, 3))}),
            P.Message(P.GT_OBJECTS, {"objects": [{"class": "person", "pos": [1.0, 2.0, 0.0]}]}),
            P.Message(P.GT_TERRAIN, {"x_min": 0.0, "y_min": 0.0, "resolution": 0.5}, {"heights": np.ones((4, 4))}),
            P.Message(P.END, {"t": 9.0, "reason": "complete"}),
        ]
        for msg in others:
            assert_passed_on(self, inj, msg)
        tel, sc, cam = one_step(1.0)
        noisy_tel, noisy_scan, noisy_cam = (inj.apply(m) for m in (tel, sc, cam))
        for name in ("t", "wp_index", "carrot"):
            self.assertEqual(noisy_tel.fields[name], tel.fields[name], name)
        self.assertEqual(noisy_scan.arrays["valid"].tobytes(), sc.arrays["valid"].tobytes())
        for name in ("t", "K", "gt_boxes"):  # the simulator's boxes are ground truth
            self.assertEqual(noisy_cam.fields[name], cam.fields[name], name)
        self.assertEqual(noisy_cam.arrays["rgb"].tobytes(), RGB.tobytes())
        self.assertEqual((noisy_tel.type, noisy_tel.drone_id), (P.TELEMETRY, 0))

    def test_the_messages_of_one_time_step_share_one_error(self):
        inj = NoiseInjector(parse_noise("pos=0.5,pos_white=0.1,att=2,att_white=0.5,seed=4"))
        errors = []
        for t in (1.0, 1.1):
            tel, sc, cam = one_step(t)
            noisy_tel, noisy_scan, noisy_cam = (inj.apply(m) for m in (tel, sc, cam))
            e = pos_error(noisy_tel, tel)
            np.testing.assert_allclose(pos_error(noisy_cam, cam), e, atol=1e-12)
            np.testing.assert_allclose(np.array(noisy_scan.fields["origin"]) - sc.fields["origin"], e, atol=1e-12)
            points = noisy_scan.arrays["points"]
            self.assertEqual((points.dtype, points.shape), (POINTS.dtype, POINTS.shape))
            np.testing.assert_allclose(points[HITS] - POINTS[HITS], np.tile(e, (3, 1)), atol=1e-5)  # float32
            self.assertEqual(points[~HITS].tobytes(), POINTS[~HITS].tobytes())  # misses are not touched
            # the attitude: one rotation about the world axes for the body and the camera
            R = world_rotation(noisy_tel, tel)
            np.testing.assert_allclose(world_rotation(noisy_cam, cam), R, atol=1e-12)
            angle = math.degrees(math.acos((np.trace(R) - 1.0) / 2.0))
            self.assertTrue(0.05 < angle < 15.0, angle)  # a small rotation, and not none
            errors.append(e)
        self.assertGreater(np.linalg.norm(errors[1] - errors[0]), 0.01)  # the next time step has its own error
        # the order inside a time step does not matter
        tel, sc, cam = one_step(1.2)
        e = pos_error(inj.apply(cam), cam)
        np.testing.assert_allclose(inj.apply(sc).arrays["points"][0] - POINTS[0], e, atol=1e-5)
        np.testing.assert_allclose(pos_error(inj.apply(tel), tel), e, atol=1e-12)

    def test_bias_has_the_set_spread_and_correlation_time(self):
        cfg = NoiseCfg(seed=11, pos_bias_m=2.0, att_bias_deg=3.0, tau_s=5.0)
        pos, att = telemetry_errors(cfg, np.arange(20000))  # one step per second
        for errors, sigma in ((pos, 2.0), (att, 3.0)):
            for axis in range(3):
                self.assertAlmostEqual(errors[:, axis].std(), sigma, delta=0.1 * sigma)
                self.assertAlmostEqual(errors[:, axis].mean(), 0.0, delta=0.15 * sigma)
            self.assertAlmostEqual(correlation(errors, 5), math.exp(-1.0), delta=0.05)  # one tau later
            self.assertAlmostEqual(correlation(errors, 10), math.exp(-2.0), delta=0.05)
        # the three axes are independent of each other
        self.assertLess(abs(np.corrcoef(pos[:, 0], pos[:, 1])[0, 1]), 0.1)
        self.assertLess(abs(np.corrcoef(att[:, 1], att[:, 2])[0, 1]), 0.1)

    def test_first_time_step_already_has_the_full_spread(self):
        first = []
        for seed in range(1500):  # the first error of 1500 flights
            pos, _ = telemetry_errors(NoiseCfg(seed=seed, pos_bias_m=2.0), [0.0])
            first.append(pos[0])
        first = np.array(first)
        self.assertAlmostEqual(first.std(), 2.0, delta=0.1)
        self.assertAlmostEqual(first.mean(), 0.0, delta=0.1)
        self.assertLess(abs(np.corrcoef(first[:-1, 0], first[1:, 0])[0, 1]), 0.1)  # seeds are independent

    def test_bias_follows_the_time_between_messages(self):
        cfg = NoiseCfg(seed=12, pos_bias_m=1.0, tau_s=5.0)
        times = np.cumsum(np.tile([1.0, 4.0], 10000))  # steps of 1 s and 4 s in turn
        pos, _ = telemetry_errors(cfg, times)
        after_4s = np.mean(pos[0:-1:2] * pos[1::2]) / np.mean(pos * pos)  # pairs (2k, 2k+1): 4 s apart
        after_1s = np.mean(pos[1:-1:2] * pos[2::2]) / np.mean(pos * pos)  # pairs (2k+1, 2k+2): 1 s apart
        self.assertAlmostEqual(after_1s, math.exp(-1.0 / 5.0), delta=0.04)
        self.assertAlmostEqual(after_4s, math.exp(-4.0 / 5.0), delta=0.04)
        self.assertAlmostEqual(pos.std(), 1.0, delta=0.1)  # the spread does not depend on the step

    def test_white_noise_is_independent_from_step_to_step_and_adds_to_the_bias(self):
        pos, att = telemetry_errors(NoiseCfg(seed=13, pos_white_m=0.5, att_white_deg=1.5), np.arange(5000) * 0.1)
        self.assertAlmostEqual(pos.std(), 0.5, delta=0.05)
        self.assertAlmostEqual(att.std(), 1.5, delta=0.15)
        self.assertLess(abs(correlation(pos, 1)), 0.05)
        self.assertLess(abs(correlation(att, 1)), 0.05)
        both, _ = telemetry_errors(NoiseCfg(seed=14, pos_bias_m=0.3, pos_white_m=0.4, tau_s=2.0), np.arange(5000))
        self.assertAlmostEqual(both.std(), 0.5, delta=0.05)  # sqrt(0.3^2 + 0.4^2)

    def test_same_seed_and_order_give_the_same_noise(self):
        stream = [m for t in (0.1, 0.2, 0.3) for m in one_step(t)]
        a, b, other = (NoiseInjector(parse_noise(spec)) for spec in (EVERYTHING, EVERYTHING, EVERYTHING + ",seed=2"))
        differs = 0
        for msg in stream:
            out_a, out_b, out_other = a.apply(msg), b.apply(msg), other.apply(msg)
            assert_same_content(self, out_a, out_b)
            differs += out_a.fields != out_other.fields
        self.assertEqual(differs, len(stream))  # another seed, another noise: in every message

    def test_every_drone_has_its_own_random_stream(self):
        cfg = parse_noise(EVERYTHING)
        alone, mixed = NoiseInjector(cfg), NoiseInjector(cfg)
        for t in (0.1, 0.2, 0.3):
            for msg0, msg1 in zip(one_step(t, drone_id=0), one_step(t, drone_id=1), strict=True):
                out0 = mixed.apply(msg0)
                out1 = mixed.apply(msg1)
                assert_same_content(self, out1, alone.apply(msg1))  # drone 1 does not depend on drone 0's messages
                self.assertNotEqual(out0.fields, out1.fields)  # and it has another error than drone 0

    def test_pixel_noise_leaves_the_navigation_error_as_it_was(self):
        nav_only = NoiseInjector(parse_noise("pos=0.5,att=1,seed=6"))
        with_pixels = NoiseInjector(parse_noise("pos=0.5,att=1,seed=6,depth=0.05,dropout=0.2,scan=0.05"))
        for t in (0.1, 0.2, 0.3):
            for msg in one_step(t):
                a, b = nav_only.apply(msg), with_pixels.apply(msg)
                for name in ("pos", "quat", "origin"):
                    self.assertEqual(a.fields.get(name), b.fields.get(name), (msg.type, name))
            self.assertNotEqual(a.arrays["depth"].tobytes(), b.arrays["depth"].tobytes())  # (the camera frame)

    def test_depth_error_grows_with_the_depth(self):
        depth = np.linspace(2.0, 40.0, 200 * 300, dtype=np.float32).reshape(200, 300)
        depth[0, :3] = [0.0, np.nan, np.inf]  # invalid pixels
        noisy = NoiseInjector(parse_noise("depth=0.05,seed=2")).apply(camera(1.0, depth=depth)).arrays["depth"]
        self.assertEqual((noisy.dtype, noisy.shape), (np.float32, (200, 300)))
        self.assertEqual(noisy[0, 0], 0.0)  # invalid stays invalid
        self.assertTrue(np.isnan(noisy[0, 1]) and np.isinf(noisy[0, 2]))
        noisy, depth = noisy[1:], depth[1:]  # the rows without the invalid pixels
        relative = noisy / depth - 1.0
        self.assertAlmostEqual(relative.mean(), 0.0, delta=0.001)
        self.assertAlmostEqual(relative.std(), 0.05, delta=0.0025)
        self.assertAlmostEqual(relative[:80].std(), 0.05, delta=0.005)  # near pixels (2 .. 17 m)
        self.assertAlmostEqual(relative[-80:].std(), 0.05, delta=0.005)  # far pixels (25 .. 40 m): the same fraction
        absolute = noisy - depth
        self.assertGreater(absolute[-80:].std(), 2.0 * absolute[:80].std())  # so more metres far away

    def test_depth_dropout_loses_the_set_fraction_of_pixels(self):
        depth = np.linspace(2.0, 40.0, 200 * 300, dtype=np.float32).reshape(200, 300)
        noisy = NoiseInjector(parse_noise("dropout=0.25,seed=2")).apply(camera(1.0, depth=depth)).arrays["depth"]
        lost = np.isinf(noisy)
        self.assertAlmostEqual(lost.mean(), 0.25, delta=0.01)
        self.assertEqual(noisy[~lost].tobytes(), depth[~lost].tobytes())  # the other pixels are exact
        self.assertAlmostEqual(lost[:, :150].mean(), lost[:, 150:].mean(), delta=0.02)  # all over the image
        both = NoiseInjector(parse_noise("dropout=0.25,depth=0.05,seed=2")).apply(camera(1.0, depth=depth))
        kept = np.isfinite(both.arrays["depth"])
        self.assertAlmostEqual(1.0 - kept.mean(), 0.25, delta=0.01)
        self.assertAlmostEqual((both.arrays["depth"][kept] / depth[kept] - 1.0).std(), 0.05, delta=0.0025)

    def test_scan_height_noise(self):
        points = np.random.default_rng(8).uniform(-20.0, 20.0, (5000, 3)).astype(np.float32)
        points[::100] = np.inf  # misses
        noisy = NoiseInjector(parse_noise("scan=0.05,seed=2")).apply(scan(1.0, points)).arrays["points"]
        hits = np.isfinite(points).all(axis=1)
        self.assertEqual(noisy[hits, :2].tobytes(), points[hits, :2].tobytes())  # x and y are exact
        dz = noisy[hits, 2] - points[hits, 2]
        self.assertAlmostEqual(dz.mean(), 0.0, delta=0.005)
        self.assertAlmostEqual(dz.std(), 0.05, delta=0.005)
        self.assertTrue(np.isinf(noisy[~hits]).all())

    def test_time_offset_moves_the_camera_by_velocity_times_offset(self):
        inj = NoiseInjector(parse_noise("lag=0.1"))
        assert_passed_on(self, inj, camera(0.5))  # no telemetry yet: no velocity to go by
        assert_passed_on(self, inj, telemetry(1.0, vel=[2.0, -1.0, 0.5]))  # the vehicle's own pose is on time
        assert_passed_on(self, inj, scan(1.0))
        cam = camera(1.0)
        noisy = inj.apply(cam)
        np.testing.assert_allclose(pos_error(noisy, cam), [0.2, -0.1, 0.05], atol=1e-12)  # v x 0.1 s
        np.testing.assert_allclose(noisy.fields["quat"], CAM_QUAT, atol=1e-12)  # no ang_vel: the attitude stays
        self.assertEqual(noisy.arrays["depth"].tobytes(), DEPTH.tobytes())
        # with an angular velocity the camera also turns, about the world axes
        for ang_vel, expected in (([0.0, 0.0, 0.5], rot_z(0.05)), ([0.4, 0.0, 0.0], rot_x(0.04))):
            inj.apply(telemetry(2.0, vel=[0.0, 0.0, 0.0], ang_vel=ang_vel))
            noisy = inj.apply(cam)
            np.testing.assert_allclose(pos_error(noisy, cam), 0.0, atol=1e-12)
            np.testing.assert_allclose(world_rotation(noisy, cam), expected, atol=1e-12)
        inj.apply(telemetry(2.5, vel=[2.0, 0.0, 0.0], ang_vel=[0.0, 0.0, 0.5]))
        inj.apply(telemetry(3.0))  # the last telemetry has no rates: the camera stays where it was
        assert_passed_on(self, inj, cam)

    def test_input_is_not_changed(self):
        inj = NoiseInjector(parse_noise(EVERYTHING))
        for t in (0.1, 0.2):
            for msg in (telemetry(t, vel=[1.0, 0.5, 0.0], ang_vel=[0.0, 0.1, 0.2]), scan(t), camera(t)):
                msg.arrays = {name: arr.copy() for name, arr in msg.arrays.items()}
                for arr in msg.arrays.values():
                    arr.flags.writeable = False  # as decoded from a read-only buffer
                before = copy.deepcopy(msg)
                out = inj.apply(msg)
                assert_same_content(self, msg, before)
                self.assertIsNot(out, msg)
                self.assertNotEqual(out.fields, before.fields)  # the noise went into the copy
                for name in ("points", "depth"):
                    if name in msg.arrays:
                        self.assertFalse(np.shares_memory(out.arrays[name], msg.arrays[name]), name)
                        self.assertNotEqual(out.arrays[name].tobytes(), msg.arrays[name].tobytes(), name)

    def test_a_message_without_its_fields_is_passed_on_unchanged(self):
        inj = NoiseInjector(parse_noise("pos=0.5,seed=1"))
        for bad in (
            P.Message(P.TELEMETRY, {"t": 1.0, "quat": [1.0, 0.0, 0.0, 0.0]}),  # no position
            P.Message(P.TELEMETRY, {"t": 1.0, "pos": [1.0, 2.0], "quat": [1.0, 0.0, 0.0, 0.0]}),
            P.Message(P.CAMERA, {"t": 1.0, "pos": None, "quat": None}, {"rgb": RGB}),
            P.Message(P.SCAN, {"t": 1.0}, {"points": np.zeros((4, 2), np.float32)}),
        ):
            self.assertIs(inj.apply(bad), bad)  # the station counts it as a bad message when it applies it
        good = telemetry(1.0)
        self.assertGreater(np.linalg.norm(pos_error(inj.apply(good), good)), 1e-3)


class TestNoiseThroughApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = short_tmpdir("dgcs_noise")
        cls.recording = Path(cls.tmp) / "small.dgcrec"
        make_small_recording(cls.recording)
        cls.clean, cls.clean_summary = cls.replay("clean", None)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp)

    @classmethod
    def replay(cls, name, noise):
        """Replay the small recording headless with the blob detector; returns the app and its summary.json."""
        cfg = AppConfig(replay=cls.recording, out_dir=Path(cls.tmp), name=name, headless=True, video=False)
        cfg.detector, cfg.linger, cfg.max_time, cfg.noise = "blob", 0.0, 60.0, noise
        app, code = run_app(cfg)
        if code != 0:
            raise RuntimeError(f"replay {name} failed with exit code {code}")
        return app, json.loads((app.run_dir / "summary.json").read_text())

    def test_constant_bias_displaces_the_pins_by_that_bias(self):
        cfg = parse_noise("pos=1,tau=1e9,seed=1")  # a correlation time of 30 years: the bias stays what it is
        first = next(m for m in read_recording(self.recording) if m.type == P.TELEMETRY)
        bias = pos_error(NoiseInjector(cfg).apply(first), first)  # what this seed adds to the drone's position
        self.assertGreater(np.linalg.norm(bias[:2]), 0.5)
        with self.assertLogs("drone_gcs.state", level="INFO") as logs:
            noisy, summary = self.replay("bias", cfg)
        self.assertIn(
            "sensor noise on the incoming messages: pos=1,tau=1e+09,seed=1", [r.getMessage() for r in logs.records]
        )
        clean_pins = self.clean.state.tracker.confirmed()
        pins = noisy.state.tracker.confirmed()
        self.assertGreaterEqual(len(clean_pins), 2)
        self.assertEqual([(p.cls, p.hits) for p in pins], [(p.cls, p.hits) for p in clean_pins])
        for pin, clean_pin in zip(pins, clean_pins, strict=True):
            np.testing.assert_allclose(pin.pos - clean_pin.pos, bias, atol=0.01)
        # the drone's track and the map moved with it
        np.testing.assert_allclose(
            np.array(noisy.state.trajectory)[:, 2:5] - np.array(self.clean.state.trajectory)[:, 2:5],
            np.tile(bias, (len(self.clean.state.trajectory), 1)),
            atol=0.01,
        )
        self.assertAlmostEqual(noisy.state.ground_level() - self.clean.state.ground_level(), bias[2], delta=0.1)
        self.assertEqual(summary["gcs"]["noise"], cfg.to_dict())
        self.assertEqual(summary["gcs"]["noise"]["pos_bias_m"], 1.0)
        self.assertIsNone(self.clean_summary["gcs"]["noise"])  # a station without --noise
        self.assertFalse(any("noise" in text for _, text in self.clean.state.events))

    def test_all_levels_zero_changes_nothing(self):
        zero, summary = self.replay("zero", NoiseCfg(seed=5))
        self.assertEqual(
            [ob.to_dict() for ob in zero.state.tracker.objects],
            [ob.to_dict() for ob in self.clean.state.tracker.objects],
        )
        np.testing.assert_array_equal(zero.state.elevation.mean(), self.clean.state.elevation.mean())
        self.assertEqual(zero.state.trajectory, self.clean.state.trajectory)
        self.assertEqual(summary["scorecard"], self.clean_summary["scorecard"])
        self.assertEqual(summary["gcs"]["noise"], NoiseCfg(seed=5).to_dict())

    def test_noisy_replay_is_repeatable(self):
        cfg = parse_noise("pos=0.3,att=2,depth=0.05,dropout=0.2,scan=0.05,lag=0.1,pos_white=0.1,att_white=0.5,seed=2")
        a, _ = self.replay("again_a", cfg)
        b, _ = self.replay("again_b", cfg)
        self.assertGreaterEqual(len(a.state.tracker.objects), 2)
        self.assertEqual(
            [ob.to_dict() for ob in a.state.tracker.objects], [ob.to_dict() for ob in b.state.tracker.objects]
        )
        np.testing.assert_array_equal(a.state.elevation.mean(), b.state.elevation.mean())
        self.assertNotEqual(
            [ob.to_dict() for ob in a.state.tracker.objects], [ob.to_dict() for ob in self.clean.state.tracker.objects]
        )
        self.assertEqual(dict(a.state.errors), {})  # every noisy message could be applied

    def test_command_line_refuses_a_bad_noise_setting(self):
        script = [sys.executable, str(ROOT / "scripts" / "gcs.py"), "--headless", "--no-video", "--detector", "blob"]
        script += ["--out-dir", self.tmp, "--name", "refused", "--replay", str(self.recording)]
        out = subprocess.run([*script, "--noise", "wind=3"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 2)  # an argparse error, before anything is created
        self.assertIn("unknown noise setting 'wind'", out.stderr)
        self.assertIn("pos_white", out.stderr)
        self.assertFalse((Path(self.tmp) / "refused").exists())

    def test_sweep_runs_the_station_per_value_and_writes_the_tables(self):
        out = Path(self.tmp) / "sweep"
        args = ["--replay", str(self.recording), "--param", "pos", "--values", "0,1", "--fixed", "tau=1e9"]
        args += ["--seeds", "1", "--out", str(out), "--", "--detector", "blob"]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = SWEEP.main(args)
        self.assertEqual(rc, 0)
        with open(out / "sweep.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(list(rows[0]), list(SWEEP.COLUMNS))
        self.assertEqual(
            [(r["param"], r["value"], r["seed"], r["ok"]) for r in rows],
            [("pos", "0", "1", "1"), ("pos", "1", "1", "1")],
        )
        self.assertEqual([r["run"] for r in rows], ["pos_0_s1", "pos_1_s1"])
        self.assertEqual([r["noise"] for r in rows], ["tau=1e9,pos=0,seed=1", "tau=1e9,pos=1,seed=1"])
        cards = []
        for row in rows:  # every row holds the scorecard of its own run folder
            summary = json.loads((out / row["run"] / "summary.json").read_text())
            self.assertEqual(summary["gcs"]["noise"], parse_noise(row["noise"]).to_dict())
            self.assertEqual(summary["gcs"]["replay"]["file"], "small.dgcrec")
            self.assertIn("colour-blob", summary["gcs"]["detector"])  # the arguments after "--" reached gcs.py
            self.assertFalse((out / row["run"] / "dashboard.mp4").exists())
            card = summary["scorecard"]
            for key in ("people_found", "people_total", "false_alarms", "pin_error_m", "people_pin_error_m"):
                self.assertEqual(row[key], str(card[key]), key)
            self.assertEqual(row["map_height_error_smooth_m"], "")  # the small recording has no terrain truth
            cards.append(card)
        self.assertEqual(cards[0], self.clean_summary["scorecard"])  # level 0 is the mission without noise
        self.assertGreater(cards[1]["pin_error_m"], cards[0]["pin_error_m"] + 0.3)  # the pins carry the bias
        text = (out / "sweep.md").read_text()
        self.assertIn("| pos (pos_bias_m) | Runs | People found | False alarms | Pin error [m] |", text)
        for row, card in zip(rows, cards, strict=True):  # one seed: the mean is the value, the spread 0
            line = (
                f"| {row['value']} | 1 / 1 | {card['people_found']:.2f} ± 0.00 of {card['people_total']} | "
                f"{card['false_alarms']:.2f} ± 0.00 | {card['pin_error_m']:.3f} ± 0.000 | "
                f"{card['people_pin_error_m']:.3f} ± 0.000 | - |"
            )
            self.assertIn(line, text)
        self.assertEqual(text.count("\n| "), 3)  # the header and one row per value
        self.assertIn("tau=1e9", text)
        self.assertIn("--detector blob", text)
        self.assertIn(text, printed.getvalue())
        # a second sweep into the same folder would mix old and new runs: refused before anything runs
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            SWEEP.main(args)
        self.assertIn("already holds runs", errors.getvalue())


class TestSweepTable(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(short_tmpdir("dgcs_sweep"))
        self.recording = self.tmp / "flight.dgcrec"
        self.recording.write_bytes(b"")  # the sweep only checks that the file is there

    def tearDown(self):
        rmtree(self.tmp)

    @staticmethod
    def row(value, seed, ok=1, found=None, alarms=None, pin=None, total=5):
        row = dict.fromkeys(SWEEP.COLUMNS)
        row.update(param="pos", value=value, seed=seed, run=f"pos_{value}_s{seed}", ok=ok, people_total=total)
        row.update(people_found=found, false_alarms=alarms, pin_error_m=pin)
        return row

    def test_mean_and_spread_over_the_seeds_against_hand_computed_numbers(self):
        rows = [
            self.row("1", 0, found=3, alarms=0, pin=0.1),
            self.row("1", 1, found=4, alarms=2, pin=0.2),
            self.row("1", 2, found=5, alarms=1, pin=None),  # found nothing it could match: no pin error
            self.row("1", 3, ok=0, found=0, alarms=9, pin=9.0),  # failed: left out
            self.row("0.5", 0, found=5, alarms=0, pin=0.04),
        ]
        table = SWEEP.summarize(rows)
        self.assertEqual([e["value"] for e in table], ["1", "0.5"])  # in the order given
        one, half = table
        self.assertEqual((one["runs"], one["ok"], one["people_total"]), (4, 3, 5))
        self.assertEqual(one["people_found"], (4.0, 1.0))  # mean and sample standard deviation of 3, 4, 5
        self.assertEqual(one["false_alarms"], (1.0, 1.0))
        self.assertAlmostEqual(one["pin_error_m"][0], 0.15)
        self.assertAlmostEqual(one["pin_error_m"][1], 0.0707107, places=6)  # of 0.1 and 0.2
        self.assertIsNone(one["map_height_error_smooth_m"])  # no run reports it
        self.assertEqual(half["pin_error_m"], (0.04, 0.0))  # one run: no spread
        text = SWEEP.markdown("pos", rows, "Recording flight.dgcrec.")
        self.assertIn("| 1 | 3 / 4 | 4.00 ± 1.00 of 5 | 1.00 ± 1.00 | 0.150 ± 0.071 | - | - |", text)
        self.assertIn("| 0.5 | 1 / 1 | 5.00 ± 0.00 of 5 | 0.00 ± 0.00 | 0.040 ± 0.000 | - | - |", text)
        self.assertIn("Failed runs (left out): pos_1_s3.", text)
        self.assertIsNone(SWEEP.mean_std([]))

    def test_a_failed_run_is_reported_and_left_out(self):
        out = self.tmp / "sweep"
        commands = []

        def fake_station(cmd, **kw):
            """Stands in for gcs.py: the seed 0 runs fail, the others write a scorecard."""
            commands.append(cmd)
            name = cmd[cmd.index("--name") + 1]
            if name.endswith("_s0"):
                return subprocess.CompletedProcess(cmd, 3, "", "Traceback\nsomething broke\n")
            run = Path(cmd[cmd.index("--out-dir") + 1]) / name
            run.mkdir(parents=True)
            card = {"people_found": 2, "people_total": 3, "false_alarms": 1, "pin_error_m": 0.5}
            (run / "summary.json").write_text(json.dumps({"scorecard": card}))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        args = ["--replay", str(self.recording), "--param", "att", "--values", "0, 2", "--seeds", "0,1"]
        args += ["--fixed", "pos=0.5", "--out", str(out), "--", "--detector", "gt", "--min-hits", "3"]
        errors = io.StringIO()
        with (
            mock.patch.object(SWEEP.subprocess, "run", fake_station),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(errors),
        ):
            rc = SWEEP.main(args)
        self.assertEqual(rc, 1)  # not every run succeeded
        # one station per value and seed, one after the other, with the command line of the README
        self.assertEqual(len(commands), 4)
        expected = [sys.executable, str(ROOT / "scripts" / "gcs.py"), "--replay", str(self.recording.resolve())]
        expected += ["--headless", "--no-video", "--noise", "pos=0.5,att=2,seed=1"]
        expected += [
            "--out-dir",
            str(out),
            "--name",
            "att_2_s1",
            "--linger",
            "0",
            "--detector",
            "gt",
            "--min-hits",
            "3",
        ]
        self.assertEqual(commands[3], expected)
        self.assertEqual([c[c.index("--name") + 1] for c in commands], ["att_0_s0", "att_0_s1", "att_2_s0", "att_2_s1"])
        self.assertIn("att_0_s0: FAILED (exit code 3)", errors.getvalue())
        self.assertIn("something broke", errors.getvalue())
        with open(out / "sweep.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        runs = [(r["run"], r["ok"], r["pin_error_m"]) for r in rows]
        expected = [("att_0_s0", "0", ""), ("att_0_s1", "1", "0.5"), ("att_2_s0", "0", ""), ("att_2_s1", "1", "0.5")]
        self.assertEqual(runs, expected)
        text = (out / "sweep.md").read_text()
        self.assertIn("| 0 | 1 / 2 | 2.00 ± 0.00 of 3 | 1.00 ± 0.00 | 0.500 ± 0.000 | - | - |", text)
        self.assertIn("Failed runs (left out): att_0_s0, att_2_s0.", text)

    def test_bad_arguments_stop_the_sweep_before_its_first_run(self):
        base = ["--replay", str(self.recording), "--out", str(self.tmp / "sweep")]
        refused = {
            "invalid choice": [*base, "--param", "seed", "--values", "0,1"],  # the seed is not a noise level
            "write pos=NUMBER": [*base, "--param", "pos", "--values", "0,much"],
            "unknown noise setting 'gps'": [*base, "--param", "pos", "--values", "0", "--fixed", "gps=1"],
            "--seeds must be whole numbers": [*base, "--param", "pos", "--values", "0", "--seeds", "a"],
            "no such file": ["--replay", str(self.tmp / "missing.dgcrec"), "--param", "pos", "--values", "0"],
        }
        with mock.patch.object(SWEEP.subprocess, "run", side_effect=AssertionError("a station was started")):
            for message, args in refused.items():
                with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as caught:
                    SWEEP.main(args)
                self.assertEqual(caught.exception.code, 2, args)
                self.assertIn(message, errors.getvalue())
        self.assertFalse((self.tmp / "sweep").exists())


if __name__ == "__main__":
    unittest.main()
