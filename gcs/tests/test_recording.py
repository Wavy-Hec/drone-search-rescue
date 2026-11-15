"""Recording the link and replaying it: file format, recorder and reader, ``Receiver(record=...)``, the detector
worker's ``wait_idle`` and replays through the app (lossless and paced). ~10 s."""

import gzip
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from drone_gcs import app as app_module
from drone_gcs import protocol as P
from drone_gcs.app import AppConfig, GCSApp, ReplaySource
from drone_gcs.detector import ColorBlobDetector, DetectorWorker
from drone_gcs.geometry import C_BODY_TO_NADIR_CAM, PinholeCamera, intrinsics_from_fov
from drone_gcs.link import Receiver, Sender
from drone_gcs.recording import MAGIC, LinkRecorder, read_recording
from drone_gcs.sim.fake_world import FakeWorld
from drone_gcs.sim.flight import CarrotFlight, lawnmower
from drone_gcs.state import GCSState

from ._util import ROOT, rmtree, short_tmpdir

N_CAMERA = 20  # camera frames in the small recording of the app tests


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def make_small_recording(path, n_camera=N_CAMERA, seed=3, camera_size=(160, 120)):
    """Record a short flight over a small fake world to ``path``, through the real link (Sender -> UNIX socket ->
    ``Receiver(record=path)``), the way ``scripts/fake_drone.py`` streams it. Returns the recorded messages per type.

    The stream: HELLO, MISSION, GT_OBJECTS, then ``n_camera`` times [4 TELEMETRY, 2 SCAN, 1 CAMERA with depth]
    (20 / 10 / 5 Hz of simulated time), then END. The drone flies the first lawnmower line over a 16 x 16 m area
    and passes several objects with the default seed. The content is the same on every call (only the HELLO
    fields and the wall-clock times differ); nothing is dropped. Takes about 1.5 s.
    """
    area = (-8.0, 8.0, -8.0, 8.0)
    world = FakeWorld(area, seed=seed)
    z = world.hmax + 4.0
    route = lawnmower(area, 4.0, z)
    flight = CarrotFlight(route, speed=3.0)
    width, height = camera_size
    K = intrinsics_from_fov(width, height, 75.0)
    dt = 0.05  # one telemetry period [s]
    tmp = short_tmpdir("dgcs_mk")
    addr = f"unix:{tmp}/mk.sock"
    try:
        with Receiver(addr, record=path) as rx:
            tx = Sender(addr, name="fake_drone", camera_queue=n_camera)  # the queue holds every frame: no drops
            if not tx.wait_connected(5.0):
                raise RuntimeError("the sender did not connect")
            ground = float(np.median(world.ground))
            tx.send_mission(area, 0.25, route, z, name="small", ground_z=ground, z_range=(world.hmin, world.hmax))
            tx.send_gt_objects(world.gt_objects())
            t = 0.0
            for k in range(4 * n_camera):
                flight.step(dt)
                t = (k + 1) * dt
                tx.send_telemetry(t, flight.p, flight.quat, flight.v, wp_index=flight.wp_index)
                if k % 2 == 0:
                    origin = flight.p + np.array([0.0, 0.0, -0.1])
                    points, valid = world.scan_grid(origin, flight.yaw)
                    tx.send_scan(t, points, valid, origin)
                if k % 4 == 0:
                    # nadir camera on a gimbal: only its yaw follows the body
                    c, s = math.cos(flight.yaw), math.sin(flight.yaw)
                    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                    cam = PinholeCamera(K, width, height, R_yaw @ C_BODY_TO_NADIR_CAM, flight.p + [0.0, 0.0, -0.08])
                    rgb, depth = world.render(cam)
                    tx.send_camera(t, rgb, K, cam.t, cam.quat, depth, frame_id=k // 4)
            tx.send_end("complete", t, flush_timeout=10.0)
            arrived = wait_for(lambda: rx.counts[P.END] == 1, 10.0)
            dropped = tx.stats()["dropped"]
            tx.close()
            counts = dict(rx.counts)
        if not arrived or dropped:
            raise RuntimeError(f"incomplete recording: END arrived {arrived}, sender dropped {dropped}")
        return counts
    finally:
        world.close()
        rmtree(tmp)


def split_frame(frame):
    """``(prefix, header, payload)`` of one encoded frame."""
    hlen, _ = P.parse_prefix(frame[: P.PREFIX_SIZE])
    return frame[: P.PREFIX_SIZE], frame[P.PREFIX_SIZE : P.PREFIX_SIZE + hlen], frame[P.PREFIX_SIZE + hlen :]


def describe(msg):
    """Everything of a message except its arrays and its receive time."""
    return (msg.type, msg.seq, msg.drone_id, msg.sender, msg.wall_time, msg.conn_id, msg.nbytes, msg.fields)


def assert_same_messages(case, got, expected):
    """Same messages in the same order, arrays bit for bit."""
    case.assertEqual([describe(m) for m in got], [describe(m) for m in expected])
    for g, e in zip(got, expected, strict=True):
        case.assertEqual(list(g.arrays), list(e.arrays))
        for name, arr in e.arrays.items():
            case.assertEqual((g.arrays[name].dtype, g.arrays[name].shape), (arr.dtype, arr.shape), name)
            case.assertEqual(g.arrays[name].tobytes(), arr.tobytes(), name)


def run_app(cfg):
    """Run a station to its end in this process; returns ``(app, exit code)``. The test runner keeps its own
    signal handlers."""
    app = GCSApp(cfg)
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = app.run()
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)
    return app, code


# three frames of two senders, with the receive times and connection ids they are recorded under
POINTS = np.array([[1.0, 1.0, 0.1], [np.inf, np.inf, np.inf], [np.nan, 2.0, -0.0]], np.float32)
VALID = np.array([True, False, False])
RGB = np.random.default_rng(1).integers(0, 255, (24, 32, 3), dtype=np.uint8)
DEPTH = np.linspace(1.0, 2.0, 24 * 32, dtype=np.float32).reshape(24, 32)
FRAMES = [
    P.encode_bytes(P.TELEMETRY, {"t": 0.5, "pos": [1.0, 2.0, 3.0], "quat": [1, 0, 0, 0]}, seq=7, sender="a"),
    P.encode_bytes(P.SCAN, {"t": 0.6}, {"points": POINTS, "valid": VALID}, seq=0, drone_id=2, sender="b"),
    P.encode_bytes(P.CAMERA, {"t": 0.7, "frame_id": 4}, {"rgb": RGB, "depth": DEPTH}, seq=8, sender="a"),
]
WALL = [1000.0, 1000.125, 1002.5]  # time.time() of the three frames (exact in binary)
CONN = [1, 2, 1]
FILE_BYTES = MAGIC + b"".join(struct.pack("<dI", w, c) + f for w, c, f in zip(WALL, CONN, FRAMES, strict=True))


def write_frames(path, frames=FRAMES, wall=WALL, conn=CONN):
    rec = LinkRecorder(path)
    for w, c, frame in zip(wall, conn, frames, strict=True):
        rec.write(w, c, *split_frame(frame))
    rec.close()
    return rec


class TestRecordingFile(unittest.TestCase):
    def setUp(self):
        self.tmp = short_tmpdir("dgcs_rec")

    def tearDown(self):
        rmtree(self.tmp)

    def check_messages(self, msgs):
        self.assertEqual([m.type for m in msgs], [P.TELEMETRY, P.SCAN, P.CAMERA])
        self.assertEqual([m.recv_time for m in msgs], [0.0, 0.125, 2.5])  # seconds since the first record
        self.assertEqual([m.conn_id for m in msgs], [1, 2, 1])
        self.assertEqual([m.nbytes for m in msgs], [len(f) for f in FRAMES])
        self.assertEqual([(m.seq, m.drone_id, m.sender) for m in msgs], [(7, 0, "a"), (0, 2, "b"), (8, 0, "a")])
        self.assertEqual(msgs[0].fields, {"t": 0.5, "pos": [1.0, 2.0, 3.0], "quat": [1, 0, 0, 0]})
        self.assertEqual(msgs[2].fields, {"t": 0.7, "frame_id": 4})
        self.assertEqual(msgs[0].arrays, {})
        originals = {"points": POINTS, "valid": VALID, "rgb": RGB, "depth": DEPTH}
        arrays = {**msgs[1].arrays, **msgs[2].arrays}
        self.assertEqual(list(arrays), list(originals))
        for name, original in originals.items():
            self.assertEqual((arrays[name].dtype, arrays[name].shape), (original.dtype, original.shape), name)
            self.assertEqual(arrays[name].tobytes(), original.tobytes(), name)  # bit-exact, inf and NaN included
            self.assertTrue(arrays[name].flags.writeable, name)

    def test_round_trip_and_file_layout(self):
        path = f"{self.tmp}/link.dgcrec"
        rec = write_frames(path)
        self.assertEqual(rec.frames, 3)
        self.assertEqual(rec.nbytes, 8 + sum(12 + len(f) for f in FRAMES))
        self.assertEqual(Path(path).read_bytes(), FILE_BYTES)  # magic, then per frame: "<dI" and the frame as sent
        self.check_messages(list(read_recording(path)))

    def test_round_trip_through_gzip(self):
        path = f"{self.tmp}/link.dgcrec.gz"
        rec = write_frames(path)
        self.assertEqual((rec.frames, rec.nbytes), (3, len(FILE_BYTES)))  # counted before compression
        self.assertEqual(Path(path).read_bytes()[:2], b"\x1f\x8b")  # really a gzip file
        with gzip.open(path) as f:
            self.assertEqual(f.read(), FILE_BYTES)
        self.check_messages(list(read_recording(path)))

    def test_jpeg_buffers_are_decoded_unless_asked_not_to(self):
        path = f"{self.tmp}/jpeg.dgcrec"
        frame = P.encode_bytes(P.CAMERA, {"t": 1.0}, {"rgb": RGB}, encodings={"rgb": "jpeg"})
        write_frames(path, [frame], [5.0], [1])
        (decoded,) = read_recording(path)
        self.assertEqual((decoded.arrays["rgb"].shape, decoded.arrays["rgb"].dtype), ((24, 32, 3), np.uint8))
        (raw,) = read_recording(path, decode_jpeg=False)
        self.assertEqual(raw.arrays["rgb"].tobytes()[:2], b"\xff\xd8")  # the JPEG file itself

    def test_write_after_close_is_ignored(self):
        path = f"{self.tmp}/closed.dgcrec"
        rec = write_frames(path)
        rec.write(2000.0, 1, *split_frame(FRAMES[0]))
        rec.close()
        self.assertEqual(rec.frames, 3)
        self.assertEqual(Path(path).read_bytes(), FILE_BYTES)

    @unittest.skipUnless(os.path.exists("/dev/full"), "needs /dev/full")
    def test_a_full_disk_ends_the_recording_without_raising(self):
        big = P.encode_bytes(P.SCAN, {}, {"data": np.zeros(100_000, np.uint8)})  # larger than the file buffer
        rec = LinkRecorder("/dev/full")
        with self.assertLogs("drone_gcs.recording", level="ERROR") as logs:
            rec.write(1.0, 1, *split_frame(big))  # the receiver's reader thread must not see an exception
        self.assertEqual((rec.frames, len(logs.records)), (0, 1))
        with self.assertNoLogs("drone_gcs.recording", level="ERROR"):
            rec.write(2.0, 1, *split_frame(big))  # the recording is over
            rec.close()
        rec = LinkRecorder("/dev/full")
        rec.write(1.0, 1, *split_frame(FRAMES[0]))  # small: still in the buffer, the error comes with close()
        with self.assertLogs("drone_gcs.recording", level="ERROR"):
            rec.close()

    def test_writers_in_several_threads_do_not_mix_their_records(self):
        path = f"{self.tmp}/threads.dgcrec"
        rec = LinkRecorder(path)
        n_threads, n_frames = 4, 60

        def writer(conn_id):
            for k in range(n_frames):  # a payload larger than the file buffer: every record is several writes
                data = np.full(40_000, conn_id * 1000 + k, np.int32)
                rec.write(float(k), conn_id, *split_frame(P.encode_bytes(P.SCAN, {"k": k}, {"data": data}, seq=k)))

        threads = [threading.Thread(target=writer, args=(c,)) for c in range(1, n_threads + 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rec.close()
        self.assertEqual(rec.frames, n_threads * n_frames)
        seen = {c: [] for c in range(1, n_threads + 1)}
        for m in read_recording(path):
            self.assertTrue((m.arrays["data"] == m.conn_id * 1000 + m.fields["k"]).all())
            seen[m.conn_id].append(m.seq)
        for c in seen:
            self.assertEqual(seen[c], list(range(n_frames)), c)  # every connection's frames, in its own order

    def test_truncated_last_record_ends_the_iteration_with_one_warning(self):
        third = len(FILE_BYTES) - 12 - len(FRAMES[2])  # offset of the third record
        cuts = {
            "in the record head": third + 5,
            "in the frame prefix": third + 12 + 7,
            "in the header": third + 12 + 16 + 3,
            "in the payload": len(FILE_BYTES) - 1,
        }
        for where, cut in cuts.items():
            path = f"{self.tmp}/cut.dgcrec"
            Path(path).write_bytes(FILE_BYTES[:cut])
            with self.assertLogs("drone_gcs.recording", level="WARNING") as logs:
                msgs = list(read_recording(path))
            self.assertEqual(len(logs.records), 1, where)
            self.assertEqual([m.type for m in msgs], [P.TELEMETRY, P.SCAN], where)
            self.assertEqual(msgs[1].arrays["points"].tobytes(), POINTS.tobytes(), where)
        # a file that ends after a complete record is simply shorter: no warning
        Path(path).write_bytes(FILE_BYTES[:third])
        with self.assertNoLogs("drone_gcs.recording", level="WARNING"):
            self.assertEqual([m.type for m in read_recording(path)], [P.TELEMETRY, P.SCAN])
        Path(path).write_bytes(MAGIC)
        self.assertEqual(list(read_recording(path)), [])

    def test_truncated_gzip_file_yields_the_records_before_the_cut(self):
        path = f"{self.tmp}/cut.dgcrec.gz"
        noise = np.random.default_rng(0).random((40, 2000))  # does not compress: half the file = half the frames
        frames = [P.encode_bytes(P.SCAN, {"k": k}, {"noise": noise[k]}) for k in range(40)]
        write_frames(path, frames, [float(k) for k in range(40)], [1] * 40)
        whole = Path(path).read_bytes()
        Path(path).write_bytes(whole[: len(whole) // 2])
        with self.assertLogs("drone_gcs.recording", level="WARNING") as logs:
            msgs = list(read_recording(path))
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(10 <= len(msgs) < 40, len(msgs))
        for k, m in enumerate(msgs):  # the first records, complete and in order
            self.assertEqual(m.fields, {"k": k})
            self.assertEqual(m.arrays["noise"].tobytes(), noise[k].tobytes())

    def test_a_file_that_is_not_a_recording_raises(self):
        path = f"{self.tmp}/other.dgcrec"
        for content in (b"", b"DGCREC", FRAMES[0], b"DGCREC2\n" + FILE_BYTES[8:]):
            Path(path).write_bytes(content)
            with self.assertRaisesRegex(P.ProtocolError, "not a drone_gcs link recording"):
                next(read_recording(path))


class TestReceiverRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = short_tmpdir("dgcs_rec")
        self.addr = f"unix:{self.tmp}/l.sock"
        self.path = f"{self.tmp}/link.dgcrec"

    def tearDown(self):
        rmtree(self.tmp)

    def test_a_station_that_cannot_listen_leaves_an_old_recording_alone(self):
        Path(self.path).write_bytes(b"an earlier recording")
        with Receiver(self.addr):  # a station is listening there already
            second = Receiver(self.addr, record=self.path)
            with self.assertRaises(RuntimeError):
                second.start()
        self.assertEqual(Path(self.path).read_bytes(), b"an earlier recording")
        self.assertIsNone(second.recorder)

    def test_recorded_sequence_equals_the_drained_sequence(self):
        drained = []
        with Receiver(self.addr, record=self.path) as rx:
            tx = Sender(self.addr, name="t")
            self.assertTrue(tx.wait_connected(3))
            tx.send_mission((0, 10, 0, 5), 0.5, np.array([[0, 0], [10, 0]], float), 4.0, name="m")
            tx.send_telemetry(0.1, [1, 2, 3], [1, 0, 0, 0], [0.5, 0, 0], wp_index=1)
            tx.send_scan(0.2, POINTS, origin=[1, 1, 3])
            tx.send_camera(0.3, RGB, np.eye(3), [1, 1, 3], [0, 1, 0, 0], DEPTH, frame_id=5)
            tx.send_gt_objects([{"class": "car", "pos": [1, 2, 0]}])
            tx.send_end("complete", 0.4)
            self.assertTrue(wait_for(lambda: rx.counts[P.END] == 1))
            drained = rx.drain()
            tx.close()
            self.assertEqual(rx.recorder.frames, 7)
            self.assertEqual(rx.recorder.nbytes, 8 + 7 * 12 + rx.total_bytes)
        recorded = list(read_recording(self.path))  # the receiver finished the file when it closed
        expected = [P.HELLO, P.MISSION, P.TELEMETRY, P.SCAN, P.CAMERA, P.GT_OBJECTS, P.END]
        self.assertEqual([m.type for m in drained], expected)
        assert_same_messages(self, recorded, drained)
        # receive times: seconds since the first frame, as the receiver's own clock measured them
        self.assertEqual(recorded[0].recv_time, 0.0)
        for rec, live in zip(recorded, drained, strict=True):
            self.assertAlmostEqual(rec.recv_time, live.recv_time - drained[0].recv_time, delta=0.5)

    def test_frames_the_inbound_queue_drops_are_still_recorded(self):
        with Receiver(self.addr, record=self.path, camera_queue=3) as rx:
            tx = Sender(self.addr, camera_queue=10)
            for k in range(6):  # the station is busy and does not drain: its queue keeps the 3 newest frames
                tx.send_camera(float(k), RGB, np.eye(3), [0, 0, 5], [0, 1, 0, 0], frame_id=k)
            self.assertTrue(wait_for(lambda: rx.counts[P.CAMERA] == 6))
            drained = rx.drain()
            tx.close()
        self.assertEqual([m.fields["frame_id"] for m in drained if m.type == P.CAMERA], [3, 4, 5])
        recorded = [m.fields["frame_id"] for m in read_recording(self.path) if m.type == P.CAMERA]
        self.assertEqual(recorded, [0, 1, 2, 3, 4, 5])

    def test_a_frame_that_does_not_decode_is_not_recorded(self):
        bad_header = b"this is not JSON"
        with Receiver(self.addr, record=self.path) as rx:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self.addr[5:])
            s.sendall(FRAMES[0] + P.PREFIX.pack(P.MAGIC, len(bad_header), 0) + bad_header)
            self.assertTrue(wait_for(lambda: len(rx.errors) == 1))
            s.close()
        self.assertEqual([m.type for m in read_recording(self.path)], [P.TELEMETRY])

    def test_no_record_no_file(self):
        with Receiver(self.addr) as rx:
            self.assertIsNone(rx.recorder)
        self.assertEqual(os.listdir(self.tmp), [])


class TestWaitIdle(unittest.TestCase):
    def test_wait_idle_returns_when_the_submitted_frames_are_done(self):
        class Slow:
            def detect(self, rgb):
                time.sleep(0.1)
                return []

        wk = DetectorWorker(Slow())
        try:
            self.assertTrue(wk.wait_idle(0.0))  # nothing submitted
            img = np.zeros((2, 2, 3), np.uint8)
            wk.submit(img, "a", key=0)
            wk.submit(img, "b", key=1)
            self.assertFalse(wk.wait_idle(0.02))  # 2 x 0.1 s of work left
            self.assertEqual(wk.poll(), [])
            t0 = time.monotonic()
            self.assertTrue(wk.wait_idle(5.0))
            self.assertLess(time.monotonic() - t0, 2.0)  # woken when the work was done, not by the timeout
            self.assertEqual([r[0] for r in wk.poll()], ["a", "b"])  # the results are ready when it returns
            self.assertFalse(wk.busy)
            self.assertEqual((wk.n_done, wk.n_dropped), (2, 0))
        finally:
            wk.close()


class TestReplaySource(unittest.TestCase):
    def setUp(self):
        self.tmp = short_tmpdir("dgcs_rec")
        self.path = f"{self.tmp}/four.dgcrec"
        frames = [P.encode_bytes(P.TELEMETRY, {"k": k}) for k in range(4)]
        write_frames(self.path, frames, [1000.0, 1000.125, 1000.25, 1005.0], [1, 1, 1, 1])
        self.nbytes = [len(f) for f in frames]

    def tearDown(self):
        rmtree(self.tmp)

    def test_paced_drain_hands_out_the_messages_that_are_due(self):
        clock = [50.0]
        with mock.patch.object(app_module.time, "monotonic", lambda: clock[0]):
            src = ReplaySource(self.path, speed=10.0).start()  # 10x: recorded 0.125 s -> due after 0.0125 s
            self.assertTrue(src.connected)
            clock[0] = 77.0  # the replay clock starts at the first drain, not at start()
            self.assertEqual([m.fields["k"] for m in src.drain()], [0])
            clock[0] = 77.0 + 0.012
            self.assertEqual(src.drain(), [])
            clock[0] = 77.0 + 0.013
            self.assertEqual([m.fields["k"] for m in src.drain()], [1])
            clock[0] = 77.0 + 0.4  # recorded 4.0 s: the third is due (0.25 s), the last (5.0 s) is not
            self.assertEqual([m.fields["k"] for m in src.drain()], [2])
            self.assertFalse(src.finished)
            self.assertEqual(src.rates(window=2.0), (1.5, sum(self.nbytes[:3]) / 2.0))  # 3 messages in the last 2 s
            clock[0] = 77.0 + 0.51
            self.assertEqual([m.fields["k"] for m in src.drain()], [3])
            self.assertTrue(src.finished)
            self.assertFalse(src.connected)
            self.assertEqual(src.drain(), [])
            src.close()
        self.assertEqual(dict(src.counts), {P.TELEMETRY: 4})
        self.assertEqual(src.total_bytes, sum(self.nbytes))
        self.assertEqual((src.dropped, src.errors, src.address_str), ({}, [], "replay:four.dgcrec"))

    def test_next_message_walks_the_recording(self):
        src = ReplaySource(self.path).start()
        got = []
        while not src.finished:
            got.append(src.next_message().fields["k"])
        self.assertEqual(got, [0, 1, 2, 3])  # finished right after the last message
        self.assertIsNone(src.next_message())
        src.close()

    def test_start_raises_for_a_bad_file(self):
        Path(self.path).write_bytes(b"something else")
        with self.assertRaises(P.ProtocolError):
            ReplaySource(self.path).start()
        with self.assertRaises(OSError):
            ReplaySource(f"{self.tmp}/missing.dgcrec").start()


class SlowBlobDetector(ColorBlobDetector):
    """The stand-in detector, 20 ms slower per frame: a station that did not wait for it would skip frames."""

    def detect(self, rgb):
        time.sleep(0.02)
        return super().detect(rgb)


def process_in_order(recording):
    """The reference for a lossless replay: every message applied in file order, every camera frame detected
    and its detections applied before the next message."""
    state = GCSState()
    detector = ColorBlobDetector()
    for msg in read_recording(recording):
        frame = state.apply(msg)
        if frame is not None:
            state.apply_detections(frame, detector.detect(frame.rgb), 0.0)
    return state


class TestReplayThroughApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = short_tmpdir("dgcs_rep")
        cls.recording = Path(cls.tmp) / "small.dgcrec"
        cls.counts = make_small_recording(cls.recording)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp)

    def replay(self, name, recording=None, **kw):
        """Replay headless without video; returns the finished app and its summary.json."""
        settings = dict(headless=True, video=False, detector="blob", linger=0.0, max_time=60.0)
        cfg = AppConfig(replay=recording or self.recording, out_dir=Path(self.tmp), name=name, **(settings | kw))
        app, code = run_app(cfg)
        self.assertEqual(code, 0)
        return app, json.loads((app.run_dir / "summary.json").read_text())

    def test_small_recording(self):
        expected = {P.HELLO: 1, P.MISSION: 1, P.GT_OBJECTS: 1, P.END: 1}
        expected |= {P.TELEMETRY: 4 * N_CAMERA, P.SCAN: 2 * N_CAMERA, P.CAMERA: N_CAMERA}
        self.assertEqual(self.counts, expected)
        msgs = list(read_recording(self.recording))
        self.assertEqual(len(msgs), sum(expected.values()))
        self.assertEqual([m.type for m in msgs[:3]], [P.HELLO, P.MISSION, P.GT_OBJECTS])
        self.assertEqual(msgs[-1].type, P.END)
        self.assertEqual([m.fields["frame_id"] for m in msgs if m.type == P.CAMERA], list(range(N_CAMERA)))

    def test_lossless_replay_skips_no_frame_and_is_repeatable(self):
        with mock.patch.object(app_module, "make_detector", return_value=SlowBlobDetector()):
            slow, summary = self.replay("slow")
        fast, _ = self.replay("fast")
        for app in (slow, fast):
            self.assertEqual(app.quit_reason, "mission end (complete)")
            self.assertEqual((app.worker.n_submitted, app.worker.n_done, app.worker.n_dropped), (N_CAMERA, N_CAMERA, 0))
        gcs = summary["gcs"]
        self.assertEqual((gcs["detector_frames"], gcs["detector_dropped_frames"]), (N_CAMERA, 0))
        self.assertEqual(gcs["replay"], {"file": "small.dgcrec", "speed": 0.0})
        self.assertIsNone(gcs["recording"])
        self.assertEqual(gcs["link_messages"], self.counts)
        self.assertEqual(gcs["link_bytes"], self.recording.stat().st_size - 8 - 12 * sum(self.counts.values()))
        # the same as plain in-order processing: each frame's detections were applied before the next message
        ref = process_in_order(self.recording)
        objects = [ob.to_dict() for ob in ref.tracker.objects]
        self.assertGreaterEqual(len(ref.tracker.confirmed()), 2)  # the flight really pins objects
        self.assertGreater(ref.elevation.n_points, 1000)
        for app in (slow, fast):
            self.assertEqual([ob.to_dict() for ob in app.state.tracker.objects], objects)
            np.testing.assert_array_equal(app.state.elevation.mean(), ref.elevation.mean())
            np.testing.assert_array_equal(app.state.elevation.count, ref.elevation.count)
        # and so are the reports of two replays, whatever the detector's speed
        self.assertEqual((slow.run_dir / "detections.json").read_text(), (fast.run_dir / "detections.json").read_text())
        for name in ("elevation_mean.npy", "hit_count.npy", "photo_filled.npy"):
            np.testing.assert_array_equal(
                np.load(slow.run_dir / "maps" / name), np.load(fast.run_dir / "maps" / name), err_msg=name
            )

    def test_lossless_replay_reads_every_mission_however_slow_the_station(self):
        parts = [Path(self.tmp) / f"part{k}.dgcrec" for k in range(2)]
        for part in parts:
            make_small_recording(part, n_camera=3)
        both = Path(self.tmp) / "two_missions.dgcrec"
        both.write_bytes(parts[0].read_bytes() + parts[1].read_bytes()[len(MAGIC) :])  # mission A, then B
        render = app_module.Dashboard.render

        def slow_render(dashboard, *args, **kwargs):  # slower than a frame at 50 fps: every round runs behind
            time.sleep(0.03)
            return render(dashboard, *args, **kwargs)

        with mock.patch.object(app_module.Dashboard, "render", slow_render):
            app, summary = self.replay("two_missions", both, fps=50.0)
        self.assertEqual(app.quit_reason, "mission end (complete)")
        self.assertEqual(len(app.reports), 1)  # mission A archived and B reported: the whole recording was read
        ids = [m.fields["mission_id"] for m in read_recording(both) if m.type == P.MISSION]
        self.assertEqual(summary["mission"]["mission_id"], ids[1])
        self.assertEqual(app.worker.n_done, 6)

    def test_lossless_replay_does_not_wait_while_input_is_left(self):
        naps = []
        with mock.patch.object(app_module.time, "sleep", naps.append):
            app, _ = self.replay("no_naps")
        self.assertEqual(app.worker.n_done, N_CAMERA)
        # the loop renders a dashboard frame between batches of messages, but only its last round (the recording
        # is finished) may wait for the next frame
        self.assertLessEqual(len(naps), 1, naps)

    def test_paced_replay_finishes_and_writes_a_report(self):
        app, summary = self.replay("paced", replay_speed=50.0)
        self.assertEqual(app.quit_reason, "mission end (complete)")
        self.assertEqual(summary["gcs"]["replay"], {"file": "small.dgcrec", "speed": 50.0})
        self.assertEqual(summary["gcs"]["link_messages"], self.counts)  # every message went through the station
        self.assertEqual(summary["messages"]["camera"], N_CAMERA)
        self.assertEqual(app.worker.n_submitted, N_CAMERA)  # (a slow machine may skip some of them, as on a live link)
        self.assertTrue((app.run_dir / "final_map_elevation.png").exists())

    def test_recording_without_end(self):
        cut = Path(self.tmp) / "no_end.dgcrec"
        cut.write_bytes(self.recording.read_bytes()[:-10])  # the station was killed while it wrote the END frame
        with self.assertLogs("drone_gcs.recording", level="WARNING"):
            app, summary = self.replay("no_end", cut)
        self.assertEqual(app.quit_reason, "end of recording")  # headless: exit_on_end by default
        self.assertEqual(summary["gcs"]["link_messages"], {k: v for k, v in self.counts.items() if k != P.END})
        self.assertEqual(app.worker.n_done, N_CAMERA)
        with self.assertLogs("drone_gcs.recording", level="WARNING"):
            app, _ = self.replay("no_end_stay", cut, exit_on_end=False, max_time=0.5)
        self.assertEqual(app.quit_reason, "max time")  # the station stays up after the recording

    def test_bad_replay_file_is_refused(self):
        bad = Path(self.tmp) / "bad.dgcrec"
        bad.write_bytes(b"not a recording")
        for path in (bad, Path(self.tmp) / "missing.dgcrec"):
            app, code = run_app(AppConfig(replay=path, out_dir=Path(self.tmp), name="bad", headless=True, video=False))
            self.assertEqual(code, 2, path)
            self.assertFalse((app.run_dir / "summary.json").exists())

    def test_record_and_replay_cannot_be_combined(self):
        with self.assertRaises(ValueError):
            GCSApp(AppConfig(record="", replay=self.recording, out_dir=Path(self.tmp), name="both"))
        self.assertFalse((Path(self.tmp) / "both").exists())

    def test_command_line(self):
        script = [sys.executable, str(ROOT / "scripts" / "gcs.py"), "--headless", "--no-video", "--detector", "blob"]
        script += ["--out-dir", self.tmp, "--linger", "0"]
        out = subprocess.run(
            [*script, "--name", "cli", "--replay", str(self.recording), "--replay-speed", "20"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        summary = json.loads((Path(self.tmp) / "cli" / "summary.json").read_text())
        self.assertEqual(summary["gcs"]["replay"], {"file": "small.dgcrec", "speed": 20.0})
        self.assertEqual(summary["gcs"]["quit_reason"], "mission end (complete)")
        refused = {
            "not allowed with argument": ["--record", "--replay", str(self.recording)],
            "--replay-speed must be": ["--replay", str(self.recording), "--replay-speed", "-1"],
        }
        for message, args in refused.items():
            out = subprocess.run(
                [*script, "--name", "refused", *args], cwd=ROOT, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(out.returncode, 2, args)  # an argparse error, before anything is created
            self.assertIn(message, out.stderr)
            self.assertFalse((Path(self.tmp) / "refused").exists())


class TestRecordThroughApp(unittest.TestCase):
    def setUp(self):
        self.tmp = short_tmpdir("dgcs_rec")

    def tearDown(self):
        rmtree(self.tmp)

    def test_station_records_the_link_it_listens_on(self):
        addr = f"unix:{self.tmp}/live.sock"
        tx = Sender(addr, reconnect_max=0.2)  # queued until the station listens
        tx.send_mission((-5, 5, -5, 5), 0.5, mission_id="A", name="A")
        for k in range(3):
            tx.send_telemetry(float(k), [k, 0, 5], [1, 0, 0, 0])
        tx.send_end("complete", 3.0, flush_timeout=0)
        cfg = AppConfig(addr=addr, out_dir=Path(self.tmp), name="live", headless=True, video=False, detector="none")
        cfg.linger, cfg.max_time, cfg.record = 0.0, 30.0, ""
        app, _ = run_app(cfg)
        tx.close(0.2)
        self.assertEqual(app.quit_reason, "mission end (complete)")
        self.assertEqual(app.record_path, app.run_dir / "link.dgcrec")  # the default file
        types = [m.type for m in read_recording(app.record_path)]
        self.assertEqual(types, [P.HELLO, P.MISSION, P.TELEMETRY, P.TELEMETRY, P.TELEMETRY, P.END])
        summary = json.loads((app.run_dir / "summary.json").read_text())
        self.assertEqual(summary["gcs"]["recording"], "link.dgcrec")
        self.assertIsNone(summary["gcs"]["replay"])
        # the recording replays to the same mission state
        again, _ = run_app(
            AppConfig(replay=app.record_path, out_dir=Path(self.tmp), name="again", headless=True, video=False,
                      detector="none", linger=0.0, max_time=30.0)
        )  # fmt: skip
        self.assertEqual(again.quit_reason, "mission end (complete)")
        self.assertEqual(again.state.trajectory, app.state.trajectory)
        self.assertEqual(again.state.mission, app.state.mission)

    def test_record_path(self):
        base = dict(addr=f"unix:{self.tmp}/p.sock", out_dir=Path(self.tmp), headless=True)
        named = GCSApp(AppConfig(name="named", record=f"{self.tmp}/flight.dgcrec.gz", **base))
        self.assertEqual(named.record_path, Path(self.tmp) / "flight.dgcrec.gz")
        off = GCSApp(AppConfig(name="off", **base))
        self.assertIsNone(off.record_path)
        off.receiver.start()
        off.receiver.close()
        self.assertIsNone(off.receiver.recorder)  # a station without --record writes no recording
        self.assertEqual(list(off.run_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
