import os
import socket
import time
import unittest

import numpy as np

from drone_gcs import protocol as P
from drone_gcs.link import BoundedMessageQueue, Receiver, Sender, parse_address

from ._util import rmtree, short_tmpdir


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def collect(rx, n, timeout=5.0):
    out = []
    t0 = time.monotonic()
    while len(out) < n and time.monotonic() - t0 < timeout:
        out += rx.drain()
        time.sleep(0.01)
    return out


class _Item:
    def __init__(self, t, n=10):
        self.type, self.nbytes = t, n


class TestQueue(unittest.TestCase):
    def test_per_type_limit_drops_oldest_and_keeps_order(self):
        q = BoundedMessageQueue({P.CAMERA: 2}, default_limit=100)
        items = [_Item(P.CAMERA), _Item(P.TELEMETRY), _Item(P.CAMERA), _Item(P.CAMERA), _Item(P.END)]
        for it in items:
            q.put(it)
        self.assertEqual(q.dropped[P.CAMERA], 1)
        out = q.drain()
        self.assertEqual([o.type for o in out], [P.TELEMETRY, P.CAMERA, P.CAMERA, P.END])
        self.assertIs(out[1], items[2])

    def test_byte_budget_never_drops_control(self):
        q = BoundedMessageQueue({}, default_limit=100, max_bytes=100)
        q.put(_Item(P.MISSION, 80))
        q.put(_Item(P.CAMERA, 50))
        q.put(_Item(P.SCAN, 50))
        types = [o.type for o in q.drain()]
        self.assertIn(P.MISSION, types)
        self.assertNotIn(P.CAMERA, types)

    def test_taken_item_stays_pending_until_done(self):
        # Sender.flush()/idle() rely on this: no moment where the queue is empty and nothing counts as in flight
        q = BoundedMessageQueue()
        a, b = _Item(P.TELEMETRY), _Item(P.END)
        q.put(a)
        q.put(b)
        self.assertIs(q.peek(), a)
        self.assertIs(q.take(0), a)
        self.assertEqual((len(q), q.pending()), (1, 2))
        q.task_done()
        self.assertIs(q.take(0), b)
        q.push_front(b)  # failed write: re-queued before task_done
        q.task_done()
        self.assertEqual(q.pending(), 1)
        self.assertIs(q.take(0), b)
        q.task_done()
        self.assertEqual(q.pending(), 0)
        self.assertIsNone(q.take(0))


class TestAddress(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_address("tcp://127.0.0.1:5000")[0], socket.AF_INET)
        self.assertEqual(parse_address(":5000")[1], ("127.0.0.1", 5000))
        fam, path, s = parse_address("unix:/tmp/x.sock")
        self.assertEqual((fam, path, s), (socket.AF_UNIX, "/tmp/x.sock", "unix:/tmp/x.sock"))
        with self.assertRaises(ValueError):
            parse_address("unix:/" + "x" * 200)


class TestLink(unittest.TestCase):
    def setUp(self):
        self.tmp = short_tmpdir()
        self.addr = f"unix:{self.tmp}/l.sock"

    def tearDown(self):
        rmtree(self.tmp)

    def test_messages_arrive_in_order_with_arrays(self):
        with Receiver(self.addr) as rx:
            tx = Sender(self.addr, name="t")
            self.assertTrue(tx.wait_connected(3))
            path = np.array([[0, 0], [10, 0], [10, 5]], float)
            self.assertTrue(tx.send_mission((0, 10, 0, 5), 0.5, path, 4.0, name="m"))
            self.assertTrue(tx.send_telemetry(0.1, [1, 2, 3], [1, 0, 0, 0], [0.5, 0, 0], wp_index=1, carrot=[2, 2, 3]))
            pts = np.array([[1, 1, 0.1], [np.inf, np.inf, np.inf]], np.float32)
            self.assertTrue(tx.send_scan(0.2, pts, origin=[1, 1, 3]))
            rgb = np.random.default_rng(1).integers(0, 255, (24, 32, 3), dtype=np.uint8)
            depth = np.full((24, 32), 3.0, np.float32)
            self.assertTrue(tx.send_camera(0.3, rgb, np.eye(3), [1, 1, 3], [0, 1, 0, 0], depth, frame_id=5))
            self.assertTrue(tx.send_gt_objects([{"class": "car", "pos": [1, 2, 0]}, ("tent", (3, 4, 0))]))
            terrain = np.arange(12, dtype=np.float32).reshape(3, 4)
            self.assertTrue(tx.send_gt_terrain(terrain, -1.0, -2.0, 0.5))
            self.assertTrue(tx.send_end("complete", 0.4))
            msgs = collect(rx, 8)
            tx.close()
            types = [m.type for m in msgs]
            # everything in call order, the latched MISSION and ground truth included
            expected = [P.HELLO, P.MISSION, P.TELEMETRY, P.SCAN, P.CAMERA, P.GT_OBJECTS, P.GT_TERRAIN, P.END]
            self.assertEqual(types, expected)
            by = {m.type: m for m in msgs}
            mid = by[P.MISSION].fields["mission_id"]
            for t in (P.TELEMETRY, P.SCAN, P.CAMERA, P.END):  # data and END carry the mission they belong to
                self.assertEqual(by[t].fields["mission_id"], mid, t)
            self.assertNotIn("mission_id", by[P.HELLO].fields)
            np.testing.assert_allclose(by[P.MISSION].arrays["planned_path"][:, 2], 4.0)
            self.assertEqual(by[P.MISSION].fields["area"], {"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 5.0})
            cam = by[P.CAMERA]
            np.testing.assert_array_equal(cam.arrays["rgb"], rgb)
            np.testing.assert_array_equal(cam.arrays["depth"], depth)
            self.assertEqual(cam.fields["frame_id"], 5)
            self.assertEqual(cam.fields["convention"], "ros_optical")
            self.assertTrue(np.isinf(by[P.SCAN].arrays["points"][1]).all())
            self.assertEqual(by[P.GT_OBJECTS].fields["objects"][1]["class"], "tent")
            np.testing.assert_array_equal(by[P.GT_TERRAIN].arrays["heights"], terrain)
            self.assertEqual(by[P.GT_TERRAIN].fields, {"x_min": -1.0, "y_min": -2.0, "resolution": 0.5})

    def test_sender_without_gcs_never_blocks_or_raises(self):
        tx = Sender(f"unix:{self.tmp}/nobody.sock", camera_queue=2)
        rgb = np.zeros((480, 640, 3), np.uint8)
        t0 = time.perf_counter()
        for i in range(50):
            self.assertTrue(tx.send_camera(i, rgb, np.eye(3), [0, 0, 1], [1, 0, 0, 0]))
            self.assertTrue(tx.send_telemetry(i, [0, 0, 1], [1, 0, 0, 0]))
        dt = time.perf_counter() - t0
        self.assertLess(dt, 1.0)
        s = tx.stats()
        self.assertFalse(s["connected"])
        self.assertEqual(s["dropped_by_type"].get(P.CAMERA), 48)
        # invalid input returns False instead of raising
        self.assertFalse(tx.send_telemetry(0, [1, 2], [1, 0, 0, 0]))
        self.assertFalse(tx.send_camera(0, np.zeros((4, 4), np.uint8), np.eye(3), [0, 0, 0], [1, 0, 0, 0]))
        self.assertGreaterEqual(tx.stats()["send_errors"], 2)
        tx.close(flush_timeout=0.1)
        Sender("unix:/nonexistent_dir_xyz/a.sock").close(0.05)  # bad path: no exception

    def test_gcs_starting_late_and_restarting_gets_latched_mission(self):
        tx = Sender(self.addr, reconnect_max=0.3)
        tx.send_mission((0, 1, 0, 1), 0.1, name="late")
        tx.send_telemetry(0, [0, 0, 1], [1, 0, 0, 0])
        time.sleep(0.3)
        with Receiver(self.addr) as rx:
            msgs = collect(rx, 3, 5)
            self.assertEqual([m.type for m in msgs][:3], [P.HELLO, P.MISSION, P.TELEMETRY])
        # GCS restarts: the sender reconnects and re-sends the mission (latched)
        with Receiver(self.addr) as rx2:
            ok = wait_for(lambda: tx.send_telemetry(1, [0, 0, 1], [1, 0, 0, 0]) and rx2.counts[P.MISSION] >= 1, 8)
            self.assertTrue(ok)
            msgs = collect(rx2, 3, 3)
            self.assertIn(P.MISSION, [m.type for m in msgs])
            self.assertEqual(msgs[1].fields["name"], "late")
        tx.close(0.2)

    def test_mission_change_while_gcs_absent_keeps_call_order(self):
        # MISSION B must neither overwrite A in the latched slot nor jump ahead of A's queued data
        tx = Sender(self.addr, reconnect_max=0.3)
        tx.send_mission((-10, 10, -10, 10), 0.25, mission_id="A", name="A")
        for k in range(3):
            tx.send_telemetry(k, [k, 0, 5], [1, 0, 0, 0])
        tx.send_end("complete", 3.0, flush_timeout=0.2)  # nobody listening: the flush times out
        tx.send_mission((-10, 10, -10, 10), 0.25, mission_id="B", name="B")
        for k in range(2):
            tx.send_telemetry(10 + k, [k, 1, 5], [1, 0, 0, 0])
        time.sleep(0.2)
        with Receiver(self.addr) as rx:
            msgs = collect(rx, 9, 6)
        tx.close(0.2)
        seq = [(m.type, m.fields.get("mission_id")) for m in msgs]
        A, B, T = "A", "B", P.TELEMETRY
        self.assertEqual(
            seq, [(P.HELLO, None), (P.MISSION, A), (T, A), (T, A), (T, A), (P.END, A), (P.MISSION, B), (T, B), (T, B)]
        )

    def test_gcs_restart_mid_mission_gets_the_mission_of_the_queued_data(self):
        tx = Sender(self.addr, reconnect_min=0.05, reconnect_max=0.2)
        with Receiver(self.addr) as rx:
            self.assertTrue(tx.wait_connected(3))
            tx.send_mission((0, 10, 0, 10), 0.25, mission_id="A", name="A")
            tx.send_telemetry(0, [0, 0, 5], [1, 0, 0, 0])
            self.assertEqual(len(collect(rx, 3)), 3)
        # GCS gone: the sender notices on its next write (that one message is lost), then queues
        tx.send_telemetry(0.5, [0, 0, 5], [1, 0, 0, 0])
        self.assertTrue(wait_for(lambda: not tx.connected, 3))
        for k in range(1, 4):
            tx.send_telemetry(k, [k, 0, 5], [1, 0, 0, 0])
        tx.send_end("complete", 4.0, flush_timeout=0)
        tx.send_mission((0, 10, 0, 10), 0.25, mission_id="B", name="B")
        tx.send_telemetry(10, [0, 1, 5], [1, 0, 0, 0])
        with Receiver(self.addr) as rx2:
            msgs = collect(rx2, 8, 6)
        tx.close(0.2)
        seq = [(m.type, m.fields.get("mission_id"), m.fields.get("t")) for m in msgs]
        T = P.TELEMETRY
        self.assertEqual(
            seq,
            [(P.HELLO, None, None), (P.MISSION, "A", None), (T, "A", 1.0), (T, "A", 2.0), (T, "A", 3.0),
             (P.END, "A", 4.0), (P.MISSION, "B", None), (T, "B", 10.0)],
        )  # fmt: skip

    def test_extra_fields_are_copied_at_call_time(self):
        # the simulator reuses its buffers, so extras must be copied before the sender thread serializes them
        tx = Sender(self.addr)
        action = np.array([0.1, 0.2, 0.3, 0.4])
        size = np.array([4.5, 1.8, 1.5])
        self.assertTrue(tx.send_telemetry(1.0, [0, 0, 5], [1, 0, 0, 0], action=action))
        self.assertTrue(tx.send_gt_objects([{"class": "car", "pos": [1, 2, 0], "size": size}]))
        action[:] = 9.0
        size[:] = 0.0
        with Receiver(self.addr) as rx:
            msgs = collect(rx, 3, 5)
        tx.close(0.2)
        by = {m.type: m for m in msgs}
        self.assertEqual(by[P.TELEMETRY].fields["action"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(by[P.GT_OBJECTS].fields["objects"][0]["size"], [4.5, 1.8, 1.5])
        self.assertFalse(tx.send_telemetry(1.0, [0, 0, 5], [1, 0, 0, 0], bad=object()))  # not JSON: refused now

    def test_slow_gcs_drops_old_camera_frames_not_the_sim(self):
        # a listener that accepts but never reads: socket buffers fill up, the sim loop must not block
        path = f"{self.tmp}/slow.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        tx = Sender(f"unix:{path}", send_timeout=1.0, sndbuf=64 * 1024)
        conn, _ = srv.accept()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
        rgb = np.zeros((480, 640, 3), np.uint8)
        t0 = time.perf_counter()
        worst = 0.0
        for i in range(60):
            t1 = time.perf_counter()
            tx.send_camera(i, rgb, np.eye(3), [0, 0, 1], [1, 0, 0, 0])
            worst = max(worst, time.perf_counter() - t1)
        self.assertLess(worst, 0.05)
        self.assertLess(time.perf_counter() - t0, 1.5)
        self.assertGreater(tx.stats()["dropped_by_type"].get(P.CAMERA, 0), 40)
        tx.close(flush_timeout=0.1)
        conn.close()
        srv.close()

    def test_tcp(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        addr = f"tcp://127.0.0.1:{port}"
        with Receiver(addr) as rx:
            tx = Sender(addr)
            tx.send_scan(1.0, np.ones((5, 3), np.float32))
            msgs = collect(rx, 2)
            tx.close()
        self.assertEqual([m.type for m in msgs], [P.HELLO, P.SCAN])

    def test_second_receiver_on_same_path_refused_and_stale_socket_cleaned(self):
        with Receiver(self.addr), self.assertRaises(RuntimeError):
            Receiver(self.addr).start()
        # stale socket file (crashed GCS) is replaced
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(self.addr[5:])
        stale.close()
        self.assertTrue(os.path.exists(self.addr[5:]))
        rx = Receiver(self.addr).start()
        rx.close()
        self.assertFalse(os.path.exists(self.addr[5:]))


if __name__ == "__main__":
    unittest.main()
