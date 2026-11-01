"""The sender must import with only the standard library + numpy (it runs inside Isaac Sim's Python)."""

import os
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path

import numpy as np

from drone_gcs import protocol as P
from drone_gcs.link import Receiver

from ._util import ROOT, rmtree, short_tmpdir

ISAAC_PY = Path(os.path.expanduser("~/anaconda3/envs/env_isaaclab/bin/python"))

SENDER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {root!r})
    import numpy as np
    from drone_gcs.link import Sender
    from drone_gcs.isaac_bridge import IsaacLabStreamer
    heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('cv2', 'torch', 'ultralytics', 'PIL', 'matplotlib', 'isaacsim', 'omni'))
    tx = Sender({addr!r}, name='isaac-env-test')
    assert tx.wait_connected(10)
    tx.send_mission((0, 10, 0, 10), 0.25, np.array([[0, 0], [10, 0]], float), 5.0, name='cross-env', note=','.join(heavy) or 'none')
    tx.send_telemetry(1.0, np.array([1.0, 2.0, 3.0]), [1, 0, 0, 0], np.float32([0.5, 0, 0]), wp_index=1)
    tx.send_scan(1.1, np.arange(30, dtype=np.float32).reshape(10, 3), valid=np.arange(10) % 2 == 0, origin=[1, 2, 3])
    rgb = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    depth = np.linspace(1, 2, 48 * 64, dtype=np.float32).reshape(48, 64, 1)
    tx.send_camera(1.2, rgb, np.eye(3), [1, 2, 3], [0, 1, 0, 0], depth, frame_id=3)
    tx.send_camera(1.3, rgb, np.eye(3), [1, 2, 3], [0, 1, 0, 0], jpeg=True, frame_id=4)
    tx.send_end('complete', 2.0)
    tx.close()
    print('numpy', np.__version__, 'heavy', heavy)
    """
)


class TestSenderImport(unittest.TestCase):
    def test_link_imports_without_heavy_modules(self):
        code = (
            f"import sys; sys.path.insert(0, {str(ROOT)!r}); import drone_gcs.link, drone_gcs.isaac_bridge, drone_gcs.geometry; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('cv2','torch','ultralytics','PIL','matplotlib')))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")

    @unittest.skipUnless(ISAAC_PY.exists(), "env_isaaclab not found")
    def test_isaaclab_env_sender_to_gcs_receiver(self):
        """Sender in env_isaaclab (numpy 2.3) -> Receiver here (numpy 2.5): arrays arrive bit-exact."""
        tmp = short_tmpdir("dgcs_x")
        try:
            addr = f"unix:{tmp}/x.sock"
            with Receiver(addr) as rx:
                env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
                out = subprocess.run(
                    [str(ISAAC_PY), "-c", SENDER_SCRIPT.format(root=str(ROOT), addr=addr)],
                    capture_output=True, text=True, timeout=120, env=env, cwd=tmp,
                )  # fmt: skip
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                msgs = []
                t0 = time.monotonic()
                while len(msgs) < 7 and time.monotonic() - t0 < 5:
                    msgs += rx.drain()
                    time.sleep(0.02)
            by = {}
            for m in msgs:
                by.setdefault(m.type, []).append(m)
            self.assertEqual(by[P.HELLO][0].fields["name"], "isaac-env-test")
            self.assertNotEqual(by[P.HELLO][0].fields["numpy"], np.__version__)  # really two numpy versions
            self.assertEqual(by[P.MISSION][0].fields["note"], "none")  # no cv2/torch/... imported by the sender
            scan = by[P.SCAN][0]
            np.testing.assert_array_equal(scan.arrays["points"], np.arange(30, dtype=np.float32).reshape(10, 3))
            np.testing.assert_array_equal(scan.arrays["valid"], np.arange(10) % 2 == 0)
            cams = by[P.CAMERA]
            rgb = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
            np.testing.assert_array_equal(cams[0].arrays["rgb"], rgb)
            self.assertEqual(cams[0].arrays["depth"].shape, (48, 64))
            self.assertAlmostEqual(float(cams[0].arrays["depth"][-1, -1]), 2.0, places=6)
            self.assertEqual(cams[1].arrays["rgb"].shape, (48, 64, 3))  # JPEG (cv2/PIL in env_isaaclab) decoded here
            self.assertEqual(by[P.END][0].fields["reason"], "complete")
            self.assertIn("numpy 2.", out.stdout)
        finally:
            rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
