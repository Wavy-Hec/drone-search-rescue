#!/usr/bin/env python3
"""Demo video: the simulator's view next to the ground station's dashboard, in step.

The two sides record on their own clocks. The simulator writes one frame per simulation step plus the sim time of
each frame (drone_lab ``sar_mission.py --video``: ``mission.mp4`` and ``mission_frames.csv``). The ground station
writes its dashboard in wall-clock time plus the sim time each frame shows (``dashboard.mp4`` and
``dashboard_frames.csv``). The demo video follows the simulator's clock: next to every simulator frame it shows the
newest dashboard frame of that moment or an earlier one. After the last simulator frame (the touchdown) the
dashboard plays on for ``--tail`` seconds next to the last view, so the end of the mission stays on screen.

Example (conda env ``vision``, from this repository)::

    python scripts/compose_video.py --gcs runs/sar_live --sim ../sim/runs/sar_live

writes ``runs/sar_live/demo.mp4`` (H.264): the simulator view scaled to the dashboard's height on the left, the
dashboard on the right (``--layout stack``: on top of each other). ``--speed 2`` plays it twice as fast with every
frame kept (20 frames/s).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def read_times(path: Path, column: str) -> np.ndarray:
    """One time per frame from a ``*_frames.csv`` file (NaN where the column is empty)."""
    with open(path, newline="") as fh:
        return np.array([float(row[column]) if row[column] else np.nan for row in csv.DictReader(fh)])


def match_frames(sim_t: np.ndarray, dash_t: np.ndarray) -> np.ndarray:
    """Index of the dashboard frame to show next to every simulator frame.

    The newest dashboard frame whose sim time is not later than the simulator frame's; before the dashboard has
    data, its first frame with data."""
    known = np.isfinite(dash_t)
    if not known.any():
        raise ValueError("the dashboard recording has no sim times (did the link carry any data?)")
    running = np.maximum.accumulate(np.where(known, dash_t, -np.inf))
    idx = np.searchsorted(running, sim_t, side="right") - 1
    return np.clip(idx, int(np.argmax(known)), len(dash_t) - 1)


class FrameReader:
    """Sequential reader that returns the frame at a non-decreasing index (repeats it when asked again)."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise OSError(f"cannot read {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.index = -1
        self.frame: np.ndarray | None = None

    def at(self, index: int) -> np.ndarray:
        while self.index < index:
            ok, frame = self.cap.read()
            if not ok:
                break  # past the end: keep the last frame
            self.frame, self.index = frame, self.index + 1
        if self.frame is None:
            raise OSError("empty video")
        return self.frame

    def close(self) -> None:
        self.cap.release()


def compose(sim: np.ndarray, dash: np.ndarray, layout: str) -> np.ndarray:
    """Scale the simulator view to the dashboard's height (side) or width (stack) and join the two."""
    dh, dw = dash.shape[:2]
    sh, sw = sim.shape[:2]
    if layout == "side":
        sim = cv2.resize(sim, (round(sw * dh / sh / 2) * 2, dh), interpolation=cv2.INTER_LANCZOS4)
        return np.hstack([sim, dash])
    sim = cv2.resize(sim, (dw, round(sh * dw / sw / 2) * 2), interpolation=cv2.INTER_LANCZOS4)
    return np.vstack([sim, dash])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gcs", type=Path, required=True, help="ground-station run folder (dashboard.mp4 + _frames.csv)")
    p.add_argument("--sim", type=Path, required=True, help="simulator run folder (mission.mp4 + mission_frames.csv)")
    p.add_argument("--out", type=Path, default=None, help="output video (default <gcs>/demo.mp4)")
    p.add_argument("--layout", choices=("side", "stack"), default="side")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed; every frame is kept")
    p.add_argument("--tail", type=float, default=4.0, help="dashboard seconds after the last simulator frame")
    p.add_argument("--crf", type=int, default=28, help="H.264 quality: lower is better and larger")
    a = p.parse_args(argv)
    gcs, sim_dir = a.gcs.expanduser(), a.sim.expanduser()
    out = (a.out or gcs / "demo.mp4").expanduser()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is needed to write the video", file=sys.stderr)
        return 2

    sim_t = read_times(sim_dir / "mission_frames.csv", "t")
    dash_t = read_times(gcs / "dashboard_frames.csv", "sim_t")
    pick = match_frames(sim_t, dash_t)
    sim_video, dash_video = FrameReader(sim_dir / "mission.mp4"), FrameReader(gcs / "dashboard.mp4")
    fps = sim_video.fps * a.speed
    tail = [
        min(int(pick[-1]) + round((k + 1) * dash_video.fps / sim_video.fps), len(dash_t) - 1)
        for k in range(round(a.tail * sim_video.fps))
    ]
    first = compose(sim_video.at(0), dash_video.at(int(pick[0])), a.layout)
    h, w = first.shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        enc.stdin.write(first.tobytes())
        for k in range(1, len(sim_t)):
            enc.stdin.write(compose(sim_video.at(k), dash_video.at(int(pick[k])), a.layout).tobytes())
        last = sim_video.at(len(sim_t) - 1)
        for j in tail:
            enc.stdin.write(compose(last, dash_video.at(j), a.layout).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
        sim_video.close()
        dash_video.close()
    if enc.returncode != 0:
        print(f"ffmpeg failed ({enc.returncode})", file=sys.stderr)
        return 1
    frames = len(sim_t) + len(tail)
    print(f"wrote {out}: {w} x {h}, {frames} frames at {fps:g} fps ({frames / fps:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
