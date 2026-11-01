"""The GCS application: receive -> map -> detect -> dashboard (window and/or mp4) -> report."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .dashboard import LAYERS, STATS, Dashboard
from .detector import DetectorWorker, detections_from_boxes, make_detector
from .link import Receiver
from .protocol import MISSION
from .report import format_summary, write_report
from .state import GCSConfig, GCSState

log = logging.getLogger("drone_gcs.app")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    addr: str | None = None
    name: str | None = None
    out_dir: Path = PROJECT_ROOT / "runs"
    headless: bool = False
    fps: float = 12.0
    size: tuple[int, int] = (1600, 900)
    detector: str = "yolo"
    weights: str | None = None
    device: str = "cpu"  # the GPU belongs to the simulator / RL jobs; "auto" or "cuda:0" to opt in
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    classes: str | None = None
    class_map: str | None = None
    layer: str = "elevation"
    stat: str = "mean"
    show_gt: bool = False
    exit_on_end: bool | None = None  # default: True when headless
    linger: float = 2.0
    max_time: float | None = None
    video: bool = True
    transcode: bool = True
    screenshot_at: list[float] = field(default_factory=list)
    auto_layer_cycle: float | None = None
    window_pos: tuple[int, int] | None = None
    gcs: GCSConfig = field(default_factory=GCSConfig)


class VideoRecorder:
    """cv2.VideoWriter (mp4v) at a fixed fps; on close, optionally re-encodes to H.264 with system ffmpeg
    (browser/Obsidian friendly). Frames are written in real time: a late tick repeats the last frame, for
    stalls up to ``max_catchup_s`` (longer gaps, e.g. a suspended machine, are cut and logged)."""

    def __init__(
        self, path: Path, fps: float, size: tuple[int, int], transcode: bool = True, max_catchup_s: float = 5.0
    ):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.stem + "_mp4v.mp4")
        self.fps = float(fps)
        self.size = size
        self.transcode = transcode
        self.max_catchup = max(1, round(max_catchup_s * self.fps))
        self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open a video writer for {self.tmp}")
        self.frames = 0
        self.t0: float | None = None
        self.lost_s = 0.0  # wall time cut out of the video by over-long stalls

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    def write(self, img: np.ndarray, now: float) -> None:
        if self.t0 is None:
            self.t0 = now
        due = int((now - self.t0 - self.lost_s) * self.fps) + 1
        n = max(1, due - self.frames)
        if n > self.max_catchup:
            gap = (n - self.max_catchup) / self.fps
            self.lost_s += gap
            log.warning("dashboard stalled for %.1f s: %.1f s left out of the video", n / self.fps, gap)
            n = self.max_catchup
        elif n > self.fps:
            log.info("dashboard stalled for %.1f s (video repeats the last frame)", n / self.fps)
        for _ in range(n):
            self.writer.write(img)
        self.frames += n

    def close(self) -> Path | None:
        self.writer.release()
        if self.frames == 0:
            self.tmp.unlink(missing_ok=True)
            return None
        if self.transcode and shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(self.tmp), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ]  # fmt: skip
            try:
                r = subprocess.run(cmd, timeout=900, capture_output=True, text=True)
                if r.returncode == 0 and self.path.exists():
                    self.tmp.unlink(missing_ok=True)
                    return self.path
                log.warning("ffmpeg transcode failed (%s); keeping the mp4v file", r.stderr.strip()[:200])
            except (OSError, subprocess.TimeoutExpired) as e:
                log.warning("ffmpeg transcode failed: %s", e)
        self.tmp.replace(self.path)
        return self.path


def decode_key(key: int) -> str | None:
    """Shortcut character of a ``cv2.waitKeyEx`` code, ``"esc"``, or ``None``.

    Qt builds of OpenCV (this one) return X keysyms for non-character keys (Left = 0xFF51, whose low byte is
    'Q'); GTK builds add the modifier state in bits 16+. So: keep the low 16 bits and accept only Latin-1."""
    if key is None or key < 0:
        return None
    k = key & 0xFFFF
    if k == 27:
        return "esc"
    if k > 0xFF or k < 32:
        return None  # arrows, function keys, Return, ...
    return chr(k).lower()


def link_label(address: str) -> str:
    """Short form of a link address for the screen: ``unix:drone_gcs.sock`` for a socket path."""
    if address.startswith("unix:"):
        return "unix:" + Path(address[5:]).name
    return address


def resolve_run_dir(out_dir: Path, name: str | None) -> Path:
    base = Path(out_dir).expanduser()
    name = name or f"{datetime.now():%Y%m%d_%H%M%S}_gcs"
    d = base / name
    k = 2
    while d.exists() and any(d.iterdir()):
        d = base / f"{name}_{k}"
        k += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


class GCSApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        if cfg.exit_on_end is None:
            cfg.exit_on_end = cfg.headless
        self.run_dir = resolve_run_dir(cfg.out_dir, cfg.name)
        self.state = GCSState(cfg.gcs)
        self.dashboard = Dashboard(cfg.size)
        self.receiver = Receiver(cfg.addr)
        self.worker: DetectorWorker | None = None
        self.detector_desc = "detector off"
        self.window: str | None = None
        self.recorder: VideoRecorder | None = None
        self.quit_reason: str | None = None
        self.ui = {
            "layer": cfg.layer if cfg.layer in LAYERS else "elevation",
            "stat": cfg.stat if cfg.stat in STATS else "mean",
            "show_gt": cfg.show_gt,
            "active_drone": None,
            "detector_on": False,
        }
        self.shots = 0
        self.loop_ms = 0.0
        self.t_start = time.monotonic()
        self.reports: list[Path] = []
        self.gen = 0  # mission generation: detector results of an archived mission are discarded

    # ------------------------------------------------------------------ setup
    def _init_detector(self) -> None:
        c = self.cfg
        if c.detector == "none":
            return
        if c.detector == "gt":
            self.detector_desc = "ground-truth boxes from the simulator"
            self.ui["detector_on"] = True
            log.info(
                "detector: %s (needs a simulator that sends them, e.g. sar_mission.py --gt_boxes)", self.detector_desc
            )
            return
        kw = {}
        if c.detector == "yolo":
            kw = dict(
                weights=c.weights,
                device=c.device,
                conf=c.conf,
                iou=c.iou,
                imgsz=c.imgsz,
                classes=c.classes,
                class_map=c.class_map,
            )
        det = make_detector(c.detector, **kw)
        det.warmup()
        self.worker = DetectorWorker(det)
        self.detector_desc = det.describe()
        self.ui["detector_on"] = True
        log.info("detector: %s", self.detector_desc)

    def _init_window(self) -> None:
        if self.cfg.headless:
            return
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            log.warning("no DISPLAY: running headless (recording only)")
            self.cfg.headless = True
            return
        title = f"drone_gcs - {self.run_dir.name}"
        try:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            if self.cfg.window_pos:
                cv2.moveWindow(title, *self.cfg.window_pos)
            self.window = title
        except cv2.error as e:
            log.warning("cannot open a window (%s): running headless", e)
            self.cfg.headless = True

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            self.quit_reason = f"signal {signal.Signals(signum).name}"

        for s in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError):  # not in the main thread
                signal.signal(s, handler)

    # ------------------------------------------------------------------ helpers
    def ui_dict(self) -> dict:
        rate, brate = self.receiver.rates()
        d = dict(self.ui)
        d["link"] = {
            "connected": self.receiver.connected,
            "msg_rate": rate,
            "byte_rate": brate,
            "address": self.receiver.address_str,
            "label": link_label(self.receiver.address_str),
            "dropped": int(sum(self.receiver.dropped.values())),
        }
        w = self.worker
        d["detector"] = {
            "on": w is not None or self.cfg.detector == "gt",
            "name": self.detector_desc,
            "kind": self.cfg.detector,
            "ms": w.avg_ms if w else 0.0,
            "fps": w.fps if w else 0.0,
            "error": w.last_error if w else "",
        }
        d["recording"] = self.recorder is not None
        d["rec_seconds"] = self.recorder.seconds if self.recorder else 0.0
        return d

    def screenshot(self, canvas: np.ndarray, tag: str = "shot") -> Path:
        d = self.run_dir / "screenshots"
        d.mkdir(exist_ok=True)
        self.shots += 1
        t = self.state.mission_time
        p = d / f"{tag}_{self.shots:02d}_T{t:06.1f}s.png"
        cv2.imwrite(str(p), canvas)
        self.state.event(f"screenshot saved: {p.name}")
        return p

    def handle_key(self, key: int, canvas: np.ndarray) -> None:
        k = decode_key(key)
        if k is None:
            return
        if k in ("q", "esc"):
            self.quit_reason = f"user ({k})"
        elif k == "m":
            i = LAYERS.index(self.ui["layer"])
            self.ui["layer"] = LAYERS[(i + 1) % len(LAYERS)]
        elif k == "e":
            self.ui["stat"] = "max" if self.ui["stat"] == "mean" else "mean"
        elif k == "g":
            self.ui["show_gt"] = not self.ui["show_gt"]
        elif k == "c" and self.state.drones:
            ids = sorted(self.state.drones)
            cur = self.ui["active_drone"]
            self.ui["active_drone"] = ids[(ids.index(cur) + 1) % len(ids)] if cur in ids else ids[0]
        elif k == "s":
            self.screenshot(canvas, "shot")

    def _apply_detector_results(self) -> None:
        """Finished detections of the current mission's frames go to the tracker; older ones are dropped."""
        if self.worker is None:
            return
        for (gen, frame), dets, ms in self.worker.poll():
            if gen == self.gen and frame.drone_id in self.state.drones:
                self.state.apply_detections(frame, dets, ms)

    def handle_message(self, msg) -> None:
        """Apply one link message (archiving the current mission when a new one starts) and send camera frames
        to the detector."""
        st = self.state
        if (
            msg.type == MISSION
            and st.mission is not None
            and msg.fields.get("mission_id") != st.mission.get("mission_id")
            and st.has_data
        ):
            self._rollover()
            st = self.state
        frame = st.apply(msg)
        if frame is None:
            return
        if self.worker is not None:
            self.worker.submit(frame.rgb, (self.gen, frame), key=frame.drone_id)  # every drone
        elif self.cfg.detector == "gt" and frame.gt_boxes is not None:
            st.apply_detections(frame, detections_from_boxes(frame.gt_boxes), 0.0)

    def _rollover(self) -> None:
        """A new MISSION (different mission_id) arrived: archive the current one and start fresh."""
        self._apply_detector_results()  # what finished so far still belongs to the old mission
        old = self.state
        sub = self.run_dir / f"mission{old.missions_seen}_{(old.mission or {}).get('name', 'survey')}"
        try:
            write_report(old, sub)
            self.reports.append(sub)
            log.info("new mission: previous one archived in %s", sub)
        except Exception:  # noqa: BLE001
            log.exception("could not archive the previous mission")
        self.state = GCSState(self.cfg.gcs)
        self.state.missions_seen = old.missions_seen
        # GT_OBJECTS is latched on the sim side: it stays in force until the sim sends a new one (a GCS that
        # connects now would get the same one), so it carries over. So does what we know about the senders.
        self.state.gt_objects = old.gt_objects
        self.state.gt_terrain = old.gt_terrain
        self.state.senders = dict(old.senders)
        self.gen += 1
        self.ui["active_drone"] = None

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        self._install_signals()
        try:
            self.receiver.start()
        except (OSError, RuntimeError) as e:
            log.error("cannot listen on %s: %s", self.receiver.address_str, e)
            return 2
        log.info("listening on %s  (run dir %s)", self.receiver.address_str, self.run_dir)
        try:
            self._init_detector()
        except Exception as e:  # noqa: BLE001
            log.error("detector disabled: %s", e)
            self.detector_desc = f"detector failed: {e}"[:80]
        self._init_window()
        if self.cfg.video:
            try:
                self.recorder = VideoRecorder(
                    self.run_dir / "dashboard.mp4", self.cfg.fps, self.cfg.size, self.cfg.transcode
                )
            except RuntimeError as e:
                log.error("%s", e)
        shots_due = sorted(self.cfg.screenshot_at)
        period = 1.0 / max(1.0, self.cfg.fps)
        next_tick = time.monotonic()
        last_cycle = time.monotonic()
        try:
            while self.quit_reason is None:
                t0 = time.monotonic()
                for msg in self.receiver.drain():
                    self.handle_message(msg)
                self._apply_detector_results()
                st = self.state
                if self.ui["active_drone"] is None and st.drones:
                    self.ui["active_drone"] = min(st.drones)
                canvas = self.dashboard.render(st, self.ui_dict())
                now = time.monotonic()
                if self.window is not None:
                    cv2.imshow(self.window, canvas)
                    key = cv2.waitKeyEx(1)
                    self.handle_key(key, canvas)
                    try:
                        if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                            self.quit_reason = "window closed"
                    except cv2.error:
                        self.quit_reason = "window closed"
                if self.recorder is not None:
                    self.recorder.write(canvas, now)
                el = now - self.t_start
                while shots_due and el >= shots_due[0]:
                    shots_due.pop(0)
                    self.screenshot(canvas, "auto")
                if self.cfg.auto_layer_cycle and now - last_cycle >= self.cfg.auto_layer_cycle:
                    last_cycle = now
                    self.ui["layer"] = LAYERS[(LAYERS.index(self.ui["layer"]) + 1) % len(LAYERS)]
                if st.ended and self.cfg.exit_on_end and time.time() - (st.end_wall or 0) >= self.cfg.linger:
                    self.quit_reason = f"mission end ({st.end_reason})"
                if self.cfg.max_time is not None and el >= self.cfg.max_time:
                    self.quit_reason = "max time"
                self.loop_ms = 0.9 * self.loop_ms + 0.1 * (time.monotonic() - t0) * 1000
                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.monotonic()
        finally:
            self.finish()
        return 0

    def finish(self) -> dict | None:
        log.info("stopping: %s", self.quit_reason)
        if self.worker is not None:
            self._apply_detector_results()  # results that finished in the meantime still count
            self.worker.close()
        self.receiver.close()
        if self.window is not None:
            try:
                cv2.destroyWindow(self.window)
                for _ in range(5):
                    cv2.waitKey(1)
            except cv2.error:
                pass
        video = self.recorder.close() if self.recorder is not None else None
        summary = None
        try:
            summary = write_report(
                self.state,
                self.run_dir,
                video_path=video,
                extra={
                    "gcs": {
                        "quit_reason": self.quit_reason,
                        "detector": self.detector_desc,
                        "detector_mean_ms": round(self.worker.avg_ms, 2) if self.worker else None,
                        "detector_frames": self.worker.n_done if self.worker else 0,
                        "detector_dropped_frames": self.worker.n_dropped if self.worker else 0,
                        "dashboard_loop_ms": round(self.loop_ms, 2),
                        "link_bytes": self.receiver.total_bytes,
                        "link_messages": dict(self.receiver.counts),
                        "link_dropped": self.receiver.dropped,
                        "link_errors": self.receiver.errors[-5:],
                        "screenshots": self.shots,
                        "archived_missions": [p.name for p in self.reports],
                    }
                },
            )
            print(format_summary(summary))
            print(f"report: {self.run_dir}")
        except Exception:  # noqa: BLE001
            log.exception("report failed")
        return summary


def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)
