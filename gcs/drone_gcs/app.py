"""The GCS application: receive (or replay a recording) -> map -> detect -> dashboard (window and/or mp4) -> report."""

from __future__ import annotations

import collections
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
from .protocol import CAMERA, MISSION, Message, ProtocolError
from .recording import read_recording
from .report import format_summary, write_report
from .state import GCSConfig, GCSState

log = logging.getLogger("drone_gcs.app")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    """Settings of :class:`GCSApp` (``scripts/gcs.py`` fills them from its command line)."""

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
    record: str | None = None  # save the link stream to this file ("" = link.dgcrec in the run folder)
    replay: Path | None = None  # process this saved link stream instead of listening on the link
    replay_speed: float = 0.0  # 0 = as fast as possible, no frame skipped; 1 = real time; N = N times faster
    gcs: GCSConfig = field(default_factory=GCSConfig)


class VideoRecorder:
    """cv2.VideoWriter (mp4v) at a fixed fps; on close, optionally re-encodes to H.264 with system ffmpeg
    (browser/Obsidian friendly). Frames are written in real time: a late tick repeats the last frame, for
    stalls up to ``max_catchup_s`` (longer gaps, e.g. a suspended machine, are cut and logged).

    ``<name>_frames.csv`` next to the video gives the simulator time shown in every frame, so the recording can be
    lined up with one made on the simulator side (``scripts/compose_video.py``)."""

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
        self.frames_csv = self.path.with_name(self.path.stem + "_frames.csv")
        self.sim_t: list[str] = []  # simulator time shown in each frame ("" before the first message)
        self.frames = 0
        self.t0: float | None = None
        self.lost_s = 0.0  # wall time cut out of the video by over-long stalls

    @property
    def seconds(self) -> float:
        """Length of the video so far [s]."""
        return self.frames / self.fps

    def write(self, img: np.ndarray, now: float, sim_t: float | None = None) -> None:
        """Write ``img`` as the frame(s) due at wall time ``now``; ``sim_t`` is the simulator time it shows."""
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
        self.sim_t += ["" if sim_t is None else f"{sim_t:.2f}"] * n
        self.frames += n

    def close(self) -> Path | None:
        """Finish the video and its frame log; returns the video, or None when no frame was written."""
        self.writer.release()
        if self.frames == 0:
            self.tmp.unlink(missing_ok=True)
            return None
        self.frames_csv.write_text("frame,sim_t\n" + "".join(f"{k},{t}\n" for k, t in enumerate(self.sim_t)))
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
    """A new, empty run folder ``out_dir/name`` (``name_2``, ... if taken; a time-stamped name if ``name`` is None)."""
    base = Path(out_dir).expanduser()
    name = name or f"{datetime.now():%Y%m%d_%H%M%S}_gcs"
    d = base / name
    k = 2
    while d.exists() and any(d.iterdir()):
        d = base / f"{name}_{k}"
        k += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


class ReplaySource:
    """Stands in for the :class:`~drone_gcs.link.Receiver` when a recording is replayed: it has the members the
    app uses of it, plus ``finished`` and :meth:`next_message`.

    ``speed`` > 0 is a paced replay: :meth:`drain` returns the messages whose recorded receive time, divided by
    ``speed``, has passed since the first call, so the station works as on a live link (a detector that is too
    slow skips frames). ``speed`` == 0 is a lossless replay: the app takes the messages one by one with
    :meth:`next_message`, as fast as it can process them.
    """

    def __init__(self, path: str | os.PathLike, speed: float = 0.0):
        self.path = Path(path)
        self.speed = float(speed)
        self.address_str = f"replay:{self.path.name}"
        self.counts: collections.Counter = collections.Counter()  # messages handed out, per type
        self.total_bytes = 0
        self.dropped: dict = {}  # a replay drops nothing
        self.errors: list[str] = []
        self.finished = False  # every recorded message has been handed out
        self._messages = read_recording(self.path)
        self._next: Message | None = None  # read from the file, not handed out yet
        self._t0: float | None = None  # time.monotonic() of the first drain()
        self._rate: collections.deque = collections.deque()  # (monotonic time, nbytes) of the messages handed out

    def start(self) -> ReplaySource:
        """Open the recording and read its first message (raises OSError or ProtocolError for a bad file)."""
        self._next = next(self._messages, None)
        self.finished = self._next is None
        return self

    def close(self) -> None:
        """Close the recording file."""
        self._messages.close()

    @property
    def connected(self) -> bool:
        """Messages are left to replay."""
        return not self.finished

    def rates(self, window: float = 2.0) -> tuple[float, float]:
        """``(messages/s, bytes/s)`` handed out over the last ``window`` seconds of wall time."""
        now = time.monotonic()
        while self._rate and now - self._rate[0][0] > window:
            self._rate.popleft()
        return len(self._rate) / window, sum(x[1] for x in self._rate) / window

    def next_message(self) -> Message | None:
        """The next recorded message, or ``None`` after the last one."""
        msg = self._next
        if msg is None:
            return None
        self._next = next(self._messages, None)
        self.finished = self._next is None
        self.counts[msg.type] += 1
        self.total_bytes += msg.nbytes
        self._rate.append((time.monotonic(), msg.nbytes))
        return msg

    def drain(self) -> list[Message]:
        """Paced replay: the messages that are due by now (the clock starts at the first call)."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        due = (now - self._t0) * self.speed  # position in the recording [s since its first message]
        out = []
        while self._next is not None and self._next.recv_time <= due:
            out.append(self.next_message())
        return out


class GCSApp:
    """The ground station: link, mission state, detector worker, dashboard window and recording, report."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        if cfg.record is not None and cfg.replay is not None:
            raise ValueError("record and replay cannot be combined (the replayed file is the recording)")
        if cfg.exit_on_end is None:
            cfg.exit_on_end = cfg.headless
        self.run_dir = resolve_run_dir(cfg.out_dir, cfg.name)
        self.state = GCSState(cfg.gcs)
        self.dashboard = Dashboard(cfg.size)
        self.record_path: Path | None = None  # where the link stream is saved
        if cfg.record is not None:
            self.record_path = Path(cfg.record) if cfg.record else self.run_dir / "link.dgcrec"
        self.replay: ReplaySource | None = None  # set when a recording is processed instead of the live link
        self.receiver: Receiver | ReplaySource
        if cfg.replay is not None:
            self.replay = self.receiver = ReplaySource(cfg.replay, cfg.replay_speed)
        else:
            self.receiver = Receiver(cfg.addr, record=self.record_path)
        self.lossless = self.replay is not None and cfg.replay_speed <= 0  # replay every message, skip no frame
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
        """UI state for the dashboard: layer, statistic, link, detector and recording status."""
        rate, brate = self.receiver.rates()
        d = dict(self.ui)
        d["link"] = {
            "connected": self.receiver.connected,
            "msg_rate": rate,
            "byte_rate": brate,
            "bytes": self.receiver.total_bytes,
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
        """Save ``canvas`` to ``<run>/screenshots/``; returns the file."""
        d = self.run_dir / "screenshots"
        d.mkdir(exist_ok=True)
        self.shots += 1
        t = self.state.mission_time
        p = d / f"{tag}_{self.shots:02d}_T{t:06.1f}s.png"
        cv2.imwrite(str(p), canvas)
        self.state.event(f"screenshot saved: {p.name}")
        return p

    def handle_key(self, key: int, canvas: np.ndarray) -> None:
        """Act on one ``cv2.waitKeyEx`` code (keys: see :func:`decode_key` and the README)."""
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

    def handle_message(self, msg: Message) -> None:
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

    def _replay_until(self, deadline: float) -> None:
        """Lossless replay: apply recorded messages until ``deadline`` (``time.monotonic()``), at least one.

        After every CAMERA message the detector finishes that frame and its result is applied before the next
        message is read. So no frame is skipped, and the outcome does not depend on how fast this computer is."""
        while self.quit_reason is None:
            msg = self.replay.next_message()
            if msg is None:
                return
            self.handle_message(msg)
            if msg.type == CAMERA and self.worker is not None:
                self.worker.wait_idle()
                self._apply_detector_results()
            if time.monotonic() >= deadline:
                return

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
        """Run until a key, the window, a signal, the mission end, the end of a replayed recording or ``max_time``
        stops it; returns the exit code."""
        self._install_signals()
        try:
            self.receiver.start()
        except (OSError, RuntimeError, ProtocolError) as e:
            if self.replay is not None:
                log.error("cannot replay %s: %s", self.replay.path, e)
            else:
                log.error("cannot listen on %s: %s", self.receiver.address_str, e)
            return 2
        if self.replay is not None:
            pace = "as fast as possible, no frame skipped" if self.lossless else f"{self.replay.speed:g}x real time"
            log.info("replaying %s, %s  (run dir %s)", self.replay.path, pace, self.run_dir)
        else:
            log.info("listening on %s  (run dir %s)", self.receiver.address_str, self.run_dir)
        if self.record_path is not None:
            log.info("recording the link to %s", self.record_path)
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
                if self.lossless:
                    self._replay_until(next_tick)  # next_tick: when this loop's dashboard frame is due
                else:
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
                    self.recorder.write(canvas, now, st.t_now)
                el = now - self.t_start
                while shots_due and el >= shots_due[0]:
                    shots_due.pop(0)
                    self.screenshot(canvas, "auto")
                if self.cfg.auto_layer_cycle and now - last_cycle >= self.cfg.auto_layer_cycle:
                    last_cycle = now
                    self.ui["layer"] = LAYERS[(LAYERS.index(self.ui["layer"]) + 1) % len(LAYERS)]
                if st.ended and self.cfg.exit_on_end and time.time() - (st.end_wall or 0) >= self.cfg.linger:
                    self.quit_reason = f"mission end ({st.end_reason})"
                if self.replay is not None and self.replay.finished and not st.ended and self.cfg.exit_on_end:
                    self.quit_reason = "end of recording"  # the recording stops before its mission's END
                if self.cfg.max_time is not None and el >= self.cfg.max_time:
                    self.quit_reason = "max time"
                self.loop_ms = 0.9 * self.loop_ms + 0.1 * (time.monotonic() - t0) * 1000
                next_tick += period
                sleep = next_tick - time.monotonic()
                input_left = self.lossless and not self.replay.finished  # then the time goes to the next messages
                if sleep <= 0:
                    next_tick = time.monotonic()
                elif not input_left:
                    time.sleep(sleep)
        finally:
            self.finish()
        return 0

    def finish(self) -> dict | None:
        """Stop the detector, link and window, close the video and write the report; returns the summary."""
        log.info("stopping: %s", self.quit_reason)
        if self.worker is not None:
            self._apply_detector_results()  # results that finished in the meantime still count
            self.worker.close()
        self.receiver.close()
        if self.replay is None and self.receiver.recorder is not None:
            rec = self.receiver.recorder
            log.info(
                "link recorded: %d frames, %.1f MB before compression, in %s", rec.frames, rec.nbytes / 1e6, rec.path
            )
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
                        "recording": self.record_path.name if self.record_path is not None else None,
                        "replay": (
                            {"file": self.replay.path.name, "speed": self.replay.speed}
                            if self.replay is not None
                            else None
                        ),
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
    """``"1600x900"`` -> ``(1600, 900)``."""
    w, h = s.lower().split("x")
    return int(w), int(h)
