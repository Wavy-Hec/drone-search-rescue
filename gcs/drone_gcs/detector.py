"""Detectors for the GCS and the worker thread that keeps inference off the UI loop.

* :class:`YoloDetector` - Ultralytics YOLO (default: the COCO nano weights in ``~/isaac_ws/assets/models``;
  later a fine-tuned model). Ultralytics is AGPL-3.0, see the README.
* :class:`ColorBlobDetector` - stand-in for the fake world (saturated colour blobs -> boxes). COCO YOLO does
  not know the synthetic shapes, so the end-to-end tests use this one.
* :class:`DetectorWorker` - runs any detector in a background thread; ``submit()`` keeps only the newest
  pending frame (older ones are dropped), ``poll()`` returns finished results with the inference time.

Boxes are ``(x1, y1, x2, y2)`` in continuous pixel coordinates of the input image (pixel ``u`` spans
``[u, u+1)``), the convention used by Ultralytics.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = os.environ.get("DRONE_GCS_WEIGHTS", "/home/hector/isaac_ws/assets/models/yolo26n.pt")
FINETUNED_WEIGHTS = PROJECT_ROOT / "models" / "detector.pt"  # written by scripts/train_detector.py


@dataclass
class Detection:
    cls_id: int
    cls: str
    conf: float
    box: tuple  # (x1, y1, x2, y2) [px]

    def to_dict(self) -> dict:
        return {"class": self.cls, "class_id": self.cls_id, "conf": round(self.conf, 4), "box": [round(b, 1) for b in self.box]}


def _parse_class_map(class_map) -> dict:
    """``{"truck": "car", "bench": None}`` or ``"truck=car,bench="`` -> dict (``None`` = drop the class)."""
    if not class_map:
        return {}
    if isinstance(class_map, dict):
        return dict(class_map)
    out = {}
    for part in str(class_map).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip() or None
    return out


class YoloDetector:
    """Ultralytics YOLO wrapper. ``rgb`` input (the GCS works in RGB); converts to BGR for Ultralytics."""

    name = "yolo"

    def __init__(
        self,
        weights: str | os.PathLike | None = None,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        classes=None,
        class_map=None,
        half: bool = False,
        max_det: int = 100,
    ):
        os.environ.setdefault("YOLO_OFFLINE", "1")  # no update checks / analytics from the GCS process
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO  # noqa: PLC0415  (heavy, optional)

        self.weights = str(weights or DEFAULT_WEIGHTS)
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.model = YOLO(self.weights, task="detect")
        self.names = {int(k): str(v) for k, v in dict(self.model.names).items()}
        if device == "auto":
            try:
                import torch  # noqa: PLC0415

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.conf, self.iou, self.imgsz, self.max_det = float(conf), float(iou), int(imgsz), int(max_det)
        self.half = bool(half) and str(device).startswith("cuda")
        self.class_map = _parse_class_map(class_map)
        self.class_ids = None
        if classes:
            wanted = {c.strip() for c in (classes.split(",") if isinstance(classes, str) else classes)}
            self.class_ids = [i for i, n in self.names.items() if n in wanted]

    def describe(self) -> str:
        return f"YOLO {Path(self.weights).name} on {self.device} (conf {self.conf:.2f}, iou {self.iou:.2f}, {self.imgsz}px)"

    def warmup(self, shape=(480, 640)) -> None:
        self.detect(np.zeros((shape[0], shape[1], 3), np.uint8))

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])
        kw = {"half": True} if self.half else {}  # only when asked: newer Ultralytics warn about 'half'
        res = self.model.predict(
            bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.class_ids,
            max_det=self.max_det,
            verbose=False,
            **kw,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        out = []
        for b, p, c in zip(xyxy, conf, cls):
            name = self.names.get(int(c), str(c))
            name = self.class_map.get(name, name)
            if name is None:
                continue
            out.append(Detection(int(c), name, float(p), tuple(float(v) for v in b)))
        return out


class ColorBlobDetector:
    """Stand-in detector: one class per saturated colour (hue window + minimum saturation/value).

    ``classes``: ``{name: (r, g, b)}``; defaults to the fake world's object palette.
    """

    name = "blob"

    def __init__(
        self,
        classes: dict | None = None,
        hue_tol: int = 9,
        min_sat: int = 130,
        min_val: int = 45,
        min_area: int = 30,
        max_area_frac: float = 0.3,
        open_px: int = 3,
    ):
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        if classes is None:
            from .sim.fake_world import OBJECT_CLASSES  # noqa: PLC0415

            classes = {k: v["color"] for k, v in OBJECT_CLASSES.items()}
        self.classes = dict(classes)
        self.names = {i: n for i, n in enumerate(self.classes)}
        self.hues = {}
        for n, rgb in self.classes.items():
            px = np.array([[rgb]], np.uint8)
            self.hues[n] = int(cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0, 0])
        self.hue_tol, self.min_sat, self.min_val = int(hue_tol), int(min_sat), int(min_val)
        self.min_area, self.max_area_frac = int(min_area), float(max_area_frac)
        self._kernel = np.ones((open_px, open_px), np.uint8) if open_px > 1 else None

    def describe(self) -> str:
        return f"colour-blob stand-in ({', '.join(self.classes)})"

    def warmup(self, shape=(480, 640)) -> None:
        pass

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        cv2 = self._cv2
        img = np.ascontiguousarray(rgb)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h = hsv[..., 0].astype(np.int16)
        sv = (hsv[..., 1] >= self.min_sat) & (hsv[..., 2] >= self.min_val)
        H, W = img.shape[:2]
        out = []
        for cid, (name, h0) in enumerate(self.hues.items()):
            dh = np.abs(h - h0)
            dh = np.minimum(dh, 180 - dh)
            m = (sv & (dh <= self.hue_tol)).astype(np.uint8)
            if self._kernel is not None:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for k in range(1, n):
                x, y, w, hh, area = (int(v) for v in stats[k])
                if area < self.min_area or area > self.max_area_frac * H * W:
                    continue
                fill = area / float(w * hh)
                conf = 0.55 + 0.4 * min(1.0, fill / 0.8)
                if x == 0 or y == 0 or x + w >= W or y + hh >= H:
                    conf *= 0.75  # truncated at the image border
                out.append(Detection(cid, name, float(min(conf, 0.99)), (float(x), float(y), float(x + w), float(y + hh))))
        return out


class NullDetector:
    name = "none"
    names: dict = {}

    def describe(self) -> str:
        return "no detector"

    def warmup(self, shape=(480, 640)) -> None:
        pass

    def detect(self, rgb):
        return []


def make_detector(kind: str = "yolo", **kw):
    """``"yolo"`` (kw: weights, device, conf, iou, imgsz, classes, class_map), ``"blob"`` or ``"none"``."""
    if kind == "yolo":
        return YoloDetector(**kw)
    if kind == "blob":
        return ColorBlobDetector()
    if kind == "none":
        return NullDetector()
    raise ValueError(f"unknown detector {kind!r} (yolo | blob | none)")


class DetectorWorker:
    """Runs ``detector.detect`` in a daemon thread. Only the newest submitted frame waits; older ones drop."""

    def __init__(self, detector, name: str = "drone_gcs-detector"):
        self.detector = detector
        self._pending = None
        self._cond = threading.Condition()
        self._results: collections.deque = collections.deque(maxlen=64)
        self._stop = False
        self.n_submitted = self.n_done = self.n_dropped = self.n_errors = 0
        self.last_ms = 0.0
        self.avg_ms = 0.0
        self.fps = 0.0
        self.last_error = ""
        self._busy = False
        self._done_times: collections.deque = collections.deque(maxlen=50)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy or self._pending is not None

    def submit(self, rgb: np.ndarray, context=None) -> None:
        with self._cond:
            if self._pending is not None:
                self.n_dropped += 1
            self._pending = (rgb, context)
            self.n_submitted += 1
            self._cond.notify()

    def poll(self) -> list[tuple]:
        """Finished results: ``[(context, detections, inference_ms), ...]``."""
        out = []
        while self._results:
            out.append(self._results.popleft())
        return out

    def close(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(0.25)
                if self._stop:
                    return
                rgb, ctx = self._pending
                self._pending = None
                self._busy = True
            t0 = time.perf_counter()
            try:
                dets = self.detector.detect(rgb)
            except Exception as e:  # noqa: BLE001  (keep the GCS alive whatever the model does)
                self.n_errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                dets = []
            ms = (time.perf_counter() - t0) * 1000.0
            now = time.perf_counter()
            self._done_times.append(now)
            while len(self._done_times) > 1 and now - self._done_times[0] > 3.0:
                self._done_times.popleft()  # throughput over the last ~3 s
            n = len(self._done_times)
            self.fps = (n - 1) / (now - self._done_times[0]) if n > 1 and now > self._done_times[0] else 0.0
            self.last_ms = ms
            self.avg_ms = ms if self.n_done == 0 else 0.85 * self.avg_ms + 0.15 * ms
            self.n_done += 1
            self._results.append((ctx, dets, ms))
            self._busy = False
