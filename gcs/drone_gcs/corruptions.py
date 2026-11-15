"""Image corruptions that make harder detector tests from a labelled split (numpy and OpenCV only).

The simulator's frames are sharp, well lit and seen from 5-12 m, and the fine-tuned detector finds nearly every
object in them. A real camera adds noise, motion blur, low light, compression and haze, debris hides parts of
people, and a drone flying higher sees everything smaller. Each function here makes one of these from a clean
frame, so that ``scripts/degrade_dataset.py`` can write corrupted copies of a split and ``scripts/eval_robustness.py``
can score the ground station's detector on them.

Images are RGB uint8 (H, W, 3), in and out. The random corruptions draw from the numpy ``Generator`` they are
given, so the same seed gives the same image. Labels are YOLO rows ``(class, cx, cy, w, h)``, normalised to the
image size; boxes in pixels are ``(x1, y1, x2, y2)`` in continuous pixel coordinates (pixel ``u`` spans
``[u, u+1)``), as in :mod:`drone_gcs.detector`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

Label = tuple[int, float, float, float, float]
"""One YOLO label: class, centre x, centre y, width, height (normalised to the image size)."""

HAZE_GREY = 200
"""Grey level the haze blends towards (a light, slightly bright sky)."""


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def gaussian_noise(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Independent Gaussian noise on every pixel and channel, std ``sigma`` [grey levels]."""
    return _to_uint8(rgb + rng.normal(0.0, sigma, rgb.shape))


def motion_blur(rgb: np.ndarray, length_px: int, angle_deg: float) -> np.ndarray:
    """Blur along a straight line of ``length_px`` pixels (odd, so the kernel is centred on the pixel).

    ``angle_deg`` is the direction of the motion, from the image's x axis (to the right) towards its y axis
    (down): 0 is horizontal, 90 vertical, 45 from the top left to the bottom right.
    """
    if length_px < 1 or length_px % 2 == 0:
        raise ValueError(f"length_px must be an odd number of pixels, got {length_px}")
    half = (length_px - 1) // 2
    dx, dy = round(half * math.cos(math.radians(angle_deg))), round(half * math.sin(math.radians(angle_deg)))
    kernel = np.zeros((length_px, length_px), np.float32)
    cv2.line(kernel, (half - dx, half - dy), (half + dx, half + dy), 1.0, thickness=1)
    return cv2.filter2D(rgb, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT)


def low_light(rgb: np.ndarray, gain: float, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """A darker exposure: the pixel values times ``gain``, then sensor noise of std ``noise_sigma`` [grey levels]."""
    return _to_uint8(rgb * gain + rng.normal(0.0, noise_sigma, rgb.shape))


def jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    """The image after JPEG encoding at ``quality`` (0-100) and decoding: block artefacts and smeared colour."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def haze(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Every pixel blended towards :data:`HAZE_GREY` by the fraction ``strength`` (0-1); the contrast (the spread
    of the pixel values) drops by the factor ``1 - strength``."""
    return _to_uint8((1.0 - strength) * rgb + strength * HAZE_GREY)


def occlude(
    rgb: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]],
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hide about ``fraction`` of every box under a strip of ground, as debris or a branch would.

    The strip crosses the box from side to side (across or down, at random) at a random place, and is ``fraction``
    of the box's height or width wide, rounded to whole pixels. Its pixels are copied from a place of the same image
    that lies outside all boxes, chosen at random; a box for which no such place exists is left as it is. The
    labels stay: the object is still there, partly visible. Boxes are pixel ``(x1, y1, x2, y2)``; a box covers the
    pixels it touches.
    """
    out = rgb.copy()
    height, width = rgb.shape[:2]
    cells = []  # (row0, row1, col0, col1) of every box, end exclusive
    for x1, y1, x2, y2 in boxes_xyxy:
        r0, r1 = max(0, math.floor(y1)), min(height, math.ceil(y2))
        c0, c1 = max(0, math.floor(x1)), min(width, math.ceil(x2))
        if r1 > r0 and c1 > c0:
            cells.append((r0, r1, c0, c1))
    in_box = np.zeros((height, width), np.int64)
    for r0, r1, c0, c1 in cells:
        in_box[r0:r1, c0:c1] = 1
    area = np.pad(in_box.cumsum(0).cumsum(1), ((1, 0), (1, 0)))  # area[r, c]: box pixels above and left of (r, c)
    for r0, r1, c0, c1 in cells:
        h, w = r1 - r0, c1 - c0
        if rng.random() < 0.5:  # a strip across the box
            sh, sw = round(fraction * h), w
            top, left = r0 + int(rng.integers(0, h - sh + 1)), c0
        else:  # a strip down the box
            sh, sw = h, round(fraction * w)
            top, left = r0, c0 + int(rng.integers(0, w - sw + 1))
        if sh == 0 or sw == 0:
            continue
        # box pixels in the sh x sw window at every top-left corner; the windows without any are the ground
        inside = area[sh:, sw:] - area[:-sh, sw:] - area[sh:, :-sw] + area[:-sh, :-sw]
        free = np.argwhere(inside == 0)
        if len(free) == 0:
            continue
        sr, sc = free[rng.integers(len(free))]
        out[top : top + sh, left : left + sw] = rgb[sr : sr + sh, sc : sc + sw]
    return out


def shrink_mosaic(images: Sequence[np.ndarray], labels: Sequence[Sequence[Label]], k: int) -> tuple[np.ndarray, list]:
    """Tile ``k * k`` frames of one size into one frame of that size, each shrunk to ``1/k`` of its size.

    Every object comes out ``k`` times smaller, as seen from ``k`` times the height with the same camera. The frames
    fill the tiles row by row (frame ``i`` goes to row ``i // k``, column ``i % k``); ``labels[i]`` are the labels
    of frame ``i``. Returns the tiled frame and its labels, in the order of the frames.
    """
    if len(images) != k * k or len(labels) != k * k:
        raise ValueError(f"shrink_mosaic needs {k * k} frames and their labels, got {len(images)} and {len(labels)}")
    height, width = images[0].shape[:2]
    if any(im.shape != images[0].shape for im in images):
        raise ValueError("shrink_mosaic needs frames of one size")
    rows = [np.hstack(images[r * k : (r + 1) * k]) for r in range(k)]
    tiled = cv2.resize(np.vstack(rows), (width, height), interpolation=cv2.INTER_AREA)
    out = []
    for i, frame_labels in enumerate(labels):
        row, col = divmod(i, k)
        out += [(c, (col + cx) / k, (row + cy) / k, w / k, h / k) for c, cx, cy, w, h in frame_labels]
    return tiled, out


def _pixel_boxes(labels: Sequence[Label], rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    height, width = rgb.shape[:2]
    return [
        ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
        for _, cx, cy, w, h in labels
    ]


@dataclass(frozen=True)
class Preset:
    """One corruption with its settings: either ``frame`` corrupts every frame on its own, or ``k`` > 1 tiles
    ``k * k`` frames into one with :func:`shrink_mosaic`."""

    about: str
    """One line for the reports."""
    frame: Callable[[np.ndarray, Sequence[Label], np.random.Generator], np.ndarray] | None = None
    """``frame(rgb, labels, rng)`` -> the corrupted frame; its labels stay the same."""
    k: int = 1


PRESETS: dict[str, Preset] = {
    "noise": Preset("sensor noise, std 15 grey levels", lambda rgb, labels, rng: gaussian_noise(rgb, 15.0, rng)),
    "blur": Preset("motion blur over 11 px, horizontal", lambda rgb, labels, rng: motion_blur(rgb, 11, 0.0)),
    "dark": Preset(
        "a quarter of the light (gain 0.25), sensor noise std 6",
        lambda rgb, labels, rng: low_light(rgb, 0.25, 6.0, rng),
    ),
    "jpeg": Preset("JPEG compression at quality 15", lambda rgb, labels, rng: jpeg(rgb, 15)),
    "haze": Preset("haze: halfway to a light grey, half the contrast", lambda rgb, labels, rng: haze(rgb, 0.5)),
    "occlude30": Preset(
        "30 % of every object under a strip of ground",
        lambda rgb, labels, rng: occlude(rgb, _pixel_boxes(labels, rgb), 0.3, rng),
    ),
    "occlude60": Preset(
        "60 % of every object under a strip of ground",
        lambda rgb, labels, rng: occlude(rgb, _pixel_boxes(labels, rgb), 0.6, rng),
    ),
    "altitude2": Preset("2 x 2 frames in one: objects half the size, as from twice the height", k=2),
    "altitude3": Preset("3 x 3 frames in one: objects a third of the size, as from three times the height", k=3),
}
"""The corrupted copies that ``scripts/degrade_dataset.py`` writes, in order."""
