"""Score a detector on labelled real aerial images, for the people only.

The ground station's detector was trained and tested on simulator images. These functions score it, or any object
with ``detect(rgb) -> list[Detection]``, on a labelled set of real images (e.g. a public drone dataset in YOLO
format), so the gap between simulator and reality can be measured. The command is ``scripts/eval_external.py``.

What :func:`evaluate_people` does with every image:

1. resize it by ``scale``, so that people are about as large as in the mission camera;
2. cut it into tiles of the mission camera's size that overlap, run the detector on every tile and keep once
   what two tiles both found, whole or cut by a tile's border (:func:`detect_tiled`);
3. match the "person" boxes to the labels, one to one (:func:`match_boxes`), and count the people found, the
   people missed and the false alarms, also per label size (:func:`size_buckets`).

Boxes are ``(x1, y1, x2, y2)`` in continuous pixel coordinates (pixel ``u`` spans ``[u, u+1)``), as in
:mod:`drone_gcs.detector`. Needs numpy and OpenCV only.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection, Detector

PERSON = "person"
"""Class name of the detections that are scored."""
MISSION_TILE = (640, 480)
"""(width, height) [px] of the mission camera's image."""
MERGE_IOS = 0.5
"""Two boxes of one class from different tiles are the same object when the smaller one lies at least this much
inside the larger one (their intersection over the smaller box's area)."""
SIZE_EDGES = (16, 32, 64, 128)
"""Borders [px] between the size buckets."""
SIZE_BUCKETS = ("<16", "16-32", "32-64", "64-128", ">=128")
"""Names of the size buckets: the longer side of a label box [px]."""


def read_yolo_labels(path: str | Path, width: int, height: int, keep_ids: Collection[int]) -> np.ndarray:
    """Boxes (N, 4) ``(x1, y1, x2, y2)`` [px] of the labels of a YOLO label file whose class id is in ``keep_ids``.

    Every line is ``<class id> <cx> <cy> <w> <h>``, normalised to the image (``width`` x ``height`` px); columns
    after the fifth are ignored. A file that does not exist gives no boxes (an image without people).
    """
    path = Path(path)
    boxes = []
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts or int(parts[0]) not in keep_ids:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(boxes, float).reshape(-1, 4)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union (N, M) of every box of ``a`` (N, 4) with every box of ``b`` (M, 4); 0 for boxes
    without area."""
    a, b = np.asarray(a, float)[:, None, :], np.asarray(b, float)[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0.0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0.0, None)
    inter = w * h
    union = (
        (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    )
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _tile_starts(size: int, tile: int, overlap: float) -> list[int]:
    """Start pixels of the tiles along one axis of ``size`` px: one tile every ``tile * (1 - overlap)`` px, and the
    last one shifted back so that it ends at the border."""
    if size <= tile:
        return [0]
    step = max(1, round(tile * (1.0 - overlap)))
    return [*range(0, size - tile, step), size - tile]


def tile_origins(
    width: int, height: int, tile: tuple[int, int] = MISSION_TILE, overlap: float = 0.2
) -> list[tuple[int, int]]:
    """Top-left corners ``(x, y)`` [px] of the tiles that cover an image of ``width`` x ``height`` px, row by row.

    ``tile`` is (width, height) [px]; neighbours share the fraction ``overlap`` of a tile, to the nearest pixel.
    The last tile of each row and column is shifted back so that it ends at the image border (it shares more).
    An image smaller than a tile along an axis gets one tile along that axis, starting at 0.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be at least 0 and below 1, got {overlap}")
    xs = _tile_starts(width, tile[0], overlap)
    return [(x, y) for y in _tile_starts(height, tile[1], overlap) for x in xs]


def box_ios(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection of two boxes over the area of the smaller one (1 when one lies inside the other); 0 when one
    of them has no area."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return max(w, 0.0) * max(h, 0.0) / smaller if smaller > 0 else 0.0


def merge_tiles(found: list[tuple[int, Detection]], ios: float = MERGE_IOS) -> list[Detection]:
    """Every object once, from the detections of overlapping tiles.

    ``found`` holds ``(tile index, detection)`` pairs, boxes in pixels of the whole image. From the most to the
    least confident, a detection joins an object kept before it when it has the same class, comes from a tile that
    object was not seen in yet, and the smaller of the two boxes lies at least ``ios`` inside the other
    (:func:`box_ios`). The object's box becomes the union of the two: a person whole in one tile and cut by the
    border of the other, or cut in both, comes out whole, with the higher confidence. Two detections of one tile
    are never merged: the detector has kept them apart already, as it does in a whole image.
    """
    kept: list[tuple[set[int], Detection]] = []
    for tile, det in sorted(found, key=lambda pair: pair[1].conf, reverse=True):
        for k, (tiles, ob) in enumerate(kept):
            if ob.cls == det.cls and tile not in tiles and box_ios(ob.box, det.box) >= ios:
                (x1, y1, x2, y2), (u1, v1, u2, v2) = ob.box, det.box
                union = (min(x1, u1), min(y1, v1), max(x2, u2), max(y2, v2))
                kept[k] = (tiles | {tile}, Detection(ob.cls_id, ob.cls, ob.conf, union))
                break
        else:
            kept.append(({tile}, det))
    return [ob for _, ob in kept]


def detect_tiled(
    detector: Detector, rgb: np.ndarray, tile: tuple[int, int] = MISSION_TILE, overlap: float = 0.2
) -> list[Detection]:
    """Detections in an RGB image (H, W, 3) of any size, in pixels of the whole image.

    The detector runs on every tile of :func:`tile_origins`, each box is shifted by its tile's corner, and an
    object that overlapping tiles both found is kept once (:func:`merge_tiles`).
    """
    height, width = rgb.shape[:2]
    found = []
    for k, (x, y) in enumerate(tile_origins(width, height, tile, overlap)):
        for d in detector.detect(rgb[y : y + tile[1], x : x + tile[0]]):
            x1, y1, x2, y2 = d.box
            found.append((k, Detection(d.cls_id, d.cls, d.conf, (x1 + x, y1 + y, x2 + x, y2 + y))))
    return merge_tiles(found)


def match_boxes(
    det_boxes: np.ndarray, det_conf: np.ndarray, gt_boxes: np.ndarray, iou: float = 0.45
) -> tuple[np.ndarray, np.ndarray]:
    """Match detections (D, 4) with confidences (D,) to label boxes (G, 4), one to one.

    The detections take their turn from the most to the least confident; each takes the label it overlaps most
    among those still free, if that IoU is at least ``iou``. Returns ``(matched, false_alarm)``: ``matched`` (G,)
    is the index of the detection that found each label, or -1 for a missed label; ``false_alarm`` (D,) is True
    for the detections that found no label.
    """
    overlaps = box_iou(det_boxes, gt_boxes)
    matched = np.full(len(gt_boxes), -1, int)
    for d in np.argsort(-det_conf, kind="stable"):
        free = np.where(matched < 0, overlaps[d], -1.0)
        if free.size and free.max() >= iou:
            matched[free.argmax()] = d
    false_alarm = np.ones(len(det_boxes), bool)
    false_alarm[matched[matched >= 0]] = False
    return matched, false_alarm


def size_buckets(boxes: np.ndarray) -> np.ndarray:
    """Index into :data:`SIZE_BUCKETS` of every box (N, 4): its longer side [px] is below 16, from 16 to below
    32, from 32 to below 64, from 64 to below 128, or 128 and more."""
    longer = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    return np.digitize(longer, SIZE_EDGES)


def scale_item(rgb: np.ndarray, boxes: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """An RGB image (H, W, 3) resized by ``scale`` and its boxes (N, 4) [px] in the resized image."""
    if scale == 1.0:
        return rgb, boxes
    height, width = rgb.shape[:2]
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR  # area averaging avoids aliasing when shrinking
    resized = cv2.resize(np.ascontiguousarray(rgb), (new_w, new_h), interpolation=interp)
    kx, ky = new_w / width, new_h / height  # the true factors, after rounding the size to whole pixels
    return resized, boxes * np.array([kx, ky, kx, ky])


def _ratio(part: int, whole: int) -> float | None:
    """``part / whole`` rounded to 4 digits; None when ``whole`` is 0."""
    return round(part / whole, 4) if whole else None


def evaluate_people(
    detector: Detector,
    items: Iterable[tuple[np.ndarray, np.ndarray]],
    scale: float = 1.0,
    tile: tuple[int, int] | None = MISSION_TILE,
    overlap: float = 0.2,
    iou: float = 0.45,
) -> dict:
    """People found, missed and false alarms of ``detector`` on labelled images.

    ``items`` yields ``(rgb, boxes)``: an RGB image (H, W, 3) uint8 and the boxes (N, 4) [px] of the people
    labelled in it. Every image is resized by ``scale`` first (the boxes with it), then cut into tiles of ``tile``
    (width, height) [px] that overlap by the fraction ``overlap`` (:func:`detect_tiled`); with ``tile=None`` the
    whole image goes to the detector. Only detections of class "person" count; they are matched to the labels
    with :func:`match_boxes` at ``iou``.

    Returns ``images``, ``people_labelled``, ``found``, ``missed``, ``false_alarms``, ``precision`` (found /
    (found + false alarms)), ``recall`` (found / labelled) and ``by_size``: ``labelled``, ``found`` and ``recall``
    per bucket of :data:`SIZE_BUCKETS`, by the longer side of the label box in the resized image. A ratio
    without anything to divide by is None.
    """
    images = false_alarms = 0
    labelled = np.zeros(len(SIZE_BUCKETS), int)
    found = np.zeros(len(SIZE_BUCKETS), int)
    for rgb, boxes in items:
        rgb, boxes = scale_item(rgb, boxes, scale)
        detections = detector.detect(rgb) if tile is None else detect_tiled(detector, rgb, tile, overlap)
        people = [d for d in detections if d.cls == PERSON]
        det_boxes = np.array([d.box for d in people], float).reshape(-1, 4)
        det_conf = np.array([d.conf for d in people], float)
        matched, false_alarm = match_boxes(det_boxes, det_conf, boxes, iou)
        bucket = size_buckets(boxes)
        labelled += np.bincount(bucket, minlength=len(SIZE_BUCKETS))
        found += np.bincount(bucket[matched >= 0], minlength=len(SIZE_BUCKETS))
        false_alarms += int(false_alarm.sum())
        images += 1
    n_labelled, n_found = int(labelled.sum()), int(found.sum())
    return {
        "images": images,
        "people_labelled": n_labelled,
        "found": n_found,
        "missed": n_labelled - n_found,
        "false_alarms": false_alarms,
        "precision": _ratio(n_found, n_found + false_alarms),
        "recall": _ratio(n_found, n_labelled),
        "by_size": {
            name: {"labelled": int(n), "found": int(k), "recall": _ratio(int(k), int(n))}
            for name, n, k in zip(SIZE_BUCKETS, labelled, found, strict=True)
        },
    }
