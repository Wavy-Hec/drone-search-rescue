"""Instance segmentation -> objects, classes and 2D boxes (numpy only).

The SAR camera returns ``instance_segmentation`` (Replicator's ``instance_segmentation_fast``) as raw int32 ids plus
``camera.data.info["instance_segmentation"]`` with ``idToLabels`` (id -> prim path of the labelled object) and
``idToSemantics`` (id -> {"class": "<label>"}); ids 0 / 1 are BACKGROUND / UNLABELLED (terrain, trees, drone).
Every object of the site lives at ``<site>/objects/<name>/...``, so the prim path tells which object a pixel
belongs to; the class comes from the semantics (callers cross-check it against the ground truth). The helpers also
accept per-mesh id maps (several ids per object are merged by object name).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assets import CLASS_IDS, SEMANTIC_TYPE
from .scenario import OBJECTS_SUBPATH


@dataclass
class InstanceBox:
    """One labelled object visible in a frame."""

    name: str
    """Object name (``person_03``) or the prim path when the pixel is not part of the site."""
    label: str | None
    """Class from the segmentation semantics (None when unlabeled)."""
    class_id: int | None
    bbox_xyxy: tuple[int, int, int, int]
    """Pixel box (x_min, y_min, x_max, y_max), inclusive."""
    pixels: int
    ids: tuple[int, ...]

    def to_dict(self) -> dict:
        """JSON-ready fields (``label`` is stored as ``class``)."""
        return {
            "name": self.name,
            "class": self.label,
            "class_id": self.class_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "pixels": self.pixels,
            "ids": list(self.ids),
        }


def _norm_map(m) -> dict[int, object]:
    if not m:
        return {}
    out = {}
    for k, v in dict(m).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


PSEUDO_LABELS = ("BACKGROUND", "UNLABELLED", "UNLABELED")
"""Replicator's pseudo classes of unlabelled pixels (ids 0 and 1 of ``instance_segmentation``)."""


def semantic_label(sem: dict | str | None) -> str | None:
    """``class`` label from one ``idToSemantics`` entry (dict, or a "class: x" string)."""
    if sem is None:
        return None
    if isinstance(sem, dict):
        val = sem.get(SEMANTIC_TYPE)
        if val is None:
            return None
        val = str(val)
    else:
        val = str(sem)
        if ":" in val:
            val = val.split(":", 1)[1]
    labels = [s.strip() for s in val.split(",") if s.strip() and s.strip().upper() not in PSEUDO_LABELS]
    return labels[0] if len(labels) == 1 else (",".join(labels) if labels else None)


def object_name_from_path(prim_path: str, site_prim_path: str = "/World/sar") -> str | None:
    """``/World/sar/objects/person_03/Root/...`` -> ``person_03`` (None for prims outside the site)."""
    prefix = f"{site_prim_path}/{OBJECTS_SUBPATH}/"
    if not isinstance(prim_path, str) or not prim_path.startswith(prefix):
        return None
    return prim_path[len(prefix) :].split("/", 1)[0]


def instance_boxes(
    seg: np.ndarray, info: dict | None, site_prim_path: str = "/World/sar", min_pixels: int = 1
) -> list[InstanceBox]:
    """Group segmentation ids by object and return one box per object (largest first).

    Args:
        seg: (H, W) or (H, W, 1) integer ids.
        info: The camera's info dict for this data type (``idToLabels``, ``idToSemantics``).
        site_prim_path: Site root used to map prim paths to object names.
        min_pixels: Drop objects with fewer visible pixels.
    """
    seg = np.asarray(seg).reshape(np.asarray(seg).shape[:2])
    info = info or {}
    id_to_path = _norm_map(info.get("idToLabels"))
    id_to_sem = _norm_map(info.get("idToSemantics") or info.get("idToSemantic"))
    groups: dict[str, list[int]] = {}
    labels: dict[str, str | None] = {}
    for i in np.unique(seg).tolist():
        path = id_to_path.get(int(i))
        label = semantic_label(id_to_sem.get(int(i)))
        if label is None:
            continue
        name = object_name_from_path(str(path), site_prim_path) or str(path)
        groups.setdefault(name, []).append(int(i))
        labels[name] = labels.get(name) or label
    boxes = []
    for name, ids in groups.items():
        mask = np.isin(seg, ids)
        n = int(mask.sum())
        if n < min_pixels:
            continue
        ys, xs = np.nonzero(mask)
        label = labels[name]
        boxes.append(
            InstanceBox(
                name=name,
                label=label,
                class_id=CLASS_IDS.get(label),
                bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                pixels=n,
                ids=tuple(ids),
            )
        )
    boxes.sort(key=lambda b: -b.pixels)
    return boxes


def class_map(seg: np.ndarray, info: dict | None) -> np.ndarray:
    """(H, W) int map of class ids (-1 for unlabeled / background)."""
    seg = np.asarray(seg).reshape(np.asarray(seg).shape[:2])
    out = np.full(seg.shape, -1, dtype=np.int16)
    info = info or {}
    id_to_sem = _norm_map(info.get("idToSemantics") or info.get("idToSemantic"))
    for i in np.unique(seg).tolist():
        cid = CLASS_IDS.get(semantic_label(id_to_sem.get(int(i))))
        if cid is not None:
            out[seg == i] = cid
    return out


CLASS_COLORS = {
    "person": (230, 25, 75),
    "vehicle": (245, 130, 48),
    "crate": (255, 225, 25),
    "barrel": (60, 180, 75),
    "pallet": (145, 30, 180),
    "cone": (70, 240, 240),
    "landing_pad": (0, 130, 200),
}
"""Display colour (RGB 0-255) per class, shared by the preview figures."""


def colorize_classes(cmap: np.ndarray, background: tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
    """RGB image of a :func:`class_map`."""
    out = np.empty((*cmap.shape, 3), dtype=np.uint8)
    out[:] = background
    for name, cid in CLASS_IDS.items():
        out[cmap == cid] = CLASS_COLORS.get(name, (255, 255, 255))
    return out
