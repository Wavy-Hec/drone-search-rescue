"""Isaac Lab camera captures (RGB + instance-id segmentation + id -> class mapping) -> YOLO detection dataset.

Input layouts (one directory, ``<k>`` = frame key, e.g. ``000123`` or ``env0_000123``):

* flat:    ``rgb_<k>.png|jpg|npy``, ``instance_<k>.npy|png`` and optionally ``instance_<k>.json``
* subdirs: ``rgb/<k>.png|jpg|npy``, ``instance/<k>.npy|png`` and optionally ``instance/<k>.json``

``instance`` is an integer (H, W) or (H, W, 1) array of instance ids (Isaac Lab ``instance_id_segmentation_fast``
with ``colorize_instance_id_segmentation=False``, or ``instance_segmentation_fast``). The JSON is that
annotator's ``info`` dict (``idToSemantics`` {id: {"class": name}} or ``idToLabels`` {id: name | {"class": name}
| prim path}) or a plain ``{id: class}``; a global ``mapping.json`` in the same format can replace the per-frame
files when ids are stable. Prim paths are turned into classes with ``prim_rules`` (substring -> class).

Labels: one ``<class> <cx> <cy> <w> <h>`` line per instance, normalised to [0, 1], box = the tight box of the
visible mask. Dropped: tiny boxes (area / side), heavily truncated boxes at the image border (visible fraction
estimated from the loose box if given, else from the median full size of the class), unknown classes.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IGNORE_LABELS = frozenset({"", "background", "unlabelled", "unlabeled", "unknown", "none"})
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".npy")
SEG_EXTS = (".npy", ".png")


@dataclass
class Box:
    cls: str
    inst: int
    x0: int  # inclusive pixel indices of the visible mask
    y0: int
    x1: int
    y1: int
    area: int  # visible pixels

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1


@dataclass
class ConvertConfig:
    classes: list[str] | None = None  # order = YOLO class ids; None = all classes found, sorted
    min_area: int = 20  # visible pixels
    min_side: int = 4  # px
    max_truncation: float = 0.6  # drop border boxes with less than 40 % of the object visible
    val_frac: float = 0.15
    split_mode: str = "block"  # block (contiguous chunks, no near-duplicate leakage) | random
    block_size: int = 50
    seed: int = 0
    image_format: str = "copy"  # copy | png | jpg
    skip_empty: bool = False
    prim_rules: dict = field(default_factory=dict)  # prim-path substring -> class


# ---------------------------------------------------------------------------------------------------- mapping
def _cls_from_value(v, prim_rules: dict) -> str | None:
    if isinstance(v, dict):
        v = v.get("class", v.get("semantic", v.get("label")))
    if v is None:
        return None
    s = str(v)
    if s.startswith("/"):  # a prim path
        for key, cls in prim_rules.items():
            if key in s:
                return cls
        return None
    # "class:car" style or "car,vehicle" style -> first token
    s = s.split(",")[0].strip()
    if ":" in s and not s.startswith("/"):
        s = s.split(":", 1)[1]
    return s


def parse_id_mapping(info: dict, prim_rules: dict | None = None) -> dict[int, str]:
    """``{instance_id: class}`` from an Isaac Lab segmentation ``info`` dict (or a plain mapping)."""
    prim_rules = prim_rules or {}
    if not isinstance(info, dict):
        return {}
    if isinstance(info.get("idToSemantics"), dict):  # instance segmentation: id -> {"class": ...}
        src = info["idToSemantics"]
    elif isinstance(info.get("idToLabels"), dict):  # semantic: id -> {"class": ...}; instance: id -> prim path
        src = info["idToLabels"]
    else:
        src = info
    out = {}
    for k, v in src.items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        c = _cls_from_value(v, prim_rules)
        if c is not None and c.lower() not in IGNORE_LABELS:
            out[i] = c
    return out


# ---------------------------------------------------------------------------------------------------- boxes
def instance_boxes(seg: np.ndarray, id_to_class: dict[int, str], min_area: int = 1, min_side: int = 1) -> tuple[list[Box], Counter]:
    """Tight boxes of every labelled instance in an (H, W) id image. Returns (boxes, dropped-by-reason)."""
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = seg[..., 0]
    dropped: Counter = Counter()
    boxes = []
    ids, counts = np.unique(seg, return_counts=True)
    for i, n in zip(ids.tolist(), counts.tolist()):
        cls = id_to_class.get(int(i))
        if cls is None:
            continue
        m = seg == i
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        b = Box(cls, int(i), int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]), int(n))
        if b.area < min_area:
            dropped["tiny_area"] += 1
            continue
        if b.w < min_side or b.h < min_side:
            dropped["tiny_side"] += 1
            continue
        boxes.append(b)
    return boxes, dropped


def touches_border(b: Box, W: int, H: int) -> tuple[bool, bool]:
    """(touches left/right, touches top/bottom)."""
    return (b.x0 <= 0 or b.x1 >= W - 1), (b.y0 <= 0 or b.y1 >= H - 1)


def visible_fraction(b: Box, W: int, H: int, class_size: tuple | None = None, loose: tuple | None = None) -> float:
    """Estimated visible fraction of a border-touching object (1.0 if it does not touch the border).

    ``loose``: unclipped box (x0, y0, x1, y1) in pixels (e.g. Isaac's bounding_box_2d_loose), else
    ``class_size``: median (w, h) of fully visible boxes of the class.
    """
    tx, ty = touches_border(b, W, H)
    if not (tx or ty):
        return 1.0
    if loose is not None:
        lx0, ly0, lx1, ly1 = loose
        full = max(1.0, (lx1 - lx0 + 1) * (ly1 - ly0 + 1))
        return float(min(1.0, b.w * b.h / full))
    if class_size is None:
        return 1.0  # nothing to compare with: keep
    fx = min(1.0, b.w / class_size[0]) if tx else 1.0
    fy = min(1.0, b.h / class_size[1]) if ty else 1.0
    return float(fx * fy)


def yolo_line(cls_id: int, b: Box, W: int, H: int) -> str:
    """``cls cx cy w h`` (normalised); the box covers pixels x0..x1 inclusive = [x0, x1 + 1) continuous."""
    cx = (b.x0 + b.x1 + 1) / 2.0 / W
    cy = (b.y0 + b.y1 + 1) / 2.0 / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {b.w / W:.6f} {b.h / H:.6f}"


def frame_labels(
    seg: np.ndarray,
    id_to_class: dict[int, str],
    class_ids: dict[str, int],
    cfg: ConvertConfig,
    class_sizes: dict | None = None,
    loose_boxes: dict | None = None,
) -> tuple[list[str], Counter]:
    """YOLO label lines of one frame and the reasons for dropped instances."""
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = seg[..., 0]
    H, W = seg.shape
    boxes, dropped = instance_boxes(seg, id_to_class, cfg.min_area, cfg.min_side)
    lines = []
    for b in boxes:
        if b.cls not in class_ids:
            dropped["unknown_class"] += 1
            continue
        vf = visible_fraction(b, W, H, (class_sizes or {}).get(b.cls), (loose_boxes or {}).get(b.inst))
        if vf < 1.0 - cfg.max_truncation:
            dropped["truncated"] += 1
            continue
        lines.append(yolo_line(class_ids[b.cls], b, W, H))
    return lines, dropped


# ---------------------------------------------------------------------------------------------------- files
@dataclass
class Frame:
    key: str
    rgb: Path
    seg: Path
    info: Path | None


def find_frames(src: str | Path) -> list[Frame]:
    """Frames of a capture directory (flat ``rgb_<k>`` / ``instance_<k>`` or ``rgb/`` + ``instance/`` subdirs)."""
    src = Path(src)
    frames = []
    if (src / "rgb").is_dir() and (src / "instance").is_dir():
        pairs = [(p.stem, p) for p in sorted((src / "rgb").iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        seg_dir, info_dir, prefix = src / "instance", src / "instance", ""
    else:
        pairs = [(p.stem[4:], p) for p in sorted(src.glob("rgb_*")) if p.suffix.lower() in IMAGE_EXTS]
        seg_dir, info_dir, prefix = src, src, "instance_"
    for key, rgb in pairs:
        seg = next((seg_dir / f"{prefix}{key}{e}" for e in SEG_EXTS if (seg_dir / f"{prefix}{key}{e}").exists()), None)
        if seg is None:
            continue
        info = info_dir / f"{prefix}{key}.json"
        frames.append(Frame(key, rgb, seg, info if info.exists() else None))
    return frames


def load_image(p: Path) -> np.ndarray:
    if p.suffix.lower() == ".npy":
        return np.load(p)
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot read {p}")
    return img


def load_seg(p: Path) -> np.ndarray:
    a = load_image(p)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.int64)


def split_keys(keys: list[str], cfg: ConvertConfig) -> dict[str, str]:
    """``{key: "train" | "val"}``. Block mode keeps runs of consecutive frames together."""
    rng = random.Random(cfg.seed)
    out = {}
    if cfg.split_mode == "random":
        idx = list(range(len(keys)))
        rng.shuffle(idx)
        n_val = int(round(cfg.val_frac * len(keys)))
        val = set(idx[:n_val])
        return {k: ("val" if i in val else "train") for i, k in enumerate(keys)}
    blocks = [keys[i : i + cfg.block_size] for i in range(0, len(keys), cfg.block_size)]
    order = list(range(len(blocks)))
    rng.shuffle(order)
    n_val = int(round(cfg.val_frac * len(keys)))
    val_blocks, n = set(), 0
    for b in order:
        if n >= n_val:
            break
        val_blocks.add(b)
        n += len(blocks[b])
    if cfg.val_frac > 0 and not val_blocks and blocks:
        val_blocks.add(order[0])
    for bi, blk in enumerate(blocks):
        for k in blk:
            out[k] = "val" if bi in val_blocks else "train"
    return out


def convert_dataset(src: str | Path, out: str | Path, cfg: ConvertConfig | None = None, mapping: dict | None = None) -> dict:
    """Convert a capture directory into ``out/{images,labels}/{train,val}`` + ``data.yaml``; returns stats."""
    cfg = cfg or ConvertConfig()
    src, out = Path(src), Path(out)
    frames = find_frames(src)
    if not frames:
        raise FileNotFoundError(f"no rgb_*/instance_* (or rgb/ + instance/) frames in {src}")
    global_map = parse_id_mapping(mapping, cfg.prim_rules) if mapping else {}

    def id_map(fr: Frame) -> dict[int, str]:
        m = dict(global_map)
        if fr.info is not None:
            m.update(parse_id_mapping(json.loads(fr.info.read_text()), cfg.prim_rules))
        return m

    # pass 1: class list and median size of fully visible boxes per class
    sizes: dict[str, list] = defaultdict(list)
    seen: set[str] = set()
    for fr in frames:
        seg = load_seg(fr.seg)
        H, W = seg.shape
        boxes, _ = instance_boxes(seg, id_map(fr), cfg.min_area, cfg.min_side)
        for b in boxes:
            seen.add(b.cls)
            tx, ty = touches_border(b, W, H)
            if not (tx or ty):
                sizes[b.cls].append((b.w, b.h))
    classes = list(cfg.classes) if cfg.classes else sorted(seen)
    class_ids = {c: i for i, c in enumerate(classes)}
    class_sizes = {c: tuple(np.median(np.array(v), axis=0)) for c, v in sizes.items() if v}

    # pass 2: write
    splits = split_keys([f.key for f in frames], cfg)
    for sub in ("images", "labels"):
        for sp in ("train", "val"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)
    stats = {"frames": len(frames), "images": Counter(), "boxes": Counter(), "per_class": defaultdict(Counter), "dropped": Counter()}
    for fr in frames:
        seg = load_seg(fr.seg)
        lines, dropped = frame_labels(seg, id_map(fr), class_ids, cfg, class_sizes)
        stats["dropped"].update(dropped)
        if not lines and cfg.skip_empty:
            stats["dropped"]["empty_frame"] += 1
            continue
        sp = splits[fr.key]
        stem = f"{fr.key}"
        if cfg.image_format == "copy" and fr.rgb.suffix.lower() != ".npy":
            dst = out / "images" / sp / f"{stem}{fr.rgb.suffix.lower()}"
            shutil.copyfile(fr.rgb, dst)
        else:
            import cv2  # noqa: PLC0415

            img = load_image(fr.rgb)
            if fr.rgb.suffix.lower() == ".npy":  # numpy captures are RGB(A)
                img = np.ascontiguousarray(img[..., :3][..., ::-1])
            ext = ".jpg" if cfg.image_format == "jpg" else ".png"
            cv2.imwrite(str(out / "images" / sp / f"{stem}{ext}"), img)
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"][sp] += 1
        stats["boxes"][sp] += len(lines)
        for ln in lines:
            stats["per_class"][sp][classes[int(ln.split()[0])]] += 1
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out / "data.yaml").write_text(
        f"# written by drone_gcs seg_to_yolo\npath: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    result = {
        "source": str(src.resolve()),
        "out": str(out.resolve()),
        "classes": classes,
        "frames": stats["frames"],
        "images": dict(stats["images"]),
        "boxes": dict(stats["boxes"]),
        "per_class": {k: dict(v) for k, v in stats["per_class"].items()},
        "dropped": dict(stats["dropped"]),
        "class_median_size_px": {k: [round(float(x), 1) for x in v] for k, v in class_sizes.items()},
        "config": {k: v for k, v in cfg.__dict__.items()},
    }
    (out / "dataset_stats.json").write_text(json.dumps(result, indent=2))
    return result


def parse_prim_rules(s: str | None) -> dict:
    """``"Car=car,Tent=tent"`` -> ``{"Car": "car", "Tent": "tent"}``."""
    out = {}
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out
