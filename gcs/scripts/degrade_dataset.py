#!/usr/bin/env python3
"""Write corrupted copies of one split of a YOLO dataset: harder tests for the detector.

For every preset of :data:`drone_gcs.corruptions.PRESETS` (default: all of them) this writes
``OUT/<preset>/images/<split>/*.png``, ``OUT/<preset>/labels/<split>/*.txt`` and ``OUT/<preset>/data.yaml`` (the
class names of the source, ``path`` = that folder, ``<split>: images/<split>``), so that
``scripts/eval_detector.py --data OUT/<preset>/data.yaml --splits <split>`` scores it like the source split. The
Ultralytics validator refuses a data.yaml without a ``train`` key, so ``train`` points at the corrupted split too:
the copies are for testing, not for training.

The noise, blur, dark, jpeg and haze presets change the pixels only, and the occlusion presets hide part of
every labelled object; their labels are those of the source frame. The altitude presets tile consecutive groups
of ``k * k`` frames (in sorted file order) into one frame of the same size, with the labels moved and shrunk to
match; a last group with fewer frames is dropped and reported. Every preset draws its random numbers from its own
generator seeded with ``--seed``, so a copy does not depend on which other presets are written.

Example::

    python scripts/degrade_dataset.py --data ../datasets/drone_sar_yolo_v2/yolo/data.yaml --split val \\
        --out runs/degraded_val --presets dark,occlude30,altitude2
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_gcs.corruptions import PRESETS, Label, shrink_mosaic  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_detector", PROJECT_ROOT / "scripts" / "eval_detector.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)  # read_labels and IMAGE_SUFFIXES; it imports Ultralytics only inside its functions


def read_data_yaml(data: Path) -> tuple[Path, dict[str, str], dict[int, str]]:
    """Dataset root, split -> image folder, and class names of a data.yaml written by seg_to_yolo.

    The same as ``eval_detector.read_names``, which needs Ultralytics' YAML reader; this one reads only that
    file's plain ``key: value`` lines and the ``id: name`` lines indented under ``names:``.
    """
    root, splits, names = data.parent, {}, {}
    for line in data.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = (s.strip() for s in line.partition(":"))
        if line[0].isspace():
            names[int(key)] = value
        elif key == "path" and value:
            root = Path(value)
        elif key in ("train", "val", "test") and value:
            splits[key] = value
    return root, splits, names


def write_frame(out: Path, split: str, stem: str, rgb: np.ndarray, labels: list[Label]) -> None:
    """One frame as ``images/<split>/<stem>.png`` and its ``labels/<split>/<stem>.txt`` under ``out``."""
    cv2.imwrite(str(out / "images" / split / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels]
    (out / "labels" / split / f"{stem}.txt").write_text("".join(line + "\n" for line in lines))


def degrade(data: Path, split: str, out: Path, presets: list[str], seed: int = 0) -> dict[str, dict]:
    """Write one corrupted copy of ``split`` per preset into ``out/<preset>`` (see the module docstring).

    Returns ``{preset: {"frames": written, "labels": written, "dropped_frames": source frames left out}}``.
    Raises ``FileExistsError`` when ``out/<preset>`` exists already (its old frames would be scored with the new).
    """
    root, splits, names = read_data_yaml(data)
    if split not in splits:
        raise ValueError(f"{data} has no split {split!r} (it has {', '.join(splits) or 'none'})")
    image_dir = root / splits[split]
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in ev.IMAGE_SUFFIXES)
    labels = ev.read_labels(root, splits[split])
    frames = []  # (stem, rgb, labels) of the source split, in sorted file order
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"cannot read {path}")
        frames.append((path.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), labels.get(path.stem, [])))
    for name in presets:
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists already: choose another output folder")
    stats = {}
    for name in presets:
        preset, dest = PRESETS[name], out / name
        for sub in ("images", "labels"):
            (dest / sub / split).mkdir(parents=True)
        rng = np.random.default_rng(seed)
        n_labels, n = 0, preset.k * preset.k
        if preset.frame is not None:
            for stem, rgb, frame_labels in frames:
                write_frame(dest, split, stem, preset.frame(rgb, frame_labels, rng), frame_labels)
                n_labels += len(frame_labels)
            written, dropped = len(frames), 0
        else:
            written, dropped = len(frames) // n, len(frames) % n
            for g in range(written):
                group = frames[g * n : (g + 1) * n]
                rgb, tiled_labels = shrink_mosaic([f[1] for f in group], [f[2] for f in group], preset.k)
                write_frame(dest, split, f"{group[0][0]}_to_{group[-1][0]}", rgb, tiled_labels)
                n_labels += len(tiled_labels)
        names_yaml = "".join(f"  {i}: {c}\n" for i, c in sorted(names.items()))
        train = "" if split == "train" else f"train: images/{split}\n"  # only because Ultralytics requires one
        (dest / "data.yaml").write_text(
            f"# written by drone_gcs degrade_dataset: {name} ({preset.about}), from {split} of {data}\n"
            f"path: {dest.resolve()}\n{train}{split}: images/{split}\nnames:\n{names_yaml}"
        )
        stats[name] = {"frames": written, "labels": n_labels, "dropped_frames": dropped}
    return stats


def parse_presets(text: str | None) -> list[str]:
    """``"dark,blur"`` -> ``["dark", "blur"]``; None or empty -> every preset. Raises ``ValueError`` on an unknown
    name."""
    names = [s.strip() for s in text.split(",") if s.strip()] if text else list(PRESETS)
    unknown = [s for s in names if s not in PRESETS]
    if unknown or not names:
        raise ValueError(f"unknown preset {', '.join(unknown) or '(none given)'}; presets: {', '.join(PRESETS)}")
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml from scripts/seg_to_yolo.py")
    p.add_argument("--split", default="val", help="the split to copy (default val)")
    p.add_argument("--out", required=True, metavar="DIR", help="one folder per preset is written in here")
    p.add_argument("--presets", default=None, help=f"comma-separated, default all: {','.join(PRESETS)}")
    p.add_argument("--seed", type=int, default=0, help="seed of the random corruptions (default 0)")
    a = p.parse_args(argv)
    try:
        presets = parse_presets(a.presets)
        stats = degrade(Path(a.data).expanduser().resolve(), a.split, Path(a.out).expanduser(), presets, a.seed)
    except (ValueError, FileExistsError) as e:
        p.error(str(e))
    for name, s in stats.items():
        line = f"{name}: {s['frames']} frames, {s['labels']} labels -> {Path(a.out) / name / 'data.yaml'}"
        if s["dropped_frames"]:
            line += f" ({s['dropped_frames']} last frames left out: too few for one more tiled frame)"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
