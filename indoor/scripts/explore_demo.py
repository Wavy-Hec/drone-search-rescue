"""Draw one exploration of a stand-in floor: the robot's map at six moments, with the frontiers it chose between.

    python scripts/explore_demo.py --seed 1 --out explore_seed1.png

Needs matplotlib, and ``drone_lab`` (the ``sim/`` project) on the path for the A*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent / "sim" / "src"):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from indoor_sar.frontier import FREE, UNKNOWN, FrontierCfg, find_frontiers  # noqa: E402
from indoor_sar.standin import coverage, explore, make_floor  # noqa: E402

# map layers are neutrals; the three things to tell apart take the first three slots of the palette
UNKNOWN_GREY, FREE_WHITE, WALL = "#c9c8c2", "#fcfcfb", "#2b2b29"
TRAIL, TARGET, OTHER = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT = "#0b0b0b", "#52514e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gain-weight", type=float, default=FrontierCfg.gain_weight)
    parser.add_argument("--out", type=Path, default=Path("explore.png"))
    args = parser.parse_args()

    resolution = 0.1
    cfg = FrontierCfg(gain_weight=args.gain_weight)
    occupied, start = make_floor(args.seed, resolution=resolution)
    moments = []

    def record(known, robot, target, path) -> None:
        others = [f for f in find_frontiers(known, robot, resolution, cfg) if f.goal != target.goal]
        moments.append((known.copy(), robot, target, path.copy(), others))

    run = explore(occupied, start, resolution, cfg, on_decision=record)
    score = coverage(occupied, run.known, start, resolution, cfg)

    # five decisions spread over the run, then the finished map
    picks = np.unique(np.linspace(0, len(moments) - 1, 5).round().astype(int))
    steps_before = _trail_index_per_decision(run.trail, [m[1] for m in moments])

    cmap = ListedColormap([UNKNOWN_GREY, FREE_WHITE, WALL])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), facecolor=FREE_WHITE, layout="constrained")
    for ax, k in zip(axes.flat, picks, strict=False):
        known, robot, target, path, others = moments[k]
        _draw_map(ax, known, cmap)
        trail = run.trail[: steps_before[k] + 1]
        ax.plot(trail[:, 1], trail[:, 0], color=TRAIL, lw=1.4, solid_capstyle="round")
        for f in others:
            ax.scatter(f.cells[:, 1], f.cells[:, 0], s=5, color=OTHER, marker="s", linewidths=0)
        ax.scatter(target.cells[:, 1], target.cells[:, 0], s=5, color=TARGET, marker="s", linewidths=0)
        ax.plot(path[:, 1], path[:, 0], color=TARGET, lw=1.4, ls=(0, (3, 2)))
        ax.scatter([target.goal[1]], [target.goal[0]], s=70, color=TARGET, marker="*", edgecolors=FREE_WHITE, zorder=4)
        ax.scatter([robot[1]], [robot[0]], s=42, color=INK, edgecolors=FREE_WHITE, linewidths=1.5, zorder=5)
        driven = np.hypot(*np.diff(trail, axis=0).T).sum() * resolution
        ax.set_title(
            f"decision {k + 1} of {len(moments)} · {driven:.0f} m driven · "
            f"{len(others) + 1} frontier{'s' if others else ''}",
            fontsize=10,
            color=INK_SOFT,
            loc="left",
        )
    ax = axes.flat[len(picks)]
    _draw_map(ax, run.known, cmap)
    ax.plot(run.trail[:, 1], run.trail[:, 0], color=TRAIL, lw=1.4, solid_capstyle="round")
    ax.scatter([run.trail[-1, 1]], [run.trail[-1, 0]], s=42, color=INK, edgecolors=FREE_WHITE, linewidths=1.5, zorder=5)
    ax.set_title(f"finished · {run.length_m:.0f} m driven · no frontier left", fontsize=10, color=INK_SOFT, loc="left")
    for ax in axes.flat[len(picks) + 1 :]:
        ax.axis("off")

    handles = [
        Patch(facecolor=UNKNOWN_GREY, label="not seen yet"),
        Patch(facecolor=FREE_WHITE, edgecolor=INK_SOFT, linewidth=0.5, label="seen: free"),
        Patch(facecolor=WALL, label="seen: wall or rubble"),
        Line2D([], [], color=TRAIL, lw=2, label="driven so far"),
        Line2D([], [], color=TARGET, lw=2, ls=(0, (3, 2)), label="chosen frontier, and the A* path to its goal (star)"),
        Line2D([], [], color=OTHER, lw=3, label="other frontiers"),
        Line2D([], [], color=INK, marker="o", ls="", markersize=6, label="robot"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, frameon=False, fontsize=9, labelcolor=INK_SOFT)
    closed_off = int((~occupied).sum()) - int((run.known == FREE).sum())
    fig.suptitle(
        f"Frontier exploration of stand-in floor {args.seed} (24 × 16 m, 0.1 m cells): "
        f"{score['coverage']:.1%} of the floor the robot can stand on is mapped, {score['wrong_cells']} wrong cells"
        + (f"; {closed_off * resolution**2:.0f} m² are closed off by rubble" if closed_off > 50 else ""),
        fontsize=12,
        color=INK,
        x=0.012,
        ha="left",
    )
    fig.savefig(args.out, dpi=200)
    print(
        f"{args.out}: {run.decisions} decisions, {run.length_m:.1f} m, coverage {score['coverage']:.4f}, "
        f"{run.target_changes} changes of target"
    )


def _draw_map(ax, known: np.ndarray, cmap) -> None:
    layer = np.where(known == UNKNOWN, 0, np.where(known == FREE, 1, 2))
    ax.imshow(layer, cmap=cmap, vmin=0, vmax=2, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _trail_index_per_decision(trail: np.ndarray, robots: list[tuple[int, int]]) -> list[int]:
    """Index into ``trail`` of the robot's cell at each decision (the trail only grows, so search forward)."""
    out, i = [], 0
    for robot in robots:
        while tuple(int(v) for v in trail[i]) != robot:
            i += 1
        out.append(i)
    return out


if __name__ == "__main__":
    main()
