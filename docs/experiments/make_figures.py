#!/usr/bin/env python3
"""Figures and tables of the CPU experiments in docs/HOW_IT_WORKS.md, made from their run folders.

Reads (a part whose runs are missing is skipped, and the script says so):

* ``gcs/runs/exp_sweep_{pos,att,depth,lag}/sweep.csv``: the noise sweeps on the stand-in recording (noise_sweep.py)
* ``gcs/runs/exp_bench_{clean,noisy}/benchmark.md``: the stand-in benchmark tables (aggregate_runs.py)
* ``gcs/runs/exp_visdrone_{sim,coco}_<setting>/external_eval.json``: both detectors on real VisDrone images
* ``sim/runs/exp_surface_{with,without}/``: the rescue plan on the stand-in report, with and without the surface map
* ``sim/runs/exp_tour_gt/``: the rescue tour on the ground truth of the demo site

Writes ``docs/experiments/*.png`` and ``docs/experiments/results.md`` (every number of the figures as a table).
The commands that make the run folders are in docs/HOW_IT_WORKS.md, section "Experiments run on the CPU".

    python docs/experiments/make_figures.py
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
GCS_RUNS, SIM_RUNS, OUT = (
    ROOT / "gcs" / "runs",
    ROOT / "sim" / "runs",
    Path(__file__).resolve().parent,
)

# light chart surface, ink and the first categorical slots of the dataviz reference palette
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
    }
)

SWEEPS = (  # parameter, panel title, x-axis label
    ("pos", "Position drift", "std of the position drift [m]"),
    ("att", "Attitude drift", "std of the attitude drift [deg]"),
    ("depth", "Depth noise", "std as a fraction of the depth"),
    ("lag", "Camera lag", "camera paired with a pose this much later [s]"),
)
VISDRONE = (  # run-folder suffix, label
    ("whole", "whole image\n(all 548 images)"),
    ("whole_first50", "whole image\n(first 50)"),
    ("scale1_first50", "tiles, x1\n(first 50)"),
    ("scale2_first50", "tiles, x2\n(first 50)"),
)
MODELS = (
    ("sim", "fine-tuned on simulator images", BLUE),
    ("coco", "stock COCO YOLO26n", ORANGE),
)


def style(ax, grid_axis: str = "y") -> None:
    """Hairline solid grid on one axis, no top/right frame."""
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for one value)."""
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


# ------------------------------------------------------------------------------------------------- noise sweeps
def read_sweep(param: str) -> list[dict] | None:
    """Per noise level: value, seeds, mean/sd of pin error, people pin error and false alarms, people found."""
    path = GCS_RUNS / f"exp_sweep_{param}" / "sweep.csv"
    if not path.is_file():
        return None
    by_value: dict[float, list[dict]] = {}
    for row in csv.DictReader(path.open()):
        if row["ok"] == "1":
            by_value.setdefault(float(row["value"]), []).append(row)
    levels = []
    for value, rows in sorted(by_value.items()):
        level = {"value": value, "seeds": len(rows)}
        for key in (
            "pin_error_m",
            "people_pin_error_m",
            "false_alarms",
            "people_found",
        ):
            level[key] = mean_sd([float(r[key]) for r in rows])
        level["people_total"] = int(rows[0]["people_total"])
        levels.append(level)
    return levels


def plot_sweeps(sweeps: dict[str, list[dict]]) -> Path:
    fig, axes = plt.subplots(2, len(SWEEPS), figsize=(14, 6.2), sharey="row")
    for col, (param, title, xlabel) in enumerate(SWEEPS):
        levels = sweeps.get(param)
        for row, (key, ylabel) in enumerate(
            (("pin_error_m", "pin error [m]"), ("false_alarms", "false alarms"))
        ):
            ax = axes[row, col]
            style(ax)
            if not levels:
                ax.text(
                    0.5,
                    0.5,
                    "not run",
                    ha="center",
                    va="center",
                    color=MUTED,
                    transform=ax.transAxes,
                )
                continue
            x = np.array([lv["value"] for lv in levels])
            m = np.array([lv[key][0] for lv in levels])
            sd = np.array([lv[key][1] for lv in levels])
            ax.fill_between(x, m - sd, m + sd, color=BLUE, alpha=0.12, linewidth=0)
            ax.plot(x, m, color=BLUE, linewidth=2, solid_capstyle="round")
            ax.plot(
                x,
                m,
                "o",
                color=BLUE,
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
            ax.annotate(
                f"{m[-1]:.2f}" if key == "pin_error_m" else f"{m[-1]:.1f}",
                (x[-1], m[-1]),
                textcoords="offset points",
                xytext=(-4, 7),
                ha="right",
                color=INK2,
                fontsize=9,
            )
            if row == 0:
                seeds = levels[0]["seeds"]
                ax.set_title(
                    f"{title}  ({seeds} noise seed{'s' if seeds > 1 else ''})",
                    loc="left",
                )
            if row == 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylim(bottom=0)
    fig.suptitle(
        "Stand-in mission, world seed 0, replayed with sensor noise: pin error (all 10 objects) and false alarms",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.925,
        "Line: mean over the noise seeds; band: one sample standard deviation. Blob detector, "
        "lossless replays of one recording (noise_sweep.py).",
        color=INK2,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "noise_sweeps.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def sweep_tables(sweeps: dict[str, list[dict]]) -> str:
    lines = []
    for param, title, xlabel in SWEEPS:
        levels = sweeps.get(param)
        if not levels:
            continue
        lines += [f"**{title}** ({xlabel})", ""]
        lines += [
            "| Level | Seeds | People found | False alarms | Pin error [m] | Pin error, people [m] |",
            "|---|---|---|---|---|---|",
        ]
        for lv in levels:
            pf, fa, pe, pp = (
                lv[k]
                for k in (
                    "people_found",
                    "false_alarms",
                    "pin_error_m",
                    "people_pin_error_m",
                )
            )
            lines.append(
                f"| {lv['value']:g} | {lv['seeds']} | {pf[0]:.1f} / {lv['people_total']} | {fa[0]:.1f} ± {fa[1]:.1f} "
                f"| {pe[0]:.3f} ± {pe[1]:.3f} | {pp[0]:.3f} ± {pp[1]:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- real images
def read_visdrone() -> dict[tuple[str, str], dict]:
    runs = {}
    for model, _, _ in MODELS:
        for setting, _ in VISDRONE:
            path = GCS_RUNS / f"exp_visdrone_{model}_{setting}" / "external_eval.json"
            if path.is_file():
                runs[(model, setting)] = json.loads(path.read_text())
    return runs


def bar_group(
    ax, labels: list[str], series: list[tuple[str, str, list[float | None]]], fmt: str
) -> None:
    """Grouped bars: one group per label, one bar per series (name, colour, values); values at the bar tips."""
    n = len(series)
    width = min(0.8 / n, 0.32)
    x = np.arange(len(labels))
    for k, (name, color, values) in enumerate(series):
        pos = x + (k - (n - 1) / 2) * (width + 0.03)
        vals = np.array([np.nan if v is None else v for v in values], float)
        ax.bar(pos, vals, width=width, color=color, label=name, linewidth=0)
        for p, v in zip(pos, vals, strict=True):
            if np.isfinite(v):
                ax.annotate(
                    fmt.format(v),
                    (p, v),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=8.5,
                    color=INK2,
                )
    ax.set_xticks(x, labels)
    style(ax)


def plot_visdrone(runs: dict) -> list[Path]:
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [label for _, label in VISDRONE]
    for ax, key, title in (
        (axes[0], "recall", "Recall: share of the labelled people found"),
        (axes[1], "precision", "Precision: share of the person boxes that are people"),
    ):
        series = [
            (name, color, [runs.get((model, s), {}).get(key) for s, _ in VISDRONE])
            for model, name, color in MODELS
        ]
        bar_group(ax, labels, series, "{:.2f}")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="upper left", fontsize=9)
    n_img = runs.get(("sim", "whole"), {}).get("images", "?")
    n_people = runs.get(("sim", "whole"), {}).get("people_labelled", "?")
    fig.suptitle(
        f"People on real drone photos: VisDrone2019-DET val ({n_img} images, {n_people} people labelled)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.905,
        "Person boxes matched one to one to the labels (pedestrian + people) at IoU 0.45. "
        "Tiles: 640 x 480 crops of the image at its size (x1) or resized to twice its size (x2), "
        "on the first 50 images.",
        color=INK2,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = OUT / "visdrone_people.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)

    setting = "whole"
    if all((m, setting) in runs for m, _, _ in MODELS):
        buckets = list(runs[("sim", setting)]["by_size"])
        fig, ax = plt.subplots(figsize=(9, 4.4))
        series = [
            (
                name,
                color,
                [runs[(model, setting)]["by_size"][b]["recall"] for b in buckets],
            )
            for model, name, color in MODELS
        ]
        counts = [runs[("sim", setting)]["by_size"][b]["labelled"] for b in buckets]
        bar_group(
            ax,
            [f"{b} px\n({c} people)" for b, c in zip(buckets, counts, strict=True)],
            series,
            "{:.2f}",
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("longer side of the label box in the photo")
        ax.set_ylabel("recall")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title(
            "Recall by person size, whole image (all 548 images)",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        path = OUT / "visdrone_by_size.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def visdrone_tables(runs: dict) -> str:
    lines = [
        "| Detector | Images given | Images | People labelled | Found | Missed | False alarms | Precision | Recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, name, _ in MODELS:
        for setting, label in VISDRONE:
            r = runs.get((model, setting))
            if r is None:
                continue
            p = "-" if r["precision"] is None else f"{r['precision']:.3f}"
            rc = "-" if r["recall"] is None else f"{r['recall']:.3f}"
            lines.append(
                f"| {name} | {label.replace(chr(10), ' ')} | {r['images']} | {r['people_labelled']} | "
                f"{r['found']} | {r['missed']} | {r['false_alarms']} | {p} | {rc} |"
            )
    lines.append("")
    for model, name, _ in MODELS:
        r = runs.get((model, "whole"))
        if r is None:
            continue
        lines += [
            f"Recall by label size, {name}, whole image (all 548):",
            "",
            "| Longer side [px] | People labelled | Found | Recall |",
            "|---|---|---|---|",
        ]
        for b, s in r["by_size"].items():
            rc = "-" if s["recall"] is None else f"{s['recall']:.3f}"
            lines.append(f"| {b} | {s['labelled']} | {s['found']} | {rc} |")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- rescue planner
def plot_surface() -> tuple[Path, str] | None:
    runs = {k: SIM_RUNS / f"exp_surface_{k}" for k in ("with", "without")}
    if not all(
        (r / "traversability.npz").is_file() and (r / "plan.json").is_file()
        for r in runs.values()
    ):
        return None
    t = {k: np.load(r / "traversability.npz") for k, r in runs.items()}
    plan = json.loads((runs["with"] / "plan.json").read_text())
    x_min, y_min, res, nx, ny = t["with"]["grid"]
    extent = (x_min, x_min + nx * res, y_min, y_min + ny * res)
    before, after = t["without"]["lethal"], t["with"]["lethal"]
    added = after & ~before
    layer = np.zeros(before.shape, np.uint8)
    layer[before] = 1
    layer[added] = 2
    cmap = ListedColormap([SURFACE, "#9c9a92", ORANGE])
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.imshow(
        layer,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    path_xy = np.asarray(plan["path_xy"])
    ax.plot(
        path_xy[:, 0], path_xy[:, 1], color=BLUE, linewidth=2.5, solid_capstyle="round"
    )
    sx, sy = plan["start_xy"]
    px, py = plan["person_xy"]
    ax.plot(
        sx,
        sy,
        "o",
        markersize=10,
        markerfacecolor=SURFACE,
        markeredgecolor=INK,
        markeredgewidth=2,
    )
    ax.plot(
        px,
        py,
        "*",
        markersize=16,
        color=INK,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    style(ax, "both")
    n_marked, n_added = (
        plan["surface_obstacle_cells"],
        plan["surface_obstacle_cells_added"],
    )
    ax.set_title(
        f"Surface-map obstacles (stand-in report of the demo site):\n{n_added} of the {n_marked} marked cells "
        "were open ground before",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    handles = [
        Patch(
            color="#9c9a92",
            label="no-go without the surface map (terrain, pins, robot radius)",
        ),
        Patch(
            color=ORANGE,
            label=f"no-go only with the surface map ({int(added.sum())} cells, inflated)",
        ),
        Line2D(
            [],
            [],
            color=BLUE,
            linewidth=2.5,
            label=f"route to the person, {plan['plan']['length_m']:.1f} m"
            if "length_m" in plan.get("plan", {})
            else "route to the person",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        fontsize=9,
        ncol=1,
    )
    fig.tight_layout()
    path = OUT / "surface_obstacles.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    table = "\n".join(
        [
            "| | Cells |",
            "|---|---|",
            f"| Marked by the surface map (more than 0.2 m above the elevation map) | {n_marked} |",
            f"| ... of them open ground for the terrain limits and the pins | {n_added} |",
            f"| No-go cells only with the surface map, after the robot-radius inflation | {int(added.sum())} |",
            f"| No-go cells without / with the surface map | {int(before.sum())} / {int(after.sum())} |",
            f"| Grid | {int(nx)} x {int(ny)} cells of {res:g} m |",
            "",
        ]
    )
    return path, table


def tour_section() -> tuple[Path, str] | None:
    run = SIM_RUNS / "exp_tour_gt"
    if not (run / "tour.json").is_file():
        return None
    tour = json.loads((run / "tour.json").read_text())
    path = OUT / "tour_ground_truth.png"
    shutil.copyfile(run / "tour.png", path)
    lines = [
        "| Leg | Person | Stop to person [m] | Length [m] | Cost | Walking time [s] |",
        "|---|---|---|---|---|---|",
    ]
    for leg in tour["legs"]:
        lines.append(
            f"| {leg['leg']} | {leg['pin']} | {leg['stop_to_person_m']:.2f} | {leg['length_m']:.1f} | "
            f"{leg['cost']:.1f} | {leg['walk_time_s']:.0f} |"
        )
    lines.append(
        f"| total | {len(tour['legs'])} people | | {tour['length_m']:.1f} | {tour['cost']:.1f} | "
        f"{tour['walk_time_s']:.0f} |"
    )
    unreachable = tour.get("unreachable", [])
    order = (
        f"Order by {tour['method']}; walking time at {tour['cruise_speed_mps']:g} m/s; not reachable within 2.5 m: "
        f"{', '.join(map(str, unreachable)) or 'none'}."
    )
    lines += ["", order, ""]
    return path, "\n".join(lines)


# ------------------------------------------------------------------------------------------------- rewards
# Reward terms of the two drone tasks, as in their configs (dt = 0.1 s per policy step: the reward manager adds
# weight * term * dt). Trajectory task: sim/src/drone_lab/tasks/track_trajectory/config/arl_robot_1/
# track_trajectory_env_cfg.py:91-110. Hover task: isaaclab_tasks/contrib/drone_arl/track_position_state_based/config/
# arl_robot_1/track_position_state_based_env_cfg.py:158-193 (Isaac Lab 3.0.0; sim/README.md names this task as the
# one the released policy was trained on, but not the Isaac Lab version it was trained with).
DT = 0.1
HOVER_GOAL = (25.0, 1.5)  # distance_to_goal_exp: weight, std [m]
TRACK_FINE, TRACK_COARSE, TRACK_VEL = (
    (15.0, 0.4),
    (5.0, 1.5),
    (5.0, 1.0),
)  # weight, std [m] or [m/s]
TOO_FAR_M = (
    2.5  # the trajectory task ends an episode beyond this distance from the reference
)


def plot_rewards() -> Path:
    """Per-step reward of the tracking terms against the error: why the trajectory task has a fine and a coarse term."""
    d = np.linspace(0, 3, 301)

    def term(weight: float, std: float) -> np.ndarray:
        return DT * weight * np.exp(-(d**2) / std**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    ax = axes[0]
    ax.plot(
        d,
        term(*HOVER_GOAL),
        color=BLUE,
        linewidth=2,
        label="distance_to_goal_exp (25, std 1.5 m)",
    )
    ax.set_title("Hover task, as configured in Isaac Lab 3.0.0: goal term", loc="left")
    ax.set_xlabel("distance from the goal [m]")
    ax.set_ylabel("reward per 0.1 s step")
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    fine, coarse = term(*TRACK_FINE), term(*TRACK_COARSE)
    ax.plot(d, fine, color=BLUE, linewidth=2, label="tracking_fine (15, std 0.4 m)")
    ax.plot(
        d, coarse, color=ORANGE, linewidth=2, label="tracking_coarse (5, std 1.5 m)"
    )
    ax.plot(
        d,
        term(*TRACK_VEL),
        color="#1baf7a",
        linewidth=2,
        label="velocity_tracking (5, std 1.0 m/s), x = speed error",
    )
    ax.axvline(TOO_FAR_M, ymax=0.45, color=MUTED, linewidth=1)  # stops below the legend
    ax.annotate(
        "episode ends\n(too_far, 2.5 m)",
        (TOO_FAR_M, 0.9),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK2,
        fontsize=9,
    )
    ax.set_title("Trajectory task: tracking terms", loc="left")
    ax.set_xlabel("distance from the reference [m] (or speed error [m/s])")
    ax.legend(loc="upper right", fontsize=9)
    for ax in axes:
        style(ax, "both")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2.6)
    fig.suptitle(
        "Reward per step against the tracking error: weight x exp(-error^2 / std^2) x 0.1 s",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "reward_terms.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sections, notes = [], []
    notes.append(f"wrote {plot_rewards().name}")
    sweeps = {p: s for p, _, _ in SWEEPS if (s := read_sweep(p))}
    if sweeps:
        notes.append(f"wrote {plot_sweeps(sweeps).name}")
        sections.append(
            "## Sensor-noise sweeps (stand-in mission, world seed 0)\n\n"
            + sweep_tables(sweeps)
        )
    else:
        notes.append("no noise sweep found (gcs/runs/exp_sweep_*)")
    bench = []
    for kind, title in (
        ("clean", "without noise"),
        ("noisy", "replayed with pos=0.5, att=1, depth=0.02"),
    ):
        md = GCS_RUNS / f"exp_bench_{kind}" / "benchmark.md"
        if md.is_file():
            body = (
                md.read_text().split("\n", 1)[1]
                if md.read_text().startswith("#")
                else md.read_text()
            )
            bench.append(f"### Stand-in benchmark, {title}\n{body}")
    if bench:
        sections.append(
            "## Stand-in benchmark (world seeds 0-4)\n\n" + "\n".join(bench)
        )
    else:
        notes.append("no benchmark table found (gcs/runs/exp_bench_*)")
    runs = read_visdrone()
    if runs:
        notes += [f"wrote {p.name}" for p in plot_visdrone(runs)]
        sections.append(
            "## People on real drone photos (VisDrone2019-DET val)\n\n"
            + visdrone_tables(runs)
        )
    else:
        notes.append("no VisDrone run found (gcs/runs/exp_visdrone_*)")
    if surface := plot_surface():
        notes.append(f"wrote {surface[0].name}")
        sections.append(
            "## Surface-map obstacles (stand-in report of the demo site)\n\n"
            + surface[1]
        )
    else:
        notes.append("no surface-obstacle plans found (sim/runs/exp_surface_*)")
    if tour := tour_section():
        notes.append(f"copied {tour[0].name}")
        sections.append("## Rescue tour (ground truth of the demo site)\n\n" + tour[1])
    else:
        notes.append("no tour found (sim/runs/exp_tour_gt)")
    header = (
        "# Results of the CPU experiments\n\nMade by `docs/experiments/make_figures.py` from the run folders; "
        "what each experiment does and how to run it: "
        "[HOW_IT_WORKS.md](../HOW_IT_WORKS.md#12-experiments-run-on-the-cpu).\n"
    )
    (OUT / "results.md").write_text(header + "\n" + "\n".join(sections))
    print("\n".join(notes + [f"wrote {OUT / 'results.md'}"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
