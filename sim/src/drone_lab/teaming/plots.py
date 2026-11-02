"""Figures of the rescue finale: the plan, the walked path, and a minimap for the video (matplotlib, Agg).

Colour roles follow :mod:`drone_lab.plotting` (same validated palette and ink tokens):

* elevation (magnitude): the neutral one-hue ramp with a hillshade;
* traversal cost (magnitude): the blue sequential ramp (light = cheap, dark = expensive);
* no-go cells (lethal after inflation): primary ink, semi-opaque; unknown cells: surface colour + outline;
* identity: planned route = categorical slot 2 (orange), walked path = slot 1 (blue), each with a surface halo;
  the raw A* path is a thin dashed muted line;
* falls use the reserved status colour *critical* and always carry an "x" marker and a "fall" label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon, Rectangle  # noqa: E402

from drone_lab.plotting import Theme, get_theme  # noqa: E402

from .planner import PlanResult  # noqa: E402
from .traversability import TraversabilityMap  # noqa: E402

__all__ = ["BLUE_SEQUENTIAL", "STATUS_CRITICAL", "STATUS_GOOD", "plot_plan", "plot_walk", "render_minimap"]

BLUE_SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
"""Blue sequential ramp (steps 100, 200, 300, 400, 500, 600, 700 of the reference palette)."""
STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"


# ------------------------------------------------------------------------------------------------- helpers
def _theme(theme) -> Theme:
    return theme if isinstance(theme, Theme) else get_theme(theme)


def _style_axes(ax, th: Theme):
    ax.set_facecolor(th.surface)
    for spine in ax.spines.values():
        spine.set_color(th.axis)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=th.ink_muted, labelcolor=th.ink_secondary, labelsize=8, width=0.6)
    ax.xaxis.label.set_color(th.ink_secondary)
    ax.yaxis.label.set_color(th.ink_secondary)


def _halo(th: Theme, width: float = 3.2):
    return [patheffects.withStroke(linewidth=width, foreground=th.surface)]


def _text_halo(th: Theme):
    return [patheffects.withStroke(linewidth=2.4, foreground=th.surface)]


def _hillshade_rgb(z: np.ndarray, th: Theme, res: float) -> np.ndarray:
    """Neutral elevation ramp with a soft hillshade; unknown cells get the surface colour."""
    known = np.isfinite(z)
    zf = np.where(known, z, np.nanmin(z) if known.any() else 0.0)
    lo, hi = (np.nanpercentile(z, 1), np.nanpercentile(z, 99)) if known.any() else (0.0, 1.0)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    cmap = LinearSegmentedColormap.from_list(f"elev_{th.name}", list(th.elevation_ramp))
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(zf, cmap=cmap, vert_exag=1.0, dx=res, dy=res, blend_mode="soft", vmin=lo, vmax=hi)[..., :3]
    surface = np.array(matplotlib.colors.to_rgb(th.surface))
    rgb[~known] = surface
    return rgb


def _scale_bar(ax, th: Theme, x0: float, y0: float, span: float):
    nice = [1, 2, 5, 10, 20, 50]
    length = max([n for n in nice if n <= 0.3 * span] or [nice[0]])
    h = 0.012 * span
    ax.add_patch(Rectangle((x0, y0), length, h, color=th.ink, lw=0, zorder=8))
    ax.text(
        x0 + length + 0.015 * span,
        y0 + 0.5 * h,
        f"{length:g} m",
        ha="left",
        va="center",
        fontsize=7.5,
        color=th.ink,
        zorder=8,
        path_effects=_text_halo(th),
    )


def _view_bounds(tmap: TraversabilityMap, pts: list[np.ndarray], margin: float, full: bool):
    g = tmap.grid
    if full or not pts:
        return g.x_min, g.x_max, g.y_min, g.y_max
    p = np.vstack([np.asarray(q, dtype=float).reshape(-1, 2) for q in pts if q is not None and len(q)])
    lo = p.min(axis=0) - margin
    hi = p.max(axis=0) + margin
    # square view, inside the grid
    c = 0.5 * (lo + hi)
    half = 0.5 * max(hi - lo)
    x0, x1 = max(g.x_min, c[0] - half), min(g.x_max, c[0] + half)
    y0, y1 = max(g.y_min, c[1] - half), min(g.y_max, c[1] + half)
    return x0, x1, y0, y1


def _draw_pins(ax, th: Theme, pins, goal_pin, *, labels=True):
    goal_id = None if goal_pin is None else goal_pin.id
    for p in pins:
        is_goal = goal_id is not None and p.id == goal_id
        if p.label == "person":
            ax.plot(p.x, p.y, marker="*", ms=13 if is_goal else 9, color=th.ink, mec=th.surface, mew=1.0, zorder=9)
        else:
            ax.plot(p.x, p.y, marker="s", ms=5, color=th.ink_secondary, mec=th.surface, mew=0.8, zorder=8)
        if labels and (is_goal or p.label == "person"):
            name = p.label if str(p.id) in ("goal", "") or str(p.id).startswith(p.label) else f"{p.label} #{p.id}"
            name = str(p.id) if str(p.id).startswith(p.label) else name
            text = f"{name} (goal)" if is_goal else name
            ax.annotate(
                text,
                (p.x, p.y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=7,
                color=th.ink if is_goal else th.ink_secondary,
                zorder=10,
                path_effects=_text_halo(th),
            )


def _draw_footprints(ax, th: Theme, tmap: TraversabilityMap):
    for fp in tmap.footprints:
        if getattr(fp, "polygon", None) is not None:
            patch = Polygon(
                np.asarray(fp.polygon), closed=True, fill=False, ec=th.ink_secondary, lw=0.8, ls=(0, (2, 1.5)), zorder=7
            )
        else:
            patch = Circle((fp.x, fp.y), fp.radius, fill=False, ec=th.ink_secondary, lw=0.8, ls=(0, (2, 1.5)), zorder=7)
        ax.add_patch(patch)


def _cost_layers(ax, tmap: TraversabilityMap, th: Theme, cmax: float):
    g = tmap.grid
    cost = np.where(np.isfinite(tmap.cost), tmap.cost, np.nan)
    cmap = LinearSegmentedColormap.from_list("cost_blue", list(BLUE_SEQUENTIAL))
    cmap.set_bad((0, 0, 0, 0))
    im = ax.imshow(
        cost, origin="lower", extent=g.extent, cmap=cmap, vmin=1.0, vmax=cmax, interpolation="nearest", zorder=1
    )
    lethal = np.where(tmap.lethal & ~tmap.unknown, 1.0, np.nan)
    ax.imshow(
        lethal,
        origin="lower",
        extent=g.extent,
        cmap=ListedColormap([th.ink]),
        alpha=0.82,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=2,
    )
    unknown = np.where(tmap.unknown, 1.0, np.nan)
    if np.any(tmap.unknown):
        ax.imshow(
            unknown,
            origin="lower",
            extent=g.extent,
            cmap=ListedColormap([th.surface]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            zorder=2,
        )
        ax.contour(
            *np.meshgrid(*g.cell_centers()),
            tmap.unknown.astype(float),
            levels=[0.5],
            colors=[th.ink_muted],
            linewidths=0.6,
            zorder=3,
        )
    return im


def _route(ax, th: Theme, xy, color, *, lw=2.0, ls="-", zorder=6, halo=True, label=None):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return None
    (line,) = ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color,
        lw=lw,
        ls=ls,
        zorder=zorder,
        solid_capstyle="round",
        path_effects=_halo(th, lw + 2.2) if halo else None,
        label=label,
    )
    return line


# ------------------------------------------------------------------------------------------------ figures
def plot_plan(
    tmap: TraversabilityMap,
    plan: PlanResult,
    out_path: str | Path,
    *,
    pins=(),
    goal_pin=None,
    source: str = "",
    robot: str = "Go2",
    theme: str | Theme = "light",
    full_view: bool = True,
) -> Path:
    """Two panels: the elevation the plan used (left) and the traversability cost with no-go cells (right),
    both with the route, the start, the goal and the pins."""
    th = _theme(theme)
    g = tmap.grid
    c = tmap.cfg
    fig = plt.figure(figsize=(13.0, 7.0), dpi=150, facecolor=th.surface)
    axes = [fig.add_axes([0.05, 0.07, 0.42, 0.72]), fig.add_axes([0.52, 0.07, 0.42, 0.72])]
    cax = fig.add_axes([0.957, 0.15, 0.010, 0.56])
    view = _view_bounds(tmap, [plan.path_xy, plan.start_xy[None], plan.goal_xy[None]], 4.0, full_view)
    finite = tmap.cost[np.isfinite(tmap.cost)]
    cmax = float(np.percentile(finite, 99)) if finite.size else 2.0
    cmax = max(cmax, 1.5)

    axes[0].imshow(
        _hillshade_rgb(tmap.elevation, th, g.resolution),
        origin="lower",
        extent=g.extent,
        interpolation="nearest",
        zorder=1,
    )
    im = _cost_layers(axes[1], tmap, th, cmax)
    for ax, panel in zip(axes, ("Elevation the robot planned on", "Traversability cost and no-go cells"), strict=False):
        _style_axes(ax, th)
        _draw_footprints(ax, th, tmap)
        if len(plan.raw_xy):
            _route(ax, th, plan.raw_xy, th.ink_muted, lw=0.8, ls=(0, (3, 2)), zorder=5, halo=False)
        if plan.success:
            _route(ax, th, plan.path_xy, th.series[1], lw=2.4, zorder=6)
        _draw_pins(ax, th, pins, goal_pin, labels=(ax is axes[0]))
        ax.plot(*plan.start_xy, marker="o", ms=9, mfc=th.surface, mec=th.ink, mew=1.8, zorder=10)
        ax.annotate(
            "start",
            plan.start_xy,
            xytext=(7, -11),
            textcoords="offset points",
            fontsize=7.5,
            color=th.ink,
            zorder=10,
            path_effects=_text_halo(th),
        )
        if plan.goal_snapped_xy is not None:
            ax.plot(*plan.goal_snapped_xy, marker="o", ms=6, color=th.series[1], mec=th.surface, mew=1.0, zorder=10)
        if goal_pin is None:
            ax.plot(*plan.goal_xy, marker="*", ms=13, color=th.ink, mec=th.surface, mew=1.0, zorder=9)
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.text(
            0.0,
            1.015,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=th.ink,
            fontweight="semibold",
        )
        _scale_bar(
            ax, th, view[0] + 0.04 * (view[1] - view[0]), view[2] + 0.04 * (view[3] - view[2]), view[1] - view[0]
        )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cost per metre (1 = flat open ground)", color=th.ink_secondary, fontsize=8)
    cb.outline.set_edgecolor(th.axis)
    cb.ax.tick_params(colors=th.ink_muted, labelcolor=th.ink_secondary, labelsize=7.5)

    handles = [
        Line2D([], [], color=th.series[1], lw=2.4, label="planned route (smoothed)"),
        Line2D([], [], color=th.ink_muted, lw=0.8, ls=(0, (3, 2)), label="A* grid path"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=th.surface, mec=th.ink, mew=1.6, label="start"),
        Line2D([], [], marker="*", ls="", ms=11, color=th.ink, label="person (goal)"),
        Patch(facecolor=th.ink, alpha=0.82, label="no-go (slope, step, roughness or object, + robot radius)"),
    ]
    if tmap.footprints:
        handles.append(Line2D([], [], color=th.ink_secondary, lw=0.8, ls=(0, (2, 1.5)), label="object footprint"))
    if np.any(tmap.unknown):
        handles.append(Patch(facecolor=th.surface, edgecolor=th.ink_muted, label="not seen by the drone"))
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.045, 0.892),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        labelcolor=th.ink_secondary,
        handlelength=2.2,
        columnspacing=1.6,
    )

    if plan.success:
        title = f"Rescue route: {plan.length_m:.1f} m to the person"
    else:
        title = "Rescue route: no safe route found"
    summ = tmap.summary()
    sub1 = (
        f"{robot} limits: slope <= {c.max_slope_deg:g} deg, step <= {c.max_step_m:g} m, roughness <= "
        f"{c.max_roughness_m:g} m, robot radius {c.robot_radius_m:g} m  |  {100 * summ['free_fraction']:.0f} % "
        f"of the map walkable"
    )
    sub2 = source
    if plan.success and plan.goal_snap_m > 0.05:
        sub2 += f"  |  stops {plan.goal_snap_m:.1f} m from the person's centre"
    if not plan.success:
        sub2 += f"  |  {plan.message}"
    fig.text(0.04, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=th.ink)
    fig.text(0.04, 0.938, sub1, ha="left", va="top", fontsize=8.5, color=th.ink_secondary)
    fig.text(0.04, 0.913, sub2, ha="left", va="top", fontsize=8.5, color=th.ink_secondary)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=th.surface)
    plt.close(fig)
    return out_path


def plot_walk(
    tmap: TraversabilityMap,
    plan: PlanResult,
    walk: dict,
    out_path: str | Path,
    *,
    pins=(),
    goal_pin=None,
    summary: dict | None = None,
    theme: str | Theme = "light",
    robot: str = "Go2",
) -> Path:
    """Map with the planned route and the walked path (left); speed and distance-to-goal over time (right).

    ``walk`` holds 1D arrays ``t, x, y`` and optionally ``speed``, ``cmd_vx``, ``dist_goal``, ``cross_track``,
    plus ``falls`` (list of ``(t, x, y)``).
    """
    th = _theme(theme)
    g = tmap.grid
    summary = summary or {}
    fig = plt.figure(figsize=(13.0, 7.0), dpi=150, facecolor=th.surface)
    ax = fig.add_axes([0.05, 0.07, 0.45, 0.76])
    ax_v = fig.add_axes([0.58, 0.50, 0.38, 0.30])
    ax_d = fig.add_axes([0.58, 0.10, 0.38, 0.30])
    t = np.asarray(walk.get("t", []), dtype=float)
    x = np.asarray(walk.get("x", []), dtype=float)
    y = np.asarray(walk.get("y", []), dtype=float)
    walked = np.column_stack([x, y]) if len(x) else np.zeros((0, 2))
    view = _view_bounds(tmap, [plan.path_xy, walked, plan.start_xy[None], plan.goal_xy[None]], 3.0, False)

    ax.imshow(
        _hillshade_rgb(tmap.elevation, th, g.resolution),
        origin="lower",
        extent=g.extent,
        interpolation="nearest",
        zorder=1,
    )
    lethal = np.where(tmap.lethal & ~tmap.unknown, 1.0, np.nan)
    ax.imshow(
        lethal,
        origin="lower",
        extent=g.extent,
        cmap=ListedColormap([th.ink]),
        alpha=0.35,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=2,
    )
    _style_axes(ax, th)
    _draw_footprints(ax, th, tmap)
    _route(ax, th, plan.path_xy, th.series[1], lw=2.0, ls=(0, (4, 2)), zorder=5)
    _route(ax, th, walked, th.series[0], lw=2.2, zorder=6)
    _draw_pins(ax, th, pins, goal_pin)
    ax.plot(*plan.start_xy, marker="o", ms=9, mfc=th.surface, mec=th.ink, mew=1.8, zorder=10)
    for k, (tf, xf, yf) in enumerate(walk.get("falls", []) or []):
        ax.plot(xf, yf, marker="X", ms=11, color=STATUS_CRITICAL, mec=th.surface, mew=1.0, zorder=11)
        ax.annotate(
            f"fall {k + 1} ({tf:.0f} s)",
            (xf, yf),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=7.5,
            color=th.ink,
            zorder=11,
            path_effects=_text_halo(th),
        )
    if len(walked):
        ax.plot(*walked[-1], marker="o", ms=7, color=th.series[0], mec=th.surface, mew=1.2, zorder=10)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _scale_bar(ax, th, view[0] + 0.04 * (view[1] - view[0]), view[2] + 0.04 * (view[3] - view[2]), view[1] - view[0])
    handles = [
        Line2D([], [], color=th.series[1], lw=2.0, ls=(0, (4, 2)), label="planned route"),
        Line2D([], [], color=th.series[0], lw=2.2, label=f"walked ({robot} policy)"),
        Line2D([], [], marker="*", ls="", ms=11, color=th.ink, label="person"),
        Patch(facecolor=th.ink, alpha=0.35, label="no-go"),
    ]
    if walk.get("falls"):
        handles.append(Line2D([], [], marker="X", ls="", ms=9, color=STATUS_CRITICAL, label="fall"))
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.045, 0.915),
        ncol=5,
        frameon=False,
        fontsize=7.8,
        labelcolor=th.ink_secondary,
    )

    # speed over time (one measure per axis; no dual axes)
    for a in (ax_v, ax_d):
        _style_axes(a, th)
        a.grid(True, color=th.grid, lw=0.5)
        a.set_axisbelow(True)
    if len(t):
        if "cmd_vx" in walk:
            ax_v.plot(t, walk["cmd_vx"], color=th.series[1], lw=1.4, label="commanded forward speed")
        if "speed" in walk:
            ax_v.plot(t, walk["speed"], color=th.series[0], lw=1.4, label="actual speed")
        ax_v.set_ylabel("speed [m/s]")
        ax_v.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=2,
            frameon=False,
            fontsize=7.5,
            labelcolor=th.ink_secondary,
        )
        if "dist_goal" in walk:
            ax_d.plot(t, walk["dist_goal"], color=th.ink_secondary, lw=1.6)
            ax_d.set_ylabel("distance to the goal [m]")
        for tf, _, _ in walk.get("falls", []) or []:
            for a in (ax_v, ax_d):
                a.axvline(tf, color=STATUS_CRITICAL, lw=1.0, ls=(0, (3, 2)))
        ax_d.set_xlabel("time [s]")
        ax_v.set_xlim(t[0], t[-1])
        ax_d.set_xlim(t[0], t[-1])
        ax_d.set_ylim(bottom=0)
    ax_v.text(0.0, 1.03, "Speed", transform=ax_v.transAxes, fontsize=9.5, color=th.ink, fontweight="semibold")
    ax_d.text(
        0.0, 1.03, "Distance to the goal", transform=ax_d.transAxes, fontsize=9.5, color=th.ink, fontweight="semibold"
    )

    ok = bool(summary.get("success"))
    title = "Rescue walk: reached the person" if ok else f"Rescue walk: {summary.get('outcome', 'did not arrive')}"
    bits = []
    if summary.get("time_s") is not None:
        bits.append(f"{summary['time_s']:.1f} s")
    if summary.get("walked_length_m") is not None:
        bits.append(f"walked {summary['walked_length_m']:.1f} m (plan {summary.get('planned_length_m', 0):.1f} m)")
    if summary.get("falls") is not None:
        bits.append(f"{summary['falls']} fall(s)")
    if summary.get("final_distance_to_person_m") is not None:
        bits.append(f"stopped {summary['final_distance_to_person_m']:.2f} m from the person")
    if summary.get("cross_track_mean_m") is not None:
        bits.append(f"mean cross-track {summary['cross_track_mean_m']:.2f} m")
    if summary.get("physics"):
        bits.append(f"physics {summary['physics']}")
    fig.text(0.04, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=th.ink)
    fig.text(0.04, 0.938, "  |  ".join(bits), ha="left", va="top", fontsize=8.5, color=th.ink_secondary)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=th.surface)
    plt.close(fig)
    return out_path


def render_minimap(
    tmap: TraversabilityMap,
    plan: PlanResult,
    *,
    pins=(),
    goal_pin=None,
    size_px: int = 300,
    margin_m: float = 3.0,
    theme: str | Theme = "light",
):
    """Square minimap image of the route for the video inset.

    Returns:
        ``(rgb (size_px, size_px, 3) uint8, to_pixel)`` where ``to_pixel(x, y) -> (col, row)`` maps world
        coordinates into the image (row 0 at the top).
    """
    th = _theme(theme)
    g = tmap.grid
    view = _view_bounds(tmap, [plan.path_xy, plan.start_xy[None], plan.goal_xy[None]], margin_m, False)
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi, facecolor=th.surface)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        _hillshade_rgb(tmap.elevation, th, g.resolution), origin="lower", extent=g.extent, interpolation="nearest"
    )
    lethal = np.where(tmap.lethal & ~tmap.unknown, 1.0, np.nan)
    ax.imshow(
        lethal,
        origin="lower",
        extent=g.extent,
        cmap=ListedColormap([th.ink]),
        alpha=0.45,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _route(ax, th, plan.path_xy, th.series[1], lw=2.0)
    _draw_pins(ax, th, pins, goal_pin, labels=False)
    if goal_pin is None:
        ax.plot(*plan.goal_xy, marker="*", ms=12, color=th.ink, mec=th.surface, mew=1.0)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    h, w = rgb.shape[:2]
    x0, x1, y0, y1 = view
    # equal aspect inside a square axes: matplotlib widens the shorter span around its centre
    span = max(x1 - x0, y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def to_pixel(x, y):
        col = (np.asarray(x) - (cx - 0.5 * span)) / span * w
        row = (1.0 - (np.asarray(y) - (cy - 0.5 * span)) / span) * h
        return col, row

    return rgb, to_pixel
