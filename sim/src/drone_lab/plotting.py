"""Survey plots (matplotlib, Agg backend).

Colour roles (validated reference palette; categorical slots 1-3 pass the CVD and normal-vision checks for all
pairs in both modes):

* elevation (magnitude): one-hue neutral ramp, low = light, high = dark (flipped in dark mode);
* map error (polarity): blue <-> red with a neutral grey midpoint and symmetric limits;
* drones (identity): categorical slots 1..3 in fixed order; drones beyond three are folded into "other";
* text uses ink tokens only; the planned path is a dashed ink line, flown trajectories are solid lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .mapping import GridSpec, map_error_stats  # noqa: E402
from .paths import Area  # noqa: E402
from .survey_io import SurveyRun, load_run  # noqa: E402


@dataclass(frozen=True)
class Theme:
    """Colour tokens of one mode."""

    name: str
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    elevation_ramp: tuple[str, ...]
    diverging: tuple[str, str, str]
    no_data: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    elevation_ramp=("#e1e0d9", "#c3c2b7", "#898781", "#52514e"),
    diverging=("#2a78d6", "#f0efec", "#e34948"),
    no_data="#fcfcfb",
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    elevation_ramp=("#2c2c2a", "#383835", "#898781", "#c3c2b7"),
    diverging=("#3987e5", "#383835", "#e66767"),
    no_data="#1a1a19",
)
THEMES = {"light": LIGHT, "dark": DARK}

PLOT_FILES = ("map_with_trajectory.png", "map_error.png", "map_3d.png", "altitude_profile.png")
"""Figures written by :func:`make_all_plots`."""


def get_theme(name: str = "light") -> Theme:
    if name not in THEMES:
        raise ValueError(f"theme must be one of {sorted(THEMES)}")
    return THEMES[name]


def _elevation_cmap(theme: Theme):
    cmap = LinearSegmentedColormap.from_list(f"elevation_{theme.name}", list(theme.elevation_ramp))
    cmap.set_bad(theme.no_data)
    return cmap


def _diverging_cmap(theme: Theme):
    cmap = LinearSegmentedColormap.from_list(f"diverging_{theme.name}", list(theme.diverging))
    cmap.set_bad(theme.no_data)
    return cmap


def _series_color(theme: Theme, index: int) -> str:
    """Fixed-order categorical slot; drones past the third share the muted 'other' colour."""
    return theme.series[index] if index < len(theme.series) else theme.ink_muted


def _style_axes(ax, theme: Theme, grid: bool = False):
    ax.set_facecolor(theme.surface)
    for spine in ax.spines.values():
        spine.set_color(theme.axis)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=theme.ink_muted, labelcolor=theme.ink_secondary, labelsize=8.5, width=0.6)
    ax.xaxis.label.set_color(theme.ink_secondary)
    ax.yaxis.label.set_color(theme.ink_secondary)
    if grid:
        ax.grid(True, color=theme.grid, linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)


def _new_figure(theme: Theme, size: tuple[float, float]):
    fig = plt.figure(figsize=size, dpi=150, facecolor=theme.surface)
    return fig


def _title(fig, theme: Theme, title: str, subtitle: str | None = None):
    fig.text(0.02, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=theme.ink)
    if subtitle:
        fig.text(0.02, 0.938, subtitle, ha="left", va="top", fontsize=9, color=theme.ink_secondary)


def _halo(theme: Theme, width: float = 2.6):
    """Surface-coloured ring under a line so it stays legible on the map."""
    return [patheffects.withStroke(linewidth=width, foreground=theme.surface)]


def _scale_bar(ax, theme: Theme, x0: float, y0: float, span: float):
    """Scale bar of a round length (about 1/4 of ``span``) with its left end at (x0, y0), in data units."""
    nice = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    length = max([n for n in nice if n <= 0.3 * span] or [nice[0]])
    h = 0.010 * span
    ax.add_patch(Rectangle((x0, y0), length, h, color=theme.ink, lw=0, zorder=6, clip_on=False))
    ax.text(x0 + length + 0.015 * span, y0 + 0.5 * h, f"{length:g} m", ha="left", va="center", fontsize=8,
            color=theme.ink, zorder=6, clip_on=False)


def _plan_and_paths(ax, run: SurveyRun, theme: Theme, *, with_markers: bool = True):
    """Planned paths (dashed ink) and flown trajectories (solid, categorical) on a top-down map."""
    handles = []
    for i, plan in enumerate(run.planned):
        ax.plot(plan[:, 0], plan[:, 1], color=theme.ink, lw=0.8, ls=(0, (4, 3)), alpha=0.9, zorder=3)
    if run.planned:
        handles.append(Line2D([], [], color=theme.ink, lw=0.8, ls=(0, (4, 3)), label="planned path"))
    for k, env_id in enumerate(run.env_ids()):
        tr = run.env_trajectory(env_id)
        if len(tr.get("x", [])) == 0:
            continue
        color = _series_color(theme, k)
        ax.plot(tr["x"], tr["y"], color=color, lw=1.1, solid_capstyle="round", zorder=4, path_effects=_halo(theme))
        if with_markers:
            ax.plot(tr["x"][0], tr["y"][0], "o", ms=6.5, mfc=color, mec=theme.surface, mew=1.0, zorder=7)
            ax.plot(tr["x"][-1], tr["y"][-1], "s", ms=6.0, mfc=color, mec=theme.surface, mew=1.0, zorder=7)
        if k < len(theme.series):
            label = "flown trajectory" if len(run.env_ids()) == 1 else f"drone {env_id}"
            handles.append(Line2D([], [], color=color, lw=1.6, label=label))
        elif k == len(theme.series):
            handles.append(Line2D([], [], color=theme.ink_muted, lw=1.6, label="other drones"))
    if with_markers:
        handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc=theme.ink_secondary, mec=theme.surface,
                              label="start"))
        handles.append(Line2D([], [], ls="", marker="s", ms=5.5, mfc=theme.ink_secondary, mec=theme.surface,
                              label="end"))
    return handles


def _area_outline(ax, area: Area, theme: Theme):
    ax.add_patch(
        Rectangle((area.x_min, area.y_min), area.width, area.height, fill=False, ec=theme.ink_secondary, lw=0.6,
                  zorder=2)
    )


def _legend(ax, handles, theme: Theme, loc="upper right", ncol=1):
    leg = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=8, frameon=True, facecolor=theme.surface,
                    edgecolor=theme.grid, framealpha=0.92, labelcolor=theme.ink_secondary)
    leg.get_frame().set_linewidth(0.6)
    return leg


def _figure_legend(fig, handles, theme: Theme, y: float = 0.895):
    """One legend row under the subtitle, above the plot."""
    ncol = len(handles) if len(handles) <= 6 else -(-len(handles) // 2)
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.085, y), ncol=ncol,
                     fontsize=8, frameon=False, labelcolor=theme.ink_secondary, handlelength=2.2, columnspacing=1.4)
    return leg


def _colorbar(fig, im, ax, theme: Theme, label: str):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, color=theme.ink_secondary, fontsize=9)
    cb.ax.tick_params(colors=theme.ink_muted, labelcolor=theme.ink_secondary, labelsize=8, width=0.6)
    cb.outline.set_edgecolor(theme.axis)
    cb.outline.set_linewidth(0.6)
    return cb


def _elevation_limits(*maps) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(m, dtype=np.float64)[np.isfinite(m)] for m in maps if m is not None])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
    return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)


"""
Figures.
"""


def plot_map_with_trajectory(run: SurveyRun, out_path: str | Path, theme: Theme = LIGHT) -> Path:
    """Top-down elevation map built from the ray hits, with planned and flown paths."""
    s = run.summary or {}
    m = s.get("map", {})
    err = (m.get("error_area") or {}).get("rmse")
    fig = _new_figure(theme, (8.2, 8.2))
    ax = fig.add_axes([0.09, 0.07, 0.74, 0.76])
    _style_axes(ax, theme)
    vmin, vmax = _elevation_limits(run.elevation, run.ground_truth)
    im = ax.imshow(run.elevation, origin="lower", extent=run.grid.extent, cmap=_elevation_cmap(theme), vmin=vmin,
                   vmax=vmax, interpolation="nearest", zorder=1)
    _area_outline(ax, run.area, theme)
    handles = _plan_and_paths(ax, run, theme)
    handles.append(Patch(facecolor=theme.no_data, edgecolor=theme.axis, lw=0.6, label="no ray hits"))
    _figure_legend(fig, handles, theme)
    pad = 0.5 * float(s.get("lawnmower_spacing_m") or 0.0) + 1.0
    x_lo, x_hi = min(run.grid.x_min, run.area.x_min - pad), max(run.grid.x_max, run.area.x_max + pad)
    y_lo, y_hi = min(run.grid.y_min, run.area.y_min - pad), max(run.grid.y_max, run.area.y_max + pad)
    y_lo -= 0.06 * (y_hi - y_lo)  # room for the scale bar below the area
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    _scale_bar(ax, theme, run.area.x_min, y_lo + 0.025 * (y_hi - y_lo), x_hi - x_lo)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cax = fig.add_axes([0.85, 0.07, 0.025, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Elevation [m]", color=theme.ink_secondary, fontsize=9)
    cb.ax.tick_params(colors=theme.ink_muted, labelcolor=theme.ink_secondary, labelsize=8, width=0.6)
    cb.outline.set_edgecolor(theme.axis)
    cov = m.get("coverage_area")
    parts = []
    if cov is not None:
        parts.append(f"coverage {100 * cov:.1f} %")
    if err is not None:
        parts.append(f"RMSE {err:.3f} m vs ground truth")
    fl = s.get("flight", {})
    if fl.get("path_length_flown_m"):
        parts.append(f"flown {fl['path_length_flown_m']:.0f} m in {fl.get('duration_s', 0):.0f} s")
    if s.get("pattern"):
        parts.insert(0, f"{s['pattern']} at {s.get('flight_z_m', float('nan')):.1f} m")
    _title(fig, theme, "Survey elevation map and flight path", " · ".join(parts) if parts else None)
    out = Path(out_path)
    fig.savefig(out, facecolor=theme.surface)
    plt.close(fig)
    return out


def plot_map_error(run: SurveyRun, out_path: str | Path, theme: Theme = LIGHT) -> Path:
    """Map minus ground truth (diverging), plus the error histogram."""
    diff = np.asarray(run.elevation, dtype=np.float64) - np.asarray(run.ground_truth, dtype=np.float64)
    in_area = run.grid.area_mask(run.area)
    stats = map_error_stats(run.elevation, run.ground_truth, in_area)
    finite = diff[np.isfinite(diff) & in_area]
    lim = float(np.percentile(np.abs(finite), 99)) if finite.size else 0.1
    lim = max(lim, 0.05)
    fig = _new_figure(theme, (11.0, 5.6))
    ax = fig.add_axes([0.06, 0.11, 0.48, 0.74])
    _style_axes(ax, theme)
    im = ax.imshow(np.where(in_area, diff, np.nan), origin="lower", extent=run.grid.extent,
                   cmap=_diverging_cmap(theme), vmin=-lim, vmax=lim, interpolation="nearest")
    _area_outline(ax, run.area, theme)
    ax.set_aspect("equal")
    ax.set_xlim(run.area.x_min - 1, run.area.x_max + 1)
    ax.set_ylim(run.area.y_min - 1, run.area.y_max + 1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cb = _colorbar(fig, im, ax, theme, "Map − ground truth [m]")
    cb.ax.yaxis.set_label_position("right")
    _legend(ax, [Patch(facecolor=theme.no_data, edgecolor=theme.axis, lw=0.6, label="no ray hits")], theme,
            loc="upper right")

    hx = fig.add_axes([0.66, 0.11, 0.31, 0.74])
    _style_axes(hx, theme, grid=True)
    if finite.size:
        bins = np.linspace(-lim, lim, 41)
        hx.hist(np.clip(finite, -lim, lim), bins=bins, color=theme.series[0], rwidth=0.82, zorder=3)
        hx.axvline(0.0, color=theme.axis, lw=0.8, zorder=2)
    hx.set_xlabel(f"Map − ground truth [m] (end bins hold |e| > {lim:.2f} m)")
    hx.set_ylabel("Cells")
    txt = (
        f"RMSE {stats['rmse']:.3f} m\nMAE {stats['mae']:.3f} m\np95 |e| {stats['p95_abs']:.3f} m\n"
        f"max |e| {stats['max_abs']:.2f} m\nbias {stats['bias']:+.3f} m"
        if stats["n_cells"]
        else "no covered cells"
    )
    hx.text(0.97, 0.96, txt, transform=hx.transAxes, ha="right", va="top", fontsize=8.5, color=theme.ink,
            bbox=dict(facecolor=theme.surface, edgecolor=theme.grid, lw=0.6, boxstyle="round,pad=0.4"))
    smooth = (run.summary.get("map", {}) or {}).get("error_area_smooth") or {}
    sub = f"red = map too high, blue = too low · {stats['n_cells']:,} covered {run.grid.resolution:g} m cells"
    if smooth.get("rmse") is not None:
        sub += f" · RMSE away from steps/walls {smooth['rmse']:.3f} m"
    _title(fig, theme, "Elevation map error", sub)
    out = Path(out_path)
    fig.savefig(out, facecolor=theme.surface)
    plt.close(fig)
    return out


def plot_map_3d(run: SurveyRun, out_path: str | Path, theme: Theme = LIGHT, max_cells: int = 140) -> Path:
    """Shaded 3D terrain (ground truth where available, else the map) with the flown trajectory."""
    surf = np.asarray(run.ground_truth, dtype=np.float64)
    if not np.any(np.isfinite(surf)):
        surf = np.asarray(run.elevation, dtype=np.float64)
    surf = np.where(np.isfinite(surf), surf, np.nanmin(surf) if np.any(np.isfinite(surf)) else 0.0)
    xs, ys = run.grid.cell_centers()
    step = max(1, int(np.ceil(max(surf.shape) / max_cells)))
    Z = surf[::step, ::step]
    X, Y = np.meshgrid(xs[::step], ys[::step])
    fig = _new_figure(theme, (9.0, 7.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.90], projection="3d")
    ax.set_facecolor(theme.surface)
    # the drone always flies above the terrain: draw paths after the surface instead of depth-sorting them
    ax.computed_zorder = False
    vmin, vmax = _elevation_limits(Z)
    ls = LightSource(azdeg=315, altdeg=40)
    norm = (np.clip(Z, vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
    rgb = ls.shade_rgb(_elevation_cmap(theme)(norm)[..., :3], Z, vert_exag=1.0, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False,
                    zorder=1)
    handles = []
    for plan in run.planned:
        ax.plot(plan[:, 0], plan[:, 1], plan[:, 2], color=theme.ink, lw=0.7, ls=(0, (4, 3)), alpha=0.8, zorder=3)
    if run.planned:
        handles.append(Line2D([], [], color=theme.ink, lw=0.8, ls=(0, (4, 3)), label="planned path"))
    for k, env_id in enumerate(run.env_ids()):
        tr = run.env_trajectory(env_id)
        if len(tr.get("x", [])) == 0:
            continue
        color = _series_color(theme, k)
        ax.plot(tr["x"], tr["y"], tr["z"], color=color, lw=1.3, zorder=4)
        ax.scatter([tr["x"][0]], [tr["y"][0]], [tr["z"][0]], color=color, s=28, marker="o",
                   edgecolors=theme.surface, linewidths=1.0, depthshade=False, zorder=5)
        ax.scatter([tr["x"][-1]], [tr["y"][-1]], [tr["z"][-1]], color=color, s=26, marker="s",
                   edgecolors=theme.surface, linewidths=1.0, depthshade=False, zorder=5)
        if k < len(theme.series):
            label = "flown trajectory" if len(run.env_ids()) == 1 else f"drone {env_id}"
            handles.append(Line2D([], [], color=color, lw=1.6, label=label))
    zmax = max(float(np.nanmax(Z)), float(np.nanmax(run.trajectory["z"])) if len(run.trajectory.get("z", [])) else 0)
    zlo, zhi = min(float(np.nanmin(Z)), 0.0), zmax + 1.0
    ax.set_zlim(zlo, zhi)
    x_span = max(float(X.max() - X.min()), 1e-6)
    z_box = 0.32
    exaggeration = z_box / ((zhi - zlo) / x_span)
    ax.set_box_aspect((1.0, float(Y.max() - Y.min()) / x_span, z_box), zoom=1.15)
    ax.view_init(elev=32, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(theme.surface, 1.0))
        axis._axinfo["grid"]["color"] = matplotlib.colors.to_rgba(theme.grid, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.line.set_color(theme.axis)
        axis.label.set_color(theme.ink_secondary)
    ax.tick_params(colors=theme.ink_muted, labelcolor=theme.ink_secondary, labelsize=7.5)
    ax.set_xlabel("x [m]", fontsize=8.5)
    ax.set_ylabel("y [m]", fontsize=8.5)
    ax.set_zlabel("z [m]", fontsize=8.5)
    if handles:
        _legend(ax, handles, theme, loc="upper right")
    sub = "terrain = ground truth" if np.any(np.isfinite(run.ground_truth)) else "terrain = elevation map"
    _title(fig, theme, "Survey flight over the terrain (3D)", sub + f" · vertical exaggeration ×{exaggeration:.1f}")
    out = Path(out_path)
    fig.savefig(out, facecolor=theme.surface)
    plt.close(fig)
    return out


def plot_altitude_profile(run: SurveyRun, out_path: str | Path, theme: Theme = LIGHT) -> Path:
    """Four panels over time: height profile (drone and terrain), height above ground, speed, carrot distance."""
    s = run.summary or {}
    fig = _new_figure(theme, (10.0, 9.4))
    n_panels = 4
    top, bottom, gap = 0.865, 0.065, 0.035
    h = (top - bottom - gap * (n_panels - 1)) / n_panels
    axes = [fig.add_axes([0.085, top - (i + 1) * h - i * gap, 0.885, h]) for i in range(n_panels)]
    for ax in axes:
        _style_axes(ax, theme, grid=True)
    handles = []
    agl_min, speed_max, carrot_max = [], [0.0], [0.0]
    for k, env_id in enumerate(run.env_ids()):
        tr = run.env_trajectory(env_id)
        if len(tr.get("time", [])) == 0:
            continue
        color = _series_color(theme, k)
        t = tr["time"]
        terrain = tr["z"] - tr["agl"]
        axes[0].fill_between(t, np.nanmin(terrain) - 0.5, terrain, color=theme.ink_muted, alpha=0.18, lw=0, zorder=2)
        axes[0].plot(t, terrain, color=theme.ink_muted, lw=0.8, zorder=3)
        axes[0].plot(t, tr["z"], color=color, lw=1.0, zorder=4)
        axes[1].plot(t, tr["agl"], color=color, lw=1.0, zorder=4)
        axes[2].plot(t, tr["speed"], color=color, lw=1.0, zorder=4)
        axes[3].plot(t, tr["carrot_dist"], color=color, lw=1.0, zorder=4)
        if np.any(np.isfinite(tr["agl"])):
            agl_min.append(float(np.nanmin(tr["agl"])))
        speed_max.append(float(np.nanmax(tr["speed"])))
        carrot_max.append(float(np.nanmax(tr["carrot_dist"])))
        if k < len(theme.series):
            label = "drone" if len(run.env_ids()) == 1 else f"drone {env_id}"
            handles.append(Line2D([], [], color=color, lw=1.6, label=label))
        elif k == len(theme.series):
            handles.append(Line2D([], [], color=theme.ink_muted, lw=1.6, label="other drones"))
    handles.append(Patch(facecolor=theme.ink_muted, alpha=0.35, edgecolor=theme.ink_muted, lw=0.6,
                         label="terrain below the drone"))
    _figure_legend(fig, handles, theme, y=0.893)

    def reference(ax, value, text, top_value):
        ax.axhline(value, color=theme.ink_secondary, lw=0.8, ls=(0, (4, 3)), zorder=3)
        ax.text(0.995, value, f"{text} ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8, color=theme.ink_secondary)
        ax.set_ylim(0, max(top_value, value) * 1.25)

    v = s.get("commanded_speed_mps")
    if v:
        reference(axes[2], v, f"commanded {v:g} m/s", max(speed_max))
    else:
        axes[2].set_ylim(bottom=0)
    lead = s.get("max_lead_m")
    if lead:
        reference(axes[3], lead, f"carrot max lead {lead:g} m", max(carrot_max))
    else:
        axes[3].set_ylim(bottom=0)
    if agl_min:
        axes[1].set_ylim(0, None)
    axes[0].set_ylabel("Height z [m]")
    axes[1].set_ylabel("Above ground [m]")
    axes[2].set_ylabel("Speed [m/s]")
    axes[3].set_ylabel("To carrot [m]")
    axes[-1].set_xlabel("Time [s]")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    xmax = max((float(np.nanmax(run.env_trajectory(e)["time"])) for e in run.env_ids()), default=1.0)
    for ax in axes:
        ax.set_xlim(0, xmax)
    sub = []
    if agl_min:
        sub.append(f"lowest clearance above ground {min(agl_min):.2f} m")
    tr_s = s.get("tracking", {})
    if tr_s.get("cross_track_rms_m") is not None:
        sub.append(f"cross-track RMS {tr_s['cross_track_rms_m']:.2f} m")
    _title(fig, theme, "Altitude, speed and tracking over time", " · ".join(sub) if sub else None)
    out = Path(out_path)
    fig.savefig(out, facecolor=theme.surface)
    plt.close(fig)
    return out


def plot_area_preview(
    ground_truth: np.ndarray,
    grid: GridSpec,
    area: Area,
    planned: list[np.ndarray],
    out_path: str | Path,
    theme: Theme = LIGHT,
    title: str = "Survey area and planned path",
    subtitle: str | None = None,
) -> Path:
    """Terrain height map with the planned path (no flight data yet)."""
    fig = _new_figure(theme, (8.2, 8.2))
    ax = fig.add_axes([0.09, 0.07, 0.74, 0.76])
    _style_axes(ax, theme)
    vmin, vmax = _elevation_limits(ground_truth)
    im = ax.imshow(ground_truth, origin="lower", extent=grid.extent, cmap=_elevation_cmap(theme), vmin=vmin,
                   vmax=vmax, interpolation="nearest")
    _area_outline(ax, area, theme)
    handles = []
    for k, plan in enumerate(planned):
        color = _series_color(theme, k)
        ax.plot(plan[:, 0], plan[:, 1], color=color, lw=1.0, ls=(0, (4, 3)), zorder=3, path_effects=_halo(theme))
        ax.plot(plan[0, 0], plan[0, 1], "o", ms=6.5, mfc=color, mec=theme.surface, mew=1.0, zorder=5)
        if k < len(theme.series):
            handles.append(Line2D([], [], color=color, lw=1.2, ls=(0, (4, 3)),
                                  label="planned path" if len(planned) == 1 else f"drone {k} plan"))
        elif k == len(theme.series):
            handles.append(Line2D([], [], color=theme.ink_muted, lw=1.2, ls=(0, (4, 3)), label="other drones"))
    handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc=theme.ink_secondary, mec=theme.surface,
                          label="start"))
    handles.append(Line2D([], [], color=theme.ink_secondary, lw=0.6, label="survey area"))
    _figure_legend(fig, handles, theme)
    y_lo, y_hi = grid.y_min, grid.y_max
    y_lo -= 0.06 * (y_hi - y_lo)
    ax.set_aspect("equal")
    ax.set_xlim(grid.x_min, grid.x_max)
    ax.set_ylim(y_lo, y_hi)
    _scale_bar(ax, theme, grid.x_min + 0.02 * (grid.x_max - grid.x_min), y_lo + 0.025 * (y_hi - y_lo),
               grid.x_max - grid.x_min)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cax = fig.add_axes([0.85, 0.07, 0.025, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Terrain height [m]", color=theme.ink_secondary, fontsize=9)
    cb.ax.tick_params(colors=theme.ink_muted, labelcolor=theme.ink_secondary, labelsize=8, width=0.6)
    cb.outline.set_edgecolor(theme.axis)
    _title(fig, theme, title, subtitle)
    out = Path(out_path)
    fig.savefig(out, facecolor=theme.surface)
    plt.close(fig)
    return out


def make_all_plots(run_dir: str | Path, theme: str = "light") -> list[Path]:
    """Write the four survey figures into ``run_dir``; returns their paths."""
    run = load_run(run_dir)
    th = get_theme(theme)
    suffix = "" if theme == "light" else f"_{theme}"
    out = []
    for name, fn in zip(PLOT_FILES, (plot_map_with_trajectory, plot_map_error, plot_map_3d, plot_altitude_profile)):
        stem, ext = name.rsplit(".", 1)
        out.append(fn(run, run.run_dir / f"{stem}{suffix}.{ext}", th))
    return out


def plot_trajectory_tracking(
    t: np.ndarray,
    flown: np.ndarray,
    reference: np.ndarray,
    out_path: str | Path,
    theme: Theme = LIGHT,
    title: str = "Trajectory tracking",
    subtitle: str | None = None,
) -> Path:
    """Reference (dashed ink) vs flown (solid) trajectory in 3D and top view, plus the error over time."""
    t = np.asarray(t, dtype=np.float64)
    flown = np.asarray(flown, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    err = np.linalg.norm(flown - reference, axis=1)
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    color = theme.series[0]
    fig = _new_figure(theme, (12.0, 6.6))
    ax3 = fig.add_axes([0.0, 0.03, 0.52, 0.82], projection="3d")
    ax3.set_facecolor(theme.surface)
    ax3.computed_zorder = False
    ax3.plot(reference[:, 0], reference[:, 1], reference[:, 2], color=theme.ink, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax3.plot(flown[:, 0], flown[:, 1], flown[:, 2], color=color, lw=1.2, zorder=3)
    ax3.scatter([flown[0, 0]], [flown[0, 1]], [flown[0, 2]], color=color, s=26, marker="o",
                edgecolors=theme.surface, linewidths=1.0, depthshade=False, zorder=4)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(theme.surface, 1.0))
        axis._axinfo["grid"]["color"] = matplotlib.colors.to_rgba(theme.grid, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.line.set_color(theme.axis)
        axis.label.set_color(theme.ink_secondary)
    ax3.tick_params(colors=theme.ink_muted, labelcolor=theme.ink_secondary, labelsize=7.5)
    ax3.set_xlabel("x [m]", fontsize=8.5)
    ax3.set_ylabel("y [m]", fontsize=8.5)
    ax3.set_zlabel("z [m]", fontsize=8.5)
    span = np.ptp(np.vstack([flown, reference]), axis=0)
    ax3.set_box_aspect(tuple(np.maximum(span, 0.25 * span.max())), zoom=0.95)
    ax3.view_init(elev=28, azim=-55)

    axt = fig.add_axes([0.60, 0.50, 0.37, 0.35])
    _style_axes(axt, theme, grid=True)
    axt.plot(reference[:, 0], reference[:, 1], color=theme.ink, lw=0.8, ls=(0, (4, 3)), zorder=2)
    axt.plot(flown[:, 0], flown[:, 1], color=color, lw=1.0, zorder=3)
    axt.set_aspect("equal", adjustable="datalim")
    axt.set_xlabel("x [m]")
    axt.set_ylabel("y [m]")

    axe = fig.add_axes([0.60, 0.09, 0.37, 0.30])
    _style_axes(axe, theme, grid=True)
    axe.plot(t, err, color=color, lw=1.0, zorder=3)
    axe.axhline(rmse, color=theme.ink_secondary, lw=0.8, ls=(0, (4, 3)), zorder=2)
    axe.text(0.995, rmse, f"RMSE {rmse:.3f} m ", transform=axe.get_yaxis_transform(), ha="right", va="bottom",
             fontsize=8, color=theme.ink_secondary)
    axe.set_ylim(0, max(float(err.max()) if len(err) else 1.0, rmse) * 1.2)
    axe.set_xlim(t.min(), t.max())
    axe.set_xlabel("Time [s]")
    axe.set_ylabel("Position error [m]")
    handles = [
        Line2D([], [], color=theme.ink, lw=0.8, ls=(0, (4, 3)), label="reference"),
        Line2D([], [], color=color, lw=1.6, label="flown"),
        Line2D([], [], ls="", marker="o", ms=6, mfc=color, mec=theme.surface, label="start"),
    ]
    _figure_legend(fig, handles, theme, y=0.885)
    _title(fig, theme, title, subtitle or f"position RMSE {rmse:.3f} m, max {float(err.max()):.3f} m")
    out = Path(out_path)
    fig.savefig(out, facecolor=theme.surface)
    plt.close(fig)
    return out
