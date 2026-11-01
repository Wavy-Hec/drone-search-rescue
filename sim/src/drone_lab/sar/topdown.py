"""Top-down object map of a SAR scenario (matplotlib, CPU only).

Shows the terrain (hill-shaded height), the survey area, the landing pad, the default lawnmower path and every
object's footprint coloured by class (trees as dashed canopies), with labels; optionally over a rendered photo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import CLASS_COLORS
from .scenario import Scenario, TerrainSite


def terrain_raster(site: TerrainSite, resolution: float = 0.1):
    """Height raster of the whole site: ``(z (ny, nx), extent)``."""
    x0, x1, y0, y1 = site.bounds
    xs = np.arange(x0 + 0.5 * resolution, x1, resolution)
    ys = np.arange(y0 + 0.5 * resolution, y1, resolution)
    xx, yy = np.meshgrid(xs, ys)
    z = site.heights(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return z, (x0, x1, y0, y1)


def _hillshade(z: np.ndarray, res: float, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(np.nan_to_num(z), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)


def plot_object_map(
    scenario: Scenario,
    site: TerrainSite,
    path: str | Path,
    *,
    photo: np.ndarray | None = None,
    photo_extent=None,
    path_xy: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = 110,
) -> Path:
    """Write the top-down object map to ``path`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    res = 0.1
    z, extent = terrain_raster(site, res)
    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    if photo is not None:
        ax.imshow(photo, extent=photo_extent or extent, origin="upper", interpolation="bilinear")
    else:
        shade = _hillshade(z, res)
        ax.imshow(z, extent=extent, origin="lower", cmap="terrain", alpha=0.9, vmin=-2.5, vmax=4.5)
        ax.imshow(shade, extent=extent, origin="lower", cmap="gray", alpha=0.35)
    x0, x1, y0, y1 = scenario.area
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.5, ls="--", zorder=3))
    if path_xy is not None and len(path_xy):
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="#4cc9f0", lw=0.9, alpha=0.8, zorder=3, label="survey path")
    halo = [pe.withStroke(linewidth=2.5, foreground="black")]
    for o in scenario.objects:
        fp = np.asarray(o.footprint_xy)
        if o.category == "tree":
            r = 0.5 * (o.half_extents[0] + o.half_extents[1])
            ax.add_patch(Circle(o.center[:2], r, fill=False, ec="#9ef01a", lw=1.2, ls="--", zorder=4))
            continue
        color = np.array(CLASS_COLORS.get(o.category, (255, 255, 255))) / 255.0
        if o.category == "landing_pad":
            ax.add_patch(Circle(o.center[:2], o.half_extents[0], fc=(*color, 0.35), ec=color, lw=2, zorder=4))
        else:
            ax.add_patch(Polygon(fp, closed=True, fc=(*color, 0.55), ec=color, lw=1.4, zorder=5))
        if o.category == "person":
            ax.plot(*o.center[:2], marker="x", color="white", ms=5, zorder=6)
            tag = f"{o.name.split('_')[-1]} {'lying' if o.pose.startswith('lying') else 'standing'}"
            ax.annotate(
                tag,
                o.center[:2],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
                color="white",
                path_effects=halo,
                zorder=7,
            )
    handles = []
    from matplotlib.lines import Line2D

    for name, rgb in CLASS_COLORS.items():
        n = sum(1 for o in scenario.objects if o.category == name)
        if n:
            handles.append(
                Line2D([0], [0], marker="s", ls="", ms=9, color=np.array(rgb) / 255.0, label=f"{name} ({n})")
            )
    n_trees = sum(1 for o in scenario.objects if o.category == "tree")
    if n_trees:
        handles.append(
            Line2D([0], [0], marker="o", ls="", ms=9, mfc="none", color="#9ef01a", label=f"tree, unlabeled ({n_trees})")
        )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    n_people = sum(1 for o in scenario.objects if o.category == "person")
    n_lying = sum(1 for o in scenario.objects if o.category == "person" and o.pose.startswith("lying"))
    ax.set_title(
        title
        or f"SAR site, seed {scenario.cfg.seed}: {n_people} people ({n_lying} lying), "
        f"flight z {scenario.flight_z:.2f} m"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
