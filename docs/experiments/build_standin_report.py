#!/usr/bin/env python3
"""A stand-in ground-station report of the demo site (SAR site seed 0), made on the CPU without flying.

The rescue planner reads a ground-station report: ``maps/elevation_mean.npy`` (the mapping rays' terrain heights),
``maps/surface_max.npy`` (the highest camera depth point per cell: the terrain and what stands on it),
``maps/grid.json`` and the confirmed pins in ``detections.json``. This script builds such a report from the site's
ground truth, as a perfect drone would see it, to try the surface-map obstacles of section 10 of HOW_IT_WORKS.md:

* the grid is the station's map of the 40 x 40 m area with 6 m of padding (209 x 209 cells of 0.25 m);
* elevation: the true terrain height at each cell centre inside the survey area (the mapping rays tile it), NaN
  outside;
* surface: the highest terrain point of 4 x 4 samples in each cell, raised to the top of every object (people,
  props, trees) standing in it, inside the area the camera saw: the survey area plus assumed margins of 3.7 m in x
  and 2.4 m in y (the camera sees a few metres past the outer survey lines);
* pins: every labelled object the camera saw, except one crate left out on purpose (a detector miss); trees are never
  pinned, as the detector has no tree class.

It also writes ``trees.json`` (the trees and the missed crate, for checking what the surface map adds).

    python docs/experiments/build_standin_report.py --out gcs/runs/exp_standin_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim" / "src"))

from drone_lab.mapping import GridSpec, MeshHeightQuery
from drone_lab.sar.scene import SarSiteCfg, scenario_for
from drone_lab.teaming.traversability import points_in_polygon
from drone_lab.terrain import SURVEY_TERRAIN_CFG, generate_terrain_mesh

AREA_HALF = 20.0  # the survey area is 40 x 40 m, centred on the origin
SEEN_HALF_X, SEEN_HALF_Y = (
    AREA_HALF + 3.7,
    AREA_HALF + 2.4,
)  # what the camera saw (assumed margins)


def cell_of(grid: GridSpec, x: float, y: float) -> tuple[int, int]:
    """(row, column) of the cell holding the point (x, y)."""
    ix, iy, _ = grid.cell_index(np.array([x]), np.array([y]))
    return int(iy[0]), int(ix[0])


def build(out: Path) -> str:
    gt = scenario_for(SarSiteCfg(seed=0)).ground_truth()
    height = MeshHeightQuery(*generate_terrain_mesh(SURVEY_TERRAIN_CFG))
    grid = GridSpec(-26.125, -26.125, 0.25, 209, 209)
    xs, ys = grid.cell_centers()
    X, Y = np.meshgrid(xs, ys)

    centre = height(np.column_stack([X.ravel(), Y.ravel()])).reshape(grid.shape)
    scanned = (np.abs(X) <= AREA_HALF) & (np.abs(Y) <= AREA_HALF)
    elevation = np.where(scanned, centre, np.nan).astype(np.float32)

    seen = (np.abs(X) <= SEEN_HALF_X) & (np.abs(Y) <= SEEN_HALF_Y)
    surface = np.full(grid.shape, -np.inf)
    k, res = 4, grid.resolution
    offsets = (
        np.arange(k) + 0.5
    ) / k * res - 0.5 * res  # 4 x 4 sample points inside each cell
    for dy in offsets:
        for dx in offsets:
            surface = np.fmax(
                surface,
                height(np.column_stack([(X + dx).ravel(), (Y + dy).ravel()])).reshape(
                    grid.shape
                ),
            )

    missed = next(
        o["name"] for o in gt["objects"] if o["category"] == "crate" and o["in_area"]
    )
    pins = []
    for o in gt["objects"]:
        inside = points_in_polygon(X, Y, o["footprint_xy"])
        if not inside.any():  # smaller than a cell: the cell it stands in
            inside[cell_of(grid, o["x"], o["y"])] = True
        surface[inside] = np.maximum(surface[inside], o["aabb_max"][2])
        if (
            o["class"] is not None
            and o["name"] != missed
            and seen[cell_of(grid, o["x"], o["y"])]
        ):
            pin = {
                "id": len(pins) + 1,
                "class": o["class"],
                "x": o["x"],
                "y": o["y"],
                "z": o["aabb_max"][2],
            }
            pins.append({**pin, "confirmed": True, "truth": o["name"]})
    surface = np.where(seen, surface, np.nan).astype(np.float32)

    (out / "maps").mkdir(parents=True, exist_ok=True)
    np.save(out / "maps" / "elevation_mean.npy", elevation)
    np.save(out / "maps" / "surface_max.npy", surface)
    area = [-AREA_HALF, AREA_HALF, -AREA_HALF, AREA_HALF]
    (out / "maps" / "grid.json").write_text(
        json.dumps({"elevation_grid": grid.to_dict(), "area": area})
    )
    (out / "detections.json").write_text(json.dumps({"objects": pins}))
    trees = [o for o in gt["objects"] if o["category"] == "tree"]
    crate = next(o for o in gt["objects"] if o["name"] == missed)
    things = [
        {"name": o["name"], "footprint_xy": o["footprint_xy"], "in_area": o["in_area"]}
        for o in trees
    ]
    things.append(
        {"name": missed, "footprint_xy": crate["footprint_xy"], "in_area": True}
    )
    (out / "trees.json").write_text(json.dumps(things))
    n_in = sum(o["in_area"] for o in trees)
    return f"report in {out}: {len(pins)} pins, missed crate {missed}, {len(trees)} trees ({n_in} in the area)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="report folder to write (maps/, detections.json)",
    )
    a = p.parse_args(argv)
    print(build(Path(a.out).expanduser()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
