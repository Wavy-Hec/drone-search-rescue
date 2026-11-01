"""Preview the survey area and the planned path on the CPU (no Isaac Sim, no flight).

Generates the terrain exactly as the simulator will (``drone_lab.terrain.SURVEY_TERRAIN_CFG`` is seeded and
deterministic), rasterises its height map, plans the survey path and writes ``area_preview.png`` plus
``ground_truth.npy`` / ``planned_path.csv`` into ``runs/<out>/``.

Usage::

    CUDA_VISIBLE_DEVICES= python scripts/preview_area.py --pattern lawnmower --spacing 4 --altitude 4
    CUDA_VISIBLE_DEVICES= python scripts/preview_area.py --pattern spiral --num_drones 2 --out preview_spiral
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from drone_lab.mapping import GridSpec, MeshHeightQuery, ground_truth_map  # noqa: E402
from drone_lab.paths import PATTERNS, build_env_paths, load_waypoints, polyline_length, resolve_pattern  # noqa: E402
from drone_lab.survey_io import write_planned_path_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", choices=PATTERNS, default=None,
                        help="Survey pattern (default lawnmower; waypoints when --waypoints is given).")
    parser.add_argument("--spacing", type=float, default=4.0)
    parser.add_argument("--altitude", type=float, default=4.0, help="Clearance above the highest terrain point [m].")
    parser.add_argument("--num_drones", type=int, default=1)
    parser.add_argument("--waypoints", type=str, default=None,
                        help="CSV/JSON waypoints file; implies --pattern waypoints.")
    parser.add_argument("--turn", choices=("round", "square"), default="round")
    parser.add_argument("--direction", choices=("x", "y"), default="x")
    parser.add_argument("--resolution", type=float, default=0.25, help="Height map cell size [m].")
    parser.add_argument("--speed", type=float, default=1.0, help="Only used to estimate the flight time.")
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--out", type=str, default=None, help="Run name under runs/ (default preview_<pattern>).")
    args = parser.parse_args(argv)
    try:
        args.pattern = resolve_pattern(args.pattern, bool(args.waypoints))
    except ValueError as exc:
        parser.error(str(exc))

    from drone_lab.plotting import get_theme, plot_area_preview
    from drone_lab.terrain import SURVEY_TERRAIN_CFG, generate_terrain_mesh, terrain_area

    vertices, faces = generate_terrain_mesh(SURVEY_TERRAIN_CFG)
    area = terrain_area(SURVEY_TERRAIN_CFG)
    band = area.inset(-0.5 * args.spacing - 0.5)
    z = vertices[band.contains(vertices[:, :2]), 2]
    flight_z = float(z.max()) + args.altitude
    custom = load_waypoints(args.waypoints, default_z=flight_z) if args.waypoints else None
    paths = build_env_paths(args.pattern, args.num_drones, area, args.spacing, flight_z, waypoints=custom,
                            direction=args.direction, turn=args.turn)
    border = SURVEY_TERRAIN_CFG.border_width
    grid = GridSpec.from_area(area, args.resolution, padding=border)
    gt, _ = ground_truth_map(vertices, faces, grid, supersample=1, query=MeshHeightQuery(vertices, faces))

    out_dir = PROJECT_ROOT / "runs" / (args.out or f"preview_{args.pattern}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ground_truth.npy", gt)
    write_planned_path_csv(out_dir / "planned_path.csv", paths)
    lengths = [polyline_length(p) for p in paths]
    info = {
        "pattern": args.pattern,
        "num_drones": args.num_drones,
        "area": list(area.as_tuple()),
        "terrain_z_min_m": float(z.min()),
        "terrain_z_max_m": float(z.max()),
        "flight_z_m": flight_z,
        "path_length_m": lengths,
        "estimated_flight_time_s": [L / args.speed for L in lengths],
        "grid": grid.to_dict(),
    }
    (out_dir / "preview.json").write_text(json.dumps(info, indent=2))
    subtitle = (
        f"{args.pattern}, {args.num_drones} drone(s) at z = {flight_z:.1f} m (terrain max {z.max():.1f} m + "
        f"{args.altitude:g} m) · path {max(lengths):.0f} m ≈ {max(lengths) / args.speed / 60:.1f} min at "
        f"{args.speed:g} m/s"
    )
    png = plot_area_preview(gt, grid, area, paths, out_dir / "area_preview.png", get_theme(args.theme),
                            subtitle=subtitle)
    print(json.dumps({k: v for k, v in info.items() if k != "grid"}, indent=2))
    print(f"[preview_area] wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
