"""Synthetic (CPU-only) survey over the real survey terrain, without Isaac Sim.

Generates the survey area with Isaac Lab's terrain generator on the CPU (``drone_lab.terrain``, deterministic and
identical to what the simulator imports), plans the same path as the Isaac Sim task, flies it with a kinematic
stand-in for the RL policy (``drone_lab.synthetic``), maps it with the same 5 m x 5 m ray grid, and writes a run
directory in the same format as ``scripts/survey_map.py`` (then plots it).

Use it to preview a pattern / spacing / altitude before spending GPU time, and as an end-to-end test of the
mapping pipeline. The flight dynamics are NOT the policy's: tracking numbers here are only indicative.

Usage (env_isaaclab, CPU only)::

    CUDA_VISIBLE_DEVICES= python scripts/synthetic_survey.py --pattern lawnmower --spacing 4 --altitude 4 \
        --speed 1.0 --out synthetic_lawnmower
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from drone_lab.paths import PATTERNS, build_env_paths, load_waypoints, resolve_pattern  # noqa: E402
from drone_lab.survey_io import format_summary, save_run  # noqa: E402


def resolve_run_dir(out: str | None, prefix: str) -> Path:
    """``--out`` as a name under ``runs/`` or as an explicit path."""
    if out and ("/" in out or out.startswith("~")):
        return Path(out).expanduser().resolve()
    name = out or f"{datetime.now():%Y%m%d_%H%M%S}_{prefix}"
    return PROJECT_ROOT / "runs" / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", choices=PATTERNS, default=None,
                        help="Survey pattern (default lawnmower; waypoints when --waypoints is given).")
    parser.add_argument("--spacing", type=float, default=4.0, help="Line/ring spacing [m].")
    parser.add_argument("--altitude", type=float, default=4.0, help="Clearance above the highest terrain point [m].")
    parser.add_argument("--speed", type=float, default=1.0, help="Carrot speed [m/s].")
    parser.add_argument("--num_drones", type=int, default=1)
    parser.add_argument("--waypoints", type=str, default=None,
                        help="CSV/JSON waypoints file; implies --pattern waypoints.")
    parser.add_argument("--turn", choices=("round", "square"), default="round")
    parser.add_argument("--direction", choices=("x", "y"), default="x")
    parser.add_argument("--map_resolution", type=float, default=0.25)
    parser.add_argument("--reducer", choices=("median", "mean", "max", "min"), default="median")
    parser.add_argument("--noise", type=float, default=0.0, help="Position noise std of the kinematic drone [m].")
    parser.add_argument("--max_time", type=float, default=None, help="Stop after this many seconds.")
    parser.add_argument("--out", type=str, default=None, help="Run name under runs/ or a path.")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.pattern = resolve_pattern(args.pattern, bool(args.waypoints))
    except ValueError as exc:
        parser.error(str(exc))

    from drone_lab.synthetic import simulate_survey
    from drone_lab.terrain import SURVEY_TERRAIN_CFG, generate_terrain_mesh, terrain_area

    t0 = time.time()
    vertices, faces = generate_terrain_mesh(SURVEY_TERRAIN_CFG)
    area = terrain_area(SURVEY_TERRAIN_CFG)
    band = area.inset(-0.5 * args.spacing - 0.5)
    z = vertices[band.contains(vertices[:, :2]), 2]
    z_min, z_max = float(z.min()), float(z.max())
    flight_z = z_max + args.altitude
    custom = load_waypoints(args.waypoints, default_z=flight_z) if args.waypoints else None
    waypoints = build_env_paths(
        args.pattern,
        args.num_drones,
        area,
        args.spacing,
        flight_z,
        waypoints=custom,
        direction=args.direction,
        turn=args.turn,
    )
    print(f"[synthetic] terrain {len(faces):,} triangles, z {z_min:.2f}..{z_max:.2f} m, flight z {flight_z:.2f} m")
    extra = {
        "source": "synthetic (kinematic stand-in for the policy, CPU)",
        "pattern": args.pattern,
        "lawnmower_spacing_m": args.spacing,
        "altitude_clearance_m": args.altitude,
        "terrain_z_min_m": z_min,
        "terrain_z_max_m": z_max,
        "flight_z_m": flight_z,
        "max_lead_m": 1.0,
        "map_reducer": args.reducer,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    result = simulate_survey(
        vertices,
        faces,
        waypoints,
        area=area,
        speed=args.speed,
        map_resolution=args.map_resolution,
        reducer=args.reducer,
        position_noise=args.noise,
        max_time=args.max_time,
        extra_summary=extra,
    )
    run_dir = resolve_run_dir(args.out, f"synthetic_{args.pattern}")
    meta = {
        "source": "synthetic",
        "map_reducer": args.reducer,
        "footprint_m": [5.0, 5.0],
        "ray_resolution_m": 0.25,
        "terrain": "drone_lab.terrain.SURVEY_TERRAIN_CFG",
    }
    save_run(
        run_dir,
        trajectory=result["trajectory"],
        planned=result["planned"],
        grid=result["grid"],
        area=result["area"],
        elevation=result["elevation"],
        count=result["count"],
        ground_truth=result["ground_truth"],
        ground_truth_range=result["ground_truth_range"],
        hits=result["hits"],
        meta=meta,
        summary=result["summary"],
    )
    print(f"[synthetic] {result['num_hits']:,} ray hits, {len(result['trajectory']['x']):,} samples, "
          f"{time.time() - t0:.1f} s wall time")
    print(format_summary(json.loads((run_dir / "summary.json").read_text())))
    if not args.no_plots:
        from drone_lab.plotting import make_all_plots

        for p in make_all_plots(run_dir):
            print(f"[synthetic] wrote {p}")
    print(f"[synthetic] run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    sys.exit(main())
