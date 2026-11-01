"""Run files and plots on a small synthetic run (numpy + matplotlib only, no torch / Isaac Lab)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from _runner import main_for, require

from drone_lab.mapping import ElevationMapper, GridSpec, ground_truth_map, heightfield_to_mesh, MeshHeightQuery
from drone_lab.paths import Area, lawnmower, resample_polyline
from drone_lab.survey_io import TrajectoryRecorder, compute_summary, load_run, save_run

AREA = Area(-6.0, 6.0, -6.0, 6.0)


def _make_run(run_dir: Path) -> None:
    """A tiny but complete run: analytic terrain, lawnmower flight, grid rays, map, ground truth, summary."""
    xs = np.arange(-9.0, 9.0001, 0.1)
    xx, yy = np.meshgrid(xs, xs)
    zz = 0.8 * np.sin(xx / 2.0) + 1.5 * ((np.abs(xx - 2) < 1.5) & (np.abs(yy + 2) < 1.0))
    v, f = heightfield_to_mesh(xs, xs, zz)
    query = MeshHeightQuery(v, f)
    plan = lawnmower(AREA, 3.0, 4.0)
    path, _ = resample_polyline(plan, 0.15)
    rec = TrajectoryRecorder()
    grid = GridSpec.from_area(AREA, 0.25)
    mapper = ElevationMapper(grid)
    o = np.arange(-2.0, 2.0001, 0.25)
    off = np.column_stack([a.ravel() for a in np.meshgrid(o, o)])
    for k, p in enumerate(path):
        rays = p[:2] + off
        mapper.add(np.column_stack([rays, query(rays)]))
        rec.record(step=[k], time=[0.1 * (k + 1)], env=[0], x=[p[0]], y=[p[1] + 0.05], z=[p[2]], speed=[1.5],
                   carrot_dist=[0.6], progress=[k / (len(path) - 1)], obs_dx=[0.1], obs_dy=[0.0], obs_dz=[0.0],
                   agl=[p[2] - query(p[None, :2])[0]])
    elev, count = mapper.finalize("median")
    gt, gt_range = ground_truth_map(v, f, grid, query=query)
    traj = rec.as_dict()
    summary = compute_summary(traj, [plan], grid=grid, area=AREA, elevation=elev, count=count, ground_truth=gt,
                              ground_truth_range=gt_range, commanded_speed=1.5,
                              extra={"pattern": "lawnmower", "flight_z_m": 4.0, "max_lead_m": 1.0})
    save_run(run_dir, trajectory=traj, planned=[plan], grid=grid, area=AREA, elevation=elev, count=count,
             ground_truth=gt, ground_truth_range=gt_range, hits=np.zeros((0, 5)), meta={"source": "test"},
             summary=summary)


def test_run_roundtrip_and_summary():
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        _make_run(run_dir)
        run = load_run(run_dir)
        assert run.grid.shape == run.elevation.shape == run.ground_truth.shape
        assert run.env_ids() == [0] and len(run.planned) == 1
        s = json.loads((run_dir / "summary.json").read_text())
        assert s["map"]["coverage_area"] > 0.99
        assert s["map"]["error_area_smooth"]["rmse"] < 0.05
        assert abs(s["tracking"]["cross_track_mean_m"] - 0.05) < 0.01
        assert "coverage of area" in (run_dir / "summary.txt").read_text()


def test_plots_written_and_not_blank():
    require("matplotlib")
    from matplotlib import image

    from drone_lab.plotting import PLOT_FILES, make_all_plots

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        _make_run(run_dir)
        for theme in ("light", "dark"):
            paths = make_all_plots(run_dir, theme=theme)
            assert len(paths) == len(PLOT_FILES)
            for p in paths:
                assert p.exists() and p.stat().st_size > 20_000, p
                img = image.imread(p)
                assert img.shape[0] > 600 and img.shape[1] > 800 and float(img[..., :3].std()) > 0.02, p


def test_plot_survey_cli():
    require("matplotlib")
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_survey.py"
    spec = importlib.util.spec_from_file_location("plot_survey", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        _make_run(run_dir)
        assert mod.main([str(run_dir)]) == 0
        assert (run_dir / "map_with_trajectory.png").exists() and (run_dir / "summary.txt").exists()


if __name__ == "__main__":
    main_for(globals(), "test_plotting")
