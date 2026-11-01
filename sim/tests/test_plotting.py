"""Run files and plots on a small synthetic run (numpy + matplotlib only, no torch / Isaac Lab)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from _runner import main_for, require

from drone_lab.mapping import (
    ElevationMapper,
    GridSpec,
    MeshHeightQuery,
    ground_truth_map,
    heightfield_to_mesh,
    truth_at_hits_map,
)
from drone_lab.paths import Area, lawnmower, resample_polyline
from drone_lab.survey_io import TrajectoryRecorder, compute_summary, load_run, save_run

AREA = Area(-6.0, 6.0, -6.0, 6.0)


def _make_run(run_dir: Path, spacing: float = 3.0) -> None:
    """A tiny but complete run: analytic terrain, lawnmower flight, grid rays, map, ground truth, summary.

    The footprint is 4 m wide, so ``spacing`` above 4 m leaves uncovered strips.
    """
    xs = np.arange(-9.0, 9.0001, 0.1)
    xx, yy = np.meshgrid(xs, xs)
    zz = 0.8 * np.sin(xx / 2.0) + 1.5 * ((np.abs(xx - 2) < 1.5) & (np.abs(yy + 2) < 1.0))
    v, f = heightfield_to_mesh(xs, xs, zz)
    query = MeshHeightQuery(v, f)
    plan = lawnmower(AREA, spacing, 4.0)
    path, _ = resample_polyline(plan, 0.15)
    rec = TrajectoryRecorder()
    grid = GridSpec.for_survey(AREA, 0.25)
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
                              extra={"pattern": "lawnmower", "flight_z_m": 4.0, "max_lead_m": 1.0},
                              ground_truth_at_hits=truth_at_hits_map(mapper.points(), grid, query))
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
        assert s["map"]["error_area_at_hits"]["rmse"] < 1e-5  # exact synthetic rays: no sensor error
        assert abs(s["tracking"]["cross_track_mean_m"] - 0.05) < 0.01
        text = (run_dir / "summary.txt").read_text()
        assert "coverage of area" in text and "map error (sensor)" in text and "PARTIAL RUN" not in text


def test_coverage_holes_are_outlined():
    """Cells without ray hits are outlined (legend swatch included); a fully covered map has none."""
    require("matplotlib")
    import matplotlib.pyplot as plt

    from drone_lab.plotting import DARK, LIGHT, _no_hit_cells, _outline_cells, make_all_plots

    with tempfile.TemporaryDirectory() as d:
        full_dir, holes_dir = Path(d) / "full", Path(d) / "holes"
        _make_run(full_dir)
        _make_run(holes_dir, spacing=6.0)  # lines 6 m apart, 4 m footprint: uncovered strips
        assert not _no_hit_cells(load_run(full_dir)).any()
        run = load_run(holes_dir)
        holes = _no_hit_cells(run)
        assert 0.2 < holes.mean() < 0.5, holes.mean()
        for theme in (LIGHT, DARK):
            fig, ax = plt.subplots()
            assert _outline_cells(ax, holes, run.grid, theme) and len(ax.collections) == 1
            assert not _outline_cells(ax, np.zeros_like(holes), run.grid, theme) and len(ax.collections) == 1
            plt.close(fig)
        for theme in ("light", "dark"):
            assert len(make_all_plots(holes_dir, theme=theme)) == 4


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
