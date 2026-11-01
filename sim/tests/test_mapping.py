"""Elevation mapping, mesh ground truth and metrics on synthetic data (numpy only)."""

from __future__ import annotations

import numpy as np

from _runner import main_for

from drone_lab.mapping import (
    ElevationMapper,
    GridSpec,
    MeshHeightQuery,
    coverage_fraction,
    elevation_map,
    ground_truth_map,
    heightfield_to_mesh,
    map_error_stats,
    mesh_top_heights,
    path_length,
    survey_map_metrics,
    tracking_stats,
)
from drone_lab.paths import Area, lawnmower, resample_polyline

AREA = Area(-15.0, 15.0, -15.0, 15.0)


def _hills(x, y):
    """Smooth synthetic terrain, heights about -1..2.5 m."""
    return 1.2 * np.sin(x / 4.0) * np.cos(y / 5.0) + 0.8 * np.exp(-((x - 5) ** 2 + (y + 4) ** 2) / 18.0)


def _terrain_mesh(fn=_hills, half=18.0, step=0.1):
    xs = np.arange(-half, half + 1e-9, step)
    ys = np.arange(-half, half + 1e-9, step)
    xx, yy = np.meshgrid(xs, ys)
    return heightfield_to_mesh(xs, ys, fn(xx, yy))


def _grid_offsets(size=5.0, res=0.25):
    o = np.arange(-size / 2, size / 2 + 1e-9, res)
    gx, gy = np.meshgrid(o, o)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _fly(vertices, faces, spacing, step=0.1, footprint=5.0):
    """Drone samples along a lawnmower path; every sample casts a grid of vertical rays."""
    path, _ = resample_polyline(lawnmower(AREA, spacing, 6.0, turn="square"), step)
    query = MeshHeightQuery(vertices, faces)
    rays = (path[:, None, :2] + _grid_offsets(footprint)[None]).reshape(-1, 2)
    return np.column_stack([rays, query(rays)]), path


def test_grid_spec_indexing():
    g = GridSpec.from_bounds(-1.0, 1.0, 0.0, 0.5, 0.25)
    assert (g.nx, g.ny) == (8, 2) and g.shape == (2, 8)
    ix, iy, valid = g.cell_index(np.array([-1.0, 0.99, 1.0, 0.1]), np.array([0.0, 0.49, 0.2, -0.01]))
    assert ix.tolist()[:2] == [0, 7] and iy.tolist()[:2] == [0, 1]
    assert valid.tolist() == [True, True, False, False]
    xs, ys = g.cell_centers()
    assert np.isclose(xs[0], -0.875) and np.isclose(ys[-1], 0.375)
    m = g.area_mask(Area(-0.5, 0.5, 0.0, 0.25))
    assert m.sum() == 4
    assert GridSpec.from_dict(g.to_dict()) == g


def test_reducers_and_empty_cells():
    g = GridSpec.from_bounds(0.0, 1.0, 0.0, 1.0, 0.5)
    pts = np.array([[0.1, 0.1, 1.0], [0.2, 0.2, 3.0], [0.3, 0.1, 2.0], [0.7, 0.2, 5.0], [np.inf, 0.0, 0.0]])
    expect = {"median": 2.0, "mean": 2.0, "max": 3.0, "min": 1.0}
    for reducer, value in expect.items():
        h, c = elevation_map(pts, g, reducer)
        assert np.isclose(h[0, 0], value), reducer
        assert np.isclose(h[0, 1], 5.0) and np.isnan(h[1, 0]) and np.isnan(h[1, 1])
        assert c.tolist() == [[3, 1], [0, 0]]
    h, _ = elevation_map(np.array([[0.1, 0.1, 1.0], [0.2, 0.1, 2.0]]), g, "median")
    assert np.isclose(h[0, 0], 1.5)  # even count: mean of the two middle values


def test_mapper_matches_batch():
    rng = np.random.default_rng(1)
    g = GridSpec.from_area(AREA, 0.5)
    pts = np.column_stack([rng.uniform(-16, 16, (5000, 2)), rng.normal(size=5000)])
    mapper = ElevationMapper(g)
    for chunk in np.array_split(pts, 7):
        mapper.add(chunk)
    h1, c1 = mapper.finalize("median")
    h2, c2 = elevation_map(pts, g, "median")
    assert np.array_equal(c1, c2) and np.allclose(h1, h2, equal_nan=True, atol=1e-5)
    assert np.array_equal(mapper.count, c2)


def test_mesh_query_exact_on_planar_and_matches_heightfield():
    plane = lambda x, y: 0.3 * x - 0.2 * y + 1.0  # noqa: E731
    v, f = _terrain_mesh(plane, half=5.0, step=0.5)
    rng = np.random.default_rng(0)
    q = rng.uniform(-4.9, 4.9, (2000, 2))
    assert np.allclose(mesh_top_heights(v, f, q), plane(q[:, 0], q[:, 1]), atol=1e-9)
    assert np.isnan(mesh_top_heights(v, f, np.array([[6.0, 0.0]]))).all()  # outside the mesh
    v, f = _terrain_mesh()
    q = rng.uniform(-17.0, 17.0, (5000, 2))
    assert np.abs(MeshHeightQuery(v, f)(q) - _hills(q[:, 0], q[:, 1])).max() < 0.01


def test_mesh_query_takes_top_surface():
    # two horizontal triangles stacked above the same xy: the higher one is the top surface
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 2], [0, 1, 2], [0, 0, 0], [0, 0, 5], [1, 0, 0]],
                 dtype=float)
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])  # the third one is vertical and ignored
    assert np.allclose(mesh_top_heights(v, f, np.array([[0.2, 0.2]])), 2.0)


def test_survey_map_low_error_and_full_coverage():
    """Known terrain + downward rays along a lawnmower with spacing < footprint: low error, ~100 % coverage."""
    v, f = _terrain_mesh()
    hits, path = _fly(v, f, spacing=4.0)
    g = GridSpec.from_area(AREA, 0.25)
    elev, count = elevation_map(hits, g, "median")
    gt, gt_range = ground_truth_map(v, f, g, supersample=3)
    stats = survey_map_metrics(elev, count, gt, gt_range, g, AREA)
    assert stats["coverage_area"] > 0.999, stats["coverage_area"]
    err = stats["error_area"]
    assert err["rmse"] < 0.03 and abs(err["bias"]) < 0.01, err
    # ground truth agrees with the analytic surface at the cell centres
    xs, ys = g.cell_centers()
    xx, yy = np.meshgrid(xs, ys)
    assert np.nanmax(np.abs(gt - _hills(xx, yy))) < 0.02


def test_coverage_drops_with_wide_spacing():
    """Spacing 8 m with a 5 m footprint leaves about 3/8 of the area unseen."""
    v, f = _terrain_mesh()
    hits, _ = _fly(v, f, spacing=8.0)
    g = GridSpec.from_area(AREA, 0.25)
    _, count = elevation_map(hits, g, "median")
    cov = coverage_fraction(count, g.area_mask(AREA))
    # 30 m extent -> 4 lines, 7.5 m apart, each covering a 5.25 m wide band (21 rays at 0.25 m)
    assert 0.6 < cov < 0.78, cov


def test_error_stats_and_tracking():
    est = np.array([[1.0, 2.0], [np.nan, 4.5]])
    gt = np.array([[1.5, 2.0], [3.0, 4.0]])
    s = map_error_stats(est, gt)
    assert s["n_cells"] == 3 and np.isclose(s["max_abs"], 0.5) and np.isclose(s["bias"], 0.0)
    assert np.isclose(s["rmse"], np.sqrt((0.25 + 0 + 0.25) / 3))
    assert map_error_stats(np.full((2, 2), np.nan), gt)["rmse"] is None
    plan = np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0]])
    traj = np.column_stack([np.linspace(0, 20, 50), np.full(50, 0.3), np.full(50, 5.4)])
    t = tracking_stats(traj, plan)
    assert np.isclose(t["cross_track_mean"], 0.3) and np.isclose(t["distance3d_max"], 0.5)
    assert np.isclose(path_length(traj), 20.0)


if __name__ == "__main__":
    main_for(globals(), "test_mapping")
