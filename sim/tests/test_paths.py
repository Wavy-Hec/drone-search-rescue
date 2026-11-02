"""Survey path generators and polyline geometry (numpy only)."""

from __future__ import annotations

import json
import math
from itertools import pairwise

import numpy as np
from _runner import main_for

from drone_lab.paths import (
    Area,
    build_env_paths,
    cumulative_length,
    distance_to_polyline,
    figure8,
    lawnmower,
    load_waypoints,
    point_at_arclength,
    polyline_length,
    resample_polyline,
    resolve_pattern,
    spiral,
)

AREA = Area(-20.0, 20.0, -20.0, 20.0)


def _area_samples(area: Area, n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(area.x_min, area.x_max, n), rng.uniform(area.y_min, area.y_max, n)])


def test_lawnmower_bounds_spacing_and_length():
    spacing, z = 4.0, 7.5
    for turn in ("round", "square"):
        path = lawnmower(AREA, spacing, z, turn=turn)
        assert path.shape[1] == 3 and np.allclose(path[:, 2], z)
        # lines stay in the area; round U-turns bulge at most half a spacing outside it
        assert AREA.contains(path, tol=0.5 * spacing + 1e-9).all()
        ys = np.unique(np.round(path[np.isclose(np.abs(path[:, 0]), 20.0), 1], 6))
        gaps = np.diff(ys)
        assert np.all(gaps <= spacing + 1e-9)
        assert ys[0] - AREA.y_min <= 0.5 * spacing + 1e-9 and AREA.y_max - ys[-1] <= 0.5 * spacing + 1e-9
        n = len(ys)
        expected = n * AREA.width + (n - 1) * (math.pi * spacing / 2 if turn == "round" else spacing)
        # round turns are 17-segment polygons: slightly shorter than the true semicircles
        assert abs(polyline_length(path) - expected) < (0.2 if turn == "round" else 1e-6), (turn, expected)


def test_lawnmower_covers_area_with_footprint():
    """Every point of the area is within half a spacing (cross-track) of a survey line."""
    spacing = 4.0
    path = lawnmower(AREA, spacing, 5.0)
    d, _ = distance_to_polyline(_area_samples(AREA), path, planar=True)
    assert d.max() <= 0.5 * spacing + 1e-6


def test_lawnmower_uneven_extent_and_direction_y():
    area = Area(0.0, 30.0, 0.0, 13.0)
    path = lawnmower(area, 4.0, 3.0, direction="y", turn="square")
    xs = np.unique(np.round(path[:, 0], 6))
    assert len(xs) == 8  # ceil(30 / 4) = 8 lines along y
    assert np.allclose(np.diff(xs), 30.0 / 8)
    assert area.contains(path, tol=1e-9).all()


def test_spiral_inside_area_and_covers_it():
    spacing = 4.0
    path = spiral(AREA, spacing, 6.0)
    assert AREA.contains(path, tol=1e-9).all()
    d, _ = distance_to_polyline(_area_samples(AREA), path, planar=True)
    # rings are one spacing apart and the outer ring is half a spacing inside: corners are the worst case
    assert d.max() <= spacing / math.sqrt(2) + 1e-6
    strip = Area(-20.0, 20.0, -20.0, 0.0)
    path2 = spiral(strip, spacing, 6.0)
    d2, _ = distance_to_polyline(_area_samples(strip), path2, planar=True)
    assert d2.max() <= spacing / math.sqrt(2) + 1e-6


def test_figure8_closed_and_inside():
    path = figure8(AREA, 6.0, loops=2)
    assert np.allclose(path[0], path[-1])
    assert AREA.contains(path, tol=1e-9).all()
    assert polyline_length(path) > 2 * 2 * AREA.width * 0.9


def test_build_env_paths_split_and_stagger():
    paths = build_env_paths("lawnmower", 3, AREA, 4.0, 7.0)
    assert len(paths) == 3
    y_ranges = sorted((p[:, 1].min(), p[:, 1].max()) for p in paths)
    for (_, hi0), (lo1, _) in pairwise(y_ranges):
        assert hi0 < lo1  # disjoint strips
    copies = build_env_paths("lawnmower", 2, AREA, 4.0, 7.0, multi_env_mode="copy")
    assert np.allclose(copies[0], copies[1])
    loops = build_env_paths("figure8", 2, AREA, 4.0, 7.0)
    assert not np.allclose(loops[0][0], loops[1][0])  # staggered starts
    assert np.allclose(polyline_length(loops[0]), polyline_length(loops[1]), rtol=1e-6)
    wp = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 10.0, 6.0]])
    same = build_env_paths("waypoints", 2, AREA, 4.0, 7.0, waypoints=wp)
    assert np.allclose(same[0], wp) and np.allclose(same[1], wp)  # open path: copied


def test_resample_and_interpolate():
    pts = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [10.0, 0.0, 1.0], [10.0, 5.0, 1.0]])
    res, s = resample_polyline(pts, 0.3)
    assert np.allclose(res[0], pts[0]) and np.allclose(res[-1], pts[-1])
    steps = np.linalg.norm(np.diff(res, axis=0), axis=1)
    assert steps.max() <= 0.3 + 1e-9
    assert abs(s[-1] - 15.0) < 1e-9 and abs(polyline_length(res) - 15.0) < 1e-9
    p = point_at_arclength(res, cumulative_length(res), [0.0, 5.0, 12.5, 99.0])
    assert np.allclose(p[:, :2], [[0, 0], [5, 0], [10, 2.5], [10, 5]])


def test_distance_to_polyline():
    line = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    q = np.array([[5.0, 2.0, 0.0], [-3.0, 4.0, 0.0], [12.0, 0.0, 1.0]])
    d, idx = distance_to_polyline(q, line)
    assert np.allclose(d, [2.0, 5.0, math.sqrt(5.0)])
    d2, _ = distance_to_polyline(q, line, planar=True)
    assert np.allclose(d2, [2.0, 5.0, 2.0])
    assert np.all(idx == 0)


def test_load_waypoints_csv_and_json():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "wp.csv"
        csv_path.write_text("x,y,z\n# comment\n0,0,5\n10,0\n10,10,6\n")
        wp = load_waypoints(csv_path, default_z=7.0)
        assert np.allclose(wp, [[0, 0, 5], [10, 0, 7], [10, 10, 6]])
        json_path = Path(d) / "wp.json"
        json_path.write_text(json.dumps({"waypoints": [[1, 2], [3, 4]]}))
        assert np.allclose(load_waypoints(json_path, default_z=2.0), [[1, 2, 2], [3, 4, 2]])


def test_resolve_pattern_from_flags():
    """A waypoint file with a generated pattern is an error, not a silently ignored file."""
    assert resolve_pattern(None, False) == "lawnmower"
    assert resolve_pattern(None, False, default=None) is None  # survey_map: keep the task config's pattern
    assert resolve_pattern("spiral", False) == "spiral"
    assert resolve_pattern(None, True) == "waypoints"
    assert resolve_pattern("waypoints", True, default=None) == "waypoints"
    for pattern, given in (("lawnmower", True), ("figure8", True), ("waypoints", False)):
        try:
            resolve_pattern(pattern, given)
        except ValueError as exc:
            assert "--waypoints" in str(exc)
        else:
            raise AssertionError(f"resolve_pattern({pattern!r}, {given}) should fail")


if __name__ == "__main__":
    main_for(globals(), "test_paths")
