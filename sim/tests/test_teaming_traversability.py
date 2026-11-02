"""Traversability on synthetic elevation grids: slope, step, roughness, unknown cells, footprints, inflation.

numpy + scipy only (runs in env_isaaclab with ``python tests/test_teaming_traversability.py`` or under pytest).
"""

from __future__ import annotations

import math

import numpy as np
from _runner import main_for


def _grid(n=60, res=0.1, x0=0.0, y0=0.0):
    from drone_lab.mapping import GridSpec

    return GridSpec(x0, y0, res, n, n)


def _interior(a, margin=6):
    return a[margin:-margin, margin:-margin]


def _cfg(**kw):
    from drone_lab.teaming.traversability import TraversabilityCfg

    return TraversabilityCfg(**kw)


def test_minmod():
    from drone_lab.teaming.traversability import minmod

    a = np.array([1.0, -2.0, 3.0, 0.0, -1.0])
    b = np.array([2.0, -1.0, -3.0, 5.0, -4.0])
    assert np.allclose(minmod(a, b), [1.0, -1.0, 0.0, 0.0, -1.0])


def test_flat_ground_is_free_and_costs_one():
    from drone_lab.teaming.traversability import compute_traversability

    g = _grid()
    t = compute_traversability(np.zeros(g.shape), g, _cfg(inflation_buffer_m=0.0))
    assert np.allclose(_interior(t.slope_deg), 0.0) and np.allclose(_interior(t.step), 0.0)
    assert np.allclose(_interior(t.roughness), 0.0, atol=1e-9)
    assert not t.lethal[5:-5, 5:-5].any()
    assert np.allclose(_interior(t.cost), 1.0)
    # only a thin band along the array edge is unknown (half the plane-fit window is outside) and inflated
    assert t.unknown[0].all() and not t.unknown[2:-2, 2:-2].any()
    assert t.free_fraction() > 0.7


def test_slope_of_an_inclined_plane_is_exact_and_resolution_independent():
    from drone_lab.teaming.traversability import compute_traversability

    for res in (0.1, 0.25):
        g = _grid(n=int(round(8.0 / res)), res=res)
        xs, ys = g.cell_centers()
        for deg in (5.0, 12.0):
            k = math.tan(math.radians(deg))
            z = k * xs[None, :] + 0.5 * k * ys[:, None]  # also inclined in y
            t = compute_traversability(z + 0.3, g, _cfg())
            expect = math.degrees(math.atan(k * math.hypot(1.0, 0.5)))
            assert np.allclose(_interior(t.slope_deg, 4), expect, atol=1e-6), (res, deg)
            # a plane has no steps and no roughness: the minmod slope removes the incline
            assert np.max(_interior(t.step, 4)) < 1e-9 and np.max(_interior(t.roughness, 4)) < 1e-6
            assert not _interior(t.lethal, 4).any()


def test_steep_slope_is_lethal_and_cost_grows_with_slope():
    from drone_lab.teaming.traversability import compute_traversability

    g = _grid(n=80, res=0.1)
    xs, _ = g.cell_centers()
    # 0..4 m gentle (8 deg), 4..8 m steep (30 deg), continuous height
    k1, k2 = math.tan(math.radians(8.0)), math.tan(math.radians(30.0))
    zx = np.where(xs < 4.0, k1 * xs, k1 * 4.0 + k2 * (xs - 4.0))
    t = compute_traversability(np.tile(zx, (g.ny, 1)), g, _cfg(max_slope_deg=22.0, inflation_buffer_m=0.0))
    row = g.ny // 2
    gentle = (xs > 1.0) & (xs < 3.0)
    steep = (xs > 5.0) & (xs < 7.0)
    assert not t.lethal[row, gentle].any() and t.lethal[row, steep].all()
    assert np.all(t.cost[row, gentle] > 1.0) and np.all(np.isfinite(t.cost[row, gentle]))
    flat = compute_traversability(np.zeros(g.shape), g, _cfg(inflation_buffer_m=0.0))
    assert np.all(t.cost[row, gentle] > flat.cost[row, gentle])


def test_ledge_step_height_is_measured_and_limits_apply():
    from drone_lab.teaming.traversability import compute_traversability

    for res in (0.1, 0.25):
        g = _grid(n=int(round(6.0 / res)), res=res)
        xs, _ = g.cell_centers()
        for h, lethal in ((0.30, True), (0.06, False)):
            z = np.tile(np.where(xs < 3.0, 0.0, h), (g.ny, 1))
            t = compute_traversability(z, g, _cfg(max_step_m=0.12, inflation_buffer_m=0.0))
            row = g.ny // 2
            edge = np.argmin(np.abs(xs - 3.0))  # first cell of the upper side (or last of the lower)
            assert abs(np.nanmax(t.step[row]) - h) < 1e-9, (res, h, np.nanmax(t.step[row]))
            far = np.abs(xs - 3.0) > 0.6
            assert np.allclose(t.step[row, far], 0.0)
            assert bool(t.lethal[row, edge]) == lethal, (res, h)
            if lethal:
                # inflated by the robot radius on both sides of the edge, free beyond it
                assert t.lethal[row, np.abs(xs - 3.0) < 0.3].all()
                inner = (xs > 0.8) & (xs < xs[-1] - 0.8)  # away from the array edge
                assert not t.lethal[row, (np.abs(xs - 3.0) > 0.4 + 0.35 + res) & inner].any()


def test_roughness_from_bumps():
    from drone_lab.teaming.traversability import compute_traversability

    rng = np.random.default_rng(0)
    g = _grid(n=60, res=0.1)
    for sigma, lethal in ((0.12, True), (0.005, False)):
        z = rng.normal(0.0, sigma, g.shape)
        t = compute_traversability(
            z, g, _cfg(max_roughness_m=0.05, max_step_m=10.0, max_slope_deg=89.0, inflation_buffer_m=0.0)
        )
        r = np.median(_interior(t.roughness))
        assert 0.6 * sigma < r < 1.2 * sigma, (sigma, r)
        assert np.mean(_interior(t.lethal)) > 0.9 if lethal else not _interior(t.lethal).any()


def test_unknown_cells_and_hole_filling():
    from drone_lab.teaming.traversability import compute_traversability, fill_small_holes

    g = _grid(n=40, res=0.25)
    z = np.zeros(g.shape)
    z[20, 20] = np.nan  # an isolated missed cell: filled
    z[5:15, 5:15] = np.nan  # a 2.5 m unseen patch: stays unknown
    filled, mask = fill_small_holes(z, 2, 5)
    assert mask[20, 20] and filled[20, 20] == 0.0 and np.isnan(filled[10, 10])
    t = compute_traversability(z, g, _cfg())
    assert not t.unknown[20, 20] and not t.lethal[20, 20]
    assert t.unknown[10, 10] and t.lethal[10, 10] and np.isinf(t.cost[10, 10])
    soft = compute_traversability(z, g, _cfg(unknown_is_lethal=False, unknown_cost=6.0, inflation_buffer_m=0.0))
    assert not soft.lethal[10, 10] and abs(soft.cost[10, 10] - 7.0) < 1e-9


def test_footprints_inflation_and_clearance():
    from drone_lab.teaming.traversability import Footprint, compute_traversability, rasterize_footprints

    g = _grid(n=100, res=0.1)  # 10 m x 10 m
    fp = Footprint(5.0, 5.0, 0.6, "person")
    mask = rasterize_footprints(g, [fp])
    xs, ys = g.cell_centers()
    d = np.hypot(xs[None, :] - 5.0, ys[:, None] - 5.0)
    assert np.array_equal(mask, d <= 0.6)
    tiny = rasterize_footprints(g, [Footprint(2.04, 2.04, 0.01)])  # smaller than a cell: still blocks its cell
    assert tiny.sum() == 1 and tiny[20, 20]
    cfg = _cfg(robot_radius_m=0.35, inflation_buffer_m=0.8, w_inflation=3.0)
    t = compute_traversability(np.zeros(g.shape), g, cfg, footprints=[fp])
    assert t.lethal_objects.sum() == mask.sum()
    inner = (np.abs(xs[None, :] - 5) < 4.3) & (np.abs(ys[:, None] - 5) < 4.3)  # away from the array edge
    # lethal = object or closer than the robot radius to it (centre-to-centre distance)
    assert not t.lethal[(d > 0.6 + 0.35 + 0.1) & inner].any() and t.lethal[d < 0.6 + 0.25].all()
    # soft band: cost decreases with distance and is 1 beyond the buffer
    ring1 = (d > 1.05) & (d < 1.15)
    ring2 = (d > 1.55) & (d < 1.65)
    far = d > 0.6 + 0.35 + 0.8 + 0.15
    assert t.cost[ring1].min() > t.cost[ring2].max() > 1.0
    core = (np.abs(xs[None, :] - 5) < 3.5) & (np.abs(ys[:, None] - 5) < 3.5)  # also clear of the edge's soft band
    assert np.allclose(t.cost[far & core], 1.0)
    assert np.isinf(t.clearance).sum() == 0 and abs(t.sample(t.clearance, np.array([[5.0, 7.05]]))[0] - 1.5) < 1e-9


def test_polygon_footprints():
    from drone_lab.teaming.traversability import Footprint, points_in_polygon, rasterize_footprints

    g = _grid(n=100, res=0.1)
    xs, ys = g.cell_centers()
    # a 2 m x 0.6 m box rotated by 30 deg around (5, 5), like a lying person or a pallet from the ground truth
    c, s_ = math.cos(math.radians(30)), math.sin(math.radians(30))
    corners = [(5 + c * dx - s_ * dy, 5 + s_ * dx + c * dy) for dx, dy in ((-1, -0.3), (1, -0.3), (1, 0.3), (-1, 0.3))]
    mask = rasterize_footprints(g, [Footprint(5.0, 5.0, 1.04, "pallet", polygon=tuple(corners))])
    # local box coordinates of every cell centre
    lx = c * (xs[None, :] - 5) + s_ * (ys[:, None] - 5)
    ly = -s_ * (xs[None, :] - 5) + c * (ys[:, None] - 5)
    inside = (np.abs(lx) < 0.95) & (np.abs(ly) < 0.25)
    outside = (np.abs(lx) > 1.1) | (np.abs(ly) > 0.4)
    assert mask[inside].all() and not mask[outside].any()
    assert abs(mask.sum() * 0.01 - 1.2) < 0.4  # ~1.2 m^2 (plus the outline cells)
    # a sliver thinner than a cell still blocks the cells under its outline
    sliver = rasterize_footprints(g, [Footprint(2, 2, 1, polygon=((1.0, 2.01), (3.0, 2.01), (3.0, 2.03), (1.0, 2.03)))])
    assert sliver[20, 10:30].all()
    assert points_in_polygon(np.array([0.5, 1.5]), np.array([0.5, 0.5]), [(0, 0), (1, 0), (1, 1), (0, 1)]).tolist() == [
        True,
        False,
    ]


def test_queries_height_and_sample():
    from drone_lab.teaming.traversability import compute_traversability

    g = _grid(n=50, res=0.2, x0=-5.0, y0=-5.0)
    xs, ys = g.cell_centers()
    z = 0.1 * xs[None, :] + 0.2 * ys[:, None]
    t = compute_traversability(z, g, _cfg())
    pts = np.array([[0.0, 0.0], [1.23, -2.5], [3.3, 3.3]])
    assert np.allclose(t.height_at(pts), 0.1 * pts[:, 0] + 0.2 * pts[:, 1], atol=1e-9)
    assert np.isnan(t.height_at(np.array([[100.0, 0.0]]))[0])
    assert np.isnan(t.sample(t.cost, np.array([[100.0, 0.0]]))[0])
    iy, ix, ok = t.world_to_cell(np.array([0.05]), np.array([-0.05]))
    cx, cy = t.cell_to_world(iy, ix)
    assert ok[0] and abs(cx[0] - 0.1) < 1e-9 and abs(cy[0] + 0.1) < 1e-9


if __name__ == "__main__":
    main_for(globals(), "test_teaming_traversability")
