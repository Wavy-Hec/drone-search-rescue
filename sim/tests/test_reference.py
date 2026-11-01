"""Reference trajectories of the tracking task (torch, CPU)."""

from __future__ import annotations

import math

from _runner import main_for, require


def _bank(n=1):
    require("torch")
    from drone_lab.reference import ReferenceBank

    return ReferenceBank(n)


def test_speed_bound_and_analytic_velocity():
    torch = require("torch")

    torch.manual_seed(3)
    bank = _bank(512)
    bank.sample(torch.arange(512))
    t = torch.linspace(0.0, 40.0, 2001)
    dt = float(t[1] - t[0])
    for e in range(0, 512, 37):
        ids = torch.full((len(t),), e)
        p, v = bank.position(t, ids), bank.velocity(t, ids)
        assert float(v.norm(dim=1).max()) <= float(bank.speed[e]) * (1 + 1e-4)
        v_num = (p[2:] - p[:-2]) / (2 * dt)
        assert float((v_num - v[1:-1]).abs().max()) < 2e-3
        assert float(p[:, :2].abs().max()) <= float(bank.size[e]) * 1.01 + 1e-6


def test_fixed_shapes_are_deterministic():
    torch = require("torch")

    t = torch.linspace(0.0, 10.0, 101)
    ids = torch.zeros(len(t), dtype=torch.long)
    for shape in ("circle", "figure8", "lissajous", "random"):
        a, b = _bank(), _bank()
        torch.manual_seed(1)
        a.sample(torch.arange(1), fixed_shape=shape, fixed_size=2.0, fixed_speed=1.0, fixed_z_amp=0.0)
        torch.manual_seed(2)
        b.sample(torch.arange(1), fixed_shape=shape, fixed_size=2.0, fixed_speed=1.0, fixed_z_amp=0.0)
        assert torch.allclose(a.position(t, ids), b.position(t, ids)), shape


def test_circle_and_figure8_geometry():
    torch = require("torch")

    t = torch.linspace(0.0, 2 * math.pi * 2.0 / 1.0, 400)
    ids = torch.zeros(len(t), dtype=torch.long)
    c = _bank()
    c.sample(torch.arange(1), fixed_shape="circle", fixed_size=2.0, fixed_speed=1.0, fixed_z_amp=0.0)
    r = c.position(t, ids)[:, :2].norm(dim=1)
    assert torch.allclose(r, torch.full_like(r, 2.0), atol=1e-4)
    assert torch.allclose(c.velocity(t, ids).norm(dim=1), torch.ones(len(t)), atol=1e-4)
    f = _bank()
    f.sample(torch.arange(1), fixed_shape="figure8", fixed_size=2.0, fixed_speed=1.0, fixed_z_amp=0.0)
    p = f.position(t, ids)
    assert abs(float(p[:, 0].max()) - 2.0) < 0.01 and abs(float(p[:, 1].max()) - 1.0) < 0.01
    z = _bank()
    z.sample(torch.arange(1), fixed_shape="circle", fixed_size=1.5, fixed_speed=1.0, fixed_z_amp=0.3)
    zz = z.position(torch.linspace(0, 60, 600), torch.zeros(600, dtype=torch.long))[:, 2]
    assert 0.0 < float(zz.max()) <= 0.3 + 1e-6  # vertical wave limited by the speed budget


if __name__ == "__main__":
    main_for(globals(), "test_reference")
