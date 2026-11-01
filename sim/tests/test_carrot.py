"""Carrot (moving target) logic: speed, lead limit, clamping, and a full follow on a lawnmower (torch, CPU)."""

from __future__ import annotations

import numpy as np

from _runner import main_for, require

from drone_lab.paths import Area, lawnmower


def _carrot():
    require("torch")
    import drone_lab.carrot as carrot

    return carrot


def test_interpolate_straight_and_padded():
    c = _carrot()
    torch = require("torch")

    short = np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]])
    long = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 4.0, 5.0]])
    pts, arc, length = c.pad_paths([short, long], 0.25)
    assert pts.shape[0] == 2 and torch.allclose(length, torch.tensor([2.0, 14.0]))
    p, seg = c.interpolate_path(pts, arc, torch.tensor([1.0, 12.0]))
    assert torch.allclose(p, torch.tensor([[1.0, 0.0, 5.0], [10.0, 2.0, 5.0]]), atol=1e-5)
    # beyond the end (and in the padding) it stays on the last point
    p, _ = c.interpolate_path(pts, arc, torch.tensor([50.0, 50.0]))
    assert torch.allclose(p, torch.tensor([[2.0, 0.0, 5.0], [10.0, 4.0, 5.0]]), atol=1e-5)
    p, _ = c.interpolate_path(pts, arc, torch.tensor([-1.0, -1.0]))
    assert torch.allclose(p[:, :2], torch.zeros(2, 2), atol=1e-6)


def test_carrot_moves_at_speed_when_drone_keeps_up():
    c = _carrot()
    torch = require("torch")

    pts, arc, _ = c.pad_paths([np.array([[0.0, 0.0, 5.0], [100.0, 0.0, 5.0]])], 0.25)
    s = torch.zeros(1)
    for _ in range(20):
        carrot_pos, _ = c.interpolate_path(pts, arc, s)
        drone = carrot_pos - torch.tensor([[0.3, 0.0, 0.0]])  # trailing 0.3 m: below slow_lead
        s, carrot_pos, gate = c.advance_carrot(s, pts, arc, drone, 1.5, 0.1, 1.0, 0.7)
        assert torch.allclose(gate, torch.ones(1))
    assert torch.allclose(s, torch.tensor([3.0]), atol=1e-4)


def test_carrot_never_runs_away_from_a_stuck_drone():
    c = _carrot()
    torch = require("torch")

    path = lawnmower(Area(-10, 10, -10, 10), 4.0, 6.0)
    pts, arc, _ = c.pad_paths([path], 0.25)
    drone = pts[:, 0].clone()
    s = torch.zeros(1)
    for _ in range(500):
        s, carrot_pos, gate = c.advance_carrot(s, pts, arc, drone, 2.0, 0.1, 1.0, 0.7)
        assert torch.linalg.norm(carrot_pos - drone) <= 1.0 + 1e-5
    assert float(s) <= 1.0 + 1e-5 and float(gate) < 1e-3


def test_clamp_offset_ball_and_box():
    c = _carrot()
    torch = require("torch")

    off = torch.tensor([[3.0, 4.0, 0.0], [0.3, -0.4, 0.2], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9]])
    out = c.clamp_offset(off, 1.0, (1.0, 1.0, 1.0))
    assert torch.allclose(out[0], torch.tensor([0.6, 0.8, 0.0]), atol=1e-6)  # direction kept, norm 1
    assert torch.allclose(out[1], off[1]) and torch.allclose(out[2], off[2])
    assert torch.allclose(torch.linalg.norm(out[3]), torch.tensor(1.0), atol=1e-6)
    box = c.clamp_offset(torch.tensor([[2.0, -0.2, -3.0]]), None, (0.5, 0.5, 1.0))
    assert torch.allclose(box, torch.tensor([[0.5, -0.2, -1.0]]))


def test_follower_stays_in_training_box_and_completes():
    """A lagging point-mass follower: the observed offset stays inside +-1 m and the survey completes."""
    c = _carrot()
    torch = require("torch")

    paths = [lawnmower(Area(-10, 10, -10, 0), 4.0, 6.0), lawnmower(Area(-10, 10, 0, 10), 4.0, 6.0)]
    pts, arc, length = c.pad_paths(paths, 0.25)
    pos = pts[:, 0].clone().double()
    vel = torch.zeros_like(pos)
    s = torch.zeros(2)
    dt, worst = 0.1, 0.0
    for _ in range(3000):
        s, carrot_pos, _ = c.advance_carrot(s, pts, arc, pos.float(), 1.2, dt, 1.0, 0.7)
        acc = 2.0 * (carrot_pos.double() - pos) - 2.5 * vel  # sluggish: lags well behind the carrot
        vel = vel + acc * dt
        pos = pos + vel * dt
        obs = c.clamp_offset((pos - carrot_pos.double()).float(), 1.0, (1.0, 1.0, 1.0))
        worst = max(worst, float(torch.linalg.norm(pos.float() - carrot_pos, dim=-1).max()))
        assert torch.all(obs.abs() <= 1.0 + 1e-6)
        if torch.all(s >= length - 1e-4):
            break
    assert torch.all(s >= length - 1e-4), (s, length)
    assert worst <= 1.0 + 0.05  # the carrot only moves while the drone is within the lead limit


if __name__ == "__main__":
    main_for(globals(), "test_carrot")
