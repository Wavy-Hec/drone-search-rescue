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


def _follow_fast_carrot(c, speed: float, max_lead: float, steps: int = 300, dt: float = 0.1):
    """Carrot commanded at ``speed`` ahead of a follower like the GPU policy: it closes the offset with a 0.75 s
    time constant and tops out at 1.33 m/s (measured). Returns (stalled steps, worst distance, carrot arclength)."""
    torch = require("torch")

    pts, arc, length = c.pad_paths([lawnmower(Area(-20, 20, -20, 20), 4.0, 7.5)], 0.25)
    s = torch.zeros(1)
    drone = pts[:, 0].clone()
    stalled, worst = 0, 0.0
    for k in range(steps):
        s_old = s.clone()
        s, carrot_pos, _ = c.advance_carrot(s, pts, arc, drone, speed, dt, max_lead, 0.7 * max_lead)
        if k >= 30 and float(s - s_old) < 1e-6:  # after the start-up, the carrot should never wait
            stalled += 1
        worst = max(worst, float(torch.linalg.norm(carrot_pos - drone)))
        v = (carrot_pos - drone) / 0.75
        drone = drone + v * torch.clamp(1.33 / torch.linalg.norm(v).clamp(min=1e-9), max=1.0) * dt
    assert float(s) < float(length[0]) - 1.0  # still mid-survey: every counted step could have advanced
    return stalled, worst, float(s)


def test_carrot_keeps_moving_at_high_commanded_speed():
    """Regression (survey flights, 2026-11-01): once ``speed * dt > max_lead - slow_lead`` (3 m/s with the
    defaults) the whole step overshot ``max_lead`` and was rejected, so the carrot went stop-and-go (waiting on
    84 % of the steps at --speed 6 on the GPU) and stalled for good at ``speed * dt >= max_lead``. The step is now
    shortened to the remaining lead: the carrot keeps moving and still never gets more than max_lead ahead."""
    c = _carrot()

    # before the fix: waited on 231 / 270 / 200 of the 270 counted steps, carrot at 26.4 / 1.0 / 15.8 m after 30 s
    for speed, max_lead in ((6.0, 1.0), (10.0, 1.0), (2.0, 0.5)):
        stalled, worst, s = _follow_fast_carrot(c, speed, max_lead)
        assert stalled == 0, (speed, max_lead, f"carrot waited on {stalled} steps")
        assert worst <= max_lead + 1e-5, (speed, max_lead, worst)
        # the follower runs at its top speed: the 1.33 m/s cap, or max_lead / 0.75 s when the lead limits it
        assert s > 0.9 * min(1.33, max_lead / 0.75) * 30.0, (speed, max_lead, s)


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
