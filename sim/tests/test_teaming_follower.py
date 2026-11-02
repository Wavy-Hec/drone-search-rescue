"""Pure-pursuit follower: command limits, steering signs, goal handling, progress, closed-loop tracking.

numpy only (runs in env_isaaclab with ``python tests/test_teaming_follower.py`` or under pytest).
"""

from __future__ import annotations

import math

import numpy as np
from _runner import main_for


def _line(length=10.0, ds=0.1):
    x = np.arange(0.0, length + 1e-9, ds)
    return np.column_stack([x, np.zeros_like(x)])


def _s_curve():
    t = np.linspace(0.0, 1.0, 400)
    x = 12.0 * t
    y = 1.8 * np.sin(2.0 * np.pi * t)
    return np.column_stack([x, y])


def test_straight_line_drives_forward_at_cruise_speed():
    from drone_lab.teaming.follower import FollowerCfg, PurePursuitFollower

    f = PurePursuitFollower(_line(), FollowerCfg(cruise_speed=0.7))
    out = f.compute(1.0, 0.0, 0.0)
    assert abs(out.vx - 0.7) < 1e-9 and abs(out.vy) < 1e-9 and abs(out.wz) < 1e-9
    assert out.mode == "track" and not out.done
    assert abs(out.progress_m - 1.0) < 1e-6 and abs(out.cross_track_m) < 1e-9
    assert abs(out.target_xy[0] - 1.8) < 1e-6


def test_steering_signs_and_lateral_pull():
    from drone_lab.teaming.follower import FollowerCfg, PurePursuitFollower

    cfg = FollowerCfg(lateral_gain=0.6, max_lateral_speed=0.25)
    # robot 0.4 m left of a path along +x, heading along it: steer right (wz < 0), slide right (vy < 0)
    f = PurePursuitFollower(_line(), cfg)
    out = f.compute(2.0, 0.4, 0.0)
    assert out.wz < 0 and out.vy < 0 and out.cross_track_m < 0
    assert abs(out.vy + 0.24) < 1e-9  # 0.6 * 0.4 m
    # mirrored on the right
    f = PurePursuitFollower(_line(), cfg)
    out = f.compute(2.0, -0.4, 0.0)
    assert out.wz > 0 and out.vy > 0 and out.cross_track_m > 0
    # heading 0.6 rad to the left of the path: turn right while driving (still under the turn-in-place angle)
    f = PurePursuitFollower(_line(), cfg)
    out = f.compute(2.0, 0.0, 0.6)
    assert out.mode == "track" and out.wz < 0 and 0.0 < out.vx < cfg.cruise_speed


def test_turns_in_place_when_facing_away():
    from drone_lab.teaming.follower import FollowerCfg, PurePursuitFollower

    cfg = FollowerCfg(turn_in_place_rad=1.0, turn_exit_rad=0.5, turn_gain=1.2)
    f = PurePursuitFollower(_line(), cfg)
    out = f.compute(2.0, 0.0, math.pi - 0.3)  # facing backwards, path to the right-hand side behind
    assert out.mode == "turn" and out.vx == 0.0 and out.vy == 0.0 and abs(out.wz) == 1.0
    # hysteresis: stays in turn mode until the error drops below turn_exit_rad
    out = f.compute(2.0, 0.0, 0.8)
    assert out.mode == "turn" and out.wz < 0
    out = f.compute(2.0, 0.0, 0.3)
    assert out.mode == "track" and out.vx > 0


def test_commands_always_within_the_policy_ranges():
    from drone_lab.teaming.follower import FollowerCfg, PurePursuitFollower, VelocityLimits

    rng = np.random.default_rng(7)
    for limits in (VelocityLimits(), VelocityLimits((-0.5, 0.6), (-0.2, 0.2), (-0.4, 0.4))):
        cfg = FollowerCfg(cruise_speed=2.0, max_lateral_speed=1.0, lateral_gain=5.0, turn_gain=10.0, limits=limits)
        f = PurePursuitFollower(_s_curve(), cfg)
        for _ in range(500):
            x, y = rng.uniform(-2, 14), rng.uniform(-4, 4)
            out = f.compute(x, y, rng.uniform(-math.pi, math.pi))
            assert limits.lin_vel_x[0] - 1e-12 <= out.vx <= limits.lin_vel_x[1] + 1e-12
            assert limits.lin_vel_y[0] - 1e-12 <= out.vy <= limits.lin_vel_y[1] + 1e-12
            assert limits.ang_vel_z[0] - 1e-12 <= out.wz <= limits.ang_vel_z[1] + 1e-12
            f.reset(rng.uniform(0, f.length))


def test_goal_reached_stops_and_stays_done():
    from drone_lab.teaming.follower import FollowerCfg, PurePursuitFollower

    cfg = FollowerCfg(goal_tolerance_m=0.3)
    f = PurePursuitFollower(_line(4.0), cfg)
    out = f.compute(3.0, 0.0, 0.0)
    assert not out.done and out.vx >= cfg.min_speed  # slows down near the goal but keeps moving
    assert out.vx < cfg.cruise_speed
    out = f.compute(3.85, 0.05, 0.0)
    assert out.done and (out.vx, out.vy, out.wz) == (0.0, 0.0, 0.0) and out.mode == "goal"
    out = f.compute(3.0, 0.5, 1.0)  # pushed away afterwards: stays done (stand still)
    assert out.done and out.vx == 0.0
    f.reset()
    assert not f.done and f.progress == 0.0


def test_progress_is_monotonic_and_windowed():
    from drone_lab.teaming.follower import FollowerCfg, PurePursuitFollower

    # a U-turn: the return leg passes 1 m from the outbound leg
    out_leg = np.column_stack([np.linspace(0, 6, 61), np.zeros(61)])
    back = np.column_stack([np.linspace(6, 0, 61), np.ones(61)])
    path = np.vstack([out_leg, [[6.0, 0.5]], back])
    f = PurePursuitFollower(path, FollowerCfg(search_ahead_m=2.5))
    out = f.compute(1.0, 0.6, 0.0)  # closer to the return leg, but that is far ahead along the path
    assert out.progress_m < 2.0
    p1 = out.progress_m
    out = f.compute(0.5, 0.0, 0.0)  # moving backwards does not reduce progress
    assert out.progress_m >= p1 - 1e-12
    f.reset(10.0)
    out = f.compute(4.0, 1.0, math.pi)  # on the return leg after a respawn there
    assert 10.0 <= out.progress_m < 12.0


def _simulate(path, cfg, x, y, yaw, dt=0.02, steps=4000, lag=0.15):
    """Kinematic robot with a first-order velocity lag (like a legged robot) under the follower."""
    from drone_lab.teaming.follower import PurePursuitFollower

    f = PurePursuitFollower(path, cfg)
    v = np.zeros(3)
    max_ct = 0.0
    log = []
    for k in range(steps):
        out = f.compute(x, y, yaw)
        log.append((out.vx, out.vy, out.wz))
        if out.done:
            return True, k * dt, max_ct, np.array(log), f
        v += (np.array([out.vx, out.vy, out.wz]) - v) * min(1.0, dt / lag)
        c, s = math.cos(yaw), math.sin(yaw)
        x += (c * v[0] - s * v[1]) * dt
        y += (s * v[0] + c * v[1]) * dt
        yaw += v[2] * dt
        if k * dt > 2.0:
            max_ct = max(max_ct, abs(out.cross_track_m))
    return False, steps * dt, max_ct, np.array(log), f


def test_closed_loop_tracks_an_s_curve_to_the_goal():
    from drone_lab.teaming.follower import FollowerCfg

    path = _s_curve()
    cfg = FollowerCfg(cruise_speed=0.7, lookahead_m=0.8)
    ok, t, max_ct, log, f = _simulate(path, cfg, x=0.0, y=-0.3, yaw=0.9)
    assert ok, "did not reach the goal"
    length = f.length
    assert t < 1.6 * length / cfg.cruise_speed + 3.0, (t, length)
    assert max_ct < 0.3, max_ct
    assert np.all(np.abs(log) <= 1.0 + 1e-12)
    # started facing 50 deg off the path: it turns while driving instead of on the spot
    ok2, _, _, log2, _ = _simulate(path, cfg, x=0.0, y=0.0, yaw=math.pi)  # facing backwards: turns on the spot first
    assert ok2 and log2[0][0] == 0.0 and abs(log2[0][2]) > 0.5


if __name__ == "__main__":
    main_for(globals(), "test_teaming_follower")
