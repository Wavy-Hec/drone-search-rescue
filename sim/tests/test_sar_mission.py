"""SAR mission plan, survey geometry, ground truth for the ground station and obstacle clearance (numpy only)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from _runner import main_for

from drone_lab.paths import cumulative_length
from drone_lab.sar.chase_view import FollowCamera, ground_below, rim_points, skirt_mesh
from drone_lab.sar.mission import PHASES, ClearanceMonitor, gt_objects, plan_mission, survey_geometry

AREA = (-20.0, 20.0, -20.0, 20.0)
HOME = (-23.0, -18.0, 0.049)


def test_plan_takes_off_surveys_and_lands_on_the_pad():
    plan = plan_mission(AREA, HOME, 7.515, 5.0)
    w = plan.waypoints
    np.testing.assert_allclose(w[0], [-23.0, -18.0, 0.549])  # start hover above the pad
    np.testing.assert_allclose(w[1], [-23.0, -18.0, 7.515])  # straight up
    np.testing.assert_allclose(w[-1], [-23.0, -18.0, 0.099])  # touchdown point just above the pad
    first, last = plan.survey_waypoints
    survey = w[first : last + 1]
    np.testing.assert_allclose(survey[:, 2], 7.515)
    lines = np.unique(np.round(survey[np.abs(np.abs(survey[:, 0]) - 20.0) < 1e-9, 1], 6))
    np.testing.assert_allclose(lines, -17.5 + 5.0 * np.arange(8))  # 8 lines, on the 0.25 m lattice
    assert plan.phase_ends[-1] == plan.length
    assert abs(plan.length - cumulative_length(w)[-1]) < 1e-9
    # 8 x 40 m lines + 7 half circles of radius 2.5 m
    survey_len = plan.phase_ends[2] - plan.phase_ends[1]
    assert abs(survey_len - (320.0 + 7 * math.pi * 2.5)) < 0.5


def test_phase_speed_and_waypoint_along_the_path():
    plan = plan_mission(AREA, HOME, 7.515, 5.0, speed=1.5, climb_speed=1.0, touchdown_speed=0.3)
    starts = (0.0, *plan.phase_ends[:-1])
    for name, a, b in zip(PHASES, starts, plan.phase_ends, strict=True):
        mid = 0.5 * (a + b)
        assert plan.phase(mid) == name
    assert plan.speed(1.0) == 1.0 and plan.speed(100.0) == 1.5 and plan.speed(plan.length - 0.01) == 0.3
    assert plan.phase(plan.length + 5.0) == "touchdown"
    assert plan.waypoint_index(0.0) == 1  # heading for the top of the climb
    assert plan.waypoint_index(plan.length) == len(plan.waypoints) - 1
    d = plan.to_dict()
    assert list(d["phases"]) == list(PHASES) and d["waypoints"] == len(plan.waypoints)


def test_survey_geometry_numbers_of_the_default_plan():
    g = survey_geometry(7.515, 5.0, 1.25, 480.0, (640, 480), min_clearance=3.9)
    assert abs(g["footprint_across_m"] - 9.887) < 1e-3 and abs(g["footprint_along_m"] - 7.415) < 1e-3
    assert abs(g["gsd_cm_per_px"] - 1.545) < 1e-3
    assert 49.0 < g["side_overlap_pct"] < 50.0
    assert g["min_swath_m"] > 5.0  # no gaps in the photo map even over the highest hill
    assert g["forward_overlap_pct"] > 98.0


def _obj(name, category, center, half, labeled=True, ground_z=0.0, yaw=0.0, pose="upright"):
    c, h = np.asarray(center, float), np.asarray(half, float)
    return SimpleNamespace(
        name=name,
        category=category,
        labeled=labeled,
        center=tuple(c),
        half_extents=tuple(h),
        aabb_min=tuple(c - h),
        aabb_max=tuple(c + h),
        ground_z=ground_z,
        yaw=yaw,
        pose=pose,
    )


def test_ground_truth_for_the_ground_station():
    objs = [
        _obj("person_00", "person", (1.0, 2.0, 0.2), (0.9, 0.3, 0.2), ground_z=0.0, pose="lying_face_up"),
        _obj("tree_00", "tree", (5.0, 5.0, 2.5), (1.0, 1.0, 2.5), labeled=False),
    ]
    gt = gt_objects(objs)
    assert [g["id"] for g in gt] == ["person_00"]  # trees are unlabelled occluders
    assert gt[0]["class"] == "person" and gt[0]["pos"] == [1.0, 2.0, 0.0] and gt[0]["top_z"] == 0.4
    assert gt[0]["size"] == [1.8, 0.6, 0.4]


def test_clearance_counts_flight_phases_only():
    objs = [
        _obj("crate_00", "crate", (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)),
        _obj("pad", "landing_pad", (9, 9, 0), (1, 1, 0)),
    ]
    mon = ClearanceMonitor(objs)
    hits = np.array([[0.0, 0.0, 1.0], [np.inf, np.inf, np.inf]])
    mon.update(0.0, (9.0, 9.0, 0.3), "takeoff", hits)  # near the pad on purpose: ignored
    mon.update(1.0, (0.0, 0.0, 5.0), "survey", hits)
    mon.update(2.0, (3.0, 0.0, 1.0), "survey")
    s = mon.summary()
    assert abs(s["terrain"]["distance_m"] - 4.0) < 1e-9
    assert abs(s["objects"]["distance_m"] - 2.5) < 1e-9 and s["objects"]["name"] == "crate_00"
    assert s["closest_m"] == 2.5


def test_follow_camera_frames_the_drone_and_its_scan_and_glides():
    cam = FollowCamera(offset=(-4.0, -6.0, 5.0), look_below=0.4, tau_xy=0.5, tau_z=1.0)
    eye, target = cam.update((10.0, 0.0, 7.5), ground_z=0.5, dt=0.1)
    np.testing.assert_allclose(target, [10.0, 0.0, 7.5 - 0.4 * 7.0])  # 40 % of the way down to the ground
    np.testing.assert_allclose(eye, target + [-4.0, -6.0, 5.0])
    _, target = cam.update((11.0, 0.0, 7.5), ground_z=0.5, dt=0.1)  # a 1 m jump is smoothed, not copied
    np.testing.assert_allclose(target[0], 10.0 + (1.0 - math.exp(-0.1 / 0.5)))
    _, high = FollowCamera().update((0.0, 0.0, 3.0), ground_z=None, dt=0.1)
    np.testing.assert_allclose(high, [0.0, 0.0, 3.0])  # no scan yet: look at the drone


def test_follow_camera_keeps_its_eye_above_the_site():
    cam = FollowCamera(offset=(-6.0, -8.0, 5.0), bounds=(-25.0, 25.0, -25.0, 25.0))
    eye, target = cam.update((-23.0, -18.0, 0.5), ground_z=0.05, dt=0.1)
    assert eye[0] == -25.0 and eye[1] == -25.0  # pulled in over the terrain instead of past its edge
    assert target[0] == -23.0 and target[1] == -18.0


def test_ground_skirt_is_a_frame_around_the_site():
    points, counts, indices = skirt_mesh((-26.0, 26.0, -26.0, 26.0), 100.0, -0.01)
    assert points.shape == (8, 3) and counts == [4] * 4 and len(indices) == 16
    np.testing.assert_allclose(points[:, 2], -0.01)
    assert set(map(tuple, points[4:, :2].tolist())) == {(-26.0, -26.0), (26.0, -26.0), (26.0, 26.0), (-26.0, 26.0)}
    for q in np.reshape(indices, (4, 4)):  # every quad faces up (counter-clockwise seen from above)
        x, y = points[q, 0], points[q, 1]
        assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0.0
    rim = rim_points((-26.0, 26.0, -26.0, 26.0), inset=0.1, n=5)
    assert rim.shape == (20, 2) and np.all(np.abs(rim) <= 25.9 + 1e-9)
    assert np.all(np.isclose(np.abs(rim), 25.9).any(axis=1))  # every point lies on the inset edge


def test_ground_below_is_the_median_scan_hit():
    hits = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [np.inf] * 3, [0.0, 0.0, 2.0]])
    assert ground_below(hits) == 2.0 and ground_below(np.full((2, 3), np.inf)) is None


if __name__ == "__main__":
    main_for(globals(), "test_sar_mission")
