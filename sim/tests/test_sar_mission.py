"""SAR mission plan, survey geometry, ground truth for the ground station and obstacle clearance (numpy only)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from _runner import main_for

from drone_lab.paths import cumulative_length
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


if __name__ == "__main__":
    main_for(globals(), "test_sar_mission")
