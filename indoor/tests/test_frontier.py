"""Frontier exploration: each step on grids small enough to check by hand, then whole floors in the stand-in.

numpy + scipy; the tests that plan a path need ``drone_lab`` (the ``sim/`` project) and skip without it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from indoor_sar.frontier import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Explorer,
    FrontierCfg,
    classify,
    distance_field,
    find_frontiers,
    frontier_mask,
    nearest_cell,
    traversable_mask,
)

SQ2 = math.sqrt(2.0)

POINT_ROBOT = FrontierCfg(robot_radius_m=0.0, min_frontier_m=1.0, gain_weight=0.0, max_start_snap_m=0.0)
"""For the hand-made grids (1 m cells): a robot with no size that goes to the nearest frontier."""


def grid_from(*rows: str) -> np.ndarray:
    """An occupancy grid from text: ``#`` occupied, ``.`` free, ``?`` unknown. The first row is ``iy = 0``."""
    values = {"#": OCCUPIED, ".": FREE, "?": UNKNOWN}
    return np.array([[values[c] for c in row] for row in rows], dtype=np.int8)


def cells_of(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(y), int(x)) for y, x in np.argwhere(mask)}


# ------------------------------------------------------------------------------------------------- classify
def test_classify_uses_the_two_thresholds():
    free, occupied, unknown = classify(np.array([[-1, 0, 25, 26, 64, 65, 100]]))
    assert free.tolist() == [[False, True, True, False, False, False, False]]
    assert occupied.tolist() == [[False, False, False, False, False, True, True]]
    assert unknown.tolist() == [[True, False, False, False, False, False, False]]


# ---------------------------------------------------------------------------------------------- traversable
def test_traversable_keeps_the_robot_radius_from_walls():
    grid = grid_from(
        "#######",
        "#.....#",
        "#.....#",
        "#.....#",
        "#######",
    )
    free, occupied, _ = classify(grid)
    # radius 0: every free cell. The wall cell next to a free cell is 1 cell away, and 1 > 0.
    assert cells_of(traversable_mask(free, occupied, 0.0)) == cells_of(free)
    # radius 1 cell: the free cells that touch a wall are exactly 1 cell from it, which is not farther than 1
    assert cells_of(traversable_mask(free, occupied, 1.0)) == {(2, 2), (2, 3), (2, 4)}
    # radius 1.5 cells: the three middle cells are 2 cells from the top and bottom walls
    assert cells_of(traversable_mask(free, occupied, 1.5)) == {(2, 2), (2, 3), (2, 4)}
    assert not traversable_mask(free, occupied, 2.0).any()


def test_unknown_cells_do_not_push_the_robot_away():
    grid = grid_from(
        "???",
        "?.?",
        "???",
    )
    free, occupied, _ = classify(grid)
    assert cells_of(traversable_mask(free, occupied, 5.0)) == {(1, 1)}


# ------------------------------------------------------------------------------------------- frontier cells
def test_frontier_cells_are_the_free_cells_that_touch_unknown():
    grid = grid_from(
        "######",
        "#...??",
        "#...??",
        "#..#??",
        "######",
    )
    free, _, unknown = classify(grid)
    # (3, 2) is free but its right neighbour is a wall, and a diagonal neighbour does not count
    assert cells_of(frontier_mask(free, unknown)) == {(1, 3), (2, 3)}


def test_outside_the_grid_counts_as_unknown():
    grid = grid_from(
        "...",
        "...",
        "...",
    )
    free, _, unknown = classify(grid)
    assert cells_of(frontier_mask(free, unknown)) == cells_of(free) - {(1, 1)}


# ------------------------------------------------------------------------------------------- distance field
def test_distance_field_on_an_open_grid_is_the_octile_distance():
    d = distance_field(np.ones((5, 6), dtype=bool), (0, 0), resolution=0.5)
    assert d[0, 0] == 0.0
    assert d[0, 5] == pytest.approx(2.5)  # 5 straight steps of 0.5 m
    assert d[4, 4] == pytest.approx(4 * SQ2 * 0.5)  # 4 diagonal steps
    assert d[4, 5] == pytest.approx((4 * SQ2 + 1) * 0.5)  # 4 diagonal steps and 1 straight


def test_distance_field_does_not_cut_corners_and_marks_what_it_cannot_reach():
    free = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    d = distance_field(free, (0, 0))
    # (0, 0) -> (1, 1) is a diagonal step past the blocked (0, 1): not allowed, so it takes two straight steps
    assert d[1, 1] == pytest.approx(2.0)
    # (1, 0) -> (2, 1) is a diagonal step past (1, 1) and (2, 0), both free: allowed
    assert d[2, 1] == pytest.approx(1.0 + SQ2)
    # (1, 1) -> (2, 2) would squeeze past the blocked (1, 2), so (2, 2) is reached from (2, 1)
    assert d[2, 2] == pytest.approx(2.0 + SQ2)
    assert np.isinf(d[0, 2])  # walled in by (0, 1) and (1, 2)
    assert np.isinf(d[0, 1])  # blocked itself
    assert np.isinf(distance_field(free, (0, 1))).all()  # a start that is not traversable reaches nothing


def test_distance_field_matches_drone_lab_astar():
    planner = pytest.importorskip("drone_lab.teaming.planner")
    rng = np.random.default_rng(0)
    checked = unreachable = 0
    for _ in range(6):
        free = rng.random((30, 34)) > 0.3
        start = tuple(int(v) for v in rng.permutation(np.argwhere(free))[0])
        field = distance_field(free, start, resolution=0.25)
        cost = np.where(free, 1.0, np.inf)
        for goal in rng.permutation(np.argwhere(free))[:40]:
            goal = (int(goal[0]), int(goal[1]))
            path, path_cost, _ = planner.astar(cost, start, goal, resolution=0.25)
            if len(path) == 0:
                assert np.isinf(field[goal])
                unreachable += 1
            else:
                assert field[goal] == pytest.approx(path_cost, abs=1e-9)
                checked += 1
    assert checked > 100 and unreachable > 5  # both cases were really tested


# ------------------------------------------------------------------------------------------------- snapping
def test_nearest_cell():
    mask = grid_from("....#", ".....", ".....") == FREE
    assert nearest_cell(mask, (1, 1), 0.0) == (1, 1)
    assert nearest_cell(mask, (0, 4), 1.0) in {(0, 3), (1, 4)}
    assert nearest_cell(mask, (0, 4), 0.5) is None
    assert nearest_cell(mask, (0, 9), 5.0) is None  # outside the grid
    assert nearest_cell(np.zeros((3, 3), dtype=bool), (1, 1), 5.0) is None


# ------------------------------------------------------------------------------------------- find_frontiers
def test_find_frontiers_in_a_corridor_open_at_both_ends():
    grid = grid_from(
        "#########",
        "?.......?",
        "#########",
    )
    frontiers = find_frontiers(grid, (1, 3), 1.0, POINT_ROBOT)
    assert [f.goal for f in frontiers] == [(1, 1), (1, 7)]  # nearest first
    assert [f.distance_m for f in frontiers] == [2.0, 4.0]
    assert [f.size_m for f in frontiers] == [1.0, 1.0]
    assert [f.score for f in frontiers] == [2.0, 4.0]  # gain_weight 0: the score is the distance


def test_touching_frontier_cells_are_one_frontier_and_its_goal_is_its_middle():
    grid = grid_from(
        "#######",
        "#.....?",
        "#.....?",
        "#.....?",
        "#.....?",
        "#.....?",
        "#######",
    )
    frontiers = find_frontiers(grid, (3, 1), 1.0, POINT_ROBOT)
    assert len(frontiers) == 1
    f = frontiers[0]
    assert cells_of(np.isin(np.arange(49).reshape(7, 7), [r * 7 + 5 for r in range(1, 6)])) == {
        (int(y), int(x)) for y, x in f.cells
    }
    assert f.goal == (3, 5)
    assert f.size_m == 5.0
    assert f.distance_m == 4.0


def test_short_frontiers_are_dropped():
    grid = grid_from(
        "####?####",
        "#.......?",
        "#.......?",
        "#.......?",
        "#########",
    )
    # the hole in the top wall makes a 1-cell frontier at (1, 4); the open right side a 3-cell one
    cfg = FrontierCfg(robot_radius_m=0.0, min_frontier_m=1.0, gain_weight=0.0)
    assert sorted(f.size_m for f in find_frontiers(grid, (2, 1), 1.0, cfg)) == [1.0, 3.0]
    cfg.min_frontier_m = 2.0
    assert [f.size_m for f in find_frontiers(grid, (2, 1), 1.0, cfg)] == [3.0]


def test_gain_weight_trades_driving_for_frontier_length():
    grid = grid_from(
        "####?######",
        "#.........?",
        "#.........?",
        "#.........?",
        "###########",
    )
    # from (2, 3): the 1-cell frontier at (1, 4) is 1.41 m away, the 3-cell one on the right 6 m away
    near_first = find_frontiers(grid, (2, 3), 1.0, FrontierCfg(robot_radius_m=0.0, min_frontier_m=1.0, gain_weight=0.0))
    assert [f.size_m for f in near_first] == [1.0, 3.0]
    assert near_first[0].distance_m == pytest.approx(SQ2)
    assert near_first[1].distance_m == pytest.approx(6.0)
    # 3 m of driving per metre of frontier: scores 1.41 - 3 = -1.59 and 6 - 9 = -3, so the long one wins
    big_first = find_frontiers(grid, (2, 3), 1.0, FrontierCfg(robot_radius_m=0.0, min_frontier_m=1.0, gain_weight=3.0))
    assert [f.size_m for f in big_first] == [3.0, 1.0]
    assert big_first[0].score == pytest.approx(6.0 - 9.0)


def _two_rooms(gap_cells: int) -> np.ndarray:
    """Two known rooms (0.1 m cells) joined by a gap in the wall between them; the far room is open to the unknown."""
    grid = np.full((40, 80), FREE, dtype=np.int8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = OCCUPIED
    grid[:, 39:41] = OCCUPIED
    grid[20 - gap_cells // 2 : 20 - gap_cells // 2 + gap_cells, 39:41] = FREE
    grid[:, 70:] = UNKNOWN
    return grid


def test_a_gap_narrower_than_the_robot_hides_the_frontier_behind_it():
    cfg = FrontierCfg(robot_radius_m=0.35, min_frontier_m=0.5)
    # the robot needs more than 2 * 0.35 m between the two sides of the gap
    assert find_frontiers(_two_rooms(gap_cells=6), (20, 10), 0.1, cfg) == []
    wide = find_frontiers(_two_rooms(gap_cells=12), (20, 10), 0.1, cfg)
    assert len(wide) == 1
    # the frontier is the last known column, wall to wall (rows 0 and 39 are walls); its goal is in the middle
    assert sorted((int(y), int(x)) for y, x in wide[0].cells) == [(y, 69) for y in range(1, 39)]
    assert wide[0].size_m == pytest.approx(3.8)
    assert wide[0].goal in {(19, 69), (20, 69)}


def test_a_wide_opening_with_a_tight_approach_is_not_taken_for_a_short_frontier():
    # A 0.9 m corridor (rows 11..19) leaves a room and is known up to column 44. The robot's centre fits only on
    # its 3 middle rows (0.35 m radius), fewer than the 5 cells of min_frontier_m, but the opening is 9 cells wide.
    grid = np.full((30, 60), OCCUPIED, dtype=np.int8)
    grid[5:25, 1:30] = FREE
    grid[11:20, 30:45] = FREE
    grid[:, 45:] = UNKNOWN
    frontiers = find_frontiers(grid, (15, 10), 0.1, FrontierCfg(robot_radius_m=0.35, min_frontier_m=0.5))
    assert len(frontiers) == 1
    assert sorted((int(y), int(x)) for y, x in frontiers[0].cells) == [(y, 44) for y in range(11, 20)]
    assert frontiers[0].size_m == pytest.approx(0.9)
    assert frontiers[0].goal == (15, 44)
    assert frontiers[0].distance_m == pytest.approx(3.4)  # 34 cells straight along row 15


def test_a_closed_room_has_nothing_to_explore():
    grid = grid_from(
        "#####",
        "#...#",
        "#...#",
        "#####",
    )
    assert find_frontiers(grid, (1, 1), 1.0, POINT_ROBOT) == []
    assert Explorer(POINT_ROBOT).update(grid, (1, 1), 1.0) is None


def test_a_robot_off_the_traversable_cells_is_snapped_or_refused():
    grid = grid_from(
        "#########",
        "#.......?",
        "#.......?",
        "#.......?",
        "#########",
    )
    cfg = FrontierCfg(robot_radius_m=1.0, min_frontier_m=1.0, gain_weight=0.0, max_start_snap_m=1.0)
    # with a 1 m radius only row 2, columns 2..7, is traversable; the robot at (1, 3) is snapped to (2, 3)
    frontiers = find_frontiers(grid, (1, 3), 1.0, cfg)
    assert [f.goal for f in frontiers] == [(2, 7)]
    assert frontiers[0].distance_m == 4.0  # from (2, 3), not from (1, 3)
    cfg.max_start_snap_m = 0.5
    assert find_frontiers(grid, (1, 3), 1.0, cfg) == []
    assert find_frontiers(grid, (9, 9), 1.0, cfg) == []  # outside the grid


# ------------------------------------------------------------------------------------------------- explorer
CORRIDOR = grid_from(
    "#########",
    "?.......?",
    "#########",
)


def test_explorer_keeps_its_target_until_another_is_better_by_the_margin():
    cfg = FrontierCfg(robot_radius_m=0.0, min_frontier_m=1.0, gain_weight=0.0, switch_margin_m=3.0)
    explorer = Explorer(cfg)
    assert explorer.update(CORRIDOR, (1, 3), 1.0).goal == (1, 1)
    # from (1, 5) the right end is 2 m away and the left end 4 m: better by 2 m, less than the 3 m margin
    assert find_frontiers(CORRIDOR, (1, 5), 1.0, cfg)[0].goal == (1, 7)
    assert explorer.update(CORRIDOR, (1, 5), 1.0).goal == (1, 1)
    assert explorer.switches == 0
    # from (1, 6): 1 m against 5 m, better by 4 m
    assert explorer.update(CORRIDOR, (1, 6), 1.0).goal == (1, 7)
    assert explorer.switches == 1


def test_explorer_takes_the_best_frontier_when_its_target_is_gone():
    cfg = FrontierCfg(robot_radius_m=0.0, min_frontier_m=1.0, gain_weight=0.0, switch_margin_m=100.0)
    explorer = Explorer(cfg)
    assert explorer.update(CORRIDOR, (1, 3), 1.0).goal == (1, 1)
    seen = CORRIDOR.copy()
    seen[1, 0] = OCCUPIED  # the left end turned out to be a wall: that frontier no longer exists
    assert explorer.update(seen, (1, 2), 1.0).goal == (1, 7)
    assert explorer.switches == 0  # not a change of mind


def test_explorer_gives_up_on_a_goal_and_then_runs_out():
    explorer = Explorer(POINT_ROBOT)
    assert explorer.update(CORRIDOR, (1, 3), 1.0).goal == (1, 1)
    explorer.give_up()
    assert explorer.abandoned == [(1, 1)]
    assert explorer.update(CORRIDOR, (1, 3), 1.0).goal == (1, 7)
    explorer.give_up()
    assert explorer.update(CORRIDOR, (1, 3), 1.0) is None
    assert explorer.target is None


# ------------------------------------------------------------------------------------------------- stand-in
def test_sense_marks_what_the_sensor_sees_and_nothing_behind_it():
    standin = pytest.importorskip("indoor_sar.standin")
    occupied = np.zeros((21, 41), dtype=bool)
    occupied[:, 25] = True  # a wall across the whole grid
    known = np.full(occupied.shape, UNKNOWN, dtype=np.int8)
    standin.sense(occupied, known, (10, 10), range_cells=8.0)
    assert (known[:, 26:] == UNKNOWN).all()  # behind the wall, and beyond the range
    assert (known[:, 25] == UNKNOWN).all()  # the wall is 15 cells away: out of range
    assert known[10, 18] == FREE and known[10, 19] == UNKNOWN  # 8 cells of range along the row
    assert known[10, 2] == FREE and known[10, 1] == UNKNOWN
    standin.sense(occupied, known, (10, 20), range_cells=8.0)
    assert known[10, 25] == OCCUPIED
    assert (known[:, 26:] == UNKNOWN).all()
    assert not (known[:, 25] == FREE).any()
    assert not ((known == FREE) & occupied).any()


def test_make_floor_is_repeatable_and_leaves_the_start_free():
    standin = pytest.importorskip("indoor_sar.standin")
    occupied, start = standin.make_floor(3)
    again, start_again = standin.make_floor(3)
    assert (occupied == again).all() and start == start_again
    assert occupied.shape == (160, 240)
    other, _ = standin.make_floor(4)
    assert (occupied != other).any()
    yy, xx = np.mgrid[0:160, 0:240]
    near_start = np.hypot(yy - start[0], xx - start[1]) <= 3.5  # the robot's radius in cells
    assert not occupied[near_start].any()


# Floor 106 with gain_weight 1 once stopped at 95 %: a 2.7 m frontier with 3 reachable cells was dropped as short.
@pytest.mark.parametrize(("seed", "gain_weight"), [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (106, 1.0)])
def test_exploration_maps_everything_the_robot_can_reach(seed, gain_weight):
    standin = pytest.importorskip("indoor_sar.standin")
    occupied, start = standin.make_floor(seed)
    run = standin.explore(occupied, start, 0.1, FrontierCfg(gain_weight=gain_weight))
    assert run.finished
    assert run.gave_up == []
    score = standin.coverage(occupied, run.known, start, 0.1)
    assert score["wrong_cells"] == 0
    assert score["coverage"] >= 0.995
    # the robot was never on a wall or on rubble, and never jumped: every move is to a neighbour cell
    assert not occupied[run.trail[:, 0], run.trail[:, 1]].any()
    assert run.snaps == 0
    assert np.abs(np.diff(run.trail, axis=0)).max() == 1
    # a guard against wandering, not a claim of efficiency: under 1 m driven per m^2 of floor it could stand on
    # (these four floors: 45 to 125 m driven for 158 to 268 m^2)
    assert run.length_m < score["reachable_cells"] * 0.1 * 0.1


def test_a_sealed_room_stays_unknown_and_the_exploration_still_ends():
    standin = pytest.importorskip("indoor_sar.standin")
    occupied = np.zeros((60, 120), dtype=bool)
    occupied[:2, :] = occupied[-2:, :] = occupied[:, :2] = occupied[:, -2:] = True
    occupied[:, 60:62] = True  # a wall with no door
    run = standin.explore(occupied, (30, 20), 0.1)
    assert run.finished
    assert (run.known[2:-2, 62:-2] == UNKNOWN).all()
    score = standin.coverage(occupied, run.known, (30, 20), 0.1)
    assert score["coverage"] == 1.0  # of what it could reach
    assert score["wrong_cells"] == 0
