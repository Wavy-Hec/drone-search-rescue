"""A 2-D stand-in for the simulator, to test the exploration without Isaac Sim, ROS 2 or SLAM.

* :func:`make_floor`: the floor plan of a damaged building: rooms, doors and rubble, as a grid of occupied cells.
* :func:`sense`: a 360-degree range sensor. It marks what it sees in the robot's map.
* :func:`explore`: the whole loop: look, pick a frontier, plan to it with drone_lab's A*, drive a few cells, look
  again, until no frontier is left.
* :func:`coverage`: how much of the floor the robot could stand on ended up in its map.

The robot knows its own cell exactly here, and the sensor makes no mistakes. That is on purpose: this checks the
exploration logic alone. Wrong poses and a noisy map are what SLAM brings, in the simulator.

Needs ``drone_lab`` (the ``sim/`` project) on the path for the A*; numpy + scipy otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from drone_lab.teaming.planner import astar

from .frontier import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Explorer,
    Frontier,
    FrontierCfg,
    classify,
    distance_field,
    nearest_cell,
    traversable_mask,
)

__all__ = ["ExplorationRun", "coverage", "explore", "make_floor", "sense"]


# --------------------------------------------------------------------------------------------- floor plan
def make_floor(
    seed: int,
    size_m: tuple[float, float] = (24.0, 16.0),
    resolution: float = 0.1,
    *,
    min_room_m: float = 3.5,
    door_m: float = 1.2,
    wall_m: float = 0.2,
    rubble: int = 14,
    rubble_radius_m: tuple[float, float] = (0.2, 0.7),
) -> tuple[np.ndarray, tuple[int, int]]:
    """A random floor of a damaged building.

    The floor is cut into rooms by walls, each with one door; then piles of rubble are dropped on it. Rubble may
    land in a doorway and close a room off: the robot cannot explore that room, and :func:`coverage` does not
    count it.

    Returns:
        ``(occupied, start)``: the ``(ny, nx)`` mask of walls and rubble, and the robot's start cell ``(iy, ix)``,
        1 m inside the lower-left corner. No rubble lands within 1.5 m of the start.
    """
    rng = np.random.default_rng(seed)
    nx, ny = round(size_m[0] / resolution), round(size_m[1] / resolution)
    wall = max(1, round(wall_m / resolution))
    door = max(1, round(door_m / resolution))
    room = max(1, round(min_room_m / resolution))
    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[:wall, :] = occupied[-wall:, :] = occupied[:, :wall] = occupied[:, -wall:] = True
    doorways = np.zeros((ny, nx), dtype=bool)  # no wall is built here, so that a new wall never closes a door

    def split(y0: int, y1: int, x0: int, x1: int) -> None:
        """Cut the room ``[y0, y1) x [x0, x1)`` in two with a wall across its longer side, then cut the halves."""
        h, w = y1 - y0, x1 - x0
        can_cut_x = w >= 2 * room + wall
        can_cut_y = h >= 2 * room + wall
        if not (can_cut_x or can_cut_y):
            return
        if can_cut_x and (w >= h or not can_cut_y):
            x = int(rng.integers(x0 + room, x1 - room - wall + 1))  # the wall is columns [x, x + wall)
            d = int(rng.integers(y0, y1 - door + 1))  # the door is rows [d, d + door)
            doorways[d : d + door, x - door : x + wall + door] = True
            occupied[y0:y1, x : x + wall] |= ~doorways[y0:y1, x : x + wall]
            split(y0, y1, x0, x)
            split(y0, y1, x + wall, x1)
        else:
            y = int(rng.integers(y0 + room, y1 - room - wall + 1))
            d = int(rng.integers(x0, x1 - door + 1))
            doorways[y - door : y + wall + door, d : d + door] = True
            occupied[y : y + wall, x0:x1] |= ~doorways[y : y + wall, x0:x1]
            split(y0, y, x0, x1)
            split(y + wall, y1, x0, x1)

    split(wall, ny - wall, wall, nx - wall)

    start = (wall + round(1.0 / resolution), wall + round(1.0 / resolution))
    yy, xx = np.mgrid[0:ny, 0:nx]
    for _ in range(rubble):
        cy, cx = rng.uniform(wall, ny - wall), rng.uniform(wall, nx - wall)
        radius = rng.uniform(*rubble_radius_m) / resolution
        if math.hypot(cy - start[0], cx - start[1]) < radius + 1.5 / resolution:
            continue
        occupied |= (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 <= radius**2
    return occupied, start


# ------------------------------------------------------------------------------------------------- sensor
def sense(
    occupied: np.ndarray, known: np.ndarray, cell: tuple[int, int], range_cells: float, n_rays: int = 720
) -> None:
    """Update the map ``known`` in place with what a range sensor at the centre of ``cell`` sees.

    Each ray is walked in steps of half a cell. The cells it crosses before the first occupied cell become FREE,
    and that occupied cell becomes OCCUPIED. Nothing behind it is touched, and neither is anything beyond
    ``range_cells``.
    """
    ny, nx = occupied.shape
    angles = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    steps = np.arange(0.0, range_cells + 0.5, 0.5)
    iy = np.floor(cell[0] + 0.5 + np.outer(np.sin(angles), steps)).astype(int)
    ix = np.floor(cell[1] + 0.5 + np.outer(np.cos(angles), steps)).astype(int)
    inside = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    hit = np.zeros(inside.shape, dtype=bool)
    hit[inside] = occupied[iy[inside], ix[inside]]
    blocked = hit | ~inside
    ended = np.cumsum(blocked, axis=1) > 0  # this sample is the end of the ray, or behind it
    first = blocked & (np.cumsum(blocked, axis=1) == 1)  # this sample is the end of the ray
    known[iy[~ended], ix[~ended]] = FREE
    known[iy[first & hit], ix[first & hit]] = OCCUPIED


# ---------------------------------------------------------------------------------------------- the loop
@dataclass
class ExplorationRun:
    """What :func:`explore` did."""

    known: np.ndarray
    """The robot's map at the end (-1 unknown, 0 free, 100 occupied)."""
    trail: np.ndarray
    """Every cell the robot was on, in order, ``(N, 2)``."""
    length_m: float
    """Distance driven."""
    decisions: int
    """How many times a frontier was picked."""
    finished: bool
    """True when it stopped because no frontier was left; False when it ran out of ``max_decisions``."""
    target_changes: int = 0
    """How many times the robot left a frontier that was still on the map for a better one."""
    snaps: int = 0
    """How many times the robot stood closer to a newly seen obstacle than its radius and was moved off."""
    gave_up: list[tuple[int, int]] = field(default_factory=list)
    """Goals the A* found no path to (none are expected: every frontier cell is reachable by construction)."""


def explore(
    occupied: np.ndarray,
    start: tuple[int, int],
    resolution: float,
    cfg: FrontierCfg | None = None,
    *,
    sensor_range_m: float = 6.0,
    replan_cells: int = 5,
    max_decisions: int = 5000,
    on_decision: Callable[[np.ndarray, tuple[int, int], Frontier, np.ndarray], None] | None = None,
) -> ExplorationRun:
    """Explore the floor ``occupied`` from ``start`` until no frontier is left.

    One decision: ask the :class:`~indoor_sar.frontier.Explorer` for a frontier, plan to its goal with A* on the
    robot's own map, drive at most ``replan_cells`` cells of that path, looking around after every cell. The
    robot stops early when its map now says that the next cell is too close to an obstacle.

    ``on_decision(known, robot_cell, frontier, path)`` is called after every plan (for figures).
    """
    cfg = cfg or FrontierCfg()
    radius_cells = cfg.robot_radius_m / resolution
    range_cells = sensor_range_m / resolution
    known = np.full(occupied.shape, UNKNOWN, dtype=np.int8)
    robot = (int(start[0]), int(start[1]))
    sense(occupied, known, robot, range_cells)
    explorer = Explorer(cfg)
    run = ExplorationRun(known=known, trail=np.zeros((0, 2), dtype=int), length_m=0.0, decisions=0, finished=False)
    trail = [robot]

    def move_to(cell: tuple[int, int]) -> tuple[int, int]:
        run.length_m += math.dist(robot, cell) * resolution
        trail.append(cell)
        sense(occupied, known, cell, range_cells)
        return cell

    def traversable() -> np.ndarray:
        free, occupied_known, _ = classify(known, cfg)
        return traversable_mask(free, occupied_known, radius_cells)

    for _ in range(max_decisions):
        target = explorer.update(known, robot, resolution)
        if target is None:
            run.finished = True
            break
        run.decisions += 1

        mask = traversable()
        at = nearest_cell(mask, robot, cfg.max_start_snap_m / resolution)
        path, _, _ = astar(np.where(mask, 1.0, np.inf), at, target.goal, resolution=resolution)
        if len(path) == 0:
            run.gave_up.append(target.goal)
            explorer.give_up()
            continue
        if on_decision is not None:
            on_decision(known, robot, target, path)
        if at != robot:
            run.snaps += 1
            robot = move_to(at)
        for cell in path[1 : 1 + replan_cells]:
            cell = (int(cell[0]), int(cell[1]))
            if not traversable()[cell]:
                break
            robot = move_to(cell)
    run.trail = np.asarray(trail, dtype=int)
    run.target_changes = explorer.switches
    return run


# ------------------------------------------------------------------------------------------------ scoring
def coverage(
    occupied: np.ndarray, known: np.ndarray, start: tuple[int, int], resolution: float, cfg: FrontierCfg | None = None
) -> dict:
    """Score the robot's map ``known`` against the true floor ``occupied``.

    Returns a dict with:

    * ``reachable_cells``: the cells the robot could stand on, starting from ``start``, if it knew the whole
      floor (free, farther than its radius from every wall and rubble pile, and connected to the start).
    * ``coverage``: the share of those cells that the robot's map shows as free.
    * ``wrong_cells``: cells the map shows as free that are occupied, or as occupied that are free.
    """
    cfg = cfg or FrontierCfg()
    reachable = np.isfinite(
        distance_field(traversable_mask(~occupied, occupied, cfg.robot_radius_m / resolution), start)
    )
    free, occupied_known, _ = classify(known, cfg)
    return {
        "reachable_cells": int(reachable.sum()),
        "coverage": float((free & reachable).sum() / max(int(reachable.sum()), 1)),
        "wrong_cells": int((free & occupied).sum() + (occupied_known & ~occupied).sum()),
    }
