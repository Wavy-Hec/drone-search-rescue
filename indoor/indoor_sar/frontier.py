"""Frontier exploration on an occupancy grid: where should the robot drive next, to see more of the building?

A **frontier** is the border between what the map knows is free and what it has not seen yet (Yamauchi, 1997).
Driving to a frontier puts unseen space in front of the sensors; when no frontier is left, everything the robot
can reach has been seen and the exploration is over.

One call of :func:`find_frontiers` does six things, each a function of its own below:

1. :func:`classify`: every cell is free, occupied or unknown.
2. :func:`traversable_mask`: the free cells where the robot's centre may be, i.e. farther than the robot's radius
   from every occupied cell.
3. :func:`distance_field`: the length of the shortest path from the robot to every traversable cell, through
   known free space only (Dijkstra on the graph of drone_lab's A*: 8 neighbours, no cutting of corners).
4. :func:`frontier_mask`: the free cells that touch an unknown cell.
5. Touching frontier cells are grouped into frontiers. A frontier is dropped when it is shorter than
   ``min_frontier_m``, or when the robot can reach none of its cells.
6. Each frontier gets a goal, the cell nearest to its middle among those the robot can reach, and a score,
   ``path length - gain_weight * frontier length``; lowest first.

A frontier's length counts all of its cells, not only the ones the robot can stand on. Next to walls and rubble
the robot's centre can use only a narrow strip of an opening, and a wide opening must not be taken for a short
frontier because of that.

:class:`Explorer` keeps what has to last from one call to the next: the frontier the robot is driving to, so that
it does not change its mind at every step, and the goals it gave up on.

The grid is a ``nav_msgs/OccupancyGrid`` as RTAB-Map and Nav2 publish it: ``(ny, nx)`` integers, ``-1`` unknown,
``0..100`` the chance in percent that the cell is occupied. Cells are ``(iy, ix)``. numpy + scipy only.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

__all__ = [
    "FREE",
    "OCCUPIED",
    "UNKNOWN",
    "Explorer",
    "Frontier",
    "FrontierCfg",
    "classify",
    "distance_field",
    "find_frontiers",
    "frontier_mask",
    "nearest_cell",
    "traversable_mask",
]

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

SQRT2 = math.sqrt(2.0)


@dataclass
class FrontierCfg:
    """Settings of the frontier search. Lengths are metres."""

    robot_radius_m: float = 0.35
    """The robot's centre keeps this far from every occupied cell."""
    min_frontier_m: float = 0.5
    """Frontiers shorter than this are ignored: gaps the robot could not use, and noise in the map."""
    gain_weight: float = 0.0
    """How many metres of extra driving one metre of frontier is worth. 0 = always the nearest frontier, which
    drove the least on the stand-in floors (12 floors: 90 m on average, against 102 m with 1 and 114 m with 3)."""
    free_max: int = 25
    """Cells with an occupancy of 0..free_max are free."""
    occupied_min: int = 65
    """Cells with an occupancy of occupied_min..100 are occupied. In between: not free, and not unknown either."""
    max_start_snap_m: float = 0.5
    """A robot standing on a cell that is not traversable (it touches a wall, or the map is a little off) is
    moved to the nearest traversable cell, if there is one this close."""
    switch_margin_m: float = 1.0
    """:class:`Explorer` leaves the frontier it is driving to only for one whose score is better by this much."""
    same_frontier_m: float = 1.0
    """:class:`Explorer` takes a frontier for the one it was driving to when it has a cell this close to the old
    goal."""
    give_up_radius_m: float = 1.0
    """After :meth:`Explorer.give_up`, frontiers whose goal is this close to the abandoned goal are skipped."""


@dataclass
class Frontier:
    """One connected piece of the border between known free space and unknown space."""

    cells: np.ndarray
    """Its cells, ``(K, 2)`` ints ``(iy, ix)``: free cells that touch unknown space. The robot can reach at least
    one of them, not necessarily all."""
    goal: tuple[int, int]
    """The cell to drive to: of the frontier's cells that the robot can reach, the nearest to its middle."""
    size_m: float
    """Its length: number of cells times the cell size."""
    distance_m: float
    """Length of the shortest known path from the robot to :attr:`goal`."""
    score: float = field(default=math.inf)
    """``distance_m - gain_weight * size_m``. Lower is better."""

    def gap_cells(self, cell: tuple[int, int]) -> float:
        """Distance from ``cell`` to the nearest cell of this frontier, in cells."""
        d = self.cells - np.asarray(cell)
        return float(np.sqrt((d * d).sum(axis=1).min()))


# ------------------------------------------------------------------------------------------------ the steps
def classify(grid: np.ndarray, cfg: FrontierCfg | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an occupancy grid into three masks: ``(free, occupied, unknown)``."""
    cfg = cfg or FrontierCfg()
    g = np.asarray(grid)
    unknown = g < 0
    occupied = g >= cfg.occupied_min
    free = (g >= 0) & (g <= cfg.free_max)
    return free, occupied, unknown


def traversable_mask(free: np.ndarray, occupied: np.ndarray, radius_cells: float) -> np.ndarray:
    """Free cells whose centre is farther than ``radius_cells`` from the centre of every occupied cell.

    These are the cells the robot's centre may be on. Unknown cells do not push the robot away: a frontier cell
    touches unknown space by definition, and the robot has to be able to stand on it.
    """
    if not occupied.any():
        return free.copy()
    clearance = ndimage.distance_transform_edt(~occupied)  # in cells, to the nearest occupied cell
    return free & (clearance > radius_cells)


def frontier_mask(candidates: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """The ``candidates`` cells with an unknown cell above, below, left or right of them.

    Outside the grid counts as unknown, because a SLAM map grows: a free cell on the edge of the grid is the edge
    of what has been seen.
    """
    u = np.pad(unknown, 1, constant_values=True)
    touches_unknown = u[:-2, 1:-1] | u[2:, 1:-1] | u[1:-1, :-2] | u[1:-1, 2:]
    return candidates & touches_unknown


def distance_field(traversable: np.ndarray, start: tuple[int, int], resolution: float = 1.0) -> np.ndarray:
    """Length of the shortest path from ``start`` to every cell, moving over ``traversable`` cells only.

    Dijkstra with the moves of ``drone_lab.teaming.planner.astar``: a step to one of the 8 neighbours costs
    ``resolution`` (straight) or ``sqrt(2) * resolution`` (diagonal), and a diagonal step is only allowed when
    both cells it squeezes past are traversable. So ``distance_field(t, s)[g]`` is the cost that A* finds from
    ``s`` to ``g`` on a grid that costs 1 per metre (the tests check this).

    Returns:
        ``(ny, nx)`` floats in the unit of ``resolution``; ``inf`` where no path exists, and everywhere when
        ``start`` is not traversable.
    """
    ny, nx = traversable.shape
    # padded flat layout, as in the A*: a ring of blocked cells around the grid removes every bounds check
    w = nx + 2
    padded = np.zeros((ny + 2, w), dtype=bool)
    padded[1:-1, 1:-1] = traversable
    free = padded.ravel().tolist()
    dist = [math.inf] * len(free)
    s = (int(start[0]) + 1) * w + (int(start[1]) + 1)
    if free[s]:
        straight = float(resolution)
        diagonal = SQRT2 * float(resolution)
        axis_moves = (1, -1, w, -w)
        # diagonal move, and the two straight neighbours it squeezes past
        diag_moves = ((w + 1, w, 1), (w - 1, w, -1), (-w + 1, -w, 1), (-w - 1, -w, -1))
        dist[s] = 0.0
        heap = [(0.0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue  # an old entry: u was already reached by a shorter path
            nd = d + straight
            for move in axis_moves:
                v = u + move
                if free[v] and nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
            nd = d + diagonal
            for move, a1, a2 in diag_moves:
                v = u + move
                if free[v] and free[u + a1] and free[u + a2] and nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
    return np.asarray(dist).reshape(ny + 2, w)[1:-1, 1:-1]


def nearest_cell(mask: np.ndarray, cell: tuple[int, int], max_cells: float) -> tuple[int, int] | None:
    """``cell`` itself if ``mask`` is set there; otherwise the nearest set cell within ``max_cells``, or None."""
    iy, ix = int(cell[0]), int(cell[1])
    ny, nx = mask.shape
    if not (0 <= iy < ny and 0 <= ix < nx):
        return None
    if mask[iy, ix]:
        return iy, ix
    if not mask.any():
        return None
    dist, (near_y, near_x) = ndimage.distance_transform_edt(~mask, return_indices=True)
    if dist[iy, ix] > max_cells:
        return None
    return int(near_y[iy, ix]), int(near_x[iy, ix])


# ------------------------------------------------------------------------------------------------ frontiers
def find_frontiers(
    grid: np.ndarray, robot_cell: tuple[int, int], resolution: float, cfg: FrontierCfg | None = None
) -> list[Frontier]:
    """The frontiers the robot can reach on ``grid``, best first (see the module docstring for the steps).

    Args:
        grid: Occupancy grid ``(ny, nx)``: -1 unknown, 0..100 occupancy in percent.
        robot_cell: The robot's cell ``(iy, ix)``.
        resolution: Cell size [m].

    Returns:
        A list sorted by :attr:`Frontier.score`. It is empty when nothing is left to explore, and also when the
        robot is outside the grid or farther than ``max_start_snap_m`` from every traversable cell.
    """
    cfg = cfg or FrontierCfg()
    free, occupied, unknown = classify(grid, cfg)
    traversable = traversable_mask(free, occupied, cfg.robot_radius_m / resolution)
    start = nearest_cell(traversable, robot_cell, cfg.max_start_snap_m / resolution)
    if start is None:
        return []
    distance = distance_field(traversable, start, resolution)
    reachable = np.isfinite(distance)

    # touching cells, diagonals included, are one frontier
    labels, count = ndimage.label(frontier_mask(free, unknown), structure=np.ones((3, 3), dtype=int))
    frontiers = []
    for cells in _cells_by_label(labels, count):
        size_m = len(cells) * resolution
        if size_m < cfg.min_frontier_m:
            continue
        standable = cells[reachable[cells[:, 0], cells[:, 1]]]
        if len(standable) == 0:
            continue
        offset = standable - cells.mean(axis=0)
        goal = standable[int(np.argmin((offset * offset).sum(axis=1)))]
        goal = (int(goal[0]), int(goal[1]))
        distance_m = float(distance[goal])
        frontiers.append(
            Frontier(
                cells=cells,
                goal=goal,
                size_m=size_m,
                distance_m=distance_m,
                score=distance_m - cfg.gain_weight * size_m,
            )
        )
    # the goal cell breaks ties, so that the order does not depend on how the labels happen to be numbered
    frontiers.sort(key=lambda f: (f.score, f.goal))
    return frontiers


def _cells_by_label(labels: np.ndarray, count: int) -> list[np.ndarray]:
    """The cells ``(K, 2)`` of each of the ``count`` labels of ``scipy.ndimage.label``, in one pass over the grid."""
    iy, ix = np.nonzero(labels)
    order = np.argsort(labels[iy, ix], kind="stable")
    cells = np.stack([iy[order], ix[order]], axis=1)
    sizes = np.bincount(labels[iy, ix], minlength=count + 1)[1:]
    return np.split(cells, np.cumsum(sizes)[:-1]) if count else []


# ------------------------------------------------------------------------------------------------- explorer
class Explorer:
    """Picks the frontier to drive to, call after call, until none is left.

    Two things make it more than "take the best frontier every time":

    * **It keeps its target.** A frontier shrinks while the robot looks at it, so its score gets worse as the
      robot approaches, and another frontier can overtake it by a hair. The explorer stays with the frontier it
      is driving to unless another one is better by ``switch_margin_m``.
    * **It can give up.** When the robot cannot get to a goal (the planner or the controller failed), call
      :meth:`give_up`; frontiers around that goal are then skipped.
    """

    def __init__(self, cfg: FrontierCfg | None = None):
        self.cfg = cfg or FrontierCfg()
        self.target: Frontier | None = None
        """The frontier of the last :meth:`update`; None before the first call and when exploration is over."""
        self.abandoned: list[tuple[int, int]] = []
        """Goal cells given up on."""
        self.switches = 0
        """How many times it left a frontier that was still on the map for a better one."""

    def update(self, grid: np.ndarray, robot_cell: tuple[int, int], resolution: float) -> Frontier | None:
        """The frontier to drive to on this map, or None when there is nothing left to explore."""
        cfg = self.cfg
        frontiers = find_frontiers(grid, robot_cell, resolution, cfg)
        skip_cells = cfg.give_up_radius_m / resolution
        frontiers = [f for f in frontiers if all(math.dist(f.goal, a) > skip_cells for a in self.abandoned)]
        if not frontiers:
            self.target = None
            return None
        best = frontiers[0]
        if self.target is not None:
            old_goal = self.target.goal
            still_there = min(frontiers, key=lambda f: f.gap_cells(old_goal))
            if still_there is not best and still_there.gap_cells(old_goal) * resolution <= cfg.same_frontier_m:
                if best.score > still_there.score - cfg.switch_margin_m:
                    best = still_there  # not better by enough: stay with the old target
                else:
                    self.switches += 1
        self.target = best
        return best

    def give_up(self) -> None:
        """Abandon the current target: it will not be offered again, nor will a frontier with a goal near it."""
        if self.target is not None:
            self.abandoned.append(self.target.goal)
            self.target = None
