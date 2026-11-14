"""One route past several goals: which person to visit first, second, ... and the route between them.

* **Stops:** every goal (a person, whose own cells are no-go) snaps to the nearest free cell that is reachable from
  the start; :func:`~drone_lab.teaming.planner.plan_path` itself does that here, so a tour stops where a single plan
  would. A goal with no such cell within ``max_goal_snap_m`` is unreachable: it is reported and left out.
* **Costs:** the cost between two stops (the start and the goals' stops) is the cost of the A* grid route between
  them (:attr:`~drone_lab.teaming.planner.PlanResult.graph_cost`), cross-slope cost included when the map has one.
* **Order:** the open route that begins at the start, visits every stop once and does not return, with the smallest
  total cost. Up to :data:`HELD_KARP_MAX_GOALS` goals the order is exact (dynamic programming over the sets of
  visited stops, Held-Karp); above that it is a nearest-neighbour order improved by 2-opt (not always the best).
* **Legs:** the ``plan_path`` route from each stop to the next goal.

World points are ``(x, y)`` [m]. numpy + scipy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .planner import PlanResult, plan_path
from .traversability import TraversabilityMap

__all__ = [
    "HELD_KARP_MAX_GOALS",
    "TourResult",
    "held_karp_order",
    "nearest_neighbour_order",
    "plan_tour",
    "route_cost",
    "two_opt",
]

HELD_KARP_MAX_GOALS = 12
"""Most goals the exact search is used for: it looks at ``n^2 2^n`` cases (about 600,000 for 12 goals)."""


@dataclass
class TourResult:
    """Outcome of :func:`plan_tour`."""

    success: bool
    """At least one goal is reachable and every leg has a route."""
    message: str
    start_xy: np.ndarray
    """Requested start [m]."""
    goals_xy: np.ndarray
    """Requested goals (N, 2) [m]."""
    order: list[int] = field(default_factory=list)
    """Indices into :attr:`goals_xy` of the reachable goals, in visiting order."""
    unreachable: list[int] = field(default_factory=list)
    """Indices into :attr:`goals_xy` of the goals that are left out."""
    legs: list[PlanResult] = field(default_factory=list)
    """``legs[k]`` leads from the previous stop (the start for ``k = 0``) to the goal ``order[k]``."""
    method: str = ""
    """How the order was found: ``held_karp`` (exact) or ``nearest_neighbour_2opt``; empty without a tour."""

    @property
    def leg_lengths_m(self) -> list[float]:
        """Length of each leg's final path [m]."""
        return [leg.length_m for leg in self.legs]

    @property
    def leg_costs(self) -> list[float]:
        """A* cost of each leg on the grid (metres weighted by cell cost): what the order minimises."""
        return [leg.graph_cost for leg in self.legs]

    @property
    def length_m(self) -> float:
        """Length of the whole tour [m]."""
        return float(sum(self.leg_lengths_m))

    @property
    def cost(self) -> float:
        """A* cost of the whole tour."""
        return float(sum(self.leg_costs))

    def to_dict(self) -> dict:
        """JSON-friendly summary (the paths themselves are stored separately)."""

        def pt(a):
            return [round(float(v), 4) for v in np.asarray(a).ravel()[:2]]

        legs = []
        for goal, leg in zip(self.order, self.legs, strict=True):
            legs.append(
                {
                    "goal": int(goal),
                    "goal_xy": pt(leg.goal_xy),
                    "from_xy": pt(leg.start_snapped_xy),
                    "stop_xy": pt(leg.goal_snapped_xy),
                    "goal_snap_m": round(leg.goal_snap_m, 3),
                    "length_m": round(leg.length_m, 3),
                    "cost": round(leg.graph_cost, 3),
                }
            )
        return {
            "success": self.success,
            "message": self.message,
            "method": self.method,
            "start_xy": pt(self.start_xy),
            "order": [int(i) for i in self.order],
            "unreachable": [int(i) for i in self.unreachable],
            "length_m": round(self.length_m, 3),
            "cost": round(self.cost, 3),
            "legs": legs,
        }


# --------------------------------------------------------------------------------------------- visiting order
def route_cost(cost: np.ndarray, order: list[int]) -> float:
    """Cost of visiting the nodes ``order`` one after another: the sum of ``cost[a, b]`` over consecutive nodes."""
    return float(sum(cost[a, b] for a, b in zip(order[:-1], order[1:], strict=True)))


def held_karp_order(cost: np.ndarray) -> list[int]:
    """Cheapest open route from node 0 through every other node once (it does not return), exactly.

    ``best[visited][j]`` is the cost of the cheapest route that starts at node 0, visits exactly the nodes of the
    set ``visited`` and ends at ``j``. A set is a bit mask: bit ``j - 1`` stands for node ``j``. Every such route is
    a shorter one (the set without ``j``, ending at some ``i``) plus the step ``i -> j``, so the sets are filled in
    from small to large.

    Args:
        cost: ``(n + 1, n + 1)`` costs between the nodes; node 0 is the start.

    Returns:
        The nodes ``1 .. n`` in visiting order.
    """
    n = len(cost) - 1
    if n == 0:
        return []
    c = np.asarray(cost, dtype=float).tolist()
    best = [[math.inf] * (n + 1) for _ in range(1 << n)]
    previous = [[0] * (n + 1) for _ in range(1 << n)]
    for j in range(1, n + 1):
        best[1 << (j - 1)][j] = c[0][j]
    # adding a node to a set gives a larger number, so counting up meets every set after the sets it grows from
    for visited in range(1, 1 << n):
        for i in range(1, n + 1):
            so_far = best[visited][i]
            if not math.isfinite(so_far):  # i is not in the set
                continue
            for j in range(1, n + 1):
                bit = 1 << (j - 1)
                if visited & bit:
                    continue
                if so_far + c[i][j] < best[visited | bit][j]:
                    best[visited | bit][j] = so_far + c[i][j]
                    previous[visited | bit][j] = i
    visited = (1 << n) - 1
    last = min(range(1, n + 1), key=lambda j: best[visited][j])
    order = []
    while last != 0:
        order.append(last)
        visited, last = visited & ~(1 << (last - 1)), previous[visited][last]
    order.reverse()
    return order


def nearest_neighbour_order(cost: np.ndarray) -> list[int]:
    """From node 0, always go on to the cheapest node not visited yet. Returns the nodes ``1 .. n`` in that order."""
    left = list(range(1, len(cost)))
    order = []
    here = 0
    while left:
        here = min(left, key=lambda j: cost[here, j])
        left.remove(here)
        order.append(here)
    return order


def two_opt(cost: np.ndarray, order: list[int]) -> list[int]:
    """Improve a visiting order by reversing stretches of it for as long as one reversal makes the route cheaper.

    The route starts at node 0, which stays in place. The result is never more expensive than ``order``.
    """
    route = [0, *order]
    best = route_cost(cost, route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                candidate_cost = route_cost(cost, candidate)
                if candidate_cost < best - 1e-9:  # a real gain, not rounding: the loop must end
                    route, best, improved = candidate, candidate_cost, True
    return route[1:]


# ---------------------------------------------------------------------------------------------------- the tour
def plan_tour(
    tmap: TraversabilityMap,
    start_xy: ArrayLike,
    goals_xy: ArrayLike,
    *,
    max_goal_snap_m: float = 2.5,
    spacing: float = 0.1,
) -> TourResult:
    """Plan one route from ``start_xy`` past every reachable goal of ``goals_xy`` (world, [m]) on ``tmap``.

    Args:
        max_goal_snap_m: A goal counts as reachable when a free cell reachable from the start lies within this
            distance of it [m]; the robot stops on the nearest such cell.
        spacing: Distance between the points of each leg's path [m].
    """
    start_xy = np.asarray(start_xy, dtype=float).ravel()[:2]
    goals_xy = np.asarray(goals_xy, dtype=float).reshape(-1, 2)
    tour = TourResult(success=False, message="no goals", start_xy=start_xy, goals_xy=goals_xy)
    if len(goals_xy) == 0:
        return tour

    # the grid route from the start to every goal on its own: which goals are reachable, the stop next to each
    # and the cost from the start
    first = [plan_path(tmap, start_xy, g, max_goal_snap_m=max_goal_snap_m, smooth=False) for g in goals_xy]
    reachable = [i for i, plan in enumerate(first) if plan.success]
    tour.unreachable = [i for i, plan in enumerate(first) if not plan.success]
    if not reachable:
        tour.message = first[0].message if first[0].start_failed else "no goal is reachable from the start"
        return tour

    # costs between the stops; node 0 is the start, node k + 1 the stop of the goal reachable[k]. A move costs the
    # same in both directions, so each pair is planned once
    stops = [first[i].goal_snapped_xy for i in reachable]
    n = len(stops)
    cost = np.zeros((n + 1, n + 1))
    for k, i in enumerate(reachable):
        cost[0, k + 1] = cost[k + 1, 0] = first[i].graph_cost
    for a in range(n):
        for b in range(a + 1, n):
            cost[a + 1, b + 1] = cost[b + 1, a + 1] = plan_path(tmap, stops[a], stops[b], smooth=False).graph_cost

    if n <= HELD_KARP_MAX_GOALS:
        nodes = held_karp_order(cost)
        tour.method = "held_karp"
    else:
        nodes = two_opt(cost, nearest_neighbour_order(cost))
        tour.method = "nearest_neighbour_2opt"
    tour.order = [reachable[node - 1] for node in nodes]

    # the legs: from each stop to the next goal, which snaps to the same stop as from the start (the free cells
    # reachable from a stop are those reachable from the start)
    here = start_xy
    for i in tour.order:
        leg = plan_path(tmap, here, goals_xy[i], max_goal_snap_m=max_goal_snap_m, spacing=spacing)
        tour.legs.append(leg)
        here = leg.goal_snapped_xy
    tour.success = all(leg.success for leg in tour.legs)
    tour.message = (
        f"tour past {n} of {len(goals_xy)} goals ({tour.method}): {tour.length_m:.1f} m, cost {tour.cost:.1f}"
        + (f"; {len(tour.unreachable)} unreachable" if tour.unreachable else "")
    )
    return tour
