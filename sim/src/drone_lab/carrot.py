"""Carrot (moving target) logic for path following with a position-tracking policy (torch only).

The built-in ``IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1`` policy was trained to fly to a fixed point
from starts within +-1 m per axis of it. To make it follow a long path, a "carrot" target slides along the path
at the requested speed and the policy observes the drone position relative to the carrot instead of relative to
its env origin. Two safeguards keep the observation inside the training distribution:

* :func:`advance_carrot` slows the carrot down (and finally stops it) when the drone lags behind, so the
  carrot never gets more than ``max_lead`` metres ahead of the drone.
* :func:`clamp_offset` limits the observed offset to a ball of radius ``max_norm`` and to per-axis limits.

All functions are batched over environments (first dimension) and work on any device.
"""

from __future__ import annotations

import numpy as np
import torch


def pad_paths(
    paths: list[np.ndarray], ds: float, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample variable-length paths and pack them into padded tensors.

    Every path is resampled at <= ``ds`` spacing; shorter paths are padded with their last point (and a
    constant arclength), which :func:`interpolate_path` handles transparently.

    Returns:
        ``(points (E, P, 3), arclength (E, P), length (E,))`` float32 tensors.
    """
    from .paths import resample_polyline

    res = [resample_polyline(p, ds) for p in paths]
    p_max = max(len(r[0]) for r in res)
    pts = np.zeros((len(res), p_max, 3), dtype=np.float32)
    arc = np.zeros((len(res), p_max), dtype=np.float32)
    for i, (rp, rs) in enumerate(res):
        n = len(rp)
        pts[i, :n] = rp
        pts[i, n:] = rp[-1]
        arc[i, :n] = rs
        arc[i, n:] = rs[-1]
    pts_t = torch.as_tensor(pts, device=device)
    arc_t = torch.as_tensor(arc, device=device)
    return pts_t, arc_t, arc_t[:, -1].clone()


def interpolate_path(
    points: torch.Tensor, arclength: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Point at arclength ``s`` on every (padded) path.

    Args:
        points: (E, P, 3) path vertices.
        arclength: (E, P) cumulative arclength (non-decreasing).
        s: (E,) query arclength; clamped to ``[0, length]``.

    Returns:
        ``(position (E, 3), segment_index (E,) long)``.
    """
    num_pts = points.shape[1]
    s = torch.minimum(torch.clamp(s, min=0.0), arclength[:, -1])
    if num_pts == 1:
        return points[:, 0].clone(), torch.zeros_like(s, dtype=torch.long)
    idx = torch.searchsorted(arclength.contiguous(), s.unsqueeze(-1).contiguous(), right=True).squeeze(-1) - 1
    idx = idx.clamp(0, num_pts - 2)
    s0 = arclength.gather(1, idx.unsqueeze(-1)).squeeze(-1)
    s1 = arclength.gather(1, (idx + 1).unsqueeze(-1)).squeeze(-1)
    t = ((s - s0) / (s1 - s0).clamp(min=1e-9)).clamp(0.0, 1.0)
    gather_idx = idx.view(-1, 1, 1).expand(-1, 1, 3)
    p0 = points.gather(1, gather_idx).squeeze(1)
    p1 = points.gather(1, gather_idx + 1).squeeze(1)
    return p0 + t.unsqueeze(-1) * (p1 - p0), idx


def advance_carrot(
    s: torch.Tensor,
    points: torch.Tensor,
    arclength: torch.Tensor,
    drone_pos: torch.Tensor,
    speed: float | torch.Tensor,
    dt: float,
    max_lead: float,
    slow_lead: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move the carrot along the path at ``speed`` without letting it run away from the drone.

    The nominal step ``speed * dt`` is scaled by a gate that is 1 while the drone-carrot distance is below
    ``slow_lead`` and falls linearly to 0 at ``max_lead``. A step that would put the carrot more than
    ``max_lead`` away from the drone is shortened to ``max_lead - distance``: the carrot moves at most that far
    along the path, and a chord is never longer than its arc, so it stays within ``max_lead``. (Rejecting the
    whole step instead made the carrot stop-and-go once ``speed * dt > max_lead - slow_lead``, e.g. above 3 m/s
    with the defaults, and stall for good at ``speed * dt >= max_lead``.)

    Args:
        s: (E,) current carrot arclength.
        points, arclength: padded paths from :func:`pad_paths`.
        drone_pos: (E, 3) drone positions (same frame as the path).
        speed: nominal carrot speed [m/s] (scalar or (E,)).
        dt: time step [s].
        max_lead: maximum drone-carrot distance [m].
        slow_lead: distance [m] where slowing starts; defaults to ``0.5 * max_lead``.

    Returns:
        ``(s_new (E,), carrot (E, 3), gate (E,))``.
    """
    if slow_lead is None:
        slow_lead = 0.5 * max_lead
    slow_lead = min(float(slow_lead), float(max_lead) - 1e-6)
    carrot, _ = interpolate_path(points, arclength, s)
    dist = torch.linalg.norm(carrot - drone_pos, dim=-1)
    gate = ((max_lead - dist) / (max_lead - slow_lead)).clamp(0.0, 1.0)
    step = speed * gate * dt
    s_try = torch.minimum(s + step, arclength[:, -1])
    carrot_try, _ = interpolate_path(points, arclength, s_try)
    ok = torch.linalg.norm(carrot_try - drone_pos, dim=-1) <= max_lead
    # too far: advance only by the remaining lead (|carrot' - drone| <= dist + arc step <= max_lead)
    s_short = torch.minimum(s + torch.minimum(step, (max_lead - dist).clamp(min=0.0)), arclength[:, -1])
    s_new = torch.where(ok, s_try, s_short)
    carrot_short, _ = interpolate_path(points, arclength, s_short)
    carrot_new = torch.where(ok.unsqueeze(-1), carrot_try, carrot_short)
    return s_new, carrot_new, gate


def clamp_offset(
    offset: torch.Tensor,
    max_norm: float | None,
    per_axis_limit: tuple[float, float, float] | float | None = None,
) -> torch.Tensor:
    """Limit an offset vector to a ball (direction preserved) and then to a per-axis box.

    Args:
        offset: (E, 3) drone position minus target position.
        max_norm: radius of the ball [m]; None disables it.
        per_axis_limit: symmetric per-axis limit [m] (scalar or 3 values); None disables it.
    """
    out = offset
    if max_norm is not None:
        norm = torch.linalg.norm(out, dim=-1, keepdim=True)
        out = out * (float(max_norm) / norm.clamp(min=1e-9)).clamp(max=1.0)
    if per_axis_limit is not None:
        lim = torch.as_tensor(per_axis_limit, dtype=out.dtype, device=out.device)
        out = torch.maximum(torch.minimum(out, lim), -lim)
    return out
