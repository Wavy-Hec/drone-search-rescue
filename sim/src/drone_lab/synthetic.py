"""CPU-only synthetic survey: a kinematic stand-in for the RL policy, flown over a terrain mesh.

Uses the same path generators, carrot logic (:mod:`drone_lab.carrot`), mapping and file formats as the Isaac Sim
survey (``scripts/survey_map.py``), so the whole pipeline can be checked without a GPU. The drone is a point mass
with a PD pull towards the carrot (a crude model of the tracking policy: lag, limited acceleration and speed);
the mapping sensor is the same 5 m x 5 m grid of vertical rays, intersected exactly with the terrain mesh.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .carrot import advance_carrot, clamp_offset, interpolate_path, pad_paths
from .mapping import ElevationMapper, GridSpec, MeshHeightQuery, ground_truth_map
from .paths import Area
from .survey_io import TrajectoryRecorder, compute_summary


def grid_offsets(size: tuple[float, float] = (5.0, 5.0), resolution: float = 0.25) -> np.ndarray:
    """Ray start offsets of Isaac Lab's ``GridPatternCfg`` (same arange rule), shape (R, 2)."""
    x = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution)
    y = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()])


def _quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return np.stack([qx, qy, qz, qw], axis=-1)


def simulate_survey(
    vertices: np.ndarray,
    faces: np.ndarray,
    waypoints: list[np.ndarray],
    *,
    area: Area,
    speed: float = 1.0,
    dt: float = 0.1,
    max_lead: float = 1.0,
    slow_lead: float = 0.7,
    path_resolution: float = 0.25,
    footprint: tuple[float, float] = (5.0, 5.0),
    ray_resolution: float = 0.25,
    map_resolution: float = 0.25,
    map_padding: float = 0.0,
    reducer: str = "median",
    kp: float = 4.0,
    kd: float = 3.2,
    max_accel: float = 4.0,
    max_speed: float = 3.0,
    position_noise: float = 0.0,
    seed: int = 0,
    hits_every: int = 5,
    hits_stride: int = 4,
    max_time: float | None = None,
    gt_supersample: int = 3,
    extra_summary: dict | None = None,
):
    """Fly every drone along its path and build the map; returns a dict with everything :func:`save_run` needs."""
    rng = np.random.default_rng(seed)
    terrain = MeshHeightQuery(vertices, faces)
    num = len(waypoints)
    pts, arc, length = pad_paths(waypoints, path_resolution)
    s = torch.zeros(num)
    pos = pts[:, 0].numpy().astype(np.float64).copy()
    vel = np.zeros((num, 3))
    offsets = grid_offsets(footprint, ray_resolution)
    grid = GridSpec.from_area(area, map_resolution, padding=map_padding)
    mapper = ElevationMapper(grid)
    rec = TrajectoryRecorder()
    hits_log = []
    if max_time is None:
        max_time = 2.5 * float(length.max()) / max(speed, 1e-3) + 60.0
    n_steps = int(math.ceil(max_time / dt))
    finished = np.zeros(num, dtype=bool)
    for k in range(n_steps):
        t = (k + 1) * dt
        # carrot first (as in the env: commands update after physics, before observations)
        s, carrot_t, gate = advance_carrot(
            s, pts, arc, torch.as_tensor(pos, dtype=torch.float32), speed, dt, max_lead, slow_lead
        )
        carrot = carrot_t.numpy().astype(np.float64)
        _, seg = interpolate_path(pts, arc, s)
        # PD pull towards the carrot, acceleration and speed limited
        acc = kp * (carrot - pos) - kd * vel
        a_norm = np.linalg.norm(acc, axis=1, keepdims=True)
        acc = acc * np.minimum(1.0, max_accel / np.maximum(a_norm, 1e-9))
        vel = vel + acc * dt
        v_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = vel * np.minimum(1.0, max_speed / np.maximum(v_norm, 1e-9))
        pos = pos + vel * dt
        if position_noise > 0:
            pos = pos + rng.normal(0.0, position_noise, size=pos.shape)
        # tilt that a quadrotor needs for this acceleration (small-angle, yaw held at 0)
        pitch = np.arctan2(acc[:, 0], 9.81)
        roll = -np.arctan2(acc[:, 1], 9.81)
        quat = _quat_from_euler(roll, pitch, np.zeros(num))
        # mapping rays: vertical, world-aligned footprint (ray_alignment="world")
        active = ~finished
        ray_xy = (pos[active, None, :2] + offsets[None, :, :]).reshape(-1, 2)
        z_hit = terrain(ray_xy)
        hits = np.column_stack([ray_xy, z_hit])
        mapper.add(hits)
        if hits_every > 0 and k % hits_every == 0:
            env_col = np.repeat(np.nonzero(active)[0], len(offsets)).astype(np.float64)
            sub = slice(None, None, max(1, hits_stride))
            keep = np.isfinite(hits[sub, 2])
            hits_log.append(np.column_stack([np.full(keep.sum(), t), env_col[sub][keep], hits[sub][keep]]))
        ground = terrain(pos[:, :2])
        offset = clamp_offset(torch.as_tensor(pos - carrot, dtype=torch.float32), max_lead, (1.0, 1.0, 1.0)).numpy()
        ids = np.nonzero(active)[0]
        rec.record(
            step=np.full(len(ids), k),
            time=np.full(len(ids), t),
            env=ids,
            x=pos[ids, 0],
            y=pos[ids, 1],
            z=pos[ids, 2],
            qx=quat[ids, 0],
            qy=quat[ids, 1],
            qz=quat[ids, 2],
            qw=quat[ids, 3],
            roll=roll[ids],
            pitch=pitch[ids],
            yaw=np.zeros(len(ids)),
            vx=vel[ids, 0],
            vy=vel[ids, 1],
            vz=vel[ids, 2],
            wx=np.zeros(len(ids)),
            wy=np.zeros(len(ids)),
            wz=np.zeros(len(ids)),
            speed=np.linalg.norm(vel[ids], axis=1),
            carrot_x=carrot[ids, 0],
            carrot_y=carrot[ids, 1],
            carrot_z=carrot[ids, 2],
            carrot_dist=np.linalg.norm(pos[ids] - carrot[ids], axis=1),
            path_s=s.numpy()[ids],
            progress=(s / length).numpy()[ids],
            path_index=seg.numpy()[ids],
            obs_dx=offset[ids, 0],
            obs_dy=offset[ids, 1],
            obs_dz=offset[ids, 2],
            gate=gate.numpy()[ids],
            agl=pos[ids, 2] - ground[ids],
        )
        end = pts[np.arange(num), -1].numpy()
        done = (s >= length - 1e-4).numpy() & (np.linalg.norm(pos - end, axis=1) < 0.3)
        finished |= done
        if finished.all():
            break
    elevation, count = mapper.finalize(reducer)
    gt, gt_range = ground_truth_map(vertices, faces, grid, gt_supersample, query=terrain)
    traj = rec.as_dict()
    summary = compute_summary(
        traj,
        waypoints,
        grid=grid,
        area=area,
        elevation=elevation,
        count=count,
        ground_truth=gt,
        ground_truth_range=gt_range,
        commanded_speed=speed,
        crashed=[False] * num,
        extra=extra_summary,
    )
    return {
        "trajectory": traj,
        "planned": waypoints,
        "grid": grid,
        "area": area,
        "elevation": elevation,
        "count": count,
        "ground_truth": gt,
        "ground_truth_range": gt_range,
        "hits": np.concatenate(hits_log) if hits_log else np.zeros((0, 5)),
        "summary": summary,
        "num_hits": mapper.num_points,
    }
