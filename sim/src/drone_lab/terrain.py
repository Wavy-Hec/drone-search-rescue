"""Procedural survey area built with the Isaac Lab 3.0 terrain generator.

The area is a 4 x 4 grid of 10 m sub-terrains (40 m x 40 m) inside a 6 m flat border, mixing rolling hills,
random rough ground, pyramids, a pit, waves, stepped blocks and stairs, with heights of up to about 3 m.

Determinism: Isaac Lab picks the sub-terrain type and difficulty with its own seeded generator
(``TerrainGeneratorCfg.seed``), but several sub-terrain functions draw from the *global* numpy RNG. The
``Seeded*`` configs below wrap those functions so every tile seeds numpy from ``(seed, difficulty, type)`` and
restores the global state afterwards. The same config therefore produces the same mesh in the simulator and
in the CPU-only preview (``scripts/preview_area.py``), on any machine. All terrain types used here are numpy
based; ``MeshRandomGridTerrainCfg`` is avoided because it draws from torch's CUDA generator when a GPU is
present. ``slope_threshold`` is disabled so the mesh has no gaps when seen from above (steep steps become
0.1 m-wide ramps instead of exactly vertical walls).

This module only needs numpy, scipy and trimesh at generation time; it does not launch Isaac Sim.
"""

from __future__ import annotations

import contextlib
import zlib
from dataclasses import MISSING

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfTerrainBaseCfg,
    HfWaveTerrainCfg,
)
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.trimesh import MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass

from .paths import Area

__all__ = [
    "SURVEY_TERRAIN_CFG",
    "SeededHfRandomUniformTerrainCfg",
    "SteppedBlocksTerrainCfg",
    "generate_terrain_mesh",
    "terrain_area",
]


def tile_seed(cfg, difficulty: float) -> int:
    """Deterministic 32-bit seed of one sub-terrain tile from its config seed, difficulty and type."""
    base = 0 if getattr(cfg, "seed", None) is None else int(cfg.seed)
    name = zlib.crc32(type(cfg).__name__.encode())
    return (base * 1_000_003 + int(round(float(difficulty) * 1e6)) * 7919 + name) % (2**32)


@contextlib.contextmanager
def numpy_seed(seed: int):
    """Temporarily seed numpy's global RNG and restore its previous state on exit."""
    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


def seeded_random_uniform_terrain(difficulty: float, cfg: HfRandomUniformTerrainCfg):
    """:func:`isaaclab.terrains.height_field.hf_terrains.random_uniform_terrain` with a per-tile seed."""
    with numpy_seed(tile_seed(cfg, difficulty)):
        return hf_terrains.random_uniform_terrain(difficulty, cfg)


@configclass
class SeededHfRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
    """Random uniform height field (hills or rough ground) that is reproducible per tile."""

    function: str = "drone_lab.terrain:seeded_random_uniform_terrain"


@height_field_to_mesh
def stepped_blocks_terrain(difficulty: float, cfg: SteppedBlocksTerrainCfg):
    """Rectangular blocks of quantised heights, stacked where they overlap ("small town" / stepped boxes).

    The tallest block height grows with ``difficulty`` from ``height_range[0]`` to ``height_range[1]``.
    Uses a local numpy generator seeded by :func:`tile_seed`, so it never touches global RNG state.
    """
    rng = np.random.default_rng(tile_seed(cfg, difficulty))
    hs = cfg.horizontal_scale
    w_px = int(cfg.size[0] / hs)
    l_px = int(cfg.size[1] / hs)
    hf = np.zeros((w_px, l_px))
    h_top = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    levels = np.arange(cfg.height_step, h_top + 1e-9, cfg.height_step)
    if len(levels) == 0:
        levels = np.array([h_top])
    for _ in range(cfg.num_blocks):
        bw = max(2, int(rng.uniform(*cfg.block_size_range) / hs))
        bl = max(2, int(rng.uniform(*cfg.block_size_range) / hs))
        x0 = int(rng.integers(0, max(1, w_px - bw)))
        y0 = int(rng.integers(0, max(1, l_px - bl)))
        h = float(rng.choice(levels))
        hf[x0 : x0 + bw, y0 : y0 + bl] = np.maximum(hf[x0 : x0 + bw, y0 : y0 + bl], h)
    height_field = np.rint(hf / cfg.vertical_scale).astype(np.int16)
    origin = np.array([0.5 * (w_px - 1) * hs, 0.5 * (l_px - 1) * hs, 0.0])
    return height_field, origin


@configclass
class SteppedBlocksTerrainCfg(HfTerrainBaseCfg):
    """Configuration for :func:`stepped_blocks_terrain`."""

    function: str = "drone_lab.terrain:stepped_blocks_terrain"

    block_size_range: tuple[float, float] = (1.0, 3.0)
    """Block edge length range [m]."""

    height_range: tuple[float, float] = MISSING
    """Tallest block height at difficulty 0 and 1 [m]."""

    height_step: float = 0.5
    """Height quantisation of the blocks [m]."""

    num_blocks: int = 12
    """Number of blocks per tile."""


SURVEY_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=17,
    curriculum=False,
    size=(10.0, 10.0),
    border_width=6.0,
    border_height=1.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    # no slope correction: with Isaac Lab's default (0.75) the height-field-to-mesh conversion shifts vertices at
    # steep tile edges and leaves ~1 % of the area uncovered from above (rays fall through 0.1 m slits)
    slope_threshold=None,
    difficulty_range=(0.4, 1.0),
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "hills": SeededHfRandomUniformTerrainCfg(
            proportion=0.22, noise_range=(0.0, 3.0), noise_step=0.05, downsampled_scale=2.0
        ),
        "rough": SeededHfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.0, 0.4), noise_step=0.05, downsampled_scale=0.5
        ),
        "pyramid": HfPyramidSlopedTerrainCfg(proportion=0.16, slope_range=(0.4, 0.7), platform_width=1.5),
        "pit": HfInvertedPyramidSlopedTerrainCfg(proportion=0.08, slope_range=(0.3, 0.45), platform_width=1.5),
        "waves": HfWaveTerrainCfg(proportion=0.10, amplitude_range=(0.6, 1.5), num_waves=2),
        "blocks": SteppedBlocksTerrainCfg(proportion=0.20, height_range=(1.5, 3.0), num_blocks=12),
        "stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.14, step_height_range=(0.25, 0.5), step_width=0.8, platform_width=1.5, border_width=0.25
        ),
    },
)
"""Survey area: 4 x 4 tiles of 10 m (40 m x 40 m) with a 6 m flat border, heights within about -2..3 m."""


def terrain_area(cfg: TerrainGeneratorCfg) -> Area:
    """Area covered by the sub-terrains (the generator centres the grid on the world origin, border excluded)."""
    half_x = 0.5 * cfg.size[0] * cfg.num_rows
    half_y = 0.5 * cfg.size[1] * cfg.num_cols
    return Area(-half_x, half_x, -half_y, half_y)


def generate_terrain_mesh(cfg: TerrainGeneratorCfg | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate the terrain on the CPU with Isaac Lab's :class:`~isaaclab.terrains.TerrainGenerator`.

    Returns:
        ``(vertices (V, 3), faces (F, 3))`` in the world frame, identical to what the simulator imports for
        the same (seeded) config.
    """
    from isaaclab.terrains import TerrainGenerator
    from isaaclab.utils import clone

    gen = TerrainGenerator(clone(cfg if cfg is not None else SURVEY_TERRAIN_CFG), device="cpu")
    mesh = gen.terrain_mesh
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
