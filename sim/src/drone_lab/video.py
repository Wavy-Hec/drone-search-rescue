"""Headless MP4 capture from a Kit visualizer (no moviepy; frames are written with imageio-ffmpeg).

Shared by ``scripts/survey_map.py`` and ``scripts/eval_trajectory.py``. The capture path is the one Isaac Lab's
own recorder uses (``isaaclab/envs/utils/video_recorder.py``, ``VideoRecorder._frame_from_visualizer``): a headless
visualizer is not "continuous", so ``env.step()`` neither syncs it nor runs the debug-marker callbacks
(``SimulationContext.is_rendering`` excludes headless visualizers). Before every frame the stage is therefore
synced with ``sim.forward()`` and the marker callbacks are dispatched, so the carrot / path / reference markers
move in the video.

Only numpy is imported at module level, so this module can be imported before the simulator is launched.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imageio.core.format import Format

    from isaaclab.sim import SimulationContext
    from isaaclab.visualizers.base_visualizer import BaseVisualizer


def find_kit_visualizer(sim: SimulationContext) -> BaseVisualizer | None:
    """The first Kit visualizer of ``sim`` that can render frames, or None."""
    for viz in getattr(sim, "visualizers", []) or []:
        if getattr(viz.cfg, "visualizer_type", None) == "kit" and hasattr(viz, "render_rgb_array"):
            return viz
    return None


def capture_frame(sim: SimulationContext, viz: BaseVisualizer) -> np.ndarray | None:
    """One RGB frame ``(H, W, 3)`` uint8 from ``viz`` (or None), with up-to-date poses and debug markers."""
    if not sim.is_rendering:
        sim.forward()
        # continuous rendering dispatches marker callbacks in update_visualizers(); an on-demand capture must too
        if viz.supports_markers():
            sim.vis_marker_registry.dispatch_callbacks()
    frame = viz.render_rgb_array()
    return None if frame is None else np.ascontiguousarray(frame[:, :, :3])


def open_mp4_writer(path: str | Path, fps: float) -> Format.Writer:
    """H.264 MP4 writer (imageio-ffmpeg). ``macro_block_size=8`` keeps 1280x720 frames unscaled."""
    import imageio.v2 as imageio

    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=8)


def playback_speedup(fps: float, step_dt: float, stride: int = 1) -> float:
    """How many times faster than real time the video plays.

    One frame is captured every ``stride`` policy steps of ``step_dt`` seconds and shown for ``1 / fps`` seconds;
    e.g. 10 Hz policy steps written at 30 fps play 3x faster than real time, at 10 fps in real time.
    """
    return float(fps) * float(step_dt) * max(1, int(stride))


def realtime_fps(step_dt: float, stride: int = 1) -> float:
    """Frame rate at which the captured frames play in real time."""
    return 1.0 / (float(step_dt) * max(1, int(stride)))
