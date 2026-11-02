"""Headless video helpers (numpy only; the simulator and visualizer are stand-ins)."""

from __future__ import annotations

import numpy as np
from _runner import main_for

from drone_lab.video import capture_frame, find_kit_visualizer, playback_speedup, realtime_fps


class _Registry:
    def __init__(self, log):
        self.log = log

    def dispatch_callbacks(self, event=None):
        self.log.append("markers")


class _Sim:
    def __init__(self, rendering: bool, visualizers=()):
        self.log: list[str] = []
        self.is_rendering = rendering
        self.visualizers = list(visualizers)
        self.vis_marker_registry = _Registry(self.log)

    def forward(self):
        self.log.append("forward")


class _Cfg:
    def __init__(self, visualizer_type):
        self.visualizer_type = visualizer_type


class _Viz:
    def __init__(self, visualizer_type="kit", markers=True, channels=4):
        self.cfg = _Cfg(visualizer_type)
        self._markers = markers
        self._channels = channels

    def supports_markers(self):
        return self._markers

    def render_rgb_array(self):
        return np.full((4, 6, self._channels), 7, dtype=np.uint8)


def test_capture_frame_dispatches_markers_when_headless():
    """Without continuous rendering the capture itself syncs the stage and runs the marker callbacks
    (as isaaclab/envs/utils/video_recorder.py does), so command markers move in the video."""
    sim = _Sim(rendering=False)
    frame = capture_frame(sim, _Viz())
    assert sim.log == ["forward", "markers"]
    assert frame.shape == (4, 6, 3) and frame.dtype == np.uint8 and frame.flags["C_CONTIGUOUS"]
    sim = _Sim(rendering=False)
    capture_frame(sim, _Viz(markers=False))
    assert sim.log == ["forward"]
    sim = _Sim(rendering=True)  # continuous rendering already did both in env.step()
    capture_frame(sim, _Viz())
    assert sim.log == []


def test_find_kit_visualizer_and_playback_speed():
    kit = _Viz("kit")
    assert find_kit_visualizer(_Sim(False, [_Viz("newton"), kit])) is kit
    assert find_kit_visualizer(_Sim(False, [_Viz("newton")])) is None
    assert find_kit_visualizer(object()) is None
    # one frame per 0.1 s policy step written at 30 fps is a 3x time-lapse; 10 fps is real time
    assert np.isclose(playback_speedup(30, 0.1), 3.0) and np.isclose(playback_speedup(10, 0.1), 1.0)
    assert np.isclose(playback_speedup(30, 0.1, stride=2), 6.0)
    assert np.isclose(realtime_fps(0.1), 10.0) and np.isclose(realtime_fps(0.1, stride=2), 5.0)


if __name__ == "__main__":
    main_for(globals(), "test_video")
