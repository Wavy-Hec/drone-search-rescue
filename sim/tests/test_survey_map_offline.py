"""``scripts/survey_map.py`` end to end without Isaac Sim: the real task config, a stand-in environment.

The simulator, the policy and ``launch_simulation`` are replaced by small fakes that follow the carrot with a
first-order lag and cast the 441 mapping rays on the real (CPU-generated) survey terrain. Everything the script
does around them runs for real: config resolution, flags, the flight-loop bookkeeping, partial-run saving on
Ctrl+C or an error, the map grid, the ground truth and the summary.

Needs torch and Isaac Lab importable (env_isaaclab); skipped elsewhere. No Kit, no GPU.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from _runner import main_for, require

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "survey_map.py"


class _Value:
    """Stands in for Isaac Lab's ProxyArray (``.torch``)."""

    def __init__(self, tensor):
        self.torch = tensor


class _FakeTerm:
    """The parts of SurveyCarrotCommand that survey_map.py reads, driven by the same carrot functions."""

    def __init__(self, cfg, num_envs: int, terrain_cfg, mesh):
        import torch

        from drone_lab.carrot import pad_paths
        from drone_lab.paths import build_env_paths, waypoints_array
        from drone_lab.terrain import terrain_area

        self.cfg = cfg
        self.area = terrain_area(terrain_cfg)
        v = mesh[0]
        z = v[self.area.inset(-0.5 * cfg.spacing - 0.5).contains(v[:, :2]), 2]
        self.terrain_z_min, self.terrain_z_max = float(z.min()), float(z.max())
        self.flight_altitude = self.terrain_z_max + cfg.altitude
        custom = waypoints_array(cfg.waypoints, self.flight_altitude) if cfg.waypoints else None
        self.waypoints = build_env_paths(cfg.pattern, num_envs, self.area, cfg.spacing, self.flight_altitude,
                                         waypoints=custom, direction=cfg.direction, turn=cfg.turn)
        self.path_points, self.path_arclength, self.path_length = pad_paths(self.waypoints, cfg.path_resolution)
        self.s = torch.zeros(num_envs)
        self.carrot_w = self.path_points[:, 0].clone()
        self.segment_index = torch.zeros(num_envs, dtype=torch.long)
        self.gate = torch.ones(num_envs)
        self.done = torch.zeros(num_envs, dtype=torch.bool)

    @property
    def progress(self):
        return self.s / self.path_length.clamp(min=1e-6)

    def update(self, pos, dt: float):
        from drone_lab.carrot import advance_carrot, interpolate_path

        self.s, self.carrot_w, self.gate = advance_carrot(self.s, self.path_points, self.path_arclength, pos,
                                                          self.cfg.speed, dt, self.cfg.max_lead, self.cfg.slow_lead)
        _, self.segment_index = interpolate_path(self.path_points, self.path_arclength, self.s)
        self.done |= self.s >= self.path_length - 1e-4

    def __str__(self):
        return f"FakeSurveyCarrotCommand({self.cfg.pattern}, {self.area.as_tuple()})"


class _FakeEnv:
    """Gymnasium-style env with the attributes survey_map.py uses; the drone lags the carrot."""

    def __init__(self, env_cfg):
        import torch

        from drone_lab.mapping import MeshHeightQuery
        from drone_lab.synthetic import grid_offsets
        from drone_lab.terrain import generate_terrain_mesh

        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = env_cfg.scene.num_envs
        self.step_dt = env_cfg.sim.dt * env_cfg.decimation
        mesh = generate_terrain_mesh(env_cfg.scene.terrain.terrain_generator)
        self._terrain = MeshHeightQuery(*mesh)
        self._offsets = grid_offsets()
        self.term = _FakeTerm(env_cfg.commands.survey_target, self.num_envs, env_cfg.scene.terrain.terrain_generator,
                              mesh)
        n = self.num_envs
        self._pos = self.term.path_points[:, 0].clone()
        self._vel = torch.zeros(n, 3)
        self.robot = type("Robot", (), {})()
        self.robot.data = type("Data", (), {})()
        self.scanner = type("Scanner", (), {"num_rays": len(self._offsets)})()
        self.scanner.data = type("ScanData", (), {})()
        self.command_manager = type("Cmd", (), {"get_term": lambda _self, name: self.term})()
        self.scene = type("Scene", (), {"sensors": {"mapping_scanner": self.scanner},
                                        "__getitem__": lambda _self, key: self.robot})()
        self.sim = type("Sim", (), {"visualizers": []})()
        self.closed = False
        self._publish()

    def _publish(self):
        import torch

        n = self.num_envs
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)
        self.robot.data.root_pos_w = _Value(self._pos.clone())
        self.robot.data.root_quat_w = _Value(quat)
        self.robot.data.root_lin_vel_w = _Value(self._vel.clone())
        self.robot.data.root_ang_vel_b = _Value(torch.zeros(n, 3))
        xy = (self._pos[:, None, :2].numpy() + self._offsets[None]).reshape(-1, 2).astype(np.float32)
        hits = np.column_stack([xy, self._terrain(xy.astype(np.float64))]).reshape(n, -1, 3)
        self.scanner.data.ray_hits_w = _Value(torch.as_tensor(hits, dtype=torch.float32))

    def _obs(self):
        import torch

        rel = self._pos - self.term.carrot_w
        policy = torch.cat([rel, self.robot.data.root_quat_w.torch, self._vel, torch.zeros(self.num_envs, 7)], dim=1)
        return {"policy": policy}

    def reset(self):
        return self._obs(), {}

    def step(self, actions):
        import torch

        new_pos = self._pos + 0.5 * (self.term.carrot_w - self._pos)  # lagging follower
        self._vel = (new_pos - self._pos) / self.step_dt
        self._pos = new_pos
        self.term.update(self._pos, self.step_dt)  # commands update after physics, as in the env
        self._publish()
        none = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.zeros(self.num_envs), none, none.clone(), {}

    def close(self):
        self.closed = True


class _Policy:
    """Zero actions; raises ``fail`` on call number ``fail_at`` (0-based) to emulate Ctrl+C or a crash."""

    def __init__(self, fail_at: int | None = None, fail: BaseException | None = None):
        self.calls = 0
        self.fail_at, self.fail = fail_at, fail

    def __call__(self, obs):
        import torch

        if self.fail_at is not None and self.calls == self.fail_at:
            raise self.fail
        self.calls += 1
        return torch.zeros(obs.shape[0], 4)


def _load_script(argv: list[str]):
    old = sys.argv
    sys.argv = ["survey_map.py", *argv]
    try:
        spec = importlib.util.spec_from_file_location("survey_map_offline", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


@contextlib.contextmanager
def _offline(mod, policy: _Policy):
    """Patch the simulator-facing calls of a loaded survey_map module; yields the created envs."""
    import gymnasium

    import drone_lab.tasks.survey.mdp as survey_mdp

    envs: list[_FakeEnv] = []

    @contextlib.contextmanager
    def fake_launch(env_cfg, args):
        yield None

    def fake_make(task, cfg=None, **kwargs):
        envs.append(_FakeEnv(cfg))
        return envs[-1]

    saved = (mod.launch_simulation, mod.load_policy, gymnasium.make, survey_mdp.terrain_mesh_from_stage)
    mod.launch_simulation = fake_launch
    mod.load_policy = lambda checkpoint, env, device: (policy, "fake")
    gymnasium.make = fake_make
    survey_mdp.terrain_mesh_from_stage = lambda prim_path="/World/ground": None  # -> CPU regeneration
    try:
        yield envs
    finally:
        mod.launch_simulation, mod.load_policy, gymnasium.make, survey_mdp.terrain_mesh_from_stage = saved


def _requirements():
    require("torch")
    require("gymnasium")
    require("isaaclab.app")
    require("isaaclab_tasks")


def test_full_flight_saves_run_with_centred_grid_and_sensor_error():
    _requirements()
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "full"
        mod = _load_script(["--checkpoint", "fake.pt", "--max_time", "30", "--no_plots", "--out", str(run_dir)])
        with _offline(mod, _Policy()) as envs:
            mod.main()
        assert envs[0].closed
        s = json.loads((run_dir / "summary.json").read_text())
        meta = json.loads((run_dir / "map_meta.json").read_text())
        assert s["interrupted"] is None and s["sim_time_s"] == 30.0 and s["ground_truth_source"] == "terrain_generator"
        assert np.isclose(meta["grid"]["x_min"], -20.125) and meta["grid"]["nx"] == 161  # GridSpec.for_survey
        assert s["map"]["error_area_at_hits"]["rmse"] < 1e-5  # exact stand-in rays
        assert s["map"]["coverage_area"] > 0.05
        n_rows = len((run_dir / "trajectory.csv").read_text().splitlines()) - 1
        assert n_rows == 300, n_rows
        assert "PARTIAL RUN" not in (run_dir / "summary.txt").read_text()


def test_ctrl_c_saves_partial_run_then_reraises():
    """Ctrl+C in the flight loop keeps every logged step, writes the plots, then exits as Ctrl+C."""
    _requirements()
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "ctrl_c"
        mod = _load_script(["--checkpoint", "fake.pt", "--out", str(run_dir)])
        with _offline(mod, _Policy(fail_at=50, fail=KeyboardInterrupt())):
            try:
                mod.main()
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("main() should re-raise KeyboardInterrupt after saving")
        s = json.loads((run_dir / "summary.json").read_text())
        assert s["interrupted"] == "stopped with Ctrl+C" and np.isclose(s["sim_time_s"], 5.0)
        assert len((run_dir / "trajectory.csv").read_text().splitlines()) - 1 == 50
        assert "PARTIAL RUN" in (run_dir / "summary.txt").read_text()
        for name in ("elevation_map.npy", "ground_truth.npy", "hits.npy", "map_with_trajectory.png", "map_error.png"):
            assert (run_dir / name).exists(), name


def test_error_saves_partial_run_then_reraises():
    _requirements()
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "error"
        mod = _load_script(["--checkpoint", "fake.pt", "--no_plots", "--out", str(run_dir)])
        with _offline(mod, _Policy(fail_at=20, fail=RuntimeError("simulated PhysX failure"))):
            try:
                mod.main()
            except RuntimeError as exc:
                assert "simulated PhysX failure" in str(exc)
            else:
                raise AssertionError("main() should re-raise the flight-loop error after saving")
        s = json.loads((run_dir / "summary.json").read_text())
        assert s["interrupted"].startswith("flight loop error: RuntimeError")
        assert len((run_dir / "trajectory.csv").read_text().splitlines()) - 1 == 20
        # an error before the first logged step leaves nothing to save, and still re-raises
        empty_dir = Path(d) / "empty"
        mod = _load_script(["--checkpoint", "fake.pt", "--no_plots", "--out", str(empty_dir)])
        with _offline(mod, _Policy(fail_at=0, fail=RuntimeError("boom"))):
            try:
                mod.main()
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError")
        assert not (empty_dir / "summary.json").exists()


def test_settle_holds_the_carrot_at_the_start():
    """--settle: the carrot waits at the path start (the drone recovers from its spawn dip), then flies at the
    commanded speed; the summary keeps the commanded speed, also after a Ctrl+C during the settle."""
    _requirements()
    from drone_lab.survey_io import read_trajectory_csv

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "settle"
        mod = _load_script(["--checkpoint", "fake.pt", "--max_time", "5", "--settle", "2", "--no_plots",
                            "--out", str(run_dir)])
        with _offline(mod, _Policy()) as envs:
            mod.main()
        traj = read_trajectory_csv(run_dir / "trajectory.csv")
        held = traj["time"] < 2.0 + 1e-6  # rows are logged after each 0.1 s step: t = 0.1 ... 2.0 is the settle
        assert held.sum() == 20 and np.all(traj["path_s"][held] == 0.0), traj["path_s"][:25]
        assert np.all(traj["path_s"][~held] > 0.0) and np.isclose(traj["path_s"][-1], 3.0, atol=0.05)
        s = json.loads((run_dir / "summary.json").read_text())
        assert np.isclose(s["settle_s"], 2.0) and s["commanded_speed_mps"] == 1.0
        assert envs[0].term.cfg.speed == 1.0
        assert "hover at the start" in (run_dir / "summary.txt").read_text()

        # stopped while still hovering: the speed is restored before the summary is written
        stop_dir = Path(d) / "settle_ctrl_c"
        mod = _load_script(["--checkpoint", "fake.pt", "--settle", "2", "--no_plots", "--out", str(stop_dir)])
        with (
            _offline(mod, _Policy(fail_at=5, fail=KeyboardInterrupt())) as envs,
            contextlib.suppress(KeyboardInterrupt),
        ):
            mod.main()
        s = json.loads((stop_dir / "summary.json").read_text())
        assert s["commanded_speed_mps"] == 1.0 and envs[0].term.cfg.speed == 1.0

        # --settle 0: the carrot starts at once (the behaviour before the option existed)
        now_dir = Path(d) / "no_settle"
        mod = _load_script(["--checkpoint", "fake.pt", "--max_time", "1", "--settle", "0", "--no_plots",
                            "--out", str(now_dir)])
        with _offline(mod, _Policy()):
            mod.main()
        traj = read_trajectory_csv(now_dir / "trajectory.csv")
        assert traj["path_s"][0] > 0.0
        assert "hover" not in (now_dir / "summary.txt").read_text()


def test_waypoints_flag_conflicts_with_generated_pattern():
    """Passing --waypoints with --pattern lawnmower exits instead of silently flying the lawnmower."""
    _requirements()
    with tempfile.TemporaryDirectory() as d:
        wp = Path(d) / "wp.csv"
        wp.write_text("x,y\n-10,-10\n10,-10\n10,10\n")
        mod = _load_script(["--checkpoint", "fake.pt", "--waypoints", str(wp), "--pattern", "lawnmower",
                            "--print_config"])
        try:
            mod.main()
        except SystemExit as exc:
            assert "--waypoints needs --pattern waypoints" in str(exc.code)
        else:
            raise AssertionError("expected SystemExit")
        import io

        mod = _load_script(["--checkpoint", "fake.pt", "--waypoints", str(wp), "--print_config"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mod.main()  # the file implies the waypoints pattern
        assert "survey: waypoints" in out.getvalue(), out.getvalue()


if __name__ == "__main__":
    main_for(globals(), "test_survey_map_offline")
