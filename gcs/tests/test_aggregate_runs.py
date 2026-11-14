"""scripts/aggregate_runs.py: the benchmark table of several runs, against hand-computed numbers."""

import contextlib
import csv
import importlib.util
import io
import json
import unittest
from pathlib import Path

from ._util import ROOT, rmtree, short_tmpdir


def load_script():
    spec = importlib.util.spec_from_file_location("aggregate_runs", ROOT / "scripts" / "aggregate_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AGG = load_script()


def write_run(root: Path, name: str, card: dict | None, *, end: str = "landed", seed=None, terrain_seed=None,
              evaluation: dict | None = None) -> Path:  # fmt: skip
    """A run folder with the parts of summary.json the aggregator reads (``card`` None: no scorecard)."""
    run = root / name
    run.mkdir(parents=True)
    summary = {
        "run": name,
        "end_reason": end,
        "mission": {"name": f"sar_seed{seed}", "seed": seed, "terrain_seed": terrain_seed},
        "evaluation": evaluation,
    }
    if card is not None:
        summary["scorecard"] = card
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def card(found, total, alarms, pin, **extra) -> dict:
    return {"people_found": found, "people_total": total, "false_alarms": alarms, "pin_error_m": pin, **extra}


class TestWilson(unittest.TestCase):
    def test_known_intervals(self):
        low, high = AGG.wilson_interval(9, 10)
        self.assertAlmostEqual(low, 0.5958, places=3)  # the textbook interval of 9 out of 10
        self.assertAlmostEqual(high, 0.9821, places=3)
        low, high = AGG.wilson_interval(10, 10)
        self.assertAlmostEqual(low, 0.7225, places=3)  # all found: the interval still reaches down to 72 %
        self.assertEqual(high, 1.0)
        low, high = AGG.wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2775, places=3)
        self.assertIsNone(AGG.wilson_interval(0, 0))

    def test_more_trials_narrow_the_interval(self):
        narrow = AGG.wilson_interval(90, 100)
        wide = AGG.wilson_interval(9, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(short_tmpdir("dgcs_agg"))

    def tearDown(self):
        rmtree(self.tmp)

    def test_read_run(self):
        extra = {"people_outside_area": {"found": 0, "total": 1}, "coverage_pct": 100.0, "flight_time_s": 376.0}
        run = write_run(self.tmp, "a", card(10, 10, 0, 0.04, **extra), seed=3, terrain_seed=17,
                        evaluation={"tp": 40, "ground_truth": 41})  # fmt: skip
        row = AGG.read_run(run)
        self.assertEqual((row["run"], row["seed"], row["terrain_seed"], row["end_reason"]), ("a", 3, 17, "landed"))
        self.assertEqual((row["people_found"], row["people_total"]), (10, 10))
        self.assertEqual((row["outside_found"], row["outside_total"]), (0, 1))
        self.assertEqual((row["objects_found"], row["objects_total"]), (40, 41))
        self.assertEqual(row["flight_time_s"], 376.0)
        self.assertIsNone(row["map_height_error_m"])  # not in this scorecard
        self.assertEqual(set(row), set(AGG.COLUMNS))
        self.assertIsNone(AGG.read_run(self.tmp / "missing"))
        self.assertIsNone(AGG.read_run(write_run(self.tmp, "no_card", None)))
        broken = self.tmp / "broken"
        broken.mkdir()
        (broken / "summary.json").write_text("{not json")
        self.assertIsNone(AGG.read_run(broken))

    def test_aggregate_against_hand_computed_numbers(self):
        rows = [
            AGG.read_run(write_run(self.tmp, "s0", card(10, 10, 0, 0.04), seed=0)),
            AGG.read_run(write_run(self.tmp, "s1", card(9, 10, 2, 0.06), seed=1, end="timeout")),
            AGG.read_run(write_run(self.tmp, "s2", card(11, 12, 1, None), seed=2)),  # no pin error reported
        ]
        agg = AGG.aggregate(rows)
        self.assertEqual((agg["runs"], agg["landed"]), (3, 2))
        people = agg["people"]
        self.assertEqual((people["found"], people["total"], people["runs"]), (30, 32, 3))
        self.assertAlmostEqual(people["rate"], 30 / 32)
        self.assertEqual(people["runs_all_found"], 1)
        self.assertEqual(people["interval95"], list(AGG.wilson_interval(30, 32)))
        self.assertAlmostEqual(people["per_run_rate"]["mean"], (1.0 + 0.9 + 11 / 12) / 3)
        self.assertAlmostEqual(people["per_run_rate"]["min"], 0.9)
        alarms = agg["false_alarms"]
        self.assertEqual((alarms["total"], alarms["runs_without"], alarms["n"]), (3, 1, 3))
        self.assertAlmostEqual(alarms["mean"], 1.0)
        self.assertEqual(alarms["max"], 2.0)
        pin = agg["pin_error_m"]  # the run without a pin error is left out of the mean, and the count says so
        self.assertEqual(pin["n"], 2)
        self.assertAlmostEqual(pin["mean"], 0.05)
        self.assertAlmostEqual(pin["std"], 0.0141421, places=6)  # sample standard deviation of 0.04 and 0.06
        self.assertIsNone(agg["map_height_error_m"])  # no run reports it
        self.assertIsNone(agg["objects"])

    def test_one_run_has_zero_spread(self):
        agg = AGG.aggregate([AGG.read_run(write_run(self.tmp, "only", card(5, 6, 0, 0.1)))])
        self.assertEqual(agg["pin_error_m"]["std"], 0.0)
        self.assertEqual(agg["people"]["runs_all_found"], 0)

    def test_main_writes_the_three_files(self):
        runs = [
            write_run(self.tmp, "bench_s0", card(10, 10, 0, 0.04, coverage_pct=100.0), seed=0),
            write_run(self.tmp, "bench_s1", card(9, 10, 2, 0.06, coverage_pct=99.5), seed=1),
            write_run(self.tmp, "bench_s2", None, seed=2),  # crashed before the report: listed, not counted
        ]
        out = self.tmp / "table"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = AGG.main([*map(str, runs), "--out", str(out)])
        self.assertEqual(rc, 0)
        with open(out / "benchmark.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["run"] for r in rows], ["bench_s0", "bench_s1"])
        self.assertEqual(rows[1]["false_alarms"], "2")
        self.assertEqual(rows[0]["pin_error_m"], "0.04")
        data = json.loads((out / "benchmark.json").read_text())
        self.assertEqual(data["failed"], ["bench_s2"])
        self.assertEqual(data["aggregate"]["people"]["found"], 19)
        text = (out / "benchmark.md").read_text()
        self.assertIn("19 / 20 (95.0 %", text)
        self.assertIn("| bench_s1 | 1 | - | landed | 9 / 10 | 2 |", text)
        self.assertIn("Without a scorecard (left out): bench_s2.", text)
        self.assertIn("19 / 20", printed.getvalue())

    def test_main_without_a_usable_run_fails(self):
        run = write_run(self.tmp, "empty", None)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = AGG.main([str(run)])
        self.assertEqual(rc, 1)
        self.assertIn("no run with a scorecard", err.getvalue())
        self.assertFalse((self.tmp / "benchmark").exists())

    def test_default_output_folder_is_next_to_the_first_run(self):
        run = write_run(self.tmp, "r0", card(1, 1, 0, 0.0))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(AGG.main([str(run)]), 0)
        self.assertTrue((self.tmp / "benchmark" / "benchmark.md").is_file())


if __name__ == "__main__":
    unittest.main()
