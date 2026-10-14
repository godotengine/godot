"""Run with python -m unittest discover -s modules/angelscript/tests -p test_interleaved_ab_benchmark.py."""
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import interleaved_ab_benchmark as runner

from interleaved_ab_benchmark import (MonitorTimeout, ProcessMonitor, build_order,
                                      paired_summary, validate_samples, wait_for_idle)
from binding_parity_benchmark import parse_output


def fixture(arm, pair=1, scale=1):
    result = []
    for i in range(17):
        for lang in ("as", "cs", "gd"):
            result.append(dict(arm=arm, pair=pair, workload=f"bench_{i}", lang=lang,
                               iters=100, min_us=10 * scale, median_us=20 * scale,
                               max_us=30 * scale, checksum=str(i)))
    return result


class InterleavedTests(unittest.TestCase):
    def test_balanced_order(self):
        self.assertEqual(build_order(4), [(1, "A"), (1, "B"), (2, "B"), (2, "A"),
                                         (3, "A"), (3, "B"), (4, "B"), (4, "A")])
        with self.assertRaises(ValueError):
            build_order(0)

    def test_original_parser_and_missing(self):
        text = "\n".join(f"BENCH|bench_{i}|{lang}|100|10|20|30|{i}" for i in range(17) for lang in ("as", "cs", "gd"))
        self.assertEqual(len(parse_output(text, "A", 1)), 51)
        with self.assertRaises(ValueError):
            parse_output(text.split("\n", 1)[1], "A", 1)

    def test_cross_arm_checksum_validation(self):
        rows = fixture("A") + fixture("B")
        validate_samples(rows, 1)
        rows[-1]["checksum"] = "wrong"
        with self.assertRaises(ValueError):
            validate_samples(rows, 1)

    def test_reject_missing_or_duplicate_rows(self):
        rows = fixture("A") + fixture("B")
        with self.assertRaises(ValueError):
            validate_samples(rows[:-1], 1)
        with self.assertRaises(ValueError):
            validate_samples(rows[:-1] + [rows[0]], 1)

    def test_reject_iteration_mismatch(self):
        rows = fixture("A") + fixture("B")
        rows[0]["iters"] += 1
        with self.assertRaises(ValueError):
            validate_samples(rows, 1)

    def test_reject_zero_timing(self):
        rows = fixture("A") + fixture("B")
        rows[0].update(min_us=0, median_us=0)
        with self.assertRaises(ValueError):
            validate_samples(rows, 1)

    def test_paired_regression(self):
        rows = [r for pair in range(1, 4) for arm, scale in (("A", 1), ("B", 2)) for r in fixture(arm, pair, scale)]
        result = paired_summary(rows, 3)[0]
        self.assertEqual(result["classification"], "supported_regression")
        self.assertEqual(result["paired_ba_median"], 2)
        self.assertEqual(result["paired_gd_ba_median"], 2)
        self.assertEqual(result["paired_normalized_ba_median"], 1)

    def test_pair_order_does_not_depend_on_input_order(self):
        rows = [r for pair in range(1, 4) for arm, scale in (("A", 2), ("B", 1)) for r in fixture(arm, pair, scale)]
        self.assertEqual(paired_summary(rows, 3), paired_summary(list(reversed(rows)), 3))

    def test_paired_estimates(self):
        rows = [r for pair in range(1, 4) for arm, scale in (("A", 2), ("B", 1)) for r in fixture(arm, pair, scale)]
        summary = paired_summary(rows, 3)
        self.assertEqual(len(summary), 17)
        self.assertEqual(summary[0]["paired_ba_median"], 0.5)
        self.assertEqual(summary[0]["paired_ba_ci_low"], 0.5)
        self.assertEqual(summary[0]["paired_ba_ci_high"], 0.5)
        self.assertEqual(summary[0]["b_as_gd_ratio"], 1.0)
        self.assertEqual(summary[0]["classification"], "supported_improvement")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleep_extra = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds + self.sleep_extra


class FakeMonitor:
    def __init__(self, clock, cpu=10, latency=0, late=False, window=1, age=0):
        self.clock, self.cpu, self.latency = clock, cpu, latency
        self.late, self.window, self.age = late, window, age
        self.deadlines = []
        self.resets = 0

    def reset(self, deadline):
        self.resets += 1

    def sample(self, deadline):
        self.deadlines.append(deadline)
        end = self.clock.now + self.latency
        self.clock.now = end if self.late else min(end, deadline)
        if end >= deadline and not self.late:
            raise MonitorTimeout("fake slow sampler")
        cpu = self.cpu.pop(0) if isinstance(self.cpu, list) else self.cpu
        return dict(cpu_percent=cpu, cpu_window_s=self.window,
                    cpu_read_monotonic=self.clock.now-self.age, timestamp=self.clock.now)


def blocked_worker(connection):
    # Deliberately never services a request; parent must time out independently.
    time.sleep(60)


def responsive_then_blocked_worker(connection):
    connection.recv()
    connection.send({"ok": True, "data": {}})
    connection.recv()
    time.sleep(60)


class IdleDeadlineTests(unittest.TestCase):
    def test_busy_gate_uses_wall_clock_not_sample_count(self):
        clock = FakeClock()
        monitor = FakeMonitor(clock, cpu=100, latency=1.5)
        result = wait_for_idle(monitor, budget=5, clock=clock, sleep=clock.sleep)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "budget_expired")
        self.assertEqual(result["elapsed_s"], 5)
        self.assertTrue(all(d <= 5 for d in monitor.deadlines))
        self.assertEqual(monitor.resets, 1)

    def test_three_fresh_idle_samples_pass(self):
        clock = FakeClock()
        result = wait_for_idle(FakeMonitor(clock), budget=60, clock=clock, sleep=clock.sleep)
        self.assertTrue(result["passed"])
        self.assertEqual(result["elapsed_s"], 3)
        self.assertEqual(result["samples"], 3)

    def test_busy_or_missing_resets_consecutive(self):
        clock = FakeClock()
        monitor = FakeMonitor(clock, cpu=[10, 10, None, 10, 25, 10, 10, 10])
        result = wait_for_idle(monitor, budget=10, clock=clock, sleep=clock.sleep)
        self.assertTrue(result["passed"])
        self.assertEqual(result["samples"], 8)

    def test_nonfinite_out_of_range_or_stale_never_pass(self):
        for kwargs in ({"cpu": None}, {"cpu": float("nan")}, {"cpu": -1},
                       {"cpu": float("inf")}, {"cpu": 101}, {"window": 4}, {"age": 3}):
            with self.subTest(kwargs=kwargs):
                clock = FakeClock()
                result = wait_for_idle(FakeMonitor(clock, **kwargs), budget=5, clock=clock, sleep=clock.sleep)
                self.assertFalse(result["passed"])

    def test_late_low_cpu_result_cannot_pass(self):
        clock = FakeClock()
        monitor = FakeMonitor(clock, latency=1, late=True)
        result = wait_for_idle(monitor, budget=5.5, clock=clock, sleep=clock.sleep)
        self.assertFalse(result["passed"])
        self.assertEqual(result["elapsed_s"], 6)
        self.assertEqual(result["overshoot_s"], 0.5)

    def test_scheduler_oversleep_does_not_start_sample(self):
        clock = FakeClock()
        clock.sleep_extra = 10
        monitor = FakeMonitor(clock)
        result = wait_for_idle(monitor, budget=5, clock=clock, sleep=clock.sleep)
        self.assertFalse(result["passed"])
        self.assertEqual(monitor.deadlines, [])
        self.assertEqual(result["overshoot_s"], 6)

    def test_exact_deadline_reply_and_reset_failure_do_not_pass(self):
        clock = FakeClock()
        result = wait_for_idle(FakeMonitor(clock, latency=1, late=True), budget=6,
                               clock=clock, sleep=clock.sleep)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "budget_expired")
        monitor = FakeMonitor(clock)
        monitor.reset = Mock(side_effect=RuntimeError("PDH initialization failed"))
        result = wait_for_idle(monitor, clock=clock, sleep=clock.sleep)
        self.assertEqual(result["reason"], "monitor_error")
        self.assertEqual(result["samples"], 0)

    def test_single_sample_timeout_is_bounded(self):
        clock = FakeClock()
        result = wait_for_idle(FakeMonitor(clock, latency=20), budget=60,
                               clock=clock, sleep=clock.sleep)
        self.assertEqual(result["reason"], "monitor_timeout")
        self.assertEqual(result["elapsed_s"], 4)

    def test_real_blocked_worker_init_and_sample_are_cancellable(self):
        for worker in (blocked_worker, responsive_then_blocked_worker):
            with self.subTest(worker=worker.__name__):
                monitor = ProcessMonitor(worker_target=worker)
                try:
                    if worker is responsive_then_blocked_worker:
                        monitor.reset(time.monotonic() + 10)
                    started = time.monotonic()
                    with self.assertRaises(MonitorTimeout):
                        monitor.sample(started + 0.2)
                    self.assertLess(time.monotonic() - started, 3)
                    with self.assertRaises(RuntimeError):
                        monitor.sample(time.monotonic() + 1)
                finally:
                    cleanup = monitor.close()
                self.assertFalse(cleanup["alive"])
                self.assertNotIn(monitor.pid, [p.pid for p in multiprocessing.active_children()])


class RunnerPolicyTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows runner policy")
    def test_failed_gate_never_launches_benchmark_or_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root / "a.exe", root / "b.exe"
            a.write_bytes(b"a")
            b.write_bytes(b"b")
            output = root / "attempt"
            args = ["runner", "--a", str(a), "--b", str(b), "--a-revision", "old", "--b-revision", "new",
                    "--output", str(output), "--affinity", "1"]
            monitor = Mock(pid=123)
            monitor.close.return_value = dict(alive=False)
            with patch.object(sys, "argv", args), patch.object(runner, "dependency_hashes", return_value={"dll": "x"}), \
                    patch.object(runner, "project_hashes", return_value={"Debug.dll": "x"}), \
                    patch.object(runner.subprocess, "run", return_value=Mock(stdout=b"power")), \
                    patch.object(runner, "ProcessMonitor", return_value=monitor), \
                    patch.object(runner, "wait_for_idle", return_value=dict(passed=False, reason="budget_expired", elapsed_s=60)), \
                    patch.object(runner, "run_child") as child:
                self.assertEqual(runner.main(), 2)
                child.assert_not_called()
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "idle_unavailable")
            self.assertEqual(result["completed_processes"], 0)
            self.assertTrue(all(result["final_identity_checks"].values()))
            self.assertEqual(json.loads((output / "processes.json").read_text()), [])
            self.assertFalse((output / "summary.csv").exists())
            self.assertEqual(len((output / "samples.csv").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
