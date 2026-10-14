"""Aborted attempts are evidence, not completed performance comparisons."""
import json
from pathlib import Path
import tempfile
import unittest

from interleaved_ab_analysis import analyze
from interleaved_ab_benchmark import verify_reference


class AnalysisTests(unittest.TestCase):
    def seed(self, root, status):
        for name, value in {"metadata": {"pairs": 6}, "processes": [],
                            "result": {"status": status},
                            "gates": [{"passed": False, "elapsed_s": 60, "reason": "budget_expired"}]}.items():
            (root / f"{name}.json").write_text(json.dumps(value))
        (root / "telemetry.jsonl").write_text(json.dumps(dict(phase="idle", cpu_percent=99,
                                                           cpu_window_s=1, gate_sample_valid=True))+"\n")

    def test_idle_unavailable_has_environment_but_no_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.seed(root, "idle_unavailable")
            result = analyze(root)
            self.assertFalse(result["performance_comparison_available"])
            self.assertEqual(result["idle_aggregate"]["cpu_percent"]["median"], 99)
            self.assertEqual(result["planned_processes"], 12)
            self.assertEqual(result["idle_gates_passed"], 0)

    def test_reject_summary_attached_to_aborted_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.seed(root, "idle_unavailable")
            (root / "summary.csv").write_text("stale summary")
            with self.assertRaises(ValueError):
                analyze(root)

    def test_reject_running_or_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.seed(root, "running")
            with self.assertRaises(ValueError):
                analyze(root)
            self.seed(root, "complete")
            (root / "summary.csv").write_text("workload\n")
            with self.assertRaises(ValueError):
                analyze(root)

    def test_preserved_reference_checks_all_content_not_paths(self):
        meta = dict(source_revisions={"A": "old", "B": "new"},
                    exes={"A": {"sha256": "aaa"}, "B": {"sha256": "bbb"}},
                    dependency_hashes={"A": {"dll": "hash"}, "B": {"dll": "hash"}},
                    project_hashes={"Debug.dll": "hash"})
        expected = dict(revisions=meta["source_revisions"], exes=meta["exes"],
                        dependencies=meta["dependency_hashes"], project=meta["project_hashes"])
        self.assertTrue(all(verify_reference(meta, expected).values()))
        for key in ("revisions", "exes", "dependencies", "project"):
            changed = dict(expected)
            changed[key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_reference(meta, changed)


if __name__ == "__main__":
    unittest.main()
