"""Run with python -m unittest discover -s modules/angelscript/tests -p test_three_language_benchmark.py."""
import unittest

from three_language_benchmark import LANGS, compute_summary

NAMES = ["bench_arith", "bench_calls", "bench_class", "bench_array", "bench_dictionary", "bench_packed",
         "bench_array_methods", "bench_string", "bench_vector", "bench_color", "bench_engine", "bench_hoist",
         "bench_typed_i64", "bench_typed_vec2", "bench_typed_map", "bench_batch_sum", "bench_dispatch"]


def fixture(runs=3, scale=None):
    scale = scale or {"as": 2.0, "cs": 1.0, "gd": 4.0}
    rows = []
    for process in range(1, runs + 1):
        for index, name in enumerate(NAMES):
            for lang in LANGS:
                median_us = int(scale[lang] * 100)
                rows.append(dict(process=process, workload=name, lang=lang, iters=100,
                                 min_us=int(median_us * 0.9), median_us=median_us, max_us=int(median_us * 1.1),
                                 checksum=str(index)))
    return rows


class ThreeLanguageTests(unittest.TestCase):
    def test_full_matrix_and_pairwise_ratios(self):
        result = compute_summary(fixture(), 3)
        self.assertEqual(len(result["summary"]), 17)
        arith = next(e for e in result["summary"] if e["workload"] == "bench_arith")
        self.assertEqual(arith["as_median_ns"], 2000)
        self.assertEqual(arith["gd_median_ns"], 4000)
        self.assertEqual(arith["cs_median_ns"], 1000)
        self.assertAlmostEqual(arith["as_gd_ratio"], 0.5)
        self.assertAlmostEqual(arith["as_cs_ratio"], 2.0)
        self.assertAlmostEqual(arith["cs_gd_ratio"], 0.25)
        self.assertEqual(arith["fastest_lang"], "cs")
        self.assertTrue(arith["as_faster_than_gd"])
        self.assertEqual(result["aggregate"]["as_gd_geomean"], 0.5)
        self.assertEqual(result["aggregate"]["cs_fastest"], sorted(NAMES))
        self.assertEqual(result["aggregate"]["as_faster_than_gd"], sorted(NAMES))

    def test_boundary_increment_is_per_iteration(self):
        rows = fixture()
        for row in rows:
            if row["workload"] == "bench_engine":
                row.update(median_us=200, min_us=180, max_us=220)
            elif row["workload"] == "bench_hoist":
                row.update(median_us=100, min_us=90, max_us=110)
        aggregate = compute_summary(rows, 3)["aggregate"]["boundary_increment_ns"]
        self.assertEqual(aggregate, {"as": 1000.0, "cs": 1000.0, "gd": 1000.0})

    def test_language_checksum_mismatch_rejected(self):
        rows = fixture()
        rows[1]["checksum"] = "wrong"
        with self.assertRaises(ValueError):
            compute_summary(rows, 3)

    def test_missing_duplicate_and_process_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compute_summary(fixture()[:-1], 3)
        with self.assertRaises(ValueError):
            compute_summary(fixture() + [fixture()[0]], 3)
        with self.assertRaises(ValueError):
            compute_summary(fixture(runs=4), 3)
        with self.assertRaises(ValueError):
            compute_summary(fixture(), 1)

    def test_cross_process_checksum_and_timing_validation(self):
        rows = fixture()
        next(r for r in rows if r["process"] == 2 and r["workload"] == "bench_array")["checksum"] = "drift"
        with self.assertRaises(ValueError):
            compute_summary(rows, 3)
        rows = fixture()
        next(r for r in rows if r["workload"] == "bench_color")["median_us"] = 0
        with self.assertRaises(ValueError):
            compute_summary(rows, 3)
        rows = fixture()
        next(r for r in rows if r["workload"] == "bench_color")["min_us"] = 999999
        with self.assertRaises(ValueError):
            compute_summary(rows, 3)

    def test_ratio_interval_collapses_for_identical_processes(self):
        arith = next(e for e in compute_summary(fixture(runs=4), 4)["summary"] if e["workload"] == "bench_arith")
        self.assertEqual(arith["as_gd_ci_low"], 0.5)
        self.assertEqual(arith["as_gd_ci_high"], 0.5)

    def test_inconclusive_ratios_are_not_forced_to_a_winner(self):
        rows = fixture(scale={"as": 1.0, "cs": 1.0, "gd": 1.0})
        entry = compute_summary(rows, 3)["summary"][0]
        self.assertEqual(entry["as_gd_ratio"], 1.0)
        self.assertFalse(entry["as_faster_than_gd"])
        self.assertFalse(entry["cs_faster_than_as"])


if __name__ == "__main__":
    unittest.main()
