#!/usr/bin/env python3
"""
Baseline QA Test Runner for Gaussian Splatting CI
Runs the core test scripts and validates results with proper error reporting
"""

import argparse
import subprocess
import sys
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_QA_BASELINE_PATH = ROOT / "tests" / "ci" / "baselines" / "qa_results.json"
DEFAULT_BASELINE_REPORT_PATH = ROOT / "baseline_qa_regression_report.json"
DEFAULT_BASELINE_SUMMARY_PATH = ROOT / "baseline_qa_regression_summary.md"
SYNTHETIC_ASSET_PREP_SCRIPT = ROOT / "tests" / "runtime" / "prepare_synthetic_assets.py"

MINIMUM_SSIM_DROP = 0.02
MINIMUM_FPS_RATIO = 0.85
MAXIMUM_TIME_RATIO = 1.20
TEST_CATEGORIES = ("ply", "pipeline", "sorting", "runtime", "module", "qa")
CATEGORY_ALIASES = {"all": None}
CLI_CATEGORY_CHOICES = tuple(CATEGORY_ALIASES.keys()) + TEST_CATEGORIES


def resolve_root_path(path_value: str) -> Path:
    """Resolve relative paths from repository root for stable CI behavior."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def normalize_test_category(category: Optional[str]) -> Optional[str]:
    """Normalize CLI category aliases to canonical internal categories."""
    if category is None:
        return None
    if category in TEST_CATEGORIES:
        return category
    if category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[category]
    valid_categories = ", ".join(CLI_CATEGORY_CHOICES)
    raise ValueError(
        f"Unsupported category '{category}'. Expected one of: {valid_categories}"
    )


def prepare_synthetic_assets() -> None:
    if not SYNTHETIC_ASSET_PREP_SCRIPT.is_file():
        raise RuntimeError(
            f"Missing synthetic asset prep script: {SYNTHETIC_ASSET_PREP_SCRIPT.relative_to(ROOT)}"
        )

    command = [sys.executable, str(SYNTHETIC_ASSET_PREP_SCRIPT), "--quiet"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        if not detail:
            detail = f"exit code {result.returncode}"
        raise RuntimeError(f"Synthetic asset prep failed: {detail}")


class BaselineQARunner:
    def __init__(self, godot_binary: str = "godot"):
        self.godot_binary = godot_binary
        self.test_results = {
            "start_time": time.time(),
            "end_time": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "tests": [],
            "summary": {
                "overall_status": "running",
                "qa_baseline": {"status": "not_run"},
            },
        }

    @staticmethod
    def _is_expected_headless_qa_skip(test_name: str, command: List[str], stdout: str, stderr: str) -> bool:
        if test_name != "QA Scene Suite":
            return False
        if "--headless" not in command:
            return False

        merged_output = f"{stdout}\n{stderr}"
        required_markers = (
            "Failed to create primary local RenderingDevice",
            "Failed to create shared local RenderingDevice",
        )
        if not all(marker in merged_output for marker in required_markers):
            return False

        allowed_error_lines = {
            'ERROR: Parameter "t" is null.',
            "ERROR: [RENDERER][ERROR] [GaussianSplatSceneDirector] Unable to acquire local RenderingDevice for shared renderer",
        }
        for line in merged_output.splitlines():
            stripped = line.strip()
            if stripped.startswith("ERROR:") and stripped not in allowed_error_lines:
                return False

        return True

    def _record_qa_baseline_skipped(
        self,
        qa_results_path: Path,
        baseline_path: Path,
        report_path: Optional[Path],
        summary_path: Optional[Path],
        reason: str,
    ) -> bool:
        comparison: Dict[str, Any] = {
            "status": "skipped",
            "mode": "compare",
            "qa_results_path": str(qa_results_path),
            "baseline_path": str(baseline_path),
            "baseline_exists": baseline_path.exists(),
            "require_baseline": False,
            "thresholds": {
                "ssim_min_delta": MINIMUM_SSIM_DROP,
                "fps_min_ratio": MINIMUM_FPS_RATIO,
                "time_max_ratio": MAXIMUM_TIME_RATIO,
            },
            "scenes_checked": 0,
            "metrics_checked": 0,
            "missing_scenes": [],
            "new_scenes": [],
            "regressions": [],
            "notes": [reason],
            "timestamp_unix": time.time(),
        }
        print(f"[WARN] {reason} (skipping comparison)")
        self.test_results["summary"]["qa_baseline"] = comparison
        self._write_baseline_artifacts(comparison, report_path, summary_path)
        return True

    def run_test(self, test: Dict, timeout: Optional[int] = None) -> Tuple[bool, str, Dict]:
        """Run a single test entry (Godot script or arbitrary command)."""

        test_name = test["name"]
        test_type = test.get("type", "godot")
        timeout = test.get("timeout", timeout or 180)
        cwd = ROOT

        if test_type == "godot":
            command = [self.godot_binary]
            if not test.get("requires_gpu", False):
                command.append("--headless")
            command.extend(["--verbose", "--script", test["script"]])
            descriptor = test["script"]
        else:
            command = test["command"]
            descriptor = " ".join(command)

        print(f"\n[TEST] Running {test_name}...")
        print(f"   Command: {' '.join(command)}")

        test_start = time.time()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=cwd,
            )

            test_duration = time.time() - test_start
            success = result.returncode == 0
            output = (result.stdout or "") + (result.stderr or "")
            details = self._parse_test_output(output)
            test_status = "passed" if success else "failed"
            expected_headless_qa_skip = False
            if not success:
                expected_headless_qa_skip = self._is_expected_headless_qa_skip(
                    test_name,
                    command,
                    result.stdout or "",
                    result.stderr or "",
                )
                if expected_headless_qa_skip:
                    details["skip_reason"] = "QA Scene Suite requires local RenderingDevice when run with current headless configuration."
                    test_status = "skipped"
                    success = True

            test_result = {
                "name": test_name,
                "type": test_type,
                "descriptor": descriptor,
                "success": success,
                "status": test_status,
                "exit_code": result.returncode,
                "duration": test_duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "details": details,
                "skipped": expected_headless_qa_skip,
            }

            if success:
                if expected_headless_qa_skip:
                    print(f"   [SKIP] SKIPPED ({test_duration:.1f}s) - local RenderingDevice unavailable in headless mode")
                    self.test_results["skipped_tests"] += 1
                else:
                    print(f"   [PASS] PASSED ({test_duration:.1f}s)")
                    self.test_results["passed_tests"] += 1
            else:
                print(f"   [FAIL] FAILED ({test_duration:.1f}s) - Exit code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                self.test_results["failed_tests"] += 1

            self.test_results["tests"].append(test_result)
            return success, output, details

        except subprocess.TimeoutExpired:
            test_duration = time.time() - test_start
            error_msg = f"Test timed out after {timeout}s"
            print(f"   [TIMEOUT] TIMEOUT ({test_duration:.1f}s)")

            test_result = {
                "name": test_name,
                "type": test_type,
                "descriptor": descriptor,
                "success": False,
                "exit_code": -1,
                "duration": test_duration,
                "stdout": "",
                "stderr": error_msg,
                "details": {"error": error_msg},
            }

            self.test_results["failed_tests"] += 1
            self.test_results["tests"].append(test_result)
            return False, error_msg, {"error": error_msg}

        except Exception as e:
            test_duration = time.time() - test_start
            error_msg = f"Exception running test: {str(e)}"
            print(f"   [EXCEPTION] EXCEPTION ({test_duration:.1f}s): {str(e)}")

            test_result = {
                "name": test_name,
                "type": test_type,
                "descriptor": descriptor,
                "success": False,
                "exit_code": -2,
                "duration": test_duration,
                "stdout": "",
                "stderr": error_msg,
                "details": {"error": error_msg},
            }

            self.test_results["failed_tests"] += 1
            self.test_results["tests"].append(test_result)
            return False, error_msg, {"error": error_msg}

    def _parse_test_output(self, output: str) -> Dict:
        """Parse test output for key metrics and details"""
        details = {}

        # Look for test results patterns
        lines = output.split('\n')
        for line in lines:
            if "Total Tests:" in line:
                try:
                    details["total_tests"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "Passed:" in line:
                try:
                    details["passed_tests"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "Failed:" in line:
                try:
                    details["failed_tests"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "sort_time=" in line:
                try:
                    # Extract sort time from performance output
                    parts = line.split("sort_time=")
                    if len(parts) > 1:
                        time_str = parts[1].split("ms")[0]
                        details["sort_time_ms"] = float(time_str)
                except:
                    pass
            elif "throughput=" in line:
                try:
                    # Extract throughput
                    parts = line.split("throughput=")
                    if len(parts) > 1:
                        throughput_str = parts[1].split("M/s")[0]
                        throughput_mps = float(throughput_str)
                        details["throughput_mps"] = throughput_mps
                except:
                    pass

        return details

    def run_all_tests(self, quick: bool = False, category: Optional[str] = None, categories: Optional[set] = None) -> bool:
        """Run baseline QA tests with optional filtering."""
        print("=== Baseline QA Test Suite ===")
        try:
            prepare_synthetic_assets()
            print("[PASS] Synthetic asset prep complete.")
        except RuntimeError as exc:
            print(f"[FAIL] {exc}")
            self.test_results["total_tests"] = 0
            self.test_results["failed_tests"] = 1
            self.test_results["end_time"] = time.time()
            return False
        try:
            category = normalize_test_category(category)
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            self.test_results["total_tests"] = 0
            self.test_results["end_time"] = time.time()
            return False

        qa_output_path = str((ROOT / "tests" / "ci" / "qa_results.json").resolve())
        tests: List[Dict] = [
            {
                "name": "PLY Loader Tests",
                "type": "godot",
                "script": "tests/ci/test_ply_loader_ci.gd",
                "category": "ply",
            },
            {
                "name": "PLY Pipeline Tests",
                "type": "godot",
                "script": "tests/ci/test_ply_pipeline_ci.gd",
                "category": "pipeline",
            },
            {
                "name": "GPU Sorting Tests",
                "type": "godot",
                "script": "tests/ci/test_gpu_sorting_ci.gd",
                "category": "sorting",
                "requires_gpu": True,
            },
            {
                "name": "Runtime Validation Suite",
                "type": "command",
                "command": [
                    sys.executable,
                    "tests/runtime/run_runtime_validation.py",
                    "--profile",
                    "release-ci",
                    "--godot-binary",
                    self.godot_binary,
                ],
                "timeout": 600,
                "category": "runtime",
            },
            {
                "name": "Module Test Suite (GaussianSplatting)",
                "type": "command",
                "command": [sys.executable, "tests/ci/run_module_tests.py", "--godot-binary", self.godot_binary],
                "timeout": 900,
                "category": "module",
            },
            {
                "name": "QA Scene Suite",
                "type": "command",
                "command": [
                    self.godot_binary,
                    "--headless",
                    "--path",
                    "tests/examples/godot/test_project",
                    "--script",
                    "res://scripts/qa_test_runner.gd",
                    "--qa-output",
                    qa_output_path,
                ],
                "timeout": 900,
                "category": "qa",
            },
        ]

        selected_tests = tests
        if categories:
            selected_tests = [test for test in tests if test.get("category") in categories]
        elif category:
            selected_tests = [test for test in tests if test.get("category") == category]
        elif quick:
            quick_categories = {"ply", "sorting"}
            selected_tests = [test for test in tests if test.get("category") in quick_categories]

        if not selected_tests:
            print(f"No tests selected for category='{category}'.")
            self.test_results["total_tests"] = 0
            self.test_results["end_time"] = time.time()
            return True

        self.test_results["total_tests"] = len(selected_tests)

        for test in selected_tests:
            self.run_test(test)

        self.test_results["end_time"] = time.time()
        return self.test_results["failed_tests"] == 0

    def _load_json_file(self, path: Path, label: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to parse {label} JSON at {path}: {exc}")
            return None

    def _metric_rule(self, metric_name: str) -> Optional[Dict[str, Any]]:
        metric = metric_name.lower()
        if "ssim" in metric:
            return {"kind": "minimum_delta", "value": MINIMUM_SSIM_DROP}
        if "fps" in metric:
            return {"kind": "minimum_ratio", "value": MINIMUM_FPS_RATIO}
        if "frame_time" in metric or metric.endswith("_ms") or metric.endswith("_time_ms"):
            return {"kind": "maximum_ratio", "value": MAXIMUM_TIME_RATIO}
        return None

    def _build_baseline_summary_markdown(self, comparison: Dict[str, Any]) -> str:
        status = str(comparison.get("status", "unknown"))
        icon = "[PASS]" if status in {"passed", "updated"} else "[WARN]" if status == "skipped" else "[FAIL]"
        lines = [
            "# QA Baseline Regression Summary",
            "",
            f"- Status: {icon} `{status}`",
            f"- Mode: `{comparison.get('mode', 'unknown')}`",
            f"- Baseline path: `{comparison.get('baseline_path', '')}`",
            f"- Baseline present: `{comparison.get('baseline_exists', False)}`",
            f"- Scenes checked: `{comparison.get('scenes_checked', 0)}`",
            f"- Metrics checked: `{comparison.get('metrics_checked', 0)}`",
            f"- Regressions: `{len(comparison.get('regressions', []))}`",
            "",
        ]

        missing_scenes = comparison.get("missing_scenes", [])
        if missing_scenes:
            lines.append("## Missing Scenes")
            for scene in missing_scenes:
                lines.append(f"- `{scene}`")
            lines.append("")

        regressions = comparison.get("regressions", [])
        if regressions:
            lines.extend(
                [
                    "## Regressions",
                    "| Scene | Metric | Baseline | Current | Rule |",
                    "| --- | --- | ---: | ---: | --- |",
                ]
            )
            for entry in regressions:
                lines.append(
                    "| {scene} | {metric} | {baseline:.6f} | {current:.6f} | {rule} |".format(
                        scene=entry.get("scene", ""),
                        metric=entry.get("metric", ""),
                        baseline=float(entry.get("baseline", 0.0)),
                        current=float(entry.get("current", 0.0)),
                        rule=entry.get("rule", ""),
                    )
                )
            lines.append("")

        new_scenes = comparison.get("new_scenes", [])
        if new_scenes:
            lines.append("## New Scenes (not in baseline)")
            for scene in new_scenes:
                lines.append(f"- `{scene}`")
            lines.append("")

        notes = comparison.get("notes", [])
        if notes:
            lines.append("## Notes")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _write_baseline_artifacts(
        self,
        comparison: Dict[str, Any],
        report_path: Optional[Path],
        summary_path: Optional[Path],
    ) -> None:
        if report_path is not None:
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
                print(f"[REPORT] QA baseline report saved to {report_path}")
            except Exception as exc:
                print(f"[WARN] Could not write QA baseline report: {exc}")

        if summary_path is not None:
            try:
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(self._build_baseline_summary_markdown(comparison), encoding="utf-8")
                print(f"[NOTE] QA baseline summary saved to {summary_path}")
            except Exception as exc:
                print(f"[WARN] Could not write QA baseline summary: {exc}")

    def compare_qa_baseline(
        self,
        qa_results_path: Path,
        baseline_path: Path,
        update_baseline: bool = False,
        require_baseline: bool = False,
        report_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
    ) -> bool:
        """Store or compare QA results against baseline snapshots."""
        comparison: Dict[str, Any] = {
            "status": "not_run",
            "mode": "update" if update_baseline else "compare",
            "qa_results_path": str(qa_results_path),
            "baseline_path": str(baseline_path),
            "baseline_exists": baseline_path.exists(),
            "require_baseline": require_baseline,
            "thresholds": {
                "ssim_min_delta": MINIMUM_SSIM_DROP,
                "fps_min_ratio": MINIMUM_FPS_RATIO,
                "time_max_ratio": MAXIMUM_TIME_RATIO,
            },
            "scenes_checked": 0,
            "metrics_checked": 0,
            "missing_scenes": [],
            "new_scenes": [],
            "regressions": [],
            "notes": [],
            "timestamp_unix": time.time(),
        }

        if not qa_results_path.exists():
            message = f"QA results not found at {qa_results_path}"
            print(f"[WARN] {message}")
            comparison["status"] = "failed"
            comparison["notes"].append(message)
            self.test_results["summary"]["qa_baseline"] = comparison
            self._write_baseline_artifacts(comparison, report_path, summary_path)
            return False

        current = self._load_json_file(qa_results_path, "QA results")
        if current is None:
            comparison["status"] = "failed"
            comparison["notes"].append("QA results JSON is invalid.")
            self.test_results["summary"]["qa_baseline"] = comparison
            self._write_baseline_artifacts(comparison, report_path, summary_path)
            return False

        current_results = current.get("results", [])
        if not isinstance(current_results, list):
            comparison["status"] = "failed"
            comparison["notes"].append("QA results payload missing list field 'results'.")
            self.test_results["summary"]["qa_baseline"] = comparison
            self._write_baseline_artifacts(comparison, report_path, summary_path)
            return False

        if update_baseline:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(qa_results_path.read_text(encoding="utf-8"), encoding="utf-8")
            comparison["baseline_exists"] = True
            comparison["status"] = "updated"
            comparison["scenes_checked"] = len(current_results)
            comparison["notes"].append(f"Baseline refreshed from current QA output ({len(current_results)} scenes).")
            print(f"[PASS] QA baseline updated at {baseline_path}")
            self.test_results["summary"]["qa_baseline"] = comparison
            self._write_baseline_artifacts(comparison, report_path, summary_path)
            return True

        if not baseline_path.exists():
            message = f"QA baseline missing at {baseline_path}"
            if require_baseline:
                print(f"[FAIL] {message}")
                comparison["status"] = "failed"
                comparison["notes"].append(f"{message} (required)")
                self.test_results["summary"]["qa_baseline"] = comparison
                self._write_baseline_artifacts(comparison, report_path, summary_path)
                return False

            print(f"[WARN] {message} (skipping comparison)")
            comparison["status"] = "skipped"
            comparison["notes"].append(f"{message} (comparison skipped)")
            self.test_results["summary"]["qa_baseline"] = comparison
            self._write_baseline_artifacts(comparison, report_path, summary_path)
            return True

        baseline = self._load_json_file(baseline_path, "QA baseline")
        if baseline is None:
            comparison["status"] = "failed"
            comparison["notes"].append("QA baseline JSON is invalid.")
            self.test_results["summary"]["qa_baseline"] = comparison
            self._write_baseline_artifacts(comparison, report_path, summary_path)
            return False

        baseline_results = baseline.get("results", [])
        if not isinstance(baseline_results, list):
            comparison["status"] = "failed"
            comparison["notes"].append("QA baseline payload missing list field 'results'.")
            self.test_results["summary"]["qa_baseline"] = comparison
            self._write_baseline_artifacts(comparison, report_path, summary_path)
            return False

        baseline_map = {
            str(entry.get("scene", "")).strip(): entry
            for entry in baseline_results
            if str(entry.get("scene", "")).strip()
        }
        current_map = {
            str(entry.get("scene", "")).strip(): entry
            for entry in current_results
            if str(entry.get("scene", "")).strip()
        }

        comparison["scenes_checked"] = len(baseline_map)
        comparison["missing_scenes"] = sorted([scene for scene in baseline_map if scene not in current_map])
        comparison["new_scenes"] = sorted([scene for scene in current_map if scene not in baseline_map])

        for scene_name in sorted(baseline_map.keys()):
            if scene_name not in current_map:
                continue
            baseline_metrics = baseline_map[scene_name].get("metrics", {}) or {}
            current_metrics = current_map[scene_name].get("metrics", {}) or {}
            if not isinstance(baseline_metrics, dict) or not isinstance(current_metrics, dict):
                continue

            for metric_name in sorted(baseline_metrics.keys()):
                baseline_value = baseline_metrics.get(metric_name)
                current_value = current_metrics.get(metric_name)
                if not isinstance(baseline_value, (int, float)) or not isinstance(current_value, (int, float)):
                    continue

                rule = self._metric_rule(metric_name)
                if rule is None:
                    continue

                baseline_num = float(baseline_value)
                current_num = float(current_value)
                comparison["metrics_checked"] += 1
                rule_kind = str(rule["kind"])
                rule_value = float(rule["value"])

                passes = True
                threshold = baseline_num
                rule_text = ""
                if rule_kind == "minimum_delta":
                    threshold = baseline_num - rule_value
                    rule_text = f"current >= baseline - {rule_value:.3f}"
                    passes = current_num >= threshold
                elif rule_kind == "minimum_ratio":
                    if abs(baseline_num) < 1e-9:
                        continue
                    threshold = baseline_num * rule_value
                    rule_text = f"current >= baseline * {rule_value:.3f}"
                    passes = current_num >= threshold
                elif rule_kind == "maximum_ratio":
                    if abs(baseline_num) < 1e-9:
                        continue
                    threshold = baseline_num * rule_value
                    rule_text = f"current <= baseline * {rule_value:.3f}"
                    passes = current_num <= threshold

                if not passes:
                    comparison["regressions"].append(
                        {
                            "scene": scene_name,
                            "metric": metric_name,
                            "baseline": baseline_num,
                            "current": current_num,
                            "threshold": threshold,
                            "rule": rule_text,
                        }
                    )

        has_regressions = bool(comparison["regressions"]) or bool(comparison["missing_scenes"])
        comparison["status"] = "failed" if has_regressions else "passed"
        self.test_results["summary"]["qa_baseline"] = comparison
        self._write_baseline_artifacts(comparison, report_path, summary_path)

        if has_regressions:
            print("\n[FAIL] QA baseline regression detected:")
            for scene_name in comparison["missing_scenes"]:
                print(f"   Missing current results for {scene_name}")
            for entry in comparison["regressions"]:
                print(
                    "   {scene}: {metric} baseline={baseline:.6f} current={current:.6f} threshold={threshold:.6f} ({rule})".format(
                        scene=entry["scene"],
                        metric=entry["metric"],
                        baseline=float(entry["baseline"]),
                        current=float(entry["current"]),
                        threshold=float(entry["threshold"]),
                        rule=str(entry["rule"]),
                    )
                )
            return False

        print(
            "[PASS] QA baseline comparison passed "
            f"({comparison['scenes_checked']} scenes, {comparison['metrics_checked']} metrics checked)"
        )
        if comparison["new_scenes"]:
            print(f"[INFO] Detected {len(comparison['new_scenes'])} new scene(s) not in baseline")
        return True

    def generate_report(self) -> None:
        """Generate comprehensive test report"""
        duration = self.test_results["end_time"] - self.test_results["start_time"]

        print("\n" + "="*60)
        print("BASELINE QA TEST REPORT")
        print("="*60)
        total_tests = self.test_results["total_tests"]
        passed_tests = self.test_results["passed_tests"]
        failed_tests = self.test_results["failed_tests"]
        skipped_tests = self.test_results["skipped_tests"]

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Skipped: {skipped_tests}")
        print(f"Duration: {duration:.1f} seconds")
        success_denominator = max(1, total_tests - skipped_tests)
        success_rate = (passed_tests / success_denominator * 100.0) if total_tests else 100.0
        print(f"Success Rate: {success_rate:.1f}%")
        self.test_results["summary"]["duration_seconds"] = duration
        self.test_results["summary"]["success_rate"] = success_rate
        baseline_status = self.test_results["summary"].get("qa_baseline", {})
        baseline_failed = isinstance(baseline_status, dict) and baseline_status.get("status") == "failed"
        overall_passed = failed_tests == 0 and not baseline_failed
        self.test_results["summary"]["overall_status"] = "passed" if overall_passed else "failed"

        # Detailed results
        print("\nDETAILED RESULTS:")
        for test in self.test_results["tests"]:
            test_status = test.get("status", "passed" if test.get("success") else "failed")
            if test_status == "skipped":
                status = "[SKIP] SKIP"
            elif test["success"]:
                status = "[PASS] PASS"
            else:
                status = "[FAIL] FAIL"
            print(f"  {status} {test['name']} ({test['duration']:.1f}s)")

            if test_status == "skipped":
                skip_reason = test.get("details", {}).get("skip_reason")
                if skip_reason:
                    print(f"       Reason: {skip_reason}")
            elif not test["success"]:
                print(f"       Exit Code: {test['exit_code']}")
                if test['stderr']:
                    print(f"       Error: {test['stderr'][:100]}...")

        # Performance summary
        print("\nPERFORMANCE METRICS:")
        for test in self.test_results["tests"]:
            if "sort_time_ms" in test["details"]:
                print(f"  {test['name']}: {test['details']['sort_time_ms']:.2f}ms sort time")
            throughput_mps = test["details"].get("throughput_mps")
            if throughput_mps is not None:
                print(f"  {test['name']}: {throughput_mps:.1f}M splats/second")

        if isinstance(baseline_status, dict) and baseline_status.get("status") != "not_run":
            print("\nQA BASELINE:")
            print(
                "  Status: {status} | scenes={scenes} | metrics={metrics} | regressions={regressions}".format(
                    status=baseline_status.get("status", "unknown"),
                    scenes=baseline_status.get("scenes_checked", 0),
                    metrics=baseline_status.get("metrics_checked", 0),
                    regressions=len(baseline_status.get("regressions", [])),
                )
            )

        # Save JSON report
        self._save_json_report()

        # CI summary
        if overall_passed:
            print("\n[PASS] ALL BASELINE QA TESTS PASSED!")
        elif failed_tests == 0 and baseline_failed:
            print("\n[WARN] QA BASELINE REGRESSION CHECK FAILED")
        else:
            print(f"\n[WARN] {self.test_results['failed_tests']} TEST(S) FAILED")

    def _save_json_report(self) -> None:
        """Save detailed JSON report for CI artifacts"""
        try:
            report_path = ROOT / "baseline_qa_results.json"
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n[REPORT] Detailed report saved to {report_path}")
        except Exception as e:
            print(f"[WARN] Could not save JSON report: {e}")

    def print_actionable_failures(self) -> None:
        """Print actionable error messages for failed tests"""
        failed_tests = [t for t in self.test_results["tests"] if not t["success"]]

        if not failed_tests:
            return

        print("\n" + "="*60)
        print("ACTIONABLE FAILURE ANALYSIS")
        print("="*60)

        for test in failed_tests:
            print(f"\n[FAIL] {test['name']} FAILED")
            descriptor = test.get('descriptor', '')
            if descriptor:
                print(f"   Command: {descriptor}")
            print(f"   Exit Code: {test['exit_code']}")
            print(f"   Duration: {test['duration']:.1f}s")

            # Analyze failure type
            if test['exit_code'] == -1:
                print("   [ISSUE] ISSUE: Test timed out")
                print("   [ACTION] ACTION: Check for infinite loops or very slow operations")
            elif test['exit_code'] == -2:
                print("   [ISSUE] ISSUE: Exception during test execution")
                print("   [ACTION] ACTION: Check test script syntax and dependencies")
            elif "RenderingDevice" in test.get('stderr', ''):
                print("   [ISSUE] ISSUE: GPU context not available")
                print("   [ACTION] ACTION: This is expected in headless CI - test should handle gracefully")
            elif "Failed to create" in test.get('stderr', ''):
                print("   [ISSUE] ISSUE: Object creation failed")
                print("   [ACTION] ACTION: Check if required modules are compiled and available")
            else:
                print("   [ISSUE] ISSUE: Functional test failure")
                print("   [ACTION] ACTION: Review test output and fix underlying implementation")

            if test['stderr']:
                print(f"   [NOTE] ERROR OUTPUT:")
                print(f"      {test['stderr'][:300]}...")


def main(argv: Optional[List[str]] = None):
    """Main entry point for baseline QA runner"""

    parser = argparse.ArgumentParser(description="Baseline QA Test Runner for Gaussian Splatting CI")
    parser.add_argument("--godot", help="Override path to the Godot binary.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast subset of checks (PLY loader + GPU sorting).",
    )
    parser.add_argument(
        "--category",
        choices=CLI_CATEGORY_CHOICES,
        help="Run only the specified test category. Use 'all' for the full suite.",
    )
    parser.add_argument(
        "--categories",
        help="Comma-separated list of test categories to run (e.g., 'ply,pipeline,runtime,module').",
    )
    parser.add_argument(
        "--qa-baseline",
        default=str(DEFAULT_QA_BASELINE_PATH.relative_to(ROOT)),
        help="Path to QA baseline JSON.",
    )
    parser.add_argument(
        "--update-qa-baseline",
        action="store_true",
        help="Update QA baseline with latest results.",
    )
    parser.add_argument(
        "--require-qa-baseline",
        action="store_true",
        help="Fail if baseline file is unavailable in compare mode.",
    )
    parser.add_argument(
        "--baseline-report",
        default=str(DEFAULT_BASELINE_REPORT_PATH.relative_to(ROOT)),
        help="Path to machine-readable QA baseline comparison JSON artifact.",
    )
    parser.add_argument(
        "--baseline-summary",
        default=str(DEFAULT_BASELINE_SUMMARY_PATH.relative_to(ROOT)),
        help="Path to human-readable QA baseline comparison Markdown artifact.",
    )
    args = parser.parse_args(argv)
    category = normalize_test_category(args.category)
    category_arg_provided = args.category is not None

    godot_binary = args.godot or os.environ.get('GODOT_BINARY', 'godot')
    if args.godot:
        os.environ['GODOT_BINARY'] = args.godot

    # Validate binary exists
    try:
        result = subprocess.run([godot_binary, '--version'],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        if result.returncode != 0:
            print(f"[FAIL] Godot binary not working: {godot_binary}")
            sys.exit(1)
        stdout_line = (result.stdout or "").strip()
        print(f"[INFO] Using Godot: {stdout_line}")
    except Exception as e:
        print(f"[FAIL] Could not find Godot binary '{godot_binary}': {e}")
        print("[ACTION] Set GODOT_BINARY environment variable or ensure 'godot' is in PATH")
        sys.exit(1)

    if args.quick and category_arg_provided:
        print("[WARN] Ignoring --quick because --category was provided.")
    if args.update_qa_baseline and args.require_qa_baseline:
        print("[WARN] Ignoring --require-qa-baseline because --update-qa-baseline was provided.")

    # Resolve --categories (plural) into a set if provided
    categories_set = None
    if args.categories:
        categories_set = {normalize_test_category(c.strip()) for c in args.categories.split(",")}

    # Run tests
    run_quick = args.quick and not category_arg_provided
    runner = BaselineQARunner(godot_binary)
    success = runner.run_all_tests(
        quick=run_quick,
        category=category,
        categories=categories_set,
    )

    qa_results_path = ROOT / "tests" / "ci" / "qa_results.json"
    qa_baseline_path = resolve_root_path(args.qa_baseline)
    baseline_report_path = resolve_root_path(args.baseline_report)
    baseline_summary_path = resolve_root_path(args.baseline_summary)
    if categories_set is not None:
        qa_ran = (None in categories_set) or ("qa" in categories_set)
    elif category is not None:
        qa_ran = category == "qa"
    else:
        qa_ran = not run_quick
    if qa_ran:
        qa_scene_result = next((test for test in runner.test_results["tests"] if test.get("name") == "QA Scene Suite"), None)
        qa_scene_skipped = bool(qa_scene_result and qa_scene_result.get("status") == "skipped")
        if qa_scene_skipped and args.update_qa_baseline:
            print("[FAIL] QA baseline update requested, but QA Scene Suite was skipped.")
            qa_ok = False
        elif qa_scene_skipped:
            skip_reason = (
                (qa_scene_result or {})
                .get("details", {})
                .get("skip_reason", "QA Scene Suite skipped; QA baseline comparison not applicable.")
            )
            qa_ok = runner._record_qa_baseline_skipped(
                qa_results_path=qa_results_path,
                baseline_path=qa_baseline_path,
                report_path=baseline_report_path,
                summary_path=baseline_summary_path,
                reason=skip_reason,
            )
        else:
            qa_ok = runner.compare_qa_baseline(
                qa_results_path=qa_results_path,
                baseline_path=qa_baseline_path,
                update_baseline=args.update_qa_baseline,
                require_baseline=args.require_qa_baseline and not args.update_qa_baseline,
                report_path=baseline_report_path,
                summary_path=baseline_summary_path,
            )
        success = success and qa_ok

    # Generate reports
    runner.generate_report()
    runner.print_actionable_failures()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
