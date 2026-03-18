#!/usr/bin/env python3
"""
Phase 1 Test Runner for Gaussian Splatting Module
Automates building, testing, benchmarking, and reporting.
"""

import os
import sys
import subprocess
import json
import time
import argparse
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing


class TestRunner:
    """Main test runner for Phase 1 tests."""

    def __init__(self, godot_path: str, project_path: str):
        self.godot_path = Path(godot_path)
        self.project_path = Path(project_path)
        self.module_path = self.project_path / "modules" / "gaussian_splatting"
        self.test_output_dir = self.module_path / "test_results"
        self.test_output_dir.mkdir(exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.system(),
            "cpu": platform.processor(),
            "cpu_count": multiprocessing.cpu_count(),
            "tests": {}
        }

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, and stderr."""
        print(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd or self.project_path,
            universal_newlines=True
        )

        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr

    def build_module(self, config: str = "debug") -> bool:
        """Build the Gaussian Splatting module."""
        print("\n=== Building Gaussian Splatting Module ===")

        scons_cmd = ["scons"]

        # Platform-specific build options
        if platform.system() == "Windows":
            scons_cmd.extend(["platform=windows", "tools=yes"])
        elif platform.system() == "Linux":
            scons_cmd.extend(["platform=linux", "tools=yes"])
        elif platform.system() == "Darwin":
            scons_cmd.extend(["platform=macos", "tools=yes"])

        # Build configuration
        if config == "debug":
            scons_cmd.extend(["debug_symbols=yes", "optimize=debug"])
        elif config == "release":
            scons_cmd.extend(["optimize=speed"])
        elif config == "profile":
            scons_cmd.extend(["debug_symbols=yes", "optimize=speed"])

        # Parallel build
        scons_cmd.append(f"-j{multiprocessing.cpu_count()}")

        start_time = time.time()
        exit_code, stdout, stderr = self.run_command(scons_cmd, self.godot_path)
        build_time = time.time() - start_time

        self.results["build"] = {
            "success": exit_code == 0,
            "time_seconds": build_time,
            "config": config
        }

        if exit_code != 0:
            print(f"Build failed:\n{stderr}")
            return False

        print(f"Build completed in {build_time:.2f} seconds")
        return True

    def run_unit_tests(self) -> bool:
        """Run unit tests using Godot's test framework."""
        print("\n=== Running Unit Tests ===")

        test_categories = [
            "test_gpu_streaming",
            "test_gpu_sorting",
            "test_phase1_integration"
        ]

        all_passed = True
        unit_test_results = {}

        for test_category in test_categories:
            print(f"\nRunning {test_category}...")

            cmd = [
                str(self.godot_path / "bin" / self._get_godot_binary()),
                "--test",
                f"--test-suite={test_category}",
                "--headless"
            ]

            start_time = time.time()
            exit_code, stdout, stderr = self.run_command(cmd)
            test_time = time.time() - start_time

            # Parse test output
            passed = 0
            failed = 0
            for line in stdout.split('\n'):
                if "PASSED" in line:
                    passed += 1
                elif "FAILED" in line:
                    failed += 1

            test_passed = exit_code == 0 and failed == 0
            all_passed = all_passed and test_passed

            unit_test_results[test_category] = {
                "passed": test_passed,
                "tests_passed": passed,
                "tests_failed": failed,
                "time_seconds": test_time
            }

            print(f"  Result: {'PASSED' if test_passed else 'FAILED'}")
            print(f"  Tests: {passed} passed, {failed} failed")
            print(f"  Time: {test_time:.2f}s")

        self.results["tests"]["unit_tests"] = unit_test_results
        return all_passed

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("\n=== Running Integration Tests ===")

        integration_script = """
extends Node

func _ready():
    var tests = preload("res://modules/gaussian_splatting/tests/test_phase1_integration.cpp")
    var runner = tests.new()
    runner.run_all_tests()
    get_tree().quit(0 if runner.all_passed() else 1)
"""

        # Write temporary test script
        test_script_path = self.test_output_dir / "integration_test.gd"
        test_script_path.write_text(integration_script)

        cmd = [
            str(self.godot_path / "bin" / self._get_godot_binary()),
            "--script",
            str(test_script_path),
            "--headless"
        ]

        start_time = time.time()
        exit_code, stdout, stderr = self.run_command(cmd)
        test_time = time.time() - start_time

        self.results["tests"]["integration"] = {
            "passed": exit_code == 0,
            "time_seconds": test_time,
            "output": stdout
        }

        print(f"Integration tests: {'PASSED' if exit_code == 0 else 'FAILED'}")
        print(f"Time: {test_time:.2f}s")

        return exit_code == 0

    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        print("\n=== Running Performance Benchmarks ===")

        benchmark_configs = [
            {"splat_count": 1000, "frames": 100},
            {"splat_count": 10000, "frames": 100},
            {"splat_count": 100000, "frames": 100},
            {"splat_count": 500000, "frames": 50}
        ]

        benchmark_results = []

        for config in benchmark_configs:
            print(f"\nBenchmarking {config['splat_count']} splats...")

            benchmark_script = f"""
extends Node

func _ready():
    var benchmark = preload("res://modules/gaussian_splatting/tests/performance_benchmark.h")
    var runner = benchmark.PerformanceBenchmark.new()

    var config = benchmark.BenchmarkConfig.new()
    config.splat_count = {config['splat_count']}
    config.frame_count = {config['frames']}

    var result = runner.run_benchmark(config)
    print(result.to_json())

    get_tree().quit(0)
"""

            # Write temporary benchmark script
            bench_script_path = self.test_output_dir / f"benchmark_{config['splat_count']}.gd"
            bench_script_path.write_text(benchmark_script)

            cmd = [
                str(self.godot_path / "bin" / self._get_godot_binary()),
                "--script",
                str(bench_script_path),
                "--headless"
            ]

            exit_code, stdout, stderr = self.run_command(cmd)

            # Parse benchmark results
            try:
                # Extract JSON from output
                json_start = stdout.find('{')
                json_end = stdout.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    result_json = stdout[json_start:json_end]
                    result = json.loads(result_json)
                    benchmark_results.append(result)

                    # Check performance targets
                    if config['splat_count'] <= 100000:
                        fps = result.get('metrics', {}).get('avg_fps', 0)
                        if fps < 60:
                            print(f"  WARNING: Failed to meet 60 FPS target ({fps:.1f} FPS)")
            except json.JSONDecodeError:
                print(f"  Failed to parse benchmark results")

        self.results["tests"]["benchmarks"] = benchmark_results

        # Check if performance targets were met
        targets_met = all(
            r.get('metrics', {}).get('avg_fps', 0) >= 60
            for r in benchmark_results
            if r.get('config', {}).get('splat_count', 0) <= 100000
        )

        return targets_met

    def run_memory_validation(self) -> bool:
        """Run memory leak detection tests."""
        print("\n=== Running Memory Validation ===")

        memory_script = """
extends Node

func _ready():
    var validator = preload("res://modules/gaussian_splatting/tests/memory_validator.h")
    var mem_validator = validator.MemoryValidator.new()

    # Run memory tests
    mem_validator.stress_test_allocation_patterns(1000)
    mem_validator.stress_test_fragmentation(100)

    # Check for leaks
    var has_leaks = not mem_validator.validate_no_leaks()

    # Generate report
    var report = mem_validator.get_report_dict()
    print(JSON.stringify(report))

    get_tree().quit(1 if has_leaks else 0)
"""

        # Write temporary memory test script
        mem_script_path = self.test_output_dir / "memory_test.gd"
        mem_script_path.write_text(memory_script)

        cmd = [
            str(self.godot_path / "bin" / self._get_godot_binary()),
            "--script",
            str(mem_script_path),
            "--headless"
        ]

        start_time = time.time()
        exit_code, stdout, stderr = self.run_command(cmd)
        test_time = time.time() - start_time

        # Parse memory report
        memory_report = {}
        try:
            json_start = stdout.find('{')
            json_end = stdout.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                report_json = stdout[json_start:json_end]
                memory_report = json.loads(report_json)
        except json.JSONDecodeError:
            print("Failed to parse memory report")

        self.results["tests"]["memory"] = {
            "passed": exit_code == 0,
            "time_seconds": test_time,
            "report": memory_report
        }

        if exit_code == 0:
            print("Memory validation: PASSED (no leaks detected)")
        else:
            print("Memory validation: FAILED (leaks detected)")
            if memory_report:
                leaked = memory_report.get('stats', {}).get('leaked_allocations', 0)
                leaked_bytes = memory_report.get('stats', {}).get('leaked_bytes', 0)
                print(f"  Leaked allocations: {leaked}")
                print(f"  Leaked bytes: {leaked_bytes}")

        return exit_code == 0

    def run_visual_validation(self) -> bool:
        """Run visual validation tests."""
        print("\n=== Running Visual Validation ===")

        # For now, just check that visual validation compiles
        # Actual visual tests would require a display

        self.results["tests"]["visual"] = {
            "skipped": True,
            "reason": "Visual tests require display"
        }

        print("Visual validation: SKIPPED (headless mode)")
        return True

    def run_synthetic_baseline_validation(self) -> bool:
        """Validate deterministic synthetic splat baseline artifacts."""
        print("\n=== Running Synthetic Baseline Validation ===")

        script_path = self.module_path / "tests" / "generate_synthetic_splat_baselines.py"
        if not script_path.exists():
            self.results["tests"]["synthetic_baselines"] = {
                "passed": False,
                "time_seconds": 0.0,
                "error": f"Baseline generator script missing: {script_path}"
            }
            print("Synthetic baseline validation: FAILED (script missing)")
            return False

        python_cmd = [sys.executable, str(script_path), "--check"]
        python_start_time = time.time()
        python_exit_code, python_stdout, python_stderr = self.run_command(python_cmd, self.project_path)
        python_test_time = time.time() - python_start_time

        cpp_cmd = [
            str(self.godot_path / "bin" / self._get_godot_binary()),
            "--test",
            "--test-case=[GaussianSplatting][Synthetic]*",
            "--headless",
        ]
        cpp_start_time = time.time()
        cpp_exit_code, cpp_stdout, cpp_stderr = self.run_command(cpp_cmd)
        cpp_test_time = time.time() - cpp_start_time

        python_passed = python_exit_code == 0
        cpp_passed = cpp_exit_code == 0
        validation_passed = python_passed and cpp_passed

        self.results["tests"]["synthetic_baselines"] = {
            "passed": validation_passed,
            "time_seconds": python_test_time + cpp_test_time,
            "python_check": {
                "passed": python_passed,
                "time_seconds": python_test_time,
                "stdout_tail": python_stdout.splitlines()[-20:],
                "stderr_tail": python_stderr.splitlines()[-20:] if python_stderr else [],
            },
            "cpp_generator_check": {
                "passed": cpp_passed,
                "time_seconds": cpp_test_time,
                "suite": "test_synthetic_splat_generators",
                "stdout_tail": cpp_stdout.splitlines()[-20:],
                "stderr_tail": cpp_stderr.splitlines()[-20:] if cpp_stderr else [],
            },
        }

        if validation_passed:
            print("Synthetic baseline validation: PASSED")
            return True

        print("Synthetic baseline validation: FAILED")
        if not python_passed and python_stderr:
            print(python_stderr.strip())
        if not cpp_passed and cpp_stderr:
            print(cpp_stderr.strip())
        return False

    def check_regression(self) -> bool:
        """Check for performance regressions against baseline."""
        print("\n=== Checking for Regressions ===")

        baseline_file = self.test_output_dir / "baseline.json"

        if not baseline_file.exists():
            print("No baseline found, skipping regression check")
            self.results["regression"] = {"skipped": True}
            return True

        with open(baseline_file, 'r') as f:
            baseline = json.load(f)

        # Compare current results with baseline
        regressions = []

        # Check benchmark regressions
        if "benchmarks" in self.results.get("tests", {}):
            for current in self.results["tests"]["benchmarks"]:
                config = current.get("config", {})
                splat_count = config.get("splat_count", 0)

                # Find matching baseline
                for base in baseline.get("tests", {}).get("benchmarks", []):
                    if base.get("config", {}).get("splat_count") == splat_count:
                        current_fps = current.get("metrics", {}).get("avg_fps", 0)
                        base_fps = base.get("metrics", {}).get("avg_fps", 0)

                        if base_fps > 0:
                            regression = (base_fps - current_fps) / base_fps
                            if regression > 0.1:  # 10% regression threshold
                                regressions.append({
                                    "test": f"benchmark_{splat_count}",
                                    "baseline_fps": base_fps,
                                    "current_fps": current_fps,
                                    "regression_percent": regression * 100
                                })

        self.results["regression"] = {
            "has_regression": len(regressions) > 0,
            "regressions": regressions
        }

        if regressions:
            print("Performance regressions detected:")
            for reg in regressions:
                print(f"  {reg['test']}: {reg['regression_percent']:.1f}% slower")
            return False
        else:
            print("No regressions detected")
            return True

    def generate_report(self):
        """Generate test report in multiple formats."""
        print("\n=== Generating Test Report ===")

        # JSON report
        json_report_path = self.test_output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"JSON report: {json_report_path}")

        # Markdown report
        md_report = self._generate_markdown_report()
        md_report_path = self.test_output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        md_report_path.write_text(md_report)
        print(f"Markdown report: {md_report_path}")

        # Console summary
        self._print_summary()

    def _generate_markdown_report(self) -> str:
        """Generate markdown formatted report."""
        report = []
        report.append("# Phase 1 Test Report\n")
        report.append(f"**Date**: {self.results['timestamp']}\n")
        report.append(f"**Platform**: {self.results['platform']}\n")
        report.append(f"**CPU**: {self.results['cpu']} ({self.results['cpu_count']} cores)\n")

        # Build results
        if "build" in self.results:
            report.append("\n## Build Results\n")
            build = self.results["build"]
            report.append(f"- **Status**: {'✅ PASSED' if build['success'] else '❌ FAILED'}\n")
            report.append(f"- **Configuration**: {build['config']}\n")
            report.append(f"- **Time**: {build['time_seconds']:.2f} seconds\n")

        # Test results
        if "tests" in self.results:
            report.append("\n## Test Results\n")

            # Unit tests
            if "unit_tests" in self.results["tests"]:
                report.append("\n### Unit Tests\n")
                for name, result in self.results["tests"]["unit_tests"].items():
                    status = "✅" if result["passed"] else "❌"
                    report.append(f"- **{name}**: {status} ")
                    report.append(f"({result['tests_passed']} passed, {result['tests_failed']} failed, ")
                    report.append(f"{result['time_seconds']:.2f}s)\n")

            # Integration tests
            if "integration" in self.results["tests"]:
                report.append("\n### Integration Tests\n")
                result = self.results["tests"]["integration"]
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                report.append(f"- **Status**: {status}\n")
                report.append(f"- **Time**: {result['time_seconds']:.2f} seconds\n")

            if "synthetic_baselines" in self.results["tests"]:
                report.append("\n### Synthetic Baseline Validation\n")
                synthetic = self.results["tests"]["synthetic_baselines"]
                status = "✅ PASSED" if synthetic.get("passed") else "❌ FAILED"
                report.append(f"- **Status**: {status}\n")
                report.append(f"- **Time**: {synthetic.get('time_seconds', 0.0):.2f} seconds\n")

            # Benchmarks
            if "benchmarks" in self.results["tests"]:
                report.append("\n### Performance Benchmarks\n")
                report.append("\n| Splat Count | Avg FPS | Frame Time (ms) | GPU Memory (MB) |\n")
                report.append("|-------------|---------|-----------------|------------------|\n")

                for bench in self.results["tests"]["benchmarks"]:
                    config = bench.get("config", {})
                    metrics = bench.get("metrics", {})
                    report.append(f"| {config.get('splat_count', 0):,} | ")
                    report.append(f"{metrics.get('avg_fps', 0):.1f} | ")
                    report.append(f"{metrics.get('avg_frame_time_ms', 0):.2f} | ")
                    report.append(f"{metrics.get('peak_gpu_memory_mb', 0):.1f} |\n")

            # Memory validation
            if "memory" in self.results["tests"]:
                report.append("\n### Memory Validation\n")
                mem = self.results["tests"]["memory"]
                status = "✅ PASSED" if mem["passed"] else "❌ FAILED"
                report.append(f"- **Status**: {status}\n")

                if mem.get("report"):
                    stats = mem["report"].get("stats", {})
                    report.append(f"- **Peak CPU Memory**: {stats.get('peak_cpu_bytes', 0) / (1024*1024):.2f} MB\n")
                    report.append(f"- **Peak GPU Memory**: {stats.get('peak_gpu_bytes', 0) / (1024*1024):.2f} MB\n")

                    if stats.get('leaked_allocations', 0) > 0:
                        report.append(f"- **⚠️ Leaks**: {stats['leaked_allocations']} allocations ")
                        report.append(f"({stats.get('leaked_bytes', 0) / 1024:.2f} KB)\n")

        # Regression check
        if "regression" in self.results:
            report.append("\n## Regression Analysis\n")
            reg = self.results["regression"]

            if reg.get("skipped"):
                report.append("- Regression check skipped (no baseline)\n")
            elif reg.get("has_regression"):
                report.append("- **⚠️ Regressions detected**:\n")
                for r in reg.get("regressions", []):
                    report.append(f"  - {r['test']}: {r['regression_percent']:.1f}% slower\n")
            else:
                report.append("- ✅ No regressions detected\n")

        return ''.join(report)

    def _print_summary(self):
        """Print test summary to console."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        total_tests = 0
        passed_tests = 0

        # Count test results
        if "tests" in self.results:
            # Unit tests
            if "unit_tests" in self.results["tests"]:
                for result in self.results["tests"]["unit_tests"].values():
                    total_tests += 1
                    if result["passed"]:
                        passed_tests += 1

            # Other tests
            for test_type in ["integration", "memory", "synthetic_baselines"]:
                if test_type in self.results["tests"]:
                    total_tests += 1
                    if self.results["tests"][test_type].get("passed"):
                        passed_tests += 1

        print(f"Tests Passed: {passed_tests}/{total_tests}")

        # Performance summary
        if "benchmarks" in self.results.get("tests", {}):
            print("\nPerformance Summary:")
            for bench in self.results["tests"]["benchmarks"]:
                config = bench.get("config", {})
                metrics = bench.get("metrics", {})
                splats = config.get("splat_count", 0)
                fps = metrics.get("avg_fps", 0)

                status = "✅" if (splats <= 100000 and fps >= 60) or splats > 100000 else "❌"
                print(f"  {splats:,} splats: {fps:.1f} FPS {status}")

        # Overall result
        all_passed = passed_tests == total_tests and total_tests > 0

        if "regression" in self.results:
            if self.results["regression"].get("has_regression"):
                all_passed = False
                print("\n⚠️ Performance regressions detected!")

        print("\n" + "="*60)
        if all_passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*60)

    def save_baseline(self):
        """Save current results as baseline for future regression checks."""
        baseline_file = self.test_output_dir / "baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Baseline saved to: {baseline_file}")

    def _get_godot_binary(self) -> str:
        """Get the Godot binary name based on platform."""
        system = platform.system()
        if system == "Windows":
            return "godot.windows.editor.x86_64.exe"
        elif system == "Linux":
            candidates = [
                "godot.linuxbsd.editor.x86_64",
                "godot.linux.editor.x86_64",
            ]
            for candidate in candidates:
                if (self.godot_path / "bin" / candidate).exists():
                    return candidate
            return candidates[0]
        elif system == "Darwin":
            return "godot.macos.editor.universal"
        else:
            return "godot"

    def run_all(self, config: str = "debug") -> bool:
        """Run all tests in sequence."""
        print("="*60)
        print("PHASE 1 TEST RUNNER")
        print("="*60)

        all_passed = True

        # Build
        if not self.build_module(config):
            print("Build failed, aborting tests")
            return False

        # Run tests
        all_passed = all_passed and self.run_synthetic_baseline_validation()
        all_passed = all_passed and self.run_unit_tests()
        all_passed = all_passed and self.run_integration_tests()
        all_passed = all_passed and self.run_performance_benchmarks()
        all_passed = all_passed and self.run_memory_validation()
        all_passed = all_passed and self.run_visual_validation()

        # Check regression
        all_passed = all_passed and self.check_regression()

        # Generate report
        self.generate_report()

        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 1 Test Runner for Gaussian Splatting")
    parser.add_argument("--godot-path", required=True, help="Path to Godot source directory")
    parser.add_argument("--project-path", default=".", help="Path to project directory")
    parser.add_argument("--config", choices=["debug", "release", "profile"], default="debug",
                       help="Build configuration")
    parser.add_argument("--save-baseline", action="store_true",
                       help="Save results as baseline for regression tests")
    parser.add_argument("--skip-build", action="store_true",
                       help="Skip building and run tests only")
    parser.add_argument("--test-only", choices=["synthetic", "unit", "integration", "benchmark", "memory"],
                       help="Run only specific test category")

    args = parser.parse_args()

    # Create test runner
    runner = TestRunner(args.godot_path, args.project_path)

    # Run tests
    if args.test_only:
        # Run specific test category
        if args.test_only == "unit":
            success = runner.run_unit_tests()
        elif args.test_only == "synthetic":
            success = runner.run_synthetic_baseline_validation()
        elif args.test_only == "integration":
            success = runner.run_integration_tests()
        elif args.test_only == "benchmark":
            success = runner.run_performance_benchmarks()
        elif args.test_only == "memory":
            success = runner.run_memory_validation()

        runner.generate_report()
    else:
        # Run all tests
        success = runner.run_all(args.config)

    # Save baseline if requested
    if args.save_baseline and success:
        runner.save_baseline()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
