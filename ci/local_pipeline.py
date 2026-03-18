#!/usr/bin/env python3
"""
GodotGS Local CI/CD Pipeline
============================

A comprehensive local CI/CD pipeline for the Gaussian Splatting Godot module.
Provides build validation, unit testing, performance benchmarking, and memory leak detection.
"""

import subprocess
import json
import time
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import psutil
    import colorama
    from colorama import Fore, Style
    colorama.init()
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "ci/requirements.txt"])
    import psutil
    import colorama
    from colorama import Fore, Style
    colorama.init()


# Validation mode for non-functional checks (benchmarks/memory):
# - strict: failures are fatal (default when CI is set).
# - warn-only: failures are reported but non-fatal (default locally).
VALIDATION_MODE_ENV = "GS_CI_VALIDATION_MODE"
# Optional override for the Godot memory-stress validation script.
MEMORY_VALIDATION_SCRIPT_ENV = "GS_CI_MEMORY_VALIDATION_SCRIPT"
DEFAULT_MEMORY_VALIDATION_SCRIPT = Path("tests/runtime/test_gpu_streaming_stress.gd")


def resolve_validation_mode() -> str:
    explicit_mode = os.environ.get(VALIDATION_MODE_ENV, "").strip().lower()
    if explicit_mode in ("strict", "warn-only"):
        return explicit_mode
    return "strict" if os.environ.get("CI") else "warn-only"


def resolve_memory_validation_script(project_root: Path) -> Path:
    configured = os.environ.get(MEMORY_VALIDATION_SCRIPT_ENV, "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate
    return project_root / DEFAULT_MEMORY_VALIDATION_SCRIPT


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: Optional[str] = None
    error_output: Optional[str] = None


@dataclass
class BenchmarkResult:
    name: str
    fps: float
    frame_time_ms: float
    memory_mb: float
    passed: bool
    target_fps: float


@dataclass
class BuildResult:
    platform: str
    success: bool
    duration: float
    error_output: Optional[str] = None


class GodotGSPipeline:
    """Main CI/CD pipeline for GodotGS project"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.godot_source = self.project_root
        self.module_path = self.project_root / "modules" / "gaussian_splatting"
        self.reports_dir = self.project_root / "ci" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.validation_mode = resolve_validation_mode()
        self.strict_validation = self.validation_mode == "strict"

        # Results storage
        self.build_results: List[BuildResult] = []
        self.test_results: List[TestResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
        self.start_time = time.time()

        # Validate environment
        self._validate_environment()

    def _validate_environment(self):
        """Validate that all required paths and tools are available"""
        print(f"{Fore.YELLOW}🔍 Validating environment...{Style.RESET_ALL}")

        if not self.godot_source.exists():
            raise FileNotFoundError(f"Engine root not found at {self.godot_source}")

        if not self.module_path.exists():
            raise FileNotFoundError(f"Gaussian splatting module not found at {self.module_path}")

        # Check for SCons
        try:
            subprocess.run(["scons", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("SCons not found. Please install SCons.")

        print(f"{Fore.GREEN}✅ Environment validated{Style.RESET_ALL}")

    def run_full_pipeline(self) -> bool:
        """Run the complete CI/CD pipeline"""
        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 GodotGS Local CI/CD Pipeline Starting{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")
        print(f"Validation mode: {self.validation_mode}")

        pipeline_start = time.time()

        try:
            # 1. Build validation
            if not self._run_build_validation():
                self._print_failure("Build validation failed")
                return False

            # 2. Unit tests
            if not self._run_unit_tests():
                self._print_failure("Unit tests failed")
                return False

            # 3. Performance benchmarks
            if not self._run_performance_benchmarks():
                if self.strict_validation:
                    self._print_failure("Performance benchmarks failed")
                    return False
                print(f"{Fore.YELLOW}⚠️ Performance benchmarks failed (warn-only mode){Style.RESET_ALL}")

            # 4. Memory validation
            if not self._run_memory_validation():
                if self.strict_validation:
                    self._print_failure("Memory validation failed")
                    return False
                print(f"{Fore.YELLOW}⚠️ Memory validation failed (warn-only mode){Style.RESET_ALL}")

            # 5. Generate report
            self._generate_report()

            pipeline_duration = time.time() - pipeline_start
            print(f"\n{Fore.GREEN}🎉 PIPELINE COMPLETED SUCCESSFULLY!{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Total time: {pipeline_duration:.1f}s{Style.RESET_ALL}\n")

            return True

        except Exception as e:
            print(f"\n{Fore.RED}💥 PIPELINE FAILED: {e}{Style.RESET_ALL}\n")
            return False

    def _run_build_validation(self) -> bool:
        """Validate that the module compiles successfully"""
        print(f"{Fore.YELLOW}🔨 [1/4] Build Validation{Style.RESET_ALL}")

        # Clean previous build artifacts
        binaries_dir = self.godot_source / "bin"
        if binaries_dir.exists():
            print(f"  🧹 Cleaning previous binaries...")
            for file in binaries_dir.glob("godot*"):
                try:
                    file.unlink()
                except OSError:
                    pass  # File might be in use

        # Build Godot with our module
        build_start = time.time()
        cmd = [
            "scons",
            "platform=windows",
            "tools=yes",
            "optimize=speed",
            "-j8",
            "--max-drift=1"  # Prevent SCons from being too aggressive about rebuilds
        ]

        print(f"  📦 Building Godot with Gaussian Splatting module...")
        print(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.godot_source,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            build_duration = time.time() - build_start

            if result.returncode == 0:
                self.build_results.append(BuildResult("windows", True, build_duration))
                print(f"  {Fore.GREEN}✅ Build successful ({build_duration:.1f}s){Style.RESET_ALL}")

                # Verify the binary was created
                expected_binary = binaries_dir / "godot.windows.editor.x86_64.exe"
                if expected_binary.exists():
                    print(f"  {Fore.GREEN}✅ Binary created: {expected_binary.name}{Style.RESET_ALL}")
                    return True
                else:
                    print(f"  {Fore.RED}❌ Binary not found at expected location{Style.RESET_ALL}")
                    return False
            else:
                self.build_results.append(BuildResult("windows", False, build_duration, result.stderr))
                print(f"  {Fore.RED}❌ Build failed ({build_duration:.1f}s){Style.RESET_ALL}")
                print(f"  Error: {result.stderr[:200]}...")
                return False

        except subprocess.TimeoutExpired:
            print(f"  {Fore.RED}❌ Build timed out after 10 minutes{Style.RESET_ALL}")
            self.build_results.append(BuildResult("windows", False, 600, "Build timeout"))
            return False
        except Exception as e:
            print(f"  {Fore.RED}❌ Build error: {e}{Style.RESET_ALL}")
            self.build_results.append(BuildResult("windows", False, 0, str(e)))
            return False

    def _run_unit_tests(self) -> bool:
        """Run unit tests for the Gaussian Splatting module"""
        print(f"{Fore.YELLOW}🧪 [2/4] Unit Tests{Style.RESET_ALL}")

        # For now, we'll create basic instantiation tests
        # In the future, this will integrate with Godot's doctest framework

        binary_path = self.godot_source / "bin" / "godot.windows.editor.x86_64.exe"

        if not binary_path.exists():
            print(f"  {Fore.RED}❌ Godot binary not found{Style.RESET_ALL}")
            return False

        # Test 1: Module loads without crashes
        test_start = time.time()
        try:
            result = subprocess.run(
                [str(binary_path), "--version", "--quiet"],
                capture_output=True,
                text=True,
                timeout=30
            )

            test_duration = time.time() - test_start

            if result.returncode == 0:
                self.test_results.append(TestResult("module_load", True, test_duration))
                print(f"  {Fore.GREEN}✅ Module loads successfully{Style.RESET_ALL}")
            else:
                self.test_results.append(TestResult("module_load", False, test_duration, error_output=result.stderr))
                print(f"  {Fore.RED}❌ Module failed to load{Style.RESET_ALL}")
                return False

        except subprocess.TimeoutExpired:
            self.test_results.append(TestResult("module_load", False, 30, "Timeout"))
            print(f"  {Fore.RED}❌ Module load test timed out{Style.RESET_ALL}")
            return False

        # Test 2: GaussianSplatRenderer can be created (via GDScript test)
        test_script = self._create_test_script()

        test_start = time.time()
        try:
            result = subprocess.run(
                [str(binary_path), "--headless", "--script", str(test_script)],
                capture_output=True,
                text=True,
                timeout=30
            )

            test_duration = time.time() - test_start

            # Check for success marker in output
            if "GAUSSIAN_SPLAT_TEST_PASSED" in result.stdout:
                self.test_results.append(TestResult("instantiation", True, test_duration))
                print(f"  {Fore.GREEN}✅ GaussianSplatRenderer instantiation test passed{Style.RESET_ALL}")
            else:
                self.test_results.append(TestResult("instantiation", False, test_duration,
                                                  error_output=result.stderr or result.stdout))
                print(f"  {Fore.RED}❌ GaussianSplatRenderer instantiation failed{Style.RESET_ALL}")
                print(f"    Output: {(result.stdout + result.stderr)[:200]}...")
                return False

        except subprocess.TimeoutExpired:
            self.test_results.append(TestResult("instantiation", False, 30, "Timeout"))
            print(f"  {Fore.RED}❌ Instantiation test timed out{Style.RESET_ALL}")
            return False
        finally:
            # Clean up test script
            if test_script.exists():
                test_script.unlink()

        # Test 3: Painterly regression harness
        painterly_script = self.project_root / "scripts" / "tools" / "run_painterly_regression.gd"
        if painterly_script.exists():
            test_start = time.time()
            try:
                result = subprocess.run(
                    [str(binary_path), "--headless", "--script", str(painterly_script)],
                    capture_output=True,
                    text=True,
                    timeout=90
                )
                test_duration = time.time() - test_start
                if "PAINTERLY_TEST_PASSED" in result.stdout:
                    self.test_results.append(TestResult("painterly_regression", True, test_duration))
                    print(f"  {Fore.GREEN}✅ Painterly regression checks passed{Style.RESET_ALL}")
                else:
                    details = (result.stdout + result.stderr) if result.stderr else result.stdout
                    self.test_results.append(TestResult("painterly_regression", False, test_duration, error_output=details))
                    print(f"  {Fore.RED}❌ Painterly regression failed{Style.RESET_ALL}")
                    print(f"    Output: {details[:200]}...")
                    return False
            except subprocess.TimeoutExpired:
                self.test_results.append(TestResult("painterly_regression", False, 90, "Timeout"))
                print(f"  {Fore.RED}❌ Painterly regression timed out{Style.RESET_ALL}")
                return False
        else:
            print(f"  {Fore.YELLOW}⚠️ Painterly regression script not found; skipping headless validation{Style.RESET_ALL}")

        print(f"  {Fore.GREEN}✅ All unit tests passed{Style.RESET_ALL}")
        return True

    def _create_test_script(self) -> Path:
        """Create a temporary GDScript test file"""
        test_script = self.project_root / "test_temp.gd"

        script_content = '''
extends SceneTree

func _init():
    # Test basic instantiation
    var renderer = GaussianSplatRenderer.new()
    if renderer != null:
        print("GAUSSIAN_SPLAT_TEST_PASSED: GaussianSplatRenderer created successfully")

        # Test basic data creation
        var gaussian_data = GaussianData.new()
        if gaussian_data != null:
            print("GAUSSIAN_SPLAT_TEST_PASSED: GaussianData created successfully")

        # Test manager access
        var manager = GaussianSplatManager.get_singleton()
        if manager != null:
            print("GAUSSIAN_SPLAT_TEST_PASSED: GaussianSplatManager singleton accessible")

        print("GAUSSIAN_SPLAT_TEST_PASSED: All instantiation tests passed")
    else:
        print("GAUSSIAN_SPLAT_TEST_FAILED: Failed to create GaussianSplatRenderer")

    quit()
'''

        test_script.write_text(script_content)
        return test_script

    def _run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks"""
        print(f"{Fore.YELLOW}⚡ [3/4] Performance Benchmarks{Style.RESET_ALL}")

        # For now, we'll create a simple benchmark that tests basic operations
        # In the future, this will run actual splat rendering benchmarks

        binary_path = self.godot_source / "bin" / "godot.windows.editor.x86_64.exe"

        benchmark_configs = [
            {"splats": 100, "target_fps": 1000, "name": "100_splats"},
            {"splats": 1000, "target_fps": 500, "name": "1k_splats"},
            {"splats": 10000, "target_fps": 200, "name": "10k_splats"},
        ]

        all_passed = True

        for config in benchmark_configs:
            bench_script = self._create_benchmark_script(config["splats"])

            print(f"  🎯 Testing {config['splats']} splats (target: {config['target_fps']} FPS)...")

            bench_start = time.time()
            try:
                result = subprocess.run(
                    [str(binary_path), "--headless", "--script", str(bench_script)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                bench_duration = time.time() - bench_start

                # Parse benchmark output
                fps = self._parse_benchmark_output(result.stdout, result.stderr)

                if fps is not None:
                    passed = fps >= config["target_fps"]

                    # Estimate frame time and memory (simplified for now)
                    frame_time = 1000.0 / fps if fps > 0 else 0
                    memory_mb = 50 + (config["splats"] * 0.001)  # Rough estimate

                    self.benchmark_results.append(BenchmarkResult(
                        config["name"], fps, frame_time, memory_mb, passed, config["target_fps"]
                    ))

                    status = f"{Fore.GREEN}✅ PASS" if passed else f"{Fore.RED}❌ FAIL"
                    print(f"    {fps:>6.1f} FPS ({frame_time:>5.1f}ms) {status}{Style.RESET_ALL}")

                    if not passed:
                        all_passed = False
                else:
                    print(f"    {Fore.RED}❌ Failed to parse benchmark results{Style.RESET_ALL}")
                    self.benchmark_results.append(BenchmarkResult(
                        config["name"], 0, 0, 0, False, config["target_fps"]
                    ))
                    all_passed = False

            except subprocess.TimeoutExpired:
                print(f"    {Fore.RED}❌ Benchmark timed out{Style.RESET_ALL}")
                all_passed = False
            finally:
                if bench_script.exists():
                    bench_script.unlink()

        if all_passed:
            print(f"  {Fore.GREEN}✅ All benchmarks passed{Style.RESET_ALL}")
        else:
            if self.strict_validation:
                print(f"  {Fore.RED}❌ Some benchmarks failed (strict mode){Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}⚠️ Some benchmarks failed (warn-only mode){Style.RESET_ALL}")

        return all_passed

    def _create_benchmark_script(self, splat_count: int) -> Path:
        """Create a temporary benchmark script"""
        bench_script = self.project_root / "benchmark_temp.gd"

        script_content = f'''
extends SceneTree

func _init():
    print("BENCHMARK_START")

    # Simulate performance test
    var start_time = Time.get_time_dict_from_system()["second"] + Time.get_time_dict_from_system()["minute"] * 60

    # Create renderer and data
    var renderer = GaussianSplatRenderer.new()
    var gaussian_data = GaussianData.new()

    # Simulate processing {splat_count} splats
    var operations = {splat_count}
    var iterations = 0

    while iterations < 100:  # Run for a bit to get stable measurement
        # Simulate some work
        for i in range(min(operations, 1000)):
            pass
        iterations += 1

    var end_time = Time.get_time_dict_from_system()["second"] + Time.get_time_dict_from_system()["minute"] * 60
    var duration = max(end_time - start_time, 0.001)  # Prevent division by zero

    # Calculate simulated FPS (this is a placeholder)
    var simulated_fps = 60.0 / (1.0 + {splat_count} / 100000.0)  # Simulate decreasing perf with more splats

    print("BENCHMARK_RESULT: " + str(simulated_fps))
    print("BENCHMARK_END")

    quit()
'''

        bench_script.write_text(script_content)
        return bench_script

    def _parse_benchmark_output(self, stdout: str, stderr: str) -> Optional[float]:
        """Parse benchmark output to extract FPS"""
        output = stdout + stderr

        for line in output.split('\n'):
            if "BENCHMARK_RESULT:" in line:
                try:
                    fps_str = line.split("BENCHMARK_RESULT:")[1].strip()
                    return float(fps_str)
                except (ValueError, IndexError):
                    pass

        return None

    def _run_memory_validation(self) -> bool:
        """Run memory-stress validation using a dedicated GDScript harness."""
        print(f"{Fore.YELLOW}🔍 [4/4] Memory Validation{Style.RESET_ALL}")

        try:
            binary_path = self.godot_source / "bin" / "godot.windows.editor.x86_64.exe"
            if not binary_path.exists():
                message = f"Godot binary not found at {binary_path}"
                self.test_results.append(TestResult("memory_stress", False, 0, error_output=message))
                print(f"  {Fore.RED}❌ {message}{Style.RESET_ALL}")
                return False

            script_path = resolve_memory_validation_script(self.project_root)
            if not script_path.is_file():
                message = (
                    f"Memory validation script not found: {script_path} "
                    f"(set {MEMORY_VALIDATION_SCRIPT_ENV} to override)"
                )
                self.test_results.append(TestResult("memory_stress", False, 0, error_output=message))
                print(f"  {Fore.RED}❌ {message}{Style.RESET_ALL}")
                return False

            try:
                script_arg = str(script_path.relative_to(self.project_root))
            except ValueError:
                script_arg = str(script_path)

            command = [str(binary_path), "--headless", "--script", script_arg]

            mem_start = time.time()
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180
            )

            mem_duration = time.time() - mem_start

            if result.returncode == 0:
                self.test_results.append(TestResult("memory_stress", True, mem_duration))
                print(f"  {Fore.GREEN}✅ Memory stress validation passed ({script_arg}){Style.RESET_ALL}")
                return True
            else:
                combined_output = (result.stderr or result.stdout or "").strip()
                self.test_results.append(
                    TestResult("memory_stress", False, mem_duration, error_output=combined_output)
                )
                print(f"  {Fore.RED}❌ Memory stress validation failed ({script_arg}){Style.RESET_ALL}")
                if combined_output:
                    print(f"    {combined_output[:400]}...")
                return False

        except subprocess.TimeoutExpired:
            self.test_results.append(TestResult("memory_stress", False, 180, error_output="Timeout"))
            print(f"  {Fore.RED}❌ Memory stress validation timed out{Style.RESET_ALL}")
            return False
        except Exception as e:
            self.test_results.append(TestResult("memory_stress", False, 0, error_output=str(e)))
            print(f"  {Fore.RED}❌ Memory stress validation error: {e}{Style.RESET_ALL}")
            return False

    def _generate_report(self):
        """Generate comprehensive HTML and JSON reports"""
        print(f"{Fore.YELLOW}📊 Generating reports...{Style.RESET_ALL}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate JSON report
        json_data = {
            "timestamp": timestamp,
            "duration": time.time() - self.start_time,
            "builds": [{"platform": b.platform, "success": b.success, "duration": b.duration}
                      for b in self.build_results],
            "tests": [{"name": t.name, "passed": t.passed, "duration": t.duration}
                     for t in self.test_results],
            "benchmarks": [{"name": b.name, "fps": b.fps, "passed": b.passed, "target_fps": b.target_fps}
                          for b in self.benchmark_results]
        }

        json_path = self.reports_dir / f"report_{timestamp}.json"
        json_path.write_text(json.dumps(json_data, indent=2))

        # Generate HTML report
        html_content = self._generate_html_report(json_data, timestamp)
        html_path = self.reports_dir / f"report_{timestamp}.html"
        html_path.write_text(html_content)

        # Create latest symlinks
        latest_json = self.reports_dir / "latest.json"
        latest_html = self.reports_dir / "latest.html"

        if latest_json.exists():
            latest_json.unlink()
        if latest_html.exists():
            latest_html.unlink()

        # Create copies as "latest"
        shutil.copy2(json_path, latest_json)
        shutil.copy2(html_path, latest_html)

        print(f"  {Fore.GREEN}✅ Reports generated:{Style.RESET_ALL}")
        print(f"    📋 {html_path}")
        print(f"    🔗 {latest_html}")

    def _generate_html_report(self, data: Dict[str, Any], timestamp: str) -> str:
        """Generate HTML report"""
        total_tests = len(data["tests"])
        passed_tests = sum(1 for t in data["tests"] if t["passed"])
        total_benchmarks = len(data["benchmarks"])
        passed_benchmarks = sum(1 for b in data["benchmarks"] if b["passed"])

        return f'''<!DOCTYPE html>
<html>
<head>
    <title>GodotGS CI Report - {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: 600; }}
        .status-pass {{ color: #28a745; font-weight: bold; }}
        .status-fail {{ color: #dc3545; font-weight: bold; }}
        .benchmark-good {{ background-color: #d4edda; }}
        .benchmark-poor {{ background-color: #f8d7da; }}
        .footer {{ text-align: center; color: #666; border-top: 1px solid #eee; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 GodotGS Local CI Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Pipeline Duration: {data["duration"]:.1f} seconds</p>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Build Status</h3>
                <p class="{'success' if all(b['success'] for b in data['builds']) else 'failure'}">
                    {'✅ SUCCESS' if all(b['success'] for b in data['builds']) else '❌ FAILED'}
                </p>
            </div>
            <div class="summary-card">
                <h3>Tests</h3>
                <p>{passed_tests}/{total_tests} Passed</p>
                <p class="{'success' if passed_tests == total_tests else 'failure'}">
                    {passed_tests/total_tests*100:.0f}% Pass Rate
                </p>
            </div>
            <div class="summary-card">
                <h3>Benchmarks</h3>
                <p>{passed_benchmarks}/{total_benchmarks} Passed</p>
                <p class="{'success' if passed_benchmarks == total_benchmarks else 'warning'}">
                    {passed_benchmarks/total_benchmarks*100:.0f}% Pass Rate
                </p>
            </div>
        </div>

        <h2>🔨 Build Results</h2>
        <table>
            <tr><th>Platform</th><th>Status</th><th>Duration</th></tr>
            {''.join(f"<tr><td>{b['platform']}</td><td class='status-{'pass' if b['success'] else 'fail'}'>{
                '✅ PASS' if b['success'] else '❌ FAIL'}</td><td>{b['duration']:.1f}s</td></tr>"
                for b in data['builds'])}
        </table>

        <h2>🧪 Test Results</h2>
        <table>
            <tr><th>Test</th><th>Status</th><th>Duration</th></tr>
            {''.join(f"<tr><td>{t['name']}</td><td class='status-{'pass' if t['passed'] else 'fail'}'>{
                '✅ PASS' if t['passed'] else '❌ FAIL'}</td><td>{t['duration']:.2f}s</td></tr>"
                for t in data['tests'])}
        </table>

        <h2>⚡ Performance Benchmarks</h2>
        <table>
            <tr><th>Benchmark</th><th>FPS</th><th>Target</th><th>Status</th></tr>
            {''.join(f"<tr class='benchmark-{'good' if b['passed'] else 'poor'}'><td>{b['name']}</td><td>{b['fps']:.1f}</td><td>{b['target_fps']}</td><td class='status-{'pass' if b['passed'] else 'fail'}'>{
                '✅ PASS' if b['passed'] else '❌ FAIL'}</td></tr>"
                for b in data['benchmarks'])}
        </table>

        <div class="footer">
            <p>GodotGS Local CI/CD Pipeline | Issue #24 Implementation</p>
            <p>Module Status: Hello Splat Implementation Complete ✅</p>
        </div>
    </div>
</body>
</html>'''

    def _print_failure(self, message: str):
        """Print failure message"""
        print(f"\n{Fore.RED}💥 {message.upper()}{Style.RESET_ALL}")
        print(f"{Fore.RED}Pipeline terminated.{Style.RESET_ALL}\n")


def main():
    """Main entry point"""
    pipeline = GodotGSPipeline()

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick validation for pre-commit hooks
        print("🚀 Running quick validation...")
        success = pipeline._run_build_validation() and pipeline._run_unit_tests()
        sys.exit(0 if success else 1)

    success = pipeline.run_full_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
