#!/usr/bin/env python3
"""
Headless build and integration tests for Gaussian Splatting module.
Runs without GUI to validate module initialization, GPU resources, and memory.
"""

import sys
import os
import subprocess
import json
import time
import tracemalloc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def find_repository_root() -> Path:
    """Find the GodotGS repository root by locating the Godot fork root."""
    current = Path(__file__).parent
    while current.parent != current:  # Stop at filesystem root
        if (current / "SConstruct").exists() and (current / "modules" / "gaussian_splatting").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find GodotGS repository root (missing SConstruct/modules layout)")

def find_godot_binary(repo_root: Path) -> Path:
    """Find Godot executable with fallback options"""
    possible_paths = [
        repo_root / "bin" / "godot.windows.editor.dev.x86_64.exe",
        repo_root / "bin" / "godot.windows.editor.x86_64.exe",
        repo_root / "bin" / "godot.linuxbsd.editor.dev.x86_64",
        repo_root / "bin" / "godot.linuxbsd.editor.x86_64",
        repo_root / "bin" / "godot.macos.editor.dev.universal",
        repo_root / "bin" / "godot.macos.editor.universal",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Try environment variable
    if os.environ.get("GODOT_BINARY"):
        env_path = Path(os.environ["GODOT_BINARY"])
        if env_path.exists():
            return env_path

    raise FileNotFoundError(f"Godot executable not found. Tried: {possible_paths}")

# Auto-detect paths
try:
    REPO_ROOT = find_repository_root()
    GODOT_PATH = REPO_ROOT
    MODULE_PATH = REPO_ROOT / "modules" / "gaussian_splatting"
    GODOT_BINARY = find_godot_binary(REPO_ROOT)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Set GODOT_BINARY environment variable or ensure repository structure is correct")
    sys.exit(1)

class HeadlessBuildTester:
    """Run comprehensive headless tests for the Gaussian Splatting module."""

    def __init__(self):
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": [],
            "summary": {}
        }
        self.process = None

    def check_prerequisites(self) -> bool:
        """Verify build environment and module presence."""
        print("🔍 Checking prerequisites...")

        checks = {
            "godot_binary": GODOT_BINARY.exists(),
            "module_exists": MODULE_PATH.exists(),
            "test_files": (MODULE_PATH / "tests").exists()
        }

        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")

        return all(checks.values())

    def test_module_registration(self) -> Dict:
        """Test that the module registers correctly with Godot."""
        print("\n🧪 Testing module registration...")

        result = {
            "test": "module_registration",
            "passed": False,
            "details": {}
        }

        try:
            # Run Godot with --list-modules
            cmd = [str(GODOT_BINARY), "--headless", "--quit", "--verbose", "--list-modules"]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            # Check for gaussian_splatting in module list
            if "gaussian_splatting" in process.stdout:
                result["passed"] = True
                result["details"]["module_found"] = True
                print("  ✅ Module registered successfully")
            else:
                result["details"]["module_found"] = False
                result["details"]["stdout"] = process.stdout
                print("  ❌ Module not found in registration")

        except subprocess.TimeoutExpired:
            result["details"]["error"] = "Timeout during module registration"
            print("  ❌ Timeout expired")
        except Exception as e:
            result["details"]["error"] = str(e)
            print(f"  ❌ Error: {e}")

        return result

    def test_gpu_initialization(self) -> Dict:
        """Test GPU resource initialization without rendering."""
        print("\n🧪 Testing GPU initialization...")

        result = {
            "test": "gpu_initialization",
            "passed": False,
            "details": {}
        }

        # Create test script for GPU validation
        test_script = """
extends Node

func _ready():
    print("Starting GPU initialization test...")

    var rs = RenderingServer
    if rs == null:
        print("ERROR: RenderingServer not available")
        get_tree().quit(1)
        return

    var rd = rs.create_local_rendering_device()
    if rd == null:
        print("ERROR: RenderingDevice not available")
        get_tree().quit(1)
        return

    print("✓ RenderingDevice created successfully")

    # Test Gaussian module components
    var manager = load("res://addons/gaussian_splatting/GaussianSplatManager.gd")
    if manager:
        print("✓ GaussianSplatManager loaded")

    var data = GaussianData.new()
    if data:
        print("✓ GaussianData instantiated")
        data.resize(1000)
        print("✓ Resized to 1000 splats")

    var stream = GaussianMemoryStream.new()
    if stream:
        print("✓ GaussianMemoryStream instantiated")
        if stream.initialize(rd, 10000) == OK:
            print("✓ Memory stream initialized")

    print("GPU initialization test completed successfully")
    get_tree().quit(0)
"""

        # Write test script
        test_project_path = MODULE_PATH / "tests" / "headless_test_project"
        test_project_path.mkdir(exist_ok=True)

        script_path = test_project_path / "test_gpu.gd"
        script_path.write_text(test_script)

        project_file = test_project_path / "project.godot"
        project_file.write_text("""
[application]
config/name="Headless GPU Test"
run/main_scene="res://test_gpu.tscn"

[rendering]
driver/threads/thread_model=2
rendering_device/staging_buffer/block_size_kb=256
rendering_device/staging_buffer/max_size_mb=128
""")

        # Create minimal scene
        scene_file = test_project_path / "test_gpu.tscn"
        scene_file.write_text("""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://test_gpu.gd" id="1"]

[node name="Test" type="Node"]
script = ExtResource("1")
""")

        try:
            cmd = [
                str(GODOT_BINARY),
                "--headless",
                "--path", str(test_project_path),
                "--quit-after", "100",
                "--verbose"
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if process.returncode == 0:
                result["passed"] = True
                result["details"]["initialization"] = "successful"
                print("  ✅ GPU initialization successful")
            else:
                result["details"]["return_code"] = process.returncode
                result["details"]["stderr"] = process.stderr[-500:]  # Last 500 chars
                print(f"  ❌ GPU initialization failed (code: {process.returncode})")

            # Parse output for specific checks
            checks = {
                "rendering_device": "RenderingDevice created" in process.stdout,
                "gaussian_data": "GaussianData instantiated" in process.stdout,
                "memory_stream": "Memory stream initialized" in process.stdout
            }

            result["details"]["component_checks"] = checks
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"    {status} {check}")

        except subprocess.TimeoutExpired:
            result["details"]["error"] = "Timeout during GPU initialization"
            print("  ❌ Timeout expired")
        except Exception as e:
            result["details"]["error"] = str(e)
            print(f"  ❌ Error: {e}")

        return result

    def test_memory_leak_detection(self) -> Dict:
        """Run memory leak detection tests."""
        print("\n🧪 Testing memory leak detection...")

        result = {
            "test": "memory_leak_detection",
            "passed": False,
            "details": {}
        }

        # Monitor memory usage during test execution
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        test_script = """
extends Node

var frame_count = 0
var data_objects = []
var stream_objects = []

func _ready():
    print("Starting memory leak test...")
    set_process(true)

func _process(_delta):
    frame_count += 1

    # Create and destroy objects repeatedly
    if frame_count % 10 == 0:
        # Create new objects
        var data = GaussianData.new()
        data.resize(10000)
        data_objects.append(data)

        var stream = GaussianMemoryStream.new()
        stream_objects.append(stream)

        # Clean up old objects
        if data_objects.size() > 10:
            data_objects.pop_front()
        if stream_objects.size() > 10:
            stream_objects.pop_front()

    # Run for 100 frames
    if frame_count >= 100:
        print("Memory leak test completed")
        data_objects.clear()
        stream_objects.clear()
        get_tree().quit(0)
"""

        test_project_path = MODULE_PATH / "tests" / "headless_test_project"
        script_path = test_project_path / "test_memory.gd"
        script_path.write_text(test_script)

        scene_file = test_project_path / "test_memory.tscn"
        scene_file.write_text("""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://test_memory.gd" id="1"]

[node name="Test" type="Node"]
script = ExtResource("1")
""")

        try:
            cmd = [
                str(GODOT_BINARY),
                "--headless",
                "--path", str(test_project_path),
                "--quit-after", "200",
                "--verbose"
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Monitor memory usage
            memory_samples = []
            start_time = time.time()

            while process.poll() is None and time.time() - start_time < 30:
                try:
                    proc_info = psutil.Process(process.pid)
                    memory_mb = proc_info.memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_mb)
                    time.sleep(0.1)
                except:
                    pass

            stdout, stderr = process.communicate(timeout=5)

            if memory_samples:
                peak_memory = max(memory_samples)
                avg_memory = sum(memory_samples) / len(memory_samples)
                memory_growth = memory_samples[-1] - memory_samples[0]

                result["details"]["memory_stats"] = {
                    "initial_mb": memory_samples[0],
                    "final_mb": memory_samples[-1],
                    "peak_mb": peak_memory,
                    "avg_mb": avg_memory,
                    "growth_mb": memory_growth
                }

                # Check for excessive memory growth (>100MB is suspicious)
                if memory_growth < 100:
                    result["passed"] = True
                    print(f"  ✅ Memory stable (growth: {memory_growth:.1f} MB)")
                else:
                    print(f"  ⚠️ Memory growth detected: {memory_growth:.1f} MB")

                print(f"    Peak: {peak_memory:.1f} MB, Avg: {avg_memory:.1f} MB")

        except subprocess.TimeoutExpired:
            process.kill()
            result["details"]["error"] = "Timeout during memory test"
            print("  ❌ Timeout expired")
        except Exception as e:
            result["details"]["error"] = str(e)
            print(f"  ❌ Error: {e}")
        finally:
            tracemalloc.stop()

        return result

    def test_component_lifecycle(self) -> Dict:
        """Test proper initialization and cleanup of components."""
        print("\n🧪 Testing component lifecycle...")

        result = {
            "test": "component_lifecycle",
            "passed": False,
            "details": {}
        }

        test_script = """
extends Node

func _ready():
    print("Starting component lifecycle test...")

    var errors = []
    var rd = RenderingServer.create_local_rendering_device()

    # Test 1: GaussianMemoryStream lifecycle
    print("Testing GaussianMemoryStream...")
    var stream = GaussianMemoryStream.new()
    if stream.initialize(rd, 10000) != OK:
        errors.append("Failed to initialize memory stream")
    stream.begin_frame(0)
    stream.end_frame()
    stream.cleanup()  # Should not crash
    print("✓ Memory stream lifecycle complete")

    # Test 2: RadixSort lifecycle
    print("Testing RadixSort...")
    var sorter = RadixSort.new()
    if sorter.initialize(rd, 1024) != OK:
        errors.append("Failed to initialize sorter")
    sorter.shutdown()  # Should not crash
    print("✓ Sorter lifecycle complete")

    # Test 3: AsyncComputePipeline lifecycle
    print("Testing AsyncComputePipeline...")
    var async_pipeline = AsyncComputePipeline.new()
    var async_available = async_pipeline.initialize(rd) == OK
    if async_available:
        async_pipeline.shutdown()
        print("✓ Async pipeline lifecycle complete")
    else:
        print("✓ Async pipeline not available (expected on some systems)")

    # Test 4: Reference counting
    print("Testing reference counting...")
    var data = GaussianData.new()
    var initial_ref = data.get_reference_count()
    var data2 = data
    if data.get_reference_count() != initial_ref + 1:
        errors.append("Reference counting failed")
    data2 = null
    print("✓ Reference counting works")

    if errors.is_empty():
        print("Component lifecycle test passed")
        get_tree().quit(0)
    else:
        for error in errors:
            print("ERROR: " + error)
        get_tree().quit(1)
"""

        test_project_path = MODULE_PATH / "tests" / "headless_test_project"
        script_path = test_project_path / "test_lifecycle.gd"
        script_path.write_text(test_script)

        scene_file = test_project_path / "test_lifecycle.tscn"
        scene_file.write_text("""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://test_lifecycle.gd" id="1"]

[node name="Test" type="Node"]
script = ExtResource("1")
""")

        try:
            cmd = [
                str(GODOT_BINARY),
                "--headless",
                "--path", str(test_project_path),
                "--quit-after", "50",
                "--verbose"
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            if process.returncode == 0:
                result["passed"] = True
                result["details"]["status"] = "all components lifecycle correct"
                print("  ✅ Component lifecycle test passed")
            else:
                result["details"]["return_code"] = process.returncode
                result["details"]["errors"] = [
                    line for line in process.stdout.split('\n')
                    if 'ERROR' in line
                ]
                print(f"  ❌ Component lifecycle test failed")

            # Check specific components
            components = [
                "Memory stream lifecycle",
                "Sorter lifecycle",
                "Reference counting"
            ]

            for component in components:
                if component in process.stdout:
                    print(f"    ✅ {component}")
                else:
                    print(f"    ❌ {component}")

        except subprocess.TimeoutExpired:
            result["details"]["error"] = "Timeout during lifecycle test"
            print("  ❌ Timeout expired")
        except Exception as e:
            result["details"]["error"] = str(e)
            print(f"  ❌ Error: {e}")

        return result

    def run_all_tests(self):
        """Execute all headless tests."""
        print("=" * 60)
        print("🚀 Running Headless Build Tests for Gaussian Splatting")
        print("=" * 60)

        if not self.check_prerequisites():
            print("\n❌ Prerequisites check failed. Exiting.")
            return False

        # Run test suite
        tests = [
            self.test_module_registration,
            self.test_gpu_initialization,
            self.test_memory_leak_detection,
            self.test_component_lifecycle
        ]

        for test_func in tests:
            result = test_func()
            self.results["tests"].append(result)

        # Generate summary
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"] if t["passed"])

        self.results["summary"] = {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }

        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {total_tests - passed_tests} ❌")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1f}%")

        # Save results to JSON
        results_file = MODULE_PATH / "tests" / f"headless_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")

        return passed_tests == total_tests

def main():
    """Main entry point for headless testing."""
    tester = HeadlessBuildTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
