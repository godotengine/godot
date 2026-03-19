#!/usr/bin/env python3
"""
Local validation script for baseline QA automation
Tests that the CI scripts work properly in the local environment
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def check_godot_binary():
    """Check if Godot binary is available"""
    godot_binary = os.environ.get('GODOT_BINARY', 'godot')

    try:
        result = subprocess.run([godot_binary, '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Godot binary available: {result.stdout.strip()}")
            return True, godot_binary
        else:
            print(f"❌ Godot binary not working: {result.stderr}")
            return False, None
    except Exception as e:
        print(f"❌ Godot binary not found: {e}")
        return False, None

def check_test_files():
    """Check that all required test files exist"""
    test_files = [
        "tests/ci/test_ply_loader_ci.gd",
        "tests/ci/test_ply_pipeline_ci.gd",
        "tests/ci/test_gpu_sorting_ci.gd",
        "tests/ci/run_baseline_qa.py"
    ]

    root_dir = Path(__file__).parent.parent.parent
    missing_files = []

    for test_file in test_files:
        file_path = root_dir / test_file
        if file_path.exists():
            print(f"✅ Found: {test_file}")
        else:
            print(f"❌ Missing: {test_file}")
            missing_files.append(test_file)

    return len(missing_files) == 0, missing_files

def test_individual_scripts(godot_binary):
    """Test individual CI scripts"""
    root_dir = Path(__file__).parent.parent.parent
    test_scripts = [
        ("tests/ci/test_ply_loader_ci.gd", "PLY Loader CI Test"),
        ("tests/ci/test_ply_pipeline_ci.gd", "PLY Pipeline CI Test"),
        ("tests/ci/test_gpu_sorting_ci.gd", "GPU Sorting CI Test")
    ]

    results = []

    for script_path, test_name in test_scripts:
        print(f"\n🧪 Testing {test_name}...")

        try:
            cmd = [godot_binary, "--headless", "--script", script_path]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=60, cwd=root_dir)

            success = result.returncode == 0
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} (exit code: {result.returncode})")

            if not success and result.stderr:
                print(f"   Error: {result.stderr[:200]}...")

            results.append({
                "name": test_name,
                "script": script_path,
                "success": success,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })

        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT (>60s)")
            results.append({
                "name": test_name,
                "script": script_path,
                "success": False,
                "exit_code": -1,
                "error": "Timeout"
            })
        except Exception as e:
            print(f"   💥 EXCEPTION: {e}")
            results.append({
                "name": test_name,
                "script": script_path,
                "success": False,
                "exit_code": -2,
                "error": str(e)
            })

    return results

def test_baseline_qa_runner(godot_binary):
    """Test the main baseline QA runner"""
    print(f"\n🎯 Testing Baseline QA Runner...")

    root_dir = Path(__file__).parent.parent.parent

    try:
        env = os.environ.copy()
        env['GODOT_BINARY'] = godot_binary

        cmd = ["python3", "tests/ci/run_baseline_qa.py", "--godot", godot_binary]
        result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=120, cwd=root_dir, env=env)

        success = result.returncode == 0
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} (exit code: {result.returncode})")

        if not success:
            print(f"   Error output: {result.stderr[:300]}...")

        # Check for results file
        results_file = root_dir / "baseline_qa_results.json"
        if results_file.exists():
            print(f"   ✅ Results file created: {results_file}")
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    print(f"   📊 Test summary: {data.get('passed_tests', 0)}/{data.get('total_tests', 0)} passed")
            except Exception as e:
                print(f"   ⚠️ Could not parse results file: {e}")
        else:
            print(f"   ⚠️ No results file created")

        return {
            "success": success,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        print(f"   💥 EXCEPTION: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def check_ci_workflow():
    """Check that CI workflow file exists and is valid"""
    workflow_file = Path(__file__).parent.parent.parent / ".github/workflows/baseline_qa.yml"

    if workflow_file.exists():
        print(f"✅ CI workflow file exists: {workflow_file}")

        # Basic YAML validation
        try:
            import yaml
            with open(workflow_file) as f:
                yaml.safe_load(f)
            print("✅ CI workflow YAML is valid")
            return True
        except ImportError:
            print("⚠️ PyYAML not available, skipping YAML validation")
            return True
        except Exception as e:
            print(f"❌ CI workflow YAML is invalid: {e}")
            return False
    else:
        print(f"❌ CI workflow file missing: {workflow_file}")
        return False

def main():
    """Main validation workflow"""
    print("=== Baseline QA Automation Validation ===")

    # Check prerequisites
    print("\n1. Checking prerequisites...")
    godot_available, godot_binary = check_godot_binary()
    files_available, missing_files = check_test_files()
    ci_valid = check_ci_workflow()

    if not godot_available:
        print("\n❌ VALIDATION FAILED: Godot binary not available")
        print("💡 Install Godot or set GODOT_BINARY environment variable")
        sys.exit(1)

    if not files_available:
        print(f"\n❌ VALIDATION FAILED: Missing test files: {missing_files}")
        sys.exit(1)

    if not ci_valid:
        print("\n❌ VALIDATION FAILED: CI workflow issues")
        sys.exit(1)

    print("\n✅ All prerequisites satisfied")

    # Test individual scripts
    print("\n2. Testing individual CI scripts...")
    script_results = test_individual_scripts(godot_binary)

    script_failures = [r for r in script_results if not r["success"]]
    if script_failures:
        print(f"\n⚠️ {len(script_failures)} individual script(s) failed")
        for failure in script_failures:
            print(f"   - {failure['name']}: {failure.get('error', 'Failed')}")
    else:
        print("\n✅ All individual scripts passed")

    # Test main runner
    print("\n3. Testing baseline QA runner...")
    runner_result = test_baseline_qa_runner(godot_binary)

    if runner_result["success"]:
        print("\n✅ Baseline QA runner passed")
    else:
        print(f"\n⚠️ Baseline QA runner failed: {runner_result.get('error', 'Unknown error')}")

    # Summary
    print("\n=== Validation Summary ===")
    total_tests = len(script_results) + 1  # +1 for runner
    passed_tests = sum(1 for r in script_results if r["success"]) + (1 if runner_result["success"] else 0)

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Baseline QA automation is ready for CI")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} validation test(s) failed")
        print("💡 Check individual test outputs for details")
        print("💡 Some failures may be expected in environments without full GPU support")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
