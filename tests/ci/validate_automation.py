#!/usr/bin/env python3
"""Local validation script for Gaussian Splatting CI automation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MarkerContract:
    issue_id: str
    description: str
    relative_path: str
    required_markers: tuple[str, ...]


ISSUE_CONTRACTS: tuple[MarkerContract, ...] = (
    MarkerContract(
        issue_id="ISSUE-041",
        description="Painterly shader performance toggle route is wired through CI evidence collection.",
        relative_path="tests/ci/collect_production_evidence.ps1",
        required_markers=(
            "scripts/tools/run_painterly_regression.gd",
            "tests/examples/godot/test_toggle_painterly.gd",
            "PAINTERLY_TEST_PASSED",
        ),
    ),
    MarkerContract(
        issue_id="ISSUE-042",
        description="Sort algorithm selection is driven through unified policy probes/traits.",
        relative_path="modules/gaussian_splatting/renderer/gpu_sorter.h",
        required_markers=(
            "struct SorterCapabilities",
            "struct PolicyProbe",
            "static PolicyDecision evaluate_auto_policy(",
        ),
    ),
    MarkerContract(
        issue_id="ISSUE-042",
        description="Factory auto-selection path continues to evaluate PolicyProbe inputs centrally.",
        relative_path="modules/gaussian_splatting/renderer/gpu_sorter.cpp",
        required_markers=(
            "PolicyDecision auto_decision = evaluate_auto_policy(",
            "_to_policy_probe(",
            "get_policy_probe(ALGORITHM_RADIX)",
        ),
    ),
    MarkerContract(
        issue_id="ISSUE-043",
        description="Theoretical complexity units/scaling assumptions stay documented in sorter contract comments.",
        relative_path="modules/gaussian_splatting/renderer/gpu_sorter.h",
        required_markers=(
            "The returned float is a unitless relative indicator (lower = fewer passes):",
            "These values do NOT reflect real-world throughput.",
            "virtual float get_theoretical_complexity() const = 0;",
        ),
    ),
    MarkerContract(
        issue_id="ISSUE-044",
        description="Clip blending guards prevent circular/self chains and enforce depth cap.",
        relative_path="modules/gaussian_splatting/animation/animation_state_machine.cpp",
        required_markers=(
            "if (p_clip_index == current_clip_index) {",
            "MAX_BLEND_CHAIN_DEPTH = 8",
            "if (blend_targets.size() >= MAX_BLEND_CHAIN_DEPTH) {",
        ),
    ),
    MarkerContract(
        issue_id="ISSUE-045",
        description="Incremental saver layout version is part of the serialized contract and validated on load.",
        relative_path="modules/gaussian_splatting/persistence/incremental_saver.h",
        required_markers=(
            "INCREMENTAL_SAVER_LAYOUT_VERSION",
            "Layout version tracks the on-disk struct layout of SplatChange / ChangeEntry.",
        ),
    ),
    MarkerContract(
        issue_id="ISSUE-045",
        description="Incremental saver read/write paths persist and enforce layout version.",
        relative_path="modules/gaussian_splatting/persistence/incremental_saver.cpp",
        required_markers=(
            "file->store_16(INCREMENTAL_SAVER_LAYOUT_VERSION);",
            "uint16_t layout_version = file->get_16();",
            "layout_version != INCREMENTAL_SAVER_LAYOUT_VERSION",
        ),
    ),
)


def check_godot_binary() -> tuple[bool, str | None]:
    """Check if Godot binary is available."""
    godot_binary = os.environ.get("GODOT_BINARY", "godot")

    try:
        result = subprocess.run(
            [godot_binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            print(f"✅ Godot binary available: {result.stdout.strip()}")
            return True, godot_binary
        print(f"❌ Godot binary not working: {result.stderr}")
        return False, None
    except Exception as exc:
        print(f"❌ Godot binary not found: {exc}")
        return False, None


def check_test_files() -> tuple[bool, list[str]]:
    """Check that all required test files exist."""
    test_files = [
        "tests/ci/test_ply_loader_ci.gd",
        "tests/ci/test_ply_pipeline_ci.gd",
        "tests/ci/test_gpu_sorting_ci.gd",
        "tests/ci/run_baseline_qa.py",
    ]

    missing_files: list[str] = []
    for test_file in test_files:
        file_path = ROOT_DIR / test_file
        if file_path.exists():
            print(f"✅ Found: {test_file}")
        else:
            print(f"❌ Missing: {test_file}")
            missing_files.append(test_file)

    return len(missing_files) == 0, missing_files


def _load_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"❌ Failed reading {path}: {exc}")
        return None


def _check_markers(text: str, markers: Iterable[str]) -> list[str]:
    missing = []
    for marker in markers:
        if marker not in text:
            missing.append(marker)
    return missing


def check_issue_contract_markers() -> tuple[bool, list[str]]:
    """Check static marker contracts for ISSUE-041..045."""
    failures: list[str] = []

    print("\n🔒 Validating issue marker contracts (ISSUE-041..045)...")
    for contract in ISSUE_CONTRACTS:
        file_path = ROOT_DIR / contract.relative_path
        text = _load_text(file_path)
        if text is None:
            failures.append(
                f"{contract.issue_id}: unable to read {contract.relative_path}"
            )
            continue

        missing_markers = _check_markers(text, contract.required_markers)
        if missing_markers:
            failures.append(
                f"{contract.issue_id}: {contract.relative_path} missing markers: "
                + ", ".join(repr(marker) for marker in missing_markers)
            )
            print(f"❌ {contract.issue_id} marker check failed: {contract.description}")
        else:
            print(f"✅ {contract.issue_id}: {contract.description}")

    return len(failures) == 0, failures


def test_individual_scripts(godot_binary: str) -> list[dict]:
    """Test individual CI scripts."""
    test_scripts = [
        ("tests/ci/test_ply_loader_ci.gd", "PLY Loader CI Test"),
        ("tests/ci/test_ply_pipeline_ci.gd", "PLY Pipeline CI Test"),
        ("tests/ci/test_gpu_sorting_ci.gd", "GPU Sorting CI Test"),
    ]

    results = []
    for script_path, test_name in test_scripts:
        print(f"\n🧪 Testing {test_name}...")
        try:
            cmd = [godot_binary, "--headless", "--script", script_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=ROOT_DIR,
                check=False,
            )

            success = result.returncode == 0
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} (exit code: {result.returncode})")
            if not success and result.stderr:
                print(f"   Error: {result.stderr[:200]}...")

            results.append(
                {
                    "name": test_name,
                    "script": script_path,
                    "success": success,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        except subprocess.TimeoutExpired:
            print("   ⏰ TIMEOUT (>60s)")
            results.append(
                {
                    "name": test_name,
                    "script": script_path,
                    "success": False,
                    "exit_code": -1,
                    "error": "Timeout",
                }
            )
        except Exception as exc:
            print(f"   💥 EXCEPTION: {exc}")
            results.append(
                {
                    "name": test_name,
                    "script": script_path,
                    "success": False,
                    "exit_code": -2,
                    "error": str(exc),
                }
            )

    return results


def test_baseline_qa_runner(godot_binary: str) -> dict:
    """Test the main baseline QA runner."""
    print("\n🎯 Testing Baseline QA Runner...")
    try:
        env = os.environ.copy()
        env["GODOT_BINARY"] = godot_binary
        cmd = [sys.executable, "tests/ci/run_baseline_qa.py", "--godot", godot_binary]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=ROOT_DIR,
            env=env,
            check=False,
        )

        success = result.returncode == 0
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} (exit code: {result.returncode})")
        if not success:
            print(f"   Error output: {result.stderr[:300]}...")

        results_file = ROOT_DIR / "baseline_qa_results.json"
        if results_file.exists():
            print(f"   ✅ Results file created: {results_file}")
            try:
                data = json.loads(results_file.read_text(encoding="utf-8"))
                print(
                    "   📊 Test summary: "
                    f"{data.get('passed_tests', 0)}/{data.get('total_tests', 0)} passed"
                )
            except Exception as exc:
                print(f"   ⚠️ Could not parse results file: {exc}")
        else:
            print("   ⚠️ No results file created")

        return {
            "success": success,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        print(f"   💥 EXCEPTION: {exc}")
        return {"success": False, "error": str(exc)}


def check_ci_workflow() -> bool:
    """Check that required CI workflow files exist and are valid YAML when parser is available."""
    workflow_files = [
        ".github/workflows/baseline_qa.yml",
        ".github/workflows/gaussian_production_gates.yml",
        ".github/workflows/gaussian_shader_validation.yml",
    ]

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    success = True
    for relative in workflow_files:
        workflow_file = ROOT_DIR / relative
        if not workflow_file.exists():
            print(f"❌ Missing CI workflow file: {relative}")
            success = False
            continue

        print(f"✅ CI workflow file exists: {relative}")
        if yaml is None:
            print("⚠️ PyYAML not available, skipping YAML parse validation")
            continue

        try:
            yaml.safe_load(workflow_file.read_text(encoding="utf-8"))
            print(f"✅ YAML valid: {relative}")
        except Exception as exc:
            print(f"❌ CI workflow YAML is invalid ({relative}): {exc}")
            success = False

    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gaussian Splatting CI automation scripts.")
    parser.add_argument(
        "--contracts-only",
        action="store_true",
        help="Run static contract/file/workflow checks only (skip Godot runtime script execution).",
    )
    return parser.parse_args()


def main() -> bool:
    """Main validation workflow."""
    args = parse_args()
    print("=== Baseline QA Automation Validation ===")

    print("\n1. Checking prerequisites and contracts...")
    files_available, missing_files = check_test_files()
    ci_valid = check_ci_workflow()
    contract_valid, contract_failures = check_issue_contract_markers()

    if not files_available:
        print(f"\n❌ VALIDATION FAILED: Missing test files: {missing_files}")
        return False
    if not ci_valid:
        print("\n❌ VALIDATION FAILED: CI workflow issues")
        return False
    if not contract_valid:
        print("\n❌ VALIDATION FAILED: Issue marker contract checks failed")
        for failure in contract_failures:
            print(f"   - {failure}")
        return False

    print("\n✅ Static prerequisites and contracts satisfied")

    if args.contracts_only:
        print("\n=== Validation Summary ===")
        print("Mode: contracts-only")
        print("Result: PASS")
        return True

    print("\n2. Checking Godot binary...")
    godot_available, godot_binary = check_godot_binary()
    if not godot_available or godot_binary is None:
        print("\n❌ VALIDATION FAILED: Godot binary not available")
        print("💡 Install Godot or set GODOT_BINARY environment variable")
        return False

    print("\n3. Testing individual CI scripts...")
    script_results = test_individual_scripts(godot_binary)
    script_failures = [result for result in script_results if not result["success"]]
    if script_failures:
        print(f"\n⚠️ {len(script_failures)} individual script(s) failed")
        for failure in script_failures:
            print(f"   - {failure['name']}: {failure.get('error', 'Failed')}")
    else:
        print("\n✅ All individual scripts passed")

    print("\n4. Testing baseline QA runner...")
    runner_result = test_baseline_qa_runner(godot_binary)
    if runner_result["success"]:
        print("\n✅ Baseline QA runner passed")
    else:
        print(f"\n⚠️ Baseline QA runner failed: {runner_result.get('error', 'Unknown error')}")

    print("\n=== Validation Summary ===")
    total_tests = len(script_results) + 1
    passed_tests = sum(1 for result in script_results if result["success"]) + (
        1 if runner_result["success"] else 0
    )
    failed_tests = total_tests - passed_tests

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")

    if failed_tests == 0:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Baseline QA automation is ready for CI")
        return True

    print(f"\n⚠️ {failed_tests} validation test(s) failed")
    print("💡 Check individual test outputs for details")
    print("💡 Some failures may be expected in environments without full GPU support")
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
