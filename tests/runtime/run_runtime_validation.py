#!/usr/bin/env python3
"""Unified runtime validation runner.

Builds and executes lightweight C++ harnesses alongside GDScript runtime checks.

Examples:
  - Default headless run:
      python3 tests/runtime/run_runtime_validation.py
  - Canonical non-headless Windows Vulkan gate:
      python3 tests/runtime/run_runtime_validation.py --gd-mode windows-vulkan
  - Profile-based stress lane:
      python3 tests/runtime/run_runtime_validation.py --profile stress-only
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "tests" / "runtime"
BUILD_DIR = RUNTIME_DIR / "build"
SYNTHETIC_ASSET_PREP_SCRIPT = RUNTIME_DIR / "prepare_synthetic_assets.py"

SKIP_MARKER = "[RUNTIME_SKIP]"
FAIL_MARKER = "[RUNTIME_FAIL]"
METRICS_MARKER = "[RUNTIME_METRICS]"
DEFAULT_SCENARIO_CONFIG = RUNTIME_DIR / "runtime_scenarios.json"


@dataclass
class TestResult:
    name: str
    command: List[str]
    duration: float
    exit_code: int
    stdout: str
    stderr: str
    status: str = "failed"
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True)
class GodotRunConfig:
    binary: str
    mode: str
    timeout: int
    extra_args: tuple[str, ...]
    fail_on_skip: bool
    allow_skip_tests: tuple[str, ...]
    project_path: Optional[Path]


@dataclass(frozen=True)
class CppBuildConfig:
    link_mode: str
    include_dirs: tuple[Path, ...] = ()
    compile_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()


CPP_TESTS: Dict[str, Path] = {
    "Runtime Modifications": RUNTIME_DIR / "test_runtime_modifications.cpp",
    "Animation Persistence": RUNTIME_DIR / "test_animation_persistence.cpp",
}

GDS_TESTS: Dict[str, Path] = {
    "Interactive State": RUNTIME_DIR / "test_interactive_state.gd",
    "GPU Streaming Stress": RUNTIME_DIR / "test_gpu_streaming_stress.gd",
    "Engine Capability Sanity": RUNTIME_DIR / "test_engine_capabilities.gd",
    "World Streaming Gate": RUNTIME_DIR / "test_world_streaming_gate.gd",
    "Streaming Residency API": RUNTIME_DIR / "test_streaming_residency_api.gd",
    "Data Flow Recent Window": RUNTIME_DIR / "test_data_flow_recent_window.gd",
    "Pipeline Trace Freshness": RUNTIME_DIR / "test_pipeline_trace_freshness.gd",
    "Monitor Lifecycle Hardening": RUNTIME_DIR / "test_monitor_lifecycle_hardening.gd",
}


def ensure_build_dir() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def ensure_synthetic_assets() -> None:
    if not SYNTHETIC_ASSET_PREP_SCRIPT.is_file():
        raise RuntimeError(
            f"Missing synthetic asset prep script: {SYNTHETIC_ASSET_PREP_SCRIPT.relative_to(ROOT)}"
        )

    command = [sys.executable, str(SYNTHETIC_ASSET_PREP_SCRIPT), "--quiet"]
    print(f"[runtime] Preparing synthetic assets: {_format_command(command)}")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    except (OSError, PermissionError) as exc:
        raise RuntimeError(f"Synthetic asset prep failed to launch: {type(exc).__name__}: {exc}") from exc

    if completed.returncode != 0:
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        detail = _first_non_empty_line(output) or f"exit code {completed.returncode}"
        raise RuntimeError(f"Synthetic asset prep failed: {detail}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gaussian runtime validation harnesses.")
    parser.add_argument(
        "--godot-binary",
        default=os.environ.get("GODOT_BINARY", "godot"),
        help="Path to Godot binary (default: GODOT_BINARY env or 'godot').",
    )
    parser.add_argument(
        "--gd-mode",
        choices=("headless", "non-headless", "windows-vulkan"),
        default=os.environ.get("GS_RUNTIME_GD_MODE", "headless"),
        help=(
            "Godot runtime mode. 'windows-vulkan' is the canonical non-headless runtime gate "
            "for issue #897."
        ),
    )
    parser.add_argument(
        "--project-path",
        default=None,
        help="Optional project path passed as '--path <dir>' for GDScript runs.",
    )
    parser.add_argument(
        "--godot-arg",
        action="append",
        default=[],
        help="Extra Godot argument (repeatable).",
    )
    parser.add_argument(
        "--gd-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each GDScript runtime test.",
    )
    parser.add_argument(
        "--cpp-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each compiled C++ harness.",
    )
    parser.add_argument(
        "--cpp-link-mode",
        choices=("standalone", "module-linked"),
        default=os.environ.get("GS_RUNTIME_CPP_LINK_MODE", "standalone"),
        help=(
            "C++ harness link mode. 'standalone' keeps mock-only harness behavior. "
            "'module-linked' requires --cpp-build-manifest."
        ),
    )
    parser.add_argument(
        "--cpp-build-manifest",
        default=None,
        help=(
            "Path to JSON manifest for module-linked C++ harness builds "
            "(include_dirs, compile_flags, link_flags)."
        ),
    )
    parser.add_argument("--skip-cpp", action="store_true", help="Skip C++ harnesses.")
    parser.add_argument("--skip-gd", action="store_true", help="Skip GDScript runtime harnesses.")
    parser.add_argument(
        "--scenario-config",
        default=str(DEFAULT_SCENARIO_CONFIG),
        help=(
            "Path to runtime scenario configuration JSON file "
            f"(default: {DEFAULT_SCENARIO_CONFIG.relative_to(ROOT)})."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=str(RUNTIME_DIR / "runtime_validation_report.json"),
        help="Path to write the JSON validation report (default: tests/runtime/runtime_validation_report.json).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Runtime scenario profile name (from --scenario-config).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available runtime scenario profiles and exit.",
    )
    parser.add_argument(
        "--gd-script",
        action="append",
        default=[],
        help=(
            "Run only the specified GDScript runtime path (relative to repo root). "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--gd-test",
        action="append",
        default=[],
        help=(
            "Run only named GDScript runtime tests from profile/config (repeatable). "
            "Mutually exclusive with --gd-script."
        ),
    )
    parser.add_argument(
        "--cpp-test",
        action="append",
        default=[],
        help="Run only named C++ runtime harness tests from profile/config (repeatable).",
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Treat runtime skip markers as failures.",
    )
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="Allow runtime skip markers without failing the run.",
    )
    parser.add_argument(
        "--allow-skip-test",
        action="append",
        default=[],
        help=(
            "Allow listed runtime tests to skip without failing, even when --fail-on-skip is set. "
            "Repeatable."
        ),
    )
    args = parser.parse_args()
    valid_cpp_link_modes = {"standalone", "module-linked"}
    if args.cpp_link_mode not in valid_cpp_link_modes:
        parser.error(
            "Invalid --cpp-link-mode '{mode}'. Expected one of: {valid}.".format(
                mode=args.cpp_link_mode,
                valid=", ".join(sorted(valid_cpp_link_modes)),
            )
        )
    if args.fail_on_skip and args.allow_skips:
        parser.error("Use only one of --fail-on-skip or --allow-skips.")
    if args.gd_script and args.gd_test:
        parser.error("Use only one of --gd-script or --gd-test.")
    return args


def _extract_marker_reasons(output: str, marker: str) -> List[str]:
    reasons: List[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        idx = line.find(marker)
        if idx == -1:
            continue
        reason = line[idx + len(marker):].strip(" :-")
        reasons.append(reason if reason else line)
    return reasons


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _first_non_empty_line(text: str) -> Optional[str]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return None


def _extract_metrics_payload(output: str) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for raw_line in output.splitlines():
        marker_index = raw_line.find(METRICS_MARKER)
        if marker_index == -1:
            continue
        payload = raw_line[marker_index + len(METRICS_MARKER):].strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            metrics = parsed
    return metrics


def _classify_result(result: TestResult, *, fail_on_skip: bool, allow_skip_tests: set[str]) -> TestResult:
    combined_output = f"{result.stdout}\n{result.stderr}"
    result.metrics = _extract_metrics_payload(combined_output)
    explicit_failures = _extract_marker_reasons(combined_output, FAIL_MARKER)
    skip_reasons = _extract_marker_reasons(combined_output, SKIP_MARKER)

    if result.exit_code != 0:
        result.status = "failed"
        if explicit_failures:
            result.reasons = explicit_failures
            return result
        if result.reasons:
            return result
        fallback = (
            _first_non_empty_line(result.stderr)
            or _first_non_empty_line(result.stdout)
            or f"Exited with code {result.exit_code}"
        )
        result.reasons = [fallback]
        return result

    if explicit_failures:
        result.status = "failed"
        result.reasons = explicit_failures
        return result

    if skip_reasons:
        if result.name in allow_skip_tests:
            result.status = "skipped"
            result.reasons = [f"Allowed runtime skip: {reason}" for reason in skip_reasons]
            return result
        if fail_on_skip:
            result.status = "failed"
            result.reasons = [f"Unexpected runtime skip: {reason}" for reason in skip_reasons]
        else:
            result.status = "skipped"
            result.reasons = skip_reasons
        return result

    result.status = "passed"
    result.reasons = []
    return result


def _format_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(name: str, command: List[str], *, cwd: Optional[Path], timeout: int) -> TestResult:
    print(f"\n[runtime] Running {name}")
    print(f"[runtime] Command: {_format_command(command)}")
    start = time.time()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or ROOT,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)
        print(f"[runtime] ❌ {name} timed out after {timeout}s")
        return TestResult(
            name=name,
            command=command,
            duration=duration,
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            status="failed",
            reasons=[f"Timed out after {timeout}s"],
        )
    except (OSError, PermissionError) as exc:
        duration = time.time() - start
        print(f"[runtime] ❌ {name} failed to launch: {type(exc).__name__}: {exc}")
        return TestResult(
            name=name,
            command=command,
            duration=duration,
            exit_code=127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
            status="failed",
            reasons=[f"{type(exc).__name__}: {exc}"],
        )

    duration = time.time() - start
    status = "ok" if completed.returncode == 0 else f"exit={completed.returncode}"
    print(f"[runtime] Completed {name} ({status}) in {duration:.1f}s")
    return TestResult(
        name=name,
        command=command,
        duration=duration,
        exit_code=completed.returncode,
        stdout=_coerce_text(completed.stdout),
        stderr=_coerce_text(completed.stderr),
    )


def _load_cpp_build_config(link_mode: str, manifest_path_value: Optional[str]) -> CppBuildConfig:
    if link_mode == "standalone":
        return CppBuildConfig(link_mode=link_mode)

    if not manifest_path_value:
        raise ValueError("module-linked C++ mode requires --cpp-build-manifest.")

    manifest_path = Path(manifest_path_value)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()

    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("C++ build manifest must be a JSON object.")

    include_values = raw.get("include_dirs", [])
    compile_values = raw.get("compile_flags", [])
    link_values = raw.get("link_flags", [])
    if not isinstance(include_values, list) or not all(isinstance(value, str) for value in include_values):
        raise ValueError("Manifest field 'include_dirs' must be a string list.")
    if not isinstance(compile_values, list) or not all(isinstance(value, str) for value in compile_values):
        raise ValueError("Manifest field 'compile_flags' must be a string list.")
    if not isinstance(link_values, list) or not all(isinstance(value, str) for value in link_values):
        raise ValueError("Manifest field 'link_flags' must be a string list.")

    include_dirs: list[Path] = []
    for include_dir in include_values:
        path = Path(include_dir)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        else:
            path = path.resolve()
        include_dirs.append(path)

    return CppBuildConfig(
        link_mode=link_mode,
        include_dirs=tuple(include_dirs),
        compile_flags=tuple(str(value) for value in compile_values),
        link_flags=tuple(str(value) for value in link_values),
    )


def compile_cpp_test(name: str, source: Path, cpp_build_config: CppBuildConfig) -> Path:
    ensure_build_dir()
    output = BUILD_DIR / source.stem
    command: List[str] = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
    ]
    for include_dir in cpp_build_config.include_dirs:
        command.append(f"-I{include_dir}")
    command.extend(cpp_build_config.compile_flags)
    command.extend([str(source), "-o", str(output)])
    command.extend(cpp_build_config.link_flags)
    print(f"\n[runtime] Compiling {name}: {source.relative_to(ROOT)}")
    print(f"[runtime] C++ link mode: {cpp_build_config.link_mode}")
    print(f"[runtime] Command: {_format_command(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except (OSError, PermissionError) as exc:
        raise RuntimeError(f"Compiler invocation failed: {type(exc).__name__}: {exc}") from exc

    if result.returncode != 0:
        detail = (
            _first_non_empty_line(result.stderr)
            or _first_non_empty_line(result.stdout)
            or f"Compiler exited with code {result.returncode}"
        )
        raise RuntimeError(detail)

    print(f"[runtime] Built {output.relative_to(ROOT)}")
    return output


def run_cpp_harnesses(timeout: int, selected_tests: Iterable[str], cpp_build_config: CppBuildConfig) -> List[TestResult]:
    try:
        cpp_tests = _resolve_named_test_map(CPP_TESTS, selected_tests, kind="C++ runtime test")
    except ValueError as exc:
        return [
            TestResult(
                name="C++ Runtime Selection",
                command=[],
                duration=0.0,
                exit_code=1,
                stdout="",
                stderr=str(exc),
                status="failed",
                reasons=[str(exc)],
            )
        ]

    results: List[TestResult] = []
    for name, source in cpp_tests.items():
        try:
            binary = compile_cpp_test(name, source, cpp_build_config)
        except RuntimeError as exc:
            result = TestResult(
                name=name,
                command=[os.environ.get("CXX", "g++"), str(source)],
                duration=0.0,
                exit_code=1,
                stdout="",
                stderr=str(exc),
                status="failed",
                reasons=[f"Compilation failed: {exc}"],
            )
            results.append(result)
            continue

        result = run_command(name, [str(binary)], cwd=ROOT, timeout=timeout)
        results.append(_classify_result(result, fail_on_skip=False, allow_skip_tests=set()))
    return results


def _godot_binary_is_available(godot_binary: str) -> Optional[str]:
    try:
        probe = subprocess.run(
            [godot_binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, PermissionError) as exc:
        return f"{type(exc).__name__}: {exc}"

    if probe.returncode != 0:
        output = (probe.stdout or "") + (probe.stderr or "")
        reason = _first_non_empty_line(output) or f"exit code {probe.returncode}"
        return f"Godot binary probe failed: {reason}"
    return None


def _resolve_mode_args(mode: str) -> List[str]:
    if mode == "headless":
        return ["--headless"]
    if mode == "windows-vulkan":
        return ["--render-thread", "safe", "--display-driver", "windows", "--rendering-driver", "vulkan"]
    return []


def _build_godot_command(config: GodotRunConfig, script: Path) -> List[str]:
    command = [config.binary, *_resolve_mode_args(config.mode)]
    if config.project_path is not None:
        command.extend(["--path", str(config.project_path)])
    command.extend(config.extra_args)
    command.extend(["--verbose", "--script", str(script.relative_to(ROOT))])
    return command


def _resolve_gd_test_map(selected_scripts: Iterable[str]) -> Dict[str, Path]:
    selected_list = [entry.strip() for entry in selected_scripts if entry and entry.strip()]
    if not selected_list:
        return dict(GDS_TESTS)

    resolved: Dict[str, Path] = {}
    for script_path_str in selected_list:
        script_path = Path(script_path_str)
        if not script_path.is_absolute():
            script_path = (ROOT / script_path).resolve()
        else:
            script_path = script_path.resolve()

        if not script_path.exists():
            raise FileNotFoundError(f"GDScript runtime test not found: {script_path_str}")

        try:
            relative = script_path.relative_to(ROOT)
        except ValueError as exc:
            raise FileNotFoundError(
                f"GDScript runtime test must be inside repository: {script_path_str}"
            ) from exc

        display_name = relative.stem.replace("_", " ").title()
        resolved[display_name] = script_path
    return resolved


def _resolve_named_test_map(
        all_tests: Dict[str, Path],
        selected_names: Iterable[str],
        *,
        kind: str,
) -> Dict[str, Path]:
    selected_list = [entry.strip() for entry in selected_names if entry and entry.strip()]
    if not selected_list:
        # Empty profile selection must stay empty; this allows profiles to explicitly disable suites.
        return {}

    resolved: Dict[str, Path] = {}
    unknown: List[str] = []
    for name in selected_list:
        if name not in all_tests:
            unknown.append(name)
            continue
        resolved[name] = all_tests[name]

    if unknown:
        available = ", ".join(sorted(all_tests.keys()))
        missing = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {kind} name(s): {missing}. Available: {available}")

    return resolved


def _load_scenario_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Scenario config must be a JSON object.")

    version = raw.get("version")
    if not isinstance(version, int):
        raise ValueError("Scenario config field 'version' must be an integer.")

    default_profile = raw.get("default_profile")
    if not isinstance(default_profile, str) or not default_profile.strip():
        raise ValueError("Scenario config field 'default_profile' must be a non-empty string.")

    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Scenario config field 'profiles' must be a non-empty object.")

    for profile_name, profile_config in profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError("Scenario profile names must be non-empty strings.")
        if not isinstance(profile_config, dict):
            raise ValueError(f"Profile '{profile_name}' must be a JSON object.")

        cpp_tests = profile_config.get("cpp_tests", [])
        gd_tests = profile_config.get("gd_tests", [])
        godot_args = profile_config.get("godot_args", [])

        if not isinstance(cpp_tests, list) or not all(isinstance(value, str) for value in cpp_tests):
            raise ValueError(f"Profile '{profile_name}' field 'cpp_tests' must be a string list.")
        if not isinstance(gd_tests, list) or not all(isinstance(value, str) for value in gd_tests):
            raise ValueError(f"Profile '{profile_name}' field 'gd_tests' must be a string list.")
        if not isinstance(godot_args, list) or not all(isinstance(value, str) for value in godot_args):
            raise ValueError(f"Profile '{profile_name}' field 'godot_args' must be a string list.")

        if "cpp_timeout" in profile_config and not isinstance(profile_config["cpp_timeout"], int):
            raise ValueError(f"Profile '{profile_name}' field 'cpp_timeout' must be an integer.")
        if "gd_timeout" in profile_config and not isinstance(profile_config["gd_timeout"], int):
            raise ValueError(f"Profile '{profile_name}' field 'gd_timeout' must be an integer.")
        if "fail_on_skip" in profile_config and not isinstance(profile_config["fail_on_skip"], bool):
            raise ValueError(f"Profile '{profile_name}' field 'fail_on_skip' must be a boolean.")

    if default_profile not in profiles:
        raise ValueError(
            f"default_profile '{default_profile}' is not defined in profiles: "
            f"{', '.join(sorted(str(name) for name in profiles.keys()))}"
        )

    return raw


def _print_profiles(config: Dict[str, object], *, source: Path) -> None:
    profiles = config.get("profiles", {})
    default_profile = config.get("default_profile", "")
    print(f"[runtime] Scenario config: {source}")
    print("[runtime] Available profiles:")
    for profile_name in sorted(profiles.keys()):
        profile_config = profiles[profile_name]
        description = ""
        if isinstance(profile_config, dict):
            description = str(profile_config.get("description", "")).strip()
        default_tag = " (default)" if profile_name == default_profile else ""
        if description:
            print(f"  - {profile_name}{default_tag}: {description}")
        else:
            print(f"  - {profile_name}{default_tag}")


def _validate_summary_schema(summary: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    required_fields = ("total", "passed", "failed", "skipped", "duration", "tests")
    for field_name in required_fields:
        if field_name not in summary:
            errors.append(f"Missing summary field '{field_name}'.")

    for integer_field in ("total", "passed", "failed", "skipped"):
        value = summary.get(integer_field)
        if not isinstance(value, int):
            errors.append(f"Field '{integer_field}' must be an integer.")
        elif value < 0:
            errors.append(f"Field '{integer_field}' must be >= 0.")

    duration = summary.get("duration")
    if not isinstance(duration, (int, float)):
        errors.append("Field 'duration' must be numeric.")
    elif duration < 0:
        errors.append("Field 'duration' must be >= 0.")

    tests = summary.get("tests")
    if not isinstance(tests, list):
        errors.append("Field 'tests' must be a list.")
        return errors

    allowed_statuses = {"passed", "failed", "skipped"}
    for index, test_entry in enumerate(tests):
        prefix = f"tests[{index}]"
        if not isinstance(test_entry, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        for required_key in ("name", "status", "reasons", "command", "duration", "exit_code", "metrics"):
            if required_key not in test_entry:
                errors.append(f"{prefix} missing '{required_key}'.")
        name = test_entry.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}.name must be a non-empty string.")
        status = test_entry.get("status")
        if not isinstance(status, str) or status not in allowed_statuses:
            errors.append(f"{prefix}.status must be one of {sorted(allowed_statuses)}.")
        reasons = test_entry.get("reasons")
        if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
            errors.append(f"{prefix}.reasons must be a string list.")
        command = test_entry.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            errors.append(f"{prefix}.command must be a string list.")
        test_duration = test_entry.get("duration")
        if not isinstance(test_duration, (int, float)) or test_duration < 0:
            errors.append(f"{prefix}.duration must be numeric >= 0.")
        exit_code = test_entry.get("exit_code")
        if not isinstance(exit_code, int):
            errors.append(f"{prefix}.exit_code must be an integer.")
        metrics = test_entry.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{prefix}.metrics must be an object.")

    if isinstance(tests, list):
        expected_total = summary.get("total")
        if isinstance(expected_total, int) and expected_total != len(tests):
            errors.append(
                f"Field 'total' ({expected_total}) does not match number of test entries ({len(tests)})."
            )

    return errors


def run_gd_tests(
        config: GodotRunConfig,
        selected_scripts: Iterable[str],
        selected_tests: Iterable[str],
) -> List[TestResult]:
    gd_scripts = [entry for entry in selected_scripts if entry and entry.strip()]
    if gd_scripts:
        try:
            gd_tests = _resolve_gd_test_map(gd_scripts)
        except FileNotFoundError as exc:
            return [
                TestResult(
                    name="GDScript Runtime Selection",
                    command=[],
                    duration=0.0,
                    exit_code=1,
                    stdout="",
                    stderr=str(exc),
                    status="failed",
                    reasons=[str(exc)],
                )
            ]
    else:
        try:
            gd_tests = _resolve_named_test_map(GDS_TESTS, selected_tests, kind="GDScript runtime test")
        except ValueError as exc:
            return [
                TestResult(
                    name="GDScript Runtime Selection",
                    command=[],
                    duration=0.0,
                    exit_code=1,
                    stdout="",
                    stderr=str(exc),
                    status="failed",
                    reasons=[str(exc)],
                )
            ]

    availability_error = _godot_binary_is_available(config.binary)
    if availability_error:
        results: List[TestResult] = []
        for name, script in gd_tests.items():
            command = _build_godot_command(config, script)
            results.append(
                TestResult(
                    name=name,
                    command=command,
                    duration=0.0,
                    exit_code=127,
                    stdout="",
                    stderr=availability_error,
                    status="failed",
                    reasons=[availability_error],
                )
            )
        return results

    print(
        "[runtime] GDScript mode="
        f"{config.mode} fail_on_skip={'yes' if config.fail_on_skip else 'no'} "
        f"allow_skip_tests={len(config.allow_skip_tests)}"
    )
    results: List[TestResult] = []
    allow_skip_tests = set(config.allow_skip_tests)
    for name, script in gd_tests.items():
        command = _build_godot_command(config, script)
        result = run_command(name, command, cwd=ROOT, timeout=config.timeout)
        results.append(
            _classify_result(
                result,
                fail_on_skip=config.fail_on_skip,
                allow_skip_tests=allow_skip_tests,
            )
        )
    return results


def summarise(results: List[TestResult]) -> Dict[str, object]:
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.status == "passed"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "duration": sum(r.duration for r in results),
        "tests": [
            {
                "name": r.name,
                "status": r.status,
                "reasons": r.reasons,
                "command": r.command,
                "duration": r.duration,
                "exit_code": r.exit_code,
                "metrics": r.metrics,
            }
            for r in results
        ],
    }
    return summary


def _print_summary(summary: Dict[str, object]) -> None:
    print("\n=== Runtime Validation Summary ===")
    print(
        "[runtime] total={total} passed={passed} failed={failed} skipped={skipped} duration={duration:.1f}s".format(
            total=summary["total"],
            passed=summary["passed"],
            failed=summary["failed"],
            skipped=summary["skipped"],
            duration=summary["duration"],
        )
    )
    schema_valid = bool(summary.get("schema_valid", False))
    if schema_valid:
        print("[runtime] summary schema validation: ok")
    else:
        print("[runtime] summary schema validation: FAILED")
    print(json.dumps(summary, indent=2))


def main() -> int:
    args = _parse_args()
    ensure_build_dir()
    try:
        ensure_synthetic_assets()
    except RuntimeError as exc:
        print(f"[runtime] ❌ {exc}")
        return 1

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = (ROOT / report_path).resolve()
    else:
        report_path = report_path.resolve()

    scenario_config_path = Path(args.scenario_config)
    if not scenario_config_path.is_absolute():
        scenario_config_path = (ROOT / scenario_config_path).resolve()
    else:
        scenario_config_path = scenario_config_path.resolve()

    try:
        scenario_config = _load_scenario_config(scenario_config_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[runtime] ❌ invalid scenario config: {exc}")
        return 1

    if args.list_profiles:
        _print_profiles(scenario_config, source=scenario_config_path)
        return 0

    cpp_build_config = CppBuildConfig(link_mode=args.cpp_link_mode)

    raw_profiles = scenario_config.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        print("[runtime] ❌ scenario config has invalid 'profiles' section.")
        return 1

    profile_name = args.profile or str(scenario_config.get("default_profile", "")).strip()
    if not profile_name:
        print("[runtime] ❌ no profile selected and scenario config has no default_profile.")
        return 1
    if profile_name not in raw_profiles:
        print(
            "[runtime] ❌ unknown profile '{profile}'. available: {available}".format(
                profile=profile_name,
                available=", ".join(sorted(str(name) for name in raw_profiles.keys())),
            )
        )
        return 1

    profile_config = raw_profiles[profile_name]
    if not isinstance(profile_config, dict):
        print(f"[runtime] ❌ scenario profile '{profile_name}' must be an object.")
        return 1

    profile_cpp_tests = [str(name) for name in profile_config.get("cpp_tests", [])]
    profile_gd_tests = [str(name) for name in profile_config.get("gd_tests", [])]
    profile_godot_args = tuple(str(value) for value in profile_config.get("godot_args", []))
    cpp_timeout = int(profile_config.get("cpp_timeout", args.cpp_timeout))
    gd_timeout = int(profile_config.get("gd_timeout", args.gd_timeout))

    selected_cpp_tests = args.cpp_test if args.cpp_test else profile_cpp_tests
    selected_gd_tests = args.gd_test if args.gd_test else profile_gd_tests
    should_run_cpp_harnesses = not args.skip_cpp and len(selected_cpp_tests) > 0
    if should_run_cpp_harnesses:
        try:
            cpp_build_config = _load_cpp_build_config(args.cpp_link_mode, args.cpp_build_manifest)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"[runtime] ❌ invalid C++ build configuration: {exc}")
            return 1
    allow_skip_tests = tuple(dict.fromkeys(str(name) for name in args.allow_skip_test if str(name).strip()))

    default_fail_on_skip = args.gd_mode != "headless"
    if args.fail_on_skip:
        fail_on_skip = True
    elif args.allow_skips:
        fail_on_skip = False
    elif "fail_on_skip" in profile_config:
        fail_on_skip = bool(profile_config.get("fail_on_skip"))
    else:
        fail_on_skip = default_fail_on_skip

    project_path = Path(args.project_path).resolve() if args.project_path else None
    gd_config = GodotRunConfig(
        binary=args.godot_binary,
        mode=args.gd_mode,
        timeout=gd_timeout,
        extra_args=tuple(profile_godot_args) + tuple(args.godot_arg),
        fail_on_skip=fail_on_skip,
        allow_skip_tests=allow_skip_tests,
        project_path=project_path,
    )

    try:
        config_display = scenario_config_path.relative_to(ROOT)
    except ValueError:
        config_display = scenario_config_path
    print(
        f"[runtime] profile='{profile_name}' scenario_config='{config_display}' "
        f"cpp_tests={len(selected_cpp_tests)} gd_tests={len(selected_gd_tests)} "
        f"cpp_link_mode={cpp_build_config.link_mode}"
    )

    all_results: List[TestResult] = []
    if should_run_cpp_harnesses:
        all_results.extend(
            run_cpp_harnesses(
                timeout=cpp_timeout,
                selected_tests=selected_cpp_tests,
                cpp_build_config=cpp_build_config,
            )
        )
    if not args.skip_gd:
        all_results.extend(run_gd_tests(gd_config, args.gd_script, selected_gd_tests))

    summary = summarise(all_results)
    summary["profile"] = profile_name
    summary["scenario_config"] = str(config_display)
    schema_errors = _validate_summary_schema(summary)
    summary["schema_valid"] = len(schema_errors) == 0
    summary["schema_errors"] = schema_errors
    if schema_errors:
        for error in schema_errors:
            print(f"[runtime] ❌ schema: {error}")
    _print_summary(summary)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    try:
        report_display = report_path.relative_to(ROOT)
    except ValueError:
        report_display = report_path
    print(f"[runtime] Report saved to {report_display}")

    return 0 if summary["failed"] == 0 and summary["schema_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
