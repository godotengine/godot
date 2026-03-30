#!/usr/bin/env python3
"""Run Gaussian Splatting module tests via Godot's built-in test runner.

If the binary was built without tests enabled, behavior is controlled by
strict/warn-only policy (strict fails, warn-only skips).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
MODULE_SOURCE_DIR = ROOT / "modules" / "gaussian_splatting"
RENDERER_DIR = MODULE_SOURCE_DIR / "renderer"
BUILD_METADATA_GUARD_SCRIPT = MODULE_SOURCE_DIR / "tests" / "check_build_metadata_consistency.py"
SHADER_DEPENDENCY_GUARD_SCRIPT = MODULE_SOURCE_DIR / "tests" / "check_shader_dependency_contract.py"
GAUSSIAN_LAYOUT_GUARD_SCRIPT = ROOT / "tests" / "ci" / "check_gaussian_layout_sync.py"
HISTORY_ARTIFACT_AUDIT_SCRIPT = ROOT / "scripts" / "repo" / "history_artifact_audit.py"
SYNTHETIC_ASSET_PREP_SCRIPT = ROOT / "tests" / "runtime" / "prepare_synthetic_assets.py"
BENCHMARK_ASSET_GUARD_SCRIPT = ROOT / "tests" / "runtime" / "check_benchmark_asset_paths.py"
SOURCE_TREES = (ROOT,)
HEADLESS_GAUSSIAN_SCOPED_TAGS: tuple[str, ...] = (
    # Only tags whose TEST_CASEs are registered at runtime belong here.
    # Standalone .cpp test files compile into the module static library but their
    # doctest registrations are stripped by the linker — only .h tests included
    # via modules_tests.gen.h actually register.  Phantom tags (zero runtime
    # tests) must NOT appear here because strict lanes fail on zero coverage.
    "Animation",
    "ComputeInfra",
    "Config",
    "Container",
    "DynamicInstance",
    "Editor",
    "Importer",
    "Node",
    "PLY",
    "Persistence",
    "SceneTree",
    "SortBenchmark",
    "Synthetic",
    "VRAMBudgetRegulator",
    "ViewTransform",
    "WorldIO",
)
UNTAGGED_GAUSSIAN_EXCLUDE_TAGS: tuple[str, ...] = HEADLESS_GAUSSIAN_SCOPED_TAGS + (
    "GeneratePLY",  # fixture generator with disk side effects; invoked explicitly by prepare_synthetic_assets.py
    "Renderer",  # only aspirational stubs currently; advisory lane below
    "RequiresGPU",
    "Thumbnail",
    "World",
)
UNTAGGED_GAUSSIAN_EXCLUDE_FILTERS: tuple[str, ...] = tuple(
    f"*GaussianSplatting*][{tag}]*" for tag in UNTAGGED_GAUSSIAN_EXCLUDE_TAGS
)
MODULE_TEST_FILTERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...] = (
    # (name, test_case_filters, test_case_exclude_filters, strict)
    # Split the canonical headless Gaussian lane into deterministic subsets so a
    # single-process crash in one area does not erase summary output for the rest
    # of the suite.
    ("GaussianSplatting [Animation]", ("*GaussianSplatting*][Animation]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [ComputeInfra]", ("*GaussianSplatting*][ComputeInfra]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [Config]", ("*GaussianSplatting*][Config]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [Container]", ("*GaussianSplatting*][Container]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [DynamicInstance]", ("*GaussianSplatting*][DynamicInstance]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [Editor]", ("*GaussianSplatting*][Editor]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [Importer]", ("*GaussianSplatting*][Importer]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [Node]", ("*GaussianSplatting*][Node]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [PLY]", ("*GaussianSplatting*][PLY]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [Persistence]", ("*GaussianSplatting*][Persistence]*",), ("*][RequiresGPU]*",), True),
    (
        "GaussianSplatting [SceneTree]",
        ("*GaussianSplatting*][SceneTree]*",),
        (
            "*][RequiresGPU]*",
            "*][Node][SceneTree]*",
            "*][Container][SceneTree]*",
            "*][DynamicInstance][SceneTree]*",
            "*][World][SceneTree]*",
        ),
        True,
    ),
    ("GaussianSplatting [SortBenchmark]", ("*GaussianSplatting*][SortBenchmark]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [Synthetic]", ("*GaussianSplatting*][Synthetic]*",), ("*][RequiresGPU]*",), False),
    ("GaussianSplatting [VRAMBudgetRegulator]", ("*GaussianSplatting*][VRAMBudgetRegulator]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [ViewTransform]", ("*GaussianSplatting*][ViewTransform]*",), ("*][RequiresGPU]*",), True),
    ("GaussianSplatting [WorldIO]", ("*GaussianSplatting*][WorldIO]*",), ("*][RequiresGPU]*",), True),
    # Safety-net lane for unscoped [GaussianSplatting] tests.  Advisory because
    # doctest's --test-case-exclude parsing is unreliable beyond ~10 repeated
    # flags, so the exclude list cannot guarantee precise filtering.  Real
    # coverage lives in the per-tag strict lanes above.
    ("GaussianSplatting [untagged]", ("*GaussianSplatting*",), UNTAGGED_GAUSSIAN_EXCLUDE_FILTERS, False),
    # Use stable description fragments instead of tag prefixes for secondary
    # lanes, as doctest matching can differ depending on how bracketed prefixes
    # are parsed in test names.
    ("GaussianSplatting [Renderer]", ("*GaussianSplatting*][Renderer]*",), ("*][RequiresGPU]*",), False),
    ("TileRenderer", ("*Shader compilation on local device*",), (), False),
    ("GPU Memory Stream", ("*Triple Buffering*",), (), False),
    ("Streaming Pipeline", ("*Concurrent LOD and visibility updates*",), (), False),
)
# Renderer-dependent (requires-RD) doctest lane.  Under Godot's --test mode
# every test here will skip because no RenderingDevice is available.  This lane
# exists for future use when a full-engine test harness is added; in the
# meantime it serves as a catalogue of renderer-dependent tests.
REQUIRES_RD_TEST_FILTERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...] = (
    ("GaussianSplatting [requires-RD]", ("*GaussianSplatting*][RequiresGPU]*",), (), False),
)
GS_RUN_GPU_TESTS_ENV = "GS_RUN_GPU_TESTS"
DISALLOWED_TRACKED_ARTIFACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Python cache directory", re.compile(r"(^|/)__pycache__/")),
    ("Python bytecode file", re.compile(r"\.pyc$")),
    ("Root screenshot dump", re.compile(r"^Screenshot [^/]+\.png$")),
    ("Runtime Linux log output", re.compile(r"^tests/runtime/linux_logs/")),
    ("Runtime log output", re.compile(r"^tests/runtime/.*\.log$")),
)
TRACKED_SYNTHETIC_PLY_PATTERN = re.compile(r"^(tests|templates)/.*\.ply$")
REQUIRED_IGNORED_PATH_PROBES: tuple[str, ...] = (
    "tests/runtime/linux_logs/.hygiene_guard_probe.log",
    "tests/runtime/windows_logs/.hygiene_guard_probe.log",
    "baseline_qa_results.json",
    "tests/ci/qa_results.json",
)
ALLOW_SETTING_TOKEN = "GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION"
ALLOW_FS_WRITE_TOKEN = "GS_CI_ALLOW_RENDER_PATH_FS_WRITE"
VALIDATION_MODE_ENV = "GS_CI_VALIDATION_MODE"
# Controls behavior when the Godot binary was built without tests enabled.
# - strict: unavailable tests are fatal.
# - warn-only: unavailable tests are logged and skipped.
# Defaults to strict in CI and warn-only locally.
TEST_AVAILABILITY_MODE_ENV = "GS_CI_MODULE_TEST_AVAILABILITY_MODE"
# Explicit override for local/debug flows that need to bypass unavailable tests.
ALLOW_TESTS_UNAVAILABLE_ENV = "GS_CI_ALLOW_TESTS_UNAVAILABLE"
HISTORY_ARTIFACT_GUARD_MODE_ENV = "GS_CI_HISTORY_ARTIFACT_GUARD_MODE"
HISTORY_ARTIFACT_GUARD_MODES = ("off", "warn", "strict")
HISTORY_ARTIFACT_MATCH_COUNT_RE = re.compile(r"Matched blob entries:\s*(\d+)")
DOCTEST_SKIP_MARKER_RE = re.compile(r"(?m)^\s*(?:Skipping(?: test)?\s*-\s+.+)$")

SETTING_MUTATION_RE = re.compile(r"->set_setting\s*\(")
FS_WRITE_RULES = (
    ("ProjectSettings save", re.compile(r"\b(?:ps|project_settings|settings)->save\s*\(")),
    ("FileAccess write open", re.compile(r"\bFileAccess::open\s*\(.*FileAccess::(?:WRITE|APPEND|READ_WRITE)\b")),
    ("ResourceSaver save", re.compile(r"\bResourceSaver::save\s*\(")),
    ("DirAccess mutation", re.compile(r"\bDirAccess::(?:make_dir(?:_recursive)?|rename|remove|copy|copy_absolute)\s*\(")),
    ("FileAccess store_* write", re.compile(r"->store_(?:8|16|32|64|float|double|string|line|csv_line|buffer)\s*\(")),
)

STATIC_FORMAT_GUARDS: tuple[tuple[str, Path, tuple[str, ...]], ...] = (
    (
        "tile_compute_rgba8_gate",
        MODULE_SOURCE_DIR / "renderer" / "render_pipeline_stages.cpp",
        (
            r"static RD::DataFormat _resolve_compute_friendly_raster_format\(RD::DataFormat p_format\)",
            r"case RD::DATA_FORMAT_R8G8B8A8_SRGB:\s*.*?return RD::DATA_FORMAT_R8G8B8A8_UNORM;",
            r"const RD::DataFormat raster_output_format = _resolve_compute_friendly_raster_format\(target_format\);",
            r"Tile fallback format override: requested=%d resolved=%d",
        ),
    ),
    (
        "render_output_default_format",
        MODULE_SOURCE_DIR / "renderer" / "render_output_orchestrator.cpp",
        (
            r"if \(target_format == RD::DATA_FORMAT_MAX\)\s*\{\s*target_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;",
        ),
    ),
    (
        "output_copy_format_mismatch_gate",
        MODULE_SOURCE_DIR / "interfaces" / "output_compositor.cpp",
        (
            r"bool format_mismatch = false;",
            r"destination_format\.format != source_format\.format",
            r"bool can_direct_copy = .*?!format_mismatch.*?;",
        ),
    ),
)

_approved_tracked_synthetic_ply_fixtures_cache: set[str] | None = None


def _run_command(args: list[str], cwd: Path = ROOT) -> tuple[int, str, str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout or "", result.stderr or ""


def _build_doctest_run_args(test_case_filters: tuple[str, ...], test_case_exclude_filters: tuple[str, ...]) -> list[str]:
    run_args = ["--headless", "--test"]
    for test_filter in test_case_filters:
        run_args.append(f"--test-case={test_filter}")
    for exclude_filter in test_case_exclude_filters:
        run_args.append(f"--test-case-exclude={exclude_filter}")
    return run_args


def _normalize_process_arg(value: str) -> str:
    # GitHub Actions outputs on Windows can include leading BOM or trailing CR/LF.
    # Also strip embedded NUL bytes defensively before subprocess invocation.
    return str(value).lstrip("\ufeff").replace("\x00", "").strip()


def _resolve_guard_base_ref(explicit_ref: str | None) -> str | None:
    def _resolve_ref(ref: str) -> str | None:
        code, out, _ = _run_command(["git", "rev-parse", "--verify", ref])
        if code == 0:
            return out.strip()
        return None

    def _merge_base(ref: str) -> str | None:
        code, out, _ = _run_command(["git", "merge-base", "HEAD", ref])
        if code == 0:
            return out.strip()
        return None

    if explicit_ref:
        if explicit_ref.lower() == "head":
            return _resolve_ref("HEAD")
        return _resolve_ref(explicit_ref) or _merge_base(explicit_ref)

    candidates = ("HEAD~1", "origin/main", "origin/master", "main", "master")
    for candidate in candidates:
        resolved = _merge_base(candidate)
        if resolved:
            return resolved
        resolved = _resolve_ref(candidate)
        if resolved:
            return resolved
    return None


def _parse_added_lines(diff_text: str) -> list[tuple[int, str]]:
    added: list[tuple[int, str]] = []
    current_new_line: int | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    for raw_line in diff_text.splitlines():
        hunk_match = hunk_re.match(raw_line)
        if hunk_match:
            current_new_line = int(hunk_match.group(1))
            continue
        if raw_line.startswith("+++ ") or raw_line.startswith("--- "):
            continue
        if current_new_line is None:
            continue

        if raw_line.startswith("+"):
            added.append((current_new_line, raw_line[1:]))
            current_new_line += 1
        elif raw_line.startswith("-"):
            continue
        else:
            current_new_line += 1

    return added


def _check_render_path_guards(base_ref: str | None) -> tuple[bool, list[str]]:
    if not RENDERER_DIR.exists():
        return True, []
    if base_ref is None:
        if os.environ.get("CI"):
            return False, ["Unable to determine git base ref for renderer guard in CI."]
        return True, ["Skipping renderer guard (no git base ref found outside CI)."]

    diff_range = f"{base_ref}...HEAD"
    changed_code, changed_out, changed_err = _run_command([
        "git", "diff", "--name-only", "--diff-filter=AMRTUXB", diff_range, "--", str(RENDERER_DIR.relative_to(ROOT))
    ])
    if changed_code != 0:
        return False, [f"Failed to enumerate renderer diffs: {changed_err.strip()}"]

    changed_files = [line.strip() for line in changed_out.splitlines() if line.strip()]
    if not changed_files:
        return True, []

    violations: list[str] = []
    for rel_path in changed_files:
        code, diff_out, diff_err = _run_command([
            "git", "diff", "--no-color", "--unified=0", diff_range, "--", rel_path
        ])
        if code != 0:
            violations.append(f"{rel_path}: failed to inspect diff ({diff_err.strip()})")
            continue

        for line_no, line_text in _parse_added_lines(diff_out):
            stripped = line_text.strip()
            if not stripped:
                continue

            if SETTING_MUTATION_RE.search(line_text) and ALLOW_SETTING_TOKEN not in line_text:
                violations.append(
                    f"{rel_path}:{line_no}: render-path set_setting mutation requires guard token "
                    f"'{ALLOW_SETTING_TOKEN}' on the same line."
                )

            for label, rule in FS_WRITE_RULES:
                if rule.search(line_text) and ALLOW_FS_WRITE_TOKEN not in line_text:
                    violations.append(
                        f"{rel_path}:{line_no}: {label} requires guard token "
                        f"'{ALLOW_FS_WRITE_TOKEN}' on the same line."
                    )

    return len(violations) == 0, violations


def _run_static_format_guards() -> tuple[bool, list[str]]:
    failures: list[str] = []
    for guard_name, file_path, required_patterns in STATIC_FORMAT_GUARDS:
        rel_path = file_path.relative_to(ROOT)
        if not file_path.is_file():
            failures.append(f"{guard_name}: missing file '{rel_path}'")
            continue
        try:
            contents = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"{guard_name}: failed reading '{rel_path}': {exc}")
            continue

        for pattern in required_patterns:
            if re.search(pattern, contents, re.MULTILINE | re.DOTALL) is None:
                failures.append(
                    f"{guard_name}: missing expected pattern '{pattern}' in '{rel_path}'"
                )

    return not failures, failures


def _run_tracked_backup_guard() -> tuple[bool, list[str]]:
    source_tree_roots = [str(path.relative_to(ROOT)) for path in SOURCE_TREES]
    code, out, err = _run_command(["git", "ls-files", "--", *source_tree_roots])
    if code != 0:
        return False, [f"Failed to enumerate tracked source files for backup guard: {err.strip()}"]

    tracked_backups = sorted(
        line.strip() for line in out.splitlines() if line.strip().endswith(".backup")
    )
    if not tracked_backups:
        return True, []

    violations = ["Tracked '*.backup' files are not allowed under source trees:"]
    violations.extend(f"  - {path}" for path in tracked_backups)
    violations.append("Remove these files from git tracking (for example, with 'git rm <path>').")
    return False, violations


def _run_tracked_artifact_guard() -> tuple[bool, list[str]]:
    approved_result = _get_approved_tracked_synthetic_ply_fixtures()
    if isinstance(approved_result, list):
        return False, approved_result
    approved_tracked_synthetic_plys = approved_result

    code, out, err = _run_command(["git", "ls-files"])
    if code != 0:
        return False, [f"Failed to enumerate tracked files for artifact guard: {err.strip()}"]

    tracked_paths = [line.strip() for line in out.splitlines() if line.strip()]
    violations: list[str] = []
    for tracked_path in tracked_paths:
        for label, pattern in DISALLOWED_TRACKED_ARTIFACT_PATTERNS:
            if pattern.search(tracked_path):
                violations.append(f"  - {tracked_path} ({label})")
                break
        else:
            if TRACKED_SYNTHETIC_PLY_PATTERN.search(tracked_path) and tracked_path not in approved_tracked_synthetic_plys:
                violations.append(f"  - {tracked_path} (Unexpected tracked synthetic PLY fixture)")

    status_code, status_out, status_err = _run_command(
        ["git", "status", "--porcelain=1", "--untracked-files=all", "--", "tests", "templates"]
    )
    if status_code != 0:
        return False, [f"Failed to inspect synthetic fixture status for artifact guard: {status_err.strip()}"]

    for raw_line in status_out.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path_text = raw_line[3:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1].strip()
        normalized_path = path_text.replace("\\", "/")
        if not TRACKED_SYNTHETIC_PLY_PATTERN.search(normalized_path):
            continue

        if normalized_path in approved_tracked_synthetic_plys:
            if status != "??":
                violations.append(
                    f"  - {normalized_path} (Approved tracked synthetic PLY fixture is dirty: git status '{status}')"
                )
            continue

        if status == "??":
            violations.append(f"  - {normalized_path} (Unexpected untracked synthetic PLY artifact)")
        else:
            violations.append(f"  - {normalized_path} (Unexpected dirty synthetic PLY artifact: git status '{status}')")

    for probe_path in REQUIRED_IGNORED_PATH_PROBES:
        check_code, _, _ = _run_command(["git", "check-ignore", "--quiet", "--no-index", probe_path])
        if check_code != 0:
            violations.append(
                f"  - Missing ignore rule for '{probe_path}' (required runtime log ignore)."
            )

    if not violations:
        return True, []

    messages = ["Tracked artifact hygiene guard failed:"]
    messages.extend(violations)
    messages.append("Remove tracked artifacts and update .gitignore patterns before merging.")
    return False, messages


def _get_approved_tracked_synthetic_ply_fixtures() -> set[str] | list[str]:
    global _approved_tracked_synthetic_ply_fixtures_cache
    if _approved_tracked_synthetic_ply_fixtures_cache is not None:
        return _approved_tracked_synthetic_ply_fixtures_cache

    if not SYNTHETIC_ASSET_PREP_SCRIPT.is_file():
        return [
            "Tracked artifact hygiene guard failed:",
            f"  - Missing synthetic asset prep script: {SYNTHETIC_ASSET_PREP_SCRIPT.relative_to(ROOT)}",
        ]

    try:
        spec = importlib.util.spec_from_file_location(
            "tests.runtime.prepare_synthetic_assets", SYNTHETIC_ASSET_PREP_SCRIPT
        )
        if spec is None or spec.loader is None:
            return [
                "Tracked artifact hygiene guard failed:",
                f"  - Unable to load synthetic asset definitions from {SYNTHETIC_ASSET_PREP_SCRIPT.relative_to(ROOT)}",
            ]
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        return [
            "Tracked artifact hygiene guard failed:",
            f"  - Failed to import synthetic asset definitions from {SYNTHETIC_ASSET_PREP_SCRIPT.relative_to(ROOT)}: {exc}",
        ]

    canonical_specs = getattr(module, "CANONICAL_SPECS", None)
    if canonical_specs is None:
        return [
            "Tracked artifact hygiene guard failed:",
            f"  - Synthetic asset prep script does not expose CANONICAL_SPECS: {SYNTHETIC_ASSET_PREP_SCRIPT.relative_to(ROOT)}",
        ]

    approved_paths: set[str] = set()
    for spec_entry in canonical_specs:
        relative_path = getattr(spec_entry, "relative_path", "")
        if not relative_path:
            continue
        normalized_path = str(relative_path).replace("\\", "/")
        if normalized_path.endswith(".ply") and TRACKED_SYNTHETIC_PLY_PATTERN.search(normalized_path):
            approved_paths.add(normalized_path)

    _approved_tracked_synthetic_ply_fixtures_cache = approved_paths
    return approved_paths


def _run_build_metadata_guard() -> tuple[bool, list[str]]:
    if not BUILD_METADATA_GUARD_SCRIPT.is_file():
        return False, [f"Missing build metadata guard script: {BUILD_METADATA_GUARD_SCRIPT.relative_to(ROOT)}"]

    code, out, err = _run_command([sys.executable, str(BUILD_METADATA_GUARD_SCRIPT)])
    output_lines = [line for line in (out + err).splitlines() if line.strip()]

    if code != 0:
        if not output_lines:
            output_lines = [f"Build metadata guard failed with exit code {code}."]
        return False, output_lines

    return True, output_lines


def _run_shader_dependency_guard() -> tuple[bool, list[str]]:
    if not SHADER_DEPENDENCY_GUARD_SCRIPT.is_file():
        return False, [
            f"Missing shader dependency guard script: {SHADER_DEPENDENCY_GUARD_SCRIPT.relative_to(ROOT)}"
        ]

    code, out, err = _run_command([sys.executable, str(SHADER_DEPENDENCY_GUARD_SCRIPT)])
    output_lines = [line for line in (out + err).splitlines() if line.strip()]

    if code != 0:
        if not output_lines:
            output_lines = [f"Shader dependency guard failed with exit code {code}."]
        return False, output_lines

    return True, output_lines


def _run_gaussian_layout_guard() -> tuple[bool, list[str]]:
    if not GAUSSIAN_LAYOUT_GUARD_SCRIPT.is_file():
        return False, [
            f"Missing Gaussian layout guard script: {GAUSSIAN_LAYOUT_GUARD_SCRIPT.relative_to(ROOT)}"
        ]

    code, out, err = _run_command([sys.executable, str(GAUSSIAN_LAYOUT_GUARD_SCRIPT)])
    output_lines = [line for line in (out + err).splitlines() if line.strip()]

    if code != 0:
        if not output_lines:
            output_lines = [f"Gaussian layout guard failed with exit code {code}."]
        return False, output_lines

    return True, output_lines


def _tests_unavailable(output: str) -> bool:
    normalized_output = " ".join(output.lower().split())
    markers = (
        "unknown option '--test'",
        "unknown option \"--test\"",
        "unknown option '--test-case'",
        "unknown option '--test-suite'",
        "testing is disabled",
        "tests are disabled",
        "support for tests is disabled",
        "compiled without support for unit test",
    )
    return any(marker in normalized_output for marker in markers)


def _env_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_ci() -> bool:
    return _env_truthy(os.environ.get("CI", ""))


def _resolve_tests_unavailable_mode(explicit_mode: str | None) -> str:
    if explicit_mode in ("strict", "warn-only"):
        return explicit_mode

    shared_mode = os.environ.get(VALIDATION_MODE_ENV, "").strip().lower()
    if shared_mode in ("strict", "warn-only"):
        return shared_mode

    return "strict" if os.environ.get("CI") else "warn-only"


def _resolve_history_artifact_guard_mode() -> tuple[str, str | None]:
    mode = os.environ.get(HISTORY_ARTIFACT_GUARD_MODE_ENV, "").strip().lower()
    if not mode:
        return "warn", None
    if mode in HISTORY_ARTIFACT_GUARD_MODES:
        return mode, None
    return (
        "warn",
        (
            f"Invalid {HISTORY_ARTIFACT_GUARD_MODE_ENV}='{mode}'. "
            "Falling back to 'warn'."
        ),
    )


def _run_history_artifact_guard(mode: str) -> tuple[bool, int, list[str]]:
    messages = [
        f"History artifact guard mode: {mode} (env: {HISTORY_ARTIFACT_GUARD_MODE_ENV})."
    ]

    if mode == "off":
        messages.append("History artifact guard skipped (mode=off).")
        return True, 0, messages

    if not HISTORY_ARTIFACT_AUDIT_SCRIPT.is_file():
        missing_msg = (
            f"Missing history artifact audit script: {HISTORY_ARTIFACT_AUDIT_SCRIPT.relative_to(ROOT)}"
        )
        if mode == "strict":
            return False, 1, messages + [missing_msg]
        messages.append(f"{missing_msg}; skipping history audit in warn mode.")
        return True, 0, messages

    code, out, err = _run_command([sys.executable, str(HISTORY_ARTIFACT_AUDIT_SCRIPT)], cwd=ROOT)
    combined_output = (out + err).strip()
    if combined_output:
        for line in combined_output.splitlines():
            messages.append(f"[history-audit] {line}")

    if code != 0:
        messages.append(f"History artifact guard failed: audit exited with code {code}.")
        return False, 1, messages

    match = HISTORY_ARTIFACT_MATCH_COUNT_RE.search(out + err)
    if match is None:
        messages.append(
            "History artifact guard failed: unable to parse 'Matched blob entries' from audit output."
        )
        return False, 1, messages

    matched_entries = int(match.group(1))
    if matched_entries <= 0:
        messages.append("History artifact guard passed: no matched history artifact entries.")
        return True, 0, messages

    entry_label = "entry" if matched_entries == 1 else "entries"
    if mode == "strict":
        messages.append(
            f"History artifact guard strict failure: found {matched_entries} matched history artifact {entry_label}."
        )
        return False, 3, messages

    messages.append(
        f"History artifact guard warning: found {matched_entries} matched history artifact {entry_label}. "
        "Continuing because mode=warn."
    )
    return True, 0, messages


def _prepare_synthetic_assets() -> tuple[bool, list[str]]:
    if not SYNTHETIC_ASSET_PREP_SCRIPT.is_file():
        return (
            False,
            [f"Missing synthetic asset prep script: {SYNTHETIC_ASSET_PREP_SCRIPT.relative_to(ROOT)}"],
        )

    code, out, err = _run_command(
        [sys.executable, str(SYNTHETIC_ASSET_PREP_SCRIPT), "--quiet"],
        cwd=ROOT,
    )

    messages = ["Preparing synthetic PLY assets for runtime and template lanes."]
    combined = (out + err).strip()
    if combined:
        messages.extend(combined.splitlines())
    if code != 0:
        messages.append(f"Synthetic asset preparation failed with exit code {code}.")
        return False, messages
    return True, messages


def _run_benchmark_asset_guard() -> tuple[bool, list[str]]:
    if not BENCHMARK_ASSET_GUARD_SCRIPT.is_file():
        return (
            False,
            [f"Missing benchmark asset guard script: {BENCHMARK_ASSET_GUARD_SCRIPT.relative_to(ROOT)}"],
        )

    code, out, err = _run_command([sys.executable, str(BENCHMARK_ASSET_GUARD_SCRIPT)], cwd=ROOT)
    output_lines = [line for line in (out + err).splitlines() if line.strip()]
    if code != 0:
        if not output_lines:
            output_lines = [f"Benchmark asset guard failed with exit code {code}."]
        return False, output_lines
    return True, output_lines


def _run_godot(godot: str, args: Iterable[str]) -> tuple[bool, bool, str]:
    normalized_godot = _normalize_process_arg(godot)
    command = [normalized_godot]
    command.extend(_normalize_process_arg(arg) for arg in args)

    if not command[0]:
        return False, False, "ValueError: empty Godot binary path after normalization"

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return False, False, f"{type(exc).__name__}: {exc} (command={command!r})"

    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 and _tests_unavailable(output):
        return True, True, output
    return result.returncode == 0, False, output


def _parse_doctest_results(output: str) -> tuple[int, int, int, int, int, bool]:
    """Parse doctest output and return counts plus whether both summary lines were found."""
    tests_match = re.search(r"test cases:\s*\d+\s*\|\s*(\d+)\s*passed\s*\|\s*(\d+)\s*failed", output)
    asserts_match = re.search(r"assertions:\s*\d+\s*\|\s*(\d+)\s*passed\s*\|\s*(\d+)\s*failed", output)
    skip_markers = len(DOCTEST_SKIP_MARKER_RE.findall(output))

    passed_tests = int(tests_match.group(1)) if tests_match else 0
    failed_tests = int(tests_match.group(2)) if tests_match else 0
    passed_asserts = int(asserts_match.group(1)) if asserts_match else 0
    failed_asserts = int(asserts_match.group(2)) if asserts_match else 0

    return (
        passed_tests,
        failed_tests,
        passed_asserts,
        failed_asserts,
        skip_markers,
        tests_match is not None and asserts_match is not None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gaussian Splatting module tests and CI guards.")
    parser.add_argument("--godot-binary", default=os.environ.get("GODOT_BINARY", "godot"),
                        help="Path to Godot binary (default: GODOT_BINARY env or 'godot').")
    parser.add_argument("--base-ref", default=os.environ.get("GS_RENDER_GUARD_BASE"),
                        help="Git base ref/commit for render-path guard diff (default: auto-detected).")
    parser.add_argument("--guard-only", "--guards-only", action="store_true",
                        help="Run guards only and skip Godot test execution.")
    parser.add_argument("--skip-render-guards", action="store_true",
                        help="Skip render-path mutation guard checks.")
    parser.add_argument("--skip-static-guards", action="store_true",
                        help="Skip static format safety guard checks.")
    parser.add_argument("--skip-build-metadata-guard", action="store_true",
                        help="Skip SCons/CMake/doc metadata consistency guard checks.")
    parser.add_argument(
        "--tests-unavailable-mode",
        choices=("strict", "warn-only"),
        default=os.environ.get(TEST_AVAILABILITY_MODE_ENV, "").strip().lower() or None,
        help=(
            "Behavior when the binary has no test runner support "
            f"(default: {TEST_AVAILABILITY_MODE_ENV}, then {VALIDATION_MODE_ENV}, then CI-aware fallback)."
        ),
    )
    parser.add_argument(
        "--allow-tests-unavailable",
        action="store_true",
        help=(
            "Explicitly allow skipping unavailable tests (non-fatal), "
            f"equivalent to setting {ALLOW_TESTS_UNAVAILABLE_ENV}=1."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Run the renderer-dependent (requires-RD) doctest lane (opt-in). "
            "Under Godot's --test mode all tests in this lane skip because no "
            "RenderingDevice is created.  Useful as a smoke-check that the "
            "tests still compile and skip gracefully. "
            f"Equivalent to setting {GS_RUN_GPU_TESTS_ENV}=1."
        ),
    )
    return parser.parse_args()


def main() -> int:
    cli_args = _parse_args()
    godot = _normalize_process_arg(cli_args.godot_binary)
    tests_unavailable_mode = _resolve_tests_unavailable_mode(cli_args.tests_unavailable_mode)
    allow_tests_unavailable = cli_args.allow_tests_unavailable or _env_truthy(
        os.environ.get(ALLOW_TESTS_UNAVAILABLE_ENV, "")
    )
    history_guard_mode, history_guard_mode_warning = _resolve_history_artifact_guard_mode()
    if history_guard_mode_warning:
        print(f"[module-tests] {history_guard_mode_warning}")

    backup_guard_ok, backup_guard_messages = _run_tracked_backup_guard()
    for message in backup_guard_messages:
        prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
        print(f"{prefix}{message}")
    if not backup_guard_ok:
        print("[module-tests] Tracked backup-file guard failed.")
        return 1
    print("[module-tests] Tracked backup-file guard passed.")

    artifact_guard_ok, artifact_guard_messages = _run_tracked_artifact_guard()
    for message in artifact_guard_messages:
        prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
        print(f"{prefix}{message}")
    if not artifact_guard_ok:
        print("[module-tests] Tracked artifact hygiene guard failed.")
        return 1
    print("[module-tests] Tracked artifact hygiene guard passed.")

    history_guard_ok, history_guard_exit_code, history_guard_messages = _run_history_artifact_guard(
        history_guard_mode
    )
    for message in history_guard_messages:
        prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
        print(f"{prefix}{message}")
    if not history_guard_ok:
        print("[module-tests] History artifact guard failed.")
        return history_guard_exit_code

    if not cli_args.skip_build_metadata_guard:
        build_metadata_ok, build_metadata_messages = _run_build_metadata_guard()
        for message in build_metadata_messages:
            prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
            print(f"{prefix}{message}")
        if not build_metadata_ok:
            print("[module-tests] Build metadata guard failed.")
            return 1
        if not build_metadata_messages:
            print("[module-tests] Build metadata guard passed.")

    shader_dependency_ok, shader_dependency_messages = _run_shader_dependency_guard()
    for message in shader_dependency_messages:
        prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
        print(f"{prefix}{message}")
    if not shader_dependency_ok:
        print("[module-tests] Shader dependency guard failed.")
        return 1
    if not shader_dependency_messages:
        print("[module-tests] Shader dependency guard passed.")

    gaussian_layout_ok, gaussian_layout_messages = _run_gaussian_layout_guard()
    for message in gaussian_layout_messages:
        prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
        print(f"{prefix}{message}")
    if not gaussian_layout_ok:
        print("[module-tests] Gaussian layout guard failed.")
        return 1
    if not gaussian_layout_messages:
        print("[module-tests] Gaussian layout guard passed.")

    benchmark_asset_guard_ok, benchmark_asset_guard_messages = _run_benchmark_asset_guard()
    for message in benchmark_asset_guard_messages:
        prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
        print(f"{prefix}{message}")
    if not benchmark_asset_guard_ok:
        print("[module-tests] Benchmark asset path guard failed.")
        return 1
    if not benchmark_asset_guard_messages:
        print("[module-tests] Benchmark asset path guard passed.")

    base_ref: str | None = None
    if not cli_args.skip_render_guards:
        base_ref = _resolve_guard_base_ref(cli_args.base_ref)

    if not cli_args.skip_render_guards:
        guard_ok, guard_messages = _check_render_path_guards(base_ref)
        for message in guard_messages:
            prefix = "[module-tests] " if not message.startswith("[module-tests]") else ""
            print(f"{prefix}{message}")
        if not guard_ok:
            print("[module-tests] Renderer guard checks failed.")
            return 1
        print("[module-tests] Renderer guard checks passed.")

    if not cli_args.skip_static_guards:
        guards_ok, guard_failures = _run_static_format_guards()
        if not guards_ok:
            print("[module-tests] Static format safety guard(s) failed:")
            for failure in guard_failures:
                print(f"[module-tests]  - {failure}")
            return 1
        print(f"[module-tests] Static format safety guards passed ({len(STATIC_FORMAT_GUARDS)} checks).")

    if cli_args.guard_only:
        print("[module-tests] Guard-only mode complete.")
        return 0

    synthetic_assets_ok, synthetic_asset_messages = _prepare_synthetic_assets()
    for message in synthetic_asset_messages:
        print(f"[module-tests] {message}")
    if not synthetic_assets_ok:
        print("[module-tests] Synthetic asset preparation failed.")
        return 1

    print(
        f"[module-tests] Tests-unavailable mode: {tests_unavailable_mode}"
        f"{' (explicit override enabled)' if allow_tests_unavailable else ''}."
    )

    test_runs = []
    for name, test_filters, exclude_filters, strict in MODULE_TEST_FILTERS:
        run_args = _build_doctest_run_args(test_filters, exclude_filters)
        test_runs.append((name, run_args, strict))

    # Opt-in requires-RD lane: activated by GS_RUN_GPU_TESTS=1 env var or --gpu flag.
    # Under --test mode every test here will skip (no RenderingDevice).  This is
    # expected — the lane validates that tagged tests compile and skip gracefully.
    run_gpu = os.environ.get(GS_RUN_GPU_TESTS_ENV, "0") == "1" or cli_args.gpu
    if run_gpu:
        for name, test_filters, exclude_filters, strict in REQUIRES_RD_TEST_FILTERS:
            run_args = _build_doctest_run_args(test_filters, exclude_filters)
            test_runs.append((name, run_args, strict))

    for name, run_args, strict in test_runs:
        ok, skipped, output = _run_godot(godot, run_args)
        if skipped:
            if tests_unavailable_mode == "strict" and not allow_tests_unavailable:
                print(
                    f"[module-tests] '{name}' unavailable: binary does not support --test "
                    "(build with tests=yes)."
                )
                if output.strip():
                    print(output.strip())
                print(
                    f"[module-tests] Failing because strict mode is active. "
                    f"Use --allow-tests-unavailable or {ALLOW_TESTS_UNAVAILABLE_ENV}=1 "
                    "for an explicit local opt-out."
                )
                return 1

            print(f"[module-tests] Skipping '{name}' (tests not enabled in binary).")
            if output.strip():
                print(output.strip())
            return 0

        if not ok:
            if not strict:
                print(
                    f"[module-tests] '{name}' crashed or failed "
                    "(advisory lane, continuing)."
                )
                if output.strip():
                    print(output.strip())
                continue
            print(f"[module-tests] '{name}' failed.")
            if output.strip():
                print(output.strip())
            return 1

        (
            passed_tests,
            failed_tests,
            passed_asserts,
            failed_asserts,
            skipped_markers,
            summary_found,
        ) = _parse_doctest_results(output)
        if not summary_found:
            print(f"[module-tests] '{name}' failed: missing doctest summary in output.")
            if output.strip():
                print(output.strip())
            return 1

        if failed_tests > 0 or failed_asserts > 0:
            print(
                f"[module-tests] '{name}' failed: {failed_tests} failed test(s), "
                f"{failed_asserts} failed assertion(s)."
            )
            if output.strip():
                print(output.strip())
            return 1

        if skipped_markers > 0:
            print(f"[module-tests] '{name}' reported {skipped_markers} skipped test marker(s) in doctest output.")
            if strict and _is_ci():
                print(
                    f"[module-tests] '{name}' failed: skipped doctest coverage is not allowed in CI."
                )
                if output.strip():
                    print(output.strip())
                return 1

        if passed_tests <= 0 or passed_asserts <= 0:
            # Keep the canonical headless Gaussian lanes strict. Secondary lanes
            # remain advisory and may not match on every platform/parser variant.
            if not strict:
                print(
                    f"[module-tests] '{name}' has no executed coverage "
                    f"(passed_tests={passed_tests}, passed_assertions={passed_asserts}); "
                    "treating this lane as advisory and continuing."
                )
                if output.strip():
                    print(output.strip())
                continue

            print(
                f"[module-tests] '{name}' failed: no executed coverage "
                f"(passed_tests={passed_tests}, passed_assertions={passed_asserts})."
            )
            if output.strip():
                print(output.strip())
            return 1

        print(
            f"[module-tests] '{name}' passed: {passed_tests} test(s), "
            f"{passed_asserts} assertion(s)."
        )

    print("[module-tests] Gaussian splatting module tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
