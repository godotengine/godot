#!/usr/bin/env python3
"""Orchestrate documentation generation tasks for Godot Gaussian Splatting."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def run(cmd: list[str], description: str) -> int:
    print(f"[docs] {description}: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, cwd=ROOT)
    except FileNotFoundError:
        print(f"[docs] Command not found: {cmd[0]}")
        return 1


def run_doxygen(config: Path) -> int:
    if shutil.which("doxygen") is None:
        print("[docs] Skipping Doxygen generation because 'doxygen' is not installed.")
        return 0
    return run(["doxygen", str(config)], "Running Doxygen")


def run_python(script: Path, *args: str) -> int:
    return run([sys.executable, str(script), *args], f"Running {script.name}")


_BENCHMARK_REPORT_SEARCH_ROOTS = [
    ROOT / "tests" / "output" / "benchmark_suite",
    ROOT / "tests" / "output" / "benchmark_evidence",
]
_BENCHMARK_REPORT_PATHS = [
    DOCS / "assets" / "data" / "benchmark_suite_report.json",
]


def _find_benchmark_report() -> Path | None:
    """Return the first existing benchmark report (fresh local data preferred over committed baseline)."""
    discovered_reports: list[Path] = []
    for root in _BENCHMARK_REPORT_SEARCH_ROOTS:
        if root.is_dir():
            discovered_reports.extend(root.rglob("benchmark_suite_report.json"))
    if discovered_reports:
        return max(discovered_reports, key=lambda path: path.stat().st_mtime)
    for candidate in _BENCHMARK_REPORT_PATHS:
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build documentation artifacts.")
    parser.add_argument("--doxygen", action="store_true", help="Generate C++ API docs via Doxygen.")
    parser.add_argument("--gdscript", action="store_true", help="Extract GDScript documentation.")
    parser.add_argument(
        "--gdscript-scope",
        choices=("public", "all"),
        default="public",
        help="Scope for GDScript extraction (`public` excludes test/internal scripts).",
    )
    parser.add_argument("--shaders", action="store_true", help="Generate shader documentation.")
    parser.add_argument(
        "--shader-include-undocumented",
        action="store_true",
        help="Include undocumented shader entries in generated reference output.",
    )
    parser.add_argument(
        "--shader-strict",
        action="store_true",
        help="Fail shader docs generation when undocumented coverage exceeds configured thresholds.",
    )
    parser.add_argument(
        "--shader-max-undocumented-functions",
        type=int,
        default=0,
        help="Allowed undocumented shader functions when --shader-strict is enabled.",
    )
    parser.add_argument(
        "--shader-max-undocumented-fields",
        type=int,
        default=0,
        help="Allowed undocumented shader uniform fields when --shader-strict is enabled.",
    )
    parser.add_argument("--engine-patch", action="store_true", help="Generate upstream engine patch report.")
    parser.add_argument(
        "--no-engine-patch",
        action="store_true",
        help="Skip engine patch generation (useful with --all for CI step isolation).",
    )
    parser.add_argument(
        "--engine-patch-config",
        default=None,
        help="Override engine patch YAML config path.",
    )
    parser.add_argument(
        "--engine-patch-upstream-ref",
        default=None,
        help="Override pinned upstream ref for engine patch generation.",
    )
    parser.add_argument(
        "--engine-patch-head-ref",
        default=None,
        help="Override head ref for engine patch generation.",
    )
    parser.add_argument(
        "--engine-patch-summary-only",
        action="store_true",
        help="Generate summary-only markdown for engine patch output.",
    )
    parser.add_argument(
        "--engine-patch-strict",
        action="store_true",
        help="Fail docs build when engine patch generation errors.",
    )
    parser.add_argument("--dashboard", action="store_true", help="Generate SVG benchmark dashboard charts.")
    parser.add_argument("--compatibility", action="store_true", help="Update compatibility matrix.")
    parser.add_argument("--project-settings", action="store_true", help="Regenerate project settings reference.")
    parser.add_argument("--benchmarks", action="store_true", help="Export benchmark data for Vega-Lite charts.")
    parser.add_argument("--report", action="store_true", help="Refresh documentation coverage reports.")
    parser.add_argument("--all", action="store_true", help="Run all documentation tasks.")
    args = parser.parse_args()

    if args.all:
        args.doxygen = args.gdscript = args.shaders = True
        args.benchmarks = args.dashboard = True
        args.compatibility = args.project_settings = args.report = True
        if not args.no_engine_patch:
            args.engine_patch = True

    exit_code = 0

    if args.doxygen:
        config = DOCS / "Doxyfile"
        exit_code |= run_doxygen(config)

    if args.gdscript:
        exit_code |= run_python(
            ROOT / "scripts" / "extract_gdscript_docs.py",
            "--scope",
            args.gdscript_scope,
        )

    if args.shaders:
        shader_args: list[str] = []
        if args.shader_include_undocumented:
            shader_args.append("--include-undocumented")
        if args.shader_strict:
            shader_args.extend(
                [
                    "--strict",
                    "--max-undocumented-functions",
                    str(args.shader_max_undocumented_functions),
                    "--max-undocumented-fields",
                    str(args.shader_max_undocumented_fields),
                ]
            )
        exit_code |= run_python(ROOT / "scripts" / "generate_shader_docs.py", *shader_args)

    if args.engine_patch and not args.no_engine_patch:
        engine_patch_args: list[str] = []
        if args.engine_patch_config:
            engine_patch_args.extend(["--config", args.engine_patch_config])
        if args.engine_patch_upstream_ref:
            engine_patch_args.extend(["--upstream-ref", args.engine_patch_upstream_ref])
        if args.engine_patch_head_ref:
            engine_patch_args.extend(["--head-ref", args.engine_patch_head_ref])
        if args.engine_patch_summary_only:
            engine_patch_args.append("--summary-only")
        if args.engine_patch_strict:
            engine_patch_args.append("--strict")

        engine_patch_exit = run_python(
            ROOT / "scripts" / "generate_engine_patch_report.py",
            *engine_patch_args,
        )
        if engine_patch_exit != 0:
            if args.engine_patch_strict:
                exit_code |= engine_patch_exit
            else:
                print(
                    "[docs] WARNING: engine patch generation failed in non-strict mode; continuing docs build."
                )

    if args.benchmarks:
        exit_code |= run_python(ROOT / "scripts" / "export_benchmark_vegalite.py")

    if args.dashboard:
        report_path = _find_benchmark_report()
        if report_path:
            exit_code |= run_python(
                ROOT / "scripts" / "generate_benchmark_suite_dashboard.py",
                "--suite-report", str(report_path),
                "--output-dir", str(DOCS / "assets" / "benchmarks"),
            )
        else:
            print("[docs] No benchmark report found; skipping dashboard generation.")

    if args.compatibility:
        exit_code |= run_python(ROOT / "scripts" / "update_compatibility_matrix.py")

    if args.project_settings:
        exit_code |= run_python(ROOT / "scripts" / "generate_project_settings_reference.py")

    if args.report:
        exit_code |= run_python(ROOT / "scripts" / "documentation_audit.py")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
