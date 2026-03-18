#!/usr/bin/env python3
"""
Run an A/B benchmark for instance-pipeline orchestration modes on the same scene.

A run consists of two passes:
- serial
- single_pass

Each pass executes the same benchmark lane scene with identical duration/asset
configuration and writes:
- per-mode JSON report from the lane script
- per-mode log (stdout/stderr + command)
- aggregate A/B report with deltas
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES: tuple[str, str] = ("serial", "single_pass")
DEFAULT_SCENE = "res://scenes/benchmark_suite/lane_instance_pipeline_ab.tscn"
DEFAULT_DURATION = 35.0
DEFAULT_WARMUP = 5.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_godot_binary(repo_root: Path) -> Path:
    return repo_root / "bin" / "godot.linuxbsd.editor.dev.x86_64"


def _default_project_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "examples" / "godot" / "test_project"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Run serial vs single_pass A/B lane benchmark.")
    parser.add_argument(
        "--godot-binary",
        default=os.environ.get("GODOT_BINARY", str(_default_godot_binary(repo_root))),
        help="Path to Godot executable.",
    )
    parser.add_argument(
        "--project-path",
        default=str(_default_project_path(repo_root)),
        help="Godot test project path.",
    )
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="Benchmark lane scene path (res://...).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Benchmark duration in seconds (default: 35).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=DEFAULT_WARMUP,
        help="Warmup duration in seconds (default: 5).",
    )
    parser.add_argument(
        "--asset",
        default="",
        help="Optional benchmark asset override path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "tests" / "output" / "instance_pipeline_ab" / _now_stamp()),
        help="Directory for mode logs/reports and A/B summary.",
    )
    parser.add_argument(
        "--timeout-scale",
        type=float,
        default=3.0,
        help="Timeout multiplier over duration (default: 3.0).",
    )
    parser.add_argument(
        "--max-avg-fps-regression-pct",
        type=float,
        default=None,
        help="Fail if single_pass avg_fps regresses vs serial by more than this %% (e.g. 5).",
    )
    parser.add_argument(
        "--max-p1-fps-regression-pct",
        type=float,
        default=None,
        help="Fail if single_pass p1_fps regresses vs serial by more than this %% (e.g. 5).",
    )
    parser.add_argument(
        "--max-p99-frame-ms-regression-pct",
        type=float,
        default=None,
        help="Fail if single_pass p99_frame_ms regresses vs serial by more than this %% (e.g. 5).",
    )
    parser.add_argument(
        "--require-gpu-timestamps",
        action="store_true",
        help="Fail the A/B run when GPU timestamp timing is unavailable in either mode.",
    )
    return parser.parse_args()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def _normalize_execution_mode_token(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("single_pass"):
        return "single_pass"
    if raw.startswith("serial"):
        return "serial"
    if raw == "auto":
        return "auto"
    return ""


def _extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    overall = report.get("overall", {})
    steady = report.get("steady_overall", {})

    summary_source = "overall"
    summary = overall if isinstance(overall, dict) else {}
    score = report.get("score")
    steady_count = 0
    if isinstance(steady, dict):
        steady_count = int(steady.get("sample_count", 0))
    if steady_count > 0:
        summary_source = "steady_overall"
        summary = steady
        if isinstance(report.get("steady_score"), (int, float)):
            score = report.get("steady_score")

    execution_mode = report.get("instancing_execution_mode", "")
    execution_path = report.get("instancing_execution_path", "")
    execution_reason = report.get("instancing_execution_reason", "")
    if not execution_mode and isinstance(summary, dict):
        execution_mode = summary.get("instance_pipeline_execution_mode", "")
    if not execution_path and isinstance(summary, dict):
        execution_path = summary.get("instance_pipeline_execution_path", "")
    if not execution_reason and isinstance(summary, dict):
        execution_reason = summary.get("instance_pipeline_execution_reason", "")

    return {
        "score": _safe_float(score),
        "avg_fps": _safe_float(summary.get("avg_fps")),
        "p1_fps": _safe_float(summary.get("p1_fps")),
        "p99_frame_ms": _safe_float(summary.get("p99_frame_ms")),
        "sample_count": int(summary.get("sample_count", 0)) if isinstance(summary, dict) else 0,
        "summary_source": summary_source,
        "instancing_mode_reported": report.get("instancing_mode", ""),
        "execution_mode": execution_mode,
        "execution_mode_normalized": _normalize_execution_mode_token(execution_mode),
        "execution_path": execution_path,
        "execution_reason": execution_reason,
        "gpu_timing_available": report.get("overall", {}).get("gpu_timing_available", None),
        "gpu_frame_time_source": report.get("overall", {}).get("gpu_time_frame_source", None),
        "true_single_pass_enabled": report.get("project_settings", {}).get(
            "rendering/gaussian_splatting/instance_pipeline/true_single_pass_enabled", None
        ),
    }


def _check_regressions(
    deltas: dict[str, dict[str, float | None]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Return a list of regression violations (empty = pass)."""
    checks: list[tuple[str, float | None, bool]] = [
        # (metric, threshold, higher_is_worse)
        ("avg_fps", args.max_avg_fps_regression_pct, False),
        ("p1_fps", args.max_p1_fps_regression_pct, False),
        ("p99_frame_ms", args.max_p99_frame_ms_regression_pct, True),
    ]
    violations: list[dict[str, Any]] = []
    for metric, threshold, higher_is_worse in checks:
        if threshold is None:
            continue
        pct = deltas.get(metric, {}).get("pct_vs_serial")
        if pct is None:
            violations.append({"metric": metric, "threshold_pct": threshold, "actual_pct": None, "reason": "missing"})
            continue
        # For FPS: regression = negative delta (single_pass < serial) → regression_pct = -pct
        # For latency: regression = positive delta (single_pass > serial) → regression_pct = pct
        regression_pct = pct if higher_is_worse else -pct
        if regression_pct > threshold:
            violations.append({
                "metric": metric,
                "threshold_pct": threshold,
                "actual_pct": round(pct, 4),
                "regression_pct": round(regression_pct, 4),
                "reason": "exceeded",
            })
    return violations


def _compute_delta(serial_value: float | None, single_pass_value: float | None) -> dict[str, float | None]:
    if serial_value is None or single_pass_value is None:
        return {"abs": None, "pct_vs_serial": None}
    delta_abs = single_pass_value - serial_value
    delta_pct = None
    if abs(serial_value) > 1e-9:
        delta_pct = (delta_abs / serial_value) * 100.0
    return {"abs": delta_abs, "pct_vs_serial": delta_pct}


def _run_mode(
    *,
    mode: str,
    godot_binary: Path,
    project_path: Path,
    scene: str,
    duration: float,
    warmup: float,
    output_dir: Path,
    asset: str,
    timeout_scale: float,
) -> dict[str, Any]:
    mode_json = output_dir / f"{mode}.json"
    mode_log = output_dir / f"{mode}.log"
    if mode_json.exists():
        mode_json.unlink()

    timeout_seconds = max(90, int(math.ceil(duration * timeout_scale)) + 45)
    cmd = [
        str(godot_binary),
        "--disable-vsync",
        "--path",
        str(project_path),
        "--scene",
        scene,
        "--benchmark-duration",
        f"{duration:.3f}",
        "--benchmark-warmup",
        f"{warmup:.3f}",
        "--benchmark-output",
        str(mode_json),
        "--benchmark-lane-tag",
        f"instance_pipeline_ab:{mode}",
        "--benchmark-headless-summary",
        f"--benchmark-instancing-mode={mode}",
    ]
    if asset:
        cmd.append(f"--benchmark-asset={asset}")

    timed_out = False
    timeout_message = ""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_message = f"Timed out after {timeout_seconds}s (duration={duration:.1f}s)."
        timeout_stdout = _coerce_text(exc.stdout)
        timeout_stderr = _coerce_text(exc.stderr)
        if timeout_message:
            timeout_stderr = f"{timeout_stderr}\n{timeout_message}" if timeout_stderr else timeout_message
        completed = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=timeout_stdout,
            stderr=timeout_stderr,
        )

    with mode_log.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.write(f"# timeout_seconds={timeout_seconds}\n")
        if timed_out:
            fh.write(f"# TIMEOUT: {timeout_message}\n")
        if completed.stdout:
            fh.write("STDOUT:\n")
            fh.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                fh.write("\n")
        if completed.stderr:
            fh.write("\nSTDERR:\n")
            fh.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                fh.write("\n")

    report: dict[str, Any] | None = None
    if mode_json.exists():
        try:
            loaded = json.loads(mode_json.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                report = loaded
        except json.JSONDecodeError:
            report = None

    metrics = _extract_metrics(report) if isinstance(report, dict) else {}
    mode_match = None
    if isinstance(metrics, dict):
        exec_mode = str(metrics.get("execution_mode_normalized", "")).strip()
        if exec_mode in MODES:
            mode_match = exec_mode == mode
    return {
        "mode": mode,
        "command": cmd,
        "exit_code": int(completed.returncode),
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "json_path": str(mode_json),
        "log_path": str(mode_log),
        "report_valid": isinstance(report, dict),
        "report": report,
        "metrics": metrics,
        "mode_match": mode_match,
    }


def _write_summary_markdown(path: Path, report: dict[str, Any]) -> None:
    serial = report["runs"].get("serial", {})
    single_pass = report["runs"].get("single_pass", {})
    deltas = report.get("deltas", {})

    def _fmt(value: Any, decimals: int = 2) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{decimals}f}"
        return "n/a"

    lines = [
        "# Instance Pipeline A/B Summary",
        "",
        f"- Scene: `{report.get('scene', '')}`",
        f"- Duration: `{report.get('duration_s', 0):.1f}s` (warmup `{report.get('warmup_s', 0):.1f}s`)",
        "",
        "| Mode | Score | Avg FPS | P1 FPS | P99 ms | Samples | Exec Mode | Exec Path | Exec Reason | GPU Timing | GPU Source | Exit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
        "| serial | "
        f"{_fmt(serial.get('metrics', {}).get('score'))} | "
        f"{_fmt(serial.get('metrics', {}).get('avg_fps'))} | "
        f"{_fmt(serial.get('metrics', {}).get('p1_fps'))} | "
        f"{_fmt(serial.get('metrics', {}).get('p99_frame_ms'))} | "
        f"{serial.get('metrics', {}).get('sample_count', 'n/a')} | "
        f"{serial.get('metrics', {}).get('execution_mode', 'n/a')} | "
        f"{serial.get('metrics', {}).get('execution_path', 'n/a')} | "
        f"{serial.get('metrics', {}).get('execution_reason', 'n/a')} | "
        f"{serial.get('metrics', {}).get('gpu_timing_available', 'n/a')} | "
        f"{serial.get('metrics', {}).get('gpu_frame_time_source', 'n/a')} | "
        f"{serial.get('exit_code', 'n/a')} |",
        "| single_pass | "
        f"{_fmt(single_pass.get('metrics', {}).get('score'))} | "
        f"{_fmt(single_pass.get('metrics', {}).get('avg_fps'))} | "
        f"{_fmt(single_pass.get('metrics', {}).get('p1_fps'))} | "
        f"{_fmt(single_pass.get('metrics', {}).get('p99_frame_ms'))} | "
        f"{single_pass.get('metrics', {}).get('sample_count', 'n/a')} | "
        f"{single_pass.get('metrics', {}).get('execution_mode', 'n/a')} | "
        f"{single_pass.get('metrics', {}).get('execution_path', 'n/a')} | "
        f"{single_pass.get('metrics', {}).get('execution_reason', 'n/a')} | "
        f"{single_pass.get('metrics', {}).get('gpu_timing_available', 'n/a')} | "
        f"{single_pass.get('metrics', {}).get('gpu_frame_time_source', 'n/a')} | "
        f"{single_pass.get('exit_code', 'n/a')} |",
        "",
        "## Delta (single_pass - serial)",
        "",
        f"- Score: `{_fmt(deltas.get('score', {}).get('abs'))}` ({_fmt(deltas.get('score', {}).get('pct_vs_serial'))}% vs serial)",
        f"- Avg FPS: `{_fmt(deltas.get('avg_fps', {}).get('abs'))}` ({_fmt(deltas.get('avg_fps', {}).get('pct_vs_serial'))}% vs serial)",
        f"- P1 FPS: `{_fmt(deltas.get('p1_fps', {}).get('abs'))}` ({_fmt(deltas.get('p1_fps', {}).get('pct_vs_serial'))}% vs serial)",
        f"- P99 frame ms: `{_fmt(deltas.get('p99_frame_ms', {}).get('abs'))}` ({_fmt(deltas.get('p99_frame_ms', {}).get('pct_vs_serial'))}% vs serial)",
        "",
    ]
    regressions = report.get("regression_violations", [])
    if regressions:
        lines.extend([
            "## Regression Violations",
            "",
        ])
        for v in regressions:
            lines.append(
                f"- **{v['metric']}**: {v['reason']} "
                f"(threshold={v['threshold_pct']}%, actual={v.get('regression_pct', v.get('actual_pct'))}%)"
            )
        lines.append("")
    elif report.get("regression_thresholds") and any(
        v is not None for v in report["regression_thresholds"].values()
    ):
        lines.extend(["## Regression Check", "", "All metrics within thresholds.", ""])

    serial_mode_match = serial.get("mode_match", None)
    single_mode_match = single_pass.get("mode_match", None)
    if serial_mode_match is not True or single_mode_match is not True:
        def _mode_status(match: Any, run: dict[str, Any]) -> str:
            exec_mode = run.get("metrics", {}).get("execution_mode", "unknown")
            if match is True:
                return f"ok ({exec_mode})"
            if match is False:
                return f"mismatch ({exec_mode})"
            return f"unknown ({exec_mode})"

        lines.extend(
            [
                "## Mode-Route Mismatch",
                "",
                f"- serial requested: `{_mode_status(serial_mode_match, serial)}`",
                f"- single_pass requested: `{_mode_status(single_mode_match, single_pass)}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    godot_binary = Path(args.godot_binary).resolve()
    project_path = Path(args.project_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not godot_binary.exists():
        print(f"ERROR: Godot binary not found: {godot_binary}", file=sys.stderr)
        return 2
    if not project_path.exists():
        print(f"ERROR: Project path not found: {project_path}", file=sys.stderr)
        return 2
    if args.duration <= 0.0:
        print("ERROR: --duration must be > 0", file=sys.stderr)
        return 2
    if args.warmup < 0.0:
        print("ERROR: --warmup must be >= 0", file=sys.stderr)
        return 2
    if args.timeout_scale <= 0.0:
        print("ERROR: --timeout-scale must be > 0", file=sys.stderr)
        return 2

    runs: dict[str, Any] = {}
    failed = False
    for mode in MODES:
        print(f"[ab] running mode={mode} scene={args.scene} duration={args.duration:.1f}s")
        run_result = _run_mode(
            mode=mode,
            godot_binary=godot_binary,
            project_path=project_path,
            scene=args.scene,
            duration=float(args.duration),
            warmup=float(args.warmup),
            output_dir=output_dir,
            asset=args.asset,
            timeout_scale=float(args.timeout_scale),
        )
        runs[mode] = run_result
        metrics = run_result.get("metrics", {})
        print(
            f"[ab] mode={mode} exit={run_result['exit_code']} "
            f"score={metrics.get('score', 'n/a')} avg_fps={metrics.get('avg_fps', 'n/a')} "
            f"p1_fps={metrics.get('p1_fps', 'n/a')} p99_ms={metrics.get('p99_frame_ms', 'n/a')} "
            f"exec_mode={metrics.get('execution_mode', 'n/a')} "
            f"exec_mode_norm={metrics.get('execution_mode_normalized', 'n/a')} "
            f"exec_path={metrics.get('execution_path', 'n/a')} "
            f"exec_reason={metrics.get('execution_reason', 'n/a')} "
            f"gpu_timing={metrics.get('gpu_timing_available', 'n/a')}"
        )
        if int(run_result["exit_code"]) != 0 or not bool(run_result.get("report_valid")):
            failed = True
        mode_match = run_result.get("mode_match")
        if mode_match is not True:
            failed = True
            expected_mode = run_result.get("mode", mode)
            actual_mode = metrics.get("execution_mode", "unknown")
            actual_path = metrics.get("execution_path", "unknown")
            status = "unknown" if mode_match is None else "mismatch"
            print(
                f"[ab] MODE_VALIDATION_FAIL: requested={expected_mode} actual={actual_mode} "
                f"path={actual_path} status={status}"
            )
        if args.require_gpu_timestamps:
            if not bool(metrics.get("gpu_timing_available")):
                failed = True
                print(
                    f"[ab] GPU_TIMING_VALIDATION_FAIL: mode={mode} "
                    f"source={metrics.get('gpu_frame_time_source', 'unknown')}"
                )

    serial_metrics = runs.get("serial", {}).get("metrics", {})
    single_metrics = runs.get("single_pass", {}).get("metrics", {})
    deltas = {
        "score": _compute_delta(_safe_float(serial_metrics.get("score")), _safe_float(single_metrics.get("score"))),
        "avg_fps": _compute_delta(_safe_float(serial_metrics.get("avg_fps")), _safe_float(single_metrics.get("avg_fps"))),
        "p1_fps": _compute_delta(_safe_float(serial_metrics.get("p1_fps")), _safe_float(single_metrics.get("p1_fps"))),
        "p99_frame_ms": _compute_delta(
            _safe_float(serial_metrics.get("p99_frame_ms")),
            _safe_float(single_metrics.get("p99_frame_ms")),
        ),
    }

    regressions = _check_regressions(deltas, args)
    if regressions:
        failed = True
        for v in regressions:
            print(f"[ab] REGRESSION: {v['metric']} {v['reason']} "
                  f"(threshold={v['threshold_pct']}%, actual={v.get('regression_pct', v.get('actual_pct'))}%)")

    report = {
        "name": "Instance Pipeline A/B Benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "godot_binary": str(godot_binary),
        "project_path": str(project_path),
        "scene": args.scene,
        "duration_s": float(args.duration),
        "warmup_s": float(args.warmup),
        "asset_override": args.asset,
        "output_dir": str(output_dir),
        "runs": runs,
        "deltas": deltas,
        "regression_thresholds": {
            "max_avg_fps_regression_pct": args.max_avg_fps_regression_pct,
            "max_p1_fps_regression_pct": args.max_p1_fps_regression_pct,
            "max_p99_frame_ms_regression_pct": args.max_p99_frame_ms_regression_pct,
        },
        "mode_validation": {
            "serial_mode_match": runs.get("serial", {}).get("mode_match"),
            "single_pass_mode_match": runs.get("single_pass", {}).get("mode_match"),
        },
        "require_gpu_timestamps": bool(args.require_gpu_timestamps),
        "regression_violations": regressions,
    }

    report_path = output_dir / "instance_pipeline_ab_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output_dir / "instance_pipeline_ab_summary.md"
    _write_summary_markdown(summary_path, report)

    print(f"[ab] report={report_path}")
    print(f"[ab] summary={summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
