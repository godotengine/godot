#!/usr/bin/env python3
"""Generate a deterministic upstream-engine patch report for this fork."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "docs" / "reference" / "engine_patch_sources.yaml"
DEFAULT_OUTPUT_MD = ROOT / "docs" / "reference" / "engine-patch.md"
DEFAULT_OUTPUT_JSON = ROOT / "docs" / "reference" / "engine-patch.json"
CACHE_REF_PREFIX = "refs/engine_patch_cache"
RENAME_THRESHOLD = "70%"
COMMIT_MARKER = "__COMMIT__"
ENGINE_SOURCE_ROOT_PREFIXES = (
    "core/",
    "drivers/",
    "editor/",
    "main/",
    "modules/",
    "platform/",
    "scene/",
    "servers/",
)
ENGINE_SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".inl",
    ".m",
    ".mm",
    ".glsl",
}
HUNK_PATTERN = re.compile(
    r"^@@\\s+-(?P<old_start>\\d+)(?:,(?P<old_count>\\d+))?\\s+\\+(?P<new_start>\\d+)(?:,(?P<new_count>\\d+))?\\s+@@\\s*(?P<context>.*)$"
)


class ReportError(RuntimeError):
    """Recoverable error for report generation."""


@dataclass(frozen=True)
class DiffEntry:
    status: str
    old_path: str
    new_path: str
    subsystem: str
    lines_added: int
    lines_deleted: int
    touchpoints: list[dict[str, Any]]


@dataclass(frozen=True)
class Config:
    upstream_repo: str
    upstream_ref: str
    label: str | None = None


def _print(msg: str) -> None:
    print(f"[engine-patch] {msg}")


def _warn(msg: str, warnings: list[str]) -> None:
    warnings.append(msg)
    _print(f"WARN: {msg}")


def _run_git(
    args: list[str],
    *,
    check: bool,
    timeout_seconds: int,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=capture_output,
        check=check,
        timeout=timeout_seconds,
    )


def _resolve_commit(ref: str, timeout_seconds: int) -> str | None:
    result = _run_git(
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        check=False,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        return None
    resolved = result.stdout.strip()
    return resolved or None


def _resolve_merge_base(left_ref: str, right_ref: str, timeout_seconds: int) -> str | None:
    result = _run_git(
        ["merge-base", left_ref, right_ref],
        check=False,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        return None
    resolved = result.stdout.strip()
    return resolved or None


def _sanitize_ref_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


def _read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        data[key] = value
    return data


def _load_config(path: Path) -> Config:
    if not path.exists():
        raise ReportError(f"Config file not found: {path}")

    data: dict[str, Any]
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ReportError(f"Config must be a mapping: {path}")
        data = loaded
    except ModuleNotFoundError:
        data = _read_simple_yaml(path)
    except Exception as exc:  # noqa: BLE001
        raise ReportError(f"Failed to parse config {path}: {exc}") from exc

    upstream_repo = str(data.get("upstream_repo", "")).strip()
    upstream_ref = str(data.get("upstream_ref", "")).strip()
    label = str(data.get("label", "")).strip() or None

    if not upstream_repo or not upstream_ref:
        raise ReportError(
            f"Config requires 'upstream_repo' and 'upstream_ref': {path}"
        )

    return Config(upstream_repo=upstream_repo, upstream_ref=upstream_ref, label=label)


def _ensure_upstream_commit(
    upstream_repo: str,
    upstream_ref: str,
    *,
    timeout_seconds: int,
) -> tuple[str, str]:
    resolved = _resolve_commit(upstream_ref, timeout_seconds)
    if resolved:
        return upstream_ref, resolved

    repo_key = _sanitize_ref_component(upstream_repo)
    ref_key = _sanitize_ref_component(upstream_ref)
    cache_ref = f"{CACHE_REF_PREFIX}/{repo_key}/{ref_key}"

    cached = _resolve_commit(cache_ref, timeout_seconds)
    if cached:
        _print(f"Using cached upstream commit from {cache_ref} ({cached[:12]})")
        return cache_ref, cached

    _print(
        "Fetching upstream ref with shallow history: "
        f"repo={upstream_repo} ref={upstream_ref}"
    )
    try:
        fetch = _run_git(
            ["fetch", "--depth=1", "--no-tags", upstream_repo, upstream_ref],
            check=False,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReportError(
            f"Timed out fetching upstream ref '{upstream_ref}' from {upstream_repo}"
        ) from exc

    if fetch.returncode != 0:
        stderr = (fetch.stderr or "").strip()
        raise ReportError(
            f"Failed to fetch upstream ref '{upstream_ref}' from {upstream_repo}: {stderr or 'unknown error'}"
        )

    fetched = _resolve_commit("FETCH_HEAD", timeout_seconds)
    if not fetched:
        raise ReportError(
            f"Fetched upstream ref '{upstream_ref}' but could not resolve FETCH_HEAD commit"
        )

    _run_git(
        ["update-ref", cache_ref, fetched],
        check=True,
        timeout_seconds=timeout_seconds,
    )
    _print(f"Cached upstream commit at {cache_ref} ({fetched[:12]})")
    return cache_ref, fetched


def _subsystem_for_path(path: str) -> str:
    parts = path.split("/")
    if not parts:
        return path
    if parts[0] in {"modules", "platform"} and len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _path_exists_in_tree(commit: str, path: str, timeout_seconds: int) -> bool:
    result = _run_git(
        ["cat-file", "-e", f"{commit}:{path}"],
        check=False,
        timeout_seconds=timeout_seconds,
    )
    return result.returncode == 0


def _parse_numstat(
    upstream_commit: str,
    head_ref: str,
    timeout_seconds: int,
) -> tuple[dict[str, tuple[int, int]], dict[tuple[str, str], tuple[int, int]]]:
    output = _run_git(
        [
            "diff",
            "--numstat",
            f"--find-renames={RENAME_THRESHOLD}",
            f"{upstream_commit}...{head_ref}",
        ],
        check=True,
        timeout_seconds=timeout_seconds,
    ).stdout

    single_path_stats: dict[str, tuple[int, int]] = {}
    rename_stats: dict[tuple[str, str], tuple[int, int]] = {}

    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue

        added_raw, deleted_raw = parts[0], parts[1]
        added = 0 if added_raw == "-" else int(added_raw)
        deleted = 0 if deleted_raw == "-" else int(deleted_raw)

        if len(parts) == 3:
            single_path_stats[parts[2]] = (added, deleted)
        else:
            old_path = parts[2]
            new_path = parts[3]
            rename_stats[(old_path, new_path)] = (added, deleted)

    return single_path_stats, rename_stats


def _parse_hunk_map(
    upstream_commit: str,
    head_ref: str,
    timeout_seconds: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    diff_text = _run_git(
        [
            "diff",
            "--no-color",
            f"--find-renames={RENAME_THRESHOLD}",
            "-U0",
            f"{upstream_commit}...{head_ref}",
        ],
        check=True,
        timeout_seconds=timeout_seconds,
    ).stdout

    hunk_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    current_old: str | None = None
    current_new: str | None = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("diff --git "):
            tokens = raw_line.split()
            if len(tokens) >= 4:
                old_token = tokens[2]
                new_token = tokens[3]
                current_old = old_token[2:] if old_token.startswith("a/") else old_token
                current_new = new_token[2:] if new_token.startswith("b/") else new_token
            else:
                current_old = None
                current_new = None
            continue

        if current_old is None or current_new is None:
            continue

        match = HUNK_PATTERN.match(raw_line)
        if not match:
            continue

        old_start = int(match.group("old_start"))
        old_count = int(match.group("old_count") or "1")
        new_start = int(match.group("new_start"))
        new_count = int(match.group("new_count") or "1")
        context = match.group("context").strip()

        key = (current_old, current_new)
        hunk_map.setdefault(key, []).append(
            {
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "context": context,
            }
        )

    return hunk_map


def _collect_entries(
    upstream_commit: str,
    head_ref: str,
    *,
    timeout_seconds: int,
    warnings: list[str],
) -> list[DiffEntry]:
    single_path_stats, rename_stats = _parse_numstat(upstream_commit, head_ref, timeout_seconds)
    hunk_map = _parse_hunk_map(upstream_commit, head_ref, timeout_seconds)

    name_status = _run_git(
        [
            "diff",
            "--name-status",
            f"--find-renames={RENAME_THRESHOLD}",
            f"{upstream_commit}...{head_ref}",
        ],
        check=True,
        timeout_seconds=timeout_seconds,
    ).stdout

    entries: list[DiffEntry] = []
    for line in name_status.splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        status_token = parts[0]
        status_letter = status_token[0]

        if status_letter not in {"M", "D", "R"}:
            continue

        if status_letter == "R":
            if len(parts) < 3:
                continue
            old_path = parts[1]
            new_path = parts[2]
            if not _path_exists_in_tree(upstream_commit, old_path, timeout_seconds):
                continue
            lines_added, lines_deleted = rename_stats.get((old_path, new_path), (0, 0))
            touchpoints = hunk_map.get((old_path, new_path), [])
            subsystem = _subsystem_for_path(old_path)
            entries.append(
                DiffEntry(
                    status="R",
                    old_path=old_path,
                    new_path=new_path,
                    subsystem=subsystem,
                    lines_added=lines_added,
                    lines_deleted=lines_deleted,
                    touchpoints=touchpoints,
                )
            )
            continue

        if len(parts) < 2:
            continue
        path = parts[1]

        if not _path_exists_in_tree(upstream_commit, path, timeout_seconds):
            continue

        lines_added, lines_deleted = single_path_stats.get(path, (0, 0))
        touchpoints = hunk_map.get((path, path), [])
        subsystem = _subsystem_for_path(path)
        entries.append(
            DiffEntry(
                status=status_letter,
                old_path=path,
                new_path=path,
                subsystem=subsystem,
                lines_added=lines_added,
                lines_deleted=lines_deleted,
                touchpoints=touchpoints,
            )
        )

    entries.sort(key=lambda entry: (entry.subsystem, entry.status, entry.old_path, entry.new_path))
    if not entries:
        _warn(
            "No upstream-origin changed files detected for configured baseline.",
            warnings,
        )
    return entries


def _collect_commit_metadata(
    entries: list[DiffEntry],
    upstream_commit: str,
    head_ref: str,
    timeout_seconds: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    target_paths: set[str] = set()
    deleted_paths: set[str] = set()
    for entry in entries:
        target_paths.add(entry.old_path)
        target_paths.add(entry.new_path)
        if entry.status == "D":
            deleted_paths.add(entry.old_path)

    if not target_paths:
        return {}, {}, {}

    log_cmd = [
        "log",
        "--name-status",
        f"--find-renames={RENAME_THRESHOLD}",
        "--format=" + COMMIT_MARKER + "%H",
        f"{upstream_commit}..{head_ref}",
        "--",
        *sorted(target_paths),
    ]
    output = _run_git(log_cmd, check=True, timeout_seconds=timeout_seconds).stdout

    last_touch: dict[str, str] = {}
    deletion_commit: dict[str, str] = {}
    pre_deletion_commit: dict[str, str] = {}

    current_commit = ""
    for line in output.splitlines():
        if line.startswith(COMMIT_MARKER):
            current_commit = line[len(COMMIT_MARKER) :].strip()
            continue

        if not current_commit or not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        status = parts[0]
        status_letter = status[0]

        paths: list[str] = []
        if status_letter == "R" and len(parts) >= 3:
            old_path = parts[1]
            new_path = parts[2]
            paths.extend([old_path, new_path])
        else:
            paths.append(parts[1])

        for path in paths:
            if path in target_paths and path not in last_touch:
                last_touch[path] = current_commit

        if status_letter == "D":
            deleted_path = parts[1]
            if deleted_path in deleted_paths and deleted_path not in deletion_commit:
                deletion_commit[deleted_path] = current_commit
                continue

        for path in paths:
            if (
                path in deleted_paths
                and path in deletion_commit
                and path not in pre_deletion_commit
                and status_letter != "D"
            ):
                pre_deletion_commit[path] = current_commit

    return last_touch, deletion_commit, pre_deletion_commit


def _short_commit(commit: str | None) -> str:
    if not commit:
        return "-"
    return commit[:12]


def _render_path(entry: DiffEntry) -> str:
    if entry.status == "R":
        return f"{entry.old_path} -> {entry.new_path}"
    return entry.old_path


def _render_hunk_range(start: int, count: int) -> str:
    if count <= 1:
        return f"L{start}"
    return f"L{start}-L{start + count - 1}"


def _entry_primary_path(entry_payload: dict[str, Any]) -> str:
    if entry_payload.get("status") == "D":
        return str(entry_payload.get("old_path", ""))
    return str(entry_payload.get("new_path", ""))


def _is_engine_source_path(path: str) -> bool:
    normalized = path.strip().lower()
    if not normalized:
        return False
    if not any(normalized.startswith(prefix) for prefix in ENGINE_SOURCE_ROOT_PREFIXES):
        return False
    return Path(normalized).suffix in ENGINE_SOURCE_EXTENSIONS


def _render_markdown(
    *,
    output_path: Path,
    report: dict[str, Any],
    summary_only: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Engine Patch Report")
    lines.append("")
    lines.append(
        "This report lists fork deltas against pinned upstream Godot source for files that originate from upstream."
    )
    lines.append("")

    status = report["status"]
    metadata = report["metadata"]

    lines.append("## Metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Status | `{status}` |")
    lines.append(f"| Upstream repo | `{metadata['upstream_repo']}` |")
    lines.append(f"| Upstream ref (configured) | `{metadata['upstream_ref']}` |")
    lines.append(f"| Upstream commit (resolved) | `{metadata['upstream_commit']}` |")
    lines.append(f"| Fork-base commit (merge-base) | `{metadata['fork_base_commit']}` |")
    lines.append(f"| Head ref | `{metadata['head_ref']}` |")
    lines.append(f"| Head commit | `{metadata['head_commit']}` |")
    lines.append(f"| Rename threshold | `-M{RENAME_THRESHOLD}` |")
    lines.append("")

    warnings = report.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if status != "ok":
        lines.append("## Report state")
        lines.append("")
        lines.append("Generation was skipped due to an upstream/config resolution error.")
        lines.append("")
        lines.append("Regenerate with:")
        lines.append("")
        lines.append("```bash")
        lines.append("python3 scripts/generate_engine_patch_report.py")
        lines.append("```")
        lines.append("")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    summary = report["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total upstream-origin changed files | {summary['total_files']} |")
    lines.append(f"| Modified (`M`) | {summary['modified']} |")
    lines.append(f"| Renamed (`R*`) | {summary['renamed']} |")
    lines.append(f"| Deleted (`D`) | {summary['deleted']} |")
    lines.append("")

    engine_summary = report.get(
        "engine_source_summary",
        {"total_files": 0, "modified": 0, "renamed": 0, "deleted": 0},
    )
    lines.append("## Engine-source-only summary")
    lines.append("")
    lines.append(
        "Filter: files under runtime engine source roots with code/shader extensions "
        "(C/C++/Obj-C headers/sources and `.glsl`)."
    )
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total engine-source changed files | {engine_summary['total_files']} |")
    lines.append(f"| Modified (`M`) | {engine_summary['modified']} |")
    lines.append(f"| Renamed (`R*`) | {engine_summary['renamed']} |")
    lines.append(f"| Deleted (`D`) | {engine_summary['deleted']} |")
    lines.append("")

    engine_source_entries = report.get("engine_source_entries", [])
    lines.append("## Engine-source-only changed files")
    lines.append("")
    lines.append("| Status | Path | Subsystem | + | - | Last touch |")
    lines.append("| --- | --- | --- | ---: | ---: | --- |")
    for entry in engine_source_entries:
        lines.append(
            "| "
            f"`{entry['status']}` | "
            f"`{entry['display_path']}` | "
            f"`{entry['subsystem']}` | "
            f"{entry['lines_added']} | "
            f"{entry['lines_deleted']} | "
            f"`{_short_commit(entry.get('last_touch_commit'))}` |"
        )
    lines.append("")

    lines.append("## Subsystem distribution")
    lines.append("")
    lines.append("| Subsystem | Files |")
    lines.append("| --- | ---: |")
    for subsystem in report["subsystems"]:
        lines.append(f"| `{subsystem['name']}` | {subsystem['count']} |")
    lines.append("")

    lines.append("## Changed upstream-origin files")
    lines.append("")
    lines.append(
        "| Status | Path | Subsystem | + | - | Last touch | Deletion commit | Pre-deletion commit |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- | --- |")
    for entry in report["entries"]:
        lines.append(
            "| "
            f"`{entry['status']}` | "
            f"`{entry['display_path']}` | "
            f"`{entry['subsystem']}` | "
            f"{entry['lines_added']} | "
            f"{entry['lines_deleted']} | "
            f"`{_short_commit(entry.get('last_touch_commit'))}` | "
            f"`{_short_commit(entry.get('deletion_commit'))}` | "
            f"`{_short_commit(entry.get('pre_deletion_commit'))}` |"
        )
    lines.append("")

    if not summary_only:
        lines.append("## Touchpoints (compact)")
        lines.append("")
        lines.append(
            "Line-range touchpoints are compact in Markdown; full per-hunk details are canonical in `engine-patch.json`."
        )
        lines.append("")
        for entry in report["entries"]:
            touchpoints = entry.get("touchpoints", [])
            if not touchpoints:
                continue
            lines.append(f"### `{entry['display_path']}`")
            lines.append("")
            max_touchpoints = 6
            for touchpoint in touchpoints[:max_touchpoints]:
                old_range = _render_hunk_range(
                    int(touchpoint["old_start"]), int(touchpoint["old_count"])
                )
                new_range = _render_hunk_range(
                    int(touchpoint["new_start"]), int(touchpoint["new_count"])
                )
                context = str(touchpoint.get("context", "")).strip()
                context_suffix = f" ({context})" if context else ""
                lines.append(
                    f"- old `{old_range}` -> new `{new_range}`{context_suffix}"
                )
            if len(touchpoints) > max_touchpoints:
                lines.append(
                    f"- ... {len(touchpoints) - max_touchpoints} additional touchpoints in JSON output"
                )
            lines.append("")

    lines.append("## Regeneration")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/generate_engine_patch_report.py")
    lines.append("```")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(
    *,
    output_md: Path,
    output_json: Path,
    report: dict[str, Any],
    summary_only: bool,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _render_markdown(output_path=output_md, report=report, summary_only=summary_only)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_skipped_report(
    *,
    reason: str,
    warnings: list[str],
    upstream_repo: str,
    upstream_ref: str,
    head_ref: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "metadata": {
            "upstream_repo": upstream_repo,
            "upstream_ref": upstream_ref,
            "upstream_commit": "-",
            "fork_base_commit": "-",
            "head_ref": head_ref,
            "head_commit": "-",
            "rename_threshold": RENAME_THRESHOLD,
            "engine_source_filter": {
                "roots": list(ENGINE_SOURCE_ROOT_PREFIXES),
                "extensions": sorted(ENGINE_SOURCE_EXTENSIONS),
            },
        },
        "warnings": [*warnings, reason],
        "summary": {
            "total_files": 0,
            "modified": 0,
            "renamed": 0,
            "deleted": 0,
        },
        "engine_source_summary": {
            "total_files": 0,
            "modified": 0,
            "renamed": 0,
            "deleted": 0,
        },
        "subsystems": [],
        "engine_source_entries": [],
        "entries": [],
    }


def _build_report(
    *,
    config: Config,
    head_ref: str,
    timeout_seconds: int,
    warnings: list[str],
) -> dict[str, Any]:
    resolved_ref, upstream_commit = _ensure_upstream_commit(
        config.upstream_repo,
        config.upstream_ref,
        timeout_seconds=timeout_seconds,
    )

    head_commit = _resolve_commit(head_ref, timeout_seconds)
    if not head_commit:
        raise ReportError(f"Failed to resolve head ref '{head_ref}' to a commit")

    fork_base_commit = _resolve_merge_base(head_commit, upstream_commit, timeout_seconds)
    if not fork_base_commit:
        raise ReportError(
            "Failed to compute fork-base commit via merge-base. "
            "Pin upstream_ref to an ancestor/fork-point commit."
        )
    if fork_base_commit != upstream_commit:
        _warn(
            f"Configured upstream ref resolved to {upstream_commit[:12]}, "
            f"but fork-base commit is {fork_base_commit[:12]}; diff uses fork-base commit.",
            warnings,
        )

    entries = _collect_entries(
        fork_base_commit,
        head_ref,
        timeout_seconds=timeout_seconds,
        warnings=warnings,
    )
    last_touch, deletion_commit, pre_deletion_commit = _collect_commit_metadata(
        entries, fork_base_commit, head_ref, timeout_seconds
    )

    entry_payloads: list[dict[str, Any]] = []
    subsystem_counts: dict[str, int] = {}
    modified = renamed = deleted = 0

    for entry in entries:
        subsystem_counts[entry.subsystem] = subsystem_counts.get(entry.subsystem, 0) + 1

        if entry.status == "M":
            modified += 1
        elif entry.status == "R":
            renamed += 1
        elif entry.status == "D":
            deleted += 1

        display_path = _render_path(entry)
        latest_path = entry.new_path if entry.status != "D" else entry.old_path

        payload = {
            "status": entry.status,
            "old_path": entry.old_path,
            "new_path": entry.new_path,
            "display_path": display_path,
            "subsystem": entry.subsystem,
            "lines_added": entry.lines_added,
            "lines_deleted": entry.lines_deleted,
            "last_touch_commit": last_touch.get(latest_path)
            or last_touch.get(entry.old_path)
            or None,
            "deletion_commit": deletion_commit.get(entry.old_path),
            "pre_deletion_commit": pre_deletion_commit.get(entry.old_path),
            "touchpoints": entry.touchpoints,
        }
        entry_payloads.append(payload)

    engine_source_entries = [
        entry for entry in entry_payloads if _is_engine_source_path(_entry_primary_path(entry))
    ]
    engine_source_modified = sum(1 for entry in engine_source_entries if entry["status"] == "M")
    engine_source_renamed = sum(1 for entry in engine_source_entries if entry["status"] == "R")
    engine_source_deleted = sum(1 for entry in engine_source_entries if entry["status"] == "D")

    subsystems = [
        {"name": name, "count": subsystem_counts[name]}
        for name in sorted(subsystem_counts.keys())
    ]

    return {
        "status": "ok",
        "metadata": {
            "upstream_repo": config.upstream_repo,
            "upstream_ref": config.upstream_ref,
            "upstream_ref_resolved": resolved_ref,
            "upstream_commit": upstream_commit,
            "fork_base_commit": fork_base_commit,
            "head_ref": head_ref,
            "head_commit": head_commit,
            "rename_threshold": RENAME_THRESHOLD,
            "label": config.label,
            "engine_source_filter": {
                "roots": list(ENGINE_SOURCE_ROOT_PREFIXES),
                "extensions": sorted(ENGINE_SOURCE_EXTENSIONS),
            },
        },
        "warnings": warnings,
        "summary": {
            "total_files": len(entry_payloads),
            "modified": modified,
            "renamed": renamed,
            "deleted": deleted,
        },
        "engine_source_summary": {
            "total_files": len(engine_source_entries),
            "modified": engine_source_modified,
            "renamed": engine_source_renamed,
            "deleted": engine_source_deleted,
        },
        "subsystems": subsystems,
        "engine_source_entries": engine_source_entries,
        "entries": entry_payloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate upstream-origin engine patch report markdown/json."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--upstream-repo", default=None, help="Override upstream repository URL.")
    parser.add_argument("--upstream-ref", default=None, help="Override pinned upstream ref.")
    parser.add_argument("--head-ref", default="HEAD", help="Head ref to compare against upstream baseline.")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD, help="Markdown output path.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="JSON output path.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write summary-focused markdown and omit compact touchpoint details.",
    )
    parser.add_argument(
        "--fetch-timeout-seconds",
        type=int,
        default=45,
        help="Timeout (seconds) for git fetch/diff/log subprocess calls.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on config/fetch/report errors instead of warning+skipping.",
    )
    args = parser.parse_args()

    warnings: list[str] = []

    try:
        config = _load_config(args.config)
        if args.upstream_repo:
            config = Config(
                upstream_repo=args.upstream_repo,
                upstream_ref=config.upstream_ref,
                label=config.label,
            )
        if args.upstream_ref:
            config = Config(
                upstream_repo=config.upstream_repo,
                upstream_ref=args.upstream_ref,
                label=config.label,
            )

        report = _build_report(
            config=config,
            head_ref=args.head_ref,
            timeout_seconds=args.fetch_timeout_seconds,
            warnings=warnings,
        )
    except ReportError as exc:
        if args.strict:
            _print(f"ERROR: {exc}")
            return 1
        _warn(str(exc), warnings)
        report = _build_skipped_report(
            reason=str(exc),
            warnings=warnings,
            upstream_repo=args.upstream_repo or "-",
            upstream_ref=args.upstream_ref or "-",
            head_ref=args.head_ref,
        )
    except subprocess.TimeoutExpired as exc:
        message = f"Timed out running git command: {' '.join(exc.cmd) if exc.cmd else 'git'}"
        if args.strict:
            _print(f"ERROR: {message}")
            return 1
        _warn(message, warnings)
        report = _build_skipped_report(
            reason=message,
            warnings=warnings,
            upstream_repo=args.upstream_repo or "-",
            upstream_ref=args.upstream_ref or "-",
            head_ref=args.head_ref,
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Unexpected error: {exc}"
        if args.strict:
            _print(f"ERROR: {message}")
            return 1
        _warn(message, warnings)
        report = _build_skipped_report(
            reason=message,
            warnings=warnings,
            upstream_repo=args.upstream_repo or "-",
            upstream_ref=args.upstream_ref or "-",
            head_ref=args.head_ref,
        )

    _write_outputs(
        output_md=args.output_md,
        output_json=args.output_json,
        report=report,
        summary_only=args.summary_only,
    )

    _print(f"Wrote markdown report: {args.output_md}")
    _print(f"Wrote json report: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
