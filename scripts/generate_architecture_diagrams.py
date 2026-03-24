#!/usr/bin/env python3
"""Generate automated architecture diagrams and coupling reports.

This script intentionally emits a small architecture pack instead of a single
all-files graph. Large Mermaid graphs become unreadable quickly in this codebase.
To preserve completeness, the generator also writes machine-readable edge lists.
"""
from __future__ import annotations

import argparse
import csv
import json
import posixpath
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "modules" / "gaussian_splatting"
DEFAULT_OUTPUT = ROOT / "docs" / "architecture" / "generated"
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".cc"}
EXCLUDED_PARTS = {".run", "docs", "doc_classes", "tests"}
LOCAL_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
CLASS_DEF_RE = re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)\b[^{;]*\{", re.MULTILINE)
TOKEN_RE = re.compile(r"\b[A-Za-z_]\w*\b")
RENDERER_POINTER_RE = re.compile(r"\bGaussianSplatRenderer\s*\*")
RENDERER_STATE_GETTER_RE = re.compile(r"(?:->|\.)\s*(get_[A-Za-z0-9_]+_state)\s*\(")
RENDERER_INCLUDE_RE = re.compile(r'#\s*include\s+"(?:.*?/)?gaussian_splat_renderer\.h"')
ORCHESTRATOR_CLASS_RE = re.compile(r"\bRender[A-Za-z0-9_]*Orchestrator\b")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relpath: str
    subsystem: str
    content: str
    includes: tuple[str, ...]
    defined_symbols: tuple[str, ...]
    tokens: frozenset[str]


def module_rel(path: Path) -> str:
    return path.relative_to(MODULE_ROOT).as_posix()


def subsystem_for(relpath: str) -> str:
    return relpath.split("/", 1)[0]


def iter_source_paths(module_root: Path, include_tests: bool) -> list[Path]:
    paths: list[Path] = []
    for path in module_root.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if path.name.endswith(".gen.h"):
            continue
        parts = set(path.relative_to(module_root).parts)
        if not include_tests and parts & EXCLUDED_PARTS:
            continue
        paths.append(path)
    return sorted(paths)


def resolve_local_include(source_path: Path, include: str) -> str | None:
    include_path = Path(include)
    candidates: list[Path] = []
    if include.startswith("modules/gaussian_splatting/"):
        candidates.append(MODULE_ROOT / Path(include).relative_to("modules/gaussian_splatting"))
    elif include.startswith("../"):
        candidates.append(source_path.parent / include_path)
    else:
        candidates.append(source_path.parent / include_path)
        candidates.append(MODULE_ROOT / include_path)

    for candidate in candidates:
        try:
            rel_candidate = candidate.relative_to(MODULE_ROOT)
        except ValueError:
            continue
        if candidate.exists():
            return posixpath.normpath(rel_candidate.as_posix())
    return None


def parse_source_file(path: Path) -> SourceFile:
    content = path.read_text(encoding="utf-8")
    local_includes = []
    for include in LOCAL_INCLUDE_RE.findall(content):
        resolved = resolve_local_include(path, include)
        if resolved:
            local_includes.append(resolved)
    relpath = module_rel(path)
    return SourceFile(
        path=path,
        relpath=relpath,
        subsystem=subsystem_for(relpath),
        content=content,
        includes=tuple(sorted(set(local_includes))),
        defined_symbols=tuple(sorted(set(CLASS_DEF_RE.findall(content)))),
        tokens=frozenset(TOKEN_RE.findall(content)),
    )


def sanitize_mermaid_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not sanitized:
        sanitized = "node"
    if sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized


def summarize_count_list(items: Iterable[tuple[str, int]], limit: int) -> list[tuple[str, int]]:
    return sorted(items, key=lambda item: (-item[1], item[0]))[:limit]


def build_symbol_index(files: list[SourceFile]) -> dict[str, str]:
    symbol_to_file: dict[str, str] = {}
    for source in files:
        for symbol in source.defined_symbols:
            if symbol[0].isupper():
                symbol_to_file.setdefault(symbol, source.relpath)
    return symbol_to_file


def build_symbol_reference_edges(files: list[SourceFile], symbol_index: dict[str, str]) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for source in files:
        for symbol in sorted(source.tokens & symbol_index.keys()):
            target = symbol_index[symbol]
            if target != source.relpath:
                edges.append(("symbol", source.relpath, target))
    return edges


def build_include_edges(files: list[SourceFile], known_paths: set[str]) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for source in files:
        for include in source.includes:
            if include in known_paths:
                edges.append(("include", source.relpath, include))
    return edges


def collect_direct_access_metrics(files: list[SourceFile]) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for source in files:
        state_getters = Counter(RENDERER_STATE_GETTER_RE.findall(source.content))
        metrics[source.relpath] = {
            "renderer_pointer_mentions": len(RENDERER_POINTER_RE.findall(source.content)),
            "renderer_state_getter_calls": sum(state_getters.values()),
            "renderer_state_getter_names": dict(sorted(state_getters.items())),
            "includes_renderer_header": bool(RENDERER_INCLUDE_RE.search(source.content)),
            "orchestrator_mentions": len(ORCHESTRATOR_CLASS_RE.findall(source.content)),
        }
    return metrics


def aggregate_subsystem_edges(files_by_rel: dict[str, SourceFile], edges: list[tuple[str, str, str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for _, source, target in edges:
        source_subsystem = files_by_rel[source].subsystem
        target_subsystem = files_by_rel[target].subsystem
        if source_subsystem != target_subsystem:
            counts[(source_subsystem, target_subsystem)] += 1
    return counts


def render_subsystem_graph(files: list[SourceFile], subsystem_edges: Counter[tuple[str, str]]) -> str:
    subsystem_counts = Counter(source.subsystem for source in files)
    lines = [
        "# Generated Subsystem Dependency Graph",
        "",
        "This Mermaid graph is generated from local `#include` edges inside `modules/gaussian_splatting`.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for subsystem, count in sorted(subsystem_counts.items()):
        node_id = sanitize_mermaid_id(subsystem)
        lines.append(f'    {node_id}["{subsystem}\\n{count} files"]')
    for (source, target), count in sorted(subsystem_edges.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        source_id = sanitize_mermaid_id(source)
        target_id = sanitize_mermaid_id(target)
        lines.append(f"    {source_id} -->|{count}| {target_id}")
    lines.extend(["```", ""])
    return "\n".join(lines)


def renderer_focus_files(files: list[SourceFile]) -> set[str]:
    focused: set[str] = set()
    for source in files:
        if source.relpath.startswith("renderer/"):
            focused.add(source.relpath)
        if source.relpath in {
            "interfaces/renderer_interfaces.h",
            "interfaces/gpu_sorting_pipeline.h",
            "interfaces/gpu_sorting_pipeline.cpp",
            "interfaces/gpu_sorting_pipeline_interfaces.h",
            "interfaces/gpu_culler.h",
            "interfaces/gpu_culler.cpp",
            "interfaces/render_device_manager.h",
            "interfaces/render_device_manager.cpp",
            "interfaces/output_compositor.h",
            "interfaces/output_compositor.cpp",
            "interfaces/output_compositor_interfaces.h",
        }:
            focused.add(source.relpath)
    return focused


def render_renderer_graph(
    files_by_rel: dict[str, SourceFile],
    include_edges: list[tuple[str, str, str]],
    symbol_edges: list[tuple[str, str, str]],
) -> str:
    focus = renderer_focus_files(list(files_by_rel.values()))
    edge_weights: Counter[tuple[str, str]] = Counter()
    for _, source, target in include_edges + symbol_edges:
        if source in focus and target in focus and source != target:
            edge_weights[(source, target)] += 1

    incoming: Counter[str] = Counter()
    outgoing: Counter[str] = Counter()
    for source, target in edge_weights:
        outgoing[source] += edge_weights[(source, target)]
        incoming[target] += edge_weights[(source, target)]

    ranked = sorted(
        focus,
        key=lambda rel: (
            -(incoming[rel] + outgoing[rel]),
            rel,
        ),
    )
    selected = set(ranked[:18])
    lines = [
        "# Generated Renderer Coupling Graph",
        "",
        "This graph combines local include edges and in-module symbol references for the renderer-focused slice.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for rel in sorted(selected):
        node_id = sanitize_mermaid_id(rel)
        label = rel.replace("modules/gaussian_splatting/", "")
        lines.append(f'    {node_id}["{label}"]')
    for (source, target), weight in sorted(edge_weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        if source in selected and target in selected:
            lines.append(
                f"    {sanitize_mermaid_id(source)} -->|{weight}| {sanitize_mermaid_id(target)}"
            )
    lines.extend(["```", ""])
    return "\n".join(lines)


def render_hotspot_report(
    files: list[SourceFile],
    files_by_rel: dict[str, SourceFile],
    include_edges: list[tuple[str, str, str]],
    symbol_edges: list[tuple[str, str, str]],
    direct_metrics: dict[str, dict[str, object]],
) -> str:
    outgoing_includes = Counter(source for _, source, _ in include_edges)
    incoming_includes = Counter(target for _, _, target in include_edges)
    subsystem_counts = Counter(source.subsystem for source in files)

    state_hotspots = [
        (rel, int(metrics["renderer_state_getter_calls"]))
        for rel, metrics in direct_metrics.items()
        if int(metrics["renderer_state_getter_calls"]) > 0
    ]
    pointer_hotspots = [
        (rel, int(metrics["renderer_pointer_mentions"]))
        for rel, metrics in direct_metrics.items()
        if int(metrics["renderer_pointer_mentions"]) > 0
    ]
    include_hotspots = [
        (rel, 1)
        for rel, metrics in direct_metrics.items()
        if bool(metrics["includes_renderer_header"])
    ]

    lines = [
        "# Generated Coupling Report",
        "",
        "This report is generated from static source heuristics. It is useful for architecture reasoning, not as a perfect semantic model.",
        "",
        "## Scope",
        "",
        f"- Source files scanned: `{len(files)}`",
        f"- Subsystems scanned: `{len(subsystem_counts)}`",
        f"- Local include edges: `{len(include_edges)}`",
        f"- Symbol-reference edges: `{len(symbol_edges)}`",
        "",
        "## Top Outgoing Include Dependencies",
        "",
        "| File | Local include edges |",
        "| --- | ---: |",
    ]
    for rel, count in summarize_count_list(outgoing_includes.items(), 15):
        lines.append(f"| `{rel}` | {count} |")

    lines.extend(
        [
            "",
            "## Top Incoming Include Dependencies",
            "",
            "| File | Incoming local includes |",
            "| --- | ---: |",
        ]
    )
    for rel, count in summarize_count_list(incoming_includes.items(), 15):
        lines.append(f"| `{rel}` | {count} |")

    lines.extend(
        [
            "",
            "## GaussianSplatRenderer Direct-Access Hotspots",
            "",
            "Files with explicit `GaussianSplatRenderer *`, renderer header includes, or `get_*_state()` calls.",
            "",
            "| File | Renderer* mentions | `get_*_state()` calls | Includes renderer header |",
            "| --- | ---: | ---: | :---: |",
        ]
    )
    direct_rows = sorted(
        (
            rel,
            int(metrics["renderer_pointer_mentions"]),
            int(metrics["renderer_state_getter_calls"]),
            "yes" if bool(metrics["includes_renderer_header"]) else "",
        )
        for rel, metrics in direct_metrics.items()
        if int(metrics["renderer_pointer_mentions"]) > 0
        or int(metrics["renderer_state_getter_calls"]) > 0
        or bool(metrics["includes_renderer_header"])
    )
    for rel, pointer_count, getter_calls, includes_header in direct_rows:
        lines.append(f"| `{rel}` | {pointer_count} | {getter_calls} | {includes_header} |")

    lines.extend(
        [
            "",
            "## Top Renderer State Accessors",
            "",
            "| File | Accessors |",
            "| --- | --- |",
        ]
    )
    for rel, _ in summarize_count_list(state_hotspots, 15):
        accessor_names = direct_metrics[rel]["renderer_state_getter_names"]
        accessor_summary = ", ".join(f"`{name}` x{count}" for name, count in accessor_names.items()) or "-"
        lines.append(f"| `{rel}` | {accessor_summary} |")

    lines.extend(
        [
            "",
            "## Files Including `gaussian_splat_renderer.h`",
            "",
            "| File |",
            "| --- |",
        ]
    )
    for rel, _ in sorted(include_hotspots):
        lines.append(f"| `{rel}` |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Include edges show build-time/module coupling.",
            "- Symbol-reference edges add a rough semantic layer but are heuristic-based.",
            "- Renderer direct-access metrics are the best signal here for renderer-centric architecture leakage.",
            "",
        ]
    )
    return "\n".join(lines)


def render_renderer_direct_access_graph(direct_metrics: dict[str, dict[str, object]]) -> str:
    ranked = sorted(
        (
            (
                rel,
                int(metrics["renderer_pointer_mentions"]),
                int(metrics["renderer_state_getter_calls"]),
                bool(metrics["includes_renderer_header"]),
            )
            for rel, metrics in direct_metrics.items()
            if int(metrics["renderer_pointer_mentions"]) > 0
            or int(metrics["renderer_state_getter_calls"]) > 0
            or bool(metrics["includes_renderer_header"])
        ),
        key=lambda item: (-(item[1] + item[2]), item[0]),
    )[:20]

    lines = [
        "# Generated Renderer Direct-Access Graph",
        "",
        "This graph highlights files most strongly coupled to `GaussianSplatRenderer` through raw pointer mentions, state getter calls, or direct header inclusion.",
        "",
        "```mermaid",
        "flowchart LR",
        '    renderer["renderer/gaussian_splat_renderer.h"]',
    ]
    for rel, pointer_mentions, state_calls, includes_header in ranked:
        node_id = sanitize_mermaid_id(rel)
        include_marker = " incl" if includes_header else ""
        label = f"{rel}\\nptr:{pointer_mentions} state:{state_calls}{include_marker}"
        lines.append(f'    {node_id}["{label}"]')
        lines.append(f"    {node_id} --> renderer")
    lines.extend(["```", ""])
    return "\n".join(lines)


def render_generated_index(files: list[SourceFile]) -> str:
    return "\n".join(
        [
            "# Generated Architecture Pack",
            "",
            "This folder is generated by `scripts/generate_architecture_diagrams.py`.",
            "",
            "## Artifacts",
            "",
            "- [subsystem-dependencies.md](subsystem-dependencies.md): Mermaid graph of local subsystem include dependencies.",
            "- [renderer-coupling.md](renderer-coupling.md): Mermaid graph focused on the renderer/orchestrator/interface slice.",
            "- [renderer-direct-access.md](renderer-direct-access.md): Mermaid graph of files most directly coupled to `GaussianSplatRenderer`.",
            "- [coupling-report.md](coupling-report.md): Hotspots for include coupling, renderer access, and state-access patterns.",
            "- [local-dependencies.csv](local-dependencies.csv): Complete machine-readable edge list.",
            "- [summary.json](summary.json): Machine-readable summary counts and hotspots.",
            "",
            "## Generation",
            "",
            "```bash",
            "python3 scripts/generate_architecture_diagrams.py",
            "```",
            "",
            f"Current scan size: `{len(files)}` source files.",
            "",
            "## Limits",
            "",
            "- This is a static approximation, not a full compiler-backed semantic model.",
            "- It intentionally avoids one giant all-file Mermaid graph because that becomes unreadable.",
            "- Use the CSV/JSON outputs when you need the complete dependency edge list.",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["kind", "source", "target", "source_subsystem", "target_subsystem"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate architecture diagrams and coupling reports.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for generated architecture artifacts.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include `tests/` in the generated graphs and reports.",
    )
    args = parser.parse_args()

    files = [parse_source_file(path) for path in iter_source_paths(MODULE_ROOT, args.include_tests)]
    files_by_rel = {source.relpath: source for source in files}
    symbol_index = build_symbol_index(files)
    include_edges = build_include_edges(files, set(files_by_rel))
    symbol_edges = build_symbol_reference_edges(files, symbol_index)
    direct_metrics = collect_direct_access_metrics(files)
    subsystem_edges = aggregate_subsystem_edges(files_by_rel, include_edges)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "subsystem-dependencies.md").write_text(
        render_subsystem_graph(files, subsystem_edges),
        encoding="utf-8",
    )
    (output_dir / "renderer-coupling.md").write_text(
        render_renderer_graph(files_by_rel, include_edges, symbol_edges),
        encoding="utf-8",
    )
    (output_dir / "renderer-direct-access.md").write_text(
        render_renderer_direct_access_graph(direct_metrics),
        encoding="utf-8",
    )
    (output_dir / "coupling-report.md").write_text(
        render_hotspot_report(files, files_by_rel, include_edges, symbol_edges, direct_metrics),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(render_generated_index(files), encoding="utf-8")

    csv_rows: list[dict[str, object]] = []
    for kind, source, target in include_edges + symbol_edges:
        csv_rows.append(
            {
                "kind": kind,
                "source": source,
                "target": target,
                "source_subsystem": files_by_rel[source].subsystem,
                "target_subsystem": files_by_rel[target].subsystem,
            }
        )
    write_csv(output_dir / "local-dependencies.csv", csv_rows)

    summary = {
        "source_files": len(files),
        "subsystems": dict(sorted(Counter(source.subsystem for source in files).items())),
        "include_edges": len(include_edges),
        "symbol_reference_edges": len(symbol_edges),
        "top_outgoing_include_files": summarize_count_list(Counter(source for _, source, _ in include_edges).items(), 15),
        "top_incoming_include_files": summarize_count_list(Counter(target for _, _, target in include_edges).items(), 15),
        "renderer_pointer_hotspots": summarize_count_list(
            (
                (rel, int(metrics["renderer_pointer_mentions"]))
                for rel, metrics in direct_metrics.items()
                if int(metrics["renderer_pointer_mentions"]) > 0
            ),
            20,
        ),
        "renderer_state_access_hotspots": summarize_count_list(
            (
                (rel, int(metrics["renderer_state_getter_calls"]))
                for rel, metrics in direct_metrics.items()
                if int(metrics["renderer_state_getter_calls"]) > 0
            ),
            20,
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[architecture] Wrote generated artifacts to {output_dir}")
    print(
        "[architecture] "
        f"scanned={len(files)} include_edges={len(include_edges)} symbol_edges={len(symbol_edges)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
