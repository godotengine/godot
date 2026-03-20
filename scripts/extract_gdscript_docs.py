#!/usr/bin/env python3
"""Extract documentation from GDScript sources into Markdown."""
from __future__ import annotations

import argparse
import fnmatch
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "api" / "gdscript_reference.md"
DEFAULT_SOURCE_ROOTS = (
    ROOT / "modules" / "gaussian_splatting",
    ROOT / "scripts",
    ROOT / "templates",
    ROOT / "tests" / "runtime",
    ROOT / "tests" / "examples" / "godot" / "test_project",
)
DEFAULT_EXCLUDE_GLOBS = ("**/addons/**",)

FUNC_PATTERN = re.compile(r"^\s*func\s+(?P<name>\w+)\s*\(")
CLASS_PATTERN = re.compile(r"^\s*class_name\s+(?P<name>[A-Za-z0-9_]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Markdown API docs from GDScript files.")
    parser.add_argument(
        "--scope",
        choices=("public", "all"),
        default="public",
        help="public: exclude test/internal/tooling scripts, all: include every discovered script.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Replacement source roots (overrides defaults). Can be repeated.",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help="If set, only scripts matching at least one repo-relative glob are included.",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Exclude scripts matching repo-relative glob(s). Can be repeated.",
    )
    parser.add_argument(
        "--include-undocumented",
        action="store_true",
        help="Include undocumented members in the generated tables.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT),
        help="Output Markdown file path.",
    )
    return parser.parse_args()


def _extract_balanced_params(line: str, open_pos: int) -> str:
    """Extract parameter text from *line* starting at the opening '(' at *open_pos*.

    Uses a depth counter so that nested parentheses (e.g. default values like
    ``PackedInt32Array()``) are handled correctly at arbitrary nesting depth.
    Returns the text between the outermost parentheses (exclusive).
    """
    depth = 0
    start = open_pos + 1
    for i in range(open_pos, len(line)):
        ch = line[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return line[start:i].strip()
    # Unbalanced – fall back to everything after the opening paren.
    return line[start:].strip()


def extract_docs(path: Path) -> dict[str, dict[str, str]]:
    docs: dict[str, dict[str, str]] = {}
    current_class = path.stem
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        class_match = CLASS_PATTERN.match(line)
        if class_match:
            current_class = class_match.group("name")
            docs.setdefault(current_class, {})
            continue

        func_match = FUNC_PATTERN.match(line)
        if func_match:
            func_name = func_match.group("name").strip()
            # Find the opening paren that the regex matched and use balanced
            # extraction so nested parens in default values are preserved.
            open_pos = line.index("(", func_match.start())
            params = _extract_balanced_params(line, open_pos)
            signature = f"{func_name}({params})" if params else f"{func_name}()"
            doc_lines = []
            lookback = idx - 1
            while lookback >= 0 and lines[lookback].strip().startswith("##"):
                doc_lines.append(lines[lookback].strip().lstrip("# "))
                lookback -= 1
            doc_lines.reverse()
            docs.setdefault(current_class, {})[signature] = " ".join(doc_lines) if doc_lines else "(undocumented)"
    return docs


def _path_matches_any(rel_path: Path, patterns: list[str]) -> bool:
    rel = rel_path.as_posix()
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern):
            return True
    return False


def _is_internal_script(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.parts and rel.parts[0] == "test_data":
        return True
    if "tests" in rel.parts:
        return True
    # Exclude build/tooling scripts that live directly under scripts/ or in
    # scripts/tools/.  Only scripts/core/ contains user-facing runtime code.
    rel_posix = rel.as_posix()
    if rel_posix.startswith("scripts/") and not rel_posix.startswith("scripts/core/"):
        return True
    return False


def _resolve_source_roots(source_roots: list[str]) -> list[Path]:
    if not source_roots:
        return list(DEFAULT_SOURCE_ROOTS)
    roots: list[Path] = []
    for value in source_roots:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        roots.append(candidate.resolve())
    return roots


def collect_sources(
    *,
    scope: str,
    source_roots: list[str],
    include_globs: list[str],
    exclude_globs: list[str],
) -> list[Path]:
    sources: list[Path] = []
    roots = _resolve_source_roots(source_roots)
    effective_excludes = list(DEFAULT_EXCLUDE_GLOBS) + list(exclude_globs)

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.gd"):
            if not path.is_file():
                continue
            if "addons" in path.parts:
                continue
            rel = path.relative_to(ROOT)
            if _path_matches_any(rel, effective_excludes):
                continue
            if scope == "public" and _is_internal_script(path):
                continue
            if include_globs and not _path_matches_any(rel, include_globs):
                continue
            sources.append(path)

    return sorted(set(sources))


def _render_member_table(members: dict[str, str]) -> list[str]:
    rows = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>Member</th>",
        "      <th>Description</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for signature, description in sorted(members.items()):
        desc = description if description and description != "(undocumented)" else "Undocumented."
        rows.extend(
            [
                "    <tr>",
                f"      <td><pre><code>{signature}</code></pre></td>",
                f"      <td>{desc}</td>",
                "    </tr>",
            ]
        )
    rows.extend(["  </tbody>", "</table>"])
    return rows


def build_reference(
    *,
    scope: str,
    source_roots: list[str],
    include_globs: list[str],
    exclude_globs: list[str],
    include_undocumented: bool,
) -> str:
    scripts = collect_sources(
        scope=scope,
        source_roots=source_roots,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    )
    sections = [
        "# GDScript API Reference",
        "",
        f"Last generated: {date.today().isoformat()}",
        "",
        f"Scope: `{scope}`",
        "",
        f"Scripts scanned: `{len(scripts)}`",
        "",
    ]
    if not include_undocumented:
        sections.extend(
            [
                "Undocumented members are omitted by default. Use `--include-undocumented` to include them.",
                "",
            ]
        )
    for script in scripts:
        docs = extract_docs(script)
        if not docs:
            continue
        sections.append("## Script")
        sections.append("")
        sections.append("```")
        sections.append(script.relative_to(ROOT).as_posix())
        sections.append("```")
        sections.append("")
        for class_name, members in docs.items():
            rendered_members: dict[str, str] = {}
            for signature, description in members.items():
                if description == "(undocumented)" and not include_undocumented:
                    continue
                rendered_members[signature] = description

            sections.append("### Class")
            sections.append("")
            sections.append("```")
            sections.append(class_name)
            sections.append("```")
            sections.append("")
            if not rendered_members:
                sections.append("No documented functions.")
                sections.append("")
                continue
            sections.extend(_render_member_table(rendered_members))
            sections.append("")
    sections.extend(
        [
            "Generated by:",
            "",
            "```",
            "scripts/extract_gdscript_docs.py",
            "```",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_reference(
            scope=args.scope,
            source_roots=args.source_root,
            include_globs=args.include_glob,
            exclude_globs=args.exclude_glob,
            include_undocumented=args.include_undocumented,
        ),
        encoding="utf-8",
    )
    print(f"[docs] Wrote GDScript reference to {output}")


if __name__ == "__main__":
    main()
