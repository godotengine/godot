#!/usr/bin/env python3
"""Analyze documentation coverage and quality metrics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
DOC_REPORT = ROOT / "docs" / "reports" / "documentation-coverage.md"
GAP_REPORT = ROOT / "docs" / "reports" / "documentation-gaps.md"
PROJECT_SOURCE_ROOTS = (
    ROOT / "modules" / "gaussian_splatting",
    ROOT / "scripts",
    ROOT / "test_data",
    ROOT / "tests" / "ci",
    ROOT / "tests" / "examples" / "godot" / "test_project",
    ROOT / "tests" / "runtime",
)
PROJECT_DOC_FILES = (
    ROOT / "AGENTS.md",
    ROOT / "BUILDING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "README.md",
)
PROJECT_DOC_ROOTS = (
    ROOT / ".github",
    ROOT / "ci",
    ROOT / "docs",
    ROOT / "modules" / "gaussian_splatting",
    ROOT / "tests" / "ci",
    ROOT / "tests" / "examples",
    ROOT / "tests" / "runtime",
)

CXX_EXTENSIONS = {".cpp", ".cc", ".hpp", ".h"}
GDSCRIPT_EXTENSIONS = {".gd"}
SHADER_EXTENSIONS = {".glsl"}

FUNCTION_DEF_CXX = re.compile(r"^[\w:<>,\s\*&]+\s+[\w:~]+\s*\([^;]*\)\s*(const)?\s*\{")
FUNCTION_DEF_GD = re.compile(r"^\s*func\s+[A-Za-z0-9_]+\s*\([^)]*\)")


def iter_source_files() -> Iterable[Path]:
    for root in PROJECT_SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in CXX_EXTENSIONS | GDSCRIPT_EXTENSIONS | SHADER_EXTENSIONS:
                continue
            yield path


@dataclass
class FileMetrics:
    path: Path
    code_lines: int
    comment_lines: int
    missing_docs: List[str]

    @property
    def comment_ratio(self) -> float:
        if self.code_lines == 0:
            return 0.0
        return self.comment_lines / self.code_lines


def count_lines(path: Path) -> FileMetrics:
    code_lines = 0
    comment_lines = 0
    missing_docs: List[str] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if path.suffix in CXX_EXTENSIONS | SHADER_EXTENSIONS:
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                comment_lines += 1
                continue
        if path.suffix in GDSCRIPT_EXTENSIONS and stripped.startswith("#"):
            comment_lines += 1
            continue
        code_lines += 1

        if path.suffix in CXX_EXTENSIONS and FUNCTION_DEF_CXX.match(stripped):
            previous = lines[idx - 1].strip() if idx > 0 else ""
            if not (previous.startswith("///") or previous.startswith("/**") or "/*" in previous):
                missing_docs.append(f"line {idx + 1}: {stripped[:60]}...")
        if path.suffix in GDSCRIPT_EXTENSIONS and FUNCTION_DEF_GD.match(stripped):
            previous = lines[idx - 1].strip() if idx > 0 else ""
            if not previous.startswith("##"):
                missing_docs.append(f"line {idx + 1}: {stripped[:60]}...")
        if path.suffix in SHADER_EXTENSIONS and FUNCTION_DEF_CXX.match(stripped):
            previous = lines[idx - 1].strip() if idx > 0 else ""
            if not previous.startswith("//"):
                missing_docs.append(f"line {idx + 1}: {stripped[:60]}...")
    return FileMetrics(path, code_lines, comment_lines, missing_docs)


VERSION_PATTERN = re.compile(r"Godot Engine\s*(?P<version>\d+\.\d+)")
VERSION_COMPONENT_PATTERN = re.compile(r'^(major|minor)\s*=\s*(\d+)\s*$', re.MULTILINE)
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def analyze_versions() -> list[str]:
    issues: list[str] = []
    readme = ROOT / "README.md"
    version_file = ROOT / "version.py"
    readme_version = None
    engine_version = None
    if readme.exists():
        match = VERSION_PATTERN.search(readme.read_text(encoding="utf-8"))
        if match:
            readme_version = match.group("version")
    if version_file.exists():
        components = {
            key: value
            for key, value in VERSION_COMPONENT_PATTERN.findall(version_file.read_text(encoding="utf-8"))
        }
        major = components.get("major")
        minor = components.get("minor")
        if major and minor:
            engine_version = f"{major}.{minor}"
    if readme_version and engine_version and readme_version != engine_version:
        issues.append(
            f"README version ({readme_version}) does not match engine version ({engine_version})."
        )
    return issues


def iter_project_markdown_files() -> Iterable[Path]:
    seen: set[Path] = set()
    for path in PROJECT_DOC_FILES:
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path
    for root in PROJECT_DOC_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            if path not in seen:
                seen.add(path)
                yield path


def find_broken_links() -> list[str]:
    issues: list[str] = []
    for md_path in iter_project_markdown_files():
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for match in LINK_PATTERN.finditer(text):
            target = match.group(1)
            if target.startswith("http") or target.startswith("mailto"):
                continue
            target_path = (md_path.parent / target).resolve()
            if not target_path.exists():
                issues.append(f"Broken link in {md_path.relative_to(ROOT)} -> {target}")
    return issues


def write_reports(metrics: list[FileMetrics]) -> None:
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    GAP_REPORT.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        "# Documentation Coverage",
        "",
        "| File | Comment/Code Ratio | Comment Lines | Code Lines |",
        "|------|--------------------|---------------|------------|",
    ]
    gaps: list[str] = []

    for metric in sorted(metrics, key=lambda m: m.comment_ratio):
        rel_path = metric.path.relative_to(ROOT)
        rows.append(
            f"| {rel_path} | {metric.comment_ratio:.2f} | {metric.comment_lines} | {metric.code_lines} |"
        )
        if metric.missing_docs:
            gaps.append(f"## {rel_path}\n" + "\n".join(f"- {item}" for item in metric.missing_docs) + "\n")

    version_issues = analyze_versions()
    link_issues = find_broken_links()

    if version_issues or link_issues:
        rows.append("\n## Quality Issues")
        for issue in version_issues + link_issues:
            rows.append(f"- {issue}")

    DOC_REPORT.write_text("\n".join(rows) + "\n", encoding="utf-8")

    gap_sections = ["# Documentation Gaps", ""]
    if gaps:
        gap_sections.extend(gaps)
    else:
        gap_sections.append("All analyzed functions include documentation.")
    GAP_REPORT.write_text("\n".join(gap_sections), encoding="utf-8")
    print(f"[audit] Wrote documentation reports to {DOC_REPORT} and {GAP_REPORT}")


def main() -> None:
    metrics = [count_lines(path) for path in iter_source_files()]
    write_reports(metrics)


if __name__ == "__main__":
    main()
