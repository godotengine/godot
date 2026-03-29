#!/usr/bin/env python3
"""Docs release acceptance checks for staged public docs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required to run docs release acceptance checks.") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(r'<a\b[^>]*href\s*=\s*(["\'])(.*?)\1', re.IGNORECASE)
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img\b([^>]*)>", re.IGNORECASE)
ALT_ATTR_RE = re.compile(r'\balt\s*=\s*(["\'])(.*?)\1', re.IGNORECASE)
SRC_ATTR_RE = re.compile(r'\bsrc\s*=\s*(["\'])(.*?)\1', re.IGNORECASE)


class MkDocsLoader(yaml.SafeLoader):
    """YAML loader that tolerates MkDocs !ENV tags."""


def _construct_env(loader: MkDocsLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        values = loader.construct_sequence(node)
        return values[-1] if values else None
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    return loader.construct_object(node)


def _construct_unknown(loader: MkDocsLoader, _suffix: str, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


MkDocsLoader.add_constructor("!ENV", _construct_env)
MkDocsLoader.add_multi_constructor("", _construct_unknown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate docs release acceptance checks.")
    parser.add_argument("--public-docs", default=".site/public-docs", help="Staged public docs root.")
    parser.add_argument("--mkdocs-config", default="mkdocs.yml", help="MkDocs config to inspect for redirects.")
    return parser.parse_args()


def iter_markdown_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.md") if path.name != "README.md")


def collect_internal_targets(text: str) -> list[str]:
    targets = list(MARKDOWN_LINK_RE.findall(text))
    targets.extend(match[1] for match in HTML_LINK_RE.findall(text))
    return targets


def resolve_internal_target(source: Path, target: str) -> Path | None:
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    path_part = urldefrag(target.split("?", 1)[0])[0]
    if not path_part:
        return None
    return (source.parent / path_part).resolve()


def find_orphan_pages(public_docs: Path) -> list[str]:
    files = iter_markdown_files(public_docs)
    public_paths = {path.resolve() for path in files}
    inbound: Counter[Path] = Counter()
    for source in files:
        text = source.read_text(encoding="utf-8", errors="ignore")
        for target in collect_internal_targets(text):
            resolved = resolve_internal_target(source, target)
            if resolved in public_paths:
                inbound[resolved] += 1

    orphans: list[str] = []
    for path in files:
        resolved = path.resolve()
        if path.name == "index.md":
            continue
        if inbound[resolved] == 0:
            orphans.append(path.relative_to(REPO_ROOT).as_posix())
    return orphans


def find_missing_alt_text(public_docs: Path) -> list[str]:
    issues: list[str] = []
    for path in iter_markdown_files(public_docs):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for match in MARKDOWN_IMAGE_RE.finditer(text):
            alt_text = match.group(1).strip()
            target = match.group(2).strip()
            if target.startswith(("http://", "https://")):
                continue
            if not alt_text:
                issues.append(f"{rel}: markdown image missing alt -> {target}")
        for match in HTML_IMAGE_RE.finditer(text):
            attrs = match.group(1)
            src_match = SRC_ATTR_RE.search(attrs)
            if src_match and src_match.group(2).startswith(("http://", "https://")):
                continue
            alt_match = ALT_ATTR_RE.search(attrs)
            if not alt_match or not alt_match.group(2).strip():
                src = src_match.group(2) if src_match else "<unknown>"
                issues.append(f"{rel}: html image missing alt -> {src}")
    return issues


def validate_redirects(mkdocs_config: Path, public_docs: Path) -> list[str]:
    config = yaml.load(mkdocs_config.read_text(encoding="utf-8"), Loader=MkDocsLoader) or {}
    redirect_maps: dict[str, str] = {}
    for plugin in config.get("plugins", []):
        if isinstance(plugin, dict) and "redirects" in plugin:
            redirects = plugin.get("redirects") or {}
            redirect_maps = redirects.get("redirect_maps") or {}
            break

    issues: list[str] = []
    seen_sources: set[str] = set()
    for source, target in redirect_maps.items():
        normalized_source = str(source).strip().lstrip("/")
        normalized_target = str(target).strip().lstrip("/")
        if not normalized_source or not normalized_target:
            issues.append(f"Redirect map has empty source or target: {source!r} -> {target!r}")
            continue
        if normalized_source == normalized_target:
            issues.append(f"Redirect map is self-referential: {normalized_source}")
        if normalized_source in seen_sources:
            issues.append(f"Redirect map has duplicate source entry: {normalized_source}")
        seen_sources.add(normalized_source)

        target_candidates = [
            public_docs / normalized_target,
            public_docs / f"{normalized_target}.md",
            public_docs / normalized_target / "index.md",
        ]
        if not any(candidate.exists() for candidate in target_candidates):
            issues.append(f"Redirect target does not resolve to a staged doc: {normalized_source} -> {normalized_target}")
    return issues


def run_command(label: str, command: list[str]) -> list[str]:
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode == 0:
        return []
    details = completed.stderr.strip() or completed.stdout.strip() or f"{label} failed."
    return [f"{label}: {details}"]


def main() -> int:
    args = parse_args()
    public_docs = (REPO_ROOT / args.public_docs).resolve()
    mkdocs_config = (REPO_ROOT / args.mkdocs_config).resolve()

    if not public_docs.exists():
        print(f"Staged public docs root does not exist: {public_docs}", file=sys.stderr)
        return 1
    if not mkdocs_config.exists():
        print(f"MkDocs config does not exist: {mkdocs_config}", file=sys.stderr)
        return 1

    failures: list[str] = []
    failures.extend(
        run_command(
            "internal link check",
            [sys.executable, "scripts/docs/check_links.py", "docs", "README.md", "BUILDING.md", "CONTRIBUTING.md"],
        )
    )
    failures.extend(run_command("media budget check", [sys.executable, "scripts/check_docs_media_budget.py"]))

    orphan_pages = find_orphan_pages(public_docs)
    missing_alt = find_missing_alt_text(public_docs)
    redirect_issues = validate_redirects(mkdocs_config, public_docs)

    failures.extend(f"public orphan page: {item}" for item in orphan_pages)
    failures.extend(missing_alt)
    failures.extend(redirect_issues)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"Docs release acceptance failed with {len(failures)} issue(s).", file=sys.stderr)
        return 1

    print("Docs release acceptance passed.")
    print(f"Checked staged public docs: {public_docs}")
    print("Validated: links, media budget, orphan pages, alt text, redirect targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
