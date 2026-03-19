#!/usr/bin/env python3
"""Stage public docs content for MkDocs/GitHub Pages publishing."""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Sequence
from urllib.parse import quote, urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDED_DIRS = ("agent_memory", "archive")
MARKDOWN_LINK_PATTERN = re.compile(r"(!?\[[^\]]+\]\()([^)]+)(\))")
HTML_ATTR_PATTERN = re.compile(
    r'(<(?:a|img|video|source)\b[^>]*?\s(?:href|src)=["\'])([^"\']+)(["\'])',
    flags=re.IGNORECASE,
)
REMOTE_URL_PATTERN = re.compile(r"^https?://", flags=re.IGNORECASE)
MEDIA_EXTENSIONS = {".avif", ".gif", ".jpeg", ".jpg", ".mp4", ".png", ".svg", ".webm", ".webp"}


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_repo_coordinates(repo_url: str) -> tuple[str, str] | None:
    normalized = repo_url.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    match = re.match(r"^https?://github\.com/([^/]+)/([^/]+)$", normalized)
    if not match:
        return None
    return match.group(1), match.group(2)


def build_blob_url(repo_url: str, ref: str, relative_path: Path, is_dir: bool) -> str:
    encoded_path = quote(relative_path.as_posix())
    kind = "tree" if is_dir else "blob"
    return f"{repo_url.rstrip('/')}/{kind}/{quote(ref)}/{encoded_path}"


def build_raw_url(repo_url: str, ref: str, relative_path: Path) -> str:
    coords = parse_repo_coordinates(repo_url)
    if not coords:
        return build_blob_url(repo_url, ref, relative_path, is_dir=False)
    owner, repo = coords
    encoded_path = quote(relative_path.as_posix())
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{quote(ref)}/{encoded_path}"


def should_exclude(relative_path: Path, excluded_dirs: set[str]) -> bool:
    return bool(relative_path.parts) and relative_path.parts[0] in excluded_dirs


def split_target_token(token: str) -> tuple[str, str]:
    stripped = token.strip()
    if not stripped:
        return token, ""
    # Support Markdown targets with optional title text.
    if stripped.startswith("<") and stripped.endswith(">"):
        return stripped[1:-1], ""
    match = re.match(r"^(\S+)(\s+.+)?$", stripped)
    if not match:
        return stripped, ""
    suffix = match.group(2) or ""
    return match.group(1), suffix


def rebuild_url(split_result, new_path: str) -> str:
    return urlunsplit(
        (
            split_result.scheme,
            split_result.netloc,
            new_path,
            split_result.query,
            split_result.fragment,
        )
    )


def rewrite_target(
    target: str,
    *,
    is_embed: bool,
    source_file: Path,
    staged_file: Path,
    source_root: Path,
    staged_root: Path,
    excluded_dirs: set[str],
    repo_url: str | None,
    ref: str | None,
) -> str:
    split_result = urlsplit(target)
    if split_result.scheme or split_result.netloc:
        return target
    if not split_result.path or split_result.path.startswith("#") or split_result.path.startswith("/"):
        return target

    decoded_path = split_result.path
    staged_candidate = (staged_file.parent / decoded_path).resolve()
    if path_is_within(staged_candidate, staged_root) and staged_candidate.exists():
        return target

    source_candidate = (source_file.parent / decoded_path).resolve()
    if not source_candidate.exists() or not path_is_within(source_candidate, REPO_ROOT):
        return target
    if path_is_within(source_candidate, source_root):
        rel_from_source = source_candidate.relative_to(source_root)
        if not should_exclude(rel_from_source, excluded_dirs):
            return target

    if not repo_url or not ref:
        return target

    rel_from_repo = source_candidate.relative_to(REPO_ROOT)
    extension = rel_from_repo.suffix.lower()
    if source_candidate.is_dir():
        rewritten = build_blob_url(repo_url, ref, rel_from_repo, is_dir=True)
    elif is_embed or extension in MEDIA_EXTENSIONS:
        rewritten = build_raw_url(repo_url, ref, rel_from_repo)
    else:
        rewritten = build_blob_url(repo_url, ref, rel_from_repo, is_dir=False)

    return rebuild_url(split_result, rewritten)


def rewrite_markdown(
    text: str,
    *,
    source_file: Path,
    staged_file: Path,
    source_root: Path,
    staged_root: Path,
    excluded_dirs: set[str],
    repo_url: str | None,
    ref: str | None,
) -> str:
    def markdown_replacer(match: re.Match[str]) -> str:
        prefix, raw_target, suffix = match.groups()
        url_path, optional_suffix = split_target_token(raw_target)
        rewritten = rewrite_target(
            url_path,
            is_embed=prefix.startswith("!["),
            source_file=source_file,
            staged_file=staged_file,
            source_root=source_root,
            staged_root=staged_root,
            excluded_dirs=excluded_dirs,
            repo_url=repo_url,
            ref=ref,
        )
        return f"{prefix}{rewritten}{optional_suffix}{suffix}"

    def html_replacer(match: re.Match[str]) -> str:
        prefix, raw_target, suffix = match.groups()
        lower_prefix = prefix.lower()
        is_embed = not lower_prefix.startswith("<a")
        rewritten = rewrite_target(
            raw_target,
            is_embed=is_embed,
            source_file=source_file,
            staged_file=staged_file,
            source_root=source_root,
            staged_root=staged_root,
            excluded_dirs=excluded_dirs,
            repo_url=repo_url,
            ref=ref,
        )
        return f"{prefix}{rewritten}{suffix}"

    rewritten = MARKDOWN_LINK_PATTERN.sub(markdown_replacer, text)
    rewritten = HTML_ATTR_PATTERN.sub(html_replacer, rewritten)
    return rewritten


def copy_docs_tree(
    *,
    source_root: Path,
    output_root: Path,
    excluded_dirs: set[str],
    repo_url: str | None,
    ref: str | None,
) -> tuple[int, int]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    rewritten_markdown = 0

    for source_file in source_root.rglob("*"):
        if source_file.is_dir():
            continue
        relative_path = source_file.relative_to(source_root)
        if should_exclude(relative_path, excluded_dirs):
            continue

        destination_file = output_root / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        if source_file.suffix.lower() in {".md", ".markdown"}:
            text = source_file.read_text(encoding="utf-8")
            rewritten = rewrite_markdown(
                text,
                source_file=source_file,
                staged_file=destination_file,
                source_root=source_root,
                staged_root=output_root,
                excluded_dirs=excluded_dirs,
                repo_url=repo_url,
                ref=ref,
            )
            destination_file.write_text(rewritten, encoding="utf-8")
            rewritten_markdown += 1
        else:
            shutil.copy2(source_file, destination_file)
        copied_files += 1

    return copied_files, rewritten_markdown


def default_repo_url() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL", "").rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip("/")
    if server and repository and REMOTE_URL_PATTERN.match(server):
        return f"{server}/{repository}"
    return None


def default_ref() -> str | None:
    return os.environ.get("GITHUB_SHA")


def parse_excluded_dirs(value: Sequence[str]) -> set[str]:
    excluded: set[str] = set()
    for item in value:
        for chunk in item.split(","):
            normalized = chunk.strip()
            if normalized:
                excluded.add(normalized)
    return excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage docs/ into a publishable MkDocs docs_dir.")
    parser.add_argument("--source", default="docs", help="Source docs directory (relative to repo root).")
    parser.add_argument(
        "--output",
        default=".site/public-docs",
        help="Output directory for staged public docs (relative to repo root).",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=list(DEFAULT_EXCLUDED_DIRS),
        help="Top-level docs directory name to exclude. May be repeated or comma-separated.",
    )
    parser.add_argument("--repo-url", default=default_repo_url(), help="GitHub repository URL for rewritten links.")
    parser.add_argument("--ref", default=default_ref(), help="Git ref/SHA used for rewritten GitHub links.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_root = (REPO_ROOT / args.source).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    excluded_dirs = parse_excluded_dirs(args.exclude_dir)

    if not source_root.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_root}")

    copied_files, rewritten_markdown = copy_docs_tree(
        source_root=source_root,
        output_root=output_root,
        excluded_dirs=excluded_dirs,
        repo_url=args.repo_url,
        ref=args.ref,
    )

    print(f"[docs-site] Source: {source_root}")
    print(f"[docs-site] Output: {output_root}")
    print(f"[docs-site] Excluded top-level directories: {sorted(excluded_dirs)}")
    if args.repo_url and args.ref:
        print(f"[docs-site] Out-of-scope links rewritten to: {args.repo_url}@{args.ref}")
    else:
        print("[docs-site] Out-of-scope links kept as-is (repo URL/ref not configured).")
    print(f"[docs-site] Copied files: {copied_files} (rewritten markdown files: {rewritten_markdown})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
