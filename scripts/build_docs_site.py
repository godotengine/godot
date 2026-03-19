#!/usr/bin/env python3
"""Build the public MkDocs site after generating and staging docs artifacts."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print(f"[docs-site] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned docs site artifacts.")
    parser.add_argument("--source", default="docs", help="Source docs directory.")
    parser.add_argument("--staging-dir", default=".site/public-docs", help="Public docs staging directory.")
    parser.add_argument("--site-dir", default=".site/site", help="MkDocs output directory.")
    parser.add_argument("--repo-url", default=os.environ.get("DOCS_REPO_URL"), help="Repository URL for staged links.")
    parser.add_argument("--ref", default=os.environ.get("GITHUB_SHA"), help="Ref/SHA for staged links.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip docs artifact generation step.")
    parser.add_argument("--skip-stage", action="store_true", help="Skip docs staging step.")
    parser.add_argument("--strict", action="store_true", help="Enable mkdocs --strict.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_generate:
        code = run([sys.executable, "scripts/build_documentation.py", "--all"])
        if code:
            return code

    if not args.skip_stage:
        stage_cmd = [
            sys.executable,
            "scripts/stage_public_docs.py",
            "--source",
            args.source,
            "--output",
            args.staging_dir,
        ]
        if args.repo_url:
            stage_cmd.extend(["--repo-url", args.repo_url])
        if args.ref:
            stage_cmd.extend(["--ref", args.ref])
        code = run(stage_cmd)
        if code:
            return code

    env = dict(os.environ)
    if args.repo_url:
        env["DOCS_REPO_URL"] = args.repo_url

    mkdocs_cmd = ["mkdocs", "build", "--config-file", "mkdocs.yml", "--site-dir", args.site_dir]
    if args.strict:
        mkdocs_cmd.append("--strict")

    return run(mkdocs_cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
