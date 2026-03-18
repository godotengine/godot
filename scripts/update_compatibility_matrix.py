#!/usr/bin/env python3
"""Generate the compatibility matrix from YAML sources."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "reference" / "compatibility_sources.yaml"
OUTPUT = ROOT / "docs" / "reference" / "compatibility-matrix.md"


def load_data() -> dict:
    if not DATA.exists():
        raise SystemExit(f"Compatibility data file missing: {DATA}")
    with DATA.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_table(platforms: dict) -> str:
    rows = [
        "# Compatibility Matrix",
        "",
        f"Last generated: {date.today().isoformat()}",
        "",
        "## Purpose",
        "Use this matrix to track validated platform compatibility status for Gaussian Splatting.",
        "",
        "## Usage",
        "| Task | Action |",
        "| --- | --- |",
        "| Review current platform status | Read the `Status` and `Notes` columns in the matrix. |",
        "| Update compatibility evidence | Edit `docs/reference/compatibility_sources.yaml` and regenerate this file. |",
        "",
        "## API",
        "Entries marked `unknown` have no validated test evidence in this repository.",
        "",
        "| Platform | Status | GPU | Driver | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for platform, entry in platforms.items():
        status = entry.get("status", "unknown")
        rows.append(
            f"| {platform} | {status} | {entry.get('gpu', '-')} | {entry.get('driver', '-')} | {entry.get('notes', '')} |"
        )

    rows.extend(
        [
            "",
            "## Examples",
            "```bash",
            "python3 scripts/update_compatibility_matrix.py",
            "```",
            "",
            "## Troubleshooting",
            "| Issue | Cause | Fix |",
            "| --- | --- | --- |",
            "| Matrix does not reflect YAML edits | The generator was not rerun. | Run `python3 scripts/update_compatibility_matrix.py`. |",
            "| A platform is still `unknown` | No validated evidence is documented. | Add evidence to `docs/reference/compatibility_sources.yaml` and regenerate. |",
            "| Missing platform row | Platform key was removed from the YAML source. | Re-add the platform entry in `docs/reference/compatibility_sources.yaml`. |",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    data = load_data()
    platforms = data.get("platforms", {})
    OUTPUT.write_text(build_table(platforms) + "\n", encoding="utf-8")
    print(f"[compatibility] Wrote matrix to {OUTPUT}")


if __name__ == "__main__":
    main()
