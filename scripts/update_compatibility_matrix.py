#!/usr/bin/env python3
"""Generate the compatibility matrix from YAML sources."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "reference" / "compatibility_sources.yaml"
OUTPUT = ROOT / "docs" / "reference" / "compatibility-matrix.md"
EVIDENCE_LADDER_FIGURE = [
    '<figure markdown="1">',
    "![Diagram of the compatibility evidence ladder with current platform positions](../assets/images/compatibility-evidence-ladder.svg){ .gs-diagram }",
    "<figcaption>The matrix is a ladder, not a badge wall: each platform only claims the highest evidence state the repository can currently prove.</figcaption>",
    "</figure>",
]


def load_data() -> dict:
    if not DATA.exists():
        raise SystemExit(f"Compatibility data file missing: {DATA}")
    with DATA.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _format_cell(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    return text.replace("|", "\\|").replace("\n", "<br>")


def _format_list(values: object) -> str:
    if values is None:
        return "-"
    if isinstance(values, str):
        return _format_cell(values)
    if not isinstance(values, list):
        return _format_cell(values)

    items = [_format_cell(item) for item in values if _format_cell(item) != "-"]
    if not items:
        return "-"
    return "<br>".join(items)


def build_table(data: dict) -> str:
    status_levels = data.get("status_levels", {})
    platforms = data.get("platforms", {})
    tested_configurations = data.get("tested_configurations", [])

    rows = [
        "# Compatibility Matrix",
        "",
        f"Last generated: {date.today().isoformat()}",
        "",
        "## Purpose",
        "Use this matrix to track evidence-backed platform status for godotGS Gaussian Splatting.",
        "",
        "## Usage",
        "| Task | Action |",
        "| --- | --- |",
        "| Review current platform status | Read the `Current state`, `Public binaries`, and `Notes` columns in the platform table. |",
        "| Update compatibility evidence | Edit `docs/reference/compatibility_sources.yaml` and regenerate this file. |",
        "",
        *EVIDENCE_LADDER_FIGURE,
        "",
        "## Evidence Levels",
        "| Level | Meaning |",
        "| --- | --- |",
    ]

    for level, description in status_levels.items():
        rows.append(f"| `{_format_cell(level)}` | {_format_cell(description)} |")

    rows.extend(
        [
            "",
            "## Platform States",
            "The state shown for each platform is the strongest evidence level currently documented in this repository.",
            "",
            "| Platform | Current state | Public binaries | Evidence | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for platform, entry in platforms.items():
        rows.append(
            "| {platform} | `{state}` | {public_binaries} | {evidence} | {notes} |".format(
                platform=_format_cell(platform),
                state=_format_cell(entry.get("state", "unknown")),
                public_binaries=_format_cell(entry.get("public_binaries", "-")),
                evidence=_format_list(entry.get("evidence", [])),
                notes=_format_cell(entry.get("notes", "")),
            )
        )

    rows.extend(
        [
            "",
            "## Published Test Environments",
            "These rows record the most concrete public environment details currently available from repo-owned automation or published artifacts. Unknown OS, adapter, or driver values are called out explicitly instead of being inferred.",
            "",
            "| Platform | State | OS / image | GPU / adapter | Driver / runtime | Evidence | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for entry in tested_configurations:
        os_version = entry.get("os_version", entry.get("hardware", ""))
        gpu = entry.get("gpu", "-")
        driver_runtime = entry.get("driver_runtime", entry.get("driver_version", entry.get("software", "")))
        rows.append(
            "| {platform} | `{state}` | {os_version} | {gpu} | {driver_runtime} | {evidence} | {notes} |".format(
                platform=_format_cell(entry.get("platform", "")),
                state=_format_cell(entry.get("state", "unknown")),
                os_version=_format_cell(os_version),
                gpu=_format_cell(gpu),
                driver_runtime=_format_cell(driver_runtime),
                evidence=_format_list(entry.get("evidence", [])),
                notes=_format_cell(entry.get("notes", "")),
            )
        )

    if not tested_configurations:
        rows.append("| - | - | - | - | - | - | No published environment rows yet. |")

    rows.extend(
        [
            "",
            "## Reading the Matrix",
            "- `build-supported` means the build system accepts the platform and repo automation can compile it.",
            "- `smoke-tested` means a minimal runtime or QA lane has passed, but not necessarily an interactive editor lane.",
            "- `editor-tested` means a non-headless editor/runtime lane has passed.",
            "- `sample-project-tested` and `production-tested` are reserved for stronger published evidence than this repo currently exposes publicly.",
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
            "| A platform only reaches `build-supported` | The repo has no documented runtime or editor evidence for it yet. | Add stronger evidence to `docs/reference/compatibility_sources.yaml` only after the lane or test result exists. |",
            "| OS, adapter, or driver fields are still generic | The lane exists, but those identifiers are not yet published in repo-owned evidence. | Capture them from the evidence run and replace the placeholder text in `docs/reference/compatibility_sources.yaml`. |",
            "| Missing platform row | Platform key was removed from the YAML source. | Re-add the platform entry in `docs/reference/compatibility_sources.yaml`. |",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    data = load_data()
    OUTPUT.write_text(build_table(data) + "\n", encoding="utf-8")
    print(f"[compatibility] Wrote matrix to {OUTPUT}")


if __name__ == "__main__":
    main()
