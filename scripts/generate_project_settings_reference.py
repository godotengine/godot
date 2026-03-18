#!/usr/bin/env python3
"""Generate a ProjectSettings reference for the Gaussian Splatting module."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reference" / "project-settings.md"
SOURCE_ROOT = ROOT / "modules" / "gaussian_splatting"

PATTERN = re.compile(r'GLOBAL_DEF\(\"(?P<key>rendering/gaussian_splatting[^\"]*)\",\s*(?P<value>[^;]+)\);')
KEY_LITERAL_PATTERN = re.compile(r'"(?P<key>rendering/gaussian_splatting[^"]*)"')

GROUP_ORDER = [
    "core",
    "world",
    "import",
    "quality",
    "streaming",
    "culling",
    "cull",
    "lod",
    "sorting",
    "debug",
    "logging",
    "diagnostics",
    "other",
]

KNOWN_GROUPS = set(GROUP_ORDER)


@dataclass
class SettingEntry:
    key: str
    value: str
    path: Path
    line: int

    @property
    def group(self) -> str:
        parts = self.key.split("/")
        if len(parts) < 3:
            return "core"
        if len(parts) == 3:
            return "core"
        group = parts[2]
        return group if group in KNOWN_GROUPS else "other"


@dataclass
class KeyReference:
    key: str
    path: Path
    line: int


def collect_entries() -> list[SettingEntry]:
    entries: list[SettingEntry] = []
    for path in SOURCE_ROOT.rglob("*.cpp"):
        rel = path.relative_to(ROOT)
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = PATTERN.search(line)
            if not match:
                continue
            key = match.group("key").strip()
            value = match.group("value").strip().rstrip(";")
            entries.append(SettingEntry(key=key, value=value, path=rel, line=idx))
    entries.sort(key=lambda entry: entry.key)
    return entries


def collect_key_references() -> dict[str, list[KeyReference]]:
    references: dict[str, list[KeyReference]] = defaultdict(list)
    for path in SOURCE_ROOT.rglob("*"):
        if path.suffix not in {".cpp", ".h"}:
            continue
        rel = path.relative_to(ROOT)
        if "tests" in rel.parts:
            continue
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in KEY_LITERAL_PATTERN.finditer(line):
                key = match.group("key").strip()
                references[key].append(KeyReference(key=key, path=rel, line=idx))
    return references


def render_table(entries: list[SettingEntry]) -> list[str]:
    rows = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>Setting</th>",
        "      <th>Default</th>",
        "      <th>Defined In</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for entry in entries:
        rows.extend(
            [
                "    <tr>",
                f"      <td><pre><code>{entry.key}</code></pre></td>",
                f"      <td><pre><code>{entry.value}</code></pre></td>",
                f"      <td><pre><code>{entry.path}:{entry.line}</code></pre></td>",
                "    </tr>",
            ]
        )
    rows.extend(["  </tbody>", "</table>"])
    return rows


def render_runtime_only_table(runtime_only: list[tuple[str, KeyReference]]) -> list[str]:
    rows = [
        "| Setting | First reference |",
        "| --- | --- |",
    ]
    for key, reference in runtime_only:
        rows.append(f"| `{key}` | `{reference.path}:{reference.line}` |")
    return rows


def render_unconsumed_table(unconsumed: list[SettingEntry]) -> list[str]:
    rows = [
        "| Setting | Registered in |",
        "| --- | --- |",
    ]
    for entry in unconsumed:
        rows.append(f"| `{entry.key}` | `{entry.path}:{entry.line}` |")
    return rows


def has_non_registration_reference(entry: SettingEntry, references: dict[str, list[KeyReference]]) -> bool:
    for reference in references.get(entry.key, []):
        if reference.path != entry.path or reference.line != entry.line:
            return True
    return False


def build_reference() -> str:
    entries = collect_entries()
    references = collect_key_references()
    grouped: dict[str, list[SettingEntry]] = {group: [] for group in GROUP_ORDER}
    for entry in entries:
        grouped.setdefault(entry.group, []).append(entry)

    registered_keys = {entry.key for entry in entries}
    runtime_only = [
        (key, references[key][0])
        for key in sorted(references.keys())
        if key not in registered_keys and not key.endswith("/")
    ]
    unconsumed = [entry for entry in entries if not has_non_registration_reference(entry, references)]

    sections: list[str] = [
        "# Project Settings Reference",
        "",
        f"Last generated: {date.today().isoformat()}",
        "",
        "## Purpose",
        "Use this reference to map Gaussian Splatting project setting keys to source definitions and runtime lookup paths.",
        "",
        "## Usage",
        "| Task | Action |",
        "| --- | --- |",
        "| Regenerate this reference | Run `python3 scripts/generate_project_settings_reference.py`. |",
        "| Audit key usage in module code | Run `rg -n \"rendering/gaussian_splatting/\" modules/gaussian_splatting --glob '*.{h,cpp}'`. |",
        "",
        "## API",
        "",
        "### Registered keys",
        "These settings are registered with `GLOBAL_DEF(...)` and grouped by key prefix.",
        "",
        "| Coverage | Count |",
        "| --- | ---: |",
        f"| Registered keys | {len(entries)} |",
        f"| Runtime-only keys | {len(runtime_only)} |",
        f"| Registered keys without additional literal lookup | {len(unconsumed)} |",
        "",
    ]

    for group in GROUP_ORDER:
        group_entries = grouped.get(group, [])
        if not group_entries:
            continue
        title = group.replace("_", " ").title()
        sections.extend([f"#### {title}", ""])
        sections.extend(render_table(group_entries))
        sections.append("")

    sections.extend(
        [
            "### Runtime-only keys",
            "These keys are used by module code but are not registered with `GLOBAL_DEF(...)`.",
            "",
        ]
    )
    if runtime_only:
        sections.extend(render_runtime_only_table(runtime_only))
    else:
        sections.append("No runtime-only keys found.")
    sections.append("")

    sections.extend(
        [
            "### Registered keys without additional literal lookup",
            "These registered keys have no additional string-literal references beyond their registration line.",
            "",
        ]
    )
    if unconsumed:
        sections.extend(render_unconsumed_table(unconsumed))
    else:
        sections.append("No registered keys without additional literal lookup found.")
    sections.append("")

    sections.extend(
        [
            "## Examples",
            "```bash",
            "python3 scripts/generate_project_settings_reference.py",
            "```",
            "",
            "```bash",
            "rg -n \"rendering/gaussian_splatting/\" modules/gaussian_splatting --glob '*.{h,cpp}'",
            "```",
            "",
            "## Troubleshooting",
            "| Issue | Cause | Fix |",
            "| --- | --- | --- |",
            "| A key exists in code but not under registered sections | The key is read at runtime without `GLOBAL_DEF(...)`. | Check the `Runtime-only keys` section and decide whether to register it. |",
            "| A registered key appears unconsumed | No extra string-literal lookup path is present in module runtime code. | Verify lookup paths or remove stale registration. |",
            "| Line references are stale | Source moved after docs were generated. | Regenerate this file after code changes. |",
        ]
    )

    return "\n".join(sections)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_reference() + "\n", encoding="utf-8")
    print(f"[docs] Wrote project settings reference to {OUTPUT}")


if __name__ == "__main__":
    main()
