#!/usr/bin/env python3
"""Generate shader documentation from GLSL comments."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "api" / "shader_reference.md"
SHADER_ROOTS = [
    ROOT / "modules" / "gaussian_splatting" / "shaders",
    ROOT / "modules" / "gaussian_splatting" / "compute",
]
KEYWORDS = {"if", "for", "while", "switch", "return"}
# Characters that form decorative banner/separator lines in comments.
_BANNER_CHARS = set("=-*#~+")


@dataclass
class CoverageStats:
    documented_functions: int = 0
    undocumented_functions: int = 0
    documented_uniform_fields: int = 0
    undocumented_uniform_fields: int = 0


def _is_banner_line(text: str) -> bool:
    """Return True if *text* is a decorative separator/banner comment."""
    stripped = text.strip()
    if not stripped:
        return False
    punct_count = sum(1 for ch in stripped if ch in _BANNER_CHARS)
    non_ws = stripped.replace(" ", "")
    if not non_ws:
        return False
    return punct_count >= 4 and punct_count / len(non_ws) >= 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate shader Markdown reference docs.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT),
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--include-undocumented",
        action="store_true",
        help="Include undocumented functions/uniform fields in output tables.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when undocumented coverage exceeds thresholds.",
    )
    parser.add_argument(
        "--max-undocumented-functions",
        type=int,
        default=0,
        help="Allowed undocumented function entries when --strict is used.",
    )
    parser.add_argument(
        "--max-undocumented-fields",
        type=int,
        default=0,
        help="Allowed undocumented uniform fields when --strict is used.",
    )
    return parser.parse_args()


@dataclass
class CoverageStats:
    documented_functions: int = 0
    undocumented_functions: int = 0
    documented_uniform_fields: int = 0
    undocumented_uniform_fields: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate shader Markdown reference docs.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT),
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--include-undocumented",
        action="store_true",
        help="Include undocumented functions/uniform fields in output tables.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when undocumented coverage exceeds thresholds.",
    )
    parser.add_argument(
        "--max-undocumented-functions",
        type=int,
        default=0,
        help="Allowed undocumented function entries when --strict is used.",
    )
    parser.add_argument(
        "--max-undocumented-fields",
        type=int,
        default=0,
        help="Allowed undocumented uniform fields when --strict is used.",
    )
    return parser.parse_args()


def iter_shader_files() -> list[Path]:
    files: list[Path] = []
    for root in SHADER_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.glsl")))
    return files


def _consume_block_comment(line: str, buffer: list[str]) -> tuple[bool, list[str]]:
    stripped = line.strip()
    if stripped.startswith("/*"):
        content = stripped.lstrip("/* ").rstrip("*/ ").strip()
        if content and not _is_banner_line(content):
            buffer.append(content)
        return (not stripped.endswith("*/")), buffer
    return False, buffer


def parse_shader(path: Path) -> list[tuple[str, list[str]]]:
    docs: list[tuple[str, list[str]]] = []
    pending: list[str] = []
    in_block_comment = False
    block_buffer: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("/*"):
            in_block_comment, block_buffer = _consume_block_comment(line, block_buffer)
            if not in_block_comment:
                pending.extend(s for s in block_buffer if not _is_banner_line(s))
                block_buffer = []
            continue
        if in_block_comment:
            if stripped.endswith("*/"):
                block_buffer.append(stripped.rstrip("*/ ").strip())
                pending.extend(s for s in block_buffer if not _is_banner_line(s))
                block_buffer = []
                in_block_comment = False
            else:
                block_buffer.append(stripped)
            continue
        if stripped.startswith("//"):
            comment_text = stripped.lstrip("/ ")
            if not _is_banner_line(comment_text):
                pending.append(comment_text)
            continue

        if stripped.endswith("{") and "(" in stripped and not stripped.startswith("#"):
            header = stripped[:-1].strip()
            before_params, params_part = header.split("(", 1)
            params = params_part.rsplit(")", 1)[0].strip()
            parts = before_params.split()
            if not parts:
                pending = []
                continue
            name = parts[-1]
            if name in KEYWORDS:
                pending = []
                continue
            if len(parts) == 1:
                pending = []
                continue
            signature = f"{name}({params})" if params else f"{name}()"
            docs.append((signature, pending))
            pending = []
        elif stripped:
            pending = []
    return docs


def parse_uniform_blocks(path: Path) -> list[tuple[str, str | None, list[tuple[str, str, str]]]]:
    blocks: list[tuple[str, str | None, list[tuple[str, str, str]]]] = []
    pending: list[str] = []
    in_block_comment = False
    block_buffer: list[str] = []
    current_block: str | None = None
    current_instance: str | None = None
    current_fields: list[tuple[str, str, str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("/*"):
            in_block_comment, block_buffer = _consume_block_comment(line, block_buffer)
            if not in_block_comment:
                pending.extend(s for s in block_buffer if not _is_banner_line(s))
                block_buffer = []
            continue
        if in_block_comment:
            if stripped.endswith("*/"):
                block_buffer.append(stripped.rstrip("*/ ").strip())
                pending.extend(s for s in block_buffer if not _is_banner_line(s))
                block_buffer = []
                in_block_comment = False
            else:
                block_buffer.append(stripped)
            continue

        if current_block is None:
            if "uniform" in stripped and "{" in stripped and not stripped.startswith("#"):
                parts = stripped.split("uniform", 1)[1].split("{", 1)[0].strip().split()
                if parts:
                    current_block = parts[0]
                    current_instance = None
                    current_fields = []
                    pending = []
            continue

        if stripped.startswith("}"):
            tail = stripped.lstrip("}").strip().rstrip(";")
            if tail:
                current_instance = tail.split()[0]
            blocks.append((current_block, current_instance, current_fields))
            current_block = None
            current_instance = None
            current_fields = []
            pending = []
            continue

        if stripped.startswith("//"):
            comment_text = stripped.lstrip("/ ").strip()
            if not _is_banner_line(comment_text):
                pending.append(comment_text)
            continue

        if not stripped or stripped.startswith("#"):
            continue

        inline_comment = ""
        if "//" in line:
            code_part, inline_comment = line.split("//", 1)
            stripped = code_part.strip()
            inline_comment = inline_comment.strip()
        if not stripped:
            if inline_comment:
                pending.append(inline_comment)
            continue

        if not stripped.endswith(";"):
            continue
        declaration = stripped.rstrip(";").strip()
        if not declaration:
            continue
        tokens = declaration.split()
        if len(tokens) < 2:
            continue
        field_type = " ".join(tokens[:-1])
        field_name = tokens[-1]
        description = inline_comment or " ".join(comment for comment in pending if comment).strip()
        if not description:
            description = "(undocumented)"
        pending = []
        current_fields.append((field_name, field_type, description))

    return blocks


def _render_function_table(entries: list[tuple[str, str]]) -> list[str]:
    rows = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>Function</th>",
        "      <th>Description</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for signature, description in entries:
        rows.extend(
            [
                "    <tr>",
                f"      <td><pre><code>{signature}</code></pre></td>",
                f"      <td>{description}</td>",
                "    </tr>",
            ]
        )
    rows.extend(["  </tbody>", "</table>"])
    return rows


def _render_uniform_table(fields: list[tuple[str, str, str]]) -> list[str]:
    rows = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>Field</th>",
        "      <th>Type</th>",
        "      <th>Description</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for field_name, field_type, description in fields:
        rows.extend(
            [
                "    <tr>",
                f"      <td><pre><code>{field_name}</code></pre></td>",
                f"      <td><pre><code>{field_type}</code></pre></td>",
                f"      <td>{description}</td>",
                "    </tr>",
            ]
        )
    rows.extend(["  </tbody>", "</table>"])
    return rows


def build_reference(*, include_undocumented: bool) -> tuple[str, CoverageStats]:
    stats = CoverageStats()
    sections = [
        "# Shader Reference",
        "",
        f"Last generated: {date.today().isoformat()}",
        "",
    ]

    for shader in iter_shader_files():
        entries = parse_shader(shader)
        uniform_blocks = parse_uniform_blocks(shader)

        rendered_functions: list[tuple[str, str]] = []
        for signature, comments in entries:
            description = " ".join(comment for comment in comments if comment).strip()
            if description:
                stats.documented_functions += 1
                rendered_functions.append((signature, description))
                continue

            stats.undocumented_functions += 1
            if include_undocumented:
                rendered_functions.append((signature, "Missing shader comment."))

        rendered_blocks: list[tuple[str, str | None, list[tuple[str, str, str]]]] = []
        for block_name, instance_name, fields in uniform_blocks:
            rendered_fields: list[tuple[str, str, str]] = []
            for field_name, field_type, description in fields:
                normalized = description.strip()
                is_documented = bool(normalized) and normalized != "(undocumented)"
                if is_documented:
                    stats.documented_uniform_fields += 1
                    rendered_fields.append((field_name, field_type, normalized))
                    continue

                stats.undocumented_uniform_fields += 1
                if include_undocumented:
                    rendered_fields.append((field_name, field_type, "Missing shader field comment."))
            if rendered_fields:
                rendered_blocks.append((block_name, instance_name, rendered_fields))

        if not rendered_functions and not rendered_blocks:
            continue

        sections.append("## Shader")
        sections.append("")
        sections.append("```")
        sections.append(shader.relative_to(ROOT).as_posix())
        sections.append("```")
        sections.append("")
        if rendered_functions:
            sections.append("### Functions")
            sections.append("")
            sections.extend(_render_function_table(rendered_functions))
            sections.append("")
        if rendered_blocks:
            sections.append("### Uniform Blocks")
            sections.append("")
            for block_name, instance_name, fields in rendered_blocks:
                sections.append("#### Block")
                sections.append("")
                sections.append("```")
                if instance_name:
                    sections.append(f"{block_name} ({instance_name})")
                else:
                    sections.append(block_name)
                sections.append("```")
                sections.append("")
                sections.extend(_render_uniform_table(fields))
                sections.append("")
        sections.append("")

    sections.insert(4, f"Coverage summary: `{stats.documented_functions}` documented functions, `{stats.undocumented_functions}` undocumented functions, `{stats.documented_uniform_fields}` documented uniform fields, `{stats.undocumented_uniform_fields}` undocumented uniform fields.")
    sections.insert(5, "")
    if not include_undocumented:
        sections.insert(6, "Undocumented entries are omitted by default. Use `--include-undocumented` to list them.")
        sections.insert(7, "")

    sections.extend(
        [
            "Generated by:",
            "",
            "```",
            "scripts/generate_shader_docs.py",
            "```",
        ]
    )
    return "\n".join(sections).strip() + "\n", stats


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()

    content, stats = build_reference(include_undocumented=args.include_undocumented)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"[docs] Wrote shader reference to {output}")

    if args.strict:
        function_ok = stats.undocumented_functions <= args.max_undocumented_functions
        fields_ok = stats.undocumented_uniform_fields <= args.max_undocumented_fields
        if not function_ok or not fields_ok:
            print(
                "[docs] Strict coverage failed: "
                f"undocumented_functions={stats.undocumented_functions} "
                f"(allowed={args.max_undocumented_functions}), "
                f"undocumented_uniform_fields={stats.undocumented_uniform_fields} "
                f"(allowed={args.max_undocumented_fields})"
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
