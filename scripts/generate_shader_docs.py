#!/usr/bin/env python3
"""Generate shader documentation from GLSL comments."""
from __future__ import annotations

from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "api" / "shader_reference.md"
SHADER_ROOTS = [
    ROOT / "modules" / "gaussian_splatting" / "shaders",
    ROOT / "modules" / "gaussian_splatting" / "compute",
]
KEYWORDS = {"if", "for", "while", "switch", "return"}

# Characters that form decorative banner/separator lines in comments.
_BANNER_CHARS = set("=-*#~+")


def _is_banner_line(text: str) -> bool:
    """Return True if *text* is a decorative separator/banner comment.

    A banner line consists primarily of repeated punctuation characters
    (``=``, ``-``, ``*``, ``#``, ``~``, ``+``) with optional whitespace.
    Short runs (fewer than 4 punctuation characters) are not treated as
    banners so that legitimate comments like ``// --flag`` survive.
    """
    stripped = text.strip()
    if not stripped:
        return False
    punct_count = sum(1 for ch in stripped if ch in _BANNER_CHARS)
    # Consider it a banner when the non-whitespace content is almost entirely
    # punctuation and there are at least 4 punctuation characters.
    non_ws = stripped.replace(" ", "")
    if not non_ws:
        return False
    return punct_count >= 4 and punct_count / len(non_ws) >= 0.75


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
            if len(parts) == 1:  # no explicit return type, likely macro
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


def _render_function_table(entries: list[tuple[str, list[str]]]) -> list[str]:
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
    for signature, comments in entries:
        description = " ".join(comment for comment in comments if comment).strip()
        if not description:
            description = "Undocumented."
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
        desc = description if description else "Undocumented."
        rows.extend(
            [
                "    <tr>",
                f"      <td><pre><code>{field_name}</code></pre></td>",
                f"      <td><pre><code>{field_type}</code></pre></td>",
                f"      <td>{desc}</td>",
                "    </tr>",
            ]
        )
    rows.extend(["  </tbody>", "</table>"])
    return rows


def build_reference() -> str:
    sections = [
        "# Shader Reference",
        "",
        f"Last generated: {date.today().isoformat()}",
        "",
    ]
    for shader in iter_shader_files():
        entries = parse_shader(shader)
        uniform_blocks = parse_uniform_blocks(shader)
        if not entries and not uniform_blocks:
            continue
        sections.append("## Shader")
        sections.append("")
        sections.append("```")
        sections.append(str(shader.relative_to(ROOT)))
        sections.append("```")
        sections.append("")
        if entries:
            sections.append("### Functions")
            sections.append("")
            sections.extend(_render_function_table(entries))
            sections.append("")
        if uniform_blocks:
            sections.append("### Uniform Blocks")
            sections.append("")
            for block_name, instance_name, fields in uniform_blocks:
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
    sections.extend(
        [
            "Generated by:",
            "",
            "```",
            "scripts/generate_shader_docs.py",
            "```",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_reference(), encoding="utf-8")
    print(f"[docs] Wrote shader reference to {OUTPUT}")


if __name__ == "__main__":
    main()
