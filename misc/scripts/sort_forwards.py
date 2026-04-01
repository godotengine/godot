#!/usr/bin/env python3
from __future__ import annotations

if __name__ != "__main__":
    raise ImportError(f'Utility script "{__file__}" should not be used as a module!')

import os
from pathlib import Path

if Path(os.getcwd()).as_posix() != (ROOT := Path(__file__).parent.parent.parent).as_posix():
    raise RuntimeError(f'Utility script "{__file__}" must be run from the repository root!')

import argparse
import re
from dataclasses import dataclass, field

RE_CLASS_FORWARD = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*(?://.*)?$")
RE_STRUCT_FORWARD = re.compile(r"^struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*(?://.*)?$")
RE_TEMPLATE = re.compile(r"^template\b")
RE_IFDEF = re.compile(r"^#\s*ifdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
RE_IFNDEF = re.compile(r"^#\s*ifndef\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
RE_IF_DEFINED = re.compile(
    r"^#\s*if\s+defined\s*(?:\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)|\s+([A-Za-z_][A-Za-z0-9_]*))\s*$"
)
RE_IF_NOT_DEFINED = re.compile(
    r"^#\s*if\s*!\s*defined\s*(?:\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)|\s+([A-Za-z_][A-Za-z0-9_]*))\s*$"
)
RE_IF = re.compile(r"^#\s*if\b")
RE_ENDIF = re.compile(r"^#\s*endif\b")
RE_ELSE_OR_ELIF = re.compile(r"^#\s*(?:else|elif)\b")
RE_PREPROCESSOR_ALLOWED = re.compile(r"^#\s*(?:if|ifdef|ifndef|else|elif|endif)\b")


def _strip_newline(line: str) -> str:
    return line.rstrip("\r\n")


def _is_class_forward_line(line: str) -> bool:
    return RE_CLASS_FORWARD.match(_strip_newline(line).strip()) is not None


def _is_struct_forward_line(line: str) -> bool:
    return RE_STRUCT_FORWARD.match(_strip_newline(line).strip()) is not None


def _is_forward_decl_line(line: str) -> bool:
    return _is_class_forward_line(line) or _is_struct_forward_line(line)


def _is_sortable_directive_line(line: str) -> bool:
    return RE_PREPROCESSOR_ALLOWED.match(_strip_newline(line).strip()) is not None


def _is_blank_line(line: str) -> bool:
    return _strip_newline(line).strip() == ""


def _is_allowed_block_line(line: str) -> bool:
    stripped = _strip_newline(line).strip()
    if RE_TEMPLATE.match(stripped):
        return True
    if _is_forward_decl_line(line):
        return True
    return _is_sortable_directive_line(line)


def _blank_has_following_allowed_line(lines: list[str], index: int) -> bool:
    i = index + 1
    while i < len(lines) and _is_blank_line(lines[i]):
        i += 1
    return i < len(lines) and _is_allowed_block_line(lines[i])


def _extract_macro_from_if(line: str) -> str | None:
    stripped = _strip_newline(line).strip()
    if m := RE_IFDEF.match(stripped):
        return m.group(1)
    if m := RE_IFNDEF.match(stripped):
        return m.group(1)
    if m := RE_IF_DEFINED.match(stripped):
        return m.group(1) or m.group(2)
    if m := RE_IF_NOT_DEFINED.match(stripped):
        return m.group(1) or m.group(2)
    return None


@dataclass
class Item:
    key: str
    lines: list[str]
    kind: str
    order: int = field(default=0)


def _extract_decl_name_and_kind(line: str) -> tuple[str, str] | None:
    stripped = _strip_newline(line).strip()
    if m := RE_CLASS_FORWARD.match(stripped):
        return m.group(1), "class"
    if m := RE_STRUCT_FORWARD.match(stripped):
        return m.group(1), "struct"
    return None


def _classify_guard_item_kind(lines: list[str]) -> str:
    has_class = any(_is_class_forward_line(line) for line in lines)
    has_struct = any(_is_struct_forward_line(line) for line in lines)
    if has_struct and not has_class:
        return "guard_struct"
    return "guard"


def _parse_decl_item(lines: list[str], i: int) -> tuple[Item, int] | None:
    stripped = _strip_newline(lines[i]).strip()
    if RE_TEMPLATE.match(stripped):
        if i + 1 >= len(lines):
            return None
        parsed_decl = _extract_decl_name_and_kind(lines[i + 1])
        if parsed_decl is None:
            return None
        decl_name, decl_kind = parsed_decl
        kind = "decl_template" if decl_kind == "class" else "decl_template_struct"
        return Item(key=decl_name.lower(), lines=[lines[i], lines[i + 1]], kind=kind), i + 2

    parsed_decl = _extract_decl_name_and_kind(lines[i])
    if parsed_decl is None:
        return None
    decl_name, decl_kind = parsed_decl
    kind = "decl" if decl_kind == "class" else "decl_struct"
    return Item(key=decl_name.lower(), lines=[lines[i]], kind=kind), i + 1


def _find_matching_endif(lines: list[str], start: int) -> tuple[int, bool] | None:
    depth = 0
    has_else_or_elif = False
    for i in range(start, len(lines)):
        stripped = _strip_newline(lines[i]).strip()
        if RE_IF.match(stripped) or RE_IFDEF.match(stripped) or RE_IFNDEF.match(stripped):
            depth += 1
        elif RE_ELSE_OR_ELIF.match(stripped) and depth == 1:
            has_else_or_elif = True
        elif RE_ENDIF.match(stripped):
            depth -= 1
            if depth == 0:
                return i, has_else_or_elif
    return None


def _sort_items(items: list[Item]) -> list[Item]:
    return sorted(items, key=lambda item: (item.key, item.order))


def _sort_forward_block_lines(lines: list[str]) -> list[str] | None:
    lines = [line for line in lines if not _is_blank_line(line)]

    items: list[Item] = []
    i = 0
    order = 0

    while i < len(lines):
        line = lines[i]

        parsed_decl = _parse_decl_item(lines, i)
        if parsed_decl is not None:
            decl_item, i = parsed_decl
            decl_item.order = order
            items.append(decl_item)
            order += 1
            continue

        if _extract_macro_from_if(line) is not None:
            match_info = _find_matching_endif(lines, i)
            if match_info is None:
                return None
            end, has_else_or_elif = match_info
            block_lines = lines[i : end + 1]

            if has_else_or_elif:
                macro = _extract_macro_from_if(line)
                if macro is None:
                    return None
                item = Item(
                    key=macro.lower(),
                    lines=block_lines,
                    kind=_classify_guard_item_kind(block_lines),
                    order=order,
                )
                items.append(item)
                order += 1
                i = end + 1
                continue

            inner_sorted = _sort_forward_block_lines(block_lines[1:-1])
            if inner_sorted is None:
                return None

            macro = _extract_macro_from_if(line)
            if macro is None:
                return None

            rebuilt = [block_lines[0], *inner_sorted, block_lines[-1]]
            item = Item(
                key=macro.lower(),
                lines=rebuilt,
                kind=_classify_guard_item_kind(rebuilt),
                order=order,
            )
            items.append(item)
            order += 1
            i = end + 1
            continue

        return None

    if not items:
        return lines

    basic_items = [item for item in items if item.kind == "decl"]
    struct_items = [item for item in items if item.kind == "decl_struct"]
    template_items = [item for item in items if item.kind == "decl_template"]
    template_struct_items = [item for item in items if item.kind == "decl_template_struct"]
    guard_items = [item for item in items if item.kind == "guard"]
    guard_struct_items = [item for item in items if item.kind == "guard_struct"]

    sorted_basic_items = _sort_items(basic_items)
    sorted_struct_items = _sort_items(struct_items)
    sorted_template_items = _sort_items(template_items)
    sorted_template_struct_items = _sort_items(template_struct_items)
    sorted_guard_items = _sort_items(guard_items)
    sorted_guard_struct_items = _sort_items(guard_struct_items)

    output: list[str] = []

    groups = [
        sorted_basic_items,
        sorted_struct_items,
        sorted_template_items,
        sorted_template_struct_items,
        sorted_guard_items,
        sorted_guard_struct_items,
    ]
    wrote_group = False
    for group in groups:
        if not group:
            continue
        if wrote_group:
            output.append("\n")
        for item in group:
            for line in item.lines:
                output.append(line)
            if item != group[-1] and item.kind in {"guard", "guard_struct"}:
                output.append("\n")
        wrote_group = True

    return output


def _sort_forward_blocks(content: str) -> str:
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    i = 0

    while i < len(lines):
        if not _is_allowed_block_line(lines[i]):
            out.append(lines[i])
            i += 1
            continue

        start = i
        has_forward_decl = False
        while i < len(lines):
            if _is_allowed_block_line(lines[i]):
                has_forward_decl = has_forward_decl or _is_forward_decl_line(lines[i])
                i += 1
                continue

            if _is_blank_line(lines[i]) and _blank_has_following_allowed_line(lines, i):
                i += 1
                continue

            break

        block = lines[start:i]
        if not has_forward_decl:
            out.extend(block)
            continue

        sorted_block = _sort_forward_block_lines(block)
        if sorted_block is None:
            out.extend(block)
            continue

        out.extend(sorted_block)

    return "".join(out)


def sort_forwards(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = _sort_forward_blocks(original)
    if updated == original:
        return False

    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(updated)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sort C++ forward class declarations and #ifdef groups")
    parser.add_argument("files", nargs="+", help="A list of files to sort")
    args = parser.parse_args()

    changed = 0
    for file in map(Path, args.files):
        changed += int(sort_forwards(file))

    return changed


try:
    raise SystemExit(main())
except KeyboardInterrupt:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)
