#!/usr/bin/env python3
"""Validate PackedGaussian layout parity between host C++ and GLSL shaders."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOST_LAYOUT = ROOT / "modules" / "gaussian_splatting" / "renderer" / "gaussian_gpu_layout.h"
SHADER_ROOTS = (
    ROOT / "modules" / "gaussian_splatting" / "shaders",
    ROOT / "modules" / "gaussian_splatting" / "compute",
)


@dataclass(frozen=True)
class Field:
    type_name: str
    name: str
    count: int | None = None


_HOST_OFFSET_RE = re.compile(r"static_assert\(offsetof\(PackedGaussian,\s*(\w+)\) == (\d+)")
_HOST_SIZE_RE = re.compile(r"static_assert\(sizeof\(PackedGaussian\) == (\d+)")
_SHADER_STRUCT_RE = re.compile(r"struct\s+Gaussian\s*\{(?P<body>.*?)\};", re.DOTALL)
_FIELD_RE = re.compile(r"^\s*(?P<type>\w+)\s+(?P<name>\w+)(?:\[(?P<count>\d+)\])?\s*;\s*$")


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _parse_host_contracts(path: Path) -> tuple[dict[str, int], int]:
    text = path.read_text(encoding="utf-8")
    offsets = {match.group(1): int(match.group(2)) for match in _HOST_OFFSET_RE.finditer(text)}
    size_match = _HOST_SIZE_RE.search(text)
    if not size_match:
        raise RuntimeError(f"Missing PackedGaussian sizeof contract in {path}")
    return offsets, int(size_match.group(1))


def _parse_shader_fields(path: Path) -> list[Field]:
    text = path.read_text(encoding="utf-8")
    match = _SHADER_STRUCT_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not find `struct Gaussian` in {path}")

    fields: list[Field] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        field_match = _FIELD_RE.match(line)
        if not field_match:
            raise RuntimeError(f"Unsupported field syntax in {path}: {raw_line.strip()}")
        count = field_match.group("count")
        fields.append(Field(field_match.group("type"), field_match.group("name"), int(count) if count else None))
    return fields


def _discover_shader_struct_files() -> tuple[Path, ...]:
    shader_files: list[Path] = []
    for root in SHADER_ROOTS:
        for path in sorted(root.rglob("*.glsl")):
            text = path.read_text(encoding="utf-8")
            if _SHADER_STRUCT_RE.search(text):
                shader_files.append(path)
    if not shader_files:
        raise RuntimeError("Could not find any GLSL files declaring `struct Gaussian`")
    return tuple(shader_files)


def _std430_type_layout(field: Field) -> tuple[int, int]:
    if field.count is not None:
        if field.type_name not in {"float", "uint"}:
            raise RuntimeError(f"Unsupported array type `{field.type_name}[{field.count}]`")
        return 4, 4 * field.count

    layouts = {
        "float": (4, 4),
        "uint": (4, 4),
        "vec2": (8, 8),
        "vec3": (16, 12),
        "vec4": (16, 16),
    }
    if field.type_name not in layouts:
        raise RuntimeError(f"Unsupported GLSL field type `{field.type_name}`")
    return layouts[field.type_name]


def _compute_shader_layout(fields: list[Field]) -> tuple[dict[str, int], int]:
    offsets: dict[str, int] = {}
    offset = 0
    max_alignment = 16

    for field in fields:
        alignment, size = _std430_type_layout(field)
        max_alignment = max(max_alignment, alignment)
        offset = _round_up(offset, alignment)
        offsets[field.name] = offset
        offset += size

    return offsets, _round_up(offset, max_alignment)


def _expected_shader_offsets(host_offsets: dict[str, int]) -> dict[str, int]:
    sh_offset = host_offsets["sh"]
    return {
        "position": host_offsets["position"],
        "opacity": host_offsets["opacity"],
        "scale": host_offsets["scale"],
        "area": host_offsets["area"],
        "rotation": host_offsets["rotation"],
        "sh_dc": sh_offset,
        "sh_encoded": sh_offset + 16,
        "normal": host_offsets["normal"],
        "stroke_age": host_offsets["stroke_age"],
        "brush_axes": host_offsets["brush_axes"],
        "painterly_meta": host_offsets["painterly_meta"],
        "sh_metadata": host_offsets["sh_metadata"],
    }


def main() -> int:
    host_offsets, host_size = _parse_host_contracts(HOST_LAYOUT)
    expected_shader_offsets = _expected_shader_offsets(host_offsets)
    shader_files = _discover_shader_struct_files()

    failures: list[str] = []
    for shader_path in shader_files:
        shader_offsets, shader_size = _compute_shader_layout(_parse_shader_fields(shader_path))
        for field_name, expected_offset in expected_shader_offsets.items():
            actual_offset = shader_offsets.get(field_name)
            if actual_offset != expected_offset:
                failures.append(
                    f"{shader_path.relative_to(ROOT)}: field `{field_name}` offset {actual_offset} != expected {expected_offset}"
                )
        if shader_size != host_size:
            failures.append(
                f"{shader_path.relative_to(ROOT)}: struct size {shader_size} != host PackedGaussian size {host_size}"
            )

    if failures:
        for failure in failures:
            print(f"[gaussian-layout-check] FAIL {failure}")
        return 1

    print("[gaussian-layout-check] PASSED")
    print("[gaussian-layout-check] PackedGaussian host/shader offsets are aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
