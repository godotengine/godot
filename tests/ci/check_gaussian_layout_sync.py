#!/usr/bin/env python3
"""Validate PackedGaussian layout parity between host C++ and shader mirrors."""

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
EMBEDDED_SHADER_MIRROR_FILES = (
    ROOT / "modules" / "gaussian_splatting" / "interfaces" / "gpu_sorting_pipeline.cpp",
)
TARGET_STRUCT_NAMES = ("PackedGaussian", "Gaussian", "GaussianQuantized")


@dataclass(frozen=True)
class RawField:
    type_name: str
    name: str
    count: int | None = None


@dataclass(frozen=True)
class StructDef:
    name: str
    fields: tuple[RawField, ...]
    alignas: int | None = None


@dataclass(frozen=True)
class FlatField:
    name: str
    base_type: str
    components: int


@dataclass(frozen=True)
class LayoutSpec:
    fields: tuple[FlatField, ...]
    offsets: dict[str, int]
    size: int
    alignment: int


_HOST_OFFSET_RE = re.compile(r"static_assert\(offsetof\(PackedGaussian,\s*(\w+)\) == (\d+)")
_FIELD_RE = re.compile(r"^\s*(?P<type>\w+)\s+(?P<name>\w+)(?:\[(?P<count>[A-Za-z_]\w*|\d+)\])?\s*;\s*$")
_CONST_RE = re.compile(r"^static constexpr \w+\s+(?P<name>\w+)\s*=\s*(?P<value>\d+)[uU]?\s*;\s*$")
_SCALAR_BASE_TYPES: dict[str, tuple[str, int, int]] = {
    "float": ("float", 4, 4),
    "uint": ("uint", 4, 4),
    "uint32_t": ("uint", 4, 4),
    "uint16_t": ("uint", 2, 2),
    "int": ("int", 4, 4),
    "int32_t": ("int", 4, 4),
}
_STD430_VECTOR_TYPES: dict[str, tuple[str, int, int]] = {
    "vec2": ("float", 8, 8),
    "vec3": ("float", 16, 12),
    "vec4": ("float", 16, 16),
    "uvec2": ("uint", 8, 8),
    "uvec3": ("uint", 16, 12),
    "uvec4": ("uint", 16, 16),
}
_VECTOR_COMPONENT_COUNTS = {
    "vec2": 2,
    "vec3": 3,
    "vec4": 4,
    "uvec2": 2,
    "uvec3": 3,
    "uvec4": 4,
}
_STRUCT_RE_TEMPLATE = r"struct(?:\s+alignas\((?P<align>\d+)\))?\s+{name}\s*\{{(?P<body>.*?)\}};"


def _struct_pattern(name: str) -> re.Pattern[str]:
    return re.compile(_STRUCT_RE_TEMPLATE.format(name=re.escape(name)), re.DOTALL)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _parse_host_contracts(path: Path) -> tuple[dict[str, int], int]:
    text = path.read_text(encoding="utf-8")
    offsets = {match.group(1): int(match.group(2)) for match in _HOST_OFFSET_RE.finditer(text)}
    size_match = re.search(r"static_assert\(sizeof\(PackedGaussian\) == (\d+)", text)
    if not size_match:
        raise RuntimeError(f"Missing PackedGaussian sizeof contract in {path}")
    return offsets, int(size_match.group(1))


def _parse_host_size_contract(path: Path, struct_name: str) -> int:
    text = path.read_text(encoding="utf-8")
    size_match = re.search(rf"static_assert\(sizeof\({re.escape(struct_name)}\) == (\d+)", text)
    if not size_match:
        raise RuntimeError(f"Missing {struct_name} sizeof contract in {path}")
    return int(size_match.group(1))


def _parse_struct_definition(path: Path, struct_name: str) -> StructDef:
    text = path.read_text(encoding="utf-8")
    match = _struct_pattern(struct_name).search(text)
    if not match:
        raise RuntimeError(f"Could not find `struct {struct_name}` in {path}")

    fields: list[RawField] = []
    constants: dict[str, int] = {}
    for raw_line in match.group("body").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        const_match = _CONST_RE.match(line)
        if const_match:
            constants[const_match.group("name")] = int(const_match.group("value"))
            continue
        if line.startswith(("static_assert", "using ", "typedef ", "friend ", "void ", "#")):
            continue
        if "(" in line:
            continue
        field_match = _FIELD_RE.match(line)
        if not field_match:
            raise RuntimeError(f"Unsupported field syntax in {path}: {raw_line.strip()}")
        count = field_match.group("count")
        if count is None:
            count_value = None
        elif count.isdigit():
            count_value = int(count)
        elif count in constants:
            count_value = constants[count]
        else:
            raise RuntimeError(f"Unknown array bound `{count}` in {path}: {raw_line.strip()}")
        fields.append(RawField(field_match.group("type"), field_match.group("name"), count_value))
    return StructDef(struct_name, tuple(fields), int(match.group("align")) if match.group("align") else None)


def _discover_shader_struct_sources() -> tuple[tuple[Path, str], ...]:
    discovered: list[tuple[Path, str]] = []
    seen: set[tuple[Path, str]] = set()

    for root in SHADER_ROOTS:
        for path in sorted(root.rglob("*.glsl")):
            text = path.read_text(encoding="utf-8")
            for struct_name in TARGET_STRUCT_NAMES:
                if _struct_pattern(struct_name).search(text):
                    key = (path, struct_name)
                    if key not in seen:
                        seen.add(key)
                        discovered.append(key)

    for path in EMBEDDED_SHADER_MIRROR_FILES:
        text = path.read_text(encoding="utf-8")
        for struct_name in TARGET_STRUCT_NAMES:
            if _struct_pattern(struct_name).search(text):
                key = (path, struct_name)
                if key not in seen:
                    seen.add(key)
                    discovered.append(key)

    if not discovered:
        raise RuntimeError("Could not find any shader mirrors declaring `struct Gaussian` or `struct PackedGaussian`")
    return tuple(discovered)


def _normalized_type_signature(field: RawField) -> tuple[str, int]:
    if field.count is not None:
        if field.type_name not in _SCALAR_BASE_TYPES:
            raise RuntimeError(f"Unsupported array type `{field.type_name}[{field.count}]`")
        base_type, _, _ = _SCALAR_BASE_TYPES[field.type_name]
        return base_type, field.count

    if field.type_name in _SCALAR_BASE_TYPES:
        base_type, _, _ = _SCALAR_BASE_TYPES[field.type_name]
        return base_type, 1
    if field.type_name in _STD430_VECTOR_TYPES:
        base_type, _, _ = _STD430_VECTOR_TYPES[field.type_name]
        components = _VECTOR_COMPONENT_COUNTS[field.type_name]
        return base_type, components
    raise RuntimeError(f"Unsupported field type `{field.type_name}`")


def _field_layout(field: RawField, mode: str) -> tuple[int, int]:
    if field.count is not None:
        if field.type_name not in _SCALAR_BASE_TYPES:
            raise RuntimeError(f"Unsupported array type `{field.type_name}[{field.count}]`")
        _, alignment, element_size = _SCALAR_BASE_TYPES[field.type_name]
        return alignment, element_size * field.count

    if field.type_name in _SCALAR_BASE_TYPES:
        _, alignment, size = _SCALAR_BASE_TYPES[field.type_name]
        return alignment, size
    if mode == "shader" and field.type_name in _STD430_VECTOR_TYPES:
        _, alignment, size = _STD430_VECTOR_TYPES[field.type_name]
        return alignment, size
    raise RuntimeError(f"Unsupported {'std430' if mode == 'shader' else 'C++'} field type `{field.type_name}`")


def _layout_struct(struct_definitions: dict[str, StructDef], struct_name: str, mode: str, prefix: str = "") -> LayoutSpec:
    struct_def = struct_definitions[struct_name]
    offset = 0
    max_alignment = struct_def.alignas or 1
    fields: list[FlatField] = []
    offsets: dict[str, int] = {}

    for field in struct_def.fields:
        if field.count is None and field.type_name in struct_definitions:
            child = _layout_struct(struct_definitions, field.type_name, mode, prefix + field.name + "_")
            offset = _round_up(offset, child.alignment)
            for child_field in child.fields:
                fields.append(child_field)
            for child_name, child_offset in child.offsets.items():
                offsets[child_name] = offset + child_offset
            offset += child.size
            max_alignment = max(max_alignment, child.alignment)
            continue

        alignment, size = _field_layout(field, mode)
        offset = _round_up(offset, alignment)
        base_type, components = _normalized_type_signature(field)
        flat_name = prefix + field.name
        fields.append(FlatField(flat_name, base_type, components))
        offsets[flat_name] = offset
        offset += size
        max_alignment = max(max_alignment, alignment)

    size = _round_up(offset, max_alignment)
    return LayoutSpec(tuple(fields), offsets, size, max_alignment)


def _format_signature(base_type: str, components: int) -> str:
    return base_type if components == 1 else f"{base_type}[{components}]"


def _build_quantized_expected_layout(host_structs: dict[str, StructDef]) -> LayoutSpec:
    host_layout = _layout_struct(host_structs, "PackedGaussianQuantized", "host")
    fields = (
        FlatField("position_chunk", "uint", 2),
        FlatField("opacity", "float", 1),
        FlatField("scale_area_lo", "uint", 1),
        FlatField("scale_area_hi", "uint", 1),
        FlatField("rotation_lo", "uint", 1),
        FlatField("rotation_hi", "uint", 1),
        FlatField("_padding", "uint", 1),
        FlatField("sh_dc", "float", 4),
        FlatField("sh_encoded_01", "uint", 2),
        FlatField("sh_encoded_23", "uint", 2),
        FlatField("sh_encoded_45", "uint", 2),
        FlatField("normal_xy", "uint", 1),
        FlatField("normal_z_stroke", "uint", 1),
    )
    offsets = {
        "position_chunk": host_layout.offsets["quantized_position"],
        "opacity": host_layout.offsets["opacity"],
        "scale_area_lo": host_layout.offsets["quantized_scale"],
        "scale_area_hi": host_layout.offsets["quantized_scale"] + 4,
        "rotation_lo": host_layout.offsets["rotation"],
        "rotation_hi": host_layout.offsets["rotation"] + 4,
        "_padding": host_layout.offsets["_pre_sh_padding"],
        "sh_dc": host_layout.offsets["sh_dc"],
        "sh_encoded_01": host_layout.offsets["sh_encoded"],
        "sh_encoded_23": host_layout.offsets["sh_encoded"] + 8,
        "sh_encoded_45": host_layout.offsets["sh_encoded"] + 16,
        "normal_xy": host_layout.offsets["normal_xy"],
        "normal_z_stroke": host_layout.offsets["normal_z_stroke"],
    }
    return LayoutSpec(fields, offsets, host_layout.size, host_layout.alignment)


def _compare_layouts(
    expected: LayoutSpec,
    actual: LayoutSpec,
    source_path: Path,
    source_struct_name: str,
    expected_label: str,
    failures: list[str],
) -> None:
    if len(actual.fields) != len(expected.fields):
        failures.append(
            f"{source_path.relative_to(ROOT)}: `struct {source_struct_name}` field count {len(actual.fields)} != expected {len(expected.fields)}"
        )

    for index, expected_field in enumerate(expected.fields):
        if index >= len(actual.fields):
            failures.append(
                f"{source_path.relative_to(ROOT)}: `struct {source_struct_name}` is missing field `{expected_field.name}` at index {index}"
            )
            continue

        actual_field = actual.fields[index]
        expected_signature = _format_signature(expected_field.base_type, expected_field.components)
        actual_signature = _format_signature(actual_field.base_type, actual_field.components)
        if actual_field.name != expected_field.name:
            failures.append(
                f"{source_path.relative_to(ROOT)}: field {index} name `{actual_field.name}` != expected `{expected_field.name}`"
            )
        if actual_signature != expected_signature:
            failures.append(
                f"{source_path.relative_to(ROOT)}: field `{actual_field.name}` type `{actual_signature}` != expected `{expected_signature}`"
            )

        expected_offset = expected.offsets[expected_field.name]
        actual_offset = actual.offsets.get(actual_field.name)
        if actual_offset != expected_offset:
            failures.append(
                f"{source_path.relative_to(ROOT)}: field `{actual_field.name}` offset {actual_offset} != expected {expected_offset}"
            )

    if actual.size != expected.size:
        failures.append(
            f"{source_path.relative_to(ROOT)}: struct size {actual.size} != expected {expected_label} size {expected.size}"
        )


def main() -> int:
    host_offsets, host_size = _parse_host_contracts(HOST_LAYOUT)
    packed_host_structs = {
        "PackedGaussian": _parse_struct_definition(HOST_LAYOUT, "PackedGaussian"),
        "PackedSphericalHarmonics": _parse_struct_definition(HOST_LAYOUT, "PackedSphericalHarmonics"),
    }
    packed_host_layout = _layout_struct(packed_host_structs, "PackedGaussian", "host")
    quantized_host_structs = {
        "PackedGaussianQuantized": _parse_struct_definition(HOST_LAYOUT, "PackedGaussianQuantized"),
    }
    quantized_host_layout = _layout_struct(quantized_host_structs, "PackedGaussianQuantized", "host")
    quantized_host_size = _parse_host_size_contract(HOST_LAYOUT, "PackedGaussianQuantized")

    failures: list[str] = []
    if packed_host_layout.size != host_size:
        failures.append(
            f"{HOST_LAYOUT.relative_to(ROOT)}: computed PackedGaussian size {packed_host_layout.size} != host contract {host_size}"
        )
    for contract_name, expected_offset in host_offsets.items():
        actual_name = "sh_dc" if contract_name == "sh" else contract_name
        actual_offset = packed_host_layout.offsets.get(actual_name)
        if actual_offset != expected_offset:
            failures.append(
                f"{HOST_LAYOUT.relative_to(ROOT)}: host field `{contract_name}` offset {actual_offset} != contract {expected_offset}"
            )
    if quantized_host_layout.size != quantized_host_size:
        failures.append(
            f"{HOST_LAYOUT.relative_to(ROOT)}: computed PackedGaussianQuantized size {quantized_host_layout.size} != host contract {quantized_host_size}"
        )

    quantized_expected_layout = _build_quantized_expected_layout(quantized_host_structs)

    for shader_path, struct_name in _discover_shader_struct_sources():
        shader_structs = {struct_name: _parse_struct_definition(shader_path, struct_name)}
        shader_layout = _layout_struct(shader_structs, struct_name, "shader")
        if struct_name in ("PackedGaussian", "Gaussian"):
            expected_layout = packed_host_layout
            expected_label = "PackedGaussian"
        elif struct_name == "GaussianQuantized":
            expected_layout = quantized_expected_layout
            expected_label = "PackedGaussianQuantized"
        else:
            raise RuntimeError(f"Unexpected discovered struct `{struct_name}`")
        _compare_layouts(expected_layout, shader_layout, shader_path, struct_name, expected_label, failures)

    if failures:
        for failure in failures:
            print(f"[gaussian-layout-check] FAIL {failure}")
        return 1

    print("[gaussian-layout-check] PASSED")
    print("[gaussian-layout-check] PackedGaussian and PackedGaussianQuantized host/mirror field signatures, offsets, and size are aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
