#!/usr/bin/env python3
"""
js_generator.py

Generates export_signatures.js from extension_api.json — a static map of every
exported Godot WASM function to its JS-side type signature.

PURPOSE
───────
The JavaScript bridge (__callGodot) needs to know the parameter and return
kinds for each exported C++ function before it can marshal arguments correctly.
Rather than inferring types at call time, this generator bakes the signatures
into a single ES module that the bridge imports at startup.

OUTPUT FORMAT
─────────────
export const GODOT_EXPORT_SIGNATURES = {
  _ClassName_method_name: { params: ["ptr", "f32", "f32"], returns: "variant*" },
  ...
};

PARAMETER KINDS
───────────────
  ptr        – receiver object pointer (always the first param of every method)
  i32        – bool or integer struct field
  i64        – int, enum, or bitfield
  f32        – float or struct float field
  f64        – double (float return type only)
  cstring    – String / StringName / NodePath (UTF-8 pointer)
  variant*   – heap-allocated Variant* (all Godot struct / packed / object returns)
  varargs    – trailing varargs sentinel (is_vararg methods)
  Array      – Godot Array passed as parallel Variant** + count
  Dictionary – Godot Dictionary passed as parallel key/value arrays + count
  Callable   – object pointer + method name pair
  Signal     – same encoding as Callable

RETURN KINDS
────────────
  void       – no return value
  i32        – bool
  i64        – int, enum, bitfield
  f64        – float (widened to double at the boundary)
  cstring    – String / StringName / NodePath
  variant*   – all struct types, packed arrays, objects, and Variant
  Callable   - returns a callable ptr.

HARDCODED ENTRIES
─────────────────
A small set of engine-bootstrap functions are not present in extension_api.json
and are emitted first with hand-written signatures (HARDCODED_FIRST).

EXCLUSIONS
──────────
Editor, VisualShader, GLTF, FBX, and ResourceImporter classes are skipped —
they are unavailable in exported WASM builds.
"""

import json
from pathlib import Path

# Godot struct types that are flattened into scalar fields at the boundary.
# Each entry maps the type name to an ordered tuple of (field_name, js_kind).
FLATTENED_STRUCTS = {
    "Vector2": (("x", "f32"), ("y", "f32")),
    "Vector2i": (("x", "i32"), ("y", "i32")),
    "Vector3": (("x", "f32"), ("y", "f32"), ("z", "f32")),
    "Vector3i": (("x", "i32"), ("y", "i32"), ("z", "i32")),
    "Vector4": (("x", "f32"), ("y", "f32"), ("z", "f32"), ("w", "f32")),
    "Vector4i": (("x", "i32"), ("y", "i32"), ("z", "i32"), ("w", "i32")),
    "Rect2": (("x", "f32"), ("y", "f32"), ("width", "f32"), ("height", "f32")),
    "Rect2i": (("x", "i32"), ("y", "i32"), ("width", "i32"), ("height", "i32")),
    "Color": (("r", "f32"), ("g", "f32"), ("b", "f32"), ("a", "f32")),
    "Plane": (("nx", "f32"), ("ny", "f32"), ("nz", "f32"), ("d", "f32")),
    "AABB": (("px", "f32"), ("py", "f32"), ("pz", "f32"), ("sx", "f32"), ("sy", "f32"), ("sz", "f32")),
    "Transform2D": (("xx", "f32"), ("xy", "f32"), ("yx", "f32"), ("yy", "f32"), ("ox", "f32"), ("oy", "f32")),
    "Transform3D": (
        ("bxx", "f32"),
        ("bxy", "f32"),
        ("bxz", "f32"),
        ("byx", "f32"),
        ("byy", "f32"),
        ("byz", "f32"),
        ("bzx", "f32"),
        ("bzy", "f32"),
        ("bzz", "f32"),
        ("ox", "f32"),
        ("oy", "f32"),
        ("oz", "f32"),
    ),
    "Quaternion": (("x", "f32"), ("y", "f32"), ("z", "f32"), ("w", "f32")),
    "Basis": (
        ("xx", "f32"),
        ("xy", "f32"),
        ("xz", "f32"),
        ("yx", "f32"),
        ("yy", "f32"),
        ("yz", "f32"),
        ("zx", "f32"),
        ("zy", "f32"),
        ("zz", "f32"),
    ),
}

# Types whose C++ export returns a heap-allocated Variant* rather than a scalar.
VARIANT_RETURN_TYPES = {
    "Vector2",
    "Vector2i",
    "Vector3",
    "Vector3i",
    "Vector4",
    "Vector4i",
    "Rect2",
    "Rect2i",
    "Color",
    "Plane",
    "AABB",
    "Transform2D",
    "Transform3D",
    "Quaternion",
    "Basis",
    "RID",
    "Callable",
    "Signal",
    "Dictionary",
    "Array",
    "PackedByteArray",
    "PackedInt32Array",
    "PackedInt64Array",
    "PackedFloat32Array",
    "PackedFloat64Array",
    "PackedStringArray",
    "PackedVector2Array",
    "PackedVector3Array",
    "PackedColorArray",
    "PackedVector4Array",
    "Variant",
}

# Engine-bootstrap functions absent from extension_api.json; emitted first.
HARDCODED_FIRST = {
    "_Engine_get_object_singleton": {"params": ["cstring"], "returns": "i64"},
    "_Engine_get_SceneTree_singleton": {"params": [], "returns": "i64"},
}

PACKED_ARRAY_TYPES = {
    "PackedByteArray",
    "PackedInt32Array",
    "PackedInt64Array",
    "PackedFloat32Array",
    "PackedFloat64Array",
    "PackedStringArray",
    "PackedVector2Array",
    "PackedVector3Array",
    "PackedColorArray",
    "PackedVector4Array",
}

_EXCLUDED_PREFIXES = ("Editor", "VisualShader", "GLTF", "FBX", "ResourceImporter")


def is_object_id(godot_type: str) -> bool:
    """Returns True if the type crosses the boundary as a raw object pointer."""
    if godot_type in ("int", "bool", "float", "Variant"):
        return False
    if godot_type in FLATTENED_STRUCTS or godot_type in PACKED_ARRAY_TYPES:
        return False
    if godot_type in ("String", "StringName", "NodePath"):
        return False
    if godot_type.startswith(("enum::", "bitfield::", "typedarray::")):
        return False
    return True


def godot_param_to_jskinds(godot_type: str) -> list[str]:
    """Maps a single Godot parameter type to one or more JS bridge kind strings."""
    if godot_type in FLATTENED_STRUCTS:
        return [kind for _, kind in FLATTENED_STRUCTS[godot_type]]
    if godot_type in PACKED_ARRAY_TYPES:
        return [godot_type]
    if godot_type in ("String", "StringName", "NodePath"):
        return ["cstring"]
    if godot_type == "bool":
        return ["i32"]
    if godot_type == "int":
        return ["i64"]
    if godot_type == "float":
        return ["f32"]
    if godot_type == "Variant":
        return ["varargs"]
    if godot_type == "Array" or godot_type.startswith("typedarray::"):
        return ["Array"]
    if godot_type == "Dictionary":
        return ["Dictionary"]
    if godot_type == "Callable":
        return ["ptr", "cstring"]
    if godot_type == "Signal":
        return ["Signal"]
    if godot_type.startswith(("enum::", "bitfield::")):
        return ["i64"]
    return ["ptr"]  # object-derived


def godot_return_to_jskind(godot_type: str) -> str:
    """Maps a Godot return type to a single JS bridge kind string."""
    if godot_type == "void":
        return "void"
    if godot_type == "bool":
        return "i32"
    if godot_type == "int":
        return "i64"
    if godot_type == "float":
        return "f64"
    if godot_type in ("String", "StringName", "NodePath"):
        return "cstring"
    if godot_type.startswith(("enum::", "bitfield::")):
        return "i64"
    # All struct types, packed arrays, objects, and Variant return as Variant*.
    if godot_type in VARIANT_RETURN_TYPES or godot_type.startswith("typedarray::"):
        return "variant*"
    return "variant*"


def generate_export_signatures(api_json_path: Path, output_js_path: Path) -> None:
    with open(api_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_list = data["classes"] if isinstance(data, dict) else data
    signatures = {}

    for cls in class_list:
        class_name = cls.get("name", "")
        if any(class_name.startswith(p) for p in _EXCLUDED_PREFIXES):
            continue

        for method in cls.get("methods", []):
            method_name = method.get("name", "")
            if not method_name:
                continue

            func_name = f"_{class_name}_{method_name}"
            if func_name in HARDCODED_FIRST:
                continue

            # Receiver pointer is always the first parameter.
            param_kinds = ["ptr"]
            for arg in method.get("arguments", []):
                param_kinds.extend(godot_param_to_jskinds(arg.get("type", "Variant")))
            if method.get("is_vararg", False):
                param_kinds.append("varargs")

            return_info = method.get("return_value") or {}
            return_type = return_info.get("type", "void") if return_info else "void"

            signatures[func_name] = {
                "params": param_kinds,
                "returns": godot_return_to_jskind(return_type),
            }

    lines = [
        "// export_signatures.js – auto-generated, do not edit",
        "export const GODOT_EXPORT_SIGNATURES = {",
    ]

    def emit(name, entry):
        params_str = ", ".join(f'"{k}"' for k in entry["params"])
        lines.append(f'  {name}: {{ params: [{params_str}], returns: "{entry["returns"]}" }},')

    for name, entry in HARDCODED_FIRST.items():
        emit(name, entry)
    for name in sorted(signatures.keys()):
        emit(name, signatures[name])

    lines.append("};")
    lines.append("")

    output_js_path.parent.mkdir(parents=True, exist_ok=True)
    output_js_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output_js_path} ({len(signatures) + len(HARDCODED_FIRST)} entries)")


if __name__ == "__main__":
    api_json = Path("extension_api.json")
    out_js = Path("bin/.web_zip/Bridge_Functions/export_signatures.js")
    generate_export_signatures(api_json, out_js)
    print("Generated JS signatures")
