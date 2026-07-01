#!/usr/bin/env python3
"""
Standalone exports generator – reads extension_api.json and writes
exports.json / exports.syms with every EMSCRIPTEN_KEEPALIVE symbol
that the WASM module must export (hand‑written variant helpers +
all generated method exports).
"""

import json
from pathlib import Path

BOOTSTRAPPED_SYMBOLS = [
    "_Engine_get_object_singleton",
    "_Engine_get_SceneTree_singleton",
]
# Hand‑written variant helpers that are always needed
VARIANT_HELPERS = [
    "_malloc",
    "_free",
    "_variant_get_type",
    "_variant_destroy",
    "_variant_free_packed",
    "_variant_as_bool",
    "_variant_as_uint64",
    "_variant_as_int64",
    "_variant_as_double",
    "_variant_as_string",
    "_variant_as_vector2",
    "_variant_as_vector2i",
    "_variant_as_rect2",
    "_variant_as_rect2i",
    "_variant_as_vector3",
    "_variant_as_vector3i",
    "_variant_as_transform2d",
    "_variant_as_vector4",
    "_variant_as_vector4i",
    "_variant_as_plane",
    "_variant_as_quaternion",
    "_variant_as_aabb",
    "_variant_as_basis",
    "_variant_as_transform3d",
    "_variant_as_projection",
    "_variant_as_color",
    "_variant_as_string_name",
    "_variant_as_node_path",
    "_variant_as_rid",
    "_variant_as_object",
    "_variant_as_callable",
    "_variant_as_signal",
    "_variant_as_dictionary",
    "_variant_as_array",
    "_variant_as_packed_byte_array",
    "_variant_packed_byte_size",
    "_variant_as_packed_int32_array",
    "_variant_packed_int32_size",
    "_variant_as_packed_int64_array",
    "_variant_packed_int64_size",
    "_variant_as_packed_float32_array",
    "_variant_packed_float32_size",
    "_variant_as_packed_float64_array",
    "_variant_packed_float64_size",
    "_variant_as_packed_string_array",
    "_variant_packed_string_count",
    "_variant_as_packed_vector2_array",
    "_variant_packed_vector2_size",
    "_variant_as_packed_vector3_array",
    "_variant_packed_vector3_size",
    "_variant_as_packed_color_array",
    "_variant_packed_color_size",
    "_variant_as_packed_vector4_array",
    "_variant_packed_vector4_size",
    "_variant_new_nil",
    "_variant_new_bool",
    "_variant_new_int64",
    "_variant_new_double",
    "_variant_new_string",
    "_variant_new_vector2i",
    "_variant_new_vector3i",
    "_variant_new_vector4i",
    "_variant_new_rect2i",
    "_variant_new_vector2",
    "_variant_new_vector3",
    "_variant_new_rect2",
    "_variant_new_color",
    "_variant_new_plane",
    "_variant_new_quaternion",
    "_variant_new_aabb",
    "_variant_new_basis",
    "_variant_new_transform2d",
    "_variant_new_transform3d",
    "_variant_new_rid",
    "_variant_new_object_id",
    "_variant_new_node_path",
    "_variant_new_string_name",
    "_variant_new_callable",
    "_variant_new_signal",
    "_variant_new_dictionary",
    "_variant_new_array",
    "_variant_array_size",
    "_variant_array_get",
    "_variant_dictionary_size",
    "_variant_dictionary_get_key",
    "_variant_dictionary_get_value",
    "_callable_get_target_id",
    "_callable_get_method",
]

# Classes whose methods should NOT be listed (same exclusion prefix used in cpp_generators)
_EXCLUDED_PREFIXES = ("Editor", "VisualShader", "GLTF", "FBX", "ResourceImporter")


def generate_exports(api_json_path: Path, output_dir: Path) -> None:
    with open(api_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        class_list = data
    elif isinstance(data, dict) and "classes" in data:
        class_list = data["classes"]
    else:
        raise ValueError("Unexpected JSON format: expected list or dict with 'classes'")

    all_exports = list(BOOTSTRAPPED_SYMBOLS) + list(VARIANT_HELPERS)

    for cls in class_list:
        class_name = cls.get("name")
        if not class_name:
            continue
        if any(class_name.startswith(prefix) for prefix in _EXCLUDED_PREFIXES):
            continue
        for method in cls.get("methods", []):
            method_name = method.get("name")
            if method_name:
                all_exports.append(f"_{class_name}_{method_name}")

    output_dir.mkdir(exist_ok=True)

    exports_path = output_dir / "exports.json"
    exports_path.write_text(json.dumps(all_exports, indent=2), encoding="utf-8")
    print(f"Generated {exports_path} ({len(all_exports)} symbols)")

    syms_path = output_dir / "exports.syms"
    syms_path.write_text("\n".join(all_exports), encoding="utf-8")
    print(f"Generated {syms_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    generate_exports(here / "extension_api.json", here / "Exported_Symbols")
    print("Generated symbols")
