#!/usr/bin/env python3
"""
cpp_generators.py – Godot API → EMSCRIPTEN_KEEPALIVE C++ Exports

Reads extension_api.json and emits per‑class .cpp files under Godot‑Wasm‑Exports/.
Each Godot method becomes a flat C function callable from JavaScript across the
WebAssembly boundary.

All generated functions follow the same pattern:
  1. Recover the live Object* from the raw uintptr_t via ObjectDB.
  2. Look up the method via ClassDB::get_method.
  3. Reconstruct flattened / packed / string / container arguments.
  4. Call the method bind, decode the return value, and return it to JS.

Two special cases (ClassDB::instantiate, PackedScene::instantiate) are
hand‑written because they need tighter control over argument construction and
resource caching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Return‑type classification

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


def is_always_variant_return(godot_type: str) -> bool:
    """True for types that must be returned as heap‑allocated Variant*."""
    return godot_type.startswith("typedarray::") or godot_type in VARIANT_RETURN_TYPES


# Primitive return types – map to (C++ type, extraction expression)
VALUE_TYPE_RETURN = {
    "bool": ("bool", "{V}.operator bool()"),
    "int": ("int64_t", "{V}.operator int64_t()"),
    "float": ("double", "{V}.operator double()"),
}

# String return types – extracted from the result Variant
STRING_RETURN = {
    "String": "_ret_val_.operator String().utf8().get_data()",
    "StringName": "_ret_val_.operator String().utf8().get_data()",
    "NodePath": "_ret_val_.operator String().utf8().get_data()",
}


# Argument reconstruction tables

# Flattened structs: field name → C++ parameter type.
FLATTENED_STRUCTS = {
    "Vector2": (("x", "float"), ("y", "float")),
    "Vector2i": (("x", "int"), ("y", "int")),
    "Vector3": (("x", "float"), ("y", "float"), ("z", "float")),
    "Vector3i": (("x", "int"), ("y", "int"), ("z", "int")),
    "Vector4": (("x", "float"), ("y", "float"), ("z", "float"), ("w", "float")),
    "Vector4i": (("x", "int"), ("y", "int"), ("z", "int"), ("w", "int")),
    "Rect2": (("x", "float"), ("y", "float"), ("width", "float"), ("height", "float")),
    "Rect2i": (("x", "int"), ("y", "int"), ("width", "int"), ("height", "int")),
    "Color": (("r", "float"), ("g", "float"), ("b", "float"), ("a", "float")),
    "Plane": (("normal_x", "float"), ("normal_y", "float"), ("normal_z", "float"), ("d", "float")),
    "AABB": (
        ("position_x", "float"),
        ("position_y", "float"),
        ("position_z", "float"),
        ("size_x", "float"),
        ("size_y", "float"),
        ("size_z", "float"),
    ),
    "Transform2D": (
        ("x_x", "float"),
        ("x_y", "float"),
        ("y_x", "float"),
        ("y_y", "float"),
        ("origin_x", "float"),
        ("origin_y", "float"),
    ),
    "Transform3D": (
        ("bx_x", "float"),
        ("bx_y", "float"),
        ("bx_z", "float"),
        ("by_x", "float"),
        ("by_y", "float"),
        ("by_z", "float"),
        ("bz_x", "float"),
        ("bz_y", "float"),
        ("bz_z", "float"),
        ("origin_x", "float"),
        ("origin_y", "float"),
        ("origin_z", "float"),
    ),
    "Quaternion": (("x", "float"), ("y", "float"), ("z", "float"), ("w", "float")),
    "Basis": (
        ("x_x", "float"),
        ("x_y", "float"),
        ("x_z", "float"),
        ("y_x", "float"),
        ("y_y", "float"),
        ("y_z", "float"),
        ("z_x", "float"),
        ("z_y", "float"),
        ("z_z", "float"),
    ),
}

# Re‑assembly expressions for structs that need more than a simple constructor.
FLATTENED_STRUCT_CONSTRUCTORS = {
    "Vector2": "Vector2({x}, {y})",
    "Vector2i": "Vector2i({x}, {y})",
    "Vector3": "Vector3({x}, {y}, {z})",
    "Vector3i": "Vector3i({x}, {y}, {z})",
    "Vector4": "Vector4({x}, {y}, {z}, {w})",
    "Vector4i": "Vector4i({x}, {y}, {z}, {w})",
    "Rect2": "Rect2(Vector2({x}, {y}), Vector2({width}, {height}))",
    "Rect2i": "Rect2i(Vector2i({x}, {y}), Vector2i({width}, {height}))",
    "Color": "Color({r}, {g}, {b}, {a})",
    "Plane": "Plane(Vector3({normal_x}, {normal_y}, {normal_z}), {d})",
    "AABB": "AABB(Vector3({position_x}, {position_y}, {position_z}), Vector3({size_x}, {size_y}, {size_z}))",
    "Transform2D": "Transform2D(Vector2({x_x}, {x_y}), Vector2({y_x}, {y_y}), Vector2({origin_x}, {origin_y}))",
    "Transform3D": "Transform3D(Basis(Vector3({bx_x}, {bx_y}, {bx_z}), Vector3({by_x}, {by_y}, {by_z}), Vector3({bz_x}, {bz_y}, {bz_z})), Vector3({origin_x}, {origin_y}, {origin_z}))",
    "Quaternion": "Quaternion({x}, {y}, {z}, {w})",
    "Basis": "Basis(Vector3({x_x}, {x_y}, {x_z}), Vector3({y_x}, {y_y}, {y_z}), Vector3({z_x}, {z_y}, {z_z}))",
}

# Maps a flattened struct back to its Godot type name (for Variant wrapping).
FLATTENED_STRUCT_TYPE = {
    "Vector2": "Vector2",
    "Vector2i": "Vector2i",
    "Vector3": "Vector3",
    "Vector3i": "Vector3i",
    "Vector4": "Vector4",
    "Vector4i": "Vector4i",
    "Rect2": "Rect2",
    "Rect2i": "Rect2i",
    "Color": "Color",
    "Plane": "Plane",
    "AABB": "AABB",
    "Transform2D": "Transform2D",
    "Transform3D": "Transform3D",
    "Quaternion": "Quaternion",
    "Basis": "Basis",
}


# Packed array helpers

# Godot type → (C++ pointer type for raw data, element type, element size expression)
PACKED_ARRAY_INFO = {
    "PackedByteArray": ("uint8_t*", "uint8_t", "sizeof(uint8_t)"),
    "PackedInt32Array": ("int32_t*", "int32_t", "sizeof(int32_t)"),
    "PackedInt64Array": ("int64_t*", "int64_t", "sizeof(int64_t)"),
    "PackedFloat32Array": ("float*", "float", "sizeof(float)"),
    "PackedFloat64Array": ("double*", "double", "sizeof(double)"),
    "PackedVector2Array": ("float*", "Vector2", None),  # reconstruct element‑wise
    "PackedVector3Array": ("float*", "Vector3", None),
    "PackedColorArray": ("float*", "Color", None),
    "PackedVector4Array": ("float*", "Vector4", None),
    "PackedStringArray": ("const char**", "String", None),
}


def is_packed_array(godot_type: str) -> bool:
    return godot_type in PACKED_ARRAY_INFO


def packed_array_reconstruction(
    godot_type: str, safe_name: str, data_ptr: str, size_var: str, lines: list[str]
) -> None:
    """Emit C++ code to rebuild a Godot packed array from a raw pointer + count."""
    info = PACKED_ARRAY_INFO[godot_type]
    container_type = godot_type
    elem_size = info[2]

    if elem_size is not None:
        # Simple memcpy for types with fixed element size
        lines.append(f"    {container_type} {safe_name};")
        lines.append(f"    {safe_name}.resize({size_var});")
        lines.append(f"    memcpy({safe_name}.ptrw(), {data_ptr}, {size_var} * {elem_size});")

    elif godot_type == "PackedStringArray":
        lines.append(f"    PackedStringArray {safe_name};")
        lines.append(f"    {safe_name}.resize({size_var});")
        lines.append(f"    for (int i = 0; i < {size_var}; i++) {{")
        lines.append(f"        {safe_name}.set(i, String::utf8({data_ptr}[i]));")
        lines.append("    }")

    elif godot_type in ("PackedVector2Array", "PackedVector3Array", "PackedColorArray", "PackedVector4Array"):
        if godot_type == "PackedVector2Array":
            lines.append(f"    PackedVector2Array {safe_name};")
            lines.append(f"    {safe_name}.resize({size_var});")
            lines.append(f"    for (int i = 0; i < {size_var}; i++) {{")
            lines.append(f"        {safe_name}.set(i, Vector2({data_ptr}[i*2], {data_ptr}[i*2+1]));")
            lines.append("    }")
        elif godot_type == "PackedVector3Array":
            lines.append(f"    PackedVector3Array {safe_name};")
            lines.append(f"    {safe_name}.resize({size_var});")
            lines.append(f"    for (int i = 0; i < {size_var}; i++) {{")
            lines.append(f"        {safe_name}.set(i, Vector3({data_ptr}[i*3], {data_ptr}[i*3+1], {data_ptr}[i*3+2]));")
            lines.append("    }")
        elif godot_type == "PackedColorArray":
            lines.append(f"    PackedColorArray {safe_name};")
            lines.append(f"    {safe_name}.resize({size_var});")
            lines.append(f"    for (int i = 0; i < {size_var}; i++) {{")
            lines.append(
                f"        {safe_name}.set(i, Color({data_ptr}[i*4], {data_ptr}[i*4+1], {data_ptr}[i*4+2], {data_ptr}[i*4+3]));"
            )
            lines.append("    }")
        else:  # PackedVector4Array
            lines.append(f"    PackedVector4Array {safe_name};")
            lines.append(f"    {safe_name}.resize({size_var});")
            lines.append(f"    for (int i = 0; i < {size_var}; i++) {{")
            lines.append(
                f"        {safe_name}.set(i, Vector4({data_ptr}[i*4], {data_ptr}[i*4+1], {data_ptr}[i*4+2], {data_ptr}[i*4+3]));"
            )
            lines.append("    }")
    else:
        lines.append(f"    // reconstruction not implemented for {godot_type}")
        lines.append(f"    Variant {safe_name} = Variant();")


# General type mapping

PARAM_TYPE_MAP = {
    "bool": "bool",
    "int": "int64_t",
    "float": "double",
    "String": "const char*",
    "StringName": "const char*",
    "NodePath": "const char*",
    "Variant": "Variant",
    "Array": "Array",
    "Dictionary": "Dictionary",
}

STRING_PARAM_TYPES = {"String", "StringName", "NodePath"}
REFCOUNTED_RETURN_TYPES = {"Resource", "RefCounted"}
_EXCLUDED_PREFIXES = ("Editor", "VisualShader", "GLTF", "FBX", "ResourceImporter")

CXX_KEYWORDS = {
    "class",
    "new",
    "delete",
    "default",
    "operator",
    "template",
    "typename",
    "enum",
    "union",
    "struct",
    "private",
    "protected",
    "public",
    "friend",
    "virtual",
    "explicit",
    "constexpr",
    "static",
    "const",
    "volatile",
    "export",
    "char",
    "int",
    "float",
    "double",
    "long",
    "short",
    "signed",
    "unsigned",
    "bool",
    "void",
    "auto",
    "register",
    "extern",
    "mutable",
    "thread_local",
    "switch",
    "case",
    "break",
    "continue",
    "goto",
    "return",
    "if",
    "else",
    "while",
    "do",
    "for",
    "try",
    "catch",
    "throw",
    "typedef",
    "using",
    "sizeof",
    "alignas",
    "alignof",
    "decltype",
    "noexcept",
    "nullptr",
    "concept",
    "requires",
    "co_await",
    "co_return",
    "co_yield",
    "consteval",
    "constinit",
    "and",
    "or",
    "xor",
    "not",
    "bitand",
    "bitor",
    "compl",
    "args",
    "type",
}


# Small helper utilities


def sanitize(name: str) -> str:
    """Escape a C++ identifier if it collides with a keyword."""
    return name if name not in CXX_KEYWORDS else name + "_"


def is_enum(godot_type: str) -> bool:
    return godot_type.startswith("enum::") or godot_type.startswith("bitfield::")


def is_refcounted(godot_type: str) -> bool:
    return godot_type in REFCOUNTED_RETURN_TYPES


def cpp_return_type(godot_type: str) -> str:
    """Map a Godot return type to the C++ return type of the export function."""
    if godot_type == "void":
        return "void"
    if is_enum(godot_type):
        return "int64_t"
    if is_object(godot_type):
        return "Variant*"
    if is_always_variant_return(godot_type):
        return "Variant*"
    if godot_type in STRING_RETURN:
        return "const char*"
    return VALUE_TYPE_RETURN[godot_type][0]


def cpp_param_type(godot_type: str) -> str:
    """Map a Godot argument type to the C++ parameter type (unflattened)."""
    if is_enum(godot_type):
        return "int64_t"
    if is_object(godot_type):
        return "uintptr_t"
    return PARAM_TYPE_MAP.get(godot_type, "Variant")


def early_return(c_ret: str) -> str:
    """Return the appropriate bail‑out statement for a given return type."""
    if c_ret == "void":
        return "return;"
    if c_ret == "Variant*":
        return "return nullptr;"
    if c_ret == "bool":
        return "return false;"
    if c_ret in ("int64_t", "double"):
        return "return 0;"
    if c_ret == "const char*":
        return 'return "";'
    return f"return {c_ret}();"


def variant_from_param(arg_name: str, godot_type: str) -> str:
    """
    Return a C++ expression that wraps a reconstructed local variable
    into a Variant suitable for the argument list.
    """
    if is_enum(godot_type):
        return f"Variant((int64_t){arg_name})"
    if godot_type in STRING_PARAM_TYPES:
        return f"Variant({arg_name})"
    if is_object(godot_type):
        return f"Variant(ObjectDB::get_instance(((Object*)(uintptr_t){arg_name})->get_instance_id()))"
    if godot_type in FLATTENED_STRUCT_TYPE:
        return f"Variant({arg_name})"
    if is_packed_array(godot_type):
        return f"Variant({arg_name})"
    # Array / Dictionary are already wrapped with Variant() at the call site
    return f"Variant({arg_name})"


def is_object(godot_type: str) -> bool:
    """Determine if a type represents a Godot Object (not a built‑in or struct)."""
    if (
        godot_type in PARAM_TYPE_MAP
        or godot_type in VALUE_TYPE_RETURN
        or godot_type in STRING_RETURN
        or godot_type in FLATTENED_STRUCTS
        or godot_type in VARIANT_RETURN_TYPES
    ):
        return False
    return not (godot_type.startswith("enum::") or godot_type.startswith("bitfield::"))


# Return‑statement generator


def emit_return_stmt(lines: list[str], return_type: str, is_refcounted_ret: bool) -> None:
    """
    Append C++ lines that decode the Variant result `_ret_val_` into the
    appropriate return type.
    """
    if is_enum(return_type):
        lines.append("    return _ret_val_.operator int64_t();")
        return

    if is_object(return_type):
        lines.append("    Object *_ret_obj_ = _ret_val_.get_validated_object();")
        lines.append("    if (!_ret_obj_) { return new Variant((int64_t)0); }")
        if is_refcounted_ret:
            lines.append("    if (Resource *_res_ = Object::cast_to<Resource>(_ret_obj_)) {")
            lines.append("        cache_loaded_resource(_res_);")
            lines.append("    }")
        lines.append("    return new Variant((int64_t)(uintptr_t)_ret_obj_);")
        return

    if is_always_variant_return(return_type):
        lines.append("    return new Variant(_ret_val_);")
        return

    if return_type in STRING_RETURN:
        expr = STRING_RETURN[return_type]
        lines.append(f"    return {expr};")
        return

    # Primitive numeric or bool
    _, extract = VALUE_TYPE_RETURN[return_type]
    expr = extract.replace("{V}", "_ret_val_")
    lines.append(f"    return {expr};")


# Per‑class file writer


def write_class_file(
    class_name: str,
    methods: list[dict[str, Any]],
    output_dir: Path,
    is_resource_loader: bool = False,
    is_refcounted_class: bool = False,
) -> None:
    """Generate one .cpp file containing all method exports for a single class."""

    lines: list[str] = []
    lines.append('#include "header/need.h"')
    lines.append('#include "header/caching_ptrs.h"')
    lines.append("")

    if is_resource_loader:
        lines.append("HashMap<ObjectID, Ref<Resource>> resource_cache;")
        lines.append("")

    lines.append('extern "C" {')
    lines.append("")

    for method in methods:
        # ── Hand‑written special cases ──────────────────────────────────────
        if class_name == "ClassDB" and method.get("name") == "instantiate":
            lines.append("""
            Variant* ClassDB_instantiate(uintptr_t p_object_ptr, const char* class_) {
                Object *_obj_ = ObjectDB::get_instance(((Object*)(uintptr_t)p_object_ptr)->get_instance_id());
                if (!_obj_) {
                    return new Variant((int64_t)0);
                }
                MethodBind *_bind_ = ClassDB::get_method("ClassDB", "instantiate");
                if (!_bind_) {
                    return new Variant((int64_t)0);
                }
                String _str_class_ = String::utf8(class_);
                StringName _sn_class_(_str_class_);
                Variant _vargs_[1];
                _vargs_[0] = Variant(_sn_class_);
                const Variant *_vargptrs_[1];
                _vargptrs_[0] = &_vargs_[0];
                Callable::CallError _call_error_;
                Variant _ret_val_ = _bind_->call(_obj_, _vargptrs_, 1, _call_error_);
                Object *_ret_obj_ = _ret_val_.get_validated_object();
                if (!_ret_obj_) {
                    return new Variant((int64_t)0);
                }
                if (Resource *_res_ = Object::cast_to<Resource>(_ret_obj_)) {
                    cache_loaded_resource(_res_);
                }
                return new Variant((int64_t)(uintptr_t)_ret_obj_);
            }
            """)
            continue

        if class_name == "PackedScene" and method.get("name") == "instantiate":
            lines.append("""
        EMSCRIPTEN_KEEPALIVE
        Variant* PackedScene_instantiate(uintptr_t p_object_ptr, int64_t edit_state) {
            Object *_obj_ = ObjectDB::get_instance(((Object*)(uintptr_t)p_object_ptr)->get_instance_id());
            if (!_obj_) {
                return new Variant((int64_t)0);
            }
            MethodBind *_bind_ = ClassDB::get_method("PackedScene", "instantiate");
            if (!_bind_) {
                return new Variant((int64_t)0);
            }
            Variant _vargs_[1];
            _vargs_[0] = Variant((int64_t)edit_state);
            const Variant *_vargptrs_[1];
            _vargptrs_[0] = &_vargs_[0];
            Callable::CallError _call_error_;
            Variant _ret_val_ = _bind_->call(_obj_, _vargptrs_, 1, _call_error_);
            Object *_ret_obj_ = _ret_val_.get_validated_object();
            if (!_ret_obj_) {
                return new Variant((int64_t)0);
            }
            if (Resource *_res_ = Object::cast_to<Resource>(_ret_obj_)) {
                cache_loaded_resource(_res_);
            }
            return new Variant((int64_t)(uintptr_t)_ret_obj_);
        }
            """)
            continue

        method_name = method.get("name")
        if not method_name:
            continue

        # ── Collect metadata ───────────────────────────────────────────────
        is_vararg = method.get("is_vararg", False)
        return_info = method.get("return_value") or {}
        return_type = return_info.get("type", "void") if return_info else "void"
        arguments = method.get("arguments", [])
        is_refcounted_ret = (
            method.get("is_refcounted", False)
            or is_refcounted(return_type)
            or (is_object(return_type) and return_type == class_name and is_refcounted_class)
        )

        func_name = f"{class_name}_{method_name}"
        c_ret_type = cpp_return_type(return_type)
        bail = early_return(c_ret_type)

        # ── Build parameter list ────────────────────────────────────────────
        params: list[str] = ["uintptr_t p_object_ptr"]

        for arg in arguments:
            arg_type = arg.get("type", "Variant")
            safe_name = sanitize(arg.get("name", "p_arg"))

            if arg_type in FLATTENED_STRUCTS:
                # Flatten structs into individual scalar params
                for fname, ctype in FLATTENED_STRUCTS[arg_type]:
                    params.append(f"{ctype} {safe_name}_{fname}")

            elif is_packed_array(arg_type):
                ptr_type = PACKED_ARRAY_INFO[arg_type][0]
                params.append(f"{ptr_type} {safe_name}_data")
                params.append(f"int {safe_name}_size")

            elif arg_type == "Callable":
                # Flattened: target pointer + method name
                params.append(f"uintptr_t {safe_name}_ptr")
                params.append(f"const char* {safe_name}_method")

            elif arg_type == "Array" or arg_type.startswith("typedarray::"):
                params.append(f"const Variant** {safe_name}_data")
                params.append(f"int {safe_name}_count")

            elif arg_type == "Dictionary":
                params.append(f"const char** {safe_name}_keys")
                params.append(f"const Variant** {safe_name}_values")
                params.append(f"int {safe_name}_count")

            else:
                params.append(f"{cpp_param_type(arg_type)} {safe_name}")

        if is_vararg:
            params.append("const Variant** p_varargs")
            params.append("int p_vararg_count")

        # ── Function header ─────────────────────────────────────────────────
        lines.append("EMSCRIPTEN_KEEPALIVE")
        lines.append(f"{c_ret_type} {func_name}({', '.join(params)}) {{")

        # ── Object / bind lookup ────────────────────────────────────────────
        lines.append(
            "    Object *_obj_ = ObjectDB::get_instance(((Object*)(uintptr_t)p_object_ptr)->get_instance_id());"
        )
        lines.append(f"    if (!_obj_) {{ {bail} }}")
        lines.append(f'    MethodBind *_bind_ = ClassDB::get_method("{class_name}", "{method_name}");')
        lines.append(f"    if (!_bind_) {{ {bail} }}")

        # ── Reconstruct non‑trivial arguments ───────────────────────────────
        for arg in arguments:
            arg_type = arg.get("type", "Variant")
            safe_name = sanitize(arg.get("name", "p_arg"))

            if arg_type in FLATTENED_STRUCTS:
                godot_type = FLATTENED_STRUCT_TYPE[arg_type]
                fields = FLATTENED_STRUCTS[arg_type]
                if arg_type in FLATTENED_STRUCT_CONSTRUCTORS:
                    fmt = FLATTENED_STRUCT_CONSTRUCTORS[arg_type]
                    param_dict = {fname: f"{safe_name}_{fname}" for fname, _ in fields}
                    expr = fmt.format(**param_dict)
                else:
                    args_str = ", ".join(f"{safe_name}_{fname}" for fname, _ in fields)
                    expr = f"{godot_type}({args_str})"
                lines.append(f"    {godot_type} {safe_name} = {expr};")
                continue

            if arg_type in STRING_PARAM_TYPES:
                local_str = f"_str_{safe_name}"
                lines.append(f"    String {local_str} = String::utf8({safe_name});")
                if arg_type == "NodePath":
                    lines.append(f"    NodePath _np_{safe_name}({local_str});")
                elif arg_type == "StringName":
                    lines.append(f"    StringName _sn_{safe_name}({local_str});")
                continue

            if arg_type == "Callable":
                lines.append(f"    Object *_obj_{safe_name}_ = (Object*)(uintptr_t){safe_name}_ptr;")
                lines.append(f"    ObjectID _id_{safe_name}_ = _obj_{safe_name}_->get_instance_id();")
                lines.append(f"    String _str_{safe_name}_method = String::utf8({safe_name}_method);")
                lines.append(f"    StringName _sn_{safe_name}_method(_str_{safe_name}_method);")
                lines.append(f"    Callable {safe_name}(_id_{safe_name}_, _sn_{safe_name}_method);")
                continue

            if is_packed_array(arg_type):
                data_var = f"{safe_name}_data"
                size_var = f"{safe_name}_size"
                packed_array_reconstruction(arg_type, safe_name, data_var, size_var, lines)
                continue

            if arg_type == "Array" or arg_type.startswith("typedarray::"):
                lines.append(f"    Array {safe_name};")
                lines.append(f"    {safe_name}.resize({safe_name}_count);")
                lines.append(f"    for (int i = 0; i < {safe_name}_count; i++) {{")
                lines.append(f"        {safe_name}[i] = *{safe_name}_data[i];")
                lines.append("    }")
                continue

            if arg_type == "Dictionary":
                lines.append(f"    Dictionary {safe_name};")
                lines.append(f"    for (int i = 0; i < {safe_name}_count; i++) {{")
                lines.append(f"        {safe_name}[String::utf8({safe_name}_keys[i])] = *{safe_name}_values[i];")
                lines.append("    }")
                continue

        # ── Vararg path ────────────────────────────────────────────────────
        if is_vararg:
            n_named = len(arguments)
            lines.append(f"    int _total_ = {n_named} + p_vararg_count;")
            lines.append("    Variant* _storage_ = new Variant[_total_];")
            lines.append("    const Variant** _ptrs_ = new const Variant*[_total_];")
            # Pack named arguments
            for i, arg in enumerate(arguments):
                arg_type = arg.get("type", "Variant")
                safe_name = sanitize(arg.get("name", "p_arg"))
                if arg_type == "String":
                    lines.append(f"    _storage_[{i}] = {variant_from_param('_str_' + safe_name, arg_type)};")
                elif arg_type == "StringName":
                    lines.append(f"    _storage_[{i}] = {variant_from_param('_sn_' + safe_name, arg_type)};")
                elif arg_type == "NodePath":
                    lines.append(f"    _storage_[{i}] = {variant_from_param('_np_' + safe_name, arg_type)};")
                elif arg_type == "Array" or arg_type.startswith("typedarray::"):
                    lines.append(f"    _storage_[{i}] = Variant({safe_name});")
                elif arg_type == "Dictionary":
                    lines.append(f"    _storage_[{i}] = Variant({safe_name});")
                else:
                    lines.append(f"    _storage_[{i}] = {variant_from_param(safe_name, arg_type)};")
                lines.append(f"    _ptrs_[{i}] = &_storage_[{i}];")

            # Object‑ID heuristic for varargs
            lines.append("    const int64_t _ADDR_MIN_ = 1024;")
            lines.append("    const int64_t _ADDR_MAX_ = 0xFFFFFFFFLL;")
            lines.append("    for (int _i_ = 0; _i_ < p_vararg_count; _i_++) {")
            lines.append("        const Variant &_v_ = *p_varargs[_i_];")
            lines.append("        if (_v_.get_type() == Variant::INT) {")
            lines.append("            int64_t _raw_ = (int64_t)_v_;")
            lines.append("            if (_raw_ >= _ADDR_MIN_ && _raw_ <= _ADDR_MAX_) {")
            lines.append("                uintptr_t _candidate_ptr_ = (uintptr_t)_raw_;")
            lines.append("                ObjectID _candidate_id_ = ((Object*)_candidate_ptr_)->get_instance_id();")
            lines.append("                Object *_resolved_ = ObjectDB::get_instance(_candidate_id_);")
            lines.append("                ptr_caching(_candidate_id_, _candidate_ptr_);")
            lines.append("                if (_resolved_) {")
            lines.append(f"                    _storage_[{n_named} + _i_] = Variant(_candidate_id_);")
            lines.append("                } else {")
            lines.append(f"                    _storage_[{n_named} + _i_] = _v_;")
            lines.append("                }")
            lines.append("            } else {")
            lines.append(f"                _storage_[{n_named} + _i_] = _v_;")
            lines.append("            }")
            lines.append("        } else {")
            lines.append(f"            _storage_[{n_named} + _i_] = _v_;")
            lines.append("        }")
            lines.append(f"        _ptrs_[{n_named} + _i_] = &_storage_[{n_named} + _i_];")
            lines.append("    }")

            call_target = "_obj_"
            lines.append("    Callable::CallError _call_error_;")
            if return_type == "void":
                lines.append(f"    _bind_->call({call_target}, _ptrs_, _total_, _call_error_);")
                lines.append("    delete[] _storage_;")
                lines.append("    delete[] _ptrs_;")
                lines.append("    return;")
            else:
                lines.append(f"    Variant _ret_val_ = _bind_->call({call_target}, _ptrs_, _total_, _call_error_);")
                lines.append("    delete[] _storage_;")
                lines.append("    delete[] _ptrs_;")
                emit_return_stmt(lines, return_type, is_refcounted_ret)
            lines.append("}")
            lines.append("")
            continue  # skip normal path

        # ── Normal (non‑vararg) path ──────────────────────────────────────
        n_args = len(arguments)
        if n_args > 0:
            lines.append(f"    Variant _vargs_[{n_args}];")
            for i, arg in enumerate(arguments):
                arg_type = arg.get("type", "Variant")
                safe_name = sanitize(arg.get("name", "p_arg"))
                if arg_type == "String":
                    var = variant_from_param("_str_" + safe_name, arg_type)
                elif arg_type == "StringName":
                    var = variant_from_param("_sn_" + safe_name, arg_type)
                elif arg_type == "NodePath":
                    var = variant_from_param("_np_" + safe_name, arg_type)
                elif arg_type == "Array" or arg_type.startswith("typedarray::"):
                    var = f"Variant({safe_name})"
                elif arg_type == "Dictionary":
                    var = f"Variant({safe_name})"
                else:
                    var = variant_from_param(safe_name, arg_type)
                lines.append(f"    _vargs_[{i}] = {var};")
            lines.append(f"    const Variant *_vargptrs_[{n_args}];")
            for i in range(n_args):
                lines.append(f"    _vargptrs_[{i}] = &_vargs_[{i}];")
            argptrs = "_vargptrs_"
            argc = n_args
        else:
            argptrs = "nullptr"
            argc = 0

        call_target = "_obj_"
        lines.append("    Callable::CallError _call_error_;")
        if return_type == "void":
            lines.append(f"    _bind_->call({call_target}, {argptrs}, {argc}, _call_error_);")
            lines.append("    return;")
        else:
            lines.append(f"    Variant _ret_val_ = _bind_->call({call_target}, {argptrs}, {argc}, _call_error_);")
            emit_return_stmt(lines, return_type, is_refcounted_ret)

        lines.append("}")
        lines.append("")

    lines.append('} // extern "C"')
    lines.append("")

    output_file = output_dir / f"{class_name}.cpp"
    output_file.write_text("\n".join(lines), encoding="utf-8")


# Entry point


def generate_all_class_files(json_path: Path, output_dir: Path) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        class_list = data
    elif isinstance(data, dict) and "classes" in data:
        class_list = data["classes"]
    else:
        raise ValueError("Unexpected JSON format: expected list or dict with 'classes'")

    output_dir.mkdir(exist_ok=True)

    for cls in class_list:
        class_name = cls.get("name")
        if not class_name:
            continue
        if any(class_name.startswith(prefix) for prefix in _EXCLUDED_PREFIXES):
            continue
        methods = cls.get("methods", [])
        if not methods:
            continue

        is_resource_loader = class_name == "ResourceLoader"
        is_refcounted_class = cls.get("is_refcounted", False)
        write_class_file(
            class_name,
            methods,
            output_dir,
            is_resource_loader=is_resource_loader,
            is_refcounted_class=is_refcounted_class,
        )


if __name__ == "__main__":
    JSON_PATH = Path("extension_api.json")
    OUTPUT_DIR = Path("Godot-Wasm-Exports")
    generate_all_class_files(JSON_PATH, OUTPUT_DIR)
    print("generated C++ bindings")
