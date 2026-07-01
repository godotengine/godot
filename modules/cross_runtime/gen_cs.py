#!/usr/bin/env python3
"""
gen_cs.py

Generates per-class C# source files from extension_api.json for the .NET 10
WebAssembly runtime, targeting the __callGodot JS bridge pattern.

ARCHITECTURE OVERVIEW
─────────────────────
Each Godot class becomes a .cs file under GodotApi/. Every method becomes a
public C# method that calls GodotBridge.CallGodot(), passing a flat object[]
of scalar arguments across the JS interop boundary into the Godot WASM module.

Two additional outputs are written to utilities/:
  GodotBridge.cs   – [JSImport] declarations for CallGodot and CallGodotPacked*
  SignalManager.cs – subscribe/unsubscribe wiring via CrossRuntimeEventSignal
  NativeStructs.cs – C# structs parsed from native_structures in the API JSON

OBJECT IDENTITY
───────────────
Godot objects are represented in C# as a ulong Id (raw Object* cast to uintptr_t
on the C++ side). Constructors accept this Id; all method calls pass it as the
first scalar argument so the C++ export can recover the live Object* via ObjectDB.
We use the pointers as the id because using the actual instance id has its coomplications
, specifically when long instance ids appear, they need special handling that just adds more
complexity unlike this one which is a much cleaner approach.

ARGUMENT MARSHALING
────────────────────
All arguments are packed into object[] before crossing the JS boundary:
  Primitives (bool, double, string)  → passed as is.
  Integer types / enums              → cast to (double)(ulong) .
  Godot structs (Vector2, Color, …)  → flattened field-by-field via STRUCT_LAYOUT.
  Object references                  → (double)(ulong)obj.Id.
  RID                                → (double)(ulong)rid.Id.
  Array / Dictionary                 → wrapped with VariantPacker.Flatten().
  Packed buffer types (1–6 args)     → routed through CallGodotPacked* using
                                       MemoryMarshal.AsBytes for zero-copy transfer.

RETURN VALUE DECODING
─────────────────────
The JS bridge returns object. Decoding is type-driven:
  Primitive types    → Convert.ToDouble / ToString / bool unbox.
  Godot structs      → JS bridge returns a JSObject; fields are read by name
                       via GetPropertyAsDouble and the struct is reconstructed.
  Packed arrays      → returned as object[]; element-wise conversion to the
                       target element type (float[], Vector2[], etc.).
  Object references  → new T(unchecked((ulong)Convert.ToDouble(ret))).
  Enums              → cast via (ulong)Convert.ToDouble.

SINGLETONS
──────────
Singleton classes expose a lazy static SingletonId property that calls
Engine.get_object_singleton() on first access. All method calls on singletons
pass SingletonId instead of an instance Id.

SIGNALS
───────
Signals are emitted as C# events backed by SignalManager.Subscribe /
Unsubscribe, which wire through CrossRuntimeEventSignal on the Godot side.
If a signal name collides with a method, the signal is chosen over the method.

PROPERTIES
──────────
get_*/is_* + set_* pairs are detected and collapsed into C# properties where
the getter and setter types are compatible. Pairs that collide with signal names,
class names, known base members, or emitted method names are emitted as plain
methods instead.

ENUM HANDLING
─────────────
Per-class enums are changed to uppercase to avoid colliding with any methods and signals.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Constants & Look‑up Tables


PACKED_BUFFER_TYPES: set[str] = {
    "PackedByteArray",
    "PackedInt32Array",
    "PackedInt64Array",
    "PackedFloat32Array",
    "PackedFloat64Array",
    "PackedVector2Array",
    "PackedVector3Array",
    "PackedVector4Array",
    "PackedColorArray",
}

# Mapping from Godot type names (as they appear in extension_api.json)
# to their C# equivalents.
TYPE_MAP: dict[str, str] = {
    "void": "void",
    "int": "long",
    "float": "double",
    "bool": "bool",
    "String": "string",
    "StringName": "string",
    "NodePath": "string",
    "Variant": "object",
    "PackedByteArray": "byte[]",
    "PackedInt32Array": "int[]",
    "PackedInt64Array": "int[]",
    "PackedFloat32Array": "float[]",
    "PackedFloat64Array": "float[]",
    "PackedStringArray": "string[]",
    "PackedVector2Array": "Vector2[]",
    "PackedVector3Array": "Vector3[]",
    "PackedColorArray": "Color[]",
    "Array": "Array",
    "Dictionary": "Dictionary<object, object>",
    "Rect2": "Rect2",
    "Rect2i": "Rect2i",
    "Vector2": "Vector2",
    "Vector2i": "Vector2i",
    "Vector3": "Vector3",
    "Vector3i": "Vector3i",
    "Vector4": "Vector4",
    "Vector4i": "Vector4i",
    "Color": "Color",
    "RID": "RID",
    "Callable": "Callable",
    "Signal": "Signal",
    "AABB": "AABB",
    "Basis": "Basis",
    "Transform2D": "Transform2D",
    "Transform3D": "Transform3D",
    "Quaternion": "Quaternion",
    "Plane": "Plane",
    "Projection": "Projection",
}

# Types that are marshaled "as‑is" (the JS side receives the whole object).
ANY_MARSHAL_TYPES: set[str] = {"Callable", "Signal", "Plane", "Quaternion", "Projection"}

# Fields returned by the JavaScript bridge for each Godot struct.
# Each entry: (JavaScript property name, C# type used in GetPropertyAs…)
STRUCT_JS_FIELDS: dict[str, list[tuple[str, str]]] = {
    "Vector2": [("X", "float"), ("Y", "float")],
    "Vector2i": [("X", "int"), ("Y", "int")],
    "Vector3": [("X", "float"), ("Y", "float"), ("Z", "float")],
    "Vector3i": [("X", "int"), ("Y", "int"), ("Z", "int")],
    "Vector4": [("X", "float"), ("Y", "float"), ("Z", "float"), ("W", "float")],
    "Vector4i": [("X", "int"), ("Y", "int"), ("Z", "int"), ("W", "int")],
    "Color": [("R", "float"), ("G", "float"), ("B", "float"), ("A", "float")],
    "Rect2": [("X", "float"), ("Y", "float"), ("Width", "float"), ("Height", "float")],
    "Rect2i": [("X", "int"), ("Y", "int"), ("Width", "int"), ("Height", "int")],
    "Plane": [("Normal.X", "float"), ("Normal.Y", "float"), ("Normal.Z", "float"), ("D", "float")],
    "Quaternion": [("X", "float"), ("Y", "float"), ("Z", "float"), ("W", "float")],
    "AABB": [
        ("Position.X", "float"),
        ("Position.Y", "float"),
        ("Position.Z", "float"),
        ("Size.X", "float"),
        ("Size.Y", "float"),
        ("Size.Z", "float"),
    ],
    "Transform2D": [
        ("Column0.X", "float"),
        ("Column0.Y", "float"),
        ("Column1.X", "float"),
        ("Column1.Y", "float"),
        ("Origin.X", "float"),
        ("Origin.Y", "float"),
    ],
    "Basis": [
        ("Column0.X", "float"),
        ("Column0.Y", "float"),
        ("Column0.Z", "float"),
        ("Column1.X", "float"),
        ("Column1.Y", "float"),
        ("Column1.Z", "float"),
        ("Column2.X", "float"),
        ("Column2.Y", "float"),
        ("Column2.Z", "float"),
    ],
    "Transform3D": [
        ("Basis.Column0.X", "float"),
        ("Basis.Column0.Y", "float"),
        ("Basis.Column0.Z", "float"),
        ("Basis.Column1.X", "float"),
        ("Basis.Column1.Y", "float"),
        ("Basis.Column1.Z", "float"),
        ("Basis.Column2.X", "float"),
        ("Basis.Column2.Y", "float"),
        ("Basis.Column2.Z", "float"),
        ("Origin.X", "float"),
        ("Origin.Y", "float"),
        ("Origin.Z", "float"),
    ],
}

# C# property layout used for flattening a struct argument into primitives.
# Each entry: (property name, child type name or None for a leaf).
STRUCT_LAYOUT: dict[str, list[tuple[str, str | None]]] = {
    "Vector2": [("X", None), ("Y", None)],
    "Vector2i": [("X", None), ("Y", None)],
    "Vector3": [("X", None), ("Y", None), ("Z", None)],
    "Vector3i": [("X", None), ("Y", None), ("Z", None)],
    "Vector4": [("X", None), ("Y", None), ("Z", None), ("W", None)],
    "Vector4i": [("X", None), ("Y", None), ("Z", None), ("W", None)],
    "Color": [("R", None), ("G", None), ("B", None), ("A", None)],
    "Rect2": [("Position", "Vector2"), ("Size", "Vector2")],
    "Rect2i": [("Position", "Vector2i"), ("Size", "Vector2i")],
    "AABB": [("Position", "Vector3"), ("Size", "Vector3")],
    "Basis": [("X", "Vector3"), ("Y", "Vector3"), ("Z", "Vector3")],
    "Transform2D": [("X", "Vector2"), ("Y", "Vector2"), ("Origin", "Vector2")],
    "Transform3D": [("Basis", "Basis"), ("Origin", "Vector3")],
}

# C# reserved keywords that must be escaped with '@'.
CS_KEYWORDS: set[str] = {
    "abstract",
    "as",
    "base",
    "bool",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "checked",
    "class",
    "const",
    "continue",
    "decimal",
    "default",
    "delegate",
    "do",
    "double",
    "else",
    "enum",
    "event",
    "explicit",
    "extern",
    "false",
    "finally",
    "fixed",
    "float",
    "for",
    "foreach",
    "goto",
    "if",
    "implicit",
    "in",
    "int",
    "interface",
    "internal",
    "is",
    "lock",
    "ulong",
    "namespace",
    "new",
    "null",
    "object",
    "operator",
    "out",
    "override",
    "params",
    "private",
    "protected",
    "public",
    "readonly",
    "ref",
    "return",
    "sbyte",
    "sealed",
    "short",
    "sizeof",
    "stackalloc",
    "static",
    "string",
    "struct",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "uint",
    "long",
    "unchecked",
    "unsafe",
    "ushort",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
}

# Classes excluded from generation.
EXCLUDED: tuple[str, ...] = ()

# Object methods that are shadowed by the generated wrapper (require `new` keyword).
_OBJECT_SHADOW_METHODS: set[str] = {"GetType", "ToString", "GetHashCode", "Equals", "MemberwiseClone"}


# Helper Utilities


def _parse_godot_type(raw: str) -> tuple[str, bool, str]:
    """Split a Godot type string into (base type, is_enum, enum_name)."""
    if raw.startswith("typedarray::"):
        return "Array", False, ""
    if raw.startswith("enum::") or raw.startswith("bitfield::"):
        inner = raw.split("::", 1)[1]
        return inner, True, inner
    return raw, False, ""


def normalize_type_name(t: str) -> str:
    """Strip prefixes (enum::, bitfield::, typedarray::) and namespace qualifiers."""
    for prefix in ("enum::", "bitfield::", "typedarray::"):
        if t.startswith(prefix):
            return "Array" if prefix == "typedarray::" else t[len(prefix) :]
    if "::" in t:
        t = t.split("::")[-1]
    return t


def sanitize_param(name: str) -> str:
    """Escape a parameter name if it collides with a C# keyword."""
    if not name or not name.strip():
        return "param"
    return f"@{name}" if name in CS_KEYWORDS else name


def sanitize_enum_member(name: str) -> str:
    """Escape an enum member name if necessary."""
    return f"@{name}" if name in CS_KEYWORDS else name


def _sanitize_type(name: str) -> str:
    """Escape a C# type name (may be qualified)."""
    if not name:
        return "object"
    if "." in name:
        parts = name.split(".", 1)
        return f"{_sanitize_type(parts[0])}.{_sanitize_type(parts[1])}"
    return f"@{name}" if name in CS_KEYWORDS else name


def to_pascal(snake: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(w.capitalize() for w in snake.split("_"))


def method_name_cs(method_name: str) -> str:
    """Convert a Godot method name to a C#‑friendly PascalCase name."""
    if method_name.startswith("_"):
        return "_" + to_pascal(method_name[1:])
    return to_pascal(method_name)


def should_exclude(name: str) -> bool:
    """Return True if the class name should be excluded from generation."""
    bare = normalize_type_name(name)
    return any(bare == p or bare.startswith(p) for p in EXCLUDED)


def _property_base_name(method_name: str) -> str | None:
    """If the method is a getter/setter, return the base property name."""
    for prefix in ("get_", "is_"):
        if method_name.startswith(prefix):
            return method_name[len(prefix) :]
    if method_name.startswith("set_"):
        return method_name[len("set_") :]
    return None


def _types_compatible(getter_cs: str, setter_cs: str) -> bool:
    """Check whether getter and setter types are compatible for a property."""
    if getter_cs.lower() == setter_cs.lower():
        return True
    collection_types = {
        "System.Collections.IDictionary",
        "System.Collections.IList",
        "Array",
        "Dictionary<object, object>",
    }
    if getter_cs == "object" and setter_cs not in collection_types:
        return True
    if setter_cs == "object" and getter_cs not in collection_types:
        return True
    if getter_cs == "ulong" and setter_cs not in {"string", "bool", "double", "object"}:
        return True
    if setter_cs == "long" and getter_cs not in {"string", "bool", "double", "object"}:
        return True
    return False


# API JSON Loading
def load_api(json_path: Path) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Load extension_api.json and return (classes, native_structures, global_enums, singletons)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, [], [], []
    return (
        data.get("classes", []),
        data.get("native_structures", []),
        data.get("global_enums", []),
        data.get("singletons", []),
    )


def build_singleton_set(singletons: list[Any]) -> dict[str, str]:
    """Build a mapping from type name → singleton name."""
    return {s["type"]: s["name"] for s in singletons}


def build_enum_to_class_map(class_list: list[Any]) -> dict[str, str]:
    """Map each enum name (possibly qualified) to its owning class."""
    m: dict[str, str] = {}
    for cls in class_list:
        cn = normalize_type_name(cls.get("name", ""))
        for e in cls.get("enums", []):
            en = normalize_type_name(e.get("name", ""))
            m[en] = cn
            m[f"{cn}.{en}"] = cn
    return m


def build_global_enum_set(global_enums: list[Any]) -> set[str]:
    """Collect all global enum names."""
    return {normalize_type_name(e.get("name", "")) for e in global_enums}


def build_global_enum_rename_map(class_list: list[Any]) -> dict[tuple[str, str], str]:
    """Create a mapping (owning_class, original_name) → new name (UPPERCASE)."""
    global_rename: dict[tuple[str, str], str] = {}
    for cls in class_list:
        cname = normalize_type_name(cls.get("name", ""))
        for enum_def in cls.get("enums", []):
            original = enum_def["name"]
            global_rename[(cname, original)] = original.upper()
    return global_rename


# Type Resolution
def _qualify_enum(enum_name: str, current_class: str, enum_owner_map: dict[str, str] | None) -> str:
    """Fully qualify an enum name when it belongs to a different class."""
    if "." in enum_name:
        p = enum_name.split(".", 1)
        return f"{_sanitize_type(p[0])}.{_sanitize_type(p[1])}"
    owner = enum_owner_map.get(enum_name) if enum_owner_map else None
    if owner and owner != current_class:
        return f"{_sanitize_type(owner)}.{_sanitize_type(enum_name)}"
    return _sanitize_type(enum_name)


def resolve_public_type(
    raw_type: Any,
    class_names: set[str],
    enum_type: Any = None,
    current_class: Any = None,
    enum_owner_map: dict[str, str] | None = None,
) -> str:
    """Convert a Godot type descriptor into a C# type name."""
    if isinstance(raw_type, str) and "*" in raw_type:
        return "nint"
    if isinstance(raw_type, str):
        cs_hint, raw_is_enum, raw_enum_name = _parse_godot_type(raw_type)
        if raw_is_enum:
            return _qualify_enum(raw_enum_name, current_class, enum_owner_map)
        if cs_hint == "Array" and raw_type.startswith("typedarray::"):
            return "Array"
        raw_type = cs_hint
    if enum_type:
        return _qualify_enum(normalize_type_name(enum_type), current_class, enum_owner_map)
    if isinstance(raw_type, str):
        if "." in raw_type:
            p = raw_type.split(".", 1)
            return f"{_sanitize_type(p[0])}.{_sanitize_type(p[1])}"
        if enum_owner_map and raw_type in enum_owner_map:
            return _qualify_enum(raw_type, current_class, enum_owner_map)
        if raw_type == "Object":
            return "GodotObject"
        if raw_type in class_names:
            return _sanitize_type(raw_type)
        if raw_type in TYPE_MAP:
            return TYPE_MAP[raw_type]
        if raw_type and raw_type[0].isupper():
            return _sanitize_type(raw_type)
    return TYPE_MAP.get(raw_type, "object")


def _apply_enum_rename(
    pt: str, cs_class: str, enum_rename: dict[str, str], global_enum_rename: dict[tuple[str, str], str] | None = None
) -> str:
    """Apply per‑class or global enum renaming to a type name."""
    if not pt:
        return pt
    if "." in pt:
        owner, bare = pt.split(".", 1)
        if owner == cs_class and bare in enum_rename:
            return f"{owner}.{enum_rename[bare]}"
        if global_enum_rename and (owner, bare) in global_enum_rename:
            return f"{owner}.{global_enum_rename[(owner, bare)]}"
    else:
        if pt in enum_rename:
            return enum_rename[pt]
    return pt


def is_struct_type(t: str) -> bool:
    """Return True if the type is a Godot struct that we flatten."""
    norm = t.split(".")[-1] if "." in t else t
    return norm in STRUCT_LAYOUT


def is_enum_type(public_type: str, enum_owner_map: dict[str, str], global_enum_set: set[str] | None = None) -> bool:
    """Return True if the type is an enum (including qualified enums)."""
    if "." in public_type:
        return True
    bare = public_type.split(".")[-1]
    if enum_owner_map and bare in enum_owner_map:
        return True
    if global_enum_set and bare in global_enum_set:
        return True
    return False


def is_any_marshal(t: str) -> bool:
    """Return True if the type is marshaled as an opaque object."""
    norm = t.split(".")[-1] if "." in t else t
    return norm in ANY_MARSHAL_TYPES


def is_godot_obj(t: str, class_names: set[str]) -> bool:
    """Return True if the type is a Godot object (not a primitive, struct, or collection)."""
    norm = t.split(".")[-1] if "." in t else t
    _NOT_GODOT_OBJ = {
        "object",
        "string",
        "bool",
        "double",
        "float",
        "ulong",
        "int",
        "short",
        "sbyte",
        "long",
        "uint",
        "ushort",
        "byte",
        "nint",
        "void",
        "Array",
        "Dictionary<object, object>",
    }
    if norm in _NOT_GODOT_OBJ:
        return False
    if norm.endswith("[]"):
        return False
    return norm in class_names or norm in {"GodotObject", "Object"}


# Argument Packing (C# → JS)
def flatten_struct_access(expr: str, type_name: str) -> list[tuple[str, str]]:
    """Recursively flatten a Godot struct into (expression, field_type) pairs."""
    norm = type_name.split(".")[-1] if "." in type_name else type_name
    layout = STRUCT_LAYOUT.get(norm)
    if layout is None:
        return [(expr, "object")]

    out: list[tuple[str, str]] = []
    for field_name, child_type in layout:
        sub = f"{expr}.{field_name}"
        if child_type is None:
            # leaf field – look up the JS bridge type
            js_fields = STRUCT_JS_FIELDS.get(norm)
            if js_fields:
                for js_field, js_type in js_fields:
                    if js_field == field_name:
                        out.append((sub, js_type))
                        break
                else:
                    out.append((sub, "float"))
            else:
                out.append((sub, "float"))
        else:
            out.extend(flatten_struct_access(sub, child_type))
    return out


def _cast_expr(expr: str, field_type: str) -> str:
    """Wrap an expression with the proper C# cast for JS interop."""
    if field_type in ("double", "float"):
        return f"(double)({expr})" if field_type == "double" else f"(float)({expr})"
    if field_type == "int":
        return f"(int)(ulong)(double)({expr})"
    return expr


def pack_arg(
    name: str, public_type: str, class_names: set[str], enum_owner_map: dict[str, str], global_enum_set: set[str]
) -> list[str]:
    """
    Return a list of C# expressions that represent a single argument when packed into object[].

    - Primitives (bool, double, string) → as‑is.
    - Objects → (double)(ulong)obj.Id.
    - Structs → flattened field by field.
    - Callable → two primitives: (double)(ulong)obj.Target, obj.Method.
    """
    norm = public_type.split(".")[-1] if "." in public_type else public_type

    # Simple pass‑through
    if norm in {"bool", "double", "string", "object"}:
        return [name]

    # Collections
    if norm == "Array":
        return [f"new object[] {{ VariantPacker.Flatten({name}.Select(v => v.Obj).ToArray()), (double){name}.Count }}"]
    if norm == "Dictionary<object, object>":
        return [
            f'new object[] {{ {name}.Keys.Select(k => k?.ToString() ?? "").ToArray(), VariantPacker.Flatten({name}.Values.ToArray()), (double){name}.Count }}'
        ]

    # Arrays of primitives (Packed*Array) are sent as Span<byte> via CallGodotPacked – never hit this branch.
    if norm.endswith("[]"):
        return [name]

    # ── Callable: flattened to target pointer + method name ──
    if norm == "Callable":
        return [f"(double)(ulong){name}.Target?.Id", f"{name}.Method"]

    # Integer types → cast to double via ulong
    if norm in {"ulong", "int", "short", "sbyte", "long", "uint", "ushort", "byte", "nint"}:
        return [f"(double)(ulong){name}"]

    # RID → its numeric id
    if norm == "RID":
        return [f"(double)(ulong){name}.Id"]

    # Opaque marshal types (legacy – kept for future use, e.g. varargs)
    if is_any_marshal(norm):
        return [name]

    # Godot structs
    if is_struct_type(norm):
        flattened = flatten_struct_access(name, norm)
        return [_cast_expr(expr, ftype) for expr, ftype in flattened]

    # Godot objects
    if is_godot_obj(norm, class_names):
        return [f"(double)(ulong){name}.Id"]

    # Enums
    if is_enum_type(public_type, enum_owner_map, global_enum_set):
        return [f"(double)(ulong){name}"]

    # Fallback – any other type
    return [f"(double)(ulong){name}"]


# Return Value Decoding (JS → C#)


def _get_struct_builder(norm: str) -> Any:
    """Return a function that writes C# code to reconstruct a specific Godot struct."""

    def rect2_builder(lines: list[str], vars: dict[str, str], _: str) -> None:
        lines.append(f"                    var __pos__ = new Vector2({vars['X']}, {vars['Y']});")
        lines.append(f"                    var __size__ = new Vector2({vars['Width']}, {vars['Height']});")
        lines.append("                    return new Rect2(__pos__, __size__);")

    def rect2i_builder(lines: list[str], vars: dict[str, str], _: str) -> None:
        lines.append(f"                    var __pos__ = new Vector2i({vars['X']}, {vars['Y']});")
        lines.append(f"                    var __size__ = new Vector2i({vars['Width']}, {vars['Height']});")
        lines.append("                    return new Rect2i(__pos__, __size__);")

    def aabb_builder(lines: list[str], vars: dict[str, str], _: str) -> None:
        lines.append(
            f"                    var __pos__ = new Vector3({vars['Position.X']}, {vars['Position.Y']}, {vars['Position.Z']});"
        )
        lines.append(
            f"                    var __size__ = new Vector3({vars['Size.X']}, {vars['Size.Y']}, {vars['Size.Z']});"
        )
        lines.append("                    return new AABB(__pos__, __size__);")

    def transform2d_builder(lines: list[str], vars: dict[str, str], _: str) -> None:
        lines.append(f"                    var __x__ = new Vector2({vars['Column0.X']}, {vars['Column0.Y']});")
        lines.append(f"                    var __y__ = new Vector2({vars['Column1.X']}, {vars['Column1.Y']});")
        lines.append(f"                    var __orig__ = new Vector2({vars['Origin.X']}, {vars['Origin.Y']});")
        lines.append("                    return new Transform2D(__x__, __y__, __orig__);")

    def basis_builder(lines: list[str], vars: dict[str, str], _: str) -> None:
        lines.append(
            f"                    var __col0__ = new Vector3({vars['Column0.X']}, {vars['Column0.Y']}, {vars['Column0.Z']});"
        )
        lines.append(
            f"                    var __col1__ = new Vector3({vars['Column1.X']}, {vars['Column1.Y']}, {vars['Column1.Z']});"
        )
        lines.append(
            f"                    var __col2__ = new Vector3({vars['Column2.X']}, {vars['Column2.Y']}, {vars['Column2.Z']});"
        )
        lines.append("                    return new Basis(__col0__, __col1__, __col2__);")

    def transform3d_builder(lines: list[str], vars: dict[str, str], _: str) -> None:
        basis_vars = {k: vars[k] for k in vars if k.startswith("Basis.")}
        lines.append(
            "                    var __basis__ = new Basis("
            f"new Vector3({basis_vars['Basis.Column0.X']}, {basis_vars['Basis.Column0.Y']}, {basis_vars['Basis.Column0.Z']}),"
            f"new Vector3({basis_vars['Basis.Column1.X']}, {basis_vars['Basis.Column1.Y']}, {basis_vars['Basis.Column1.Z']}),"
            f"new Vector3({basis_vars['Basis.Column2.X']}, {basis_vars['Basis.Column2.Y']}, {basis_vars['Basis.Column2.Z']}));"
        )
        lines.append(
            f"                    var __origin__ = new Vector3({vars['Origin.X']}, {vars['Origin.Y']}, {vars['Origin.Z']});"
        )
        lines.append("                    return new Transform3D(__basis__, __origin__);")

    def plane_builder(lines: list[str], vars: dict[str, str], _: str) -> None:
        lines.append(
            f"                    var __normal__ = new Vector3({vars['Normal.X']}, {vars['Normal.Y']}, {vars['Normal.Z']});"
        )
        lines.append(f"                    return new Plane(__normal__, {vars['D']});")

    builders = {
        "Rect2": rect2_builder,
        "Rect2i": rect2i_builder,
        "AABB": aabb_builder,
        "Transform2D": transform2d_builder,
        "Basis": basis_builder,
        "Transform3D": transform3d_builder,
        "Plane": plane_builder,
    }
    return builders.get(norm)


def _build_struct_return(tmp: str, norm: str, public_ret: str) -> list[str]:
    """Generate C# code to reconstruct a Godot struct from a JSObject."""
    fields = STRUCT_JS_FIELDS.get(norm)
    if fields is None:
        return [f"            return ({public_ret}){tmp};"]

    layout = STRUCT_LAYOUT.get(norm)
    is_simple = all(child_type is None for _, child_type in layout) if layout else True

    lines = [
        f"            if ({tmp} is System.Runtime.InteropServices.JavaScript.JSObject __js__)",
        "            {",
        "                try",
        "                {",
    ]

    field_vars: dict[str, str] = {}
    for i, (field, ftype) in enumerate(fields):
        vname = f"__f{i}__"
        lines.append(f'                    var {vname} = ({ftype})__js__.GetPropertyAsDouble("{field}");')
        field_vars[field] = vname

    if is_simple:
        ctor_args = [field_vars[f] for f, _ in fields]
        lines.append(f"                    return new {norm}({', '.join(ctor_args)});")
    else:
        builder = _get_struct_builder(norm)
        if builder is None:
            return [f"            return default({public_ret});"]
        builder(lines, field_vars, norm)

    lines.append("                }")
    lines.append("                finally { __js__.Dispose(); }")
    lines.append("            }")
    lines.append("            return default;")
    return lines


def build_return_statement(
    native_call: str,
    public_ret: str,
    class_names: set[str],
    enum_owner_map: dict[str, str],
    global_enum_set: set[str],
    used_names: set[str] | None = None,
) -> list[str]:
    """
    Generate C# code that takes the result of a CallGodot() invocation and
    converts it to the correct C# return type.
    """
    tmp = "__ret__"
    if used_names:
        i = 0
        while tmp in used_names:
            tmp = f"__ret{i}__"
            i += 1
    norm = public_ret.split(".")[-1] if "." in public_ret else public_ret

    # ── Packed‑array return types ──────────────────────────────────────
    _PACKED_RETURN = {
        "byte[]": (None, False),
        "int[]": ("(int)Convert.ToDouble(x)", False),
        "long[]": ("(long)Convert.ToDouble(x)", False),
        "float[]": ("(float)Convert.ToDouble(x)", False),
        "double[]": ("Convert.ToDouble(x)", False),
        "string[]": ('x?.ToString() ?? ""', False),
        "Vector2[]": ('new Vector2((float)js.GetPropertyAsDouble("X"), (float)js.GetPropertyAsDouble("Y"))', True),
        "Vector3[]": (
            'new Vector3((float)js.GetPropertyAsDouble("X"), (float)js.GetPropertyAsDouble("Y"), (float)js.GetPropertyAsDouble("Z"))',
            True,
        ),
        "Vector4[]": (
            'new Vector4((float)js.GetPropertyAsDouble("X"), (float)js.GetPropertyAsDouble("Y"), (float)js.GetPropertyAsDouble("Z"), (float)js.GetPropertyAsDouble("W"))',
            True,
        ),
        "Color[]": (
            'new Color((float)js.GetPropertyAsDouble("R"), (float)js.GetPropertyAsDouble("G"), (float)js.GetPropertyAsDouble("B"), (float)js.GetPropertyAsDouble("A"))',
            True,
        ),
    }

    if norm in _PACKED_RETURN:
        conv_expr, needs_js = _PACKED_RETURN[norm]
        lines = [f"            var {tmp} = {native_call};"]
        if norm == "byte[]":
            lines.append(f"            return {tmp} as byte[] ?? new byte[0];")
            return lines

        raw = f"__raw_{tmp}__"
        lines.append(f"            var {raw} = {tmp} as object[] ?? new object[0];")
        if needs_js:
            lines.append(f"            return System.Array.ConvertAll({raw}, x =>")
            lines.append("            {")
            lines.append("                var js = x as System.Runtime.InteropServices.JavaScript.JSObject;")
            lines.append(f"                return {conv_expr};")
            lines.append("            });")
        else:
            lines.append(f"            return System.Array.ConvertAll({raw}, x => {conv_expr});")
        return lines

    # ── Simple types ───────────────────────────────────────────────────
    if norm == "void":
        return [f"            {native_call};"]
    if norm == "bool":
        return [
            f"            var {tmp} = {native_call};",
            f"            return {tmp} is bool __b__ ? __b__ : System.Convert.ToBoolean({tmp});",
        ]
    if norm == "double":
        return [f"            var {tmp} = {native_call};", f"            return System.Convert.ToDouble({tmp});"]
    if norm == "string":
        return [f"            var {tmp} = {native_call};", f"            return {tmp}?.ToString() ?? string.Empty;"]
    if norm == "nint":
        return [
            f"            var {tmp} = {native_call};",
            f"            return (nint)(ulong)System.Convert.ToDouble({tmp});",
        ]
    if norm == "ulong":
        return [f"            var {tmp} = {native_call};", f"            return (ulong)System.Convert.ToDouble({tmp});"]
    if norm == "RID":
        return [
            f"            var {tmp} = {native_call};",
            f"            return new RID(unchecked((ulong)System.Convert.ToDouble({tmp})));",
        ]

    # ── Godot structs ──────────────────────────────────────────────────
    if norm in STRUCT_JS_FIELDS:
        lines = [f"            var {tmp} = {native_call};"]
        lines.extend(_build_struct_return(tmp, norm, public_ret))
        return lines

    # ── Godot objects ──────────────────────────────────────────────────
    if is_godot_obj(norm, class_names):
        return [
            f"            var {tmp} = {native_call};",
            f"            return new {norm}(unchecked((ulong)System.Convert.ToDouble({tmp})));",
        ]

    # ── Opaque marshal types (Callable, Signal, etc.) ─────────────────
    if is_any_marshal(norm):
        return [f"            var {tmp} = {native_call};", f"            return ({norm}){tmp};"]

    # ── Enums ──────────────────────────────────────────────────────────
    if is_enum_type(public_ret, enum_owner_map, global_enum_set):
        return [
            f"            var {tmp} = {native_call};",
            f"            return ({public_ret})(ulong)System.Convert.ToDouble({tmp});",
        ]

    # ── Collections (Array, Dictionary) ────────────────────────────────
    if norm == "Array":
        return [
            f"            var {tmp} = {native_call};",
            f"            var {tmp}_raw = {tmp} as object[] ?? new object[0];",
            f"            return new Array({tmp}_raw.Select(Variant.CreateFrom));",
        ]
    if norm == "Dictionary<object, object>":
        return [
            f"            var {tmp} = {native_call};",
            f"            var {tmp}_raw = {tmp} as object[] ?? new object[0];",
            f"            var {tmp}_dict = new Dictionary<object, object>();",
            f"            for (int _i_ = 0; _i_ + 1 < {tmp}_raw.Length; _i_ += 2)",
            f"                {tmp}_dict[{tmp}_raw[_i_]] = {tmp}_raw[_i_ + 1];",
            f"            return {tmp}_dict;",
        ]
    if norm == "object":
        return [f"            var {tmp} = {native_call};", f"            return ({norm}){tmp};"]

    # Fallback for unrecognized array types
    if norm.endswith("[]"):
        return [f"            var {tmp} = {native_call};", f"            return ({norm}){tmp};"]

    return [
        f"            var {tmp} = {native_call};",
        f"            return ({public_ret})(ulong)System.Convert.ToDouble({tmp});",
    ]


def write_class_enums(lines: list[str], enums: list[dict[str, Any]], enum_rename: dict[str, str]) -> None:
    """Append C# enum declarations to the class body."""
    for enum_def in enums:
        original_name = enum_def["name"]
        enum_name = _sanitize_type(enum_rename.get(original_name, original_name))
        values = enum_def.get("values", [])
        if not values:
            continue
        lines.append(f"        public enum {enum_name} : long")
        lines.append("        {")
        for v in values:
            lines.append(f"            {sanitize_enum_member(v['name'])} = {v['value']},")
        lines.append("        }")
        lines.append("")


# Extra static methods for the Engine class.
_ENGINE_EXTRA_METHODS = """        public static ulong get_SceneTree_singleton()
        {
            var __ret__ = CallGodot("_Engine_get_SceneTree_singleton", new object[] { });
            return unchecked((ulong)System.Convert.ToDouble(__ret__));
        }

        public static ulong get_object_singleton(string name)
        {
            var __ret__ = CallGodot("_Engine_get_object_singleton", new object[] { name });
            return unchecked((ulong)System.Convert.ToDouble(__ret__));
        }
"""


def _resolve_signal_arg_type(
    arg: dict[str, Any],
    class_names: set[str],
    enum_owner_map: dict[str, str],
    global_enum_set: set[str],
    current_class: str,
) -> str:
    return resolve_public_type(arg.get("type", "Variant"), class_names, None, current_class, enum_owner_map)


def _format_method_default(default_value: Any, public_type: str) -> str | None:
    """Format a Godot default value as a C# literal expression."""
    if default_value is None:
        return None

    norm = public_type.split(".")[-1] if "." in public_type else public_type
    raw = str(default_value).strip()

    if norm == "bool":
        if isinstance(default_value, bool):
            return "true" if default_value else "false"
        lowered = raw.lower()
        return lowered if lowered in ("true", "false") else None
    if norm == "string":
        if raw in ("", '""', '""'):
            return '""'
        if raw.startswith('"') and raw.endswith('"'):
            return raw
        return None
    if norm in {"double", "float"}:
        try:
            float(raw.rstrip("fFdD"))
            return raw.rstrip("fFdD")
        except ValueError:
            return None
    if norm in {"ulong", "int", "short", "sbyte", "uint", "ushort", "byte", "nint"}:
        try:
            int(raw, 0)
            return raw
        except ValueError:
            return None
    if "." in public_type:  # enum
        try:
            int(raw, 0)
            return f"({public_type}){raw}"
        except ValueError:
            return None

    return None


# Main Class Wrapper Generator


def write_class_wrapper(
    out_dir: Path,
    class_name: str,
    methods: list[dict[str, Any]],
    enums: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    base_class: str,
    class_names: set[str],
    enum_owner_map: dict[str, str],
    global_enum_set: set[str],
    is_singleton: bool = False,
    singleton_name: str = "",
    is_instantiable: bool = False,
    global_enum_rename: dict[tuple[str, str], str] | None = None,
) -> int:
    """Write one C# file for a Godot class. Returns the number of emitted methods."""

    class_name = normalize_type_name(class_name)
    cs_class = _sanitize_type(class_name)
    base_class = normalize_type_name(base_class)
    base = "GodotObject" if base_class in ("Object", "") else _sanitize_type(base_class)

    effective_enum_rename: dict[tuple[str, str], str] = global_enum_rename if global_enum_rename is not None else {}

    # ── Local enum rename map (uppercase to avoid collisions) ──────────
    enum_rename: dict[str, str] = {}
    for enum_def in enums:
        original = enum_def["name"]
        enum_rename[original] = original.upper()

    # ── Signal pascal names (used for collision detection) ────────────
    signal_pascal_names: set[str] = set()
    for sig in signals:
        sig_name = sig.get("name", "")
        if sig_name:
            signal_pascal_names.add(to_pascal(sig_name))

    # ── Emitted method names (also for collision detection) ────────────
    emitted_method_count = 0
    emitted_method_names: set[str] = set()
    for m in methods:
        mname = m["name"]
        if mname.startswith("_") and mname not in ("_ready", "_process", "_physics_process"):
            continue
        emitted_method_names.add(sanitize_param(method_name_cs(mname)))
        emitted_method_count += 1

    # ── Property detection ─────────────────────────────────────────────
    property_map: dict[str, dict[str, Any]] = {}
    for m in methods:
        mname = m["name"]
        base_prop = _property_base_name(mname)
        if base_prop is None:
            continue
        prop_pascal = to_pascal(base_prop)
        if not prop_pascal or not (prop_pascal[0].isalpha() or prop_pascal[0] == "_"):
            continue
        if prop_pascal not in property_map:
            property_map[prop_pascal] = {"getter": None, "setter": None, "cs_type": None}
        real_args = m.get("arguments", [])
        if mname.startswith("set_") and len(real_args) >= 1:
            extra_args = real_args[1:]
            if all(a.get("default_value") is not None for a in extra_args):
                property_map[prop_pascal]["setter"] = m
        elif mname.startswith(("get_", "is_")) and len(real_args) == 0:
            property_map[prop_pascal]["getter"] = m

    # ── Validate and filter properties ─────────────────────────────────
    props_to_remove: list[str] = []
    for prop_pascal, acc in property_map.items():
        getter = acc["getter"]
        setter = acc["setter"]

        # Collision with signals, built‑in members, class name, or emitted methods
        if (
            prop_pascal in signal_pascal_names
            or prop_pascal == "Id"
            or prop_pascal == cs_class
            or prop_pascal in emitted_method_names
        ):
            props_to_remove.append(prop_pascal)
            continue

        if getter is None:
            props_to_remove.append(prop_pascal)
            continue

        ret_info = getter.get("return_value", {})
        getter_cs = resolve_public_type(
            ret_info.get("type", "void"),
            class_names,
            ret_info.get("enum_type"),
            class_name,
            enum_owner_map,
        )
        getter_cs = _apply_enum_rename(getter_cs, cs_class, enum_rename, effective_enum_rename)
        if getter_cs == "void":
            props_to_remove.append(prop_pascal)
            continue

        if setter is not None:
            set_arg = setter["arguments"][0]
            setter_cs = resolve_public_type(
                set_arg.get("type", ""),
                class_names,
                set_arg.get("enum_type"),
                class_name,
                enum_owner_map,
            )
            setter_cs = _apply_enum_rename(setter_cs, cs_class, enum_rename, effective_enum_rename)
            if not _types_compatible(getter_cs, setter_cs):
                props_to_remove.append(prop_pascal)
                continue

        acc["cs_type"] = getter_cs

    for p in props_to_remove:
        del property_map[p]

    # ── Constructor lines ──────────────────────────────────────────────
    if class_name == "Object":
        ctor_lines = [""]
    else:
        ctor_lines = [f"        public {cs_class}(ulong id) : base(id) {{ }}"]

    if is_singleton and class_name != "SceneTree":
        sname = singleton_name or class_name
        ctor_lines.append(f'        public {cs_class}() : base(Engine.get_object_singleton("{sname}")) {{ }}')
    elif is_instantiable and class_name != "SceneTree":
        if class_name != "Object":
            ctor_lines.append(f'        public {cs_class}() : base(ClassDB.Instantiate("{class_name}")) {{ }}')
    elif class_name == "SceneTree":
        ctor_lines.append(f"        public {cs_class}() : base(Engine.get_SceneTree_singleton()) {{ }}")

    # ── File header ────────────────────────────────────────────────────
    if class_name == "Object":
        lines = [
            "using System;",
            "using System.Collections.Generic;",
            "using System.Runtime.InteropServices;",
            "using static Godot.GodotBridge;",
            "",
            "namespace Godot",
            "{",
            "    public class GodotObject",
            "    {",
            " public ulong Id { get; protected set; }",
            "",
            "public GodotObject(ulong id)",
            "{",
            "    Id = id;",
            "}",
        ]
    else:
        lines = [
            "using System;",
            "using System.Collections.Generic;",
            "using System.Runtime.InteropServices;",
            "using static Godot.GodotBridge;",
            "using System.Linq;",
            "",
            "namespace Godot",
            "{",
            f"    public partial class {cs_class} : {base}",
            "    {",
        ]
    lines.extend(ctor_lines)
    lines.append("")

    if is_singleton:
        sname = singleton_name or class_name
        lines += [
            "        private static ulong _singletonId;",
            "        private static ulong SingletonId",
            "        {",
            "            get",
            "            {",
            "                if (_singletonId == 0)",
            f'                    _singletonId = Engine.get_object_singleton("{sname}");',
            "                return _singletonId;",
            "            }",
            "        }",
            "",
        ]

    if class_name == "Engine":
        lines.append(_ENGINE_EXTRA_METHODS)

    write_class_enums(lines, enums, enum_rename)

    # ── Signal events ──────────────────────────────────────────────────
    for sig in signals:
        sig_name = sig.get("name", "")
        if not sig_name:
            continue
        sig_pascal = to_pascal(sig_name)
        cs_arg_types = [
            _resolve_signal_arg_type(arg, class_names, enum_owner_map, global_enum_set, class_name)
            for arg in sig.get("arguments", [])
        ]
        delegate_type = f"Action<{', '.join(cs_arg_types)}>" if cs_arg_types else "Action"
        lines += [
            f"        public event {delegate_type} {sig_pascal}",
            "        {",
            f'            add    => SignalManager.Subscribe(Id, "{sig_name}", value);',
            f'            remove => SignalManager.Unsubscribe(Id, "{sig_name}", value);',
            "        }",
            "",
        ]

    # ── Method generation ──────────────────────────────────────────────
    for m in methods:
        mname = m["name"]
        if mname.startswith("_") and mname not in ("_ready", "_process", "_physics_process"):
            continue
        if method_name_cs(mname) in signal_pascal_names:
            continue

        is_static = m.get("is_static", False) or is_singleton
        ret_info = m.get("return_value", {})
        public_ret = resolve_public_type(
            ret_info.get("type", "void"),
            class_names,
            ret_info.get("enum_type"),
            class_name,
            enum_owner_map,
        )
        public_ret = _apply_enum_rename(public_ret, cs_class, enum_rename, effective_enum_rename)
        is_vararg = m.get("is_vararg", False)

        # --- Packed-array bulk methods (1‑6 packed args, void return) ---
        packed_args_info = [
            (i, a) for i, a in enumerate(m.get("arguments", [])) if a.get("type", "") in PACKED_BUFFER_TYPES
        ]
        packed_count = len(packed_args_info)
        ret_is_void = public_ret == "void"

        if not is_vararg and ret_is_void and 1 <= packed_count <= 6:
            args_list = m.get("arguments", [])
            cs_method = sanitize_param(method_name_cs(mname))

            public_params: list[str] = []
            param_names: list[str] = []
            used: set[str] = set()
            for a in args_list:
                raw_name = (a.get("name", "") or "").strip() or f"p{len(param_names)}"
                pname = sanitize_param(raw_name)
                c = 1
                while pname in used:
                    pname = f"{pname}_{c}"
                    c += 1
                used.add(pname)
                param_names.append(pname)

                if a.get("type", "") in PACKED_BUFFER_TYPES:
                    elem_type = TYPE_MAP.get(a["type"], "byte[]")
                    public_params.append(f"{elem_type} {pname}")
                else:
                    pt = resolve_public_type(
                        a.get("type", ""),
                        class_names,
                        a.get("enum_type"),
                        class_name,
                        enum_owner_map,
                    )
                    pt = _apply_enum_rename(pt, cs_class, enum_rename, effective_enum_rename)
                    public_params.append(f"{pt} {pname}")

            static_kw = "static " if is_static else ""
            lines.append(f"        public {static_kw}void {cs_method}({', '.join(public_params)})")
            lines.append("        {")

            fixed_items: list[str] = []
            if not is_static:
                fixed_items.append("(double)(ulong)SingletonId" if is_singleton else "(double)(ulong)Id")

            buffer_decls: list[str] = []
            buf_idx = 0
            for a, pname in zip(args_list, param_names):
                if a.get("type", "") in PACKED_BUFFER_TYPES:
                    bname = f"__buf{buf_idx}__"
                    buffer_decls.append(f"            Span<byte> {bname} = MemoryMarshal.AsBytes({pname}.AsSpan());")
                    buf_idx += 1
                else:
                    pt = resolve_public_type(
                        a.get("type", ""),
                        class_names,
                        a.get("enum_type"),
                        class_name,
                        enum_owner_map,
                    )
                    pt = _apply_enum_rename(pt, cs_class, enum_rename, effective_enum_rename)
                    fixed_items.extend(pack_arg(pname, pt, class_names, enum_owner_map, global_enum_set))

            for decl in buffer_decls:
                lines.append(decl)

            buf_names = [f"__buf{i}__" for i in range(packed_count)]
            fn_name = f'"_{class_name}_{mname}"'

            if packed_count == 1 and len(args_list) == 1 and not is_static:
                lines.append(f"            CallGodotPacked({fn_name}, {fixed_items[0]}, {buf_names[0]});")
            else:
                call_name = f"CallGodotPacked{packed_count}"
                args_literal = ", ".join(fixed_items)
                lines.append(f"            var __args__ = new object[] {{ {args_literal} }};")
                lines.append(f"            {call_name}({fn_name}, __args__, {', '.join(buf_names)});")

            lines.append("        }")
            lines.append("")
            continue

        # --- Special case: ClassDB.instantiate ---
        if class_name == "ClassDB" and mname == "instantiate":
            lines += [
                "        public static ulong Instantiate(string @class)",
                "        {",
                '            ulong classDbId = Engine.get_object_singleton("ClassDB");',
                '            var __ret__ = CallGodot("_ClassDB_instantiate", new object[] { (double)classDbId, @class });',
                "            return unchecked((ulong)System.Convert.ToDouble(__ret__));",
                "        }",
                "",
            ]
            continue

        # --- General method ---
        public_params = []
        arg_meta: list[tuple[str, str]] = []
        used = set()

        resolved: list[tuple[str, str, str | None]] = []
        for a in m.get("arguments", []):
            raw_name = (a.get("name", "") or "").strip() or f"p{len(resolved)}"
            base_pname = sanitize_param(raw_name)
            pname = base_pname
            c = 1
            while pname in used:
                pname = f"{base_pname}_{c}"
                c += 1
            used.add(pname)

            pt = resolve_public_type(
                a.get("type", ""),
                class_names,
                a.get("enum_type"),
                class_name,
                enum_owner_map,
            )
            pt = _apply_enum_rename(pt, cs_class, enum_rename, effective_enum_rename)
            default_expr = _format_method_default(a.get("default_value", None), pt)
            resolved.append((pname, pt, default_expr))

        last_required = -1
        for i, (_, _, d) in enumerate(resolved):
            if d is None:
                last_required = i

        for i, (pname, pt, default_expr) in enumerate(resolved):
            if i <= last_required:
                default_expr = None
            if default_expr is not None:
                public_params.append(f"{pt} {pname} = {default_expr}")
            else:
                public_params.append(f"{pt} {pname}")
            arg_meta.append((pname, pt))

        if is_vararg:
            vararg_pname = "varargs"
            suffix = 0
            while vararg_pname in used:
                vararg_pname = f"varargs{suffix}"
                suffix += 1
            public_params.append(f"params object[] {vararg_pname}")
            used.add(vararg_pname)

        cs_method = sanitize_param(method_name_cs(mname))
        is_virtual = m.get("is_virtual", False)
        static_kw = "static " if is_static else ""
        virtual_kw = "virtual " if is_virtual and not is_static else ""
        new_kw = "new " if cs_method in _OBJECT_SHADOW_METHODS else ""

        lines.append(
            f"        public {new_kw}{static_kw}{virtual_kw}{public_ret} {cs_method}({', '.join(public_params)})"
        )
        lines.append("        {")

        fixed_items = []
        if is_static:
            if is_singleton:
                fixed_items.append("(double)(ulong)SingletonId")
        else:
            fixed_items.append("(double)(ulong)SingletonId" if is_singleton else "(double)(ulong)Id")
        for pname, pt in arg_meta:
            fixed_items.extend(pack_arg(pname, pt, class_names, enum_owner_map, global_enum_set))

        if is_vararg:
            fixed_literal = ", ".join(fixed_items) if fixed_items else ""
            lines.append(
                f"            var __args__ = new System.Collections.Generic.List<object> {{ {fixed_literal} }};"
            )
            lines.append(f"            __args__.Add(VariantPacker.Flatten({vararg_pname}));")
            native_call = f'CallGodot("_{class_name}_{mname}", __args__.ToArray())'
        else:
            args_literal = ", ".join(fixed_items) if fixed_items else ""
            native_call = f'CallGodot("_{class_name}_{mname}", new object[] {{ {args_literal} }})'

        for line in build_return_statement(native_call, public_ret, class_names, enum_owner_map, global_enum_set, used):
            lines.append(line)
        lines.append("        }")
        lines.append("")

    # ── Properties ─────────────────────────────────────────────────────
    _KNOWN_BASE_MEMBERS = {"Id", "Ready"}
    for prop_pascal, acc in property_map.items():
        getter = acc["getter"]
        setter = acc["setter"]
        prop_cs = acc["cs_type"]
        getter_cs_name = sanitize_param(method_name_cs(getter["name"]))
        new_kw = "new " if prop_pascal in _KNOWN_BASE_MEMBERS else ""

        if setter is not None:
            setter_cs_name = sanitize_param(method_name_cs(setter["name"]))
            lines += [
                f"        public {new_kw}{prop_cs} {prop_pascal}",
                "        {",
                f"            get => {getter_cs_name}();",
                f"            set => {setter_cs_name}(value);",
                "        }",
                "",
            ]
        else:
            lines += [
                f"        public {new_kw}{prop_cs} {prop_pascal}",
                "        {",
                f"            get => {getter_cs_name}();",
                "        }",
                "",
            ]

    lines.append("    }")
    lines.append("}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if cs_class == "Object":
        (out_dir / "GodotObject.cs").write_text("\n".join(lines))
    else:
        (out_dir / f"{cs_class}.cs").write_text("\n".join(lines))
    return emitted_method_count


# Native Structure Writer


C_TYPE_MAP = {
    "void": "void",
    "int": "int",
    "unsigned int": "uint",
    "ulong": "ulong",
    "unsigned ulong": "ulong",
    "ulong ulong": "ulong",
    "unsigned ulong ulong": "ulong",
    "int8_t": "sbyte",
    "uint8_t": "byte",
    "int16_t": "short",
    "uint16_t": "ushort",
    "int32_t": "int",
    "uint32_t": "uint",
    "int64_t": "ulong",
    "uint64_t": "ulong",
    "float": "float",
    "double": "double",
    "real_t": "double",
    "bool": "bool",
    "char": "char",
    "String": "string",
    "StringName": "string",
    "NodePath": "string",
    "Variant": "object",
    "Rect2": "Rect2",
    "Rect2i": "Rect2i",
    "Vector2": "Vector2",
    "Vector2i": "Vector2i",
    "Vector3": "Vector3",
    "Vector3i": "Vector3i",
    "Vector4": "Vector4",
    "Vector4i": "Vector4i",
    "Color": "Color",
    "RID": "RID",
    "Callable": "Callable",
    "Signal": "Signal",
    "ObjectID": "ulong",
    "Object *": "nint",
}


def _resolve_c_type(c: str) -> tuple[str, bool]:
    s = c.strip()
    if s in C_TYPE_MAP:
        cs = C_TYPE_MAP[s]
        return cs, cs == "nint"
    if "*" in s:
        return "nint", True
    return "ulong", False


def parse_native_struct_format(fmt: str) -> list[tuple[str, str, str | None, int | None, bool]]:
    fields = []
    for part in fmt.split(";"):
        part = part.strip()
        if not part:
            continue
        default_val = None
        eq = re.search(r"\s*=\s*(.+)$", part)
        if eq:
            default_val = eq.group(1).strip()
            part = part[: eq.start()].strip()
        array_size = None
        arr = re.search(r"\[(\d+)\]\s*$", part)
        if arr:
            array_size = int(arr.group(1))
            part = part[: arr.start()].strip()
        m = re.search(r"^(.*?)\s*(\*+)?\s*(\w+)\s*$", part)
        if not m:
            continue
        type_base = m.group(1).strip()
        stars = m.group(2) or ""
        name = m.group(3)
        c_type = (type_base + " " + stars).strip() if stars else type_base
        cs_type, is_ptr = _resolve_c_type(c_type)
        fields.append((cs_type, name, default_val, array_size, is_ptr))
    return fields


def _norm_default(v: str) -> str:
    return "0" if v in ("0.f", "0.", "0.0f", "0.0") else v


def write_native_structs(
    utilities_dir: Path,
    native_structs: list[Any],
    class_names: set[str],
    enum_owner_map: dict[str, str],
) -> None:
    if not native_structs:
        return
    lines = ["using System;", "using System.Runtime.InteropServices;", "", "namespace Godot", "{"]
    for sd in native_structs:
        sname = _sanitize_type(sd["name"])
        fields = parse_native_struct_format(sd["format"])
        has_defaults = any(dv is not None for _, _, dv, _, _ in fields)
        lines += [f"    public struct {sname}", "    {"]
        for cs_type, name, default_val, array_size, is_ptr in fields:
            safe = sanitize_param(name)
            if array_size:
                base = "nint" if is_ptr else cs_type
                lines.append(f"        public {base}[] {safe} = new {base}[{array_size}];")
            elif default_val is not None:
                lines.append(f"        public {cs_type} {safe} = {_norm_default(default_val)};")
            else:
                lines.append(f"        public {cs_type} {safe};")
        if has_defaults or any(a is not None for _, _, _, a, _ in fields):
            lines += ["", f"        public {sname}() {{ }}"]
        lines += ["    }", ""]
    lines.append("}")
    (utilities_dir / "NativeStructs.cs").write_text("\n".join(lines))


# Entry Point


def generate_all(json_path: Path, base_dir: Path) -> None:
    class_list, native_structs, global_enums, singletons = load_api(json_path)
    class_names = {normalize_type_name(c["name"]) for c in class_list}
    enum_owner_map = build_enum_to_class_map(class_list)
    global_enum_set = build_global_enum_set(global_enums)
    singleton_map = build_singleton_set(singletons)
    global_enum_rename = build_global_enum_rename_map(class_list)

    godot_api_dir = base_dir / "GodotApi"
    utilities_dir = base_dir / "utilities"

    write_native_structs(utilities_dir, native_structs, class_names, enum_owner_map)

    generated_classes = 0
    generated_methods = 0

    for cls in class_list:
        cname = normalize_type_name(cls["name"])
        if should_exclude(cname):
            continue
        is_sing = cname in singleton_map
        sing_nm = singleton_map.get(cname, cname)
        is_inst = cls.get("is_instantiable", False) and not is_sing
        emitted = write_class_wrapper(
            godot_api_dir,
            cname,
            cls.get("methods", []),
            cls.get("enums", []),
            cls.get("signals", []),
            cls.get("inherits", "Object"),
            class_names,
            enum_owner_map,
            global_enum_set,
            is_singleton=is_sing,
            singleton_name=sing_nm,
            is_instantiable=is_inst,
            global_enum_rename=global_enum_rename,
        )
        generated_classes += 1
        generated_methods += emitted


if __name__ == "__main__":
    generate_all(
        Path("extension_api.json"),
        Path("NET"),
    )
    print("generated .NET bindings")
