
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union


strings_l10n: Dict[str, str] = {}

CLASS_GROUPS: Dict[str, str] = {
    "global": "Globals",
    "node": "Nodes",
    "resource": "Resources",
    "object": "Other objects",
    "editor": "Editor-only",
    "variant": "Variant types",
}
CLASS_GROUPS_BASE: Dict[str, str] = {
    "node": "Node",
    "resource": "Resource",
    "object": "Object",
    "variant": "Variant",
}
# Sync with editor\register_editor_types.cpp
EDITOR_CLASSES: List[str] = [
    "FileSystemDock",
    "ScriptCreateDialog",
    "ScriptEditor",
    "ScriptEditorBase",
]
# Sync with the types mentioned in https://docs.godotengine.org/en/stable/tutorials/scripting/c_sharp/c_sharp_differences.html
CLASSES_WITH_CSHARP_DIFFERENCES: List[str] = [
    "@GlobalScope",
    "String",
    "StringName",
    "NodePath",
    "Signal",
    "Callable",
    "RID",
    "Basis",
    "Transform2D",
    "Transform3D",
    "Rect2",
    "Rect2i",
    "AABB",
    "Quaternion",
    "Projection",
    "Color",
    "Array",
    "Dictionary",
    "PackedByteArray",
    "PackedColorArray",
    "PackedFloat32Array",
    "PackedFloat64Array",
    "PackedInt32Array",
    "PackedInt64Array",
    "PackedStringArray",
    "PackedVector2Array",
    "PackedVector3Array",
    "PackedVector4Array",
    "Variant",
]

PACKED_ARRAY_TYPES: List[str] = [
    "PackedByteArray",
    "PackedColorArray",
    "PackedFloat32Array",
    "Packedfloat64Array",
    "PackedInt32Array",
    "PackedInt64Array",
    "PackedStringArray",
    "PackedVector2Array",
    "PackedVector3Array",
    "PackedVector4Array",
]
