static func get_type(property: Dictionary, is_return: bool = false) -> String:
	match property.type:
		TYPE_NIL:
			if property.usage & PROPERTY_USAGE_NIL_IS_VARIANT:
				return "Variant"
			return "void" if is_return else "null"
		TYPE_INT:
			if property.usage & PROPERTY_USAGE_CLASS_IS_ENUM:
				if property.class_name == &"":
					return "<unknown enum>"
				return property.class_name
		TYPE_ARRAY:
			if property.hint == PROPERTY_HINT_ARRAY_TYPE:
				if str(property.hint_string).is_empty():
					return "Array[<unknown type>]"
				return "Array[%s]" % property.hint_string
		TYPE_OBJECT:
			if not str(property.class_name).is_empty():
				return property.class_name
	return variant_get_type_name(property.type)

static func get_property_signature(property: Dictionary, is_static: bool = false) -> String:
	var result: String = ""
	if not (property.usage & PROPERTY_USAGE_SCRIPT_VARIABLE):
		printerr("Missing `PROPERTY_USAGE_SCRIPT_VARIABLE` flag.")
	if property.usage & PROPERTY_USAGE_DEFAULT:
		result += "@export "
	if is_static:
		result += "static "
	result += "var " + property.name + ": " + get_type(property)
	return result

static func get_method_signature(method: Dictionary, is_signal: bool = false) -> String:
	var result: String = ""
	if method.flags & METHOD_FLAG_STATIC:
		result += "static "
	result += ("signal " if is_signal else "func ") + method.name + "("

	var args: Array[Dictionary] = method.args
	var default_args: Array = method.default_args
	var mandatory_argc: int = args.size() - default_args.size()
	for i in args.size():
		if i > 0:
			result += ", "
		var arg: Dictionary = args[i]
		result += arg.name + ": " + get_type(arg)
		if i >= mandatory_argc:
			result += " = " + var_to_str(default_args[i - mandatory_argc])

	result += ")"
	if is_signal:
		if get_type(method.return, true) != "void":
			printerr("Signal return type must be `void`.")
	else:
		result += " -> " + get_type(method.return, true)
	return result

static func variant_get_type_name(type: Variant.Type) -> String:
	match type:
		TYPE_NIL:
			return "Nil" # `Nil` in core, `null` in GDScript.
		TYPE_BOOL:
			return "bool"
		TYPE_INT:
			return "int"
		TYPE_FLOAT:
			return "float"
		TYPE_STRING:
			return "String"
		TYPE_VECTOR2:
			return "Vector2"
		TYPE_VECTOR2I:
			return "Vector2i"
		TYPE_RECT2:
			return "Rect2"
		TYPE_RECT2I:
			return "Rect2i"
		TYPE_VECTOR3:
			return "Vector3"
		TYPE_VECTOR3I:
			return "Vector3i"
		TYPE_TRANSFORM2D:
			return "Transform2D"
		TYPE_VECTOR4:
			return "Vector4"
		TYPE_VECTOR4I:
			return "Vector4i"
		TYPE_PLANE:
			return "Plane"
		TYPE_QUATERNION:
			return "Quaternion"
		TYPE_AABB:
			return "AABB"
		TYPE_BASIS:
			return "Basis"
		TYPE_TRANSFORM3D:
			return "Transform3D"
		TYPE_PROJECTION:
			return "Projection"
		TYPE_COLOR:
			return "Color"
		TYPE_STRING_NAME:
			return "StringName"
		TYPE_NODE_PATH:
			return "NodePath"
		TYPE_RID:
			return "RID"
		TYPE_OBJECT:
			return "Object"
		TYPE_CALLABLE:
			return "Callable"
		TYPE_SIGNAL:
			return "Signal"
		TYPE_DICTIONARY:
			return "Dictionary"
		TYPE_ARRAY:
			return "Array"
		TYPE_PACKED_BYTE_ARRAY:
			return "PackedByteArray"
		TYPE_PACKED_INT32_ARRAY:
			return "PackedInt32Array"
		TYPE_PACKED_INT64_ARRAY:
			return "PackedInt64Array"
		TYPE_PACKED_FLOAT32_ARRAY:
			return "PackedFloat32Array"
		TYPE_PACKED_FLOAT64_ARRAY:
			return "PackedFloat64Array"
		TYPE_PACKED_STRING_ARRAY:
			return "PackedStringArray"
		TYPE_PACKED_VECTOR2_ARRAY:
			return "PackedVector2Array"
		TYPE_PACKED_VECTOR3_ARRAY:
			return "PackedVector3Array"
		TYPE_PACKED_COLOR_ARRAY:
			return "PackedColorArray"
	push_error("Argument `type` is invalid. Use `TYPE_*` constants.")
	return "<invalid type>"
