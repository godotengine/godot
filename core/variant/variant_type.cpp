/**************************************************************************/
/*  variant_type.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "variant_type.h"

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"

String VariantType::get_type_name(Type p_type) {
	switch (p_type) {
		case Type::NIL: {
			return "Nil";
		}

		// Atomic types.
		case Type::BOOL: {
			return "bool";
		}
		case Type::INT: {
			return "int";
		}
		case Type::FLOAT: {
			return "float";
		}
		case Type::STRING: {
			return "String";
		}

		// Math types.
		case Type::VECTOR2: {
			return "Vector2";
		}
		case Type::VECTOR2I: {
			return "Vector2i";
		}
		case Type::RECT2: {
			return "Rect2";
		}
		case Type::RECT2I: {
			return "Rect2i";
		}
		case Type::TRANSFORM2D: {
			return "Transform2D";
		}
		case Type::VECTOR3: {
			return "Vector3";
		}
		case Type::VECTOR3I: {
			return "Vector3i";
		}
		case Type::VECTOR4: {
			return "Vector4";
		}
		case Type::VECTOR4I: {
			return "Vector4i";
		}
		case Type::PLANE: {
			return "Plane";
		}
		case Type::AABB: {
			return "AABB";
		}
		case Type::QUATERNION: {
			return "Quaternion";
		}
		case Type::BASIS: {
			return "Basis";
		}
		case Type::TRANSFORM3D: {
			return "Transform3D";
		}
		case Type::PROJECTION: {
			return "Projection";
		}

		// Miscellaneous types.
		case Type::COLOR: {
			return "Color";
		}
		case Type::RID: {
			return "RID";
		}
		case Type::OBJECT: {
			return "Object";
		}
		case Type::CALLABLE: {
			return "Callable";
		}
		case Type::SIGNAL: {
			return "Signal";
		}
		case Type::STRING_NAME: {
			return "StringName";
		}
		case Type::NODE_PATH: {
			return "NodePath";
		}
		case Type::DICTIONARY: {
			return "Dictionary";
		}
		case Type::ARRAY: {
			return "Array";
		}

		// Arrays.
		case Type::PACKED_BYTE_ARRAY: {
			return "PackedByteArray";
		}
		case Type::PACKED_INT32_ARRAY: {
			return "PackedInt32Array";
		}
		case Type::PACKED_INT64_ARRAY: {
			return "PackedInt64Array";
		}
		case Type::PACKED_FLOAT32_ARRAY: {
			return "PackedFloat32Array";
		}
		case Type::PACKED_FLOAT64_ARRAY: {
			return "PackedFloat64Array";
		}
		case Type::PACKED_STRING_ARRAY: {
			return "PackedStringArray";
		}
		case Type::PACKED_VECTOR2_ARRAY: {
			return "PackedVector2Array";
		}
		case Type::PACKED_VECTOR3_ARRAY: {
			return "PackedVector3Array";
		}
		case Type::PACKED_COLOR_ARRAY: {
			return "PackedColorArray";
		}
		case Type::PACKED_VECTOR4_ARRAY: {
			return "PackedVector4Array";
		}
		default: {
		}
	}

	return "";
}

static HashMap<String, VariantType::Type> _init_type_name_map() {
	HashMap<String, VariantType::Type> type_names;
	for (int i = 0; i < VariantType::VARIANT_MAX; i++) {
		type_names[get_type_name((VariantType::Type)i)] = (VariantType::Type)i;
	}
	return type_names;
}

VariantType::Type VariantType::get_type_by_name(const String &p_type_name) {
	static HashMap<String, VariantType::Type> type_names = _init_type_name_map();

	const Type *ptr = type_names.getptr(p_type_name);
	return (ptr == nullptr) ? Type::VARIANT_MAX : *ptr;
}
bool VariantType::can_convert(Type p_type_from, Type p_type_to) {
	if (p_type_from == p_type_to) {
		return true;
	}
	if (p_type_to == Type::NIL) { //nil can convert to anything
		return true;
	}

	if (p_type_from == Type::NIL) {
		return (p_type_to == Type::OBJECT);
	}

	const VariantType::Type *valid_types = nullptr;
	const VariantType::Type *invalid_types = nullptr;

	switch (p_type_to) {
		case Type::BOOL: {
			static const Type valid[] = {
				Type::INT,
				Type::FLOAT,
				Type::STRING,
				Type::NIL,
			};

			valid_types = valid;
		} break;
		case Type::INT: {
			static const Type valid[] = {
				Type::BOOL,
				Type::FLOAT,
				Type::STRING,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::FLOAT: {
			static const Type valid[] = {
				Type::BOOL,
				Type::INT,
				Type::STRING,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::STRING: {
			static const Type invalid[] = {
				Type::OBJECT,
				Type::NIL
			};

			invalid_types = invalid;
		} break;
		case Type::VECTOR2: {
			static const Type valid[] = {
				Type::VECTOR2I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR2I: {
			static const Type valid[] = {
				Type::VECTOR2,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::RECT2: {
			static const Type valid[] = {
				Type::RECT2I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::RECT2I: {
			static const Type valid[] = {
				Type::RECT2,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::TRANSFORM2D: {
			static const Type valid[] = {
				Type::TRANSFORM3D,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::VECTOR3: {
			static const Type valid[] = {
				Type::VECTOR3I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR3I: {
			static const Type valid[] = {
				Type::VECTOR3,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR4: {
			static const Type valid[] = {
				Type::VECTOR4I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR4I: {
			static const Type valid[] = {
				Type::VECTOR4,
				Type::NIL,
			};

			valid_types = valid;

		} break;

		case Type::QUATERNION: {
			static const Type valid[] = {
				Type::BASIS,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::BASIS: {
			static const Type valid[] = {
				Type::QUATERNION,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::TRANSFORM3D: {
			static const Type valid[] = {
				Type::TRANSFORM2D,
				Type::QUATERNION,
				Type::BASIS,
				Type::PROJECTION,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::PROJECTION: {
			static const Type valid[] = {
				Type::TRANSFORM3D,
				Type::NIL
			};

			valid_types = valid;

		} break;

		case Type::COLOR: {
			static const Type valid[] = {
				Type::STRING,
				Type::INT,
				Type::NIL,
			};

			valid_types = valid;

		} break;

		case Type::RID: {
			static const Type valid[] = {
				Type::OBJECT,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::OBJECT: {
			static const Type valid[] = {
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::STRING_NAME: {
			static const Type valid[] = {
				Type::STRING,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::NODE_PATH: {
			static const Type valid[] = {
				Type::STRING,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::ARRAY: {
			static const Type valid[] = {
				Type::PACKED_BYTE_ARRAY,
				Type::PACKED_INT32_ARRAY,
				Type::PACKED_INT64_ARRAY,
				Type::PACKED_FLOAT32_ARRAY,
				Type::PACKED_FLOAT64_ARRAY,
				Type::PACKED_STRING_ARRAY,
				Type::PACKED_COLOR_ARRAY,
				Type::PACKED_VECTOR2_ARRAY,
				Type::PACKED_VECTOR3_ARRAY,
				Type::PACKED_VECTOR4_ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case Type::PACKED_BYTE_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::PACKED_INT32_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;
		} break;
		case Type::PACKED_INT64_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;
		} break;
		case Type::PACKED_FLOAT32_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::PACKED_FLOAT64_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::PACKED_STRING_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;
		} break;
		case Type::PACKED_VECTOR2_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;

		} break;
		case Type::PACKED_VECTOR3_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;

		} break;
		case Type::PACKED_COLOR_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::PACKED_VECTOR4_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;

		} break;
		default: {
		}
	}

	if (valid_types) {
		int i = 0;
		while (valid_types[i] != Type::NIL) {
			if (p_type_from == valid_types[i]) {
				return true;
			}
			i++;
		}

	} else if (invalid_types) {
		int i = 0;
		while (invalid_types[i] != Type::NIL) {
			if (p_type_from == invalid_types[i]) {
				return false;
			}
			i++;
		}

		return true;
	}

	return false;
}

bool VariantType::can_convert_strict(Type p_type_from, Type p_type_to) {
	if (p_type_from == p_type_to) {
		return true;
	}
	if (p_type_to == Type::NIL) { //nil can convert to anything
		return true;
	}

	if (p_type_from == Type::NIL) {
		return (p_type_to == Type::OBJECT);
	}

	const Type *valid_types = nullptr;

	switch (p_type_to) {
		case Type::BOOL: {
			static const Type valid[] = {
				Type::INT,
				Type::FLOAT,
				//Type::STRING,
				Type::NIL,
			};

			valid_types = valid;
		} break;
		case Type::INT: {
			static const Type valid[] = {
				Type::BOOL,
				Type::FLOAT,
				//Type::STRING,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::FLOAT: {
			static const Type valid[] = {
				Type::BOOL,
				Type::INT,
				//Type::STRING,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::STRING: {
			static const Type valid[] = {
				Type::NODE_PATH,
				Type::STRING_NAME,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::VECTOR2: {
			static const Type valid[] = {
				Type::VECTOR2I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR2I: {
			static const Type valid[] = {
				Type::VECTOR2,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::RECT2: {
			static const Type valid[] = {
				Type::RECT2I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::RECT2I: {
			static const Type valid[] = {
				Type::RECT2,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::TRANSFORM2D: {
			static const Type valid[] = {
				Type::TRANSFORM3D,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::VECTOR3: {
			static const Type valid[] = {
				Type::VECTOR3I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR3I: {
			static const Type valid[] = {
				Type::VECTOR3,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR4: {
			static const Type valid[] = {
				Type::VECTOR4I,
				Type::NIL,
			};

			valid_types = valid;

		} break;
		case Type::VECTOR4I: {
			static const Type valid[] = {
				Type::VECTOR4,
				Type::NIL,
			};

			valid_types = valid;

		} break;

		case Type::QUATERNION: {
			static const Type valid[] = {
				Type::BASIS,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::BASIS: {
			static const Type valid[] = {
				Type::QUATERNION,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::TRANSFORM3D: {
			static const Type valid[] = {
				Type::TRANSFORM2D,
				Type::QUATERNION,
				Type::BASIS,
				Type::PROJECTION,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::PROJECTION: {
			static const Type valid[] = {
				Type::TRANSFORM3D,
				Type::NIL
			};

			valid_types = valid;

		} break;

		case Type::COLOR: {
			static const Type valid[] = {
				Type::STRING,
				Type::INT,
				Type::NIL,
			};

			valid_types = valid;

		} break;

		case Type::RID: {
			static const Type valid[] = {
				Type::OBJECT,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::OBJECT: {
			static const Type valid[] = {
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::STRING_NAME: {
			static const Type valid[] = {
				Type::STRING,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::NODE_PATH: {
			static const Type valid[] = {
				Type::STRING,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::ARRAY: {
			static const Type valid[] = {
				Type::PACKED_BYTE_ARRAY,
				Type::PACKED_INT32_ARRAY,
				Type::PACKED_INT64_ARRAY,
				Type::PACKED_FLOAT32_ARRAY,
				Type::PACKED_FLOAT64_ARRAY,
				Type::PACKED_STRING_ARRAY,
				Type::PACKED_COLOR_ARRAY,
				Type::PACKED_VECTOR2_ARRAY,
				Type::PACKED_VECTOR3_ARRAY,
				Type::PACKED_VECTOR4_ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case Type::PACKED_BYTE_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::PACKED_INT32_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;
		} break;
		case Type::PACKED_INT64_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;
		} break;
		case Type::PACKED_FLOAT32_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::PACKED_FLOAT64_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;
		} break;
		case Type::PACKED_STRING_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;
		} break;
		case Type::PACKED_VECTOR2_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;

		} break;
		case Type::PACKED_VECTOR3_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;

		} break;
		case Type::PACKED_COLOR_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};

			valid_types = valid;

		} break;
		case Type::PACKED_VECTOR4_ARRAY: {
			static const Type valid[] = {
				Type::ARRAY,
				Type::NIL
			};
			valid_types = valid;

		} break;
		default: {
		}
	}

	if (valid_types) {
		int i = 0;
		while (valid_types[i] != Type::NIL) {
			if (p_type_from == valid_types[i]) {
				return true;
			}
			i++;
		}
	}

	return false;
}

bool VariantType::is_type_shared(Type p_type) {
	switch (p_type) {
		case OBJECT:
		case DICTIONARY:
		case ARRAY:
		// NOTE: Packed array constructors **do** copies (unlike `Array()` and `Dictionary()`),
		// whereas they pass by reference when inside a `Variant`.
		case PACKED_BYTE_ARRAY:
		case PACKED_INT32_ARRAY:
		case PACKED_INT64_ARRAY:
		case PACKED_FLOAT32_ARRAY:
		case PACKED_FLOAT64_ARRAY:
		case PACKED_STRING_ARRAY:
		case PACKED_VECTOR2_ARRAY:
		case PACKED_VECTOR3_ARRAY:
		case PACKED_COLOR_ARRAY:
		case PACKED_VECTOR4_ARRAY:
			return true;
		default:
			return false;
	}
}
