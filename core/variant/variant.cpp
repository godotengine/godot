/*************************************************************************/
/*  variant.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "variant.h"

#include "core/core_string_names.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "core/io/resource.h"
#include "core/math/math_funcs.h"
#include "core/string/print_string.h"
#include "core/variant/variant_parser.h"

String Variant::get_type_name(Variant::Type p_type) {
	switch (p_type) {
		case NIL: {
			return "Nil";
		} break;

		// atomic types
		case BOOL: {
			return "bool";
		} break;
		case INT: {
			return "int";

		} break;
		case FLOAT: {
			return "float";

		} break;
		case STRING: {
			return "String";
		} break;

		// math types
		case VECTOR2: {
			return "Vector2";
		} break;
		case VECTOR2I: {
			return "Vector2i";
		} break;
		case RECT2: {
			return "Rect2";
		} break;
		case RECT2I: {
			return "Rect2i";
		} break;
		case TRANSFORM2D: {
			return "Transform2D";
		} break;
		case VECTOR3: {
			return "Vector3";
		} break;
		case VECTOR3I: {
			return "Vector3i";
		} break;
		case PLANE: {
			return "Plane";

		} break;
		case AABB: {
			return "AABB";
		} break;
		case QUATERNION: {
			return "Quaternion";

		} break;
		case BASIS: {
			return "Basis";

		} break;
		case TRANSFORM3D: {
			return "Transform3D";

		} break;

		// misc types
		case COLOR: {
			return "Color";

		} break;
		case RID: {
			return "RID";
		} break;
		case OBJECT: {
			return "Object";
		} break;
		case CALLABLE: {
			return "Callable";
		} break;
		case SIGNAL: {
			return "Signal";
		} break;
		case STRING_NAME: {
			return "StringName";

		} break;
		case NODE_PATH: {
			return "NodePath";

		} break;
		case DICTIONARY: {
			return "Dictionary";

		} break;
		case ARRAY: {
			return "Array";

		} break;

		// arrays
		case PACKED_BYTE_ARRAY: {
			return "PackedByteArray";

		} break;
		case PACKED_INT32_ARRAY: {
			return "PackedInt32Array";

		} break;
		case PACKED_INT64_ARRAY: {
			return "PackedInt64Array";

		} break;
		case PACKED_FLOAT32_ARRAY: {
			return "PackedFloat32Array";

		} break;
		case PACKED_FLOAT64_ARRAY: {
			return "PackedFloat64Array";

		} break;
		case PACKED_STRING_ARRAY: {
			return "PackedStringArray";
		} break;
		case PACKED_VECTOR2_ARRAY: {
			return "PackedVector2Array";

		} break;
		case PACKED_VECTOR3_ARRAY: {
			return "PackedVector3Array";

		} break;
		case PACKED_COLOR_ARRAY: {
			return "PackedColorArray";

		} break;
		default: {
		}
	}

	return "";
}

bool Variant::can_convert(Variant::Type p_type_from, Variant::Type p_type_to) {
	if (p_type_from == p_type_to) {
		return true;
	}
	if (p_type_to == NIL && p_type_from != NIL) { //nil can convert to anything
		return true;
	}

	if (p_type_from == NIL) {
		return (p_type_to == OBJECT);
	}

	const Type *valid_types = nullptr;
	const Type *invalid_types = nullptr;

	switch (p_type_to) {
		case BOOL: {
			static const Type valid[] = {
				INT,
				FLOAT,
				STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case INT: {
			static const Type valid[] = {
				BOOL,
				FLOAT,
				STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case FLOAT: {
			static const Type valid[] = {
				BOOL,
				INT,
				STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case STRING: {
			static const Type invalid[] = {
				OBJECT,
				NIL
			};

			invalid_types = invalid;
		} break;
		case VECTOR2: {
			static const Type valid[] = {
				VECTOR2I,
				NIL,
			};

			valid_types = valid;

		} break;
		case VECTOR2I: {
			static const Type valid[] = {
				VECTOR2,
				NIL,
			};

			valid_types = valid;

		} break;
		case RECT2: {
			static const Type valid[] = {
				RECT2I,
				NIL,
			};

			valid_types = valid;

		} break;
		case RECT2I: {
			static const Type valid[] = {
				RECT2,
				NIL,
			};

			valid_types = valid;

		} break;
		case TRANSFORM2D: {
			static const Type valid[] = {
				TRANSFORM3D,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR3: {
			static const Type valid[] = {
				VECTOR3I,
				NIL,
			};

			valid_types = valid;

		} break;
		case VECTOR3I: {
			static const Type valid[] = {
				VECTOR3,
				NIL,
			};

			valid_types = valid;

		} break;

		case QUATERNION: {
			static const Type valid[] = {
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;
		case BASIS: {
			static const Type valid[] = {
				QUATERNION,
				NIL
			};

			valid_types = valid;

		} break;
		case TRANSFORM3D: {
			static const Type valid[] = {
				TRANSFORM2D,
				QUATERNION,
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;

		case COLOR: {
			static const Type valid[] = {
				STRING,
				INT,
				NIL,
			};

			valid_types = valid;

		} break;

		case RID: {
			static const Type valid[] = {
				OBJECT,
				NIL
			};

			valid_types = valid;
		} break;
		case OBJECT: {
			static const Type valid[] = {
				NIL
			};

			valid_types = valid;
		} break;
		case STRING_NAME: {
			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case NODE_PATH: {
			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case ARRAY: {
			static const Type valid[] = {
				PACKED_BYTE_ARRAY,
				PACKED_INT32_ARRAY,
				PACKED_INT64_ARRAY,
				PACKED_FLOAT32_ARRAY,
				PACKED_FLOAT64_ARRAY,
				PACKED_STRING_ARRAY,
				PACKED_COLOR_ARRAY,
				PACKED_VECTOR2_ARRAY,
				PACKED_VECTOR3_ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case PACKED_BYTE_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case PACKED_INT32_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case PACKED_INT64_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case PACKED_FLOAT32_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case PACKED_FLOAT64_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case PACKED_STRING_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case PACKED_VECTOR2_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case PACKED_VECTOR3_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case PACKED_COLOR_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;

		} break;
		default: {
		}
	}

	if (valid_types) {
		int i = 0;
		while (valid_types[i] != NIL) {
			if (p_type_from == valid_types[i]) {
				return true;
			}
			i++;
		}

	} else if (invalid_types) {
		int i = 0;
		while (invalid_types[i] != NIL) {
			if (p_type_from == invalid_types[i]) {
				return false;
			}
			i++;
		}

		return true;
	}

	return false;
}

bool Variant::can_convert_strict(Variant::Type p_type_from, Variant::Type p_type_to) {
	if (p_type_from == p_type_to) {
		return true;
	}
	if (p_type_to == NIL && p_type_from != NIL) { //nil can convert to anything
		return true;
	}

	if (p_type_from == NIL) {
		return (p_type_to == OBJECT);
	}

	const Type *valid_types = nullptr;

	switch (p_type_to) {
		case BOOL: {
			static const Type valid[] = {
				INT,
				FLOAT,
				//STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case INT: {
			static const Type valid[] = {
				BOOL,
				FLOAT,
				//STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case FLOAT: {
			static const Type valid[] = {
				BOOL,
				INT,
				//STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case STRING: {
			static const Type valid[] = {
				NODE_PATH,
				STRING_NAME,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR2: {
			static const Type valid[] = {
				VECTOR2I,
				NIL,
			};

			valid_types = valid;

		} break;
		case VECTOR2I: {
			static const Type valid[] = {
				VECTOR2,
				NIL,
			};

			valid_types = valid;

		} break;
		case RECT2: {
			static const Type valid[] = {
				RECT2I,
				NIL,
			};

			valid_types = valid;

		} break;
		case RECT2I: {
			static const Type valid[] = {
				RECT2,
				NIL,
			};

			valid_types = valid;

		} break;
		case TRANSFORM2D: {
			static const Type valid[] = {
				TRANSFORM3D,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR3: {
			static const Type valid[] = {
				VECTOR3I,
				NIL,
			};

			valid_types = valid;

		} break;
		case VECTOR3I: {
			static const Type valid[] = {
				VECTOR3,
				NIL,
			};

			valid_types = valid;

		} break;

		case QUATERNION: {
			static const Type valid[] = {
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;
		case BASIS: {
			static const Type valid[] = {
				QUATERNION,
				NIL
			};

			valid_types = valid;

		} break;
		case TRANSFORM3D: {
			static const Type valid[] = {
				TRANSFORM2D,
				QUATERNION,
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;

		case COLOR: {
			static const Type valid[] = {
				STRING,
				INT,
				NIL,
			};

			valid_types = valid;

		} break;

		case RID: {
			static const Type valid[] = {
				OBJECT,
				NIL
			};

			valid_types = valid;
		} break;
		case OBJECT: {
			static const Type valid[] = {
				NIL
			};

			valid_types = valid;
		} break;
		case STRING_NAME: {
			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case NODE_PATH: {
			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case ARRAY: {
			static const Type valid[] = {
				PACKED_BYTE_ARRAY,
				PACKED_INT32_ARRAY,
				PACKED_INT64_ARRAY,
				PACKED_FLOAT32_ARRAY,
				PACKED_FLOAT64_ARRAY,
				PACKED_STRING_ARRAY,
				PACKED_COLOR_ARRAY,
				PACKED_VECTOR2_ARRAY,
				PACKED_VECTOR3_ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case PACKED_BYTE_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case PACKED_INT32_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case PACKED_INT64_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case PACKED_FLOAT32_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case PACKED_FLOAT64_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case PACKED_STRING_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case PACKED_VECTOR2_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case PACKED_VECTOR3_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case PACKED_COLOR_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;

		} break;
		default: {
		}
	}

	if (valid_types) {
		int i = 0;
		while (valid_types[i] != NIL) {
			if (p_type_from == valid_types[i]) {
				return true;
			}
			i++;
		}
	}

	return false;
}

bool Variant::operator==(const Variant &p_variant) const {
	return hash_compare(p_variant);
}

bool Variant::operator!=(const Variant &p_variant) const {
	// Don't use `!hash_compare(p_variant)` given it makes use of OP_EQUAL
	if (type != p_variant.type) { //evaluation of operator== needs to be more strict
		return true;
	}
	bool v;
	Variant r;
	evaluate(OP_NOT_EQUAL, *this, p_variant, r, v);
	return r;
}

bool Variant::operator<(const Variant &p_variant) const {
	if (type != p_variant.type) { //if types differ, then order by type first
		return type < p_variant.type;
	}
	bool v;
	Variant r;
	evaluate(OP_LESS, *this, p_variant, r, v);
	return r;
}

bool Variant::is_zero() const {
	switch (type) {
		case NIL: {
			return true;
		} break;

		// atomic types
		case BOOL: {
			return !(_data._bool);
		} break;
		case INT: {
			return _data._int == 0;

		} break;
		case FLOAT: {
			return _data._float == 0;

		} break;
		case STRING: {
			return *reinterpret_cast<const String *>(_data._mem) == String();

		} break;

		// math types
		case VECTOR2: {
			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2();

		} break;
		case VECTOR2I: {
			return *reinterpret_cast<const Vector2i *>(_data._mem) == Vector2i();

		} break;
		case RECT2: {
			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2();

		} break;
		case RECT2I: {
			return *reinterpret_cast<const Rect2i *>(_data._mem) == Rect2i();

		} break;
		case TRANSFORM2D: {
			return *_data._transform2d == Transform2D();

		} break;
		case VECTOR3: {
			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3();

		} break;
		case VECTOR3I: {
			return *reinterpret_cast<const Vector3i *>(_data._mem) == Vector3i();

		} break;
		case PLANE: {
			return *reinterpret_cast<const Plane *>(_data._mem) == Plane();

		} break;
		case AABB: {
			return *_data._aabb == ::AABB();
		} break;
		case QUATERNION: {
			return *reinterpret_cast<const Quaternion *>(_data._mem) == Quaternion();

		} break;
		case BASIS: {
			return *_data._basis == Basis();

		} break;
		case TRANSFORM3D: {
			return *_data._transform3d == Transform3D();

		} break;

		// misc types
		case COLOR: {
			return *reinterpret_cast<const Color *>(_data._mem) == Color();

		} break;
		case RID: {
			return *reinterpret_cast<const ::RID *>(_data._mem) == ::RID();
		} break;
		case OBJECT: {
			return _get_obj().obj == nullptr;
		} break;
		case CALLABLE: {
			return reinterpret_cast<const Callable *>(_data._mem)->is_null();
		} break;
		case SIGNAL: {
			return reinterpret_cast<const Signal *>(_data._mem)->is_null();
		} break;
		case STRING_NAME: {
			return *reinterpret_cast<const StringName *>(_data._mem) != StringName();

		} break;
		case NODE_PATH: {
			return reinterpret_cast<const NodePath *>(_data._mem)->is_empty();

		} break;
		case DICTIONARY: {
			return reinterpret_cast<const Dictionary *>(_data._mem)->is_empty();

		} break;
		case ARRAY: {
			return reinterpret_cast<const Array *>(_data._mem)->is_empty();

		} break;

		// arrays
		case PACKED_BYTE_ARRAY: {
			return PackedArrayRef<uint8_t>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_INT32_ARRAY: {
			return PackedArrayRef<int32_t>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_INT64_ARRAY: {
			return PackedArrayRef<int64_t>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_FLOAT32_ARRAY: {
			return PackedArrayRef<float>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_FLOAT64_ARRAY: {
			return PackedArrayRef<double>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_STRING_ARRAY: {
			return PackedArrayRef<String>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_VECTOR2_ARRAY: {
			return PackedArrayRef<Vector2>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_VECTOR3_ARRAY: {
			return PackedArrayRef<Vector3>::get_array(_data.packed_array).size() == 0;

		} break;
		case PACKED_COLOR_ARRAY: {
			return PackedArrayRef<Color>::get_array(_data.packed_array).size() == 0;

		} break;
		default: {
		}
	}

	return false;
}

bool Variant::is_one() const {
	switch (type) {
		case NIL: {
			return true;
		} break;

		// atomic types
		case BOOL: {
			return _data._bool;
		} break;
		case INT: {
			return _data._int == 1;

		} break;
		case FLOAT: {
			return _data._float == 1;

		} break;
		case VECTOR2: {
			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2(1, 1);

		} break;
		case VECTOR2I: {
			return *reinterpret_cast<const Vector2i *>(_data._mem) == Vector2i(1, 1);

		} break;
		case RECT2: {
			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2(1, 1, 1, 1);

		} break;
		case RECT2I: {
			return *reinterpret_cast<const Rect2i *>(_data._mem) == Rect2i(1, 1, 1, 1);

		} break;
		case VECTOR3: {
			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3(1, 1, 1);

		} break;
		case VECTOR3I: {
			return *reinterpret_cast<const Vector3i *>(_data._mem) == Vector3i(1, 1, 1);

		} break;
		case PLANE: {
			return *reinterpret_cast<const Plane *>(_data._mem) == Plane(1, 1, 1, 1);

		} break;
		case COLOR: {
			return *reinterpret_cast<const Color *>(_data._mem) == Color(1, 1, 1, 1);

		} break;

		default: {
			return !is_zero();
		}
	}

	return false;
}

bool Variant::is_null() const {
	if (type == OBJECT && _get_obj().obj) {
		return false;
	} else {
		return true;
	}
}

void Variant::reference(const Variant &p_variant) {
	switch (type) {
		case NIL:
		case BOOL:
		case INT:
		case FLOAT:
			break;
		default:
			clear();
	}

	type = p_variant.type;

	switch (p_variant.type) {
		case NIL: {
			// none
		} break;

		// atomic types
		case BOOL: {
			_data._bool = p_variant._data._bool;
		} break;
		case INT: {
			_data._int = p_variant._data._int;
		} break;
		case FLOAT: {
			_data._float = p_variant._data._float;
		} break;
		case STRING: {
			memnew_placement(_data._mem, String(*reinterpret_cast<const String *>(p_variant._data._mem)));
		} break;

		// math types
		case VECTOR2: {
			memnew_placement(_data._mem, Vector2(*reinterpret_cast<const Vector2 *>(p_variant._data._mem)));
		} break;
		case VECTOR2I: {
			memnew_placement(_data._mem, Vector2i(*reinterpret_cast<const Vector2i *>(p_variant._data._mem)));
		} break;
		case RECT2: {
			memnew_placement(_data._mem, Rect2(*reinterpret_cast<const Rect2 *>(p_variant._data._mem)));
		} break;
		case RECT2I: {
			memnew_placement(_data._mem, Rect2i(*reinterpret_cast<const Rect2i *>(p_variant._data._mem)));
		} break;
		case TRANSFORM2D: {
			_data._transform2d = memnew(Transform2D(*p_variant._data._transform2d));
		} break;
		case VECTOR3: {
			memnew_placement(_data._mem, Vector3(*reinterpret_cast<const Vector3 *>(p_variant._data._mem)));
		} break;
		case VECTOR3I: {
			memnew_placement(_data._mem, Vector3i(*reinterpret_cast<const Vector3i *>(p_variant._data._mem)));
		} break;
		case PLANE: {
			memnew_placement(_data._mem, Plane(*reinterpret_cast<const Plane *>(p_variant._data._mem)));
		} break;

		case AABB: {
			_data._aabb = memnew(::AABB(*p_variant._data._aabb));
		} break;
		case QUATERNION: {
			memnew_placement(_data._mem, Quaternion(*reinterpret_cast<const Quaternion *>(p_variant._data._mem)));

		} break;
		case BASIS: {
			_data._basis = memnew(Basis(*p_variant._data._basis));

		} break;
		case TRANSFORM3D: {
			_data._transform3d = memnew(Transform3D(*p_variant._data._transform3d));
		} break;

		// misc types
		case COLOR: {
			memnew_placement(_data._mem, Color(*reinterpret_cast<const Color *>(p_variant._data._mem)));

		} break;
		case RID: {
			memnew_placement(_data._mem, ::RID(*reinterpret_cast<const ::RID *>(p_variant._data._mem)));
		} break;
		case OBJECT: {
			memnew_placement(_data._mem, ObjData);

			if (p_variant._get_obj().obj && p_variant._get_obj().id.is_ref_counted()) {
				RefCounted *ref_counted = static_cast<RefCounted *>(p_variant._get_obj().obj);
				if (!ref_counted->reference()) {
					_get_obj().obj = nullptr;
					_get_obj().id = ObjectID();
					break;
				}
			}

			_get_obj().obj = const_cast<Object *>(p_variant._get_obj().obj);
			_get_obj().id = p_variant._get_obj().id;

		} break;
		case CALLABLE: {
			memnew_placement(_data._mem, Callable(*reinterpret_cast<const Callable *>(p_variant._data._mem)));
		} break;
		case SIGNAL: {
			memnew_placement(_data._mem, Signal(*reinterpret_cast<const Signal *>(p_variant._data._mem)));
		} break;
		case STRING_NAME: {
			memnew_placement(_data._mem, StringName(*reinterpret_cast<const StringName *>(p_variant._data._mem)));

		} break;
		case NODE_PATH: {
			memnew_placement(_data._mem, NodePath(*reinterpret_cast<const NodePath *>(p_variant._data._mem)));

		} break;
		case DICTIONARY: {
			memnew_placement(_data._mem, Dictionary(*reinterpret_cast<const Dictionary *>(p_variant._data._mem)));

		} break;
		case ARRAY: {
			memnew_placement(_data._mem, Array(*reinterpret_cast<const Array *>(p_variant._data._mem)));

		} break;

		// arrays
		case PACKED_BYTE_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<uint8_t> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<uint8_t>::create();
			}

		} break;
		case PACKED_INT32_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<int32_t> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<int32_t>::create();
			}

		} break;
		case PACKED_INT64_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<int64_t> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<int64_t>::create();
			}

		} break;
		case PACKED_FLOAT32_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<float> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<float>::create();
			}

		} break;
		case PACKED_FLOAT64_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<double> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<double>::create();
			}

		} break;
		case PACKED_STRING_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<String> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<String>::create();
			}

		} break;
		case PACKED_VECTOR2_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<Vector2> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<Vector2>::create();
			}

		} break;
		case PACKED_VECTOR3_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<Vector3> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<Vector3>::create();
			}

		} break;
		case PACKED_COLOR_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<Color> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<Color>::create();
			}

		} break;
		default: {
		}
	}
}

void Variant::zero() {
	switch (type) {
		case NIL:
			break;
		case BOOL:
			this->_data._bool = false;
			break;
		case INT:
			this->_data._int = 0;
			break;
		case FLOAT:
			this->_data._float = 0;
			break;
		case VECTOR2:
			*reinterpret_cast<Vector2 *>(this->_data._mem) = Vector2();
			break;
		case VECTOR2I:
			*reinterpret_cast<Vector2i *>(this->_data._mem) = Vector2i();
			break;
		case RECT2:
			*reinterpret_cast<Rect2 *>(this->_data._mem) = Rect2();
			break;
		case RECT2I:
			*reinterpret_cast<Rect2i *>(this->_data._mem) = Rect2i();
			break;
		case VECTOR3:
			*reinterpret_cast<Vector3 *>(this->_data._mem) = Vector3();
			break;
		case VECTOR3I:
			*reinterpret_cast<Vector3i *>(this->_data._mem) = Vector3i();
			break;
		case PLANE:
			*reinterpret_cast<Plane *>(this->_data._mem) = Plane();
			break;
		case QUATERNION:
			*reinterpret_cast<Quaternion *>(this->_data._mem) = Quaternion();
			break;
		case COLOR:
			*reinterpret_cast<Color *>(this->_data._mem) = Color();
			break;
		default:
			this->clear();
			break;
	}
}

void Variant::_clear_internal() {
	switch (type) {
		case STRING: {
			reinterpret_cast<String *>(_data._mem)->~String();
		} break;
		/*
		// no point, they don't allocate memory
		VECTOR3,
		PLANE,
		QUATERNION,
		COLOR,
		VECTOR2,
		RECT2
		*/
		case TRANSFORM2D: {
			memdelete(_data._transform2d);
		} break;
		case AABB: {
			memdelete(_data._aabb);
		} break;
		case BASIS: {
			memdelete(_data._basis);
		} break;
		case TRANSFORM3D: {
			memdelete(_data._transform3d);
		} break;

			// misc types
		case STRING_NAME: {
			reinterpret_cast<StringName *>(_data._mem)->~StringName();
		} break;
		case NODE_PATH: {
			reinterpret_cast<NodePath *>(_data._mem)->~NodePath();
		} break;
		case OBJECT: {
			if (_get_obj().id.is_ref_counted()) {
				//we are safe that there is a reference here
				RefCounted *ref_counted = static_cast<RefCounted *>(_get_obj().obj);
				if (ref_counted->unreference()) {
					memdelete(ref_counted);
				}
			}
			_get_obj().obj = nullptr;
			_get_obj().id = ObjectID();
		} break;
		case RID: {
			// not much need probably
			// Can't seem to use destructor + scoping operator, so hack.
			typedef ::RID RID_Class;
			reinterpret_cast<RID_Class *>(_data._mem)->~RID_Class();
		} break;
		case CALLABLE: {
			reinterpret_cast<Callable *>(_data._mem)->~Callable();
		} break;
		case SIGNAL: {
			reinterpret_cast<Signal *>(_data._mem)->~Signal();
		} break;
		case DICTIONARY: {
			reinterpret_cast<Dictionary *>(_data._mem)->~Dictionary();
		} break;
		case ARRAY: {
			reinterpret_cast<Array *>(_data._mem)->~Array();
		} break;
		// arrays
		case PACKED_BYTE_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_INT32_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_INT64_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_FLOAT32_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_FLOAT64_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_STRING_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_VECTOR2_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_VECTOR3_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case PACKED_COLOR_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		default: {
		} /* not needed */
	}
}

Variant::operator signed int() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator unsigned int() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator int64_t() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator uint64_t() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator ObjectID() const {
	if (type == INT) {
		return ObjectID(_data._int);
	} else if (type == OBJECT) {
		return _get_obj().id;
	} else {
		return ObjectID();
	}
}

#ifdef NEED_LONG_INT
Variant::operator signed long() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}

	return 0;
}

Variant::operator unsigned long() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}

	return 0;
}
#endif

Variant::operator signed short() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator unsigned short() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator signed char() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator unsigned char() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator char32_t() const {
	return operator unsigned int();
}

Variant::operator float() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1.0 : 0.0;
		case INT:
			return (float)_data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_float();
		default: {
			return 0;
		}
	}
}

Variant::operator double() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1.0 : 0.0;
		case INT:
			return (double)_data._int;
		case FLOAT:
			return _data._float;
		case STRING:
			return operator String().to_float();
		default: {
			return 0;
		}
	}
}

Variant::operator StringName() const {
	if (type == STRING_NAME) {
		return *reinterpret_cast<const StringName *>(_data._mem);
	} else if (type == STRING) {
		return *reinterpret_cast<const String *>(_data._mem);
	}

	return StringName();
}

struct _VariantStrPair {
	String key;
	String value;

	bool operator<(const _VariantStrPair &p) const {
		return key < p.key;
	}
};

Variant::operator String() const {
	return stringify(0);
}

template <class T>
String stringify_vector(const T &vec, int recursion_count) {
	String str("[");
	for (int i = 0; i < vec.size(); i++) {
		if (i > 0) {
			str += ", ";
		}
		str = str + Variant(vec[i]).stringify(recursion_count);
	}
	str += "]";
	return str;
}

String Variant::stringify(int recursion_count) const {
	switch (type) {
		case NIL:
			return "null";
		case BOOL:
			return _data._bool ? "true" : "false";
		case INT:
			return itos(_data._int);
		case FLOAT:
			return rtos(_data._float);
		case STRING:
			return *reinterpret_cast<const String *>(_data._mem);
		case VECTOR2:
			return operator Vector2();
		case VECTOR2I:
			return operator Vector2i();
		case RECT2:
			return operator Rect2();
		case RECT2I:
			return operator Rect2i();
		case TRANSFORM2D:
			return operator Transform2D();
		case VECTOR3:
			return operator Vector3();
		case VECTOR3I:
			return operator Vector3i();
		case PLANE:
			return operator Plane();
		case AABB:
			return operator ::AABB();
		case QUATERNION:
			return operator Quaternion();
		case BASIS:
			return operator Basis();
		case TRANSFORM3D:
			return operator Transform3D();
		case STRING_NAME:
			return operator StringName();
		case NODE_PATH:
			return operator NodePath();
		case COLOR:
			return operator Color();
		case DICTIONARY: {
			const Dictionary &d = *reinterpret_cast<const Dictionary *>(_data._mem);
			if (recursion_count > MAX_RECURSION) {
				ERR_PRINT("Max recursion reached");
				return "{...}";
			}

			String str("{");
			List<Variant> keys;
			d.get_key_list(&keys);

			Vector<_VariantStrPair> pairs;

			recursion_count++;
			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
				_VariantStrPair sp;
				sp.key = E->get().stringify(recursion_count);
				sp.value = d[E->get()].stringify(recursion_count);

				pairs.push_back(sp);
			}

			for (int i = 0; i < pairs.size(); i++) {
				if (i > 0) {
					str += ", ";
				}
				str += pairs[i].key + ":" + pairs[i].value;
			}
			str += "}";

			return str;
		} break;
		case PACKED_VECTOR2_ARRAY: {
			return stringify_vector(operator Vector<Vector2>(), recursion_count);
		} break;
		case PACKED_VECTOR3_ARRAY: {
			return stringify_vector(operator Vector<Vector3>(), recursion_count);
		} break;
		case PACKED_COLOR_ARRAY: {
			return stringify_vector(operator Vector<Color>(), recursion_count);
		} break;
		case PACKED_STRING_ARRAY: {
			return stringify_vector(operator Vector<String>(), recursion_count);
		} break;
		case PACKED_BYTE_ARRAY: {
			return stringify_vector(operator Vector<uint8_t>(), recursion_count);
		} break;
		case PACKED_INT32_ARRAY: {
			return stringify_vector(operator Vector<int32_t>(), recursion_count);
		} break;
		case PACKED_INT64_ARRAY: {
			return stringify_vector(operator Vector<int64_t>(), recursion_count);
		} break;
		case PACKED_FLOAT32_ARRAY: {
			return stringify_vector(operator Vector<float>(), recursion_count);
		} break;
		case PACKED_FLOAT64_ARRAY: {
			return stringify_vector(operator Vector<double>(), recursion_count);
		} break;
		case ARRAY: {
			Array arr = operator Array();
			if (recursion_count > MAX_RECURSION) {
				ERR_PRINT("Max recursion reached");
				return "[...]";
			}

			String str = stringify_vector(arr, recursion_count);
			return str;

		} break;
		case OBJECT: {
			if (_get_obj().obj) {
				if (!_get_obj().id.is_ref_counted() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
					return "[Freed Object]";
				}

				return _get_obj().obj->to_string();
			} else {
				return "[Object:null]";
			}

		} break;
		case CALLABLE: {
			const Callable &c = *reinterpret_cast<const Callable *>(_data._mem);
			return c;
		} break;
		case SIGNAL: {
			const Signal &s = *reinterpret_cast<const Signal *>(_data._mem);
			return s;
		} break;
		case RID: {
			const ::RID &s = *reinterpret_cast<const ::RID *>(_data._mem);
			return "RID(" + itos(s.get_id()) + ")";
		} break;
		default: {
			return "[" + get_type_name(type) + "]";
		}
	}

	return "";
}

String Variant::to_json_string() const {
	JSON json;
	return json.stringify(*this);
}

Variant::operator Vector2() const {
	if (type == VECTOR2) {
		return *reinterpret_cast<const Vector2 *>(_data._mem);
	} else if (type == VECTOR2I) {
		return *reinterpret_cast<const Vector2i *>(_data._mem);
	} else if (type == VECTOR3) {
		return Vector2(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y);
	} else if (type == VECTOR3I) {
		return Vector2(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y);
	} else {
		return Vector2();
	}
}

Variant::operator Vector2i() const {
	if (type == VECTOR2I) {
		return *reinterpret_cast<const Vector2i *>(_data._mem);
	} else if (type == VECTOR2) {
		return *reinterpret_cast<const Vector2 *>(_data._mem);
	} else if (type == VECTOR3) {
		return Vector2(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y);
	} else if (type == VECTOR3I) {
		return Vector2(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y);
	} else {
		return Vector2i();
	}
}

Variant::operator Rect2() const {
	if (type == RECT2) {
		return *reinterpret_cast<const Rect2 *>(_data._mem);
	} else if (type == RECT2I) {
		return *reinterpret_cast<const Rect2i *>(_data._mem);
	} else {
		return Rect2();
	}
}

Variant::operator Rect2i() const {
	if (type == RECT2I) {
		return *reinterpret_cast<const Rect2i *>(_data._mem);
	} else if (type == RECT2) {
		return *reinterpret_cast<const Rect2 *>(_data._mem);
	} else {
		return Rect2i();
	}
}

Variant::operator Vector3() const {
	if (type == VECTOR3) {
		return *reinterpret_cast<const Vector3 *>(_data._mem);
	} else if (type == VECTOR3I) {
		return *reinterpret_cast<const Vector3i *>(_data._mem);
	} else if (type == VECTOR2) {
		return Vector3(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0);
	} else if (type == VECTOR2I) {
		return Vector3(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0);
	} else {
		return Vector3();
	}
}

Variant::operator Vector3i() const {
	if (type == VECTOR3I) {
		return *reinterpret_cast<const Vector3i *>(_data._mem);
	} else if (type == VECTOR3) {
		return *reinterpret_cast<const Vector3 *>(_data._mem);
	} else if (type == VECTOR2) {
		return Vector3i(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0);
	} else if (type == VECTOR2I) {
		return Vector3i(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0);
	} else {
		return Vector3i();
	}
}

Variant::operator Plane() const {
	if (type == PLANE) {
		return *reinterpret_cast<const Plane *>(_data._mem);
	} else {
		return Plane();
	}
}

Variant::operator ::AABB() const {
	if (type == AABB) {
		return *_data._aabb;
	} else {
		return ::AABB();
	}
}

Variant::operator Basis() const {
	if (type == BASIS) {
		return *_data._basis;
	} else if (type == QUATERNION) {
		return *reinterpret_cast<const Quaternion *>(_data._mem);
	} else if (type == TRANSFORM3D) { // unexposed in Variant::can_convert?
		return _data._transform3d->basis;
	} else {
		return Basis();
	}
}

Variant::operator Quaternion() const {
	if (type == QUATERNION) {
		return *reinterpret_cast<const Quaternion *>(_data._mem);
	} else if (type == BASIS) {
		return *_data._basis;
	} else if (type == TRANSFORM3D) {
		return _data._transform3d->basis;
	} else {
		return Quaternion();
	}
}

Variant::operator Transform3D() const {
	if (type == TRANSFORM3D) {
		return *_data._transform3d;
	} else if (type == BASIS) {
		return Transform3D(*_data._basis, Vector3());
	} else if (type == QUATERNION) {
		return Transform3D(Basis(*reinterpret_cast<const Quaternion *>(_data._mem)), Vector3());
	} else if (type == TRANSFORM2D) {
		const Transform2D &t = *_data._transform2d;
		Transform3D m;
		m.basis.elements[0][0] = t.elements[0][0];
		m.basis.elements[1][0] = t.elements[0][1];
		m.basis.elements[0][1] = t.elements[1][0];
		m.basis.elements[1][1] = t.elements[1][1];
		m.origin[0] = t.elements[2][0];
		m.origin[1] = t.elements[2][1];
		return m;
	} else {
		return Transform3D();
	}
}

Variant::operator Transform2D() const {
	if (type == TRANSFORM2D) {
		return *_data._transform2d;
	} else if (type == TRANSFORM3D) {
		const Transform3D &t = *_data._transform3d;
		Transform2D m;
		m.elements[0][0] = t.basis.elements[0][0];
		m.elements[0][1] = t.basis.elements[1][0];
		m.elements[1][0] = t.basis.elements[0][1];
		m.elements[1][1] = t.basis.elements[1][1];
		m.elements[2][0] = t.origin[0];
		m.elements[2][1] = t.origin[1];
		return m;
	} else {
		return Transform2D();
	}
}

Variant::operator Color() const {
	if (type == COLOR) {
		return *reinterpret_cast<const Color *>(_data._mem);
	} else if (type == STRING) {
		return Color(operator String());
	} else if (type == INT) {
		return Color::hex(operator int());
	} else {
		return Color();
	}
}

Variant::operator NodePath() const {
	if (type == NODE_PATH) {
		return *reinterpret_cast<const NodePath *>(_data._mem);
	} else if (type == STRING) {
		return NodePath(operator String());
	} else {
		return NodePath();
	}
}

Variant::operator ::RID() const {
	if (type == RID) {
		return *reinterpret_cast<const ::RID *>(_data._mem);
	} else if (type == OBJECT && _get_obj().obj == nullptr) {
		return ::RID();
	} else if (type == OBJECT && _get_obj().obj) {
#ifdef DEBUG_ENABLED
		if (EngineDebugger::is_active()) {
			ERR_FAIL_COND_V_MSG(ObjectDB::get_instance(_get_obj().id) == nullptr, ::RID(), "Invalid pointer (object was freed).");
		}
#endif
		Callable::CallError ce;
		Variant ret = _get_obj().obj->call(CoreStringNames::get_singleton()->get_rid, nullptr, 0, ce);
		if (ce.error == Callable::CallError::CALL_OK && ret.get_type() == Variant::RID) {
			return ret;
		}
		return ::RID();
	} else {
		return ::RID();
	}
}

Variant::operator Object *() const {
	if (type == OBJECT) {
		return _get_obj().obj;
	} else {
		return nullptr;
	}
}

Object *Variant::get_validated_object_with_check(bool &r_previously_freed) const {
	if (type == OBJECT) {
		Object *instance = ObjectDB::get_instance(_get_obj().id);
		r_previously_freed = !instance && _get_obj().id != ObjectID();
		return instance;
	} else {
		r_previously_freed = false;
		return nullptr;
	}
}

Object *Variant::get_validated_object() const {
	if (type == OBJECT) {
		return ObjectDB::get_instance(_get_obj().id);
	} else {
		return nullptr;
	}
}

Variant::operator Dictionary() const {
	if (type == DICTIONARY) {
		return *reinterpret_cast<const Dictionary *>(_data._mem);
	} else {
		return Dictionary();
	}
}

Variant::operator Callable() const {
	if (type == CALLABLE) {
		return *reinterpret_cast<const Callable *>(_data._mem);
	} else {
		return Callable();
	}
}

Variant::operator Signal() const {
	if (type == SIGNAL) {
		return *reinterpret_cast<const Signal *>(_data._mem);
	} else {
		return Signal();
	}
}

template <class DA, class SA>
inline DA _convert_array(const SA &p_array) {
	DA da;
	da.resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		da.set(i, Variant(p_array.get(i)));
	}

	return da;
}

template <class DA>
inline DA _convert_array_from_variant(const Variant &p_variant) {
	switch (p_variant.get_type()) {
		case Variant::ARRAY: {
			return _convert_array<DA, Array>(p_variant.operator Array());
		}
		case Variant::PACKED_BYTE_ARRAY: {
			return _convert_array<DA, Vector<uint8_t>>(p_variant.operator Vector<uint8_t>());
		}
		case Variant::PACKED_INT32_ARRAY: {
			return _convert_array<DA, Vector<int32_t>>(p_variant.operator Vector<int32_t>());
		}
		case Variant::PACKED_INT64_ARRAY: {
			return _convert_array<DA, Vector<int64_t>>(p_variant.operator Vector<int64_t>());
		}
		case Variant::PACKED_FLOAT32_ARRAY: {
			return _convert_array<DA, Vector<float>>(p_variant.operator Vector<float>());
		}
		case Variant::PACKED_FLOAT64_ARRAY: {
			return _convert_array<DA, Vector<double>>(p_variant.operator Vector<double>());
		}
		case Variant::PACKED_STRING_ARRAY: {
			return _convert_array<DA, Vector<String>>(p_variant.operator Vector<String>());
		}
		case Variant::PACKED_VECTOR2_ARRAY: {
			return _convert_array<DA, Vector<Vector2>>(p_variant.operator Vector<Vector2>());
		}
		case Variant::PACKED_VECTOR3_ARRAY: {
			return _convert_array<DA, Vector<Vector3>>(p_variant.operator Vector<Vector3>());
		}
		case Variant::PACKED_COLOR_ARRAY: {
			return _convert_array<DA, Vector<Color>>(p_variant.operator Vector<Color>());
		}
		default: {
			return DA();
		}
	}
}

Variant::operator Array() const {
	if (type == ARRAY) {
		return *reinterpret_cast<const Array *>(_data._mem);
	} else {
		return _convert_array_from_variant<Array>(*this);
	}
}

Variant::operator Vector<uint8_t>() const {
	if (type == PACKED_BYTE_ARRAY) {
		return static_cast<PackedArrayRef<uint8_t> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<uint8_t>>(*this);
	}
}

Variant::operator Vector<int32_t>() const {
	if (type == PACKED_INT32_ARRAY) {
		return static_cast<PackedArrayRef<int32_t> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<int>>(*this);
	}
}

Variant::operator Vector<int64_t>() const {
	if (type == PACKED_INT64_ARRAY) {
		return static_cast<PackedArrayRef<int64_t> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<int64_t>>(*this);
	}
}

Variant::operator Vector<float>() const {
	if (type == PACKED_FLOAT32_ARRAY) {
		return static_cast<PackedArrayRef<float> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<float>>(*this);
	}
}

Variant::operator Vector<double>() const {
	if (type == PACKED_FLOAT64_ARRAY) {
		return static_cast<PackedArrayRef<double> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<double>>(*this);
	}
}

Variant::operator Vector<String>() const {
	if (type == PACKED_STRING_ARRAY) {
		return static_cast<PackedArrayRef<String> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<String>>(*this);
	}
}

Variant::operator Vector<Vector3>() const {
	if (type == PACKED_VECTOR3_ARRAY) {
		return static_cast<PackedArrayRef<Vector3> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<Vector3>>(*this);
	}
}

Variant::operator Vector<Vector2>() const {
	if (type == PACKED_VECTOR2_ARRAY) {
		return static_cast<PackedArrayRef<Vector2> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<Vector2>>(*this);
	}
}

Variant::operator Vector<Color>() const {
	if (type == PACKED_COLOR_ARRAY) {
		return static_cast<PackedArrayRef<Color> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<Vector<Color>>(*this);
	}
}

/* helpers */

Variant::operator Vector<::RID>() const {
	Array va = operator Array();
	Vector<::RID> rids;
	rids.resize(va.size());
	for (int i = 0; i < rids.size(); i++) {
		rids.write[i] = va[i];
	}
	return rids;
}

Variant::operator Vector<Plane>() const {
	Array va = operator Array();
	Vector<Plane> planes;
	int va_size = va.size();
	if (va_size == 0) {
		return planes;
	}

	planes.resize(va_size);
	Plane *w = planes.ptrw();

	for (int i = 0; i < va_size; i++) {
		w[i] = va[i];
	}

	return planes;
}

Variant::operator Vector<Face3>() const {
	Vector<Vector3> va = operator Vector<Vector3>();
	Vector<Face3> faces;
	int va_size = va.size();
	if (va_size == 0) {
		return faces;
	}

	faces.resize(va_size / 3);
	Face3 *w = faces.ptrw();
	const Vector3 *r = va.ptr();

	for (int i = 0; i < va_size; i++) {
		w[i / 3].vertex[i % 3] = r[i];
	}

	return faces;
}

Variant::operator Vector<Variant>() const {
	Array va = operator Array();
	Vector<Variant> variants;
	int va_size = va.size();
	if (va_size == 0) {
		return variants;
	}

	variants.resize(va_size);
	Variant *w = variants.ptrw();
	for (int i = 0; i < va_size; i++) {
		w[i] = va[i];
	}

	return variants;
}

Variant::operator Vector<StringName>() const {
	Vector<String> from = operator Vector<String>();
	Vector<StringName> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}

Variant::operator Side() const {
	return (Side) operator int();
}

Variant::operator Orientation() const {
	return (Orientation) operator int();
}

Variant::operator IPAddress() const {
	if (type == PACKED_FLOAT32_ARRAY || type == PACKED_INT32_ARRAY || type == PACKED_FLOAT64_ARRAY || type == PACKED_INT64_ARRAY || type == PACKED_BYTE_ARRAY) {
		Vector<int> addr = operator Vector<int>();
		if (addr.size() == 4) {
			return IPAddress(addr.get(0), addr.get(1), addr.get(2), addr.get(3));
		}
	}

	return IPAddress(operator String());
}

Variant::Variant(bool p_bool) {
	type = BOOL;
	_data._bool = p_bool;
}

Variant::Variant(signed int p_int) {
	type = INT;
	_data._int = p_int;
}

Variant::Variant(unsigned int p_int) {
	type = INT;
	_data._int = p_int;
}

#ifdef NEED_LONG_INT

Variant::Variant(signed long p_int) {
	type = INT;
	_data._int = p_int;
}

Variant::Variant(unsigned long p_int) {
	type = INT;
	_data._int = p_int;
}
#endif

Variant::Variant(int64_t p_int) {
	type = INT;
	_data._int = p_int;
}

Variant::Variant(uint64_t p_int) {
	type = INT;
	_data._int = p_int;
}

Variant::Variant(signed short p_short) {
	type = INT;
	_data._int = p_short;
}

Variant::Variant(unsigned short p_short) {
	type = INT;
	_data._int = p_short;
}

Variant::Variant(signed char p_char) {
	type = INT;
	_data._int = p_char;
}

Variant::Variant(unsigned char p_char) {
	type = INT;
	_data._int = p_char;
}

Variant::Variant(float p_float) {
	type = FLOAT;
	_data._float = p_float;
}

Variant::Variant(double p_double) {
	type = FLOAT;
	_data._float = p_double;
}

Variant::Variant(const ObjectID &p_id) {
	type = INT;
	_data._int = p_id;
}

Variant::Variant(const StringName &p_string) {
	type = STRING_NAME;
	memnew_placement(_data._mem, StringName(p_string));
}

Variant::Variant(const String &p_string) {
	type = STRING;
	memnew_placement(_data._mem, String(p_string));
}

Variant::Variant(const char *const p_cstring) {
	type = STRING;
	memnew_placement(_data._mem, String((const char *)p_cstring));
}

Variant::Variant(const char32_t *p_wstring) {
	type = STRING;
	memnew_placement(_data._mem, String(p_wstring));
}

Variant::Variant(const Vector3 &p_vector3) {
	type = VECTOR3;
	memnew_placement(_data._mem, Vector3(p_vector3));
}

Variant::Variant(const Vector3i &p_vector3i) {
	type = VECTOR3I;
	memnew_placement(_data._mem, Vector3i(p_vector3i));
}

Variant::Variant(const Vector2 &p_vector2) {
	type = VECTOR2;
	memnew_placement(_data._mem, Vector2(p_vector2));
}

Variant::Variant(const Vector2i &p_vector2i) {
	type = VECTOR2I;
	memnew_placement(_data._mem, Vector2i(p_vector2i));
}

Variant::Variant(const Rect2 &p_rect2) {
	type = RECT2;
	memnew_placement(_data._mem, Rect2(p_rect2));
}

Variant::Variant(const Rect2i &p_rect2i) {
	type = RECT2I;
	memnew_placement(_data._mem, Rect2i(p_rect2i));
}

Variant::Variant(const Plane &p_plane) {
	type = PLANE;
	memnew_placement(_data._mem, Plane(p_plane));
}

Variant::Variant(const ::AABB &p_aabb) {
	type = AABB;
	_data._aabb = memnew(::AABB(p_aabb));
}

Variant::Variant(const Basis &p_matrix) {
	type = BASIS;
	_data._basis = memnew(Basis(p_matrix));
}

Variant::Variant(const Quaternion &p_quaternion) {
	type = QUATERNION;
	memnew_placement(_data._mem, Quaternion(p_quaternion));
}

Variant::Variant(const Transform3D &p_transform) {
	type = TRANSFORM3D;
	_data._transform3d = memnew(Transform3D(p_transform));
}

Variant::Variant(const Transform2D &p_transform) {
	type = TRANSFORM2D;
	_data._transform2d = memnew(Transform2D(p_transform));
}

Variant::Variant(const Color &p_color) {
	type = COLOR;
	memnew_placement(_data._mem, Color(p_color));
}

Variant::Variant(const NodePath &p_node_path) {
	type = NODE_PATH;
	memnew_placement(_data._mem, NodePath(p_node_path));
}

Variant::Variant(const ::RID &p_rid) {
	type = RID;
	memnew_placement(_data._mem, ::RID(p_rid));
}

Variant::Variant(const Object *p_object) {
	type = OBJECT;

	memnew_placement(_data._mem, ObjData);

	if (p_object) {
		if (p_object->is_ref_counted()) {
			RefCounted *ref_counted = const_cast<RefCounted *>(static_cast<const RefCounted *>(p_object));
			if (!ref_counted->init_ref()) {
				_get_obj().obj = nullptr;
				_get_obj().id = ObjectID();
				return;
			}
		}

		_get_obj().obj = const_cast<Object *>(p_object);
		_get_obj().id = p_object->get_instance_id();
	} else {
		_get_obj().obj = nullptr;
		_get_obj().id = ObjectID();
	}
}

Variant::Variant(const Callable &p_callable) {
	type = CALLABLE;
	memnew_placement(_data._mem, Callable(p_callable));
}

Variant::Variant(const Signal &p_callable) {
	type = SIGNAL;
	memnew_placement(_data._mem, Signal(p_callable));
}

Variant::Variant(const Dictionary &p_dictionary) {
	type = DICTIONARY;
	memnew_placement(_data._mem, Dictionary(p_dictionary));
}

Variant::Variant(const Array &p_array) {
	type = ARRAY;
	memnew_placement(_data._mem, Array(p_array));
}

Variant::Variant(const Vector<Plane> &p_array) {
	type = ARRAY;

	Array *plane_array = memnew_placement(_data._mem, Array);

	plane_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		plane_array->operator[](i) = Variant(p_array[i]);
	}
}

Variant::Variant(const Vector<::RID> &p_array) {
	type = ARRAY;

	Array *rid_array = memnew_placement(_data._mem, Array);

	rid_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		rid_array->set(i, Variant(p_array[i]));
	}
}

Variant::Variant(const Vector<uint8_t> &p_byte_array) {
	type = PACKED_BYTE_ARRAY;

	_data.packed_array = PackedArrayRef<uint8_t>::create(p_byte_array);
}

Variant::Variant(const Vector<int32_t> &p_int32_array) {
	type = PACKED_INT32_ARRAY;
	_data.packed_array = PackedArrayRef<int32_t>::create(p_int32_array);
}

Variant::Variant(const Vector<int64_t> &p_int64_array) {
	type = PACKED_INT64_ARRAY;
	_data.packed_array = PackedArrayRef<int64_t>::create(p_int64_array);
}

Variant::Variant(const Vector<float> &p_float32_array) {
	type = PACKED_FLOAT32_ARRAY;
	_data.packed_array = PackedArrayRef<float>::create(p_float32_array);
}

Variant::Variant(const Vector<double> &p_float64_array) {
	type = PACKED_FLOAT64_ARRAY;
	_data.packed_array = PackedArrayRef<double>::create(p_float64_array);
}

Variant::Variant(const Vector<String> &p_string_array) {
	type = PACKED_STRING_ARRAY;
	_data.packed_array = PackedArrayRef<String>::create(p_string_array);
}

Variant::Variant(const Vector<Vector3> &p_vector3_array) {
	type = PACKED_VECTOR3_ARRAY;
	_data.packed_array = PackedArrayRef<Vector3>::create(p_vector3_array);
}

Variant::Variant(const Vector<Vector2> &p_vector2_array) {
	type = PACKED_VECTOR2_ARRAY;
	_data.packed_array = PackedArrayRef<Vector2>::create(p_vector2_array);
}

Variant::Variant(const Vector<Color> &p_color_array) {
	type = PACKED_COLOR_ARRAY;
	_data.packed_array = PackedArrayRef<Color>::create(p_color_array);
}

Variant::Variant(const Vector<Face3> &p_face_array) {
	Vector<Vector3> vertices;
	int face_count = p_face_array.size();
	vertices.resize(face_count * 3);

	if (face_count) {
		const Face3 *r = p_face_array.ptr();
		Vector3 *w = vertices.ptrw();

		for (int i = 0; i < face_count; i++) {
			for (int j = 0; j < 3; j++) {
				w[i * 3 + j] = r[i].vertex[j];
			}
		}
	}

	type = NIL;

	*this = vertices;
}

/* helpers */
Variant::Variant(const Vector<Variant> &p_array) {
	type = NIL;
	Array arr;
	arr.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		arr[i] = p_array[i];
	}
	*this = arr;
}

Variant::Variant(const Vector<StringName> &p_array) {
	type = NIL;
	Vector<String> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

void Variant::operator=(const Variant &p_variant) {
	if (unlikely(this == &p_variant)) {
		return;
	}

	if (unlikely(type != p_variant.type)) {
		reference(p_variant);
		return;
	}

	switch (p_variant.type) {
		case NIL: {
			// none
		} break;

		// atomic types
		case BOOL: {
			_data._bool = p_variant._data._bool;
		} break;
		case INT: {
			_data._int = p_variant._data._int;
		} break;
		case FLOAT: {
			_data._float = p_variant._data._float;
		} break;
		case STRING: {
			*reinterpret_cast<String *>(_data._mem) = *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

		// math types
		case VECTOR2: {
			*reinterpret_cast<Vector2 *>(_data._mem) = *reinterpret_cast<const Vector2 *>(p_variant._data._mem);
		} break;
		case VECTOR2I: {
			*reinterpret_cast<Vector2i *>(_data._mem) = *reinterpret_cast<const Vector2i *>(p_variant._data._mem);
		} break;
		case RECT2: {
			*reinterpret_cast<Rect2 *>(_data._mem) = *reinterpret_cast<const Rect2 *>(p_variant._data._mem);
		} break;
		case RECT2I: {
			*reinterpret_cast<Rect2i *>(_data._mem) = *reinterpret_cast<const Rect2i *>(p_variant._data._mem);
		} break;
		case TRANSFORM2D: {
			*_data._transform2d = *(p_variant._data._transform2d);
		} break;
		case VECTOR3: {
			*reinterpret_cast<Vector3 *>(_data._mem) = *reinterpret_cast<const Vector3 *>(p_variant._data._mem);
		} break;
		case VECTOR3I: {
			*reinterpret_cast<Vector3i *>(_data._mem) = *reinterpret_cast<const Vector3i *>(p_variant._data._mem);
		} break;
		case PLANE: {
			*reinterpret_cast<Plane *>(_data._mem) = *reinterpret_cast<const Plane *>(p_variant._data._mem);
		} break;

		case AABB: {
			*_data._aabb = *(p_variant._data._aabb);
		} break;
		case QUATERNION: {
			*reinterpret_cast<Quaternion *>(_data._mem) = *reinterpret_cast<const Quaternion *>(p_variant._data._mem);
		} break;
		case BASIS: {
			*_data._basis = *(p_variant._data._basis);
		} break;
		case TRANSFORM3D: {
			*_data._transform3d = *(p_variant._data._transform3d);
		} break;

		// misc types
		case COLOR: {
			*reinterpret_cast<Color *>(_data._mem) = *reinterpret_cast<const Color *>(p_variant._data._mem);
		} break;
		case RID: {
			*reinterpret_cast<::RID *>(_data._mem) = *reinterpret_cast<const ::RID *>(p_variant._data._mem);
		} break;
		case OBJECT: {
			if (_get_obj().id.is_ref_counted()) {
				//we are safe that there is a reference here
				RefCounted *ref_counted = static_cast<RefCounted *>(_get_obj().obj);
				if (ref_counted->unreference()) {
					memdelete(ref_counted);
				}
			}

			if (p_variant._get_obj().obj && p_variant._get_obj().id.is_ref_counted()) {
				RefCounted *ref_counted = static_cast<RefCounted *>(p_variant._get_obj().obj);
				if (!ref_counted->reference()) {
					_get_obj().obj = nullptr;
					_get_obj().id = ObjectID();
					break;
				}
			}

			_get_obj().obj = const_cast<Object *>(p_variant._get_obj().obj);
			_get_obj().id = p_variant._get_obj().id;

		} break;
		case CALLABLE: {
			*reinterpret_cast<Callable *>(_data._mem) = *reinterpret_cast<const Callable *>(p_variant._data._mem);
		} break;
		case SIGNAL: {
			*reinterpret_cast<Signal *>(_data._mem) = *reinterpret_cast<const Signal *>(p_variant._data._mem);
		} break;

		case STRING_NAME: {
			*reinterpret_cast<StringName *>(_data._mem) = *reinterpret_cast<const StringName *>(p_variant._data._mem);
		} break;
		case NODE_PATH: {
			*reinterpret_cast<NodePath *>(_data._mem) = *reinterpret_cast<const NodePath *>(p_variant._data._mem);
		} break;
		case DICTIONARY: {
			*reinterpret_cast<Dictionary *>(_data._mem) = *reinterpret_cast<const Dictionary *>(p_variant._data._mem);
		} break;
		case ARRAY: {
			*reinterpret_cast<Array *>(_data._mem) = *reinterpret_cast<const Array *>(p_variant._data._mem);
		} break;

		// arrays
		case PACKED_BYTE_ARRAY: {
			_data.packed_array = PackedArrayRef<uint8_t>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_INT32_ARRAY: {
			_data.packed_array = PackedArrayRef<int32_t>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_INT64_ARRAY: {
			_data.packed_array = PackedArrayRef<int64_t>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_FLOAT32_ARRAY: {
			_data.packed_array = PackedArrayRef<float>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_FLOAT64_ARRAY: {
			_data.packed_array = PackedArrayRef<double>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_STRING_ARRAY: {
			_data.packed_array = PackedArrayRef<String>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_VECTOR2_ARRAY: {
			_data.packed_array = PackedArrayRef<Vector2>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_VECTOR3_ARRAY: {
			_data.packed_array = PackedArrayRef<Vector3>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case PACKED_COLOR_ARRAY: {
			_data.packed_array = PackedArrayRef<Color>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		default: {
		}
	}
}

Variant::Variant(const IPAddress &p_address) {
	type = STRING;
	memnew_placement(_data._mem, String(p_address));
}

Variant::Variant(const Variant &p_variant) {
	reference(p_variant);
}

uint32_t Variant::hash() const {
	return recursive_hash(0);
}

uint32_t Variant::recursive_hash(int recursion_count) const {
	switch (type) {
		case NIL: {
			return 0;
		} break;
		case BOOL: {
			return _data._bool ? 1 : 0;
		} break;
		case INT: {
			return hash_one_uint64((uint64_t)_data._int);
		} break;
		case FLOAT: {
			return hash_djb2_one_float(_data._float);
		} break;
		case STRING: {
			return reinterpret_cast<const String *>(_data._mem)->hash();
		} break;

		// math types
		case VECTOR2: {
			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Vector2 *>(_data._mem)->x);
			return hash_djb2_one_float(reinterpret_cast<const Vector2 *>(_data._mem)->y, hash);
		} break;
		case VECTOR2I: {
			uint32_t hash = hash_djb2_one_32((uint32_t) reinterpret_cast<const Vector2i *>(_data._mem)->x);
			return hash_djb2_one_32((uint32_t) reinterpret_cast<const Vector2i *>(_data._mem)->y, hash);
		} break;
		case RECT2: {
			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->position.x);
			hash = hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->position.y, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->size.x, hash);
			return hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->size.y, hash);
		} break;
		case RECT2I: {
			uint32_t hash = hash_djb2_one_32((uint32_t) reinterpret_cast<const Rect2i *>(_data._mem)->position.x);
			hash = hash_djb2_one_32((uint32_t) reinterpret_cast<const Rect2i *>(_data._mem)->position.y, hash);
			hash = hash_djb2_one_32((uint32_t) reinterpret_cast<const Rect2i *>(_data._mem)->size.x, hash);
			return hash_djb2_one_32((uint32_t) reinterpret_cast<const Rect2i *>(_data._mem)->size.y, hash);
		} break;
		case TRANSFORM2D: {
			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 2; j++) {
					hash = hash_djb2_one_float(_data._transform2d->elements[i][j], hash);
				}
			}

			return hash;
		} break;
		case VECTOR3: {
			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Vector3 *>(_data._mem)->x);
			hash = hash_djb2_one_float(reinterpret_cast<const Vector3 *>(_data._mem)->y, hash);
			return hash_djb2_one_float(reinterpret_cast<const Vector3 *>(_data._mem)->z, hash);
		} break;
		case VECTOR3I: {
			uint32_t hash = hash_djb2_one_32((uint32_t) reinterpret_cast<const Vector3i *>(_data._mem)->x);
			hash = hash_djb2_one_32((uint32_t) reinterpret_cast<const Vector3i *>(_data._mem)->y, hash);
			return hash_djb2_one_32((uint32_t) reinterpret_cast<const Vector3i *>(_data._mem)->z, hash);
		} break;
		case PLANE: {
			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->normal.x);
			hash = hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->normal.y, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->normal.z, hash);
			return hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->d, hash);

		} break;
		case AABB: {
			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {
				hash = hash_djb2_one_float(_data._aabb->position[i], hash);
				hash = hash_djb2_one_float(_data._aabb->size[i], hash);
			}

			return hash;

		} break;
		case QUATERNION: {
			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Quaternion *>(_data._mem)->x);
			hash = hash_djb2_one_float(reinterpret_cast<const Quaternion *>(_data._mem)->y, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Quaternion *>(_data._mem)->z, hash);
			return hash_djb2_one_float(reinterpret_cast<const Quaternion *>(_data._mem)->w, hash);

		} break;
		case BASIS: {
			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					hash = hash_djb2_one_float(_data._basis->elements[i][j], hash);
				}
			}

			return hash;

		} break;
		case TRANSFORM3D: {
			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					hash = hash_djb2_one_float(_data._transform3d->basis.elements[i][j], hash);
				}
				hash = hash_djb2_one_float(_data._transform3d->origin[i], hash);
			}

			return hash;

		} break;

		// misc types
		case COLOR: {
			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->r);
			hash = hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->g, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->b, hash);
			return hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->a, hash);

		} break;
		case RID: {
			return hash_djb2_one_64(reinterpret_cast<const ::RID *>(_data._mem)->get_id());
		} break;
		case OBJECT: {
			return hash_djb2_one_64(make_uint64_t(_get_obj().obj));
		} break;
		case STRING_NAME: {
			return reinterpret_cast<const StringName *>(_data._mem)->hash();
		} break;
		case NODE_PATH: {
			return reinterpret_cast<const NodePath *>(_data._mem)->hash();
		} break;
		case DICTIONARY: {
			return reinterpret_cast<const Dictionary *>(_data._mem)->recursive_hash(recursion_count);

		} break;
		case CALLABLE: {
			return reinterpret_cast<const Callable *>(_data._mem)->hash();

		} break;
		case SIGNAL: {
			const Signal &s = *reinterpret_cast<const Signal *>(_data._mem);
			uint32_t hash = s.get_name().hash();
			return hash_djb2_one_64(s.get_object_id(), hash);
		} break;
		case ARRAY: {
			const Array &arr = *reinterpret_cast<const Array *>(_data._mem);
			return arr.recursive_hash(recursion_count);

		} break;
		case PACKED_BYTE_ARRAY: {
			const Vector<uint8_t> &arr = PackedArrayRef<uint8_t>::get_array(_data.packed_array);
			int len = arr.size();
			if (likely(len)) {
				const uint8_t *r = arr.ptr();
				return hash_djb2_buffer((uint8_t *)&r[0], len);
			} else {
				return hash_djb2_one_64(0);
			}

		} break;
		case PACKED_INT32_ARRAY: {
			const Vector<int32_t> &arr = PackedArrayRef<int32_t>::get_array(_data.packed_array);
			int len = arr.size();
			if (likely(len)) {
				const int32_t *r = arr.ptr();
				return hash_djb2_buffer((uint8_t *)&r[0], len * sizeof(int32_t));
			} else {
				return hash_djb2_one_64(0);
			}

		} break;
		case PACKED_INT64_ARRAY: {
			const Vector<int64_t> &arr = PackedArrayRef<int64_t>::get_array(_data.packed_array);
			int len = arr.size();
			if (likely(len)) {
				const int64_t *r = arr.ptr();
				return hash_djb2_buffer((uint8_t *)&r[0], len * sizeof(int64_t));
			} else {
				return hash_djb2_one_64(0);
			}

		} break;
		case PACKED_FLOAT32_ARRAY: {
			const Vector<float> &arr = PackedArrayRef<float>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const float *r = arr.ptr();
				return hash_djb2_buffer((uint8_t *)&r[0], len * sizeof(float));
			} else {
				return hash_djb2_one_float(0.0);
			}

		} break;
		case PACKED_FLOAT64_ARRAY: {
			const Vector<double> &arr = PackedArrayRef<double>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const double *r = arr.ptr();
				return hash_djb2_buffer((uint8_t *)&r[0], len * sizeof(double));
			} else {
				return hash_djb2_one_float(0.0);
			}

		} break;
		case PACKED_STRING_ARRAY: {
			uint32_t hash = 5831;
			const Vector<String> &arr = PackedArrayRef<String>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const String *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_32(r[i].hash(), hash);
				}
			}

			return hash;
		} break;
		case PACKED_VECTOR2_ARRAY: {
			uint32_t hash = 5831;
			const Vector<Vector2> &arr = PackedArrayRef<Vector2>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const Vector2 *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_float(r[i].x, hash);
					hash = hash_djb2_one_float(r[i].y, hash);
				}
			}

			return hash;
		} break;
		case PACKED_VECTOR3_ARRAY: {
			uint32_t hash = 5831;
			const Vector<Vector3> &arr = PackedArrayRef<Vector3>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const Vector3 *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_float(r[i].x, hash);
					hash = hash_djb2_one_float(r[i].y, hash);
					hash = hash_djb2_one_float(r[i].z, hash);
				}
			}

			return hash;
		} break;
		case PACKED_COLOR_ARRAY: {
			uint32_t hash = 5831;
			const Vector<Color> &arr = PackedArrayRef<Color>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const Color *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_float(r[i].r, hash);
					hash = hash_djb2_one_float(r[i].g, hash);
					hash = hash_djb2_one_float(r[i].b, hash);
					hash = hash_djb2_one_float(r[i].a, hash);
				}
			}

			return hash;
		} break;
		default: {
		}
	}

	return 0;
}

#define hash_compare_scalar(p_lhs, p_rhs) \
	((p_lhs) == (p_rhs)) || (Math::is_nan(p_lhs) && Math::is_nan(p_rhs))

#define hash_compare_vector2(p_lhs, p_rhs)         \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x)) && \
			(hash_compare_scalar((p_lhs).y, (p_rhs).y))

#define hash_compare_vector3(p_lhs, p_rhs)                 \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x)) &&         \
			(hash_compare_scalar((p_lhs).y, (p_rhs).y)) && \
			(hash_compare_scalar((p_lhs).z, (p_rhs).z))

#define hash_compare_quaternion(p_lhs, p_rhs)              \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x)) &&         \
			(hash_compare_scalar((p_lhs).y, (p_rhs).y)) && \
			(hash_compare_scalar((p_lhs).z, (p_rhs).z)) && \
			(hash_compare_scalar((p_lhs).w, (p_rhs).w))

#define hash_compare_color(p_lhs, p_rhs)                   \
	(hash_compare_scalar((p_lhs).r, (p_rhs).r)) &&         \
			(hash_compare_scalar((p_lhs).g, (p_rhs).g)) && \
			(hash_compare_scalar((p_lhs).b, (p_rhs).b)) && \
			(hash_compare_scalar((p_lhs).a, (p_rhs).a))

#define hash_compare_packed_array(p_lhs, p_rhs, p_type, p_compare_func) \
	const Vector<p_type> &l = PackedArrayRef<p_type>::get_array(p_lhs); \
	const Vector<p_type> &r = PackedArrayRef<p_type>::get_array(p_rhs); \
                                                                        \
	if (l.size() != r.size())                                           \
		return false;                                                   \
                                                                        \
	const p_type *lr = l.ptr();                                         \
	const p_type *rr = r.ptr();                                         \
                                                                        \
	for (int i = 0; i < l.size(); ++i) {                                \
		if (!p_compare_func((lr[i]), (rr[i])))                          \
			return false;                                               \
	}                                                                   \
                                                                        \
	return true

bool Variant::hash_compare(const Variant &p_variant, int recursion_count) const {
	if (type != p_variant.type) {
		return false;
	}

	switch (type) {
		case INT: {
			return _data._int == p_variant._data._int;
		} break;

		case FLOAT: {
			return hash_compare_scalar(_data._float, p_variant._data._float);
		} break;

		case STRING: {
			return *reinterpret_cast<const String *>(_data._mem) == *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

		case VECTOR2: {
			const Vector2 *l = reinterpret_cast<const Vector2 *>(_data._mem);
			const Vector2 *r = reinterpret_cast<const Vector2 *>(p_variant._data._mem);

			return hash_compare_vector2(*l, *r);
		} break;
		case VECTOR2I: {
			const Vector2i *l = reinterpret_cast<const Vector2i *>(_data._mem);
			const Vector2i *r = reinterpret_cast<const Vector2i *>(p_variant._data._mem);
			return *l == *r;
		} break;

		case RECT2: {
			const Rect2 *l = reinterpret_cast<const Rect2 *>(_data._mem);
			const Rect2 *r = reinterpret_cast<const Rect2 *>(p_variant._data._mem);

			return (hash_compare_vector2(l->position, r->position)) &&
					(hash_compare_vector2(l->size, r->size));
		} break;
		case RECT2I: {
			const Rect2i *l = reinterpret_cast<const Rect2i *>(_data._mem);
			const Rect2i *r = reinterpret_cast<const Rect2i *>(p_variant._data._mem);

			return *l == *r;
		} break;

		case TRANSFORM2D: {
			Transform2D *l = _data._transform2d;
			Transform2D *r = p_variant._data._transform2d;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector2(l->elements[i], r->elements[i]))) {
					return false;
				}
			}

			return true;
		} break;

		case VECTOR3: {
			const Vector3 *l = reinterpret_cast<const Vector3 *>(_data._mem);
			const Vector3 *r = reinterpret_cast<const Vector3 *>(p_variant._data._mem);

			return hash_compare_vector3(*l, *r);
		} break;
		case VECTOR3I: {
			const Vector3i *l = reinterpret_cast<const Vector3i *>(_data._mem);
			const Vector3i *r = reinterpret_cast<const Vector3i *>(p_variant._data._mem);

			return *l == *r;
		} break;

		case PLANE: {
			const Plane *l = reinterpret_cast<const Plane *>(_data._mem);
			const Plane *r = reinterpret_cast<const Plane *>(p_variant._data._mem);

			return (hash_compare_vector3(l->normal, r->normal)) &&
					(hash_compare_scalar(l->d, r->d));
		} break;

		case AABB: {
			const ::AABB *l = _data._aabb;
			const ::AABB *r = p_variant._data._aabb;

			return (hash_compare_vector3(l->position, r->position) &&
					(hash_compare_vector3(l->size, r->size)));

		} break;

		case QUATERNION: {
			const Quaternion *l = reinterpret_cast<const Quaternion *>(_data._mem);
			const Quaternion *r = reinterpret_cast<const Quaternion *>(p_variant._data._mem);

			return hash_compare_quaternion(*l, *r);
		} break;

		case BASIS: {
			const Basis *l = _data._basis;
			const Basis *r = p_variant._data._basis;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector3(l->elements[i], r->elements[i]))) {
					return false;
				}
			}

			return true;
		} break;

		case TRANSFORM3D: {
			const Transform3D *l = _data._transform3d;
			const Transform3D *r = p_variant._data._transform3d;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector3(l->basis.elements[i], r->basis.elements[i]))) {
					return false;
				}
			}

			return hash_compare_vector3(l->origin, r->origin);
		} break;

		case COLOR: {
			const Color *l = reinterpret_cast<const Color *>(_data._mem);
			const Color *r = reinterpret_cast<const Color *>(p_variant._data._mem);

			return hash_compare_color(*l, *r);
		} break;

		case ARRAY: {
			const Array &l = *(reinterpret_cast<const Array *>(_data._mem));
			const Array &r = *(reinterpret_cast<const Array *>(p_variant._data._mem));

			if (!l.recursive_equal(r, recursion_count + 1)) {
				return false;
			}

			return true;
		} break;

		case DICTIONARY: {
			const Dictionary &l = *(reinterpret_cast<const Dictionary *>(_data._mem));
			const Dictionary &r = *(reinterpret_cast<const Dictionary *>(p_variant._data._mem));

			if (!l.recursive_equal(r, recursion_count + 1)) {
				return false;
			}

			return true;
		} break;

		// This is for floating point comparisons only.
		case PACKED_FLOAT32_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, float, hash_compare_scalar);
		} break;

		case PACKED_FLOAT64_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, double, hash_compare_scalar);
		} break;

		case PACKED_VECTOR2_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Vector2, hash_compare_vector2);
		} break;

		case PACKED_VECTOR3_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Vector3, hash_compare_vector3);
		} break;

		case PACKED_COLOR_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Color, hash_compare_color);
		} break;

		default:
			bool v;
			Variant r;
			evaluate(OP_EQUAL, *this, p_variant, r, v);
			return r;
	}

	return false;
}

bool Variant::is_ref_counted() const {
	return type == OBJECT && _get_obj().id.is_ref_counted();
}

Vector<Variant> varray() {
	return Vector<Variant>();
}

Vector<Variant> varray(const Variant &p_arg1) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	return v;
}

Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	return v;
}

Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	return v;
}

Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	v.push_back(p_arg4);
	return v;
}

Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	v.push_back(p_arg4);
	v.push_back(p_arg5);
	return v;
}

void Variant::static_assign(const Variant &p_variant) {
}

bool Variant::is_shared() const {
	switch (type) {
		case OBJECT:
			return true;
		case ARRAY:
			return true;
		case DICTIONARY:
			return true;
		default: {
		}
	}

	return false;
}

Variant Variant::call(const StringName &p_method, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;
	int argc = 0;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL) {
			break;
		}
		argc++;
	}

	Callable::CallError error;

	Variant ret;
	call(p_method, argptr, argc, ret, error);

	switch (error.error) {
		case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT: {
			String err = "Invalid type for argument #" + itos(error.argument) + ", expected '" + Variant::get_type_name(Variant::Type(error.expected)) + "'.";
			ERR_PRINT(err.utf8().get_data());

		} break;
		case Callable::CallError::CALL_ERROR_INVALID_METHOD: {
			String err = "Invalid method '" + p_method + "' for type '" + Variant::get_type_name(type) + "'.";
			ERR_PRINT(err.utf8().get_data());
		} break;
		case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {
			String err = "Too many arguments for method '" + p_method + "'";
			ERR_PRINT(err.utf8().get_data());
		} break;
		default: {
		}
	}

	return ret;
}

void Variant::construct_from_string(const String &p_string, Variant &r_value, ObjectConstruct p_obj_construct, void *p_construct_ud) {
	r_value = Variant();
}

String Variant::get_construct_string() const {
	String vars;
	VariantWriter::write_to_string(*this, vars);

	return vars;
}

String Variant::get_call_error_text(const StringName &p_method, const Variant **p_argptrs, int p_argcount, const Callable::CallError &ce) {
	String err_text;

	if (ce.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
		int errorarg = ce.argument;
		if (p_argptrs) {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from " + Variant::get_type_name(p_argptrs[errorarg]->get_type()) + " to " + Variant::get_type_name(Variant::Type(ce.expected)) + ".";
		} else {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from [missing argptr, type unknown] to " + Variant::get_type_name(Variant::Type(ce.expected)) + ".";
		}
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Callable::CallError::CALL_ERROR_INVALID_METHOD) {
		err_text = "Method not found.";
	} else if (ce.error == Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
		err_text = "Instance is null";
	} else if (ce.error == Callable::CallError::CALL_OK) {
		return "Call OK";
	}
	return "'" + String(p_method) + "': " + err_text;
}

String Variant::get_call_error_text(Object *p_base, const StringName &p_method, const Variant **p_argptrs, int p_argcount, const Callable::CallError &ce) {
	String err_text;

	if (ce.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
		int errorarg = ce.argument;
		if (p_argptrs) {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from " + Variant::get_type_name(p_argptrs[errorarg]->get_type()) + " to " + Variant::get_type_name(Variant::Type(ce.expected)) + ".";
		} else {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from [missing argptr, type unknown] to " + Variant::get_type_name(Variant::Type(ce.expected)) + ".";
		}
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Callable::CallError::CALL_ERROR_INVALID_METHOD) {
		err_text = "Method not found.";
	} else if (ce.error == Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
		err_text = "Instance is null";
	} else if (ce.error == Callable::CallError::CALL_OK) {
		return "Call OK";
	}

	String class_name = p_base->get_class();
	Ref<Resource> script = p_base->get_script();
	if (script.is_valid() && script->get_path().is_resource_file()) {
		class_name += "(" + script->get_path().get_file() + ")";
	}
	return "'" + class_name + "::" + String(p_method) + "': " + err_text;
}

String Variant::get_callable_error_text(const Callable &p_callable, const Variant **p_argptrs, int p_argcount, const Callable::CallError &ce) {
	String err_text;

	if (ce.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
		int errorarg = ce.argument;
		if (p_argptrs) {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from " + Variant::get_type_name(p_argptrs[errorarg]->get_type()) + " to " + Variant::get_type_name(Variant::Type(ce.expected)) + ".";
		} else {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from [missing argptr, type unknown] to " + Variant::get_type_name(Variant::Type(ce.expected)) + ".";
		}
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Callable::CallError::CALL_ERROR_INVALID_METHOD) {
		err_text = "Method not found.";
	} else if (ce.error == Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
		err_text = "Instance is null";
	} else if (ce.error == Callable::CallError::CALL_OK) {
		return "Call OK";
	}

	return String(p_callable) + " : " + err_text;
}

String vformat(const String &p_text, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4, const Variant &p5) {
	Array args;
	if (p1.get_type() != Variant::NIL) {
		args.push_back(p1);

		if (p2.get_type() != Variant::NIL) {
			args.push_back(p2);

			if (p3.get_type() != Variant::NIL) {
				args.push_back(p3);

				if (p4.get_type() != Variant::NIL) {
					args.push_back(p4);

					if (p5.get_type() != Variant::NIL) {
						args.push_back(p5);
					}
				}
			}
		}
	}

	bool error = false;
	String fmt = p_text.sprintf(args, &error);

	ERR_FAIL_COND_V_MSG(error, String(), fmt);

	return fmt;
}

void Variant::register_types() {
	_register_variant_operators();
	_register_variant_methods();
	_register_variant_setters_getters();
	_register_variant_constructors();
	_register_variant_destructors();
	_register_variant_utility_functions();
}
void Variant::unregister_types() {
	_unregister_variant_operators();
	_unregister_variant_methods();
	_unregister_variant_setters_getters();
	_unregister_variant_destructors();
	_unregister_variant_utility_functions();
}
