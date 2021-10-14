/*************************************************************************/
/*  variant_internal.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VARIANT_INTERNAL_H
#define VARIANT_INTERNAL_H

#include "variant.h"

// For use when you want to access the internal pointer of a Variant directly.
// Use with caution. You need to be sure that the type is correct.
class VariantInternal {
public:
	// Set type.
	_FORCE_INLINE_ static void initialize(Variant *v, Variant::Type p_type) {
		v->clear();
		v->type = p_type;

		switch (p_type) {
			case Variant::STRING:
				init_string(v);
				break;
			case Variant::TRANSFORM2D:
				init_transform2d(v);
				break;
			case Variant::AABB:
				init_aabb(v);
				break;
			case Variant::BASIS:
				init_basis(v);
				break;
			case Variant::TRANSFORM3D:
				init_transform(v);
				break;
			case Variant::STRING_NAME:
				init_string_name(v);
				break;
			case Variant::NODE_PATH:
				init_node_path(v);
				break;
			case Variant::CALLABLE:
				init_callable(v);
				break;
			case Variant::SIGNAL:
				init_signal(v);
				break;
			case Variant::DICTIONARY:
				init_dictionary(v);
				break;
			case Variant::ARRAY:
				init_array(v);
				break;
			case Variant::PACKED_BYTE_ARRAY:
				init_byte_array(v);
				break;
			case Variant::PACKED_INT32_ARRAY:
				init_int32_array(v);
				break;
			case Variant::PACKED_INT64_ARRAY:
				init_int64_array(v);
				break;
			case Variant::PACKED_FLOAT32_ARRAY:
				init_float32_array(v);
				break;
			case Variant::PACKED_FLOAT64_ARRAY:
				init_float64_array(v);
				break;
			case Variant::PACKED_STRING_ARRAY:
				init_string_array(v);
				break;
			case Variant::PACKED_VECTOR2_ARRAY:
				init_vector2_array(v);
				break;
			case Variant::PACKED_VECTOR3_ARRAY:
				init_vector3_array(v);
				break;
			case Variant::PACKED_COLOR_ARRAY:
				init_color_array(v);
				break;
			case Variant::OBJECT:
				init_object(v);
				break;
			default:
				break;
		}
	}

	// Atomic types.
	_FORCE_INLINE_ static bool *get_bool(Variant *v) { return &v->_data._bool; }
	_FORCE_INLINE_ static const bool *get_bool(const Variant *v) { return &v->_data._bool; }
	_FORCE_INLINE_ static int64_t *get_int(Variant *v) { return &v->_data._int; }
	_FORCE_INLINE_ static const int64_t *get_int(const Variant *v) { return &v->_data._int; }
	_FORCE_INLINE_ static double *get_float(Variant *v) { return &v->_data._float; }
	_FORCE_INLINE_ static const double *get_float(const Variant *v) { return &v->_data._float; }
	_FORCE_INLINE_ static String *get_string(Variant *v) { return reinterpret_cast<String *>(v->_data._mem); }
	_FORCE_INLINE_ static const String *get_string(const Variant *v) { return reinterpret_cast<const String *>(v->_data._mem); }

	// Math types.
	_FORCE_INLINE_ static Vector2 *get_vector2(Variant *v) { return reinterpret_cast<Vector2 *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector2 *get_vector2(const Variant *v) { return reinterpret_cast<const Vector2 *>(v->_data._mem); }
	_FORCE_INLINE_ static Vector2i *get_vector2i(Variant *v) { return reinterpret_cast<Vector2i *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector2i *get_vector2i(const Variant *v) { return reinterpret_cast<const Vector2i *>(v->_data._mem); }
	_FORCE_INLINE_ static Rect2 *get_rect2(Variant *v) { return reinterpret_cast<Rect2 *>(v->_data._mem); }
	_FORCE_INLINE_ static const Rect2 *get_rect2(const Variant *v) { return reinterpret_cast<const Rect2 *>(v->_data._mem); }
	_FORCE_INLINE_ static Rect2i *get_rect2i(Variant *v) { return reinterpret_cast<Rect2i *>(v->_data._mem); }
	_FORCE_INLINE_ static const Rect2i *get_rect2i(const Variant *v) { return reinterpret_cast<const Rect2i *>(v->_data._mem); }
	_FORCE_INLINE_ static Vector3 *get_vector3(Variant *v) { return reinterpret_cast<Vector3 *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector3 *get_vector3(const Variant *v) { return reinterpret_cast<const Vector3 *>(v->_data._mem); }
	_FORCE_INLINE_ static Vector3i *get_vector3i(Variant *v) { return reinterpret_cast<Vector3i *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector3i *get_vector3i(const Variant *v) { return reinterpret_cast<const Vector3i *>(v->_data._mem); }
	_FORCE_INLINE_ static Transform2D *get_transform2d(Variant *v) { return v->_data._transform2d; }
	_FORCE_INLINE_ static const Transform2D *get_transform2d(const Variant *v) { return v->_data._transform2d; }
	_FORCE_INLINE_ static Plane *get_plane(Variant *v) { return reinterpret_cast<Plane *>(v->_data._mem); }
	_FORCE_INLINE_ static const Plane *get_plane(const Variant *v) { return reinterpret_cast<const Plane *>(v->_data._mem); }
	_FORCE_INLINE_ static Quaternion *get_quaternion(Variant *v) { return reinterpret_cast<Quaternion *>(v->_data._mem); }
	_FORCE_INLINE_ static const Quaternion *get_quaternion(const Variant *v) { return reinterpret_cast<const Quaternion *>(v->_data._mem); }
	_FORCE_INLINE_ static ::AABB *get_aabb(Variant *v) { return v->_data._aabb; }
	_FORCE_INLINE_ static const ::AABB *get_aabb(const Variant *v) { return v->_data._aabb; }
	_FORCE_INLINE_ static Basis *get_basis(Variant *v) { return v->_data._basis; }
	_FORCE_INLINE_ static const Basis *get_basis(const Variant *v) { return v->_data._basis; }
	_FORCE_INLINE_ static Transform3D *get_transform(Variant *v) { return v->_data._transform3d; }
	_FORCE_INLINE_ static const Transform3D *get_transform(const Variant *v) { return v->_data._transform3d; }

	// Misc types.
	_FORCE_INLINE_ static Color *get_color(Variant *v) { return reinterpret_cast<Color *>(v->_data._mem); }
	_FORCE_INLINE_ static const Color *get_color(const Variant *v) { return reinterpret_cast<const Color *>(v->_data._mem); }
	_FORCE_INLINE_ static StringName *get_string_name(Variant *v) { return reinterpret_cast<StringName *>(v->_data._mem); }
	_FORCE_INLINE_ static const StringName *get_string_name(const Variant *v) { return reinterpret_cast<const StringName *>(v->_data._mem); }
	_FORCE_INLINE_ static NodePath *get_node_path(Variant *v) { return reinterpret_cast<NodePath *>(v->_data._mem); }
	_FORCE_INLINE_ static const NodePath *get_node_path(const Variant *v) { return reinterpret_cast<const NodePath *>(v->_data._mem); }
	_FORCE_INLINE_ static ::RID *get_rid(Variant *v) { return reinterpret_cast<::RID *>(v->_data._mem); }
	_FORCE_INLINE_ static const ::RID *get_rid(const Variant *v) { return reinterpret_cast<const ::RID *>(v->_data._mem); }
	_FORCE_INLINE_ static Callable *get_callable(Variant *v) { return reinterpret_cast<Callable *>(v->_data._mem); }
	_FORCE_INLINE_ static const Callable *get_callable(const Variant *v) { return reinterpret_cast<const Callable *>(v->_data._mem); }
	_FORCE_INLINE_ static Signal *get_signal(Variant *v) { return reinterpret_cast<Signal *>(v->_data._mem); }
	_FORCE_INLINE_ static const Signal *get_signal(const Variant *v) { return reinterpret_cast<const Signal *>(v->_data._mem); }
	_FORCE_INLINE_ static Dictionary *get_dictionary(Variant *v) { return reinterpret_cast<Dictionary *>(v->_data._mem); }
	_FORCE_INLINE_ static const Dictionary *get_dictionary(const Variant *v) { return reinterpret_cast<const Dictionary *>(v->_data._mem); }
	_FORCE_INLINE_ static Array *get_array(Variant *v) { return reinterpret_cast<Array *>(v->_data._mem); }
	_FORCE_INLINE_ static const Array *get_array(const Variant *v) { return reinterpret_cast<const Array *>(v->_data._mem); }

	// Typed arrays.
	_FORCE_INLINE_ static PackedByteArray *get_byte_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<uint8_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedByteArray *get_byte_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<uint8_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedInt32Array *get_int32_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<int32_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedInt32Array *get_int32_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<int32_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedInt64Array *get_int64_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<int64_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedInt64Array *get_int64_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<int64_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedFloat32Array *get_float32_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<float> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedFloat32Array *get_float32_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<float> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedFloat64Array *get_float64_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<double> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedFloat64Array *get_float64_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<double> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedStringArray *get_string_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<String> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedStringArray *get_string_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<String> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedVector2Array *get_vector2_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<Vector2> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedVector2Array *get_vector2_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<Vector2> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedVector3Array *get_vector3_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<Vector3> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedVector3Array *get_vector3_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<Vector3> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedColorArray *get_color_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<Color> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedColorArray *get_color_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<Color> *>(v->_data.packed_array)->array; }

	_FORCE_INLINE_ static Object **get_object(Variant *v) { return (Object **)&v->_get_obj().obj; }
	_FORCE_INLINE_ static const Object **get_object(const Variant *v) { return (const Object **)&v->_get_obj().obj; }

	_FORCE_INLINE_ static const ObjectID get_object_id(const Variant *v) { return v->_get_obj().id; }

	template <class T>
	_FORCE_INLINE_ static void init_generic(Variant *v) {
		v->type = GetTypeInfo<T>::VARIANT_TYPE;
	}

	// Should be in the same order as Variant::Type for consistency.
	// Those primitive and vector types don't need an `init_` method:
	// Nil, bool, float, Vector2/i, Rect2/i, Vector3/i, Plane, Quat, Color, RID.
	// Object is a special case, handled via `object_assign_null`.
	_FORCE_INLINE_ static void init_string(Variant *v) {
		memnew_placement(v->_data._mem, String);
		v->type = Variant::STRING;
	}

	_FORCE_INLINE_ static void init_transform2d(Variant *v) {
		v->_data._transform2d = memnew(Transform2D);
		v->type = Variant::TRANSFORM2D;
	}
	_FORCE_INLINE_ static void init_aabb(Variant *v) {
		v->_data._aabb = memnew(AABB);
		v->type = Variant::AABB;
	}
	_FORCE_INLINE_ static void init_basis(Variant *v) {
		v->_data._basis = memnew(Basis);
		v->type = Variant::BASIS;
	}
	_FORCE_INLINE_ static void init_transform(Variant *v) {
		v->_data._transform3d = memnew(Transform3D);
		v->type = Variant::TRANSFORM3D;
	}
	_FORCE_INLINE_ static void init_string_name(Variant *v) {
		memnew_placement(v->_data._mem, StringName);
		v->type = Variant::STRING_NAME;
	}
	_FORCE_INLINE_ static void init_node_path(Variant *v) {
		memnew_placement(v->_data._mem, NodePath);
		v->type = Variant::NODE_PATH;
	}
	_FORCE_INLINE_ static void init_callable(Variant *v) {
		memnew_placement(v->_data._mem, Callable);
		v->type = Variant::CALLABLE;
	}
	_FORCE_INLINE_ static void init_signal(Variant *v) {
		memnew_placement(v->_data._mem, Signal);
		v->type = Variant::SIGNAL;
	}
	_FORCE_INLINE_ static void init_dictionary(Variant *v) {
		memnew_placement(v->_data._mem, Dictionary);
		v->type = Variant::DICTIONARY;
	}
	_FORCE_INLINE_ static void init_array(Variant *v) {
		memnew_placement(v->_data._mem, Array);
		v->type = Variant::ARRAY;
	}
	_FORCE_INLINE_ static void init_byte_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<uint8_t>::create(Vector<uint8_t>());
		v->type = Variant::PACKED_BYTE_ARRAY;
	}
	_FORCE_INLINE_ static void init_int32_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<int32_t>::create(Vector<int32_t>());
		v->type = Variant::PACKED_INT32_ARRAY;
	}
	_FORCE_INLINE_ static void init_int64_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<int64_t>::create(Vector<int64_t>());
		v->type = Variant::PACKED_INT64_ARRAY;
	}
	_FORCE_INLINE_ static void init_float32_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<float>::create(Vector<float>());
		v->type = Variant::PACKED_FLOAT32_ARRAY;
	}
	_FORCE_INLINE_ static void init_float64_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<double>::create(Vector<double>());
		v->type = Variant::PACKED_FLOAT64_ARRAY;
	}
	_FORCE_INLINE_ static void init_string_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<String>::create(Vector<String>());
		v->type = Variant::PACKED_STRING_ARRAY;
	}
	_FORCE_INLINE_ static void init_vector2_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<Vector2>::create(Vector<Vector2>());
		v->type = Variant::PACKED_VECTOR2_ARRAY;
	}
	_FORCE_INLINE_ static void init_vector3_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<Vector3>::create(Vector<Vector3>());
		v->type = Variant::PACKED_VECTOR3_ARRAY;
	}
	_FORCE_INLINE_ static void init_color_array(Variant *v) {
		v->_data.packed_array = Variant::PackedArrayRef<Color>::create(Vector<Color>());
		v->type = Variant::PACKED_COLOR_ARRAY;
	}
	_FORCE_INLINE_ static void init_object(Variant *v) {
		object_assign_null(v);
		v->type = Variant::OBJECT;
	}

	_FORCE_INLINE_ static void clear(Variant *v) {
		v->clear();
	}

	static void object_assign(Variant *v, const Object *o); // Needs RefCounted, so it's implemented elsewhere.

	_FORCE_INLINE_ static void object_assign(Variant *v, const Variant *o) {
		object_assign(v, o->_get_obj().obj);
	}

	_FORCE_INLINE_ static void object_assign_null(Variant *v) {
		v->_get_obj().obj = nullptr;
		v->_get_obj().id = ObjectID();
	}

	_FORCE_INLINE_ static void *get_opaque_pointer(Variant *v) {
		switch (v->type) {
			case Variant::NIL:
				return nullptr;
			case Variant::BOOL:
				return get_bool(v);
			case Variant::INT:
				return get_int(v);
			case Variant::FLOAT:
				return get_float(v);
			case Variant::STRING:
				return get_string(v);
			case Variant::VECTOR2:
				return get_vector2(v);
			case Variant::VECTOR2I:
				return get_vector2i(v);
			case Variant::VECTOR3:
				return get_vector3(v);
			case Variant::VECTOR3I:
				return get_vector3i(v);
			case Variant::RECT2:
				return get_rect2(v);
			case Variant::RECT2I:
				return get_rect2i(v);
			case Variant::TRANSFORM3D:
				return get_transform(v);
			case Variant::TRANSFORM2D:
				return get_transform2d(v);
			case Variant::QUATERNION:
				return get_quaternion(v);
			case Variant::PLANE:
				return get_plane(v);
			case Variant::BASIS:
				return get_basis(v);
			case Variant::AABB:
				return get_aabb(v);
			case Variant::COLOR:
				return get_color(v);
			case Variant::STRING_NAME:
				return get_string_name(v);
			case Variant::NODE_PATH:
				return get_node_path(v);
			case Variant::RID:
				return get_rid(v);
			case Variant::CALLABLE:
				return get_callable(v);
			case Variant::SIGNAL:
				return get_signal(v);
			case Variant::DICTIONARY:
				return get_dictionary(v);
			case Variant::ARRAY:
				return get_array(v);
			case Variant::PACKED_BYTE_ARRAY:
				return get_byte_array(v);
			case Variant::PACKED_INT32_ARRAY:
				return get_int32_array(v);
			case Variant::PACKED_INT64_ARRAY:
				return get_int64_array(v);
			case Variant::PACKED_FLOAT32_ARRAY:
				return get_float32_array(v);
			case Variant::PACKED_FLOAT64_ARRAY:
				return get_float64_array(v);
			case Variant::PACKED_STRING_ARRAY:
				return get_string_array(v);
			case Variant::PACKED_VECTOR2_ARRAY:
				return get_vector2_array(v);
			case Variant::PACKED_VECTOR3_ARRAY:
				return get_vector3_array(v);
			case Variant::PACKED_COLOR_ARRAY:
				return get_color_array(v);
			case Variant::OBJECT:
				return v->_get_obj().obj;
			case Variant::VARIANT_MAX:
				ERR_FAIL_V(nullptr);
		}
		ERR_FAIL_V(nullptr);
	}

	_FORCE_INLINE_ static const void *get_opaque_pointer(const Variant *v) {
		switch (v->type) {
			case Variant::NIL:
				return nullptr;
			case Variant::BOOL:
				return get_bool(v);
			case Variant::INT:
				return get_int(v);
			case Variant::FLOAT:
				return get_float(v);
			case Variant::STRING:
				return get_string(v);
			case Variant::VECTOR2:
				return get_vector2(v);
			case Variant::VECTOR2I:
				return get_vector2i(v);
			case Variant::VECTOR3:
				return get_vector3(v);
			case Variant::VECTOR3I:
				return get_vector3i(v);
			case Variant::RECT2:
				return get_rect2(v);
			case Variant::RECT2I:
				return get_rect2i(v);
			case Variant::TRANSFORM3D:
				return get_transform(v);
			case Variant::TRANSFORM2D:
				return get_transform2d(v);
			case Variant::QUATERNION:
				return get_quaternion(v);
			case Variant::PLANE:
				return get_plane(v);
			case Variant::BASIS:
				return get_basis(v);
			case Variant::AABB:
				return get_aabb(v);
			case Variant::COLOR:
				return get_color(v);
			case Variant::STRING_NAME:
				return get_string_name(v);
			case Variant::NODE_PATH:
				return get_node_path(v);
			case Variant::RID:
				return get_rid(v);
			case Variant::CALLABLE:
				return get_callable(v);
			case Variant::SIGNAL:
				return get_signal(v);
			case Variant::DICTIONARY:
				return get_dictionary(v);
			case Variant::ARRAY:
				return get_array(v);
			case Variant::PACKED_BYTE_ARRAY:
				return get_byte_array(v);
			case Variant::PACKED_INT32_ARRAY:
				return get_int32_array(v);
			case Variant::PACKED_INT64_ARRAY:
				return get_int64_array(v);
			case Variant::PACKED_FLOAT32_ARRAY:
				return get_float32_array(v);
			case Variant::PACKED_FLOAT64_ARRAY:
				return get_float64_array(v);
			case Variant::PACKED_STRING_ARRAY:
				return get_string_array(v);
			case Variant::PACKED_VECTOR2_ARRAY:
				return get_vector2_array(v);
			case Variant::PACKED_VECTOR3_ARRAY:
				return get_vector3_array(v);
			case Variant::PACKED_COLOR_ARRAY:
				return get_color_array(v);
			case Variant::OBJECT:
				return v->_get_obj().obj;
			case Variant::VARIANT_MAX:
				ERR_FAIL_V(nullptr);
		}
		ERR_FAIL_V(nullptr);
	}
};

template <class T>
struct VariantGetInternalPtr {
};

template <>
struct VariantGetInternalPtr<bool> {
	static bool *get_ptr(Variant *v) { return VariantInternal::get_bool(v); }
	static const bool *get_ptr(const Variant *v) { return VariantInternal::get_bool(v); }
};

template <>
struct VariantGetInternalPtr<int8_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<uint8_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<int16_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<uint16_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<int32_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<uint32_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<int64_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<uint64_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<char32_t> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<ObjectID> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<Error> {
	static int64_t *get_ptr(Variant *v) { return VariantInternal::get_int(v); }
	static const int64_t *get_ptr(const Variant *v) { return VariantInternal::get_int(v); }
};

template <>
struct VariantGetInternalPtr<float> {
	static double *get_ptr(Variant *v) { return VariantInternal::get_float(v); }
	static const double *get_ptr(const Variant *v) { return VariantInternal::get_float(v); }
};

template <>
struct VariantGetInternalPtr<double> {
	static double *get_ptr(Variant *v) { return VariantInternal::get_float(v); }
	static const double *get_ptr(const Variant *v) { return VariantInternal::get_float(v); }
};

template <>
struct VariantGetInternalPtr<String> {
	static String *get_ptr(Variant *v) { return VariantInternal::get_string(v); }
	static const String *get_ptr(const Variant *v) { return VariantInternal::get_string(v); }
};

template <>
struct VariantGetInternalPtr<Vector2> {
	static Vector2 *get_ptr(Variant *v) { return VariantInternal::get_vector2(v); }
	static const Vector2 *get_ptr(const Variant *v) { return VariantInternal::get_vector2(v); }
};

template <>
struct VariantGetInternalPtr<Vector2i> {
	static Vector2i *get_ptr(Variant *v) { return VariantInternal::get_vector2i(v); }
	static const Vector2i *get_ptr(const Variant *v) { return VariantInternal::get_vector2i(v); }
};

template <>
struct VariantGetInternalPtr<Rect2> {
	static Rect2 *get_ptr(Variant *v) { return VariantInternal::get_rect2(v); }
	static const Rect2 *get_ptr(const Variant *v) { return VariantInternal::get_rect2(v); }
};

template <>
struct VariantGetInternalPtr<Rect2i> {
	static Rect2i *get_ptr(Variant *v) { return VariantInternal::get_rect2i(v); }
	static const Rect2i *get_ptr(const Variant *v) { return VariantInternal::get_rect2i(v); }
};

template <>
struct VariantGetInternalPtr<Vector3> {
	static Vector3 *get_ptr(Variant *v) { return VariantInternal::get_vector3(v); }
	static const Vector3 *get_ptr(const Variant *v) { return VariantInternal::get_vector3(v); }
};

template <>
struct VariantGetInternalPtr<Vector3i> {
	static Vector3i *get_ptr(Variant *v) { return VariantInternal::get_vector3i(v); }
	static const Vector3i *get_ptr(const Variant *v) { return VariantInternal::get_vector3i(v); }
};

template <>
struct VariantGetInternalPtr<Transform2D> {
	static Transform2D *get_ptr(Variant *v) { return VariantInternal::get_transform2d(v); }
	static const Transform2D *get_ptr(const Variant *v) { return VariantInternal::get_transform2d(v); }
};

template <>
struct VariantGetInternalPtr<Transform3D> {
	static Transform3D *get_ptr(Variant *v) { return VariantInternal::get_transform(v); }
	static const Transform3D *get_ptr(const Variant *v) { return VariantInternal::get_transform(v); }
};

template <>
struct VariantGetInternalPtr<Plane> {
	static Plane *get_ptr(Variant *v) { return VariantInternal::get_plane(v); }
	static const Plane *get_ptr(const Variant *v) { return VariantInternal::get_plane(v); }
};

template <>
struct VariantGetInternalPtr<Quaternion> {
	static Quaternion *get_ptr(Variant *v) { return VariantInternal::get_quaternion(v); }
	static const Quaternion *get_ptr(const Variant *v) { return VariantInternal::get_quaternion(v); }
};

template <>
struct VariantGetInternalPtr<::AABB> {
	static ::AABB *get_ptr(Variant *v) { return VariantInternal::get_aabb(v); }
	static const ::AABB *get_ptr(const Variant *v) { return VariantInternal::get_aabb(v); }
};

template <>
struct VariantGetInternalPtr<Basis> {
	static Basis *get_ptr(Variant *v) { return VariantInternal::get_basis(v); }
	static const Basis *get_ptr(const Variant *v) { return VariantInternal::get_basis(v); }
};

//

template <>
struct VariantGetInternalPtr<Color> {
	static Color *get_ptr(Variant *v) { return VariantInternal::get_color(v); }
	static const Color *get_ptr(const Variant *v) { return VariantInternal::get_color(v); }
};

template <>
struct VariantGetInternalPtr<StringName> {
	static StringName *get_ptr(Variant *v) { return VariantInternal::get_string_name(v); }
	static const StringName *get_ptr(const Variant *v) { return VariantInternal::get_string_name(v); }
};

template <>
struct VariantGetInternalPtr<NodePath> {
	static NodePath *get_ptr(Variant *v) { return VariantInternal::get_node_path(v); }
	static const NodePath *get_ptr(const Variant *v) { return VariantInternal::get_node_path(v); }
};

template <>
struct VariantGetInternalPtr<::RID> {
	static ::RID *get_ptr(Variant *v) { return VariantInternal::get_rid(v); }
	static const ::RID *get_ptr(const Variant *v) { return VariantInternal::get_rid(v); }
};

template <>
struct VariantGetInternalPtr<Callable> {
	static Callable *get_ptr(Variant *v) { return VariantInternal::get_callable(v); }
	static const Callable *get_ptr(const Variant *v) { return VariantInternal::get_callable(v); }
};

template <>
struct VariantGetInternalPtr<Signal> {
	static Signal *get_ptr(Variant *v) { return VariantInternal::get_signal(v); }
	static const Signal *get_ptr(const Variant *v) { return VariantInternal::get_signal(v); }
};

template <>
struct VariantGetInternalPtr<Dictionary> {
	static Dictionary *get_ptr(Variant *v) { return VariantInternal::get_dictionary(v); }
	static const Dictionary *get_ptr(const Variant *v) { return VariantInternal::get_dictionary(v); }
};

template <>
struct VariantGetInternalPtr<Array> {
	static Array *get_ptr(Variant *v) { return VariantInternal::get_array(v); }
	static const Array *get_ptr(const Variant *v) { return VariantInternal::get_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedByteArray> {
	static PackedByteArray *get_ptr(Variant *v) { return VariantInternal::get_byte_array(v); }
	static const PackedByteArray *get_ptr(const Variant *v) { return VariantInternal::get_byte_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedInt32Array> {
	static PackedInt32Array *get_ptr(Variant *v) { return VariantInternal::get_int32_array(v); }
	static const PackedInt32Array *get_ptr(const Variant *v) { return VariantInternal::get_int32_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedInt64Array> {
	static PackedInt64Array *get_ptr(Variant *v) { return VariantInternal::get_int64_array(v); }
	static const PackedInt64Array *get_ptr(const Variant *v) { return VariantInternal::get_int64_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedFloat32Array> {
	static PackedFloat32Array *get_ptr(Variant *v) { return VariantInternal::get_float32_array(v); }
	static const PackedFloat32Array *get_ptr(const Variant *v) { return VariantInternal::get_float32_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedFloat64Array> {
	static PackedFloat64Array *get_ptr(Variant *v) { return VariantInternal::get_float64_array(v); }
	static const PackedFloat64Array *get_ptr(const Variant *v) { return VariantInternal::get_float64_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedStringArray> {
	static PackedStringArray *get_ptr(Variant *v) { return VariantInternal::get_string_array(v); }
	static const PackedStringArray *get_ptr(const Variant *v) { return VariantInternal::get_string_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedVector2Array> {
	static PackedVector2Array *get_ptr(Variant *v) { return VariantInternal::get_vector2_array(v); }
	static const PackedVector2Array *get_ptr(const Variant *v) { return VariantInternal::get_vector2_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedVector3Array> {
	static PackedVector3Array *get_ptr(Variant *v) { return VariantInternal::get_vector3_array(v); }
	static const PackedVector3Array *get_ptr(const Variant *v) { return VariantInternal::get_vector3_array(v); }
};

template <>
struct VariantGetInternalPtr<PackedColorArray> {
	static PackedColorArray *get_ptr(Variant *v) { return VariantInternal::get_color_array(v); }
	static const PackedColorArray *get_ptr(const Variant *v) { return VariantInternal::get_color_array(v); }
};

template <class T>
struct VariantInternalAccessor {
};

template <>
struct VariantInternalAccessor<bool> {
	static _FORCE_INLINE_ bool get(const Variant *v) { return *VariantInternal::get_bool(v); }
	static _FORCE_INLINE_ void set(Variant *v, bool p_value) { *VariantInternal::get_bool(v) = p_value; }
};

#define VARIANT_ACCESSOR_NUMBER(m_type)                                                                        \
	template <>                                                                                                \
	struct VariantInternalAccessor<m_type> {                                                                   \
		static _FORCE_INLINE_ m_type get(const Variant *v) { return (m_type)*VariantInternal::get_int(v); }    \
		static _FORCE_INLINE_ void set(Variant *v, m_type p_value) { *VariantInternal::get_int(v) = p_value; } \
	};

VARIANT_ACCESSOR_NUMBER(int8_t)
VARIANT_ACCESSOR_NUMBER(uint8_t)
VARIANT_ACCESSOR_NUMBER(int16_t)
VARIANT_ACCESSOR_NUMBER(uint16_t)
VARIANT_ACCESSOR_NUMBER(int32_t)
VARIANT_ACCESSOR_NUMBER(uint32_t)
VARIANT_ACCESSOR_NUMBER(int64_t)
VARIANT_ACCESSOR_NUMBER(uint64_t)
VARIANT_ACCESSOR_NUMBER(char32_t)
VARIANT_ACCESSOR_NUMBER(Error)
VARIANT_ACCESSOR_NUMBER(Side)

template <>
struct VariantInternalAccessor<ObjectID> {
	static _FORCE_INLINE_ ObjectID get(const Variant *v) { return ObjectID(*VariantInternal::get_int(v)); }
	static _FORCE_INLINE_ void set(Variant *v, ObjectID p_value) { *VariantInternal::get_int(v) = p_value; }
};

template <>
struct VariantInternalAccessor<float> {
	static _FORCE_INLINE_ float get(const Variant *v) { return *VariantInternal::get_float(v); }
	static _FORCE_INLINE_ void set(Variant *v, float p_value) { *VariantInternal::get_float(v) = p_value; }
};

template <>
struct VariantInternalAccessor<double> {
	static _FORCE_INLINE_ double get(const Variant *v) { return *VariantInternal::get_float(v); }
	static _FORCE_INLINE_ void set(Variant *v, double p_value) { *VariantInternal::get_float(v) = p_value; }
};

template <>
struct VariantInternalAccessor<String> {
	static _FORCE_INLINE_ const String &get(const Variant *v) { return *VariantInternal::get_string(v); }
	static _FORCE_INLINE_ void set(Variant *v, const String &p_value) { *VariantInternal::get_string(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Vector2> {
	static _FORCE_INLINE_ const Vector2 &get(const Variant *v) { return *VariantInternal::get_vector2(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Vector2 &p_value) { *VariantInternal::get_vector2(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Vector2i> {
	static _FORCE_INLINE_ const Vector2i &get(const Variant *v) { return *VariantInternal::get_vector2i(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Vector2i &p_value) { *VariantInternal::get_vector2i(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Rect2> {
	static _FORCE_INLINE_ const Rect2 &get(const Variant *v) { return *VariantInternal::get_rect2(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Rect2 &p_value) { *VariantInternal::get_rect2(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Rect2i> {
	static _FORCE_INLINE_ const Rect2i &get(const Variant *v) { return *VariantInternal::get_rect2i(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Rect2i &p_value) { *VariantInternal::get_rect2i(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Vector3> {
	static _FORCE_INLINE_ const Vector3 &get(const Variant *v) { return *VariantInternal::get_vector3(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Vector3 &p_value) { *VariantInternal::get_vector3(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Vector3i> {
	static _FORCE_INLINE_ const Vector3i &get(const Variant *v) { return *VariantInternal::get_vector3i(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Vector3i &p_value) { *VariantInternal::get_vector3i(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Transform2D> {
	static _FORCE_INLINE_ const Transform2D &get(const Variant *v) { return *VariantInternal::get_transform2d(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Transform2D &p_value) { *VariantInternal::get_transform2d(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Transform3D> {
	static _FORCE_INLINE_ const Transform3D &get(const Variant *v) { return *VariantInternal::get_transform(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Transform3D &p_value) { *VariantInternal::get_transform(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Plane> {
	static _FORCE_INLINE_ const Plane &get(const Variant *v) { return *VariantInternal::get_plane(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Plane &p_value) { *VariantInternal::get_plane(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Quaternion> {
	static _FORCE_INLINE_ const Quaternion &get(const Variant *v) { return *VariantInternal::get_quaternion(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Quaternion &p_value) { *VariantInternal::get_quaternion(v) = p_value; }
};

template <>
struct VariantInternalAccessor<AABB> {
	static _FORCE_INLINE_ const AABB &get(const Variant *v) { return *VariantInternal::get_aabb(v); }
	static _FORCE_INLINE_ void set(Variant *v, const AABB &p_value) { *VariantInternal::get_aabb(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Basis> {
	static _FORCE_INLINE_ const Basis &get(const Variant *v) { return *VariantInternal::get_basis(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Basis &p_value) { *VariantInternal::get_basis(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Color> {
	static _FORCE_INLINE_ const Color &get(const Variant *v) { return *VariantInternal::get_color(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Color &p_value) { *VariantInternal::get_color(v) = p_value; }
};

template <>
struct VariantInternalAccessor<StringName> {
	static _FORCE_INLINE_ const StringName &get(const Variant *v) { return *VariantInternal::get_string_name(v); }
	static _FORCE_INLINE_ void set(Variant *v, const StringName &p_value) { *VariantInternal::get_string_name(v) = p_value; }
};

template <>
struct VariantInternalAccessor<NodePath> {
	static _FORCE_INLINE_ const NodePath &get(const Variant *v) { return *VariantInternal::get_node_path(v); }
	static _FORCE_INLINE_ void set(Variant *v, const NodePath &p_value) { *VariantInternal::get_node_path(v) = p_value; }
};

template <>
struct VariantInternalAccessor<::RID> {
	static _FORCE_INLINE_ const ::RID &get(const Variant *v) { return *VariantInternal::get_rid(v); }
	static _FORCE_INLINE_ void set(Variant *v, const ::RID &p_value) { *VariantInternal::get_rid(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Callable> {
	static _FORCE_INLINE_ const Callable &get(const Variant *v) { return *VariantInternal::get_callable(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Callable &p_value) { *VariantInternal::get_callable(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Signal> {
	static _FORCE_INLINE_ const Signal &get(const Variant *v) { return *VariantInternal::get_signal(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Signal &p_value) { *VariantInternal::get_signal(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Dictionary> {
	static _FORCE_INLINE_ const Dictionary &get(const Variant *v) { return *VariantInternal::get_dictionary(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Dictionary &p_value) { *VariantInternal::get_dictionary(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Array> {
	static _FORCE_INLINE_ const Array &get(const Variant *v) { return *VariantInternal::get_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Array &p_value) { *VariantInternal::get_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedByteArray> {
	static _FORCE_INLINE_ const PackedByteArray &get(const Variant *v) { return *VariantInternal::get_byte_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedByteArray &p_value) { *VariantInternal::get_byte_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedInt32Array> {
	static _FORCE_INLINE_ const PackedInt32Array &get(const Variant *v) { return *VariantInternal::get_int32_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedInt32Array &p_value) { *VariantInternal::get_int32_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedInt64Array> {
	static _FORCE_INLINE_ const PackedInt64Array &get(const Variant *v) { return *VariantInternal::get_int64_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedInt64Array &p_value) { *VariantInternal::get_int64_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedFloat32Array> {
	static _FORCE_INLINE_ const PackedFloat32Array &get(const Variant *v) { return *VariantInternal::get_float32_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedFloat32Array &p_value) { *VariantInternal::get_float32_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedFloat64Array> {
	static _FORCE_INLINE_ const PackedFloat64Array &get(const Variant *v) { return *VariantInternal::get_float64_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedFloat64Array &p_value) { *VariantInternal::get_float64_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedStringArray> {
	static _FORCE_INLINE_ const PackedStringArray &get(const Variant *v) { return *VariantInternal::get_string_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedStringArray &p_value) { *VariantInternal::get_string_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedVector2Array> {
	static _FORCE_INLINE_ const PackedVector2Array &get(const Variant *v) { return *VariantInternal::get_vector2_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedVector2Array &p_value) { *VariantInternal::get_vector2_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedVector3Array> {
	static _FORCE_INLINE_ const PackedVector3Array &get(const Variant *v) { return *VariantInternal::get_vector3_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedVector3Array &p_value) { *VariantInternal::get_vector3_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<PackedColorArray> {
	static _FORCE_INLINE_ const PackedColorArray &get(const Variant *v) { return *VariantInternal::get_color_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const PackedColorArray &p_value) { *VariantInternal::get_color_array(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Object *> {
	static _FORCE_INLINE_ Object *get(const Variant *v) { return const_cast<Object *>(*VariantInternal::get_object(v)); }
	static _FORCE_INLINE_ void set(Variant *v, const Object *p_value) { *VariantInternal::get_object(v) = const_cast<Object *>(p_value); }
};

template <>
struct VariantInternalAccessor<Variant> {
	static _FORCE_INLINE_ Variant &get(Variant *v) { return *v; }
	static _FORCE_INLINE_ const Variant &get(const Variant *v) { return *v; }
	static _FORCE_INLINE_ void set(Variant *v, const Variant &p_value) { *v = p_value; }
};

template <>
struct VariantInternalAccessor<Vector<Variant>> {
	static _FORCE_INLINE_ Vector<Variant> get(const Variant *v) {
		Vector<Variant> ret;
		int s = VariantInternal::get_array(v)->size();
		ret.resize(s);
		for (int i = 0; i < s; i++) {
			ret.write[i] = VariantInternal::get_array(v)->get(i);
		}

		return ret;
	}
	static _FORCE_INLINE_ void set(Variant *v, const Vector<Variant> &p_value) {
		int s = p_value.size();
		VariantInternal::get_array(v)->resize(s);
		for (int i = 0; i < s; i++) {
			VariantInternal::get_array(v)->set(i, p_value[i]);
		}
	}
};

template <class T>
struct VariantInitializer {
};

template <>
struct VariantInitializer<bool> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<bool>(v); }
};

#define INITIALIZER_INT(m_type)                                                                    \
	template <>                                                                                    \
	struct VariantInitializer<m_type> {                                                            \
		static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<int64_t>(v); } \
	};

INITIALIZER_INT(uint8_t)
INITIALIZER_INT(int8_t)
INITIALIZER_INT(uint16_t)
INITIALIZER_INT(int16_t)
INITIALIZER_INT(uint32_t)
INITIALIZER_INT(int32_t)
INITIALIZER_INT(uint64_t)
INITIALIZER_INT(int64_t)
INITIALIZER_INT(char32_t)
INITIALIZER_INT(Error)
INITIALIZER_INT(ObjectID)

template <>
struct VariantInitializer<double> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<double>(v); }
};

template <>
struct VariantInitializer<float> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<double>(v); }
};

template <>
struct VariantInitializer<String> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_string(v); }
};

template <>
struct VariantInitializer<Vector2> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Vector2>(v); }
};

template <>
struct VariantInitializer<Vector2i> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Vector2i>(v); }
};

template <>
struct VariantInitializer<Rect2> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Rect2>(v); }
};

template <>
struct VariantInitializer<Rect2i> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Rect2i>(v); }
};

template <>
struct VariantInitializer<Vector3> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Vector3>(v); }
};

template <>
struct VariantInitializer<Vector3i> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Vector3i>(v); }
};

template <>
struct VariantInitializer<Transform2D> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_transform2d(v); }
};

template <>
struct VariantInitializer<Plane> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Plane>(v); }
};

template <>
struct VariantInitializer<Quaternion> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Quaternion>(v); }
};

template <>
struct VariantInitializer<AABB> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_aabb(v); }
};

template <>
struct VariantInitializer<Basis> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_basis(v); }
};

template <>
struct VariantInitializer<Transform3D> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_transform(v); }
};

template <>
struct VariantInitializer<Color> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<Color>(v); }
};

template <>
struct VariantInitializer<StringName> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_string_name(v); }
};

template <>
struct VariantInitializer<NodePath> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_node_path(v); }
};

template <>
struct VariantInitializer<::RID> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_generic<::RID>(v); }
};

template <>
struct VariantInitializer<Callable> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_callable(v); }
};

template <>
struct VariantInitializer<Signal> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_signal(v); }
};

template <>
struct VariantInitializer<Dictionary> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_dictionary(v); }
};

template <>
struct VariantInitializer<Array> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_array(v); }
};

template <>
struct VariantInitializer<PackedByteArray> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_byte_array(v); }
};

template <>
struct VariantInitializer<PackedInt32Array> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_int32_array(v); }
};

template <>
struct VariantInitializer<PackedInt64Array> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_int64_array(v); }
};

template <>
struct VariantInitializer<PackedFloat32Array> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_float32_array(v); }
};

template <>
struct VariantInitializer<PackedFloat64Array> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_float64_array(v); }
};

template <>
struct VariantInitializer<PackedStringArray> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_string_array(v); }
};

template <>
struct VariantInitializer<PackedVector2Array> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_vector2_array(v); }
};

template <>
struct VariantInitializer<PackedVector3Array> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_vector3_array(v); }
};

template <>
struct VariantInitializer<PackedColorArray> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_color_array(v); }
};

template <>
struct VariantInitializer<Object *> {
	static _FORCE_INLINE_ void init(Variant *v) { VariantInternal::init_object(v); }
};

template <class T>
struct VariantZeroAssigner {
};

template <>
struct VariantZeroAssigner<bool> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_bool(v) = false; }
};

template <>
struct VariantZeroAssigner<int64_t> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_int(v) = 0; }
};

template <>
struct VariantZeroAssigner<double> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_float(v) = 0.0; }
};

template <>
struct VariantZeroAssigner<float> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_float(v) = 0.0; }
};

template <>
struct VariantZeroAssigner<String> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<Vector2> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_vector2(v) = Vector2(); }
};

template <>
struct VariantZeroAssigner<Vector2i> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_vector2i(v) = Vector2i(); }
};

template <>
struct VariantZeroAssigner<Rect2> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_rect2(v) = Rect2(); }
};

template <>
struct VariantZeroAssigner<Rect2i> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_rect2i(v) = Rect2i(); }
};

template <>
struct VariantZeroAssigner<Vector3> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_vector3(v) = Vector3(); }
};

template <>
struct VariantZeroAssigner<Vector3i> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_vector3i(v) = Vector3i(); }
};

template <>
struct VariantZeroAssigner<Transform2D> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_transform2d(v) = Transform2D(); }
};

template <>
struct VariantZeroAssigner<Plane> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_plane(v) = Plane(); }
};

template <>
struct VariantZeroAssigner<Quaternion> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_quaternion(v) = Quaternion(); }
};

template <>
struct VariantZeroAssigner<AABB> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_aabb(v) = AABB(); }
};

template <>
struct VariantZeroAssigner<Basis> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_basis(v) = Basis(); }
};

template <>
struct VariantZeroAssigner<Transform3D> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_transform(v) = Transform3D(); }
};

template <>
struct VariantZeroAssigner<Color> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_color(v) = Color(); }
};

template <>
struct VariantZeroAssigner<StringName> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<NodePath> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<::RID> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_rid(v) = RID(); }
};

template <>
struct VariantZeroAssigner<Callable> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<Signal> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<Dictionary> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_dictionary(v) = Dictionary(); }
};

template <>
struct VariantZeroAssigner<Array> {
	static _FORCE_INLINE_ void zero(Variant *v) { *VariantInternal::get_array(v) = Array(); }
};

template <>
struct VariantZeroAssigner<PackedByteArray> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedInt32Array> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedInt64Array> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedFloat32Array> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedFloat64Array> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedStringArray> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedVector2Array> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedVector3Array> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <>
struct VariantZeroAssigner<PackedColorArray> {
	static _FORCE_INLINE_ void zero(Variant *v) {}
};

template <class T>
struct VariantTypeChanger {
	static _FORCE_INLINE_ void change(Variant *v) {
		if (v->get_type() != GetTypeInfo<T>::VARIANT_TYPE || GetTypeInfo<T>::VARIANT_TYPE >= Variant::PACKED_BYTE_ARRAY) { //second condition removed by optimizer
			VariantInternal::clear(v);
			VariantInitializer<T>::init(v);
		}
	}
	static _FORCE_INLINE_ void change_and_reset(Variant *v) {
		if (v->get_type() != GetTypeInfo<T>::VARIANT_TYPE || GetTypeInfo<T>::VARIANT_TYPE >= Variant::PACKED_BYTE_ARRAY) { //second condition removed by optimizer
			VariantInternal::clear(v);
			VariantInitializer<T>::init(v);
		}

		VariantZeroAssigner<T>::zero(v);
	}
};

template <class T>
struct VariantTypeAdjust {
	_FORCE_INLINE_ static void adjust(Variant *r_ret) {
		VariantTypeChanger<typename GetSimpleTypeT<T>::type_t>::change(r_ret);
	}
};

template <>
struct VariantTypeAdjust<Variant> {
	_FORCE_INLINE_ static void adjust(Variant *r_ret) {
		// Do nothing for variant.
	}
};

template <>
struct VariantTypeAdjust<Object *> {
	_FORCE_INLINE_ static void adjust(Variant *r_ret) {
		VariantInternal::clear(r_ret);
		*r_ret = (Object *)nullptr;
	}
};

// GDNative extension helpers.

template <class T>
struct VariantTypeConstructor {
	_FORCE_INLINE_ static void variant_from_type(void *p_variant, void *p_value) {
		Variant *variant = reinterpret_cast<Variant *>(p_variant);
		VariantInitializer<T>::init(variant);
		VariantInternalAccessor<T>::set(variant, *((T *)p_value));
	}

	_FORCE_INLINE_ static void type_from_variant(void *p_value, void *p_variant) {
		*((T *)p_value) = VariantInternalAccessor<T>::get(reinterpret_cast<Variant *>(p_variant));
	}
};

template <>
struct VariantTypeConstructor<Object *> {
	_FORCE_INLINE_ static void variant_from_type(void *p_variant, void *p_value) {
		Variant *variant = reinterpret_cast<Variant *>(p_variant);
		VariantInitializer<Object *>::init(variant);
		Object *value = *(reinterpret_cast<Object **>(p_value));
		if (value) {
			VariantInternalAccessor<Object *>::set(variant, value);
			VariantInternalAccessor<ObjectID>::set(variant, value->get_instance_id());
		}
	}

	_FORCE_INLINE_ static void type_from_variant(void *p_value, void *p_variant) {
		Object **value = reinterpret_cast<Object **>(p_value);
		*value = VariantInternalAccessor<Object *>::get(reinterpret_cast<Variant *>(p_variant));
	}
};

#endif // VARIANT_INTERNAL_H
