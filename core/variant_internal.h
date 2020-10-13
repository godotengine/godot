/*************************************************************************/
/*  variant_internal.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
	_FORCE_INLINE_ static void initialize(Variant *v, Variant::Type p_type) { v->type = p_type; }

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
	_FORCE_INLINE_ static Quat *get_quat(Variant *v) { return reinterpret_cast<Quat *>(v->_data._mem); }
	_FORCE_INLINE_ static const Quat *get_quat(const Variant *v) { return reinterpret_cast<const Quat *>(v->_data._mem); }
	_FORCE_INLINE_ static ::AABB *get_aabb(Variant *v) { return v->_data._aabb; }
	_FORCE_INLINE_ static const ::AABB *get_aabb(const Variant *v) { return v->_data._aabb; }
	_FORCE_INLINE_ static Basis *get_basis(Variant *v) { return v->_data._basis; }
	_FORCE_INLINE_ static const Basis *get_basis(const Variant *v) { return v->_data._basis; }
	_FORCE_INLINE_ static Transform *get_transform(Variant *v) { return v->_data._transform; }
	_FORCE_INLINE_ static const Transform *get_transform(const Variant *v) { return v->_data._transform; }

	// Misc types.
	_FORCE_INLINE_ static Color *get_color(Variant *v) { return reinterpret_cast<Color *>(v->_data._mem); }
	_FORCE_INLINE_ static const Color *get_color(const Variant *v) { return reinterpret_cast<const Color *>(v->_data._mem); }
	_FORCE_INLINE_ static StringName *get_string_name(Variant *v) { return reinterpret_cast<StringName *>(v->_data._mem); }
	_FORCE_INLINE_ static const StringName *get_string_name(const Variant *v) { return reinterpret_cast<const StringName *>(v->_data._mem); }
	_FORCE_INLINE_ static NodePath *get_node_path(Variant *v) { return reinterpret_cast<NodePath *>(v->_data._mem); }
	_FORCE_INLINE_ static const NodePath *get_node_path(const Variant *v) { return reinterpret_cast<const NodePath *>(v->_data._mem); }
	_FORCE_INLINE_ static RID *get_rid(Variant *v) { return reinterpret_cast<RID *>(v->_data._mem); }
	_FORCE_INLINE_ static const RID *get_rid(const Variant *v) { return reinterpret_cast<const RID *>(v->_data._mem); }
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
struct VariantGetInternalPtr<Transform> {
	static Transform *get_ptr(Variant *v) { return VariantInternal::get_transform(v); }
	static const Transform *get_ptr(const Variant *v) { return VariantInternal::get_transform(v); }
};

template <>
struct VariantGetInternalPtr<Plane> {
	static Plane *get_ptr(Variant *v) { return VariantInternal::get_plane(v); }
	static const Plane *get_ptr(const Variant *v) { return VariantInternal::get_plane(v); }
};

template <>
struct VariantGetInternalPtr<Quat> {
	static Quat *get_ptr(Variant *v) { return VariantInternal::get_quat(v); }
	static const Quat *get_ptr(const Variant *v) { return VariantInternal::get_quat(v); }
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
struct VariantGetInternalPtr<RID> {
	static RID *get_ptr(Variant *v) { return VariantInternal::get_rid(v); }
	static const RID *get_ptr(const Variant *v) { return VariantInternal::get_rid(v); }
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
VARIANT_ACCESSOR_NUMBER(Margin)

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
struct VariantInternalAccessor<Transform> {
	static _FORCE_INLINE_ const Transform &get(const Variant *v) { return *VariantInternal::get_transform(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Transform &p_value) { *VariantInternal::get_transform(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Plane> {
	static _FORCE_INLINE_ const Plane &get(const Variant *v) { return *VariantInternal::get_plane(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Plane &p_value) { *VariantInternal::get_plane(v) = p_value; }
};

template <>
struct VariantInternalAccessor<Quat> {
	static _FORCE_INLINE_ const Quat &get(const Variant *v) { return *VariantInternal::get_quat(v); }
	static _FORCE_INLINE_ void set(Variant *v, const Quat &p_value) { *VariantInternal::get_quat(v) = p_value; }
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
struct VariantInternalAccessor<RID> {
	static _FORCE_INLINE_ const RID &get(const Variant *v) { return *VariantInternal::get_rid(v); }
	static _FORCE_INLINE_ void set(Variant *v, const RID &p_value) { *VariantInternal::get_rid(v) = p_value; }
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

#endif // VARIANT_INTERNAL_H
