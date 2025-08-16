/**************************************************************************/
/*  variant_internal.hpp                                                  */
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

#pragma once

#include <gdextension_interface.h>
#include <godot_cpp/variant/variant.hpp>

namespace godot {
// For use when you want to access the internal pointer of a Variant directly.
// Use with caution. You need to be sure that the type is correct.

namespace internal {
template <typename T>
struct VariantInternalType {};

template <>
struct VariantInternalType<bool> {
	static constexpr Variant::Type type = Variant::BOOL;
};
template <>
struct VariantInternalType<int64_t> {
	static constexpr Variant::Type type = Variant::INT;
};
template <>
struct VariantInternalType<double> {
	static constexpr Variant::Type type = Variant::FLOAT;
};
template <>
struct VariantInternalType<String> {
	static constexpr Variant::Type type = Variant::STRING;
};
template <>
struct VariantInternalType<Vector2> {
	static constexpr Variant::Type type = Variant::VECTOR2;
};
template <>
struct VariantInternalType<Vector2i> {
	static constexpr Variant::Type type = Variant::VECTOR2I;
};
template <>
struct VariantInternalType<Rect2> {
	static constexpr Variant::Type type = Variant::RECT2;
};
template <>
struct VariantInternalType<Rect2i> {
	static constexpr Variant::Type type = Variant::RECT2I;
};
template <>
struct VariantInternalType<Vector3> {
	static constexpr Variant::Type type = Variant::VECTOR3;
};
template <>
struct VariantInternalType<Vector3i> {
	static constexpr Variant::Type type = Variant::VECTOR3I;
};
template <>
struct VariantInternalType<Transform2D> {
	static constexpr Variant::Type type = Variant::TRANSFORM2D;
};
template <>
struct VariantInternalType<Vector4> {
	static constexpr Variant::Type type = Variant::VECTOR4;
};
template <>
struct VariantInternalType<Vector4i> {
	static constexpr Variant::Type type = Variant::VECTOR4I;
};
template <>
struct VariantInternalType<Plane> {
	static constexpr Variant::Type type = Variant::PLANE;
};
template <>
struct VariantInternalType<Quaternion> {
	static constexpr Variant::Type type = Variant::QUATERNION;
};
template <>
struct VariantInternalType<AABB> {
	static constexpr Variant::Type type = Variant::AABB;
};
template <>
struct VariantInternalType<Basis> {
	static constexpr Variant::Type type = Variant::BASIS;
};
template <>
struct VariantInternalType<Transform3D> {
	static constexpr Variant::Type type = Variant::TRANSFORM3D;
};
template <>
struct VariantInternalType<Projection> {
	static constexpr Variant::Type type = Variant::PROJECTION;
};
template <>
struct VariantInternalType<Color> {
	static constexpr Variant::Type type = Variant::COLOR;
};
template <>
struct VariantInternalType<StringName> {
	static constexpr Variant::Type type = Variant::STRING_NAME;
};
template <>
struct VariantInternalType<NodePath> {
	static constexpr Variant::Type type = Variant::NODE_PATH;
};
template <>
struct VariantInternalType<RID> {
	static constexpr Variant::Type type = Variant::RID;
};
template <>
struct VariantInternalType<Object *> {
	static constexpr Variant::Type type = Variant::OBJECT;
};
template <>
struct VariantInternalType<Callable> {
	static constexpr Variant::Type type = Variant::CALLABLE;
};
template <>
struct VariantInternalType<Signal> {
	static constexpr Variant::Type type = Variant::SIGNAL;
};
template <>
struct VariantInternalType<Dictionary> {
	static constexpr Variant::Type type = Variant::DICTIONARY;
};
template <>
struct VariantInternalType<Array> {
	static constexpr Variant::Type type = Variant::ARRAY;
};
template <>
struct VariantInternalType<PackedByteArray> {
	static constexpr Variant::Type type = Variant::PACKED_BYTE_ARRAY;
};
template <>
struct VariantInternalType<PackedInt32Array> {
	static constexpr Variant::Type type = Variant::PACKED_INT32_ARRAY;
};
template <>
struct VariantInternalType<PackedInt64Array> {
	static constexpr Variant::Type type = Variant::PACKED_INT64_ARRAY;
};
template <>
struct VariantInternalType<PackedFloat32Array> {
	static constexpr Variant::Type type = Variant::PACKED_FLOAT32_ARRAY;
};
template <>
struct VariantInternalType<PackedFloat64Array> {
	static constexpr Variant::Type type = Variant::PACKED_FLOAT64_ARRAY;
};
template <>
struct VariantInternalType<PackedStringArray> {
	static constexpr Variant::Type type = Variant::PACKED_STRING_ARRAY;
};
template <>
struct VariantInternalType<PackedVector2Array> {
	static constexpr Variant::Type type = Variant::PACKED_VECTOR2_ARRAY;
};
template <>
struct VariantInternalType<PackedVector3Array> {
	static constexpr Variant::Type type = Variant::PACKED_VECTOR3_ARRAY;
};
template <>
struct VariantInternalType<PackedColorArray> {
	static constexpr Variant::Type type = Variant::PACKED_COLOR_ARRAY;
};
template <>
struct VariantInternalType<PackedVector4Array> {
	static constexpr Variant::Type type = Variant::PACKED_VECTOR4_ARRAY;
};
} //namespace internal

class VariantInternal {
	friend class Variant;

	static GDExtensionVariantGetInternalPtrFunc get_internal_func[Variant::VARIANT_MAX];

	static void init_bindings();

public:
	template <typename T>
	_FORCE_INLINE_ static T *get_internal_value(Variant *v) {
		return static_cast<T *>(get_internal_func[internal::VariantInternalType<T>::type](v));
	}

	template <typename T>
	_FORCE_INLINE_ static const T *get_internal_value(const Variant *v) {
		return static_cast<const T *>(get_internal_func[internal::VariantInternalType<T>::type](const_cast<Variant *>(v)));
	}

	// Atomic types.
	_FORCE_INLINE_ static bool *get_bool(Variant *v) { return get_internal_value<bool>(v); }
	_FORCE_INLINE_ static const bool *get_bool(const Variant *v) { return get_internal_value<bool>(v); }
	_FORCE_INLINE_ static int64_t *get_int(Variant *v) { return get_internal_value<int64_t>(v); }
	_FORCE_INLINE_ static const int64_t *get_int(const Variant *v) { return get_internal_value<int64_t>(v); }
	_FORCE_INLINE_ static double *get_float(Variant *v) { return get_internal_value<double>(v); }
	_FORCE_INLINE_ static const double *get_float(const Variant *v) { return get_internal_value<double>(v); }
	_FORCE_INLINE_ static String *get_string(Variant *v) { return get_internal_value<String>(v); }
	_FORCE_INLINE_ static const String *get_string(const Variant *v) { return get_internal_value<String>(v); }

	// Math types.
	_FORCE_INLINE_ static Vector2 *get_vector2(Variant *v) { return get_internal_value<Vector2>(v); }
	_FORCE_INLINE_ static const Vector2 *get_vector2(const Variant *v) { return get_internal_value<Vector2>(v); }
	_FORCE_INLINE_ static Vector2i *get_vector2i(Variant *v) { return get_internal_value<Vector2i>(v); }
	_FORCE_INLINE_ static const Vector2i *get_vector2i(const Variant *v) { return get_internal_value<Vector2i>(v); }
	_FORCE_INLINE_ static Rect2 *get_rect2(Variant *v) { return get_internal_value<Rect2>(v); }
	_FORCE_INLINE_ static const Rect2 *get_rect2(const Variant *v) { return get_internal_value<Rect2>(v); }
	_FORCE_INLINE_ static Rect2i *get_rect2i(Variant *v) { return get_internal_value<Rect2i>(v); }
	_FORCE_INLINE_ static const Rect2i *get_rect2i(const Variant *v) { return get_internal_value<Rect2i>(v); }
	_FORCE_INLINE_ static Vector3 *get_vector3(Variant *v) { return get_internal_value<Vector3>(v); }
	_FORCE_INLINE_ static const Vector3 *get_vector3(const Variant *v) { return get_internal_value<Vector3>(v); }
	_FORCE_INLINE_ static Vector3i *get_vector3i(Variant *v) { return get_internal_value<Vector3i>(v); }
	_FORCE_INLINE_ static const Vector3i *get_vector3i(const Variant *v) { return get_internal_value<Vector3i>(v); }
	_FORCE_INLINE_ static Vector4 *get_vector4(Variant *v) { return get_internal_value<Vector4>(v); }
	_FORCE_INLINE_ static const Vector4 *get_vector4(const Variant *v) { return get_internal_value<Vector4>(v); }
	_FORCE_INLINE_ static Vector4i *get_vector4i(Variant *v) { return get_internal_value<Vector4i>(v); }
	_FORCE_INLINE_ static const Vector4i *get_vector4i(const Variant *v) { return get_internal_value<Vector4i>(v); }
	_FORCE_INLINE_ static Transform2D *get_transform2d(Variant *v) { return get_internal_value<Transform2D>(v); }
	_FORCE_INLINE_ static const Transform2D *get_transform2d(const Variant *v) { return get_internal_value<Transform2D>(v); }
	_FORCE_INLINE_ static Plane *get_plane(Variant *v) { return get_internal_value<Plane>(v); }
	_FORCE_INLINE_ static const Plane *get_plane(const Variant *v) { return get_internal_value<Plane>(v); }
	_FORCE_INLINE_ static Quaternion *get_quaternion(Variant *v) { return get_internal_value<Quaternion>(v); }
	_FORCE_INLINE_ static const Quaternion *get_quaternion(const Variant *v) { return get_internal_value<Quaternion>(v); }
	_FORCE_INLINE_ static AABB *get_aabb(Variant *v) { return get_internal_value<AABB>(v); }
	_FORCE_INLINE_ static const AABB *get_aabb(const Variant *v) { return get_internal_value<AABB>(v); }
	_FORCE_INLINE_ static Basis *get_basis(Variant *v) { return get_internal_value<Basis>(v); }
	_FORCE_INLINE_ static const Basis *get_basis(const Variant *v) { return get_internal_value<Basis>(v); }
	_FORCE_INLINE_ static Transform3D *get_transform(Variant *v) { return get_internal_value<Transform3D>(v); }
	_FORCE_INLINE_ static const Transform3D *get_transform(const Variant *v) { return get_internal_value<Transform3D>(v); }
	_FORCE_INLINE_ static Projection *get_projection(Variant *v) { return get_internal_value<Projection>(v); }
	_FORCE_INLINE_ static const Projection *get_projection(const Variant *v) { return get_internal_value<Projection>(v); }

	// Misc types.
	_FORCE_INLINE_ static Color *get_color(Variant *v) { return get_internal_value<Color>(v); }
	_FORCE_INLINE_ static const Color *get_color(const Variant *v) { return get_internal_value<Color>(v); }
	_FORCE_INLINE_ static StringName *get_string_name(Variant *v) { return get_internal_value<StringName>(v); }
	_FORCE_INLINE_ static const StringName *get_string_name(const Variant *v) { return get_internal_value<StringName>(v); }
	_FORCE_INLINE_ static NodePath *get_node_path(Variant *v) { return get_internal_value<NodePath>(v); }
	_FORCE_INLINE_ static const NodePath *get_node_path(const Variant *v) { return get_internal_value<NodePath>(v); }
	_FORCE_INLINE_ static RID *get_rid(Variant *v) { return get_internal_value<RID>(v); }
	_FORCE_INLINE_ static const RID *get_rid(const Variant *v) { return get_internal_value<RID>(v); }
	_FORCE_INLINE_ static Callable *get_callable(Variant *v) { return get_internal_value<Callable>(v); }
	_FORCE_INLINE_ static const Callable *get_callable(const Variant *v) { return get_internal_value<Callable>(v); }
	_FORCE_INLINE_ static Signal *get_signal(Variant *v) { return get_internal_value<Signal>(v); }
	_FORCE_INLINE_ static const Signal *get_signal(const Variant *v) { return get_internal_value<Signal>(v); }
	_FORCE_INLINE_ static Dictionary *get_dictionary(Variant *v) { return get_internal_value<Dictionary>(v); }
	_FORCE_INLINE_ static const Dictionary *get_dictionary(const Variant *v) { return get_internal_value<Dictionary>(v); }
	_FORCE_INLINE_ static Array *get_array(Variant *v) { return get_internal_value<Array>(v); }
	_FORCE_INLINE_ static const Array *get_array(const Variant *v) { return get_internal_value<Array>(v); }

	// Typed arrays.
	_FORCE_INLINE_ static PackedByteArray *get_byte_array(Variant *v) { return get_internal_value<PackedByteArray>(v); }
	_FORCE_INLINE_ static const PackedByteArray *get_byte_array(const Variant *v) { return get_internal_value<PackedByteArray>(v); }
	_FORCE_INLINE_ static PackedInt32Array *get_int32_array(Variant *v) { return get_internal_value<PackedInt32Array>(v); }
	_FORCE_INLINE_ static const PackedInt32Array *get_int32_array(const Variant *v) { return get_internal_value<PackedInt32Array>(v); }
	_FORCE_INLINE_ static PackedInt64Array *get_int64_array(Variant *v) { return get_internal_value<PackedInt64Array>(v); }
	_FORCE_INLINE_ static const PackedInt64Array *get_int64_array(const Variant *v) { return get_internal_value<PackedInt64Array>(v); }
	_FORCE_INLINE_ static PackedFloat32Array *get_float32_array(Variant *v) { return get_internal_value<PackedFloat32Array>(v); }
	_FORCE_INLINE_ static const PackedFloat32Array *get_float32_array(const Variant *v) { return get_internal_value<PackedFloat32Array>(v); }
	_FORCE_INLINE_ static PackedFloat64Array *get_float64_array(Variant *v) { return get_internal_value<PackedFloat64Array>(v); }
	_FORCE_INLINE_ static const PackedFloat64Array *get_float64_array(const Variant *v) { return get_internal_value<PackedFloat64Array>(v); }
	_FORCE_INLINE_ static PackedStringArray *get_string_array(Variant *v) { return get_internal_value<PackedStringArray>(v); }
	_FORCE_INLINE_ static const PackedStringArray *get_string_array(const Variant *v) { return get_internal_value<PackedStringArray>(v); }
	_FORCE_INLINE_ static PackedVector2Array *get_vector2_array(Variant *v) { return get_internal_value<PackedVector2Array>(v); }
	_FORCE_INLINE_ static const PackedVector2Array *get_vector2_array(const Variant *v) { return get_internal_value<PackedVector2Array>(v); }
	_FORCE_INLINE_ static PackedVector3Array *get_vector3_array(Variant *v) { return get_internal_value<PackedVector3Array>(v); }
	_FORCE_INLINE_ static const PackedVector3Array *get_vector3_array(const Variant *v) { return get_internal_value<PackedVector3Array>(v); }
	_FORCE_INLINE_ static PackedColorArray *get_color_array(Variant *v) { return get_internal_value<PackedColorArray>(v); }
	_FORCE_INLINE_ static const PackedColorArray *get_color_array(const Variant *v) { return get_internal_value<PackedColorArray>(v); }
	_FORCE_INLINE_ static PackedVector4Array *get_vector4_array(Variant *v) { return get_internal_value<PackedVector4Array>(v); }
	_FORCE_INLINE_ static const PackedVector4Array *get_vector4_array(const Variant *v) { return get_internal_value<PackedVector4Array>(v); }

	_FORCE_INLINE_ static Object **get_object(Variant *v) { return get_internal_value<Object *>(v); }
	_FORCE_INLINE_ static const Object **get_object(const Variant *v) { return (const Object **)get_internal_value<Object *>(v); }

	_FORCE_INLINE_ static void *get_opaque_pointer(Variant *v) {
		switch (v->get_type()) {
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
			case Variant::VECTOR4:
				return get_vector4(v);
			case Variant::VECTOR4I:
				return get_vector4i(v);
			case Variant::RECT2:
				return get_rect2(v);
			case Variant::RECT2I:
				return get_rect2i(v);
			case Variant::TRANSFORM3D:
				return get_transform(v);
			case Variant::PROJECTION:
				return get_projection(v);
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
			case Variant::PACKED_VECTOR4_ARRAY:
				return get_vector4_array(v);
			case Variant::OBJECT:
				return get_object(v);
			case Variant::VARIANT_MAX:
				ERR_FAIL_V(nullptr);
		}
		ERR_FAIL_V(nullptr);
	}

	_FORCE_INLINE_ static const void *get_opaque_pointer(const Variant *v) {
		switch (v->get_type()) {
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
			case Variant::VECTOR4:
				return get_vector4(v);
			case Variant::VECTOR4I:
				return get_vector4i(v);
			case Variant::RECT2:
				return get_rect2(v);
			case Variant::RECT2I:
				return get_rect2i(v);
			case Variant::TRANSFORM3D:
				return get_transform(v);
			case Variant::PROJECTION:
				return get_projection(v);
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
			case Variant::PACKED_VECTOR4_ARRAY:
				return get_vector4_array(v);
			case Variant::OBJECT:
				return get_object(v);
			case Variant::VARIANT_MAX:
				ERR_FAIL_V(nullptr);
		}
		ERR_FAIL_V(nullptr);
	}
};

template <typename T>
struct VariantGetInternalPtr {
	static internal::VariantInternalType<T> *get_ptr(Variant *v) { return VariantInternal::get_internal_value<T>(v); }
	static const internal::VariantInternalType<T> *get_ptr(const Variant *v) { return VariantInternal::get_internal_value<T>(v); }
};

template <typename T>
struct can_set_variant_internal_value {
	static const bool value = true;
};

template <>
struct can_set_variant_internal_value<Object *> {
	static const bool value = false;
};

template <typename T>
struct VariantInternalAccessor {
	static _FORCE_INLINE_ const T &get(const Variant *v) { return *VariantInternal::get_internal_value<T>(v); }

	// Enable set() only for those types where we can set (all but Object *).
	template <typename U = T, typename = std::enable_if_t<can_set_variant_internal_value<U>::value>>
	static _FORCE_INLINE_ void set(Variant *v, const internal::VariantInternalType<U> &p_value) {
		*VariantInternal::get_internal_value<U>(v) = p_value;
	}
};

template <typename T, std::enable_if_t<can_set_variant_internal_value<T>::value>>
struct VariantDefaultInitializer {
	static _FORCE_INLINE_ void init(Variant *v) { *VariantInternal::get_internal_value<T>(v) = T(); }
};

} // namespace godot
