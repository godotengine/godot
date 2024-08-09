/**************************************************************************/
/*  typed_set.h                                                           */
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

#ifndef TYPED_SET_H
#define TYPED_SET_H

#include "core/object/object.h"
#include "core/variant/binder_common.h"
#include "core/variant/method_ptrcall.h"
#include "core/variant/set.h"
#include "core/variant/type_info.h"
#include "core/variant/variant.h"

template <typename T>
class TypedSet : public Set {
public:
	_FORCE_INLINE_ void operator=(const Set &p_set) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_set), "Cannot assign a set with a different element type.");
		_ref(p_set);
	}
	_FORCE_INLINE_ TypedSet(const Variant &p_variant) :
			TypedSet(Set(p_variant)) {
	}
	_FORCE_INLINE_ TypedSet(const Set &p_set) {
		set_typed(Variant::OBJECT, T::get_class_static(), Variant());
		if (is_same_typed(p_set)) {
			_ref(p_set);
		} else {
			assign(p_set);
		}
	}
	_FORCE_INLINE_ TypedSet() {
		set_typed(Variant::OBJECT, T::get_class_static(), Variant());
	}
};

template <typename T>
struct VariantInternalAccessor<TypedSet<T>> {
	static _FORCE_INLINE_ TypedSet<T> get(const Variant *v) { return *VariantInternal::get_set(v); }
	static _FORCE_INLINE_ void set(Variant *v, const TypedSet<T> &p_set) { *VariantInternal::get_set(v) = p_set; }
};
template <typename T>
struct VariantInternalAccessor<const TypedSet<T> &> {
	static _FORCE_INLINE_ TypedSet<T> get(const Variant *v) { return *VariantInternal::get_set(v); }
	static _FORCE_INLINE_ void set(Variant *v, const TypedSet<T> &p_set) { *VariantInternal::get_set(v) = p_set; }
};

//specialization for the rest of variant types

#define MAKE_TYPED_SET(m_type, m_variant_type)                                                              \
	template <>                                                                                             \
	class TypedSet<m_type> : public Set {                                                                   \
	public:                                                                                                 \
		_FORCE_INLINE_ void operator=(const Set &p_set) {                                                   \
			ERR_FAIL_COND_MSG(!is_same_typed(p_set), "Cannot assign a set with a different element type."); \
			_ref(p_set);                                                                                    \
		}                                                                                                   \
		_FORCE_INLINE_ TypedSet(const Variant &p_variant) :                                                 \
				TypedSet(Set(p_variant)) {                                                                  \
		}                                                                                                   \
		_FORCE_INLINE_ TypedSet(const Set &p_set) {                                                         \
			set_typed(m_variant_type, StringName(), Variant());                                             \
			if (is_same_typed(p_set)) {                                                                     \
				_ref(p_set);                                                                                \
			} else {                                                                                        \
				assign(p_set);                                                                              \
			}                                                                                               \
		}                                                                                                   \
		_FORCE_INLINE_ TypedSet() {                                                                         \
			set_typed(m_variant_type, StringName(), Variant());                                             \
		}                                                                                                   \
	};

// All Variant::OBJECT types are intentionally omitted from this list because they are handled by
// the unspecialized TypedArray definition.
MAKE_TYPED_SET(bool, Variant::BOOL)
MAKE_TYPED_SET(uint8_t, Variant::INT)
MAKE_TYPED_SET(int8_t, Variant::INT)
MAKE_TYPED_SET(uint16_t, Variant::INT)
MAKE_TYPED_SET(int16_t, Variant::INT)
MAKE_TYPED_SET(uint32_t, Variant::INT)
MAKE_TYPED_SET(int32_t, Variant::INT)
MAKE_TYPED_SET(uint64_t, Variant::INT)
MAKE_TYPED_SET(int64_t, Variant::INT)
MAKE_TYPED_SET(float, Variant::FLOAT)
MAKE_TYPED_SET(double, Variant::FLOAT)
MAKE_TYPED_SET(String, Variant::STRING)
MAKE_TYPED_SET(Vector2, Variant::VECTOR2)
MAKE_TYPED_SET(Vector2i, Variant::VECTOR2I)
MAKE_TYPED_SET(Rect2, Variant::RECT2)
MAKE_TYPED_SET(Rect2i, Variant::RECT2I)
MAKE_TYPED_SET(Vector3, Variant::VECTOR3)
MAKE_TYPED_SET(Vector3i, Variant::VECTOR3I)
MAKE_TYPED_SET(Transform2D, Variant::TRANSFORM2D)
MAKE_TYPED_SET(Vector4, Variant::VECTOR4)
MAKE_TYPED_SET(Vector4i, Variant::VECTOR4I)
MAKE_TYPED_SET(Plane, Variant::PLANE)
MAKE_TYPED_SET(Quaternion, Variant::QUATERNION)
MAKE_TYPED_SET(AABB, Variant::AABB)
MAKE_TYPED_SET(Basis, Variant::BASIS)
MAKE_TYPED_SET(Transform3D, Variant::TRANSFORM3D)
MAKE_TYPED_SET(Projection, Variant::PROJECTION)
MAKE_TYPED_SET(Color, Variant::COLOR)
MAKE_TYPED_SET(StringName, Variant::STRING_NAME)
MAKE_TYPED_SET(NodePath, Variant::NODE_PATH)
MAKE_TYPED_SET(RID, Variant::RID)
MAKE_TYPED_SET(Callable, Variant::CALLABLE)
MAKE_TYPED_SET(Signal, Variant::SIGNAL)
MAKE_TYPED_SET(Dictionary, Variant::DICTIONARY)
MAKE_TYPED_SET(Set, Variant::SET)
MAKE_TYPED_SET(Array, Variant::ARRAY)
MAKE_TYPED_SET(PackedByteArray, Variant::PACKED_BYTE_ARRAY)
MAKE_TYPED_SET(PackedInt32Array, Variant::PACKED_INT32_ARRAY)
MAKE_TYPED_SET(PackedInt64Array, Variant::PACKED_INT64_ARRAY)
MAKE_TYPED_SET(PackedFloat32Array, Variant::PACKED_FLOAT32_ARRAY)
MAKE_TYPED_SET(PackedFloat64Array, Variant::PACKED_FLOAT64_ARRAY)
MAKE_TYPED_SET(PackedStringArray, Variant::PACKED_STRING_ARRAY)
MAKE_TYPED_SET(PackedVector2Array, Variant::PACKED_VECTOR2_ARRAY)
MAKE_TYPED_SET(PackedVector3Array, Variant::PACKED_VECTOR3_ARRAY)
MAKE_TYPED_SET(PackedColorArray, Variant::PACKED_COLOR_ARRAY)
MAKE_TYPED_SET(PackedVector4Array, Variant::PACKED_VECTOR4_ARRAY)
MAKE_TYPED_SET(IPAddress, Variant::STRING)

template <typename T>
struct PtrToArg<TypedSet<T>> {
	_FORCE_INLINE_ static TypedSet<T> convert(const void *p_ptr) {
		return TypedSet<T>(*reinterpret_cast<const Set *>(p_ptr));
	}
	typedef Set EncodeT;
	_FORCE_INLINE_ static void encode(TypedSet<T> p_val, void *p_ptr) {
		*(Set *)p_ptr = p_val;
	}
};

template <typename T>
struct PtrToArg<const TypedSet<T> &> {
	typedef Set EncodeT;
	_FORCE_INLINE_ static TypedSet<T> convert(const void *p_ptr) {
		return TypedSet<T>(*reinterpret_cast<const Set *>(p_ptr));
	}
};

template <typename T>
struct GetTypeInfo<TypedSet<T>> {
	static const Variant::Type VARIANT_TYPE = Variant::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, T::get_class_static());
	}
};

template <typename T>
struct GetTypeInfo<const TypedSet<T> &> {
	static const Variant::Type VARIANT_TYPE = Variant::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, T::get_class_static());
	}
};

#define MAKE_TYPED_SET_INFO(m_type, m_variant_type)                                                                          \
	template <>                                                                                                              \
	struct GetTypeInfo<TypedSet<m_type>> {                                                                                   \
		static const Variant::Type VARIANT_TYPE = Variant::ARRAY;                                                            \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;                                        \
		static inline PropertyInfo get_class_info() {                                                                        \
			return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, Variant::get_type_name(m_variant_type)); \
		}                                                                                                                    \
	};                                                                                                                       \
	template <>                                                                                                              \
	struct GetTypeInfo<const TypedSet<m_type> &> {                                                                           \
		static const Variant::Type VARIANT_TYPE = Variant::ARRAY;                                                            \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;                                        \
		static inline PropertyInfo get_class_info() {                                                                        \
			return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, Variant::get_type_name(m_variant_type)); \
		}                                                                                                                    \
	};

MAKE_TYPED_SET_INFO(bool, Variant::BOOL)
MAKE_TYPED_SET_INFO(uint8_t, Variant::INT)
MAKE_TYPED_SET_INFO(int8_t, Variant::INT)
MAKE_TYPED_SET_INFO(uint16_t, Variant::INT)
MAKE_TYPED_SET_INFO(int16_t, Variant::INT)
MAKE_TYPED_SET_INFO(uint32_t, Variant::INT)
MAKE_TYPED_SET_INFO(int32_t, Variant::INT)
MAKE_TYPED_SET_INFO(uint64_t, Variant::INT)
MAKE_TYPED_SET_INFO(int64_t, Variant::INT)
MAKE_TYPED_SET_INFO(float, Variant::FLOAT)
MAKE_TYPED_SET_INFO(double, Variant::FLOAT)
MAKE_TYPED_SET_INFO(String, Variant::STRING)
MAKE_TYPED_SET_INFO(Vector2, Variant::VECTOR2)
MAKE_TYPED_SET_INFO(Vector2i, Variant::VECTOR2I)
MAKE_TYPED_SET_INFO(Rect2, Variant::RECT2)
MAKE_TYPED_SET_INFO(Rect2i, Variant::RECT2I)
MAKE_TYPED_SET_INFO(Vector3, Variant::VECTOR3)
MAKE_TYPED_SET_INFO(Vector3i, Variant::VECTOR3I)
MAKE_TYPED_SET_INFO(Transform2D, Variant::TRANSFORM2D)
MAKE_TYPED_SET_INFO(Vector4, Variant::VECTOR4)
MAKE_TYPED_SET_INFO(Vector4i, Variant::VECTOR4I)
MAKE_TYPED_SET_INFO(Plane, Variant::PLANE)
MAKE_TYPED_SET_INFO(Quaternion, Variant::QUATERNION)
MAKE_TYPED_SET_INFO(AABB, Variant::AABB)
MAKE_TYPED_SET_INFO(Basis, Variant::BASIS)
MAKE_TYPED_SET_INFO(Transform3D, Variant::TRANSFORM3D)
MAKE_TYPED_SET_INFO(Projection, Variant::PROJECTION)
MAKE_TYPED_SET_INFO(Color, Variant::COLOR)
MAKE_TYPED_SET_INFO(StringName, Variant::STRING_NAME)
MAKE_TYPED_SET_INFO(NodePath, Variant::NODE_PATH)
MAKE_TYPED_SET_INFO(RID, Variant::RID)
MAKE_TYPED_SET_INFO(Callable, Variant::CALLABLE)
MAKE_TYPED_SET_INFO(Signal, Variant::SIGNAL)
MAKE_TYPED_SET_INFO(Dictionary, Variant::DICTIONARY)
MAKE_TYPED_SET_INFO(Set, Variant::SET)
MAKE_TYPED_SET_INFO(Array, Variant::ARRAY)
MAKE_TYPED_SET_INFO(PackedByteArray, Variant::PACKED_BYTE_ARRAY)
MAKE_TYPED_SET_INFO(PackedInt32Array, Variant::PACKED_INT32_ARRAY)
MAKE_TYPED_SET_INFO(PackedInt64Array, Variant::PACKED_INT64_ARRAY)
MAKE_TYPED_SET_INFO(PackedFloat32Array, Variant::PACKED_FLOAT32_ARRAY)
MAKE_TYPED_SET_INFO(PackedFloat64Array, Variant::PACKED_FLOAT64_ARRAY)
MAKE_TYPED_SET_INFO(PackedStringArray, Variant::PACKED_STRING_ARRAY)
MAKE_TYPED_SET_INFO(PackedVector2Array, Variant::PACKED_VECTOR2_ARRAY)
MAKE_TYPED_SET_INFO(PackedVector3Array, Variant::PACKED_VECTOR3_ARRAY)
MAKE_TYPED_SET_INFO(PackedColorArray, Variant::PACKED_COLOR_ARRAY)
MAKE_TYPED_SET_INFO(PackedVector4Array, Variant::PACKED_VECTOR4_ARRAY)
MAKE_TYPED_SET_INFO(IPAddress, Variant::STRING)

#undef MAKE_TYPED_SET
#undef MAKE_TYPED_SET_INFO

#endif // TYPED_SET_H
