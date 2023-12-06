/**************************************************************************/
/*  typed_array.h                                                         */
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

#ifndef TYPED_ARRAY_H
#define TYPED_ARRAY_H

#include "core/object/object.h"
#include "core/variant/array.h"
#include "core/variant/binder_common.h"
#include "core/variant/method_ptrcall.h"
#include "core/variant/type_info.h"
#include "core/variant/variant.h"

template <class T>
class TypedArray : public Array {
public:
	_FORCE_INLINE_ void operator=(const Array &p_array) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign an array with a different element type.");
		_ref(p_array);
	}
	_FORCE_INLINE_ TypedArray(const Variant &p_variant) :
			Array(Array(p_variant), VariantType::OBJECT, T::get_class_static(), Variant()) {
	}
	_FORCE_INLINE_ TypedArray(const Array &p_array) :
			Array(p_array, VariantType::OBJECT, T::get_class_static(), Variant()) {
	}
	_FORCE_INLINE_ TypedArray() {
		set_typed(VariantType::OBJECT, T::get_class_static(), Variant());
	}
};

template <class T>
struct VariantInternalAccessor<TypedArray<T>> {
	static _FORCE_INLINE_ TypedArray<T> get(const Variant *v) { return *VariantInternal::get_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const TypedArray<T> &p_array) { *VariantInternal::get_array(v) = p_array; }
};
template <class T>
struct VariantInternalAccessor<const TypedArray<T> &> {
	static _FORCE_INLINE_ TypedArray<T> get(const Variant *v) { return *VariantInternal::get_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const TypedArray<T> &p_array) { *VariantInternal::get_array(v) = p_array; }
};

//specialization for the rest of variant types

#define MAKE_TYPED_ARRAY(m_type, m_variant_type)                                                                 \
	template <>                                                                                                  \
	class TypedArray<m_type> : public Array {                                                                    \
	public:                                                                                                      \
		_FORCE_INLINE_ void operator=(const Array &p_array) {                                                    \
			ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign an array with a different element type."); \
			_ref(p_array);                                                                                       \
		}                                                                                                        \
		_FORCE_INLINE_ TypedArray(const Variant &p_variant) :                                                    \
				Array(Array(p_variant), m_variant_type, StringName(), Variant()) {                               \
		}                                                                                                        \
		_FORCE_INLINE_ TypedArray(const Array &p_array) :                                                        \
				Array(p_array, m_variant_type, StringName(), Variant()) {                                        \
		}                                                                                                        \
		_FORCE_INLINE_ TypedArray() {                                                                            \
			set_typed(m_variant_type, StringName(), Variant());                                                  \
		}                                                                                                        \
	};

MAKE_TYPED_ARRAY(bool, VariantType::BOOL)
MAKE_TYPED_ARRAY(uint8_t, VariantType::INT)
MAKE_TYPED_ARRAY(int8_t, VariantType::INT)
MAKE_TYPED_ARRAY(uint16_t, VariantType::INT)
MAKE_TYPED_ARRAY(int16_t, VariantType::INT)
MAKE_TYPED_ARRAY(uint32_t, VariantType::INT)
MAKE_TYPED_ARRAY(int32_t, VariantType::INT)
MAKE_TYPED_ARRAY(uint64_t, VariantType::INT)
MAKE_TYPED_ARRAY(int64_t, VariantType::INT)
MAKE_TYPED_ARRAY(float, VariantType::FLOAT)
MAKE_TYPED_ARRAY(double, VariantType::FLOAT)
MAKE_TYPED_ARRAY(String, VariantType::STRING)
MAKE_TYPED_ARRAY(Vector2, VariantType::VECTOR2)
MAKE_TYPED_ARRAY(Vector2i, VariantType::VECTOR2I)
MAKE_TYPED_ARRAY(Rect2, VariantType::RECT2)
MAKE_TYPED_ARRAY(Rect2i, VariantType::RECT2I)
MAKE_TYPED_ARRAY(Vector3, VariantType::VECTOR3)
MAKE_TYPED_ARRAY(Vector3i, VariantType::VECTOR3I)
MAKE_TYPED_ARRAY(Transform2D, VariantType::TRANSFORM2D)
MAKE_TYPED_ARRAY(Plane, VariantType::PLANE)
MAKE_TYPED_ARRAY(Quaternion, VariantType::QUATERNION)
MAKE_TYPED_ARRAY(AABB, VariantType::AABB)
MAKE_TYPED_ARRAY(Basis, VariantType::BASIS)
MAKE_TYPED_ARRAY(Transform3D, VariantType::TRANSFORM3D)
MAKE_TYPED_ARRAY(Color, VariantType::COLOR)
MAKE_TYPED_ARRAY(StringName, VariantType::STRING_NAME)
MAKE_TYPED_ARRAY(NodePath, VariantType::NODE_PATH)
MAKE_TYPED_ARRAY(RID, VariantType::RID)
MAKE_TYPED_ARRAY(Callable, VariantType::CALLABLE)
MAKE_TYPED_ARRAY(Signal, VariantType::SIGNAL)
MAKE_TYPED_ARRAY(Dictionary, VariantType::DICTIONARY)
MAKE_TYPED_ARRAY(Array, VariantType::ARRAY)
MAKE_TYPED_ARRAY(Vector<uint8_t>, VariantType::PACKED_BYTE_ARRAY)
MAKE_TYPED_ARRAY(Vector<int32_t>, VariantType::PACKED_INT32_ARRAY)
MAKE_TYPED_ARRAY(Vector<int64_t>, VariantType::PACKED_INT64_ARRAY)
MAKE_TYPED_ARRAY(Vector<float>, VariantType::PACKED_FLOAT32_ARRAY)
MAKE_TYPED_ARRAY(Vector<double>, VariantType::PACKED_FLOAT64_ARRAY)
MAKE_TYPED_ARRAY(Vector<String>, VariantType::PACKED_STRING_ARRAY)
MAKE_TYPED_ARRAY(Vector<Vector2>, VariantType::PACKED_VECTOR2_ARRAY)
MAKE_TYPED_ARRAY(Vector<Vector3>, VariantType::PACKED_VECTOR3_ARRAY)
MAKE_TYPED_ARRAY(Vector<Color>, VariantType::PACKED_COLOR_ARRAY)
MAKE_TYPED_ARRAY(IPAddress, VariantType::STRING)

template <class T>
struct PtrToArg<TypedArray<T>> {
	_FORCE_INLINE_ static TypedArray<T> convert(const void *p_ptr) {
		return TypedArray<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
	typedef Array EncodeT;
	_FORCE_INLINE_ static void encode(TypedArray<T> p_val, void *p_ptr) {
		*(Array *)p_ptr = p_val;
	}
};

template <class T>
struct PtrToArg<const TypedArray<T> &> {
	typedef Array EncodeT;
	_FORCE_INLINE_ static TypedArray<T> convert(const void *p_ptr) {
		return TypedArray<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
};

template <class T>
struct GetTypeInfo<TypedArray<T>> {
	static const VariantType VARIANT_TYPE = VariantType::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, T::get_class_static());
	}
};

template <class T>
struct GetTypeInfo<const TypedArray<T> &> {
	static const VariantType VARIANT_TYPE = VariantType::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, T::get_class_static());
	}
};

#define MAKE_TYPED_ARRAY_INFO(m_type, m_variant_type)                                                                            \
	template <>                                                                                                                  \
	struct GetTypeInfo<TypedArray<m_type>> {                                                                                     \
		static const VariantType VARIANT_TYPE = VariantType::ARRAY;                                                              \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;                                            \
		static inline PropertyInfo get_class_info() {                                                                            \
			return PropertyInfo(VariantType::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, Variant::get_type_name(m_variant_type)); \
		}                                                                                                                        \
	};                                                                                                                           \
	template <>                                                                                                                  \
	struct GetTypeInfo<const TypedArray<m_type> &> {                                                                             \
		static const VariantType VARIANT_TYPE = VariantType::ARRAY;                                                              \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;                                            \
		static inline PropertyInfo get_class_info() {                                                                            \
			return PropertyInfo(VariantType::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, Variant::get_type_name(m_variant_type)); \
		}                                                                                                                        \
	};

MAKE_TYPED_ARRAY_INFO(bool, VariantType::BOOL)
MAKE_TYPED_ARRAY_INFO(uint8_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(int8_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(uint16_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(int16_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(uint32_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(int32_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(uint64_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(int64_t, VariantType::INT)
MAKE_TYPED_ARRAY_INFO(float, VariantType::FLOAT)
MAKE_TYPED_ARRAY_INFO(double, VariantType::FLOAT)
MAKE_TYPED_ARRAY_INFO(String, VariantType::STRING)
MAKE_TYPED_ARRAY_INFO(Vector2, VariantType::VECTOR2)
MAKE_TYPED_ARRAY_INFO(Vector2i, VariantType::VECTOR2I)
MAKE_TYPED_ARRAY_INFO(Rect2, VariantType::RECT2)
MAKE_TYPED_ARRAY_INFO(Rect2i, VariantType::RECT2I)
MAKE_TYPED_ARRAY_INFO(Vector3, VariantType::VECTOR3)
MAKE_TYPED_ARRAY_INFO(Vector3i, VariantType::VECTOR3I)
MAKE_TYPED_ARRAY_INFO(Transform2D, VariantType::TRANSFORM2D)
MAKE_TYPED_ARRAY_INFO(Plane, VariantType::PLANE)
MAKE_TYPED_ARRAY_INFO(Quaternion, VariantType::QUATERNION)
MAKE_TYPED_ARRAY_INFO(AABB, VariantType::AABB)
MAKE_TYPED_ARRAY_INFO(Basis, VariantType::BASIS)
MAKE_TYPED_ARRAY_INFO(Transform3D, VariantType::TRANSFORM3D)
MAKE_TYPED_ARRAY_INFO(Color, VariantType::COLOR)
MAKE_TYPED_ARRAY_INFO(StringName, VariantType::STRING_NAME)
MAKE_TYPED_ARRAY_INFO(NodePath, VariantType::NODE_PATH)
MAKE_TYPED_ARRAY_INFO(RID, VariantType::RID)
MAKE_TYPED_ARRAY_INFO(Callable, VariantType::CALLABLE)
MAKE_TYPED_ARRAY_INFO(Signal, VariantType::SIGNAL)
MAKE_TYPED_ARRAY_INFO(Dictionary, VariantType::DICTIONARY)
MAKE_TYPED_ARRAY_INFO(Array, VariantType::ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<uint8_t>, VariantType::PACKED_BYTE_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<int32_t>, VariantType::PACKED_INT32_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<int64_t>, VariantType::PACKED_INT64_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<float>, VariantType::PACKED_FLOAT32_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<double>, VariantType::PACKED_FLOAT64_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<String>, VariantType::PACKED_STRING_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<Vector2>, VariantType::PACKED_VECTOR2_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<Vector3>, VariantType::PACKED_VECTOR3_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<Color>, VariantType::PACKED_COLOR_ARRAY)
MAKE_TYPED_ARRAY_INFO(IPAddress, VariantType::STRING)

#undef MAKE_TYPED_ARRAY
#undef MAKE_TYPED_ARRAY_INFO

#endif // TYPED_ARRAY_H
