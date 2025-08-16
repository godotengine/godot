/**************************************************************************/
/*  typed_array.hpp                                                       */
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

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

template <typename T>
class TypedArray : public Array {
public:
	_FORCE_INLINE_ void operator=(const Array &p_array) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign an array with a different element type.");
		Array::operator=(p_array);
	}
	_FORCE_INLINE_ TypedArray(const Variant &p_variant) :
			TypedArray(Array(p_variant)) {
	}
	_FORCE_INLINE_ TypedArray(const Array &p_array) {
		set_typed(Variant::OBJECT, T::get_class_static(), Variant());
		if (is_same_typed(p_array)) {
			Array::operator=(p_array);
		} else {
			assign(p_array);
		}
	}
	_FORCE_INLINE_ TypedArray(std::initializer_list<Variant> p_init) :
			TypedArray(Array(p_init)) {}
	_FORCE_INLINE_ TypedArray() {
		set_typed(Variant::OBJECT, T::get_class_static(), Variant());
	}
};

// specialization for the rest of variant types

#define MAKE_TYPED_ARRAY(m_type, m_variant_type)                                                                 \
	template <>                                                                                                  \
	class TypedArray<m_type> : public Array {                                                                    \
	public:                                                                                                      \
		_FORCE_INLINE_ void operator=(const Array &p_array) {                                                    \
			ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign an array with a different element type."); \
			Array::operator=(p_array);                                                                           \
		}                                                                                                        \
		_FORCE_INLINE_ TypedArray(std::initializer_list<Variant> p_init) :                                       \
				Array(Array(p_init), m_variant_type, StringName(), Variant()) {                                  \
		}                                                                                                        \
		_FORCE_INLINE_ TypedArray(const Variant &p_variant) :                                                    \
				TypedArray(Array(p_variant)) {                                                                   \
		}                                                                                                        \
		_FORCE_INLINE_ TypedArray(const Array &p_array) {                                                        \
			set_typed(m_variant_type, StringName(), Variant());                                                  \
			if (is_same_typed(p_array)) {                                                                        \
				Array::operator=(p_array);                                                                       \
			} else {                                                                                             \
				assign(p_array);                                                                                 \
			}                                                                                                    \
		}                                                                                                        \
		_FORCE_INLINE_ TypedArray() {                                                                            \
			set_typed(m_variant_type, StringName(), Variant());                                                  \
		}                                                                                                        \
	};

// All Variant::OBJECT types are intentionally omitted from this list because they are handled by
// the unspecialized TypedArray definition.
MAKE_TYPED_ARRAY(bool, Variant::BOOL)
MAKE_TYPED_ARRAY(uint8_t, Variant::INT)
MAKE_TYPED_ARRAY(int8_t, Variant::INT)
MAKE_TYPED_ARRAY(uint16_t, Variant::INT)
MAKE_TYPED_ARRAY(int16_t, Variant::INT)
MAKE_TYPED_ARRAY(uint32_t, Variant::INT)
MAKE_TYPED_ARRAY(int32_t, Variant::INT)
MAKE_TYPED_ARRAY(uint64_t, Variant::INT)
MAKE_TYPED_ARRAY(int64_t, Variant::INT)
MAKE_TYPED_ARRAY(float, Variant::FLOAT)
MAKE_TYPED_ARRAY(double, Variant::FLOAT)
MAKE_TYPED_ARRAY(String, Variant::STRING)
MAKE_TYPED_ARRAY(Vector2, Variant::VECTOR2)
MAKE_TYPED_ARRAY(Vector2i, Variant::VECTOR2I)
MAKE_TYPED_ARRAY(Rect2, Variant::RECT2)
MAKE_TYPED_ARRAY(Rect2i, Variant::RECT2I)
MAKE_TYPED_ARRAY(Vector3, Variant::VECTOR3)
MAKE_TYPED_ARRAY(Vector3i, Variant::VECTOR3I)
MAKE_TYPED_ARRAY(Transform2D, Variant::TRANSFORM2D)
MAKE_TYPED_ARRAY(Vector4, Variant::VECTOR4)
MAKE_TYPED_ARRAY(Vector4i, Variant::VECTOR4I)
MAKE_TYPED_ARRAY(Plane, Variant::PLANE)
MAKE_TYPED_ARRAY(Quaternion, Variant::QUATERNION)
MAKE_TYPED_ARRAY(AABB, Variant::AABB)
MAKE_TYPED_ARRAY(Basis, Variant::BASIS)
MAKE_TYPED_ARRAY(Transform3D, Variant::TRANSFORM3D)
MAKE_TYPED_ARRAY(Projection, Variant::PROJECTION)
MAKE_TYPED_ARRAY(Color, Variant::COLOR)
MAKE_TYPED_ARRAY(StringName, Variant::STRING_NAME)
MAKE_TYPED_ARRAY(NodePath, Variant::NODE_PATH)
MAKE_TYPED_ARRAY(RID, Variant::RID)
MAKE_TYPED_ARRAY(Callable, Variant::CALLABLE)
MAKE_TYPED_ARRAY(Signal, Variant::SIGNAL)
MAKE_TYPED_ARRAY(Dictionary, Variant::DICTIONARY)
MAKE_TYPED_ARRAY(Array, Variant::ARRAY)
MAKE_TYPED_ARRAY(PackedByteArray, Variant::PACKED_BYTE_ARRAY)
MAKE_TYPED_ARRAY(PackedInt32Array, Variant::PACKED_INT32_ARRAY)
MAKE_TYPED_ARRAY(PackedInt64Array, Variant::PACKED_INT64_ARRAY)
MAKE_TYPED_ARRAY(PackedFloat32Array, Variant::PACKED_FLOAT32_ARRAY)
MAKE_TYPED_ARRAY(PackedFloat64Array, Variant::PACKED_FLOAT64_ARRAY)
MAKE_TYPED_ARRAY(PackedStringArray, Variant::PACKED_STRING_ARRAY)
MAKE_TYPED_ARRAY(PackedVector2Array, Variant::PACKED_VECTOR2_ARRAY)
MAKE_TYPED_ARRAY(PackedVector3Array, Variant::PACKED_VECTOR3_ARRAY)
MAKE_TYPED_ARRAY(PackedVector4Array, Variant::PACKED_VECTOR4_ARRAY)
MAKE_TYPED_ARRAY(PackedColorArray, Variant::PACKED_COLOR_ARRAY)
// If the IPAddress struct is added to godot-cpp, the following could also be added:
//MAKE_TYPED_ARRAY(IPAddress, Variant::STRING)

#undef MAKE_TYPED_ARRAY

} // namespace godot
