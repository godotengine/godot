/**************************************************************************/
/*  type_info.h                                                           */
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

#include "core/templates/simple_type.h"
#include "core/typedefs.h"

#include <type_traits>

namespace GodotTypeInfo {
enum Metadata {
	METADATA_NONE,
	METADATA_INT_IS_INT8,
	METADATA_INT_IS_INT16,
	METADATA_INT_IS_INT32,
	METADATA_INT_IS_INT64,
	METADATA_INT_IS_UINT8,
	METADATA_INT_IS_UINT16,
	METADATA_INT_IS_UINT32,
	METADATA_INT_IS_UINT64,
	METADATA_REAL_IS_FLOAT,
	METADATA_REAL_IS_DOUBLE,
	METADATA_INT_IS_CHAR16,
	METADATA_INT_IS_CHAR32,
};
}

// If the compiler fails because it's trying to instantiate the primary 'GetTypeInfo' template
// instead of one of the specializations, it's most likely because the type 'T' is not supported.
// If 'T' is a class that inherits 'Object', make sure it can see the actual class declaration
// instead of a forward declaration. You can always forward declare 'T' in a header file, and then
// include the actual declaration of 'T' in the source file where 'GetTypeInfo<T>' is instantiated.
template <typename T, typename = void>
struct GetTypeInfo;

template <typename T>
struct GetTypeInfo<T, std::enable_if_t<!std::is_same_v<T, GetSimpleTypeT<T>>>> : GetTypeInfo<GetSimpleTypeT<T>> {};

#define MAKE_TYPE_INFO(m_type, m_var_type)                                            \
	template <>                                                                       \
	struct GetTypeInfo<m_type> {                                                      \
		static const Variant::Type VARIANT_TYPE = m_var_type;                         \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE; \
		static inline PropertyInfo get_class_info() {                                 \
			return PropertyInfo(VARIANT_TYPE, String());                              \
		}                                                                             \
	};

#define MAKE_TYPE_INFO_WITH_META(m_type, m_var_type, m_metadata)    \
	template <>                                                     \
	struct GetTypeInfo<m_type> {                                    \
		static const Variant::Type VARIANT_TYPE = m_var_type;       \
		static const GodotTypeInfo::Metadata METADATA = m_metadata; \
		static inline PropertyInfo get_class_info() {               \
			return PropertyInfo(VARIANT_TYPE, String());            \
		}                                                           \
	};

MAKE_TYPE_INFO(bool, Variant::BOOL)
MAKE_TYPE_INFO_WITH_META(uint8_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_UINT8)
MAKE_TYPE_INFO_WITH_META(int8_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_INT8)
MAKE_TYPE_INFO_WITH_META(uint16_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_UINT16)
MAKE_TYPE_INFO_WITH_META(int16_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_INT16)
MAKE_TYPE_INFO_WITH_META(uint32_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_UINT32)
MAKE_TYPE_INFO_WITH_META(int32_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_INT32)
MAKE_TYPE_INFO_WITH_META(uint64_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_UINT64)
MAKE_TYPE_INFO_WITH_META(int64_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_INT64)
MAKE_TYPE_INFO_WITH_META(char16_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_CHAR16)
MAKE_TYPE_INFO_WITH_META(char32_t, Variant::INT, GodotTypeInfo::METADATA_INT_IS_CHAR32)
MAKE_TYPE_INFO_WITH_META(float, Variant::FLOAT, GodotTypeInfo::METADATA_REAL_IS_FLOAT)
MAKE_TYPE_INFO_WITH_META(double, Variant::FLOAT, GodotTypeInfo::METADATA_REAL_IS_DOUBLE)

MAKE_TYPE_INFO(String, Variant::STRING)
MAKE_TYPE_INFO(Vector2, Variant::VECTOR2)
MAKE_TYPE_INFO(Rect2, Variant::RECT2)
MAKE_TYPE_INFO(Vector3, Variant::VECTOR3)
MAKE_TYPE_INFO(Vector2i, Variant::VECTOR2I)
MAKE_TYPE_INFO(Rect2i, Variant::RECT2I)
MAKE_TYPE_INFO(Vector3i, Variant::VECTOR3I)
MAKE_TYPE_INFO(Vector4, Variant::VECTOR4)
MAKE_TYPE_INFO(Vector4i, Variant::VECTOR4I)
MAKE_TYPE_INFO(Transform2D, Variant::TRANSFORM2D)
MAKE_TYPE_INFO(Plane, Variant::PLANE)
MAKE_TYPE_INFO(Quaternion, Variant::QUATERNION)
MAKE_TYPE_INFO(AABB, Variant::AABB)
MAKE_TYPE_INFO(Basis, Variant::BASIS)
MAKE_TYPE_INFO(Transform3D, Variant::TRANSFORM3D)
MAKE_TYPE_INFO(Projection, Variant::PROJECTION)
MAKE_TYPE_INFO(Color, Variant::COLOR)
MAKE_TYPE_INFO(StringName, Variant::STRING_NAME)
MAKE_TYPE_INFO(NodePath, Variant::NODE_PATH)
MAKE_TYPE_INFO(RID, Variant::RID)
MAKE_TYPE_INFO(Callable, Variant::CALLABLE)
MAKE_TYPE_INFO(Signal, Variant::SIGNAL)
MAKE_TYPE_INFO(Dictionary, Variant::DICTIONARY)
MAKE_TYPE_INFO(Array, Variant::ARRAY)
MAKE_TYPE_INFO(PackedByteArray, Variant::PACKED_BYTE_ARRAY)
MAKE_TYPE_INFO(PackedInt32Array, Variant::PACKED_INT32_ARRAY)
MAKE_TYPE_INFO(PackedInt64Array, Variant::PACKED_INT64_ARRAY)
MAKE_TYPE_INFO(PackedFloat32Array, Variant::PACKED_FLOAT32_ARRAY)
MAKE_TYPE_INFO(PackedFloat64Array, Variant::PACKED_FLOAT64_ARRAY)
MAKE_TYPE_INFO(PackedStringArray, Variant::PACKED_STRING_ARRAY)
MAKE_TYPE_INFO(PackedVector2Array, Variant::PACKED_VECTOR2_ARRAY)
MAKE_TYPE_INFO(PackedVector3Array, Variant::PACKED_VECTOR3_ARRAY)
MAKE_TYPE_INFO(PackedColorArray, Variant::PACKED_COLOR_ARRAY)
MAKE_TYPE_INFO(PackedVector4Array, Variant::PACKED_VECTOR4_ARRAY)

MAKE_TYPE_INFO(IPAddress, Variant::STRING)

//objectID
template <>
struct GetTypeInfo<ObjectID> {
	static const Variant::Type VARIANT_TYPE = Variant::INT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_INT_IS_UINT64;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::INT, String(), PROPERTY_HINT_INT_IS_OBJECTID);
	}
};

//for variant
template <>
struct GetTypeInfo<Variant> {
	static const Variant::Type VARIANT_TYPE = Variant::NIL;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::NIL, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
	}
};

#define MAKE_TEMPLATE_TYPE_INFO(m_template, m_type, m_var_type)                       \
	template <>                                                                       \
	struct GetTypeInfo<m_template<m_type>> {                                          \
		static const Variant::Type VARIANT_TYPE = m_var_type;                         \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE; \
		static inline PropertyInfo get_class_info() {                                 \
			return PropertyInfo(VARIANT_TYPE, String());                              \
		}                                                                             \
	};

MAKE_TEMPLATE_TYPE_INFO(Vector, Variant, Variant::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, RID, Variant::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Plane, Variant::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Face3, Variant::PACKED_VECTOR3_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, StringName, Variant::PACKED_STRING_ARRAY)

template <typename T>
struct GetTypeInfo<T *, std::enable_if_t<std::is_base_of_v<Object, T>>> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(StringName(T::get_class_static()));
	}
};

namespace GodotTypeInfo {
namespace Internal {
inline String enum_qualified_name_to_class_info_name(const String &p_qualified_name) {
	Vector<String> parts = p_qualified_name.split("::", false);
	if (parts.size() <= 2) {
		return String(".").join(parts);
	}
	// Contains namespace. We only want the class and enum names.
	return parts[parts.size() - 2] + "." + parts[parts.size() - 1];
}
} // namespace Internal
} // namespace GodotTypeInfo

#define MAKE_ENUM_TYPE_INFO(m_enum)                                                                                                          \
	template <>                                                                                                                              \
	struct GetTypeInfo<m_enum> {                                                                                                             \
		static const Variant::Type VARIANT_TYPE = Variant::INT;                                                                              \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;                                                        \
		static inline PropertyInfo get_class_info() {                                                                                        \
			return PropertyInfo(Variant::INT, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_ENUM, \
					GodotTypeInfo::Internal::enum_qualified_name_to_class_info_name(String(#m_enum)));                                       \
		}                                                                                                                                    \
	};

template <typename T>
inline StringName __constant_get_enum_name(T param, const String &p_constant) {
	return GetTypeInfo<T>::get_class_info().class_name;
}

#define MAKE_BITFIELD_TYPE_INFO(m_enum)                                                                                                          \
	template <>                                                                                                                                  \
	struct GetTypeInfo<m_enum> {                                                                                                                 \
		static const Variant::Type VARIANT_TYPE = Variant::INT;                                                                                  \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;                                                            \
		static inline PropertyInfo get_class_info() {                                                                                            \
			return PropertyInfo(Variant::INT, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_BITFIELD, \
					GodotTypeInfo::Internal::enum_qualified_name_to_class_info_name(String(#m_enum)));                                           \
		}                                                                                                                                        \
	};                                                                                                                                           \
	template <>                                                                                                                                  \
	struct GetTypeInfo<BitField<m_enum>> {                                                                                                       \
		static const Variant::Type VARIANT_TYPE = Variant::INT;                                                                                  \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;                                                            \
		static inline PropertyInfo get_class_info() {                                                                                            \
			return PropertyInfo(Variant::INT, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_BITFIELD, \
					GodotTypeInfo::Internal::enum_qualified_name_to_class_info_name(String(#m_enum)));                                           \
		}                                                                                                                                        \
	};

template <typename T>
inline StringName __constant_get_bitfield_name(T param, const String &p_constant) {
	return GetTypeInfo<BitField<T>>::get_class_info().class_name;
}
#define CLASS_INFO(m_type) (GetTypeInfo<m_type *>::get_class_info())

// No initialization by default, except for scalar types.
template <typename T>
struct ZeroInitializer {
	static void initialize(T &value) {
		if constexpr (std::is_scalar_v<T>) {
			value = {};
		}
	}
};
