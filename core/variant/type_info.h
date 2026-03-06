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

#include "core/object/object.h"
#include "core/templates/simple_type.h"
#include "core/typedefs.h"
#include "core/variant/variant.h"

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
	METADATA_OBJECT_IS_REQUIRED,
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

#define MAKE_TYPE_INFO(m_type, m_var_type) \
	template <> \
	struct GetTypeInfo<m_type> { \
		static const VariantType::Type VARIANT_TYPE = m_var_type; \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE; \
		static inline PropertyInfo get_class_info() { \
			return PropertyInfo(VARIANT_TYPE, String()); \
		} \
	};

#define MAKE_TYPE_INFO_WITH_META(m_type, m_var_type, m_metadata) \
	template <> \
	struct GetTypeInfo<m_type> { \
		static const VariantType::Type VARIANT_TYPE = m_var_type; \
		static const GodotTypeInfo::Metadata METADATA = m_metadata; \
		static inline PropertyInfo get_class_info() { \
			return PropertyInfo(VARIANT_TYPE, String()); \
		} \
	};

MAKE_TYPE_INFO(bool, VariantType::BOOL)
MAKE_TYPE_INFO_WITH_META(uint8_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_UINT8)
MAKE_TYPE_INFO_WITH_META(int8_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_INT8)
MAKE_TYPE_INFO_WITH_META(uint16_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_UINT16)
MAKE_TYPE_INFO_WITH_META(int16_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_INT16)
MAKE_TYPE_INFO_WITH_META(uint32_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_UINT32)
MAKE_TYPE_INFO_WITH_META(int32_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_INT32)
MAKE_TYPE_INFO_WITH_META(uint64_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_UINT64)
MAKE_TYPE_INFO_WITH_META(int64_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_INT64)
MAKE_TYPE_INFO_WITH_META(char16_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_CHAR16)
MAKE_TYPE_INFO_WITH_META(char32_t, VariantType::INT, GodotTypeInfo::METADATA_INT_IS_CHAR32)
MAKE_TYPE_INFO_WITH_META(float, VariantType::FLOAT, GodotTypeInfo::METADATA_REAL_IS_FLOAT)
MAKE_TYPE_INFO_WITH_META(double, VariantType::FLOAT, GodotTypeInfo::METADATA_REAL_IS_DOUBLE)

MAKE_TYPE_INFO(String, VariantType::STRING)
MAKE_TYPE_INFO(Vector2, VariantType::VECTOR2)
MAKE_TYPE_INFO(Rect2, VariantType::RECT2)
MAKE_TYPE_INFO(Vector3, VariantType::VECTOR3)
MAKE_TYPE_INFO(Vector2i, VariantType::VECTOR2I)
MAKE_TYPE_INFO(Rect2i, VariantType::RECT2I)
MAKE_TYPE_INFO(Vector3i, VariantType::VECTOR3I)
MAKE_TYPE_INFO(Vector4, VariantType::VECTOR4)
MAKE_TYPE_INFO(Vector4i, VariantType::VECTOR4I)
MAKE_TYPE_INFO(Transform2D, VariantType::TRANSFORM2D)
MAKE_TYPE_INFO(Plane, VariantType::PLANE)
MAKE_TYPE_INFO(Quaternion, VariantType::QUATERNION)
MAKE_TYPE_INFO(AABB, VariantType::AABB)
MAKE_TYPE_INFO(Basis, VariantType::BASIS)
MAKE_TYPE_INFO(Transform3D, VariantType::TRANSFORM3D)
MAKE_TYPE_INFO(Projection, VariantType::PROJECTION)
MAKE_TYPE_INFO(Color, VariantType::COLOR)
MAKE_TYPE_INFO(StringName, VariantType::STRING_NAME)
MAKE_TYPE_INFO(NodePath, VariantType::NODE_PATH)
MAKE_TYPE_INFO(RID, VariantType::RID)
MAKE_TYPE_INFO(Callable, VariantType::CALLABLE)
MAKE_TYPE_INFO(Signal, VariantType::SIGNAL)
MAKE_TYPE_INFO(Dictionary, VariantType::DICTIONARY)
MAKE_TYPE_INFO(Array, VariantType::ARRAY)
MAKE_TYPE_INFO(PackedByteArray, VariantType::PACKED_BYTE_ARRAY)
MAKE_TYPE_INFO(PackedInt32Array, VariantType::PACKED_INT32_ARRAY)
MAKE_TYPE_INFO(PackedInt64Array, VariantType::PACKED_INT64_ARRAY)
MAKE_TYPE_INFO(PackedFloat32Array, VariantType::PACKED_FLOAT32_ARRAY)
MAKE_TYPE_INFO(PackedFloat64Array, VariantType::PACKED_FLOAT64_ARRAY)
MAKE_TYPE_INFO(PackedStringArray, VariantType::PACKED_STRING_ARRAY)
MAKE_TYPE_INFO(PackedVector2Array, VariantType::PACKED_VECTOR2_ARRAY)
MAKE_TYPE_INFO(PackedVector3Array, VariantType::PACKED_VECTOR3_ARRAY)
MAKE_TYPE_INFO(PackedColorArray, VariantType::PACKED_COLOR_ARRAY)
MAKE_TYPE_INFO(PackedVector4Array, VariantType::PACKED_VECTOR4_ARRAY)

MAKE_TYPE_INFO(IPAddress, VariantType::STRING)

//objectID
template <>
struct GetTypeInfo<ObjectID> {
	static const VariantType::Type VARIANT_TYPE = VariantType::INT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_INT_IS_UINT64;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::INT, String(), PROPERTY_HINT_INT_IS_OBJECTID);
	}
};

//for variant
template <>
struct GetTypeInfo<Variant> {
	static const VariantType::Type VARIANT_TYPE = VariantType::NIL;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::NIL, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
	}
};

#define MAKE_TEMPLATE_TYPE_INFO(m_template, m_type, m_var_type) \
	template <> \
	struct GetTypeInfo<m_template<m_type>> { \
		static const VariantType::Type VARIANT_TYPE = m_var_type; \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE; \
		static inline PropertyInfo get_class_info() { \
			return PropertyInfo(VARIANT_TYPE, String()); \
		} \
	};

MAKE_TEMPLATE_TYPE_INFO(Vector, Variant, VariantType::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, RID, VariantType::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Plane, VariantType::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Face3, VariantType::PACKED_VECTOR3_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, StringName, VariantType::PACKED_STRING_ARRAY)

template <typename T>
struct GetTypeInfo<T *, std::enable_if_t<std::is_base_of_v<Object, T>>> {
	static const VariantType::Type VARIANT_TYPE = VariantType::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(StringName(T::get_class_static()));
	}
};

template <typename T>
struct GetTypeInfo<Ref<T>> {
	static const VariantType::Type VARIANT_TYPE = VariantType::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

template <typename T>
class RequiredParam;

template <typename T>
class RequiredResult;

template <typename T>
struct GetTypeInfo<RequiredParam<T>, std::enable_if_t<std::is_base_of_v<Object, T>>> {
	static const VariantType::Type VARIANT_TYPE = VariantType::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_OBJECT_IS_REQUIRED;

	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}

	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(StringName(T::get_class_static()));
	}
};

template <typename T>
struct GetTypeInfo<RequiredResult<T>, std::enable_if_t<std::is_base_of_v<Object, T>>> {
	static const VariantType::Type VARIANT_TYPE = VariantType::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_OBJECT_IS_REQUIRED;

	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}

	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
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

#define MAKE_ENUM_TYPE_INFO(m_enum, m_bound_name) \
	template <> \
	struct GetTypeInfo<m_enum> { \
		static const VariantType::Type VARIANT_TYPE = VariantType::INT; \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE; \
		static inline PropertyInfo get_class_info() { \
			return PropertyInfo(VariantType::INT, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_ENUM, \
					GodotTypeInfo::Internal::enum_qualified_name_to_class_info_name(String(#m_bound_name))); \
		} \
	};

template <typename T>
inline StringName __constant_get_enum_name(T param) {
	return GetTypeInfo<T>::get_class_info().class_name;
}

inline StringName __constant_get_enum_value_name(const char *p_name) {
	return String(p_name).get_slice("::", 1);
}

#define MAKE_BITFIELD_TYPE_INFO(m_enum, m_bound_name) \
	template <> \
	struct GetTypeInfo<m_enum> { \
		static const VariantType::Type VARIANT_TYPE = VariantType::INT; \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE; \
		static inline PropertyInfo get_class_info() { \
			return PropertyInfo(VariantType::INT, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_BITFIELD, \
					GodotTypeInfo::Internal::enum_qualified_name_to_class_info_name(String(#m_bound_name))); \
		} \
	}; \
	template <> \
	struct GetTypeInfo<BitField<m_enum>> { \
		static const VariantType::Type VARIANT_TYPE = VariantType::INT; \
		static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE; \
		static inline PropertyInfo get_class_info() { \
			return PropertyInfo(VariantType::INT, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_BITFIELD, \
					GodotTypeInfo::Internal::enum_qualified_name_to_class_info_name(String(#m_bound_name))); \
		} \
	};

template <typename T>
inline StringName __constant_get_bitfield_name(T param) {
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

namespace GodotTypeInfo {
namespace Internal {

template <typename T>
VariantType::Type get_variant_type() {
	if constexpr (std::is_base_of_v<Object, T>) {
		return VariantType::Type::OBJECT;
	} else {
		return GetTypeInfo<T>::VARIANT_TYPE;
	}
}

template <typename T>
const String get_object_class_name_or_empty() {
	if constexpr (std::is_base_of_v<Object, T>) {
		return T::get_class_static();
	} else {
		return "";
	}
}

template <typename T>
const String get_variant_type_identifier() {
	if constexpr (std::is_base_of_v<Object, T>) {
		return T::get_class_static();
	} else if constexpr (std::is_same_v<Variant, T>) {
		return "Variant";
	} else {
		return VariantType::get_type_name(GetTypeInfo<T>::VARIANT_TYPE);
	}
}

} //namespace Internal
} //namespace GodotTypeInfo

template <typename T>
struct GetTypeInfo<TypedArray<T>> {
	static const VariantType::Type VARIANT_TYPE = VariantType::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, GodotTypeInfo::Internal::get_variant_type_identifier<T>());
	}
};

template <typename K, typename V>
struct GetTypeInfo<TypedDictionary<K, V>> {
	static const VariantType::Type VARIANT_TYPE = VariantType::DICTIONARY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(VariantType::DICTIONARY, String(), PROPERTY_HINT_DICTIONARY_TYPE,
				vformat("%s;%s", GodotTypeInfo::Internal::get_variant_type_identifier<K>(), GodotTypeInfo::Internal::get_variant_type_identifier<V>()));
	}
};
