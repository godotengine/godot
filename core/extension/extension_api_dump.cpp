/**************************************************************************/
/*  extension_api_dump.cpp                                                */
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

#include "extension_api_dump.h"

#include "core/config/engine.h"
#include "core/core_constants.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/templates/pair.h"
#include "core/version.h"

#ifdef TOOLS_ENABLED

static String get_builtin_or_variant_type_name(const Variant::Type p_type) {
	if (p_type == Variant::NIL) {
		return "Variant";
	} else {
		return Variant::get_type_name(p_type);
	}
}

static String get_property_info_type_name(const PropertyInfo &p_info) {
	if (p_info.type == Variant::INT && (p_info.hint == PROPERTY_HINT_INT_IS_POINTER)) {
		if (p_info.hint_string.is_empty()) {
			return "void*";
		} else {
			return p_info.hint_string + "*";
		}
	}
	if (p_info.type == Variant::ARRAY && (p_info.hint == PROPERTY_HINT_ARRAY_TYPE)) {
		return String("typedarray::") + p_info.hint_string;
	}
	if (p_info.type == Variant::INT && (p_info.usage & (PROPERTY_USAGE_CLASS_IS_ENUM))) {
		return String("enum::") + String(p_info.class_name);
	}
	if (p_info.type == Variant::INT && (p_info.usage & (PROPERTY_USAGE_CLASS_IS_BITFIELD))) {
		return String("bitfield::") + String(p_info.class_name);
	}
	if (p_info.type == Variant::INT && (p_info.usage & PROPERTY_USAGE_ARRAY)) {
		return "int";
	}
	if (p_info.class_name != StringName()) {
		return p_info.class_name;
	}
	if (p_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
		return p_info.hint_string;
	}
	if (p_info.type == Variant::NIL && (p_info.usage & PROPERTY_USAGE_NIL_IS_VARIANT)) {
		return "Variant";
	}
	if (p_info.type == Variant::NIL) {
		return "void";
	}
	return get_builtin_or_variant_type_name(p_info.type);
}

static String get_type_meta_name(const GodotTypeInfo::Metadata metadata) {
	static const char *argmeta[11] = { "none", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float", "double" };
	return argmeta[metadata];
}

Dictionary GDExtensionAPIDump::generate_extension_api() {
	Dictionary api_dump;

	{
		//header
		Dictionary header;
		header["version_major"] = VERSION_MAJOR;
		header["version_minor"] = VERSION_MINOR;
#if VERSION_PATCH
		header["version_patch"] = VERSION_PATCH;
#else
		header["version_patch"] = 0;
#endif
		header["version_status"] = VERSION_STATUS;
		header["version_build"] = VERSION_BUILD;
		header["version_full_name"] = VERSION_FULL_NAME;

		api_dump["header"] = header;
	}

	const uint32_t vec3_elems = 3;
	const uint32_t vec4_elems = 4;
	const uint32_t ptrsize_32 = 4;
	const uint32_t ptrsize_64 = 8;
	static const char *build_config_name[4] = { "float_32", "float_64", "double_32", "double_64" };

	{
		//type sizes
		constexpr struct {
			Variant::Type type;
			uint32_t size_32_bits_real_float;
			uint32_t size_64_bits_real_float;
			uint32_t size_32_bits_real_double;
			uint32_t size_64_bits_real_double;

			// For compile-time size check.
			constexpr uint32_t operator[](int index) const {
				switch (index) {
#ifndef REAL_T_IS_DOUBLE
					case sizeof(uint32_t):
						return size_32_bits_real_float;
					case sizeof(uint64_t):
						return size_64_bits_real_float;
#else // REAL_T_IS_DOUBLE
					case sizeof(uint32_t):
						return size_32_bits_real_double;
					case sizeof(uint64_t):
						return size_64_bits_real_double;
#endif
				}
				return -1;
			}
		} type_size_array[Variant::VARIANT_MAX + 1] = {
			{ Variant::NIL, 0, 0, 0, 0 },
			{ Variant::BOOL, sizeof(uint8_t), sizeof(uint8_t), sizeof(uint8_t), sizeof(uint8_t) },
			{ Variant::INT, sizeof(int64_t), sizeof(int64_t), sizeof(int64_t), sizeof(int64_t) },
			{ Variant::FLOAT, sizeof(double), sizeof(double), sizeof(double), sizeof(double) },
			{ Variant::STRING, ptrsize_32, ptrsize_64, ptrsize_32, ptrsize_64 },
			{ Variant::VECTOR2, 2 * sizeof(float), 2 * sizeof(float), 2 * sizeof(double), 2 * sizeof(double) },
			{ Variant::VECTOR2I, 2 * sizeof(int32_t), 2 * sizeof(int32_t), 2 * sizeof(int32_t), 2 * sizeof(int32_t) },
			{ Variant::RECT2, 4 * sizeof(float), 4 * sizeof(float), 4 * sizeof(double), 4 * sizeof(double) },
			{ Variant::RECT2I, 4 * sizeof(int32_t), 4 * sizeof(int32_t), 4 * sizeof(int32_t), 4 * sizeof(int32_t) },
			{ Variant::VECTOR3, vec3_elems * sizeof(float), vec3_elems * sizeof(float), vec3_elems * sizeof(double), vec3_elems * sizeof(double) },
			{ Variant::VECTOR3I, 3 * sizeof(int32_t), 3 * sizeof(int32_t), 3 * sizeof(int32_t), 3 * sizeof(int32_t) },
			{ Variant::TRANSFORM2D, 6 * sizeof(float), 6 * sizeof(float), 6 * sizeof(double), 6 * sizeof(double) },
			{ Variant::VECTOR4, 4 * sizeof(float), 4 * sizeof(float), 4 * sizeof(double), 4 * sizeof(double) },
			{ Variant::VECTOR4I, 4 * sizeof(int32_t), 4 * sizeof(int32_t), 4 * sizeof(int32_t), 4 * sizeof(int32_t) },
			{ Variant::PLANE, (vec3_elems + 1) * sizeof(float), (vec3_elems + 1) * sizeof(float), (vec3_elems + 1) * sizeof(double), (vec3_elems + 1) * sizeof(double) },
			{ Variant::QUATERNION, 4 * sizeof(float), 4 * sizeof(float), 4 * sizeof(double), 4 * sizeof(double) },
			{ Variant::AABB, (vec3_elems * 2) * sizeof(float), (vec3_elems * 2) * sizeof(float), (vec3_elems * 2) * sizeof(double), (vec3_elems * 2) * sizeof(double) },
			{ Variant::BASIS, (vec3_elems * 3) * sizeof(float), (vec3_elems * 3) * sizeof(float), (vec3_elems * 3) * sizeof(double), (vec3_elems * 3) * sizeof(double) },
			{ Variant::TRANSFORM3D, (vec3_elems * 4) * sizeof(float), (vec3_elems * 4) * sizeof(float), (vec3_elems * 4) * sizeof(double), (vec3_elems * 4) * sizeof(double) },
			{ Variant::PROJECTION, (vec4_elems * 4) * sizeof(float), (vec4_elems * 4) * sizeof(float), (vec4_elems * 4) * sizeof(double), (vec4_elems * 4) * sizeof(double) },
			{ Variant::COLOR, 4 * sizeof(float), 4 * sizeof(float), 4 * sizeof(float), 4 * sizeof(float) },
			{ Variant::STRING_NAME, ptrsize_32, ptrsize_64, ptrsize_32, ptrsize_64 },
			{ Variant::NODE_PATH, ptrsize_32, ptrsize_64, ptrsize_32, ptrsize_64 },
			{ Variant::RID, sizeof(uint64_t), sizeof(uint64_t), sizeof(uint64_t), sizeof(uint64_t) },
			{ Variant::OBJECT, ptrsize_32, ptrsize_64, ptrsize_32, ptrsize_64 },
			{ Variant::CALLABLE, sizeof(Callable), sizeof(Callable), sizeof(Callable), sizeof(Callable) }, // Hardcoded align.
			{ Variant::SIGNAL, sizeof(Signal), sizeof(Signal), sizeof(Signal), sizeof(Signal) }, // Hardcoded align.
			{ Variant::DICTIONARY, ptrsize_32, ptrsize_64, ptrsize_32, ptrsize_64 },
			{ Variant::ARRAY, ptrsize_32, ptrsize_64, ptrsize_32, ptrsize_64 },
			{ Variant::PACKED_BYTE_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_INT32_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_INT64_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_FLOAT32_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_FLOAT64_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_STRING_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_VECTOR2_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_VECTOR3_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::PACKED_COLOR_ARRAY, ptrsize_32 * 2, ptrsize_64 * 2, ptrsize_32 * 2, ptrsize_64 * 2 },
			{ Variant::VARIANT_MAX, sizeof(uint64_t) + sizeof(float) * 4, sizeof(uint64_t) + sizeof(float) * 4, sizeof(uint64_t) + sizeof(double) * 4, sizeof(uint64_t) + sizeof(double) * 4 },
		};

		// Validate sizes at compile time for the current build configuration.
		static_assert(type_size_array[Variant::BOOL][sizeof(void *)] == sizeof(GDExtensionBool), "Size of bool mismatch");
		static_assert(type_size_array[Variant::INT][sizeof(void *)] == sizeof(GDExtensionInt), "Size of int mismatch");
		static_assert(type_size_array[Variant::FLOAT][sizeof(void *)] == sizeof(double), "Size of float mismatch");
		static_assert(type_size_array[Variant::STRING][sizeof(void *)] == sizeof(String), "Size of String mismatch");
		static_assert(type_size_array[Variant::VECTOR2][sizeof(void *)] == sizeof(Vector2), "Size of Vector2 mismatch");
		static_assert(type_size_array[Variant::VECTOR2I][sizeof(void *)] == sizeof(Vector2i), "Size of Vector2i mismatch");
		static_assert(type_size_array[Variant::RECT2][sizeof(void *)] == sizeof(Rect2), "Size of Rect2 mismatch");
		static_assert(type_size_array[Variant::RECT2I][sizeof(void *)] == sizeof(Rect2i), "Size of Rect2i mismatch");
		static_assert(type_size_array[Variant::VECTOR3][sizeof(void *)] == sizeof(Vector3), "Size of Vector3 mismatch");
		static_assert(type_size_array[Variant::VECTOR3I][sizeof(void *)] == sizeof(Vector3i), "Size of Vector3i mismatch");
		static_assert(type_size_array[Variant::TRANSFORM2D][sizeof(void *)] == sizeof(Transform2D), "Size of Transform2D mismatch");
		static_assert(type_size_array[Variant::VECTOR4][sizeof(void *)] == sizeof(Vector4), "Size of Vector4 mismatch");
		static_assert(type_size_array[Variant::VECTOR4I][sizeof(void *)] == sizeof(Vector4i), "Size of Vector4i mismatch");
		static_assert(type_size_array[Variant::PLANE][sizeof(void *)] == sizeof(Plane), "Size of Plane mismatch");
		static_assert(type_size_array[Variant::QUATERNION][sizeof(void *)] == sizeof(Quaternion), "Size of Quaternion mismatch");
		static_assert(type_size_array[Variant::AABB][sizeof(void *)] == sizeof(AABB), "Size of AABB mismatch");
		static_assert(type_size_array[Variant::BASIS][sizeof(void *)] == sizeof(Basis), "Size of Basis mismatch");
		static_assert(type_size_array[Variant::TRANSFORM3D][sizeof(void *)] == sizeof(Transform3D), "Size of Transform3D mismatch");
		static_assert(type_size_array[Variant::PROJECTION][sizeof(void *)] == sizeof(Projection), "Size of Projection mismatch");
		static_assert(type_size_array[Variant::COLOR][sizeof(void *)] == sizeof(Color), "Size of Color mismatch");
		static_assert(type_size_array[Variant::STRING_NAME][sizeof(void *)] == sizeof(StringName), "Size of StringName mismatch");
		static_assert(type_size_array[Variant::NODE_PATH][sizeof(void *)] == sizeof(NodePath), "Size of NodePath mismatch");
		static_assert(type_size_array[Variant::RID][sizeof(void *)] == sizeof(RID), "Size of RID mismatch");
		static_assert(type_size_array[Variant::OBJECT][sizeof(void *)] == sizeof(Object *), "Size of Object mismatch");
		static_assert(type_size_array[Variant::CALLABLE][sizeof(void *)] == sizeof(Callable), "Size of Callable mismatch");
		static_assert(type_size_array[Variant::SIGNAL][sizeof(void *)] == sizeof(Signal), "Size of Signal mismatch");
		static_assert(type_size_array[Variant::DICTIONARY][sizeof(void *)] == sizeof(Dictionary), "Size of Dictionary mismatch");
		static_assert(type_size_array[Variant::ARRAY][sizeof(void *)] == sizeof(Array), "Size of Array mismatch");
		static_assert(type_size_array[Variant::PACKED_BYTE_ARRAY][sizeof(void *)] == sizeof(PackedByteArray), "Size of PackedByteArray mismatch");
		static_assert(type_size_array[Variant::PACKED_INT32_ARRAY][sizeof(void *)] == sizeof(PackedInt32Array), "Size of PackedInt32Array mismatch");
		static_assert(type_size_array[Variant::PACKED_INT64_ARRAY][sizeof(void *)] == sizeof(PackedInt64Array), "Size of PackedInt64Array mismatch");
		static_assert(type_size_array[Variant::PACKED_FLOAT32_ARRAY][sizeof(void *)] == sizeof(PackedFloat32Array), "Size of PackedFloat32Array mismatch");
		static_assert(type_size_array[Variant::PACKED_FLOAT64_ARRAY][sizeof(void *)] == sizeof(PackedFloat64Array), "Size of PackedFloat64Array mismatch");
		static_assert(type_size_array[Variant::PACKED_STRING_ARRAY][sizeof(void *)] == sizeof(PackedStringArray), "Size of PackedStringArray mismatch");
		static_assert(type_size_array[Variant::PACKED_VECTOR2_ARRAY][sizeof(void *)] == sizeof(PackedVector2Array), "Size of PackedVector2Array mismatch");
		static_assert(type_size_array[Variant::PACKED_VECTOR3_ARRAY][sizeof(void *)] == sizeof(PackedVector3Array), "Size of PackedVector3Array mismatch");
		static_assert(type_size_array[Variant::PACKED_COLOR_ARRAY][sizeof(void *)] == sizeof(PackedColorArray), "Size of PackedColorArray mismatch");
		static_assert(type_size_array[Variant::VARIANT_MAX][sizeof(void *)] == sizeof(Variant), "Size of Variant mismatch");

		Array core_type_sizes;

		for (int i = 0; i < 4; i++) {
			Dictionary d;
			d["build_configuration"] = build_config_name[i];
			Array sizes;
			for (int j = 0; j <= Variant::VARIANT_MAX; j++) {
				Variant::Type t = type_size_array[j].type;
				String name = t == Variant::VARIANT_MAX ? String("Variant") : Variant::get_type_name(t);
				Dictionary d2;
				d2["name"] = name;
				uint32_t size = 0;
				switch (i) {
					case 0:
						size = type_size_array[j].size_32_bits_real_float;
						break;
					case 1:
						size = type_size_array[j].size_64_bits_real_float;
						break;
					case 2:
						size = type_size_array[j].size_32_bits_real_double;
						break;
					case 3:
						size = type_size_array[j].size_64_bits_real_double;
						break;
				}
				d2["size"] = size;
				sizes.push_back(d2);
			}
			d["sizes"] = sizes;
			core_type_sizes.push_back(d);
		}
		api_dump["builtin_class_sizes"] = core_type_sizes;
	}

	{
		// Member offsets, meta types and sizes.

#define REAL_MEMBER_OFFSET(type, member) \
	{                                    \
		type,                            \
				member,                  \
				"float",                 \
				sizeof(float),           \
				"float",                 \
				sizeof(float),           \
				"double",                \
				sizeof(double),          \
				"double",                \
				sizeof(double),          \
	}

#define INT32_MEMBER_OFFSET(type, member) \
	{                                     \
		type,                             \
				member,                   \
				"int32",                  \
				sizeof(int32_t),          \
				"int32",                  \
				sizeof(int32_t),          \
				"int32",                  \
				sizeof(int32_t),          \
				"int32",                  \
				sizeof(int32_t),          \
	}

#define INT32_BASED_BUILTIN_MEMBER_OFFSET(type, member, member_type, member_elems) \
	{                                                                              \
		type,                                                                      \
				member,                                                            \
				member_type,                                                       \
				sizeof(int32_t) * member_elems,                                    \
				member_type,                                                       \
				sizeof(int32_t) * member_elems,                                    \
				member_type,                                                       \
				sizeof(int32_t) * member_elems,                                    \
				member_type,                                                       \
				sizeof(int32_t) * member_elems,                                    \
	}

#define REAL_BASED_BUILTIN_MEMBER_OFFSET(type, member, member_type, member_elems) \
	{                                                                             \
		type,                                                                     \
				member,                                                           \
				member_type,                                                      \
				sizeof(float) * member_elems,                                     \
				member_type,                                                      \
				sizeof(float) * member_elems,                                     \
				member_type,                                                      \
				sizeof(double) * member_elems,                                    \
				member_type,                                                      \
				sizeof(double) * member_elems,                                    \
	}

		struct {
			Variant::Type type;
			const char *member;
			const char *member_meta_32_bits_real_float;
			const uint32_t member_size_32_bits_real_float;
			const char *member_meta_64_bits_real_float;
			const uint32_t member_size_64_bits_real_float;
			const char *member_meta_32_bits_real_double;
			const uint32_t member_size_32_bits_real_double;
			const char *member_meta_64_bits_real_double;
			const uint32_t member_size_64_bits_real_double;
		} member_offset_array[] = {
			// Vector2
			REAL_MEMBER_OFFSET(Variant::VECTOR2, "x"),
			REAL_MEMBER_OFFSET(Variant::VECTOR2, "y"),
			// Vector2i
			INT32_MEMBER_OFFSET(Variant::VECTOR2I, "x"),
			INT32_MEMBER_OFFSET(Variant::VECTOR2I, "y"),
			// Rect2
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::RECT2, "position", "Vector2", 2),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::RECT2, "size", "Vector2", 2),
			// Rect2i
			INT32_BASED_BUILTIN_MEMBER_OFFSET(Variant::RECT2I, "position", "Vector2i", 2),
			INT32_BASED_BUILTIN_MEMBER_OFFSET(Variant::RECT2I, "size", "Vector2i", 2),
			// Vector3
			REAL_MEMBER_OFFSET(Variant::VECTOR3, "x"),
			REAL_MEMBER_OFFSET(Variant::VECTOR3, "y"),
			REAL_MEMBER_OFFSET(Variant::VECTOR3, "z"),
			// Vector3i
			INT32_MEMBER_OFFSET(Variant::VECTOR3I, "x"),
			INT32_MEMBER_OFFSET(Variant::VECTOR3I, "y"),
			INT32_MEMBER_OFFSET(Variant::VECTOR3I, "z"),
			// Transform2D
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::TRANSFORM2D, "x", "Vector2", 2),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::TRANSFORM2D, "y", "Vector2", 2),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::TRANSFORM2D, "origin", "Vector2", 2),
			// Vector4
			REAL_MEMBER_OFFSET(Variant::VECTOR4, "x"),
			REAL_MEMBER_OFFSET(Variant::VECTOR4, "y"),
			REAL_MEMBER_OFFSET(Variant::VECTOR4, "z"),
			REAL_MEMBER_OFFSET(Variant::VECTOR4, "w"),
			// Vector4i
			INT32_MEMBER_OFFSET(Variant::VECTOR4I, "x"),
			INT32_MEMBER_OFFSET(Variant::VECTOR4I, "y"),
			INT32_MEMBER_OFFSET(Variant::VECTOR4I, "z"),
			INT32_MEMBER_OFFSET(Variant::VECTOR4I, "w"),
			// Plane
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::PLANE, "normal", "Vector3", vec3_elems),
			REAL_MEMBER_OFFSET(Variant::PLANE, "d"),
			// Quaternion
			REAL_MEMBER_OFFSET(Variant::QUATERNION, "x"),
			REAL_MEMBER_OFFSET(Variant::QUATERNION, "y"),
			REAL_MEMBER_OFFSET(Variant::QUATERNION, "z"),
			REAL_MEMBER_OFFSET(Variant::QUATERNION, "w"),
			// AABB
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::AABB, "position", "Vector3", vec3_elems),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::AABB, "size", "Vector3", vec3_elems),
			// Basis (remember that basis vectors are flipped!)
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::BASIS, "x", "Vector3", vec3_elems),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::BASIS, "y", "Vector3", vec3_elems),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::BASIS, "z", "Vector3", vec3_elems),
			// Transform3D
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::TRANSFORM3D, "basis", "Basis", vec3_elems * 3),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::TRANSFORM3D, "origin", "Vector3", vec3_elems),
			// Projection
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::PROJECTION, "x", "Vector4", vec4_elems),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::PROJECTION, "y", "Vector4", vec4_elems),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::PROJECTION, "z", "Vector4", vec4_elems),
			REAL_BASED_BUILTIN_MEMBER_OFFSET(Variant::PROJECTION, "w", "Vector4", vec4_elems),
			// Color (always composed of 4bytes floats)
			{ Variant::COLOR, "r", "float", sizeof(float), "float", sizeof(float), "float", sizeof(float), "float", sizeof(float) },
			{ Variant::COLOR, "g", "float", sizeof(float), "float", sizeof(float), "float", sizeof(float), "float", sizeof(float) },
			{ Variant::COLOR, "b", "float", sizeof(float), "float", sizeof(float), "float", sizeof(float), "float", sizeof(float) },
			{ Variant::COLOR, "a", "float", sizeof(float), "float", sizeof(float), "float", sizeof(float), "float", sizeof(float) },
			// End marker, must stay last
			{ Variant::NIL, nullptr, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0 },
		};

		Array core_type_member_offsets;

		for (int i = 0; i < 4; i++) {
			Dictionary d;
			d["build_configuration"] = build_config_name[i];
			Array type_offsets;
			uint32_t idx = 0;

			Variant::Type previous_type = Variant::NIL;

			Dictionary d2;
			Array members;
			uint32_t offset = 0;

			while (true) {
				Variant::Type t = member_offset_array[idx].type;
				if (t != previous_type) {
					if (previous_type != Variant::NIL) {
						d2["members"] = members;
						type_offsets.push_back(d2);
					}
					if (t == Variant::NIL) {
						break;
					}

					String name = t == Variant::VARIANT_MAX ? String("Variant") : Variant::get_type_name(t);
					d2 = Dictionary();
					members = Array();
					offset = 0;
					d2["name"] = name;
					previous_type = t;
				}
				Dictionary d3;
				const char *member_meta = nullptr;
				uint32_t member_size = 0;
				switch (i) {
					case 0:
						member_meta = member_offset_array[idx].member_meta_32_bits_real_float;
						member_size = member_offset_array[idx].member_size_32_bits_real_float;
						break;
					case 1:
						member_meta = member_offset_array[idx].member_meta_64_bits_real_float;
						member_size = member_offset_array[idx].member_size_64_bits_real_float;
						break;
					case 2:
						member_meta = member_offset_array[idx].member_meta_32_bits_real_double;
						member_size = member_offset_array[idx].member_size_32_bits_real_double;
						break;
					case 3:
						member_meta = member_offset_array[idx].member_meta_64_bits_real_double;
						member_size = member_offset_array[idx].member_size_64_bits_real_double;
						break;
				}
				d3["member"] = member_offset_array[idx].member;
				d3["offset"] = offset;
				d3["meta"] = member_meta;
				offset += member_size;
				members.push_back(d3);
				idx++;
			}
			d["classes"] = type_offsets;
			core_type_member_offsets.push_back(d);
		}
		api_dump["builtin_class_member_offsets"] = core_type_member_offsets;
	}

	{
		// Global enums and constants.
		Array constants;
		HashMap<String, List<Pair<String, int64_t>>> enum_list;
		HashMap<String, bool> enum_is_bitfield;

		for (int i = 0; i < CoreConstants::get_global_constant_count(); i++) {
			int64_t value = CoreConstants::get_global_constant_value(i);
			String enum_name = CoreConstants::get_global_constant_enum(i);
			String name = CoreConstants::get_global_constant_name(i);
			bool bitfield = CoreConstants::is_global_constant_bitfield(i);
			if (!enum_name.is_empty()) {
				enum_list[enum_name].push_back(Pair<String, int64_t>(name, value));
				enum_is_bitfield[enum_name] = bitfield;
			} else {
				Dictionary d;
				d["name"] = name;
				d["value"] = value;
				d["is_bitfield"] = bitfield;
				constants.push_back(d);
			}
		}

		api_dump["global_constants"] = constants;

		Array enums;
		for (const KeyValue<String, List<Pair<String, int64_t>>> &E : enum_list) {
			Dictionary d1;
			d1["name"] = E.key;
			d1["is_bitfield"] = enum_is_bitfield[E.key];
			Array values;
			for (const Pair<String, int64_t> &F : E.value) {
				Dictionary d2;
				d2["name"] = F.first;
				d2["value"] = F.second;
				values.push_back(d2);
			}
			d1["values"] = values;
			enums.push_back(d1);
		}

		api_dump["global_enums"] = enums;
	}
	{
		Array utility_funcs;

		List<StringName> utility_func_names;
		Variant::get_utility_function_list(&utility_func_names);

		for (const StringName &name : utility_func_names) {
			Dictionary func;
			func["name"] = String(name);
			if (Variant::has_utility_function_return_value(name)) {
				Variant::Type rt = Variant::get_utility_function_return_type(name);
				func["return_type"] = rt == Variant::NIL ? String("Variant") : Variant::get_type_name(rt);
			}
			switch (Variant::get_utility_function_type(name)) {
				case Variant::UTILITY_FUNC_TYPE_MATH:
					func["category"] = "math";
					break;
				case Variant::UTILITY_FUNC_TYPE_RANDOM:
					func["category"] = "random";
					break;
				case Variant::UTILITY_FUNC_TYPE_GENERAL:
					func["category"] = "general";
					break;
			}
			bool vararg = Variant::is_utility_function_vararg(name);
			func["is_vararg"] = Variant::is_utility_function_vararg(name);
			func["hash"] = Variant::get_utility_function_hash(name);
			Array arguments;
			int argcount = Variant::get_utility_function_argument_count(name);
			for (int i = 0; i < argcount; i++) {
				Dictionary arg;
				String argname = vararg ? "arg" + itos(i + 1) : Variant::get_utility_function_argument_name(name, i);
				arg["name"] = argname;
				arg["type"] = get_builtin_or_variant_type_name(Variant::get_utility_function_argument_type(name, i));
				//no default value support in utility functions
				arguments.push_back(arg);
			}

			if (arguments.size()) {
				func["arguments"] = arguments;
			}

			utility_funcs.push_back(func);
		}

		api_dump["utility_functions"] = utility_funcs;
	}

	{
		// builtin types

		Array builtins;

		for (int i = 0; i < Variant::VARIANT_MAX; i++) {
			if (i == Variant::OBJECT) {
				continue;
			}

			Variant::Type type = Variant::Type(i);

			Dictionary d;
			d["name"] = Variant::get_type_name(type);
			if (Variant::has_indexing(type)) {
				d["indexing_return_type"] = get_builtin_or_variant_type_name(Variant::get_indexed_element_type(type));
			}

			d["is_keyed"] = Variant::is_keyed(type);

			{
				//members
				Array members;

				List<StringName> member_names;
				Variant::get_member_list(type, &member_names);
				for (const StringName &member_name : member_names) {
					Dictionary d2;
					d2["name"] = String(member_name);
					d2["type"] = get_builtin_or_variant_type_name(Variant::get_member_type(type, member_name));
					members.push_back(d2);
				}
				if (members.size()) {
					d["members"] = members;
				}
			}
			{
				//constants
				Array constants;

				List<StringName> constant_names;
				Variant::get_constants_for_type(type, &constant_names);
				for (const StringName &constant_name : constant_names) {
					Dictionary d2;
					d2["name"] = String(constant_name);
					Variant constant = Variant::get_constant_value(type, constant_name);
					d2["type"] = get_builtin_or_variant_type_name(constant.get_type());
					d2["value"] = constant.get_construct_string();
					constants.push_back(d2);
				}
				if (constants.size()) {
					d["constants"] = constants;
				}
			}
			{
				//enums
				Array enums;

				List<StringName> enum_names;
				Variant::get_enums_for_type(type, &enum_names);
				for (const StringName &enum_name : enum_names) {
					Dictionary enum_dict;
					enum_dict["name"] = String(enum_name);

					List<StringName> enumeration_names;
					Variant::get_enumerations_for_enum(type, enum_name, &enumeration_names);

					Array values;

					for (const StringName &enumeration : enumeration_names) {
						Dictionary values_dict;
						values_dict["name"] = String(enumeration);
						values_dict["value"] = Variant::get_enum_value(type, enum_name, enumeration);
						values.push_back(values_dict);
					}

					if (values.size()) {
						enum_dict["values"] = values;
					}
					enums.push_back(enum_dict);
				}

				if (enums.size()) {
					d["enums"] = enums;
				}
			}
			{
				//operators
				Array operators;

				for (int j = 0; j < Variant::VARIANT_MAX; j++) {
					for (int k = 0; k < Variant::OP_MAX; k++) {
						Variant::Type rt = Variant::get_operator_return_type(Variant::Operator(k), type, Variant::Type(j));
						if (rt != Variant::NIL) {
							Dictionary d2;
							d2["name"] = Variant::get_operator_name(Variant::Operator(k));
							if (k != Variant::OP_NEGATE && k != Variant::OP_POSITIVE && k != Variant::OP_NOT && k != Variant::OP_BIT_NEGATE) {
								d2["right_type"] = get_builtin_or_variant_type_name(Variant::Type(j));
							}
							d2["return_type"] = get_builtin_or_variant_type_name(Variant::get_operator_return_type(Variant::Operator(k), type, Variant::Type(j)));
							operators.push_back(d2);
						}
					}
				}
				if (operators.size()) {
					d["operators"] = operators;
				}
			}
			{
				//methods
				Array methods;

				List<StringName> method_names;
				Variant::get_builtin_method_list(type, &method_names);
				for (const StringName &method_name : method_names) {
					Dictionary d2;
					d2["name"] = String(method_name);
					if (Variant::has_builtin_method_return_value(type, method_name)) {
						Variant::Type ret_type = Variant::get_builtin_method_return_type(type, method_name);
						d2["return_type"] = ret_type == Variant::NIL ? String("Variant") : Variant::get_type_name(ret_type);
					}
					d2["is_vararg"] = Variant::is_builtin_method_vararg(type, method_name);
					d2["is_const"] = Variant::is_builtin_method_const(type, method_name);
					d2["is_static"] = Variant::is_builtin_method_static(type, method_name);
					d2["hash"] = Variant::get_builtin_method_hash(type, method_name);

					Vector<Variant> default_args = Variant::get_builtin_method_default_arguments(type, method_name);

					Array arguments;
					int argcount = Variant::get_builtin_method_argument_count(type, method_name);
					for (int j = 0; j < argcount; j++) {
						Dictionary d3;
						d3["name"] = Variant::get_builtin_method_argument_name(type, method_name, j);
						d3["type"] = get_builtin_or_variant_type_name(Variant::get_builtin_method_argument_type(type, method_name, j));

						if (j >= (argcount - default_args.size())) {
							int dargidx = j - (argcount - default_args.size());
							d3["default_value"] = default_args[dargidx].get_construct_string();
						}
						arguments.push_back(d3);
					}

					if (arguments.size()) {
						d2["arguments"] = arguments;
					}

					methods.push_back(d2);
				}
				if (methods.size()) {
					d["methods"] = methods;
				}
			}
			{
				//constructors
				Array constructors;

				for (int j = 0; j < Variant::get_constructor_count(type); j++) {
					Dictionary d2;
					d2["index"] = j;

					Array arguments;
					int argcount = Variant::get_constructor_argument_count(type, j);
					for (int k = 0; k < argcount; k++) {
						Dictionary d3;
						d3["name"] = Variant::get_constructor_argument_name(type, j, k);
						d3["type"] = get_builtin_or_variant_type_name(Variant::get_constructor_argument_type(type, j, k));
						arguments.push_back(d3);
					}
					if (arguments.size()) {
						d2["arguments"] = arguments;
					}
					constructors.push_back(d2);
				}

				if (constructors.size()) {
					d["constructors"] = constructors;
				}
			}
			{
				//destructor
				d["has_destructor"] = Variant::has_destructor(type);
			}

			builtins.push_back(d);
		}

		api_dump["builtin_classes"] = builtins;
	}

	{
		// classes
		Array classes;

		List<StringName> class_list;

		ClassDB::get_class_list(&class_list);

		class_list.sort_custom<StringName::AlphCompare>();

		for (const StringName &class_name : class_list) {
			Dictionary d;
			d["name"] = String(class_name);
			d["is_refcounted"] = ClassDB::is_parent_class(class_name, "RefCounted");
			d["is_instantiable"] = ClassDB::can_instantiate(class_name);
			StringName parent_class = ClassDB::get_parent_class(class_name);
			if (parent_class != StringName()) {
				d["inherits"] = String(parent_class);
			}

			{
				ClassDB::APIType api = ClassDB::get_api_type(class_name);
				static const char *api_type[5] = { "core", "editor", "extension", "editor_extension" };
				d["api_type"] = api_type[api];
			}

			{
				//constants
				Array constants;
				List<String> constant_list;
				ClassDB::get_integer_constant_list(class_name, &constant_list, true);
				for (const String &F : constant_list) {
					StringName enum_name = ClassDB::get_integer_constant_enum(class_name, F);
					if (enum_name != StringName()) {
						continue; //enums will be handled on their own
					}

					Dictionary d2;
					d2["name"] = String(F);
					d2["value"] = ClassDB::get_integer_constant(class_name, F);

					constants.push_back(d2);
				}

				if (constants.size()) {
					d["constants"] = constants;
				}
			}
			{
				//enum
				Array enums;
				List<StringName> enum_list;
				ClassDB::get_enum_list(class_name, &enum_list, true);
				for (const StringName &F : enum_list) {
					Dictionary d2;
					d2["name"] = String(F);
					d2["is_bitfield"] = ClassDB::is_enum_bitfield(class_name, F);

					Array values;
					List<StringName> enum_constant_list;
					ClassDB::get_enum_constants(class_name, F, &enum_constant_list, true);
					for (List<StringName>::Element *G = enum_constant_list.front(); G; G = G->next()) {
						Dictionary d3;
						d3["name"] = String(G->get());
						d3["value"] = ClassDB::get_integer_constant(class_name, G->get());
						values.push_back(d3);
					}

					d2["values"] = values;

					enums.push_back(d2);
				}

				if (enums.size()) {
					d["enums"] = enums;
				}
			}
			{
				//methods
				Array methods;
				List<MethodInfo> method_list;
				ClassDB::get_method_list(class_name, &method_list, true);
				for (const MethodInfo &F : method_list) {
					StringName method_name = F.name;
					if ((F.flags & METHOD_FLAG_VIRTUAL) && !(F.flags & METHOD_FLAG_OBJECT_CORE)) {
						//virtual method
						const MethodInfo &mi = F;
						Dictionary d2;
						d2["name"] = String(method_name);
						d2["is_const"] = (F.flags & METHOD_FLAG_CONST) ? true : false;
						d2["is_static"] = (F.flags & METHOD_FLAG_STATIC) ? true : false;
						d2["is_vararg"] = false;
						d2["is_virtual"] = true;
						// virtual functions have no hash since no MethodBind is involved
						bool has_return = mi.return_val.type != Variant::NIL || (mi.return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT);
						Array arguments;
						for (int i = (has_return ? -1 : 0); i < mi.arguments.size(); i++) {
							PropertyInfo pinfo = i == -1 ? mi.return_val : mi.arguments[i];
							Dictionary d3;

							if (i >= 0) {
								d3["name"] = pinfo.name;
							}

							d3["type"] = get_property_info_type_name(pinfo);

							if (mi.get_argument_meta(i) > 0) {
								d3["meta"] = get_type_meta_name((GodotTypeInfo::Metadata)mi.get_argument_meta(i));
							}

							if (i == -1) {
								d2["return_value"] = d3;
							} else {
								arguments.push_back(d3);
							}
						}

						if (arguments.size()) {
							d2["arguments"] = arguments;
						}

						methods.push_back(d2);

					} else if (F.name.begins_with("_")) {
						//hidden method, ignore

					} else {
						Dictionary d2;
						d2["name"] = String(method_name);

						MethodBind *method = ClassDB::get_method(class_name, method_name);
						if (!method) {
							continue;
						}

						d2["is_const"] = method->is_const();
						d2["is_vararg"] = method->is_vararg();
						d2["is_static"] = method->is_static();
						d2["is_virtual"] = false;
						d2["hash"] = method->get_hash();

						Vector<Variant> default_args = method->get_default_arguments();

						Array arguments;
						for (int i = (method->has_return() ? -1 : 0); i < method->get_argument_count(); i++) {
							PropertyInfo pinfo = i == -1 ? method->get_return_info() : method->get_argument_info(i);
							Dictionary d3;

							if (i >= 0) {
								d3["name"] = pinfo.name;
							}
							d3["type"] = get_property_info_type_name(pinfo);

							if (method->get_argument_meta(i) > 0) {
								d3["meta"] = get_type_meta_name(method->get_argument_meta(i));
							}

							if (i >= 0 && i >= (method->get_argument_count() - default_args.size())) {
								int dargidx = i - (method->get_argument_count() - default_args.size());
								d3["default_value"] = default_args[dargidx].get_construct_string();
							}

							if (i == -1) {
								d2["return_value"] = d3;
							} else {
								arguments.push_back(d3);
							}
						}

						if (arguments.size()) {
							d2["arguments"] = arguments;
						}

						methods.push_back(d2);
					}
				}

				if (methods.size()) {
					d["methods"] = methods;
				}
			}

			{
				//signals
				Array signals;
				List<MethodInfo> signal_list;
				ClassDB::get_signal_list(class_name, &signal_list, true);
				for (const MethodInfo &F : signal_list) {
					StringName signal_name = F.name;
					Dictionary d2;
					d2["name"] = String(signal_name);

					Array arguments;

					for (int i = 0; i < F.arguments.size(); i++) {
						Dictionary d3;
						d3["name"] = F.arguments[i].name;
						d3["type"] = get_property_info_type_name(F.arguments[i]);
						if (F.get_argument_meta(i) > 0) {
							d3["meta"] = get_type_meta_name((GodotTypeInfo::Metadata)F.get_argument_meta(i));
						}
						arguments.push_back(d3);
					}
					if (arguments.size()) {
						d2["arguments"] = arguments;
					}

					signals.push_back(d2);
				}

				if (signals.size()) {
					d["signals"] = signals;
				}
			}
			{
				//properties
				Array properties;
				List<PropertyInfo> property_list;
				ClassDB::get_property_list(class_name, &property_list, true);
				for (const PropertyInfo &F : property_list) {
					if (F.usage & PROPERTY_USAGE_CATEGORY || F.usage & PROPERTY_USAGE_GROUP || F.usage & PROPERTY_USAGE_SUBGROUP || (F.type == Variant::NIL && F.usage & PROPERTY_USAGE_ARRAY)) {
						continue; //not real properties
					}
					if (F.name.begins_with("_")) {
						continue; //hidden property
					}
					if (F.name.find("/") >= 0) {
						// Ignore properties with '/' (slash) in the name. These are only meant for use in the inspector.
						continue;
					}
					StringName property_name = F.name;
					Dictionary d2;
					d2["type"] = get_property_info_type_name(F);
					d2["name"] = String(property_name);
					StringName setter = ClassDB::get_property_setter(class_name, F.name);
					if (!(setter == "")) {
						d2["setter"] = setter;
					}
					StringName getter = ClassDB::get_property_getter(class_name, F.name);
					if (!(getter == "")) {
						d2["getter"] = getter;
					}
					int index = ClassDB::get_property_index(class_name, F.name);
					if (index != -1) {
						d2["index"] = index;
					}
					properties.push_back(d2);
				}

				if (properties.size()) {
					d["properties"] = properties;
				}
			}

			classes.push_back(d);
		}

		api_dump["classes"] = classes;
	}

	{
		// singletons

		Array singletons;
		List<Engine::Singleton> singleton_list;
		Engine::get_singleton()->get_singletons(&singleton_list);

		for (const Engine::Singleton &s : singleton_list) {
			Dictionary d;
			d["name"] = s.name;
			if (s.class_name != StringName()) {
				d["type"] = String(s.class_name);
			} else {
				d["type"] = String(s.ptr->get_class());
			}
			singletons.push_back(d);
		}

		if (singletons.size()) {
			api_dump["singletons"] = singletons;
		}
	}

	{
		Array native_structures;

		List<StringName> native_structs;
		ClassDB::get_native_struct_list(&native_structs);
		native_structs.sort_custom<StringName::AlphCompare>();

		for (const StringName &E : native_structs) {
			String code = ClassDB::get_native_struct_code(E);

			Dictionary d;
			d["name"] = String(E);
			d["format"] = code;

			native_structures.push_back(d);
		}

		api_dump["native_structures"] = native_structures;
	}

	return api_dump;
}

void GDExtensionAPIDump::generate_extension_json_file(const String &p_path) {
	Dictionary api = generate_extension_api();
	Ref<JSON> json;
	json.instantiate();

	String text = json->stringify(api, "\t", false) + "\n";
	Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_MSG(fa.is_null(), vformat("Cannot open file '%s' for writing.", p_path));
	fa->store_string(text);
}

#endif // TOOLS_ENABLED
