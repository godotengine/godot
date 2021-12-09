/*************************************************************************/
/*  extension_api_dump.cpp                                               */
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

#include "extension_api_dump.h"
#include "core/config/engine.h"
#include "core/core_constants.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/templates/pair.h"
#include "core/version.h"

#ifdef TOOLS_ENABLED

static String get_type_name(const PropertyInfo &p_info) {
	if (p_info.type == Variant::INT && (p_info.hint == PROPERTY_HINT_INT_IS_POINTER)) {
		if (p_info.hint_string.is_empty()) {
			return "void*";
		} else {
			return p_info.hint_string + "*";
		}
	}
	if (p_info.type == Variant::INT && (p_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM)) {
		return String("enum::") + String(p_info.class_name);
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
	return Variant::get_type_name(p_info.type);
}

Dictionary NativeExtensionAPIDump::generate_extension_api() {
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
			{ Variant::PLANE, (vec3_elems + 1) * sizeof(float), (vec3_elems + 1) * sizeof(float), (vec3_elems + 1) * sizeof(double), (vec3_elems + 1) * sizeof(double) },
			{ Variant::QUATERNION, 4 * sizeof(float), 4 * sizeof(float), 4 * sizeof(double), 4 * sizeof(double) },
			{ Variant::AABB, (vec3_elems * 2) * sizeof(float), (vec3_elems * 2) * sizeof(float), (vec3_elems * 2) * sizeof(double), (vec3_elems * 2) * sizeof(double) },
			{ Variant::BASIS, (vec3_elems * 3) * sizeof(float), (vec3_elems * 3) * sizeof(float), (vec3_elems * 3) * sizeof(double), (vec3_elems * 3) * sizeof(double) },
			{ Variant::TRANSFORM3D, (vec3_elems * 4) * sizeof(float), (vec3_elems * 4) * sizeof(float), (vec3_elems * 4) * sizeof(double), (vec3_elems * 4) * sizeof(double) },
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
		static_assert(type_size_array[Variant::BOOL][sizeof(void *)] == sizeof(GDNativeBool), "Size of bool mismatch");
		static_assert(type_size_array[Variant::INT][sizeof(void *)] == sizeof(GDNativeInt), "Size of int mismatch");
		static_assert(type_size_array[Variant::FLOAT][sizeof(void *)] == sizeof(double), "Size of float mismatch");
		static_assert(type_size_array[Variant::STRING][sizeof(void *)] == sizeof(String), "Size of String mismatch");
		static_assert(type_size_array[Variant::VECTOR2][sizeof(void *)] == sizeof(Vector2), "Size of Vector2 mismatch");
		static_assert(type_size_array[Variant::VECTOR2I][sizeof(void *)] == sizeof(Vector2i), "Size of Vector2i mismatch");
		static_assert(type_size_array[Variant::RECT2][sizeof(void *)] == sizeof(Rect2), "Size of Rect2 mismatch");
		static_assert(type_size_array[Variant::RECT2I][sizeof(void *)] == sizeof(Rect2i), "Size of Rect2i mismatch");
		static_assert(type_size_array[Variant::VECTOR3][sizeof(void *)] == sizeof(Vector3), "Size of Vector3 mismatch");
		static_assert(type_size_array[Variant::VECTOR3I][sizeof(void *)] == sizeof(Vector3i), "Size of Vector3i mismatch");
		static_assert(type_size_array[Variant::TRANSFORM2D][sizeof(void *)] == sizeof(Transform2D), "Size of Transform2D mismatch");
		static_assert(type_size_array[Variant::PLANE][sizeof(void *)] == sizeof(Plane), "Size of Plane mismatch");
		static_assert(type_size_array[Variant::QUATERNION][sizeof(void *)] == sizeof(Quaternion), "Size of Quaternion mismatch");
		static_assert(type_size_array[Variant::AABB][sizeof(void *)] == sizeof(AABB), "Size of AABB mismatch");
		static_assert(type_size_array[Variant::BASIS][sizeof(void *)] == sizeof(Basis), "Size of Basis mismatch");
		static_assert(type_size_array[Variant::TRANSFORM3D][sizeof(void *)] == sizeof(Transform3D), "Size of Transform3D mismatch");
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
				uint32_t size;
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
		// Member offsets sizes.
		struct {
			Variant::Type type;
			const char *member;
			uint32_t offset_32_bits_real_float;
			uint32_t offset_64_bits_real_float;
			uint32_t offset_32_bits_real_double;
			uint32_t offset_64_bits_real_double;
		} member_offset_array[] = {
			{ Variant::VECTOR2, "x", 0, 0, 0, 0 },
			{ Variant::VECTOR2, "y", sizeof(float), sizeof(float), sizeof(double), sizeof(double) },
			{ Variant::VECTOR2I, "x", 0, 0, 0, 0 },
			{ Variant::VECTOR2I, "y", sizeof(int32_t), sizeof(int32_t), sizeof(int32_t), sizeof(int32_t) },
			{ Variant::RECT2, "position", 0, 0, 0, 0 },
			{ Variant::RECT2, "size", 2 * sizeof(Vector2), 2 * sizeof(float), 2 * sizeof(double), 2 * sizeof(double) },
			{ Variant::RECT2I, "position", 0, 0, 0, 0 },
			{ Variant::RECT2I, "size", 2 * sizeof(int32_t), 2 * sizeof(int32_t), 2 * sizeof(int32_t), 2 * sizeof(int32_t) },
			{ Variant::VECTOR3, "x", 0, 0, 0, 0 },
			{ Variant::VECTOR3, "y", sizeof(float), sizeof(float), sizeof(double), sizeof(double) },
			{ Variant::VECTOR3, "z", 2 * sizeof(float), 2 * sizeof(float), 2 * sizeof(double), 2 * sizeof(double) },
			{ Variant::VECTOR3I, "x", 0, 0, 0, 0 },
			{ Variant::VECTOR3I, "y", sizeof(int32_t), sizeof(int32_t), sizeof(int32_t), sizeof(int32_t) },
			{ Variant::VECTOR3I, "z", 2 * sizeof(int32_t), 2 * sizeof(int32_t), 2 * sizeof(int32_t), 2 * sizeof(int32_t) },
			{ Variant::TRANSFORM2D, "x", 0, 0, 0, 0 },
			{ Variant::TRANSFORM2D, "y", 2 * sizeof(float), 2 * sizeof(float), 2 * sizeof(double), 2 * sizeof(double) },
			{ Variant::TRANSFORM2D, "origin", 4 * sizeof(float), 4 * sizeof(float), 4 * sizeof(double), 4 * sizeof(double) },
			{ Variant::PLANE, "normal", 0, 0, 0, 0 },
			{ Variant::PLANE, "d", vec3_elems * sizeof(float), vec3_elems * sizeof(float), vec3_elems * sizeof(double), vec3_elems * sizeof(double) },
			{ Variant::QUATERNION, "x", 0, 0, 0, 0 },
			{ Variant::QUATERNION, "y", sizeof(float), sizeof(float), sizeof(double), sizeof(double) },
			{ Variant::QUATERNION, "z", 2 * sizeof(float), 2 * sizeof(float), 2 * sizeof(double), 2 * sizeof(double) },
			{ Variant::QUATERNION, "w", 3 * sizeof(float), 3 * sizeof(float), 3 * sizeof(double), 3 * sizeof(double) },
			{ Variant::AABB, "position", 0, 0, 0, 0 },
			{ Variant::AABB, "size", vec3_elems * sizeof(float), vec3_elems * sizeof(float), vec3_elems * sizeof(double), vec3_elems * sizeof(double) },
			// Remember that basis vectors are flipped!
			{ Variant::BASIS, "x", 0, 0, 0, 0 },
			{ Variant::BASIS, "y", vec3_elems * sizeof(float), vec3_elems * sizeof(float), vec3_elems * sizeof(double), vec3_elems * sizeof(double) },
			{ Variant::BASIS, "z", vec3_elems * 2 * sizeof(float), vec3_elems * 2 * sizeof(float), vec3_elems * 2 * sizeof(double), vec3_elems * 2 * sizeof(double) },
			{ Variant::TRANSFORM3D, "basis", 0, 0, 0, 0 },
			{ Variant::TRANSFORM3D, "origin", (vec3_elems * 3) * sizeof(float), (vec3_elems * 3) * sizeof(float), (vec3_elems * 3) * sizeof(double), (vec3_elems * 3) * sizeof(double) },
			{ Variant::COLOR, "x", 0, 0, 0, 0 },
			{ Variant::COLOR, "y", sizeof(float), sizeof(float), sizeof(float), sizeof(float) },
			{ Variant::COLOR, "z", 2 * sizeof(float), 2 * sizeof(float), 2 * sizeof(float), 2 * sizeof(float) },
			{ Variant::COLOR, "w", 3 * sizeof(float), 3 * sizeof(float), 3 * sizeof(float), 3 * sizeof(float) },
			{ Variant::NIL, nullptr, 0, 0, 0, 0 },
		};

		Array core_type_member_offsets;

		for (int i = 0; i < 4; i++) {
			Dictionary d;
			d["build_configuration"] = build_config_name[i];
			Array type_offsets;
			uint32_t idx = 0;

			Variant::Type last_type = Variant::NIL;

			Dictionary d2;
			Array members;

			while (true) {
				Variant::Type t = member_offset_array[idx].type;
				if (t != last_type) {
					if (last_type != Variant::NIL) {
						d2["members"] = members;
						type_offsets.push_back(d2);
					}
					if (t == Variant::NIL) {
						break;
					}

					String name = t == Variant::VARIANT_MAX ? String("Variant") : Variant::get_type_name(t);
					d2 = Dictionary();
					members = Array();
					d2["name"] = name;
					last_type = t;
				}
				Dictionary d3;
				uint32_t offset;
				switch (i) {
					case 0:
						offset = member_offset_array[idx].offset_32_bits_real_float;
						break;
					case 1:
						offset = member_offset_array[idx].offset_64_bits_real_float;
						break;
					case 2:
						offset = member_offset_array[idx].offset_32_bits_real_double;
						break;
					case 3:
						offset = member_offset_array[idx].offset_64_bits_real_double;
						break;
				}
				d3["member"] = member_offset_array[idx].member;
				d3["offset"] = offset;
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
		Map<String, List<Pair<String, int>>> enum_list;

		for (int i = 0; i < CoreConstants::get_global_constant_count(); i++) {
			int value = CoreConstants::get_global_constant_value(i);
			String enum_name = CoreConstants::get_global_constant_enum(i);
			String name = CoreConstants::get_global_constant_name(i);
			if (!enum_name.is_empty()) {
				enum_list[enum_name].push_back(Pair<String, int>(name, value));
			} else {
				Dictionary d;
				d["name"] = name;
				d["value"] = value;
				constants.push_back(d);
			}
		}

		api_dump["global_constants"] = constants;

		Array enums;
		for (const KeyValue<String, List<Pair<String, int>>> &E : enum_list) {
			Dictionary d1;
			d1["name"] = E.key;
			Array values;
			for (const Pair<String, int> &F : E.value) {
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
				Variant::Type argtype = Variant::get_utility_function_argument_type(name, i);
				arg["type"] = argtype == Variant::NIL ? String("Variant") : Variant::get_type_name(argtype);
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
				Variant::Type index_type = Variant::get_indexed_element_type(type);
				d["indexing_return_type"] = index_type == Variant::NIL ? String("Variant") : Variant::get_type_name(index_type);
			}

			d["is_keyed"] = Variant::ValidatedKeyedSetter(type);

			{
				//members
				Array members;

				List<StringName> member_names;
				Variant::get_member_list(type, &member_names);
				for (const StringName &member_name : member_names) {
					Dictionary d2;
					d2["name"] = String(member_name);
					d2["type"] = Variant::get_type_name(Variant::get_member_type(type, member_name));
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
					d2["type"] = Variant::get_type_name(constant.get_type());
					d2["value"] = constant.get_construct_string();
					constants.push_back(d2);
				}
				if (constants.size()) {
					d["constants"] = constants;
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
								d2["right_type"] = Variant::get_type_name(Variant::Type(j));
							}
							d2["return_type"] = Variant::get_type_name(Variant::get_operator_return_type(Variant::Operator(k), type, Variant::Type(j)));
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
						Variant::Type argtype = Variant::get_builtin_method_argument_type(type, method_name, j);
						d3["type"] = argtype == Variant::NIL ? String("Variant") : Variant::get_type_name(argtype);

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
						d3["type"] = Variant::get_type_name(Variant::get_constructor_argument_type(type, j, k));
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

							d3["type"] = get_type_name(pinfo);

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
							d3["type"] = get_type_name(pinfo);

							if (method->get_argument_meta(i) > 0) {
								static const char *argmeta[11] = { "none", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float", "double" };
								d3["meta"] = argmeta[method->get_argument_meta(i)];
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
						d3["type"] = get_type_name(F.arguments[i]);
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
					if (F.usage & PROPERTY_USAGE_CATEGORY || F.usage & PROPERTY_USAGE_GROUP || F.usage & PROPERTY_USAGE_SUBGROUP) {
						continue; //not real properties
					}
					if (F.name.begins_with("_")) {
						continue; //hidden property
					}
					StringName property_name = F.name;
					Dictionary d2;
					d2["type"] = get_type_name(F);
					d2["name"] = String(property_name);
					d2["setter"] = ClassDB::get_property_setter(class_name, F.name);
					d2["getter"] = ClassDB::get_property_getter(class_name, F.name);
					d2["index"] = ClassDB::get_property_index(class_name, F.name);
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

		// AudioStream structures
		{
			Dictionary d;
			d["name"] = "AudioFrame";
			d["format"] = "float left,float right";

			native_structures.push_back(d);
		}

		// TextServer structures
		{
			Dictionary d;
			d["name"] = "Glyph";
			d["format"] = "int start,int end,uint8_t count,uint8_t repeat,uint16_t flags,float x_off,float y_off,float advance,RID font_rid,int font_size,int32_t index";

			native_structures.push_back(d);
		}
		{
			Dictionary d;
			d["name"] = "CaretInfo";
			d["format"] = "Rect2 leading_caret,Rect2 trailing_caret,TextServer::Direction leading_direction,TextServer::Direction trailing_direction";

			native_structures.push_back(d);
		}

		api_dump["native_structures"] = native_structures;
	}

	return api_dump;
}

void NativeExtensionAPIDump::generate_extension_json_file(const String &p_path) {
	Dictionary api = generate_extension_api();
	Ref<JSON> json;
	json.instantiate();

	String text = json->stringify(api, "\t", false);
	FileAccessRef fa = FileAccess::open(p_path, FileAccess::WRITE);
	CharString cs = text.ascii();
	fa->store_buffer((const uint8_t *)cs.ptr(), cs.length());
	fa->close();
}
#endif
