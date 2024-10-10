/**************************************************************************/
/*  test_json_native.h                                                    */
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

#ifndef TEST_JSON_NATIVE_H
#define TEST_JSON_NATIVE_H

#include "core/io/json.h"

namespace TestJSONNative {

bool compare_variants(Variant variant_1, Variant variant_2, int depth = 0) {
	if (depth > 100) {
		return false;
	}
	if (variant_1.get_type() == Variant::RID && variant_2.get_type() == Variant::RID) {
		return true;
	}
	if (variant_1.get_type() == Variant::CALLABLE || variant_2.get_type() == Variant::CALLABLE) {
		return true;
	}

	List<PropertyInfo> variant_1_properties;
	variant_1.get_property_list(&variant_1_properties);
	List<PropertyInfo> variant_2_properties;
	variant_2.get_property_list(&variant_2_properties);

	if (variant_1_properties.size() != variant_2_properties.size()) {
		return false;
	}

	for (List<PropertyInfo>::Element *E = variant_1_properties.front(); E; E = E->next()) {
		String name = E->get().name;
		Variant variant_1_value = variant_1.get(name);
		Variant variant_2_value = variant_2.get(name);

		if (!compare_variants(variant_1_value, variant_2_value, depth + 1)) {
			return false;
		}
	}

	return true;
}

TEST_CASE("[JSON][Native][SceneTree] Conversion between native and JSON formats") {
	for (int variant_i = 0; variant_i < Variant::VARIANT_MAX; variant_i++) {
		Variant::Type type = static_cast<Variant::Type>(variant_i);
		Variant native_data;
		Callable::CallError error;

		if (type == Variant::Type::INT || type == Variant::Type::FLOAT) {
			Variant value = int64_t(INT64_MAX);
			const Variant *args[] = { &value };
			Variant::construct(type, native_data, args, 1, error);
		} else if (type == Variant::Type::OBJECT) {
			Ref<JSON> json = memnew(JSON);
			native_data = json;
		} else if (type == Variant::Type::DICTIONARY) {
			Dictionary dictionary;
			dictionary["key"] = "value";
			native_data = dictionary;
		} else if (type == Variant::Type::ARRAY) {
			Array array;
			array.push_back("element1");
			array.push_back("element2");
			native_data = array;
		} else if (type == Variant::Type::PACKED_BYTE_ARRAY) {
			PackedByteArray packed_array;
			packed_array.push_back(1);
			packed_array.push_back(2);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_INT32_ARRAY) {
			PackedInt32Array packed_array;
			packed_array.push_back(INT32_MIN);
			packed_array.push_back(INT32_MAX);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_INT64_ARRAY) {
			PackedInt64Array packed_array;
			packed_array.push_back(INT64_MIN);
			packed_array.push_back(INT64_MAX);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_FLOAT32_ARRAY) {
			PackedFloat32Array packed_array;
			packed_array.push_back(FLT_MIN);
			packed_array.push_back(FLT_MAX);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_FLOAT64_ARRAY) {
			PackedFloat64Array packed_array;
			packed_array.push_back(DBL_MIN);
			packed_array.push_back(DBL_MAX);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_STRING_ARRAY) {
			PackedStringArray packed_array;
			packed_array.push_back("string1");
			packed_array.push_back("string2");
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_VECTOR2_ARRAY) {
			PackedVector2Array packed_array;
			Vector2 vector(1.0, 2.0);
			packed_array.push_back(vector);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_VECTOR3_ARRAY) {
			PackedVector3Array packed_array;
			Vector3 vector(1.0, 2.0, 3.0);
			packed_array.push_back(vector);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_COLOR_ARRAY) {
			PackedColorArray packed_array;
			Color color(1.0, 1.0, 1.0);
			packed_array.push_back(color);
			native_data = packed_array;
		} else if (type == Variant::Type::PACKED_VECTOR4_ARRAY) {
			PackedVector4Array packed_array;
			Vector4 vector(1.0, 2.0, 3.0, 4.0);
			packed_array.push_back(vector);
			native_data = packed_array;
		} else {
			Variant::construct(type, native_data, nullptr, 0, error);
		}
		Variant json_converted_from_native = JSON::from_native(native_data, true, true);
		Variant variant_native_converted = JSON::to_native(json_converted_from_native, true, true);
		CHECK_MESSAGE(compare_variants(native_data, variant_native_converted),
				vformat("Conversion from native to JSON type %s and back successful. \nNative: %s \nNative Converted: %s \nError: %s\nConversion from native to JSON type %s successful: %s",
						Variant::get_type_name(type),
						native_data,
						variant_native_converted,
						itos(error.error),
						Variant::get_type_name(type),
						json_converted_from_native));
	}
}
} // namespace TestJSONNative

#endif // TEST_JSON_NATIVE_H
