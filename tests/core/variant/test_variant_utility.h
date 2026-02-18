/**************************************************************************/
/*  test_variant_utility.h                                                */
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

#include "core/variant/variant_utility.h"

#include "tests/test_macros.h"

namespace TestVariantUtility {

TEST_CASE("[VariantUtility] Type conversion") {
	Variant converted;
	converted = VariantUtilityFunctions::type_convert("Hi!", Variant::Type::NIL);
	CHECK(converted.get_type() == Variant::Type::NIL);
	CHECK(converted == Variant());

	converted = VariantUtilityFunctions::type_convert("Hi!", Variant::Type::INT);
	CHECK(converted.get_type() == Variant::Type::INT);
	CHECK(converted == Variant(0));

	converted = VariantUtilityFunctions::type_convert("123", Variant::Type::INT);
	CHECK(converted.get_type() == Variant::Type::INT);
	CHECK(converted == Variant(123));

	converted = VariantUtilityFunctions::type_convert(123, Variant::Type::STRING);
	CHECK(converted.get_type() == Variant::Type::STRING);
	CHECK(converted == Variant("123"));

	converted = VariantUtilityFunctions::type_convert(123.4, Variant::Type::INT);
	CHECK(converted.get_type() == Variant::Type::INT);
	CHECK(converted == Variant(123));

	converted = VariantUtilityFunctions::type_convert(5, Variant::Type::VECTOR2);
	CHECK(converted.get_type() == Variant::Type::VECTOR2);
	CHECK(converted == Variant(Vector2(0, 0)));

	converted = VariantUtilityFunctions::type_convert(Vector3(1, 2, 3), Variant::Type::VECTOR2);
	CHECK(converted.get_type() == Variant::Type::VECTOR2);
	CHECK(converted == Variant(Vector2(1, 2)));

	converted = VariantUtilityFunctions::type_convert(Vector2(1, 2), Variant::Type::VECTOR4);
	CHECK(converted.get_type() == Variant::Type::VECTOR4);
	CHECK(converted == Variant(Vector4(1, 2, 0, 0)));

	converted = VariantUtilityFunctions::type_convert(Vector4(1.2, 3.4, 5.6, 7.8), Variant::Type::VECTOR3I);
	CHECK(converted.get_type() == Variant::Type::VECTOR3I);
	CHECK(converted == Variant(Vector3i(1, 3, 5)));

	{
		Basis basis = Basis::from_scale(Vector3(1.2, 3.4, 5.6));
		Transform3D transform = Transform3D(basis, Vector3());

		converted = VariantUtilityFunctions::type_convert(transform, Variant::Type::BASIS);
		CHECK(converted.get_type() == Variant::Type::BASIS);
		CHECK(converted == basis);

		converted = VariantUtilityFunctions::type_convert(basis, Variant::Type::TRANSFORM3D);
		CHECK(converted.get_type() == Variant::Type::TRANSFORM3D);
		CHECK(converted == transform);

		converted = VariantUtilityFunctions::type_convert(basis, Variant::Type::STRING);
		CHECK(converted.get_type() == Variant::Type::STRING);
		CHECK(converted == Variant("[X: (1.2, 0.0, 0.0), Y: (0.0, 3.4, 0.0), Z: (0.0, 0.0, 5.6)]"));
	}

	{
		Array arr = { 1.2, 3.4, 5.6 };
		PackedFloat64Array packed = { 1.2, 3.4, 5.6 };

		converted = VariantUtilityFunctions::type_convert(arr, Variant::Type::PACKED_FLOAT64_ARRAY);
		CHECK(converted.get_type() == Variant::Type::PACKED_FLOAT64_ARRAY);
		CHECK(converted == packed);

		converted = VariantUtilityFunctions::type_convert(packed, Variant::Type::ARRAY);
		CHECK(converted.get_type() == Variant::Type::ARRAY);
		CHECK(converted == arr);
	}

	{
		// Check that using Variant::call_utility_function also works.
		Vector<const Variant *> args;
		Variant data_arg = "Hi!";
		args.push_back(&data_arg);
		Variant type_arg = Variant::Type::NIL;
		args.push_back(&type_arg);
		Callable::CallError call_error;
		Variant::call_utility_function("type_convert", &converted, (const Variant **)args.ptr(), 2, call_error);
		CHECK(converted.get_type() == Variant::Type::NIL);
		CHECK(converted == Variant());

		type_arg = Variant::Type::INT;
		Variant::call_utility_function("type_convert", &converted, (const Variant **)args.ptr(), 2, call_error);
		CHECK(converted.get_type() == Variant::Type::INT);
		CHECK(converted == Variant(0));

		data_arg = "123";
		Variant::call_utility_function("type_convert", &converted, (const Variant **)args.ptr(), 2, call_error);
		CHECK(converted.get_type() == Variant::Type::INT);
		CHECK(converted == Variant(123));
	}
}

TEST_CASE("[VariantUtility] error_string") {
	CHECK(VariantUtilityFunctions::error_string(OK) == "OK");
	CHECK(VariantUtilityFunctions::error_string(ERR_CANT_OPEN) == "Can't open");
	CHECK(VariantUtilityFunctions::error_string(ERR_PRINTER_ON_FIRE) == "Printer on fire");
	CHECK(VariantUtilityFunctions::error_string(ERR_MAX) == "(invalid error code)");
}

TEST_CASE("[VariantUtility] type_string") {
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::NIL) == "Nil");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::INT) == "int");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::FLOAT) == "float");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::STRING) == "String");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::VECTOR2) == "Vector2");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::VECTOR4I) == "Vector4i");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::COLOR) == "Color");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::ARRAY) == "Array");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::DICTIONARY) == "Dictionary");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::PACKED_BYTE_ARRAY) == "PackedByteArray");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::PACKED_INT32_ARRAY) == "PackedInt32Array");
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::PACKED_FLOAT64_ARRAY) == "PackedFloat64Array");
	ERR_PRINT_OFF;
	CHECK(VariantUtilityFunctions::type_string(Variant::Type::VARIANT_MAX) == "<invalid type>");
	ERR_PRINT_ON;
}

TEST_CASE("[VariantUtility] str_to_var") {
	const Array arr = { 1.2, 3.4, 5.600000000001 };
	Variant converted = VariantUtilityFunctions::str_to_var("[1.2, 3.4, 5.600000000001]");
	CHECK(converted.get_type() == Variant::Type::ARRAY);
	CHECK(converted == arr);

	const PackedFloat64Array packed_arr = { 1.2, 3.4, 5.600000000001 };
	converted = VariantUtilityFunctions::str_to_var("PackedFloat64Array(1.2, 3.4, 5.600000000001)");
	CHECK(converted.get_type() == Variant::Type::PACKED_FLOAT64_ARRAY);
	CHECK(converted == PackedFloat64Array({ 1.2, 3.4, 5.600000000001 }));
}

TEST_CASE("[VariantUtility] var_to_str") {
	const Array arr = { 1.2, 3.4, 5.600000000001 };
	Variant converted = VariantUtilityFunctions::var_to_str(arr);
	CHECK(converted.get_type() == Variant::Type::STRING);
	CHECK(converted == Variant("[1.2, 3.4, 5.600000000001]"));

	const PackedFloat64Array packed_arr = { 1.2, 3.4, 5.600000000001 };
	converted = VariantUtilityFunctions::var_to_str(packed_arr);
	CHECK(converted.get_type() == Variant::Type::STRING);
	CHECK(converted == Variant("PackedFloat64Array(1.2, 3.4, 5.600000000001)"));
}

TEST_CASE("[VariantUtility] var_to_bytes") {
	const Array array = { 1.2, 3.4, 5.600000000001 };
	Variant converted = VariantUtilityFunctions::var_to_bytes(array);
	CHECK(converted.get_type() == Variant::Type::PACKED_BYTE_ARRAY);
	// clang-format off
	CHECK(converted == PackedByteArray({
		0x1c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
		0x03, 0x00, 0x01, 0x00, 0x33, 0x33, 0x33, 0x33,
		0x33, 0x33, 0xf3, 0x3f, 0x03, 0x00, 0x01, 0x00,
		0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x0b, 0x40,
		0x03, 0x00, 0x01, 0x00, 0xcc, 0x6a, 0x66, 0x66,
		0x66, 0x66, 0x16, 0x40,
	}));
	// clang-format on
}

TEST_CASE("[VariantUtility] var_to_bytes_with_objects") {
	Ref<Resource> resource = memnew(Resource);
	resource->set_meta("example", 123.456);
	const Array array = { 42, resource, true };

	Variant converted = VariantUtilityFunctions::var_to_bytes(array);
	CHECK(converted.get_type() == Variant::Type::PACKED_BYTE_ARRAY);
	// clang-format off
	CHECK(converted == PackedByteArray({
		0x1c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
		0x02, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00,
		0x18, 0x00, 0x01, 0x00, 0x54, 0x00, 0x00, 0x58,
		0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,
	}));
	// clang-format on

	converted = VariantUtilityFunctions::var_to_bytes_with_objects(array);
	CHECK(converted.get_type() == Variant::Type::PACKED_BYTE_ARRAY);
	// clang-format off
	CHECK(converted == PackedByteArray({
		0x1c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
		0x02, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00,
		0x18, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
		0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
		0x04, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
		0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
		0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x74,
		0x6f, 0x5f, 0x73, 0x63, 0x65, 0x6e, 0x65, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x0d, 0x00, 0x00, 0x00, 0x72, 0x65, 0x73, 0x6f,
		0x75, 0x72, 0x63, 0x65, 0x5f, 0x6e, 0x61, 0x6d,
		0x65, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
		0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
		0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61,
		0x2f, 0x65, 0x78, 0x61, 0x6d, 0x70, 0x6c, 0x65,
		0x03, 0x00, 0x01, 0x00, 0x77, 0xbe, 0x9f, 0x1a,
		0x2f, 0xdd, 0x5e, 0x40, 0x01, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,
	}));
	// clang-format on
}

TEST_CASE("[VariantUtility] bytes_to_var") {
	// clang-format off
	const PackedByteArray packed_array = {
		0x1c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
		0x03, 0x00, 0x01, 0x00, 0x33, 0x33, 0x33, 0x33,
		0x33, 0x33, 0xf3, 0x3f, 0x03, 0x00, 0x01, 0x00,
		0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x0b, 0x40,
		0x03, 0x00, 0x01, 0x00, 0xcc, 0x6a, 0x66, 0x66,
		0x66, 0x66, 0x16, 0x40,
	};
	// clang-format on
	const Variant converted = VariantUtilityFunctions::bytes_to_var(packed_array);
	CHECK(converted.get_type() == Variant::Type::ARRAY);
	CHECK(converted == Array({ 1.2, 3.4, 5.600000000001 }));
}

TEST_CASE("[VariantUtility] bytes_to_var_with_objects") {
	// clang-format off
	const PackedByteArray packed_byte_array_with_objects = {
		0x1c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
		0x02, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00,
		0x18, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
		0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
		0x04, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
		0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
		0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x74,
		0x6f, 0x5f, 0x73, 0x63, 0x65, 0x6e, 0x65, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x0d, 0x00, 0x00, 0x00, 0x72, 0x65, 0x73, 0x6f,
		0x75, 0x72, 0x63, 0x65, 0x5f, 0x6e, 0x61, 0x6d,
		0x65, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
		0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
		0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61,
		0x2f, 0x65, 0x78, 0x61, 0x6d, 0x70, 0x6c, 0x65,
		0x03, 0x00, 0x01, 0x00, 0x77, 0xbe, 0x9f, 0x1a,
		0x2f, 0xdd, 0x5e, 0x40, 0x01, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,
	};
	// clang-format on

	ERR_PRINT_OFF;
	// Not allowed with `bytes_to_var()`, as it contains a type that derives from Object.
	Variant converted = VariantUtilityFunctions::bytes_to_var(packed_byte_array_with_objects);
	ERR_PRINT_ON;
	CHECK(converted.get_type() == Variant::Type::NIL);

	converted = VariantUtilityFunctions::bytes_to_var_with_objects(packed_byte_array_with_objects);
	CHECK(converted.get_type() == Variant::Type::ARRAY);
	const Array array = converted;
	CHECK(int(array[0]) == 42);
	const Ref<Resource> resource = array[1];
	CHECK(resource.is_valid());
	CHECK(Math::is_equal_approx(double(resource->get_meta("example")), 123.456));
	CHECK(bool(array[2]) == true);
}

TEST_CASE("[VariantUtility] hash") {
	Variant variant;
	CHECK(VariantUtilityFunctions::hash(variant) == 0);

	variant = 42;
	CHECK(VariantUtilityFunctions::hash(variant) == 0x7f576bfb);

	variant = "Hello world!";
	CHECK(VariantUtilityFunctions::hash(variant) == 0xc11f34e2);

	variant = StringName("Hello world!");
	CHECK(VariantUtilityFunctions::hash(variant) == 0xc11f34e2);

	variant = { 1, 2, 3 };
	CHECK(VariantUtilityFunctions::hash(variant) == 0xe61420f0);

	variant = { 1.0, 2.0, 3.0 };
	CHECK(VariantUtilityFunctions::hash(variant) == 0x1f19e9ff);

	variant = PackedInt32Array({ 1, 2, 3 });
	CHECK(VariantUtilityFunctions::hash(variant) == 0x3296a8c7);

	variant = PackedInt64Array({ 1, 2, 3 });
	CHECK(VariantUtilityFunctions::hash(variant) == 0x361c2f77);

	variant = PackedFloat32Array({ 1, 2, 3 });
	CHECK(VariantUtilityFunctions::hash(variant) == 0x9abe67ed);

	variant = PackedFloat64Array({ 1, 2, 3 });
	CHECK(VariantUtilityFunctions::hash(variant) == 0xb7b8f019);
}

TEST_CASE("[VariantUtility] is_same") {
	const Variant variant1 = 42;
	const Variant variant2 = 42;
	const Variant variant3 = "Hello world!";
	const Variant variant4 = StringName("Hello world!");

	CHECK(VariantUtilityFunctions::is_same(variant1, variant2) == true);
	CHECK(VariantUtilityFunctions::is_same(variant1, variant3) == false);
	CHECK(VariantUtilityFunctions::is_same(variant3, variant4) == false);
	CHECK(VariantUtilityFunctions::is_same(variant1, variant4) == false);
}

TEST_CASE("[VariantUtility] join_string") {
	const Variant variant_1 = "Hello ";
	const Variant variant_2 = "world";
	const Variant variant_3 = "!";
	const Variant *variants[3] = { &variant_1, &variant_2, &variant_3 };
	Variant result = VariantUtilityFunctions::join_string(variants, 3);
	CHECK(result.get_type() == Variant::Type::STRING);
	CHECK(result == Variant("Hello world!"));
}

} // namespace TestVariantUtility
