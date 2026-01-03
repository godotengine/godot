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

#include "scene/main/node.h"
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

TEST_CASE("[VariantUtility] deep_equals") {
	SUBCASE("Different types") {
		Variant v_int = 5;
		Variant v_str = "hello";
		Variant v_arr = Array();
		Variant v_dict = Dictionary();
		Variant v_null_obj = Variant();
		Node *node = memnew(Node);

		CHECK_FALSE(VariantUtilityFunctions::deep_equals(v_int, v_str));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(v_str, v_int));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(v_int, v_arr));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(v_arr, v_dict));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(v_dict, node));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(node, v_null_obj));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(v_int, v_null_obj));

		memdelete(node);
	}

	SUBCASE("Basic types") {
		CHECK(VariantUtilityFunctions::deep_equals(Variant(10), Variant(10)));
		CHECK(VariantUtilityFunctions::deep_equals(Variant(3.14), Variant(3.14)));
		CHECK(VariantUtilityFunctions::deep_equals(Variant(true), Variant(true)));
		CHECK(VariantUtilityFunctions::deep_equals(Variant("test"), Variant("test")));
		CHECK(VariantUtilityFunctions::deep_equals(Variant(Vector2(1, 2)), Variant(Vector2(1, 2))));
		CHECK(VariantUtilityFunctions::deep_equals(Variant(), Variant())); // Nil == Nil

		CHECK_FALSE(VariantUtilityFunctions::deep_equals(Variant(10), Variant(11)));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(Variant(3.14), Variant(3.141)));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(Variant(true), Variant(false)));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(Variant("test"), Variant("Test")));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(Variant("test"), Variant("")));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(Variant(Vector2(1, 2)), Variant(Vector2(1, 3))));
		CHECK_FALSE(VariantUtilityFunctions::deep_equals(Variant(5), Variant()));
	}

	SUBCASE("Arrays") {
		Array a1, a2;
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(a1, a2), "Empty arrays should be equal.");

		a1.append(1);
		a1.append("hello");
		a2.append(1);
		a2.append("hello");
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(a1, a2), "Arrays with identical basic elements should be equal.");

		Array a3;
		a3.append(1);
		a3.append("world");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(a1, a3), "Arrays with different element values should not be equal.");

		Array a4;
		a4.append(1);
		a4.append(2);
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(a1, a4), "Arrays with different element types at the same index should not be equal.");

		Array a5;
		a5.append(1);
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(a1, a5), "Arrays with different sizes should not be equal.");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(a5, a1), "Arrays with different sizes should not be equal.");

		Array nested1, nested2, inner1, inner2;
		inner1.append(true);
		inner1.append(2.5);
		inner2.append(true);
		inner2.append(2.5);
		nested1.append("outer");
		nested1.append(inner1);
		nested2.append("outer");
		nested2.append(inner2);
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(nested1, nested2), "Arrays with equal nested arrays should be equal.");

		Array nested3, inner3;
		inner3.append(true);
		inner3.append(3.0);
		nested3.append("outer");
		nested3.append(inner3);
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(nested1, nested3), "Arrays with unequal nested arrays should not be equal.");

		Array ca1, ca2;
		ca1.append(1);
		ca1.append(ca1);
		ca2.append(1);
		ca2.append(ca2);
		ERR_PRINT_OFF
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(ca1, ca2), "Arrays with circular references should be equal as the maximum recursion depth is reached.");
		ERR_PRINT_ON

		ca1.clear();
		ca2.clear();
	}

	SUBCASE("Dictionaries") {
		Dictionary d1, d2;
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(d1, d2), "Empty dictionaries should be equal.");

		d1["key1"] = 100;
		d1["key2"] = "val";
		d2["key1"] = 100;
		d2["key2"] = "val";
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(d1, d2), "Dictionaries with identical key-value pairs should be equal.");

		Dictionary d3;
		d3["key2"] = "val";
		d3["key1"] = 100;
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(d1, d3), "Dictionaries with identical key-value pairs in different order should be equal.");

		Dictionary d4;
		d4["key1"] = 100;
		d4["key2"] = "VAL";
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(d1, d4), "Dictionaries with different values for the same key should not be equal.");

		Dictionary d5;
		d5["key1"] = 100;
		d5["key_other"] = "val";
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(d1, d5), "Dictionaries with different keys should not be equal.");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(d5, d1), "Dictionaries with different keys should not be equal.");

		Dictionary d6;
		d6["key1"] = 100;
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(d1, d6), "Dictionaries with different sizes should not be equal.");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(d6, d1), "Dictionaries with different sizes should not be equal.");

		Dictionary nested_d1, nested_d2, inner_d1, inner_d2;
		inner_d1["sub_key"] = 9.9;
		inner_d2["sub_key"] = 9.9;
		nested_d1["level1"] = inner_d1;
		nested_d1["another"] = false;
		nested_d2["level1"] = inner_d2;
		nested_d2["another"] = false;
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(nested_d1, nested_d2), "Dictionaries with equal nested dictionaries should be equal.");

		Dictionary nested_d3, inner_d3;
		inner_d3["sub_key"] = 10.0;
		nested_d3["level1"] = inner_d3;
		nested_d3["another"] = false;
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(nested_d1, nested_d3), "Dictionaries with unequal nested dictionaries should not be equal.");

		Dictionary cd1, cd2;
		cd1["key"] = 1;
		cd1["self"] = cd1;
		cd2["key"] = 1;
		cd2["self"] = cd2;

		ERR_PRINT_OFF
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(cd1, cd2), "Dictionaries with circular references should be equal as the maximum recursion depth is reached.");
		ERR_PRINT_ON

		cd1.clear();
		cd2.clear();
	}

	SUBCASE("Objects") {
		Variant null_obj1;
		Variant null_obj2;
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(null_obj1, null_obj2), "Two null object Variants should be equal.");

		Node *node1 = memnew(Node);
		Node *node2 = memnew(Node);
		Node *node3 = memnew(Node);

		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(null_obj1, node1), "A null object Variant and a non-null object Variant should not be equal.");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(node1, null_obj1), "A non-null object Variant and a null object Variant should not be equal.");

		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(node1, node2), "Two distinct Node instances with default properties should be equal.");

		node1->set_name("TestNode");
		node2->set_name("TestNode");
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(node1, node2), "Two Node instances with the same property value should be equal.");

		node3->set_name("AnotherNode");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(node1, node3), "Two Node instances with different property values should not be equal.");

		Object *base_obj1_ptr = memnew(Object);
		Object *base_obj2_ptr = memnew(Object);
		Variant v_base_obj1 = base_obj1_ptr;
		Variant v_base_obj2 = base_obj2_ptr;

		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(v_base_obj1, v_base_obj2), "Two distinct base Object instances should be equal if properties match.");

		base_obj1_ptr->set_meta("info", "data1");
		base_obj2_ptr->set_meta("info", "data1");
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(v_base_obj1, v_base_obj2), "Two objects with the same meta values should be equal.");

		base_obj2_ptr->set_meta("info", "data2");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(v_base_obj1, v_base_obj2), "Two objects with different meta values should not be equal.");

		memdelete(base_obj1_ptr);
		memdelete(base_obj2_ptr);
		memdelete(node1);
		memdelete(node2);
		memdelete(node3);
	}

	SUBCASE("Mixed Recursive Types") {
		Array mixed_a1, mixed_a2;
		Dictionary mixed_d1a, mixed_d1b;
		mixed_d1a["value"] = 50;
		mixed_d1b["value"] = 50;
		mixed_a1.append(mixed_d1a);
		mixed_a1.append("common");
		mixed_a2.append(mixed_d1b);
		mixed_a2.append("common");
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(mixed_a1, mixed_a2), "Arrays containing equal dictionaries should be equal.");

		Array mixed_a3;
		Dictionary mixed_d1c;
		mixed_d1c["value"] = 51;
		mixed_a3.append(mixed_d1c);
		mixed_a3.append("common");
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(mixed_a1, mixed_a3), "Arrays containing unequal dictionaries should not be equal.");

		Dictionary mixed_d2a, mixed_d2b;
		Array mixed_a1a, mixed_a1b;
		mixed_a1a.append(true);
		mixed_a1b.append(true);
		mixed_d2a["list"] = mixed_a1a;
		mixed_d2a["id"] = 1;
		mixed_d2b["list"] = mixed_a1b;
		mixed_d2b["id"] = 1;
		CHECK_MESSAGE(VariantUtilityFunctions::deep_equals(mixed_d2a, mixed_d2b), "Dictionaries containing equal arrays should be equal.");

		Dictionary mixed_d2c;
		Array mixed_a1c;
		mixed_a1c.append(false);
		mixed_d2c["list"] = mixed_a1c;
		mixed_d2c["id"] = 1;
		CHECK_FALSE_MESSAGE(VariantUtilityFunctions::deep_equals(mixed_d2a, mixed_d2c), "Dictionaries containing unequal arrays should not be equal.");
	}
}

} // namespace TestVariantUtility
