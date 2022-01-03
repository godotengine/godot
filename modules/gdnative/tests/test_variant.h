/*************************************************************************/
/*  test_variant.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_GDNATIVE_VARIANT_H
#define TEST_GDNATIVE_VARIANT_H

#include <gdnative/gdnative.h>
#include <gdnative/variant.h>

#include "tests/test_macros.h"

namespace TestGDNativeVariant {

TEST_CASE("[GDNative Variant] New Variant with copy") {
	godot_variant src;
	godot_variant_new_int(&src, 42);

	godot_variant copy;
	godot_variant_new_copy(&copy, &src);

	CHECK(godot_variant_as_int(&copy) == 42);
	CHECK(godot_variant_get_type(&copy) == GODOT_VARIANT_TYPE_INT);

	godot_variant_destroy(&src);
	godot_variant_destroy(&copy);
}

TEST_CASE("[GDNative Variant] New Variant with Nil") {
	godot_variant val;
	godot_variant_new_nil(&val);

	CHECK(godot_variant_get_type(&val) == GODOT_VARIANT_TYPE_NIL);

	godot_variant_destroy(&val);
}

TEST_CASE("[GDNative Variant] New Variant with bool") {
	godot_variant val;
	godot_variant_new_bool(&val, true);

	CHECK(godot_variant_as_bool(&val));
	CHECK(godot_variant_get_type(&val) == GODOT_VARIANT_TYPE_BOOL);

	godot_variant_destroy(&val);
}

TEST_CASE("[GDNative Variant] New Variant with float") {
	godot_variant val;
	godot_variant_new_float(&val, 4.2);

	CHECK(godot_variant_as_float(&val) == 4.2);
	CHECK(godot_variant_get_type(&val) == GODOT_VARIANT_TYPE_FLOAT);

	godot_variant_destroy(&val);
}

TEST_CASE("[GDNative Variant] New Variant with String") {
	String str = "something";

	godot_variant val;
	godot_variant_new_string(&val, (godot_string *)&str);
	godot_string gd_str = godot_variant_as_string(&val);
	String *result = (String *)&gd_str;

	CHECK(*result == String("something"));
	CHECK(godot_variant_get_type(&val) == GODOT_VARIANT_TYPE_STRING);

	godot_variant_destroy(&val);
	godot_string_destroy(&gd_str);
}

TEST_CASE("[GDNative Variant] Variant call") {
	String str("something");
	godot_variant self;
	godot_variant_new_string(&self, (godot_string *)&str);

	godot_variant ret;

	godot_string_name method;
	godot_string_name_new_with_latin1_chars(&method, "is_valid_identifier");

	godot_variant_call_error error;
	godot_variant_call(&self, &method, nullptr, 0, &ret, &error);

	CHECK(godot_variant_get_type(&ret) == GODOT_VARIANT_TYPE_BOOL);
	CHECK(godot_variant_as_bool(&ret));

	godot_variant_destroy(&ret);
	godot_variant_destroy(&self);
	godot_string_name_destroy(&method);
}

TEST_CASE("[GDNative Variant] Variant evaluate") {
	godot_variant one;
	godot_variant_new_int(&one, 1);
	godot_variant two;
	godot_variant_new_int(&two, 2);

	godot_variant three;
	godot_variant_new_nil(&three);
	bool valid = false;

	godot_variant_evaluate(GODOT_VARIANT_OP_ADD, &one, &two, &three, &valid);

	CHECK(godot_variant_get_type(&three) == GODOT_VARIANT_TYPE_INT);
	CHECK(godot_variant_as_int(&three) == 3);
	CHECK(valid);

	godot_variant_destroy(&one);
	godot_variant_destroy(&two);
	godot_variant_destroy(&three);
}

TEST_CASE("[GDNative Variant] Variant set/get named") {
	godot_string_name x;
	godot_string_name_new_with_latin1_chars(&x, "x");

	Vector2 vec(0, 0);
	godot_variant self;
	godot_variant_new_vector2(&self, (godot_vector2 *)&vec);

	godot_variant set;
	godot_variant_new_float(&set, 1.0);

	bool set_valid = false;
	godot_variant_set_named(&self, &x, &set, &set_valid);

	bool get_valid = false;
	godot_variant get = godot_variant_get_named(&self, &x, &get_valid);

	CHECK(get_valid);
	CHECK(set_valid);
	CHECK(godot_variant_get_type(&get) == GODOT_VARIANT_TYPE_FLOAT);
	CHECK(godot_variant_as_float(&get) == 1.0);

	godot_string_name_destroy(&x);
	godot_variant_destroy(&self);
	godot_variant_destroy(&set);
	godot_variant_destroy(&get);
}

TEST_CASE("[GDNative Variant] Get utility function argument name") {
	godot_string_name function;
	godot_string_name_new_with_latin1_chars(&function, "pow");

	godot_string arg_name = godot_variant_get_utility_function_argument_name(&function, 0);

	String *arg_name_str = (String *)&arg_name;

	CHECK(*arg_name_str == "base");

	godot_string_destroy(&arg_name);
	godot_string_name_destroy(&function);
}

TEST_CASE("[GDNative Variant] Get utility function list") {
	int count = godot_variant_get_utility_function_count();

	godot_string_name *c_list = (godot_string_name *)godot_alloc(count * sizeof(godot_string_name));
	godot_variant_get_utility_function_list(c_list);

	List<StringName> cpp_list;
	Variant::get_utility_function_list(&cpp_list);

	godot_string_name *cur = c_list;

	for (const StringName &E : cpp_list) {
		const StringName &cpp_name = E;
		StringName *c_name = (StringName *)cur++;

		CHECK(*c_name == cpp_name);
	}

	godot_free(c_list);
}
} // namespace TestGDNativeVariant

#endif // TEST_GDNATIVE_VARIANT_H
