/**************************************************************************/
/*  test_callable.h                                                       */
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

#ifndef TEST_CALLABLE_H
#define TEST_CALLABLE_H

#include "core/object/class_db.h"
#include "core/object/object.h"

#include "tests/test_macros.h"

namespace TestCallable {

class TestClass : public Object {
	GDCLASS(TestClass, Object);

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("test_func_1", "foo", "bar"), &TestClass::test_func_1);
		ClassDB::bind_method(D_METHOD("test_func_2", "foo", "bar", "baz"), &TestClass::test_func_2);
		ClassDB::bind_static_method("TestClass", D_METHOD("test_func_5", "foo", "bar"), &TestClass::test_func_5);
		ClassDB::bind_static_method("TestClass", D_METHOD("test_func_6", "foo", "bar", "baz"), &TestClass::test_func_6);

		{
			MethodInfo mi;
			mi.name = "test_func_7";
			mi.arguments.push_back(PropertyInfo(Variant::INT, "foo"));
			mi.arguments.push_back(PropertyInfo(Variant::INT, "bar"));

			ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "test_func_7", &TestClass::test_func_7, mi, varray(), false);
		}

		{
			MethodInfo mi;
			mi.name = "test_func_8";
			mi.arguments.push_back(PropertyInfo(Variant::INT, "foo"));
			mi.arguments.push_back(PropertyInfo(Variant::INT, "bar"));
			mi.arguments.push_back(PropertyInfo(Variant::INT, "baz"));

			ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "test_func_8", &TestClass::test_func_8, mi, varray(), false);
		}
	}

public:
	void test_func_1(int p_foo, int p_bar) {}
	void test_func_2(int p_foo, int p_bar, int p_baz) {}

	int test_func_3(int p_foo, int p_bar) const { return 0; }
	int test_func_4(int p_foo, int p_bar, int p_baz) const { return 0; }

	static void test_func_5(int p_foo, int p_bar) {}
	static void test_func_6(int p_foo, int p_bar, int p_baz) {}

	void test_func_7(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {}
	void test_func_8(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {}
};

TEST_CASE("[Callable] Argument count") {
	TestClass *my_test = memnew(TestClass);

	// Bound methods tests.

	// Test simple methods.
	Callable callable_1 = Callable(my_test, "test_func_1");
	CHECK_EQ(callable_1.get_argument_count(), 2);
	Callable callable_2 = Callable(my_test, "test_func_2");
	CHECK_EQ(callable_2.get_argument_count(), 3);
	Callable callable_3 = Callable(my_test, "test_func_5");
	CHECK_EQ(callable_3.get_argument_count(), 2);
	Callable callable_4 = Callable(my_test, "test_func_6");
	CHECK_EQ(callable_4.get_argument_count(), 3);

	// Test vararg methods.
	Callable callable_vararg_1 = Callable(my_test, "test_func_7");
	CHECK_MESSAGE(callable_vararg_1.get_argument_count() == 2, "vararg Callable should return the number of declared arguments");
	Callable callable_vararg_2 = Callable(my_test, "test_func_8");
	CHECK_MESSAGE(callable_vararg_2.get_argument_count() == 3, "vararg Callable should return the number of declared arguments");

	// Callable MP tests.

	// Test simple methods.
	Callable callable_mp_1 = callable_mp(my_test, &TestClass::test_func_1);
	CHECK_EQ(callable_mp_1.get_argument_count(), 2);
	Callable callable_mp_2 = callable_mp(my_test, &TestClass::test_func_2);
	CHECK_EQ(callable_mp_2.get_argument_count(), 3);
	Callable callable_mp_3 = callable_mp(my_test, &TestClass::test_func_3);
	CHECK_EQ(callable_mp_3.get_argument_count(), 2);
	Callable callable_mp_4 = callable_mp(my_test, &TestClass::test_func_4);
	CHECK_EQ(callable_mp_4.get_argument_count(), 3);

	// Test static methods.
	Callable callable_mp_static_1 = callable_mp_static(&TestClass::test_func_5);
	CHECK_EQ(callable_mp_static_1.get_argument_count(), 2);
	Callable callable_mp_static_2 = callable_mp_static(&TestClass::test_func_6);
	CHECK_EQ(callable_mp_static_2.get_argument_count(), 3);

	// Test bind.
	Callable callable_mp_bind_1 = callable_mp_2.bind(1);
	CHECK_MESSAGE(callable_mp_bind_1.get_argument_count() == 2, "bind should subtract from the argument count");
	Callable callable_mp_bind_2 = callable_mp_2.bind(1, 2);
	CHECK_MESSAGE(callable_mp_bind_2.get_argument_count() == 1, "bind should subtract from the argument count");

	// Test unbind.
	Callable callable_mp_unbind_1 = callable_mp_2.unbind(1);
	CHECK_MESSAGE(callable_mp_unbind_1.get_argument_count() == 4, "unbind should add to the argument count");
	Callable callable_mp_unbind_2 = callable_mp_2.unbind(2);
	CHECK_MESSAGE(callable_mp_unbind_2.get_argument_count() == 5, "unbind should add to the argument count");

	memdelete(my_test);
}
} // namespace TestCallable

#endif // TEST_CALLABLE_H
