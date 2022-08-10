/*************************************************************************/
/*  test_string_name.cpp                                                 */
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

#include "test_string_name.h"
#include "core/os/os.h"
#include "core/string_name.h"
#include <stdio.h>
#include <wchar.h>

namespace TestStringName {
#define CHECK(X)                                          \
	if (!(X)) {                                           \
		OS::get_singleton()->print("\tFAIL at %s\n", #X); \
		return false;                                     \
	} else {                                              \
		OS::get_singleton()->print("\tPASS\n");           \
	}

bool test_empty_string_names() {
	OS::get_singleton()->print("\n\ntest_empty_string_names\n");

	CHECK(StringName("") == StringName());
	CHECK(!StringName(""))
	CHECK(!StringName())
	CHECK(StringName().is_empty())
	CHECK(StringName("").is_empty())
	CHECK(!StringName("hi there").is_empty())

	return true;
}

bool test_string_name_assignments() {
	OS::get_singleton()->print("\n\ntest_string_name_assignments\n");

	CHECK(StringName("asdf") == "asdf")
	CHECK(StringName("") != StringName("asdf"))

	StringName a;
	CHECK(!a);

	StringName b;
	b = "";
	CHECK(!b);
	b = "asdf";
	CHECK(b);

	return true;
}

bool test_string_comparators_and_search() {
	OS::get_singleton()->print("\n\ntest_string_name_comparators\n");

	CHECK(StringName("asdf") > StringName())

	StringName test_search("test string search");
	CHECK(StringName::search("test string search"))

	CHECK(!StringName::search("test_string_comparators_and_search definitely should be no stringname this unique"))

	return true;
}

typedef bool (*TestFunc)();

TestFunc test_funcs[] = {

	test_empty_string_names,
	test_string_name_assignments,
	test_string_comparators_and_search,
	nullptr

};

MainLoop *test() {
	/** A character length != wchar_t may be forced, so the tests won't work */

	ERR_FAIL_COND_V(sizeof(CharType) != sizeof(wchar_t), nullptr);

	int count = 0;
	int passed = 0;

	while (true) {
		if (!test_funcs[count]) {
			break;
		}
		bool pass = test_funcs[count]();
		if (pass) {
			passed++;
		}
		OS::get_singleton()->print("\t%s\n", pass ? "PASS" : "FAILED");

		count++;
	}

	OS::get_singleton()->print("\n\n\n");
	OS::get_singleton()->print("*************\n");
	OS::get_singleton()->print("***TOTALS!***\n");
	OS::get_singleton()->print("*************\n");

	OS::get_singleton()->print("Passed %i of %i tests\n", passed, count);

	return nullptr;
}
} // namespace TestStringName
