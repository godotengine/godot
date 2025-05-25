/**************************************************************************/
/*  test_expression.cpp                                                   */
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

#include "test_expression.h"

#include "core/math/expression.h"
#include "core/os/os.h"

#define CHECK_MESSAGE(X, msg)                                      \
	if (!(X)) {                                                    \
		OS::get_singleton()->print("\tFAIL at %s: %s\n", #X, msg); \
		return false;                                              \
	} else {                                                       \
		OS::get_singleton()->print("\tPASS\n");                    \
	}

namespace TestExpression {

bool floating_point_notation() {
	OS::get_singleton()->print("\n\nTest 1: Floating-point notation\n");

	Expression expression;

	CHECK_MESSAGE(
			expression.parse("2.") == OK,
			"The expression should parse successfully.");
	CHECK_MESSAGE(
			Math::is_equal_approx(expression.execute(Array()), 2.0),
			"The expression should return the expected result.");

	CHECK_MESSAGE(
			expression.parse("(2.)") == OK,
			"The expression should parse successfully.");
	CHECK_MESSAGE(
			Math::is_equal_approx(expression.execute(Array()), 2.0),
			"The expression should return the expected result.");

	CHECK_MESSAGE(
			expression.parse(".3") == OK,
			"The expression should parse successfully.");
	CHECK_MESSAGE(
			Math::is_equal_approx(expression.execute(Array()), 0.3),
			"The expression should return the expected result.");

	CHECK_MESSAGE(
			expression.parse("2.+5.") == OK,
			"The expression should parse successfully.");
	CHECK_MESSAGE(
			Math::is_equal_approx(expression.execute(Array()), 7.0),
			"The expression should return the expected result.");

	CHECK_MESSAGE(
			expression.parse(".3-.8") == OK,
			"The expression should parse successfully.");
	CHECK_MESSAGE(
			Math::is_equal_approx(expression.execute(Array()), -0.5),
			"The expression should return the expected result.");

	CHECK_MESSAGE(
			expression.parse("2.+.2") == OK,
			"The expression should parse successfully.");
	CHECK_MESSAGE(
			Math::is_equal_approx(expression.execute(Array()), 2.2),
			"The expression should return the expected result.");

	CHECK_MESSAGE(
			expression.parse(".0*0.") == OK,
			"The expression should parse successfully.");
	CHECK_MESSAGE(
			Math::is_equal_approx(expression.execute(Array()), 0.0),
			"The expression should return the expected result.");

	return true;
}

typedef bool (*TestFunc)();

TestFunc test_funcs[] = {
	floating_point_notation,
	nullptr
};

MainLoop *test() {
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
} // namespace TestExpression
