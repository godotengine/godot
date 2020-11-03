/*************************************************************************/
/*  test_array.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_ARRAY_H
#define TEST_ARRAY_H

#include "core/array.h"

#include "tests/test_macros.h"

namespace TestArray {

static inline Array build_array() {
	return Array();
}
template <typename... Targs>
static inline Array build_array(Variant item, Targs... Fargs) {
	Array a = build_array(Fargs...);
	a.push_front(item);
	return a;
}
static inline Dictionary build_dictionary() {
	return Dictionary();
}
template <typename... Targs>
static inline Dictionary build_dictionary(Variant key, Variant item, Targs... Fargs) {
	Dictionary d = build_dictionary(Fargs...);
	d[key] = item;
	return d;
}

TEST_CASE("[Array] Duplicate array") {
	// a = [1, [2, 2], {3: 3}]
	Array a = build_array(1, build_array(2, 2), build_dictionary(3, 3));

	// Deep copy
	Array deep_a = a.duplicate(true);
	CHECK_MESSAGE(deep_a.id() != a.id(), "Should create a new array");
	CHECK_MESSAGE(Array(deep_a[1]).id() != Array(a[1]).id(), "Should clone nested array");
	CHECK_MESSAGE(Dictionary(deep_a[2]).id() != Dictionary(a[2]).id(), "Should clone nested dictionary");
	CHECK_EQ(deep_a, a);
	deep_a.push_back(1);
	CHECK_NE(deep_a, a);
	deep_a.pop_back();
	Array(deep_a[1]).push_back(1);
	CHECK_NE(deep_a, a);
	Array(deep_a[1]).pop_back();
	CHECK_EQ(deep_a, a);

	// Shallow copy
	Array shallow_a = a.duplicate(false);
	CHECK_MESSAGE(shallow_a.id() != a.id(), "Should create a new array");
	CHECK_MESSAGE(Array(shallow_a[1]).id() == Array(a[1]).id(), "Should keep nested array");
	CHECK_MESSAGE(Dictionary(shallow_a[2]).id() == Dictionary(a[2]).id(), "Should keep nested dictionary");
	CHECK_EQ(shallow_a, a);
	Array(shallow_a).push_back(1);
	CHECK_NE(shallow_a, a);
}

TEST_CASE("[Array] Duplicate recursive array") {
	// Self recursive
	Array a;
	a.push_back(a);

	Array a_shallow = a.duplicate(false);
	CHECK_EQ(a, a_shallow);

	// Deep copy of recursive array endup with recursion limit and return
	// an invalid result (multiple nested arrays), the point is we should
	// not end up with a segfault and an error log should be printed
	ERR_PRINT_OFF;
	a.duplicate(true);
	ERR_PRINT_ON;

	// Nested recursive
	Array a1;
	Array a2;
	a2.push_back(a1);
	a1.push_back(a2);

	Array a1_shallow = a1.duplicate(false);
	CHECK_EQ(a1, a1_shallow);

	// Same deep copy issue as above
	ERR_PRINT_OFF;
	a1.duplicate(true);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Array teardown will leak memory
	a.clear();
	a1.clear();
	a2.clear();
}

TEST_CASE("[Array] Hash array") {
	// a = [1, [2, 2], {3: 3}]
	Array a = build_array(1, build_array(2, 2), build_dictionary(3, 3));
	uint32_t original_hash = a.hash();

	a.push_back(1);
	CHECK_NE(a.hash(), original_hash);

	a.pop_back();
	CHECK_EQ(a.hash(), original_hash);

	Array(a[1]).push_back(1);
	CHECK_NE(a.hash(), original_hash);
	Array(a[1]).pop_back();
	CHECK_EQ(a.hash(), original_hash);

	(Dictionary(a[2]))[1] = 1;
	CHECK_NE(a.hash(), original_hash);
	Dictionary(a[2]).erase(1);
	CHECK_EQ(a.hash(), original_hash);

	Array a2 = a.duplicate(true);
	CHECK_EQ(a2.hash(), a.hash());
}

TEST_CASE("[Array] Hash recursive array") {
	Array a1;
	a1.push_back(a1);

	Array a2;
	a2.push_back(a2);

	// Hash should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_EQ(a1.hash(), a2.hash());
	ERR_PRINT_ON;

	// Break the recursivity otherwise Array teardown will leak memory
	a1.clear();
	a2.clear();
}

TEST_CASE("[Array] Empty comparison") {
	Array a1;
	Array a2;

	// test both operator== and operator!=
	CHECK_EQ(a1, a2);
	CHECK_FALSE(a1 != a2);
}

TEST_CASE("[Array] Flat comparison") {
	Array a1 = build_array(1);
	Array a2 = build_array(1);
	Array other_a = build_array(2);

	// test both operator== and operator!=
	CHECK_EQ(a1, a1); // compare self
	CHECK_FALSE(a1 != a1);
	CHECK_EQ(a1, a2); // different equivalent arrays
	CHECK_FALSE(a1 != a2);
	CHECK_NE(a1, other_a); // different arrays with different content
	CHECK_FALSE(a1 == other_a);
}

TEST_CASE("[Array] Nested array comparison") {
	// a1 = [[[1], 2], 3]
	Array a1 = build_array(build_array(build_array(1), 2), 3);

	Array a2 = a1.duplicate(true);

	// other_a = [[[1, 0], 2], 3]
	Array other_a = build_array(build_array(build_array(1, 0), 2), 3);

	// test both operator== and operator!=
	CHECK_EQ(a1, a1); // compare self
	CHECK_FALSE(a1 != a1);
	CHECK_EQ(a1, a2); // different equivalent arrays
	CHECK_FALSE(a1 != a2);
	CHECK_NE(a1, other_a); // different arrays with different content
	CHECK_FALSE(a1 == other_a);
}

TEST_CASE("[Array] Nested dictionary comparison") {
	// a1 = [{1: 2}, 3]
	Array a1 = build_array(build_dictionary(1, 2), 3);

	Array a2 = a1.duplicate(true);

	// other_a = [{1: 0}, 3]
	Array other_a = build_array(build_dictionary(1, 0), 3);

	// test both operator== and operator!=
	CHECK_EQ(a1, a1); // compare self
	CHECK_FALSE(a1 != a1);
	CHECK_EQ(a1, a2); // different equivalent arrays
	CHECK_FALSE(a1 != a2);
	CHECK_NE(a1, other_a); // different arrays with different content
	CHECK_FALSE(a1 == other_a);
}

TEST_CASE("[Array] Recursive comparison") {
	Array a1;
	a1.push_back(a1);

	Array a2;
	a2.push_back(a2);

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_EQ(a1, a2);
	CHECK_FALSE(a1 != a2);
	ERR_PRINT_ON;

	a1.push_back(1);
	a2.push_back(1);

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_EQ(a1, a2);
	CHECK_FALSE(a1 != a2);
	ERR_PRINT_ON;

	a1.push_back(1);
	a2.push_back(2);

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_NE(a1, a2);
	CHECK_FALSE(a1 == a2);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Array tearndown will leak memory
	a1.clear();
	a2.clear();
}

TEST_CASE("[Array] Recursive self comparison") {
	Array a1;
	Array a2;
	a2.push_back(a1);
	a1.push_back(a2);

	CHECK_EQ(a1, a1);
	CHECK_FALSE(a1 != a1);

	// Break the recursivity otherwise Array tearndown will leak memory
	a1.clear();
	a2.clear();
}

} // namespace TestArray

#endif // TEST_ARRAY_H
