/*************************************************************************/
/*  test_array.h                                                         */
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

#ifndef TEST_ARRAY_H
#define TEST_ARRAY_H

#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/vector.h"
#include "core/variant/array.h"
#include "core/variant/container_type_validate.h"
#include "core/variant/variant.h"
#include "tests/test_macros.h"
#include "tests/test_tools.h"

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

TEST_CASE("[Array] size(), clear(), and is_empty()") {
	Array arr;
	CHECK(arr.size() == 0);
	CHECK(arr.is_empty());
	arr.push_back(1);
	CHECK(arr.size() == 1);
	arr.clear();
	CHECK(arr.is_empty());
	CHECK(arr.size() == 0);
}

TEST_CASE("[Array] Assignment and comparison operators") {
	Array arr1;
	Array arr2;
	arr1.push_back(1);
	CHECK(arr1 != arr2);
	CHECK(arr1 > arr2);
	CHECK(arr1 >= arr2);
	arr2.push_back(2);
	CHECK(arr1 != arr2);
	CHECK(arr1 < arr2);
	CHECK(arr1 <= arr2);
	CHECK(arr2 > arr1);
	CHECK(arr2 >= arr1);
	Array arr3 = arr2;
	CHECK(arr3 == arr2);
}

TEST_CASE("[Array] append_array()") {
	Array arr1;
	Array arr2;
	arr1.push_back(1);
	arr1.append_array(arr2);
	CHECK(arr1.size() == 1);
	arr2.push_back(2);
	arr1.append_array(arr2);
	CHECK(arr1.size() == 2);
	CHECK(int(arr1[0]) == 1);
	CHECK(int(arr1[1]) == 2);
}

TEST_CASE("[Array] resize(), insert(), and erase()") {
	Array arr;
	arr.resize(2);
	CHECK(arr.size() == 2);
	arr.insert(0, 1);
	CHECK(int(arr[0]) == 1);
	arr.insert(0, 2);
	CHECK(int(arr[0]) == 2);
	arr.erase(2);
	CHECK(int(arr[0]) == 1);
}

TEST_CASE("[Array] front() and back()") {
	Array arr;
	arr.push_back(1);
	CHECK(int(arr.front()) == 1);
	CHECK(int(arr.back()) == 1);
	arr.push_back(3);
	CHECK(int(arr.front()) == 1);
	CHECK(int(arr.back()) == 3);
}

TEST_CASE("[Array] has() and count()") {
	Array arr;
	arr.push_back(1);
	arr.push_back(1);
	CHECK(arr.has(1));
	CHECK(!arr.has(2));
	CHECK(arr.count(1) == 2);
	CHECK(arr.count(2) == 0);
}

TEST_CASE("[Array] remove()") {
	Array arr;
	arr.push_back(1);
	arr.push_back(2);
	arr.remove(0);
	CHECK(arr.size() == 1);
	CHECK(int(arr[0]) == 2);
	arr.remove(0);
	CHECK(arr.size() == 0);

	// The array is now empty; try to use `remove()` again.
	// Normally, this prints an error message so we silence it.
	ERR_PRINT_OFF;
	arr.remove(0);
	ERR_PRINT_ON;

	CHECK(arr.size() == 0);
}

TEST_CASE("[Array] get()") {
	Array arr;
	arr.push_back(1);
	CHECK(int(arr.get(0)) == 1);
}

TEST_CASE("[Array] sort()") {
	Array arr;

	arr.push_back(3);
	arr.push_back(4);
	arr.push_back(2);
	arr.push_back(1);
	arr.sort();
	int val = 1;
	for (int i = 0; i < arr.size(); i++) {
		CHECK(int(arr[i]) == val);
		val++;
	}
}

TEST_CASE("[Array] push_front(), pop_front(), pop_back()") {
	Array arr;
	arr.push_front(1);
	arr.push_front(2);
	CHECK(int(arr[0]) == 2);
	arr.pop_front();
	CHECK(int(arr[0]) == 1);
	CHECK(arr.size() == 1);
	arr.push_front(2);
	arr.push_front(3);
	arr.pop_back();
	CHECK(int(arr[1]) == 2);
	CHECK(arr.size() == 2);
}

TEST_CASE("[Array] pop_at()") {
	ErrorDetector ed;

	Array arr;
	arr.push_back(2);
	arr.push_back(4);
	arr.push_back(6);
	arr.push_back(8);
	arr.push_back(10);

	REQUIRE(int(arr.pop_at(2)) == 6);
	REQUIRE(arr.size() == 4);
	CHECK(int(arr[0]) == 2);
	CHECK(int(arr[1]) == 4);
	CHECK(int(arr[2]) == 8);
	CHECK(int(arr[3]) == 10);

	REQUIRE(int(arr.pop_at(2)) == 8);
	REQUIRE(arr.size() == 3);
	CHECK(int(arr[0]) == 2);
	CHECK(int(arr[1]) == 4);
	CHECK(int(arr[2]) == 10);

	// Negative index.
	REQUIRE(int(arr.pop_at(-1)) == 10);
	REQUIRE(arr.size() == 2);
	CHECK(int(arr[0]) == 2);
	CHECK(int(arr[1]) == 4);

	// Invalid pop.
	ed.clear();
	ERR_PRINT_OFF;
	const Variant ret = arr.pop_at(-15);
	ERR_PRINT_ON;
	REQUIRE(ret.is_null());
	CHECK(ed.has_error);

	REQUIRE(int(arr.pop_at(0)) == 2);
	REQUIRE(arr.size() == 1);
	CHECK(int(arr[0]) == 4);

	REQUIRE(int(arr.pop_at(0)) == 4);
	REQUIRE(arr.is_empty());

	// Pop from empty array.
	ed.clear();
	REQUIRE(arr.pop_at(24).is_null());
	CHECK_FALSE(ed.has_error);
}

TEST_CASE("[Array] max() and min()") {
	Array arr;
	arr.push_back(3);
	arr.push_front(4);
	arr.push_back(5);
	arr.push_back(2);
	int max = int(arr.max());
	int min = int(arr.min());
	CHECK(max == 5);
	CHECK(min == 2);
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
