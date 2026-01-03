/**************************************************************************/
/*  test_array.h                                                          */
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

#include "core/variant/array.h"
#include "tests/test_macros.h"
#include "tests/test_tools.h"

namespace TestArray {
TEST_CASE("[Array] initializer list") {
	Array arr = { 0, 1, "test", true, { 0.0, 1.0 } };
	CHECK(arr.size() == 5);
	CHECK(arr[0] == Variant(0));
	CHECK(arr[1] == Variant(1));
	CHECK(arr[2] == Variant("test"));
	CHECK(arr[3] == Variant(true));
	CHECK(arr[4] == Variant({ 0.0, 1.0 }));

	arr = { "reassign" };
	CHECK(arr.size() == 1);
	CHECK(arr[0] == Variant("reassign"));

	TypedArray<int> typed_arr = { 0, 1, 2 };
	CHECK(typed_arr.size() == 3);
	CHECK(typed_arr[0] == Variant(0));
	CHECK(typed_arr[1] == Variant(1));
	CHECK(typed_arr[2] == Variant(2));
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

TEST_CASE("[Array] fill()") {
	Array arr;
	arr.resize(5);
	arr.fill(7);
	Array expected = { 7, 7, 7, 7, 7 };
	CHECK_EQ(arr, expected);

	Array empty;
	empty.fill(7);
	Array empty_expected;
	CHECK_EQ(empty, empty_expected);
}

TEST_CASE("[Array] reverse()") {
	Array odd_sized = { 1, 2, 3 };
	odd_sized.reverse();
	Array odd_sized_expected = { 3, 2, 1 };
	CHECK_EQ(odd_sized, odd_sized_expected);

	Array even_sized = { "a", "b", "c", "d" };
	even_sized.reverse();
	Array even_sized_expected = { "d", "c", "b", "a" };
	CHECK_EQ(even_sized, even_sized_expected);

	Array empty;
	empty.reverse();
	Array empty_expected;
	CHECK_EQ(empty, empty_expected);
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

TEST_CASE("[Array] append()") {
	Array arr;
	arr.append(1);
	arr.append(2);
	arr.append(3);
	arr.append("test");
	Array expected = { 1, 2, 3, "test" };
	CHECK_EQ(arr, expected);
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
	arr.resize(0);
	CHECK(arr.size() == 0);
	arr.insert(0, 8);
	CHECK(arr.size() == 1);
	arr.insert(1, 16);
	CHECK(int(arr[1]) == 16);
	arr.insert(-1, 3);
	CHECK(int(arr[1]) == 3);
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
	Array arr = { 1, 1 };
	CHECK(arr.has(1));
	CHECK(!arr.has(2));
	CHECK(arr.count(1) == 2);
	CHECK(arr.count(2) == 0);
}

TEST_CASE("[Array] remove_at()") {
	Array arr = { 1, 2 };
	arr.remove_at(0);
	CHECK(arr.size() == 1);
	CHECK(int(arr[0]) == 2);
	arr.remove_at(0);
	CHECK(arr.size() == 0);

	// Negative index.
	arr.push_back(3);
	arr.push_back(4);
	arr.remove_at(-1);
	CHECK(arr.size() == 1);
	CHECK(int(arr[0]) == 3);
	arr.remove_at(-1);
	CHECK(arr.size() == 0);

	// The array is now empty; try to use `remove_at()` again.
	// Normally, this prints an error message so we silence it.
	ERR_PRINT_OFF;
	arr.remove_at(0);
	ERR_PRINT_ON;

	CHECK(arr.size() == 0);
}

TEST_CASE("[Array] get() and set()") {
	Array arr = { 1, 2, 3 };
	CHECK_EQ(int(arr.get(0)), 1);
	CHECK_EQ(int(arr.get(1)), 2);
	CHECK_EQ(int(arr.get(2)), 3);

	arr.set(1, 5);
	CHECK_EQ(int(arr.get(1)), 5);
}

TEST_CASE("[Array] sort() and bsearch()") {
	Array arr = { 3, 4, 2, 1 };
	arr.sort();
	Array expected = { 1, 2, 3, 4 };
	CHECK_EQ(arr, expected);

	CHECK_EQ(arr.bsearch(1), 0);
	CHECK_EQ(arr.bsearch(3), 2);
	CHECK_EQ(arr.bsearch(-100), 0);
	CHECK_EQ(arr.bsearch(100), 4);
}

static bool _order_descending(int p_a, int p_b) {
	return p_b < p_a;
}

TEST_CASE("[Array] sort_custom() and bsearch_custom()") {
	Array arr = { 3, 4, 2, 1 };
	arr.sort_custom(callable_mp_static(_order_descending));
	Array expected = { 4, 3, 2, 1 };
	CHECK_EQ(arr, expected);

	CHECK_EQ(arr.bsearch_custom(1, callable_mp_static(_order_descending)), 3);
	CHECK_EQ(arr.bsearch_custom(4, callable_mp_static(_order_descending)), 0);
	CHECK_EQ(arr.bsearch_custom(100, callable_mp_static(_order_descending)), 0);
	CHECK_EQ(arr.bsearch_custom(-100, callable_mp_static(_order_descending)), 4);
}

static bool _is_even(int p_num) {
	return p_num % 2 == 0;
}

static bool _is_odd(int p_num) {
	return p_num % 2 == 1;
}

TEST_CASE("[Array] filter(), any(), all()") {
	Array nums = { 1, 2, 3, 4, 5, 6, 7 };
	CHECK(nums.any(callable_mp_static(_is_odd)));
	CHECK(nums.any(callable_mp_static(_is_even)));
	CHECK(!nums.all(callable_mp_static(_is_odd)));
	CHECK(!nums.all(callable_mp_static(_is_even)));

	Array odd = nums.filter(callable_mp_static(_is_odd));
	Array odd_expected = { 1, 3, 5, 7 };
	CHECK_EQ(odd, odd_expected);

	Array even = nums.filter(callable_mp_static(_is_even));
	Array even_expected = { 2, 4, 6 };
	CHECK_EQ(even, even_expected);

	CHECK(odd.all(callable_mp_static(_is_odd)));
	CHECK(odd.any(callable_mp_static(_is_odd)));
	CHECK(!odd.all(callable_mp_static(_is_even)));
	CHECK(!odd.any(callable_mp_static(_is_even)));
}

static int _add(int p_a, int p_b) {
	return p_a + p_b;
}

TEST_CASE("[Array] map() and reduce()") {
	Array array = { 1, 2, 3, 4, 5 };
	Array mapped = array.map(callable_mp_static(_add).bind(5));
	Array mapped_expected = { 6, 7, 8, 9, 10 };
	CHECK_EQ(mapped, mapped_expected);

	Variant sum = 0;
	CHECK_EQ(int(array.reduce(callable_mp_static(_add), sum)), 15);
}

static int _get_y_callable(Vector2i const &p_v) {
	return p_v.y;
}

TEST_CASE("[Array] sort_by_key()") {
	Array sort_array = { Vector2i(0, 1), Vector2i(1, 1), Vector2i(0, 0), Vector2i(1, 0), Vector2i(0, 2), Vector2i(1, 2) };
	Array expected = { Vector2i(0, 0), Vector2i(1, 0), Vector2i(0, 1), Vector2i(1, 1), Vector2i(0, 2), Vector2i(1, 2) };
	Array expected_reversed = { Vector2i(0, 2), Vector2i(1, 2), Vector2i(0, 1), Vector2i(1, 1), Vector2i(0, 0), Vector2i(1, 0) };

	sort_array.sort_by_key(callable_mp_static(_get_y_callable));
	CHECK_EQ(sort_array, expected);

	sort_array.sort_by_key(callable_mp_static(_get_y_callable), true);
	CHECK_EQ(sort_array, expected_reversed);
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

	Array arr = { 2, 4, 6, 8, 10 };

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

TEST_CASE("[Array] slice()") {
	Array array = { 0, 1, 2, 3, 4, 5 };

	Array slice0 = array.slice(0, 0);
	CHECK(slice0.size() == 0);

	Array slice1 = array.slice(1, 3);
	CHECK(slice1.size() == 2);
	CHECK(slice1[0] == Variant(1));
	CHECK(slice1[1] == Variant(2));

	Array slice2 = array.slice(1, -1);
	CHECK(slice2.size() == 4);
	CHECK(slice2[0] == Variant(1));
	CHECK(slice2[1] == Variant(2));
	CHECK(slice2[2] == Variant(3));
	CHECK(slice2[3] == Variant(4));

	Array slice3 = array.slice(3);
	CHECK(slice3.size() == 3);
	CHECK(slice3[0] == Variant(3));
	CHECK(slice3[1] == Variant(4));
	CHECK(slice3[2] == Variant(5));

	Array slice4 = array.slice(2, -2);
	CHECK(slice4.size() == 2);
	CHECK(slice4[0] == Variant(2));
	CHECK(slice4[1] == Variant(3));

	Array slice5 = array.slice(-2);
	CHECK(slice5.size() == 2);
	CHECK(slice5[0] == Variant(4));
	CHECK(slice5[1] == Variant(5));

	Array slice6 = array.slice(2, 42);
	CHECK(slice6.size() == 4);
	CHECK(slice6[0] == Variant(2));
	CHECK(slice6[1] == Variant(3));
	CHECK(slice6[2] == Variant(4));
	CHECK(slice6[3] == Variant(5));

	Array slice7 = array.slice(4, 0, -2);
	CHECK(slice7.size() == 2);
	CHECK(slice7[0] == Variant(4));
	CHECK(slice7[1] == Variant(2));

	Array slice8 = array.slice(5, 0, -2);
	CHECK(slice8.size() == 3);
	CHECK(slice8[0] == Variant(5));
	CHECK(slice8[1] == Variant(3));
	CHECK(slice8[2] == Variant(1));

	Array slice9 = array.slice(10, 0, -2);
	CHECK(slice9.size() == 3);
	CHECK(slice9[0] == Variant(5));
	CHECK(slice9[1] == Variant(3));
	CHECK(slice9[2] == Variant(1));

	Array slice10 = array.slice(2, -10, -1);
	CHECK(slice10.size() == 3);
	CHECK(slice10[0] == Variant(2));
	CHECK(slice10[1] == Variant(1));
	CHECK(slice10[2] == Variant(0));

	ERR_PRINT_OFF;
	Array slice11 = array.slice(4, 1);
	CHECK(slice11.size() == 0);

	Array slice12 = array.slice(3, -4);
	CHECK(slice12.size() == 0);
	ERR_PRINT_ON;

	Array slice13 = Array().slice(1);
	CHECK(slice13.size() == 0);

	Array slice14 = array.slice(6);
	CHECK(slice14.size() == 0);
}

TEST_CASE("[Array] Duplicate array") {
	// a = [1, [2, 2], {3: 3}]
	Array a = { 1, { 2, 2 }, Dictionary({ { 3, 3 } }) };

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

	// Deep copy of recursive array ends up with recursion limit and return
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
	Array a = { 1, { 2, 2 }, Dictionary({ { 3, 3 } }) };
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
	Array a1 = { 1 };
	Array a2 = { 1 };
	Array other_a = { 2 };

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
	Array a1 = { { { 1 }, 2 }, 3 };

	Array a2 = a1.duplicate(true);

	// other_a = [[[1, 0], 2], 3]
	Array other_a = { { { 1, 0 }, 2 }, 3 };

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
	Array a1 = { Dictionary({ { 1, 2 } }), 3 };

	Array a2 = a1.duplicate(true);

	// other_a = [{1: 0}, 3]
	Array other_a = { Dictionary({ { 1, 0 } }), 3 };

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

TEST_CASE("[Array] Iteration") {
	Array a1 = { 1, 2, 3 };
	Array a2 = { 1, 2, 3 };

	int idx = 0;
	for (Variant &E : a1) {
		CHECK_EQ(int(a2[idx]), int(E));
		idx++;
	}

	CHECK_EQ(idx, a1.size());

	idx = 0;

	for (const Variant &E : (const Array &)a1) {
		CHECK_EQ(int(a2[idx]), int(E));
		idx++;
	}

	CHECK_EQ(idx, a1.size());

	a1.clear();
}

TEST_CASE("[Array] Iteration and modification") {
	Array a1 = { 1, 2, 3 };
	Array a2 = { 2, 3, 4 };
	Array a3 = { 1, 2, 3 };
	Array a4 = { 1, 2, 3 };
	a3.make_read_only();

	int idx = 0;
	for (Variant &E : a1) {
		E = a2[idx];
		idx++;
	}

	CHECK_EQ(a1, a2);

	// Ensure read-only is respected.
	idx = 0;
	for (Variant &E : a3) {
		E = a2[idx];
	}

	CHECK_EQ(a3, a4);

	a1.clear();
	a2.clear();
	a4.clear();
}

TEST_CASE("[Array] Typed copying") {
	TypedArray<int> a1 = { 1 };
	TypedArray<double> a2 = { 1.0 };

	Array a3 = a1;
	TypedArray<int> a4 = a3;

	Array a5 = a2;
	TypedArray<int> a6 = a5;

	a3[0] = 2;
	a4[0] = 3;

	// Same typed TypedArray should be shared.
	CHECK_EQ(a1[0], Variant(3));
	CHECK_EQ(a3[0], Variant(3));
	CHECK_EQ(a4[0], Variant(3));

	a5[0] = 2.0;
	a6[0] = 3.0;

	// Different typed TypedArray should not be shared.
	CHECK_EQ(a2[0], Variant(2.0));
	CHECK_EQ(a5[0], Variant(2.0));
	CHECK_EQ(a6[0], Variant(3.0));

	a1.clear();
	a2.clear();
	a3.clear();
	a4.clear();
	a5.clear();
	a6.clear();
}

TEST_CASE("[Array] find() and rfind()") {
	Array array = { "a", "b", "c", "a", "b", "c" };

	CHECK_EQ(array.find("a"), 0);
	CHECK_EQ(array.find("c"), 2);
	CHECK_EQ(array.find("a", 1), 3);

	CHECK_EQ(array.rfind("b"), 4);
	CHECK_EQ(array.rfind("c", -2), 2);
}

static bool _find_custom_callable(const Variant &p_val) {
	return (int)p_val % 2 == 0;
}

TEST_CASE("[Array] Test find_custom") {
	Array a1 = { 1, 3, 4, 5, 8, 9 };
	// Find first even number.
	int index = a1.find_custom(callable_mp_static(_find_custom_callable));
	CHECK_EQ(index, 2);
}

TEST_CASE("[Array] Test rfind_custom") {
	Array a1 = { 1, 3, 4, 5, 8, 9 };
	// Find last even number.
	int index = a1.rfind_custom(callable_mp_static(_find_custom_callable));
	CHECK_EQ(index, 4);
}

TEST_CASE("[Array] Test typed arrays") {
	Array arr1;
	CHECK_FALSE(arr1.is_typed());

	arr1.set_typed(Variant::FLOAT, StringName(), Variant());
	CHECK(arr1.is_typed());
	CHECK_EQ(arr1.get_typed_builtin(), Variant::FLOAT);

	arr1.push_back(1);
	CHECK_EQ(arr1.size(), 1);

	ERR_PRINT_OFF;
	arr1.push_back("test wrong type");
	CHECK_EQ(arr1.size(), 1);
	ERR_PRINT_ON;

	Array arr2;
	arr2.set_typed(Variant::INT, StringName(), Variant());
	CHECK_FALSE(arr1.is_same_typed(arr2));

	Array arr3;
	arr3.set_typed(Variant::OBJECT, "Node", Variant());
	CHECK_EQ(arr3.get_typed_class_name(), "Node");
}

} // namespace TestArray
