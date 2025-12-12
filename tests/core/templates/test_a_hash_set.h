/**************************************************************************/
/*  test_a_hash_set.h                                                     */
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

#include "core/templates/a_hash_set.h"

#include "tests/test_macros.h"

namespace TestAHashSet {

TEST_CASE("[AHashSet] List initialization") {
	AHashSet<int> set{ 0, 1, 2, 3, 4 };

	CHECK(set.size() == 5);
	CHECK(set.has(0));
	CHECK(set.has(1));
	CHECK(set.has(2));
	CHECK(set.has(3));
	CHECK(set.has(4));
}

TEST_CASE("[AHashSet] List initialization with existing elements") {
	AHashSet<int> set{ 0, 0, 0, 0, 0 };

	CHECK(set.size() == 1);
	CHECK(set.has(0));
}

TEST_CASE("[AHashSet] Insert element") {
	AHashSet<int> set;
	AHashSet<int>::Iterator e = set.insert(42);

	CHECK(e);
	CHECK(*e == 42);
	CHECK(set.has(42));
	CHECK(set.find(42));
	set.reset();
}

TEST_CASE("[AHashSet] Insert existing element") {
	AHashSet<int> set;
	set.insert(42);
	set.insert(42);

	CHECK(set.has(42));
	CHECK(set.size() == 1);
}

TEST_CASE("[AHashSet] Insert, iterate and remove many elements") {
	const int elem_max = 1234;
	AHashSet<int> set;
	for (int i = 0; i < elem_max; i++) {
		set.insert(i);
	}

	//insert order should have been kept
	int idx = 0;
	for (const int &K : set) {
		CHECK(idx == K);
		CHECK(set.has(idx));
		idx++;
	}

	Vector<int> elems_still_valid;

	for (int i = 0; i < elem_max; i++) {
		if ((i % 5) == 0) {
			set.erase(i);
		} else {
			elems_still_valid.push_back(i);
		}
	}

	CHECK(elems_still_valid.size() == set.size());

	for (int i = 0; i < elems_still_valid.size(); i++) {
		CHECK(set.has(elems_still_valid[i]));
	}
}

TEST_CASE("[AHashSet] Insert, iterate and remove many strings") {
	// This tests a key that uses allocation, to see if any leaks occur

	uint64_t pre_mem = Memory::get_mem_usage();
	const int elem_max = 432;
	AHashSet<String> set;

	// To not print WARNING: Excessive collision count (NN), is the right hash function being used?
	ERR_PRINT_OFF;
	for (int i = 0; i < elem_max; i++) {
		set.insert(itos(i));
	}
	ERR_PRINT_ON;

	//insert order should have been kept
	int idx = 0;
	for (const String &K : set) {
		CHECK(itos(idx) == K);
		CHECK(set.has(itos(idx)));
		idx++;
	}

	Vector<String> elems_still_valid;

	for (int i = 0; i < elem_max; i++) {
		if ((i % 5) == 0) {
			set.erase(itos(i));
		} else {
			elems_still_valid.push_back(itos(i));
		}
	}

	CHECK(elems_still_valid.size() == set.size());

	for (int i = 0; i < elems_still_valid.size(); i++) {
		CHECK(set.has(elems_still_valid[i]));
	}

	elems_still_valid.clear();
	set.reset();

	CHECK(Memory::get_mem_usage() == pre_mem);
}

TEST_CASE("[AHashSet] Erase via element") {
	AHashSet<int> set;
	AHashSet<int>::Iterator e = set.insert(42);
	set.remove(e);
	CHECK(!set.has(42));
	CHECK(!set.find(42));
}

TEST_CASE("[AHashSet] Erase via key") {
	AHashSet<int> set;
	set.insert(42);
	set.insert(49);
	set.erase(42);
	CHECK(!set.has(42));
	CHECK(!set.find(42));
}

TEST_CASE("[AHashSet] Insert and erase half elements") {
	AHashSet<int> set;
	set.insert(1);
	set.insert(2);
	set.insert(3);
	set.insert(4);
	set.erase(1);
	set.erase(3);

	CHECK(set.size() == 2);
	CHECK(set.has(2));
	CHECK(set.has(4));
}

TEST_CASE("[AHashSet] Size") {
	AHashSet<int> set;
	set.insert(42);
	set.insert(123);
	set.insert(123);
	set.insert(0);
	set.insert(123485);

	CHECK(set.size() == 4);
}

TEST_CASE("[AHashSet] Iteration") {
	AHashSet<int> set;
	set.insert(42);
	set.insert(123);
	set.insert(0);
	set.insert(123485);

	Vector<int> expected;
	expected.push_back(42);
	expected.push_back(123);
	expected.push_back(0);
	expected.push_back(123485);

	int idx = 0;
	for (const int &E : set) {
		CHECK(expected[idx] == E);
		++idx;
	}
}

TEST_CASE("[AHashSet] Copy") {
	AHashSet<int> set;
	set.insert(42);
	set.insert(123);
	set.insert(0);
	set.insert(123485);

	Vector<int> expected;
	expected.push_back(42);
	expected.push_back(123);
	expected.push_back(0);
	expected.push_back(123485);

	AHashSet<int> copy_assign = set;

	int idx = 0;
	for (const int &E : copy_assign) {
		CHECK(expected[idx] == E);
		++idx;
	}

	AHashSet<int> copy_construct(set);

	idx = 0;
	for (const int &E : copy_construct) {
		CHECK(expected[idx] == E);
		++idx;
	}
}

TEST_CASE("[AHashSet] Equality") {
	// Empty sets.
	CHECK(AHashSet<int>{} == AHashSet<int>{});
	CHECK(AHashSet<int>{} != AHashSet<int>{ 1, 2, 3 });
	CHECK(AHashSet<int>{ 1, 2, 3 } != AHashSet<int>{});

	// Different length.
	CHECK(AHashSet<int>{ 1, 2, 3 } != AHashSet<int>{ 1, 2, 3, 4 });
	CHECK(AHashSet<int>{ 1, 2, 3, 4 } != AHashSet<int>{ 4, 3, 2 });

	// Same length.
	CHECK(AHashSet<int>{ 1, 2, 3 } == AHashSet<int>{ 1, 2, 3 });
	CHECK(AHashSet<int>{ 1, 2, 3 } == AHashSet<int>{ 3, 2, 1 });
	CHECK(AHashSet<int>{ 1, 2, 3 } != AHashSet<int>{ 1, 2, 8 });
}

} // namespace TestAHashSet
