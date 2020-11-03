/*************************************************************************/
/*  test_dictionary.h                                                    */
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

#ifndef TEST_DICTIONARY_H
#define TEST_DICTIONARY_H

#include "core/dictionary.h"

#include "tests/test_macros.h"

namespace TestDictionary {

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

TEST_CASE("[Dictionary] Duplicate dictionary") {
	// d = {1: {1: 1}, {2: 2}: [2], [3]: 3}
	Dictionary k2 = build_dictionary(2, 2);
	Array k3 = build_array(3);
	Dictionary d = build_dictionary(1, build_dictionary(1, 1), k2, build_array(2), k3, 3);

	// Deep copy
	Dictionary deep_d = d.duplicate(true);
	CHECK_MESSAGE(deep_d.id() != d.id(), "Should create a new dictionary");
	CHECK_MESSAGE(Dictionary(deep_d[1]).id() != Dictionary(d[1]).id(), "Should clone nested dictionary");
	CHECK_MESSAGE(Array(deep_d[k2]).id() != Array(d[k2]).id(), "Should clone nested array");
	CHECK_EQ(deep_d, d);
	deep_d[0] = 0;
	CHECK_NE(deep_d, d);
	deep_d.erase(0);
	Dictionary(deep_d[1]).operator[](0) = 0;
	CHECK_NE(deep_d, d);
	Dictionary(deep_d[1]).erase(0);
	CHECK_EQ(deep_d, d);
	// Keys should also be copied
	k2[0] = 0;
	CHECK_NE(deep_d, d);
	k2.erase(0);
	CHECK_EQ(deep_d, d);
	k3.push_back(0);
	CHECK_NE(deep_d, d);
	k3.pop_back();
	CHECK_EQ(deep_d, d);

	// Shallow copy
	Dictionary shallow_d = d.duplicate(false);
	CHECK_MESSAGE(shallow_d.id() != d.id(), "Should create a new array");
	CHECK_MESSAGE(Dictionary(shallow_d[1]).id() == Dictionary(d[1]).id(), "Should keep nested dictionary");
	CHECK_MESSAGE(Array(shallow_d[2]).id() == Array(d[2]).id(), "Should keep nested array");
	CHECK_EQ(shallow_d, d);
	shallow_d[0] = 0;
	CHECK_NE(shallow_d, d);
	shallow_d.erase(0);
#if 0 // TODO: recursion in dict key currently is buggy
	// Keys should also be shallowed
	k2[0] = 0;
	CHECK_EQ(shallow_d, d);
	k2.erase(0);
	k3.push_back(0);
	CHECK_EQ(shallow_d, d);
#endif
}

TEST_CASE("[Dictionary] Duplicate recursive dictionary") {
	// Self recursive
	Dictionary d;
	d[1] = d;

	Dictionary d_shallow = d.duplicate(false);
	CHECK_EQ(d, d_shallow);

	// Deep copy of recursive dictionary endup with recursion limit and return
	// an invalid result (multiple nested dictionaries), the point is we should
	// not end up with a segfault and an error log should be printed
	ERR_PRINT_OFF;
	d.duplicate(true);
	ERR_PRINT_ON;

	// Nested recursive
	Dictionary d1;
	Dictionary d2;
	d1[2] = d2;
	d2[1] = d1;

	Dictionary d1_shallow = d1.duplicate(false);
	CHECK_EQ(d1, d1_shallow);

	// Same deep copy issue as above
	ERR_PRINT_OFF;
	d1.duplicate(true);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary teardown will leak memory
	d.clear();
	d1.clear();
	d2.clear();
}

#if 0 // TODO: recursion in dict key is currently buggy
TEST_CASE("[Dictionary] Duplicate recursive dictionary on keys") {
	// Self recursive
	Dictionary d;
	d[d] = d;

	Dictionary d_shallow = d.duplicate(false);
	CHECK_EQ(d, d_shallow);

	// // Deep copy of recursive dictionary endup with recursion limit and return
	// // an invalid result (multiple nested dictionaries), the point is we should
	// // not end up with a segfault and an error log should be printed
	// ERR_PRINT_OFF;
	// d.duplicate(true);
	// ERR_PRINT_ON;

	// // Nested recursive
	// Dictionary d1;
	// Dictionary d2;
	// d1[d2] = d2;
	// d2[d1] = d1;

	// Dictionary d1_shallow = d1.duplicate(false);
	// CHECK_EQ(d1, d1_shallow);

	// // Same deep copy issue as above
	// ERR_PRINT_OFF;
	// d1.duplicate(true);
	// ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary teardown will leak memory
	d.clear();
	d1.clear();
	d2.clear();
}
#endif

TEST_CASE("[Dictionary] Hash dictionary") {
	// d = {1: {1: 1}, {2: 2}: [2], [3]: 3}
	Dictionary k2 = build_dictionary(2, 2);
	Array k3 = build_array(3);
	Dictionary d = build_dictionary(1, build_dictionary(1, 1), k2, build_array(2), k3, 3);
	uint32_t original_hash = d.hash();

	// Modify dict change the hash
	d[0] = 0;
	CHECK_NE(d.hash(), original_hash);
	d.erase(0);
	CHECK_EQ(d.hash(), original_hash);

	// Modify nested item change the hash
	Dictionary(d[1]).operator[](0) = 0;
	CHECK_NE(d.hash(), original_hash);
	Dictionary(d[1]).erase(0);
	Array(d[k2]).push_back(0);
	CHECK_NE(d.hash(), original_hash);
	Array(d[k2]).pop_back();

	// Modify a key change the hash
	k2[0] = 0;
	CHECK_NE(d.hash(), original_hash);
	k2.erase(0);
	CHECK_EQ(d.hash(), original_hash);
	k3.push_back(0);
	CHECK_NE(d.hash(), original_hash);
	k3.pop_back();
	CHECK_EQ(d.hash(), original_hash);

	// Duplication doesn't change the hash
	Dictionary d2 = d.duplicate(true);
	CHECK_EQ(d2.hash(), original_hash);
}

TEST_CASE("[Dictionary] Hash recursive dictionary") {
	Dictionary d;
	d[1] = d;

	// Hash should reach recursion limit, we just make sure this doesn't blow up
	ERR_PRINT_OFF;
	d.hash();
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary teardown will leak memory
	d.clear();
}

#if 0 // TODO: recursion in dict key is currently buggy
TEST_CASE("[Dictionary] Hash recursive dictionary on keys") {
	Dictionary d;
	d[d] = 1;

	// Hash should reach recursion limit, we just make sure this doesn't blow up
	ERR_PRINT_OFF;
	d.hash();
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary teardown will leak memory
	d.clear();
}
#endif

TEST_CASE("[Dictionary] Empty comparison") {
	Dictionary d1;
	Dictionary d2;

	// test both operator== and operator!=
	CHECK_EQ(d1, d2);
	CHECK_FALSE(d1 != d2);
}

TEST_CASE("[Dictionary] Flat comparison") {
	Dictionary d1 = build_dictionary(1, 1);
	Dictionary d2 = build_dictionary(1, 1);
	Dictionary other_d = build_dictionary(2, 1);

	// test both operator== and operator!=
	CHECK_EQ(d1, d1); // compare self
	CHECK_FALSE(d1 != d1);
	CHECK_EQ(d1, d2); // different equivalent arrays
	CHECK_FALSE(d1 != d2);
	CHECK_NE(d1, other_d); // different arrays with different content
	CHECK_FALSE(d1 == other_d);
}

TEST_CASE("[Dictionary] Nested dictionary comparison") {
	// d1 = {1: {2: {3: 4}}}
	Dictionary d1 = build_dictionary(1, build_dictionary(2, build_dictionary(3, 4)));

	Dictionary d2 = d1.duplicate(true);

	// other_d = {1: {2: {3: 0}}}
	Dictionary other_d = build_dictionary(1, build_dictionary(2, build_dictionary(3, 0)));

	// test both operator== and operator!=
	CHECK_EQ(d1, d1); // compare self
	CHECK_FALSE(d1 != d1);
	CHECK_EQ(d1, d2); // different equivalent arrays
	CHECK_FALSE(d1 != d2);
	CHECK_NE(d1, other_d); // different arrays with different content
	CHECK_FALSE(d1 == other_d);
}

TEST_CASE("[Dictionary] Nested array comparison") {
	// d1 = {1: [2, 3]}
	Dictionary d1 = build_dictionary(1, build_array(2, 3));

	Dictionary d2 = d1.duplicate(true);

	// other_d = {1: [2, 0]}
	Dictionary other_d = build_dictionary(1, build_array(2, 0));

	// test both operator== and operator!=
	CHECK_EQ(d1, d1); // compare self
	CHECK_FALSE(d1 != d1);
	CHECK_EQ(d1, d2); // different equivalent arrays
	CHECK_FALSE(d1 != d2);
	CHECK_NE(d1, other_d); // different arrays with different content
	CHECK_FALSE(d1 == other_d);
}

TEST_CASE("[Dictionary] Recursive comparison") {
	Dictionary d1;
	d1[1] = d1;

	Dictionary d2;
	d2[1] = d2;

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_EQ(d1, d2);
	CHECK_FALSE(d1 != d2);
	ERR_PRINT_ON;

	d1[2] = 2;
	d2[2] = 2;

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_EQ(d1, d2);
	CHECK_FALSE(d1 != d2);
	ERR_PRINT_ON;

	d1[3] = 3;
	d2[3] = 0;

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_NE(d1, d2);
	CHECK_FALSE(d1 == d2);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary teardown will leak memory
	d1.clear();
	d2.clear();
}

#if 0 // TODO: recursion in dict key is currently buggy
TEST_CASE("[Dictionary] Recursive comparison on keys") {
	Dictionary d1;
	// Hash computation should reach recursion limit
	ERR_PRINT_OFF;
	d1[d1] = 1;
	ERR_PRINT_ON;

	Dictionary d2;
	// Hash computation should reach recursion limit
	ERR_PRINT_OFF;
	d2[d2] = 1;
	ERR_PRINT_ON;

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_EQ(d1, d2);
	CHECK_FALSE(d1 != d2);
	ERR_PRINT_ON;

	d1[2] = 2;
	d2[2] = 2;

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_EQ(d1, d2);
	CHECK_FALSE(d1 != d2);
	ERR_PRINT_ON;

	d1[3] = 3;
	d2[3] = 0;

	// Comparison should reach recursion limit
	ERR_PRINT_OFF;
	CHECK_NE(d1, d2);
	CHECK_FALSE(d1 == d2);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary teardown will leak memory
	d1.clear();
	d2.clear();
}
#endif

TEST_CASE("[Dictionary] Recursive self comparison") {
	Dictionary d1;
	Dictionary d2;
	d1[1] = d2;
	d2[1] = d1;

	CHECK_EQ(d1, d1);
	CHECK_FALSE(d1 != d1);

	// Break the recursivity otherwise Dictionary teardown will leak memory
	d1.clear();
	d2.clear();
}

} // namespace TestDictionary

#endif // TEST_DICTIONARY_H
