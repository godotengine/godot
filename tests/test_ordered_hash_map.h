/*************************************************************************/
/*  test_ordered_hash_map.h                                              */
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

#ifndef TEST_ORDERED_HASH_MAP_H
#define TEST_ORDERED_HASH_MAP_H

#include "core/ordered_hash_map.h"
#include "core/os/os.h"
#include "core/pair.h"
#include "core/vector.h"

#include "tests/test_macros.h"

namespace TestOrderedHashMap {

TEST_CASE("[OrderedHashMap] Insert element") {
	OrderedHashMap<int, int> map;
	OrderedHashMap<int, int>::Element e = map.insert(42, 84);

	CHECK(e);
	CHECK(e.key() == 42);
	CHECK(e.get() == 84);
	CHECK(e.value() == 84);
	CHECK(map[42] == 84);
	CHECK(map.has(42));
	CHECK(map.find(42));
}

TEST_CASE("[OrderedHashMap] Overwrite element") {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(42, 1234);

	CHECK(map[42] == 1234);
}

TEST_CASE("[OrderedHashMap] Erase via element") {
	OrderedHashMap<int, int> map;
	OrderedHashMap<int, int>::Element e = map.insert(42, 84);

	map.erase(e);
	CHECK(!e);
	CHECK(!map.has(42));
	CHECK(!map.find(42));
}

TEST_CASE("[OrderedHashMap] Erase via key") {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.erase(42);
	CHECK(!map.has(42));
	CHECK(!map.find(42));
}

TEST_CASE("[OrderedHashMap] Size") {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 84);
	map.insert(123, 84);
	map.insert(0, 84);
	map.insert(123485, 84);

	CHECK(map.size() == 4);
}

TEST_CASE("[OrderedHashMap] Iteration") {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);
	map.insert(123485, 1238888);
	map.insert(123, 111111);

	Vector<Pair<int, int>> expected;
	expected.push_back(Pair<int, int>(42, 84));
	expected.push_back(Pair<int, int>(123, 111111));
	expected.push_back(Pair<int, int>(0, 12934));
	expected.push_back(Pair<int, int>(123485, 1238888));

	int idx = 0;
	for (OrderedHashMap<int, int>::Element E = map.front(); E; E = E.next()) {
		CHECK(expected[idx] == Pair<int, int>(E.key(), E.value()));
		++idx;
	}
}

TEST_CASE("[OrderedHashMap] Const iteration") {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);
	map.insert(123485, 1238888);
	map.insert(123, 111111);

	const OrderedHashMap<int, int> const_map = map;

	Vector<Pair<int, int>> expected;
	expected.push_back(Pair<int, int>(42, 84));
	expected.push_back(Pair<int, int>(123, 111111));
	expected.push_back(Pair<int, int>(0, 12934));
	expected.push_back(Pair<int, int>(123485, 1238888));

	int idx = 0;
	for (OrderedHashMap<int, int>::ConstElement E = const_map.front(); E; E = E.next()) {
		CHECK(expected[idx] == Pair<int, int>(E.key(), E.value()));
		++idx;
	}
}

} // namespace TestOrderedHashMap

#endif // TEST_ORDERED_HASH_MAP_H
