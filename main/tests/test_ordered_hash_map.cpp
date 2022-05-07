/*************************************************************************/
/*  test_ordered_hash_map.cpp                                            */
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

#include "test_ordered_hash_map.h"

#include "core/ordered_hash_map.h"
#include "core/os/os.h"
#include "core/pair.h"
#include "core/vector.h"

namespace TestOrderedHashMap {

bool test_insert() {
	OrderedHashMap<int, int> map;
	OrderedHashMap<int, int>::Element e = map.insert(42, 84);

	return e && e.key() == 42 && e.get() == 84 && e.value() == 84 && map[42] == 84 && map.has(42) && map.find(42);
}

bool test_insert_overwrite() {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(42, 1234);

	return map[42] == 1234;
}

bool test_erase_via_element() {
	OrderedHashMap<int, int> map;
	OrderedHashMap<int, int>::Element e = map.insert(42, 84);

	map.erase(e);
	return !e && !map.has(42) && !map.find(42);
}

bool test_erase_via_key() {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.erase(42);
	return !map.has(42) && !map.find(42);
}

bool test_size() {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 84);
	map.insert(123, 84);
	map.insert(0, 84);
	map.insert(123485, 84);

	return map.size() == 4;
}

bool test_iteration() {
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
		if (expected[idx] != Pair<int, int>(E.key(), E.value())) {
			return false;
		}
		++idx;
	}
	return true;
}

bool test_const_iteration(const OrderedHashMap<int, int> &map) {
	Vector<Pair<int, int>> expected;
	expected.push_back(Pair<int, int>(42, 84));
	expected.push_back(Pair<int, int>(123, 111111));
	expected.push_back(Pair<int, int>(0, 12934));
	expected.push_back(Pair<int, int>(123485, 1238888));

	int idx = 0;
	for (OrderedHashMap<int, int>::ConstElement E = map.front(); E; E = E.next()) {
		if (expected[idx] != Pair<int, int>(E.key(), E.value())) {
			return false;
		}
		++idx;
	}
	return true;
}

bool test_const_iteration() {
	OrderedHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);
	map.insert(123485, 1238888);
	map.insert(123, 111111);

	return test_const_iteration(map);
}

typedef bool (*TestFunc)();

TestFunc test_funcs[] = {

	test_insert,
	test_insert_overwrite,
	test_erase_via_element,
	test_erase_via_key,
	test_size,
	test_iteration,
	test_const_iteration,
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
} // namespace TestOrderedHashMap
