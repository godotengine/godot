/*************************************************************************/
/*  test_dictionary.h                                                    */
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

#ifndef TEST_DICTIONARY_H
#define TEST_DICTIONARY_H

#include "core/templates/ordered_hash_map.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "tests/test_macros.h"

namespace TestDictionary {

TEST_CASE("[Dictionary] Assignment using bracket notation ([])") {
	Dictionary map;
	map["Hello"] = 0;
	CHECK(int(map["Hello"]) == 0);
	map["Hello"] = 3;
	CHECK(int(map["Hello"]) == 3);
	map["World!"] = 4;
	CHECK(int(map["World!"]) == 4);

	// Test non-string keys, since keys can be of any Variant type.
	map[12345] = -5;
	CHECK(int(map[12345]) == -5);
	map[false] = 128;
	CHECK(int(map[false]) == 128);
	map[Vector2(10, 20)] = 30;
	CHECK(int(map[Vector2(10, 20)]) == 30);
	map[0] = 400;
	CHECK(int(map[0]) == 400);
	// Check that assigning 0 doesn't overwrite the value for `false`.
	CHECK(int(map[false]) == 128);
}

TEST_CASE("[Dictionary] == and != operators") {
	Dictionary map1;
	Dictionary map2;
	CHECK(map1 != map2);
	map1[1] = 3;
	map2 = map1;
	CHECK(map1 == map2);
}

TEST_CASE("[Dictionary] get_key_lists()") {
	Dictionary map;
	List<Variant> keys;
	List<Variant> *ptr = &keys;
	map.get_key_list(ptr);
	CHECK(keys.is_empty());
	map[1] = 3;
	map.get_key_list(ptr);
	CHECK(keys.size() == 1);
	CHECK(int(keys[0]) == 1);
	map[2] = 4;
	map.get_key_list(ptr);
	CHECK(keys.size() == 3);
}

TEST_CASE("[Dictionary] get_key_at_index()") {
	Dictionary map;
	map[4] = 3;
	Variant val = map.get_key_at_index(0);
	CHECK(int(val) == 4);
	map[3] = 1;
	val = map.get_key_at_index(0);
	CHECK(int(val) == 4);
	val = map.get_key_at_index(1);
	CHECK(int(val) == 3);
}

TEST_CASE("[Dictionary] getptr()") {
	Dictionary map;
	map[1] = 3;
	Variant *key = map.getptr(1);
	CHECK(int(*key) == 3);
	key = map.getptr(2);
	CHECK(key == nullptr);
}

TEST_CASE("[Dictionary] get_valid()") {
	Dictionary map;
	map[1] = 3;
	Variant val = map.get_valid(1);
	CHECK(int(val) == 3);
}
TEST_CASE("[Dictionary] get()") {
	Dictionary map;
	map[1] = 3;
	Variant val = map.get(1, -1);
	CHECK(int(val) == 3);
}

TEST_CASE("[Dictionary] size(), empty() and clear()") {
	Dictionary map;
	CHECK(map.size() == 0);
	CHECK(map.is_empty());
	map[1] = 3;
	CHECK(map.size() == 1);
	CHECK(!map.is_empty());
	map.clear();
	CHECK(map.size() == 0);
	CHECK(map.is_empty());
}

TEST_CASE("[Dictionary] has() and has_all()") {
	Dictionary map;
	CHECK(map.has(1) == false);
	map[1] = 3;
	CHECK(map.has(1));
	Array keys;
	keys.push_back(1);
	CHECK(map.has_all(keys));
	keys.push_back(2);
	CHECK(map.has_all(keys) == false);
}

TEST_CASE("[Dictionary] keys() and values()") {
	Dictionary map;
	Array keys = map.keys();
	Array values = map.values();
	CHECK(keys.is_empty());
	CHECK(values.is_empty());
	map[1] = 3;
	keys = map.keys();
	values = map.values();
	CHECK(int(keys[0]) == 1);
	CHECK(int(values[0]) == 3);
}
} // namespace TestDictionary
#endif // TEST_DICTIONARY_H
