/*************************************************************************/
/*  test_oa_hash_map.cpp                                                 */
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

#include "test_dictionary.h"
#include "tests/test_macros.h"


#include "core/variant/dictionary.h"
#include "core/templates/ordered_hash_map.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/variant.h"

namespace TestDictionary {



MainLoop *test() {
	// test get_key_list.
	{
		Dictionary map;
        List<Variant> keys;
        auto ptr = &keys;
        map.get_key_list(ptr);
        CHECK(keys.empty());
        map[1] = 3;
        map.get_key_list(ptr);
        CHECK(keys.size() == 1);
        CHECK(int(keys[0]) == 1);
        map[2] = 4;
        map.get_key_list(ptr);
        CHECK(keys.size() == 3);
	}
	// test get_key_at_index
	{
		Dictionary map;
        map[4] = 3;
        auto val = map.get_key_at_index(0);
        CHECK(int(val) == 4);
        map[3] = 1;
        val = map.get_key_at_index(0);
        CHECK(int(val) == 3);
        val = map.get_key_at_index(1);
        CHECK(int(val) == 4);
	}
	// test [] assignment
	{
		Dictionary map;

        map["Hello"] = 0;
        CHECK(int(map["Hello"]) == 0);
        map["Hello"] = 3;
        CHECK(int(map["Hello"]) == 3);
        map["World!"] = 4;
        CHECK(int(map["World"]) == 4);
	}
	// test getptr
	{
		Dictionary map;
        map[1] = 3;
        auto key = map.getptr(1);
        CHECK(int(*key) == 3);
        key = map.getptr(2);
        CHECK(key == nullptr);
	}
	// test get_valid
	{
        Dictionary map;
        map[1] = 3;
        auto val = map.get_valid(1);
        CHECK(int(val) == 3);
	}
	// test get
	{
		Dictionary map;
        map[1] = 3;
        auto val = map.get(1, -1);
        CHECK(int(val) == 3);
	}
	// test size and empty and clear
	{
        Dictionary map;
        CHECK(map.size() == 0);
        CHECK(map.empty());
        map[1] = 3;
        CHECK(map.size() == 1);
        CHECK(!map.empty());
        map.clear();
        CHECK(map.size() == 0);
        CHECK(map.empty());
	}
	// test has and has_all
	{
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
	// test == and !=
	{
		Dictionary map1;
        Dictionary map2;
        CHECK(map1 == map2);
        map1[1] = 3;
        CHECK(map1 != map2);
        map2[1] = 3;
        CHECK(map1 == map2);
	}
	// test keys and values
	{
        Dictionary map;
        auto keys = map.keys();
        auto values = map.values();
        CHECK(keys.empty());
        CHECK(values.empty());
        map[1] = 3;
        CHECK(int(keys[0]) == 1);
        CHECK(int(values[0]) == 3);
	}
	return nullptr;
}
} // namespace TestDictionary
