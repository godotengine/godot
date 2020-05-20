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

#include "test_oa_hash_map.h"

#include "core/oa_hash_map.h"
#include "core/os/os.h"

namespace TestOAHashMap {

struct CountedItem {
	static int count;

	int id = -1;
	bool destroyed = false;

	CountedItem() {
		count++;
	}

	CountedItem(int p_id) :
			id(p_id) {
		count++;
	}

	CountedItem(const CountedItem &p_other) :
			id(p_other.id) {
		count++;
	}

	CountedItem &operator=(const CountedItem &p_other) = default;

	~CountedItem() {
		CRASH_COND(destroyed);
		count--;
		destroyed = true;
	}
};

int CountedItem::count;

MainLoop *test() {
	OS::get_singleton()->print("\n\n\nHello from test\n");

	// test element tracking.
	{
		OAHashMap<int, int> map;

		map.set(42, 1337);
		map.set(1337, 21);
		map.set(42, 11880);

		int value = 0;
		map.lookup(42, value);

		OS::get_singleton()->print("capacity  %d\n", map.get_capacity());
		OS::get_singleton()->print("elements  %d\n", map.get_num_elements());

		OS::get_singleton()->print("map[42] = %d\n", value);
	}

	// rehashing and deletion
	{
		OAHashMap<int, int> map;

		for (int i = 0; i < 500; i++) {
			map.set(i, i * 2);
		}

		for (int i = 0; i < 500; i += 2) {
			map.remove(i);
		}

		uint32_t num_elems = 0;
		for (int i = 0; i < 500; i++) {
			int tmp;
			if (map.lookup(i, tmp) && tmp == i * 2) {
				num_elems++;
			}
		}

		OS::get_singleton()->print("elements %d == %d.\n", map.get_num_elements(), num_elems);
	}

	// iteration
	{
		OAHashMap<String, int> map;

		map.set("Hello", 1);
		map.set("World", 2);
		map.set("Godot rocks", 42);

		for (OAHashMap<String, int>::Iterator it = map.iter(); it.valid; it = map.next_iter(it)) {
			OS::get_singleton()->print("map[\"%s\"] = %d\n", it.key->utf8().get_data(), *it.value);
		}
	}

	// stress test / test for issue #22928
	{
		OAHashMap<int, int> map;
		int dummy = 0;
		const int N = 1000;
		uint32_t *keys = new uint32_t[N];

		Math::seed(0);

		// insert a couple of random keys (with a dummy value, which is ignored)
		for (int i = 0; i < N; i++) {
			keys[i] = Math::rand();
			map.set(keys[i], dummy);

			if (!map.lookup(keys[i], dummy)) {
				OS::get_singleton()->print("could not find 0x%X despite it was just inserted!\n", unsigned(keys[i]));
			}
		}

		// check whether the keys are still present
		for (int i = 0; i < N; i++) {
			if (!map.lookup(keys[i], dummy)) {
				OS::get_singleton()->print("could not find 0x%X despite it has been inserted previously! (not checking the other keys, breaking...)\n", unsigned(keys[i]));
				break;
			}
		}

		delete[] keys;
	}

	// regression test / test for issue related to #31402
	{
		OS::get_singleton()->print("test for issue #31402 started...\n");

		const int num_test_values = 12;
		int test_values[num_test_values] = { 0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264 };

		int dummy = 0;
		OAHashMap<int, int> map;
		map.clear();

		for (int i = 0; i < num_test_values; ++i) {
			map.set(test_values[i], dummy);
		}

		OS::get_singleton()->print("test for issue #31402 passed.\n");
	}

	// test collision resolution, should not crash or run indefinitely
	{
		OAHashMap<int, int> map(4);
		map.set(1, 1);
		map.set(5, 1);
		map.set(9, 1);
		map.set(13, 1);
		map.remove(5);
		map.remove(9);
		map.remove(13);
		map.set(5, 1);
	}

	// test memory management of items, should not crash or leak items
	{
		// Exercise different patterns of removal
		for (int i = 0; i < 4; ++i) {
			{
				OAHashMap<String, CountedItem> map;
				int id = 0;
				for (int j = 0; j < 100; ++j) {
					map.insert(itos(j), CountedItem(id));
				}
				if (i <= 1) {
					for (int j = 0; j < 100; ++j) {
						map.remove(itos(j));
					}
				}
				if (i % 2 == 0) {
					map.clear();
				}
			}

			if (CountedItem::count != 0) {
				OS::get_singleton()->print("%d != 0 (not performing the other test sub-cases, breaking...)\n", CountedItem::count);
				break;
			}
		}
	}

	// Test map with 0 capacity.
	{
		OAHashMap<int, String> original_map(0);
		original_map.set(1, "1");
		OS::get_singleton()->print("OAHashMap 0 capacity initialization passed.\n");
	}

	// Test copy constructor.
	{
		OAHashMap<int, String> original_map;
		original_map.set(1, "1");
		original_map.set(2, "2");
		original_map.set(3, "3");
		original_map.set(4, "4");
		original_map.set(5, "5");

		OAHashMap<int, String> map_copy(original_map);

		bool pass = true;
		for (
				OAHashMap<int, String>::Iterator it = original_map.iter();
				it.valid;
				it = original_map.next_iter(it)) {
			if (map_copy.lookup_ptr(*it.key) == nullptr) {
				pass = false;
			}
			if (*it.value != *map_copy.lookup_ptr(*it.key)) {
				pass = false;
			}
		}
		if (pass) {
			OS::get_singleton()->print("OAHashMap copy constructor test passed.\n");
		} else {
			OS::get_singleton()->print("OAHashMap copy constructor test FAILED.\n");
		}

		map_copy.set(1, "Random String");
		if (*map_copy.lookup_ptr(1) == *original_map.lookup_ptr(1)) {
			OS::get_singleton()->print("OAHashMap copy constructor, atomic copy test FAILED.\n");
		} else {
			OS::get_singleton()->print("OAHashMap copy constructor, atomic copy test passed.\n");
		}
	}

	// Test assign operator.
	{
		OAHashMap<int, String> original_map;
		original_map.set(1, "1");
		original_map.set(2, "2");
		original_map.set(3, "3");
		original_map.set(4, "4");
		original_map.set(5, "5");

		OAHashMap<int, String> map_copy(100000);
		map_copy.set(1, "Just a string.");
		map_copy = original_map;

		bool pass = true;
		for (
				OAHashMap<int, String>::Iterator it = map_copy.iter();
				it.valid;
				it = map_copy.next_iter(it)) {
			if (original_map.lookup_ptr(*it.key) == nullptr) {
				pass = false;
			}
			if (*it.value != *original_map.lookup_ptr(*it.key)) {
				pass = false;
			}
		}
		if (pass) {
			OS::get_singleton()->print("OAHashMap assign operation test passed.\n");
		} else {
			OS::get_singleton()->print("OAHashMap assign operation test FAILED.\n");
		}

		map_copy.set(1, "Random String");
		if (*map_copy.lookup_ptr(1) == *original_map.lookup_ptr(1)) {
			OS::get_singleton()->print("OAHashMap assign operation atomic copy test FAILED.\n");
		} else {
			OS::get_singleton()->print("OAHashMap assign operation atomic copy test passed.\n");
		}
	}

	return nullptr;
}

} // namespace TestOAHashMap
