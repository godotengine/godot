/*************************************************************************/
/*  test_oa_hash_map.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/os/os.h"

#include "core/oa_hash_map.h"

namespace TestOAHashMap {

MainLoop *test() {

	OS::get_singleton()->print("\n\n\nHello from test\n");

	// test element tracking.
	{
		OAHashMap<int, int> map;

		map.set(42, 1337);
		map.set(1337, 21);
		map.set(42, 11880);

		int value;
		map.lookup(42, &value);

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
			if (map.lookup(i, &tmp))
				num_elems++;
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
			OS::get_singleton()->print("map[\"%s\"] = %d\n", it.key->utf8().get_data(), *it.data);
		}
	}

	return NULL;
}
} // namespace TestOAHashMap
