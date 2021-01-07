/*************************************************************************/
/*  test_lru.h                                                           */
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

#ifndef TEST_LRU_H
#define TEST_LRU_H

#include "core/templates/lru.h"
#include "core/templates/vector.h"

#include "tests/test_macros.h"

namespace TestLRU {

TEST_CASE("[LRU] Store and read") {
	LRUCache<int, int> lru;

	lru.set_capacity(3);
	lru.insert(1, 1);
	lru.insert(50, 2);
	lru.insert(100, 5);

	CHECK(lru.has(1));
	CHECK(lru.has(50));
	CHECK(lru.has(100));
	CHECK(!lru.has(200));

	CHECK(lru.get(1) == 1);
	CHECK(lru.get(50) == 2);
	CHECK(lru.get(100) == 5);

	CHECK(lru.getptr(1) != nullptr);
	CHECK(lru.getptr(1000) == nullptr);

	lru.insert(600, 600); // Erase <50>
	CHECK(lru.has(600));
	CHECK(!lru.has(50));
}

TEST_CASE("[LRU] Resize and clear") {
	LRUCache<int, int> lru;

	lru.set_capacity(3);
	lru.insert(1, 1);
	lru.insert(2, 2);
	lru.insert(3, 3);

	CHECK(lru.get_capacity() == 3);

	lru.set_capacity(5);
	CHECK(lru.get_capacity() == 5);

	CHECK(lru.has(1));
	CHECK(lru.has(2));
	CHECK(lru.has(3));
	CHECK(!lru.has(4));

	lru.set_capacity(2);
	CHECK(lru.get_capacity() == 2);

	CHECK(!lru.has(1));
	CHECK(lru.has(2));
	CHECK(lru.has(3));
	CHECK(!lru.has(4));

	lru.clear();
	CHECK(!lru.has(1));
	CHECK(!lru.has(2));
	CHECK(!lru.has(3));
	CHECK(!lru.has(4));
}
} // namespace TestLRU

#endif // TEST_LRU_H
