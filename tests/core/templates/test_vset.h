/**************************************************************************/
/*  test_vset.h                                                           */
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

#include "core/templates/vset.h"

#include "tests/test_macros.h"

namespace TestVSet {

template <typename T>
class TestClass : public VSet<T> {
public:
	int _find(const T &p_val, bool &r_exact) const {
		return VSet<T>::_find(p_val, r_exact);
	}
};

TEST_CASE("[VSet] _find and _find_exact correctness.") {
	TestClass<int> set;

	// insert some values
	set.insert(10);
	set.insert(20);
	set.insert(30);
	set.insert(40);
	set.insert(50);

	// data should be sorted
	CHECK(set.size() == 5);
	CHECK(set[0] == 10);
	CHECK(set[1] == 20);
	CHECK(set[2] == 30);
	CHECK(set[3] == 40);
	CHECK(set[4] == 50);

	// _find_exact return exact position for existing elements
	CHECK(set.find(10) == 0);
	CHECK(set.find(30) == 2);
	CHECK(set.find(50) == 4);

	// _find_exact return -1 for non-existing elements
	CHECK(set.find(15) == -1);
	CHECK(set.find(0) == -1);
	CHECK(set.find(60) == -1);

	// test _find
	bool exact;

	// existing elements
	CHECK(set._find(10, exact) == 0);
	CHECK(exact == true);

	CHECK(set._find(30, exact) == 2);
	CHECK(exact == true);

	// non-existing elements
	CHECK(set._find(25, exact) == 2);
	CHECK(exact == false);

	CHECK(set._find(35, exact) == 3);
	CHECK(exact == false);

	CHECK(set._find(5, exact) == 0);
	CHECK(exact == false);

	CHECK(set._find(60, exact) == 5);
	CHECK(exact == false);
}

} // namespace TestVSet
