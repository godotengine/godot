/**************************************************************************/
/*  test_bit_set.h                                                        */
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

#ifndef TEST_BIT_SET_H
#define TEST_BIT_SET_H

#include "core/templates/bit_set.h"

#include "tests/test_macros.h"

namespace TestBitSet {

TEST_CASE("[BitSet] Set elements") {
	BitSet set;
	set.resize(6);
	set.set(0, true);
	set.set(3, true);
	set.set(5, true);

	CHECK(set.size() == 6);
	CHECK(set.get(0) == true);
	CHECK(set.get(1) == false);
	CHECK(set.get(2) == false);
	CHECK(set.get(3) == true);
	CHECK(set.get(4) == false);
	CHECK(set.get(5) == true);

	set.set(0, false);
	set.set(1, true);
	set.set(3, false);
	set.set(4, true);

	CHECK(set.get(0) == false);
	CHECK(set.get(1) == true);
	CHECK(set.get(2) == false);
	CHECK(set.get(3) == false);
	CHECK(set.get(4) == true);
	CHECK(set.get(5) == true);
}

TEST_CASE("[BitSet] Clear and resize defaults") {
	BitSet set;
	set.resize(3);
	set.set(0, true);
	set.set(1, true);
	set.set(2, true);
	set.clear();

	CHECK(set.size() == 0);

	// After resizing, all values should be defaulted to zero.
	set.resize(3);

	CHECK(set.size() == 3);
	CHECK(set.get(0) == false);
	CHECK(set.get(1) == false);
	CHECK(set.get(2) == false);

	// Special case where the bit set should've cleared the individual bit to simulate
	// the behavior a vector does when resizing to a bigger size and defaulting to zero,
	// even if the contents might've been set before.
	set.set(0, true);
	set.set(1, true);
	set.set(2, true);
	set.resize(2);
	set.resize(12);

	CHECK(set.get(0) == true);
	CHECK(set.get(1) == true);
	CHECK(set.get(2) == false);
	CHECK(set.get(11) == false);
}

TEST_CASE("[BitSet] Copy") {
	BitSet set;
	set.resize(3);
	set.set(0, true);
	set.set(1, false);
	set.set(2, true);

	BitSet set_copy = set;
	CHECK(set_copy.size() == 3);
	CHECK(set_copy.get(0) == true);
	CHECK(set_copy.get(1) == false);
	CHECK(set_copy.get(2) == true);
}

TEST_CASE("[BitSet] Large set") {
	uint32_t large_count = 10000;
	BitSet set;
	set.resize(large_count);
	set.set(100, true);
	set.set(350, true);
	set.set(5000, true);
	set.set(1234, true);

	CHECK(set.get(100) == true);
	CHECK(set.get(350) == true);
	CHECK(set.get(5000) == true);
	CHECK(set.get(1234) == true);
}
} // namespace TestBitSet

#endif // TEST_BIT_SET_H
