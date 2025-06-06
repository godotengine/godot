/**************************************************************************/
/*  test_fixed_array.h                                                    */
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

#include "core/templates/fixed_array.h"

#include "tests/test_macros.h"

namespace TestFixedArray {

struct MoveOnly {
	MoveOnly() = default;
	MoveOnly(const MoveOnly &p) = delete;
	MoveOnly(MoveOnly &&p) = default;
	MoveOnly &operator=(const MoveOnly &p) = delete;
	MoveOnly &operator=(MoveOnly &&p) = default;
};

TEST_CASE("[FixedArray] Basic Checks") {
	FixedArray<uint16_t, 3> vector = { 1, 2, 3 };
	CHECK_EQ(vector.size(), 3);
	CHECK_EQ(vector[0], 1);
	CHECK_EQ(vector[1], 2);
	CHECK_EQ(vector[2], 3);

	FixedArray<uint16_t, 3> vector1 = vector;
	CHECK_EQ(vector1.size(), 3);
	CHECK_EQ(vector1[0], 1);
	CHECK_EQ(vector1[1], 2);
	CHECK_EQ(vector1[2], 3);

	// Test that move-only types can be used.
	FixedArray<MoveOnly, 3> a;
	FixedArray<MoveOnly, 3> b(a);
	a = std::move(b);
}

TEST_CASE("[FixedArray] Alignment Checks") {
	FixedArray<uint16_t, 4> vector_uint16;
	CHECK((size_t)&vector_uint16[0] % alignof(uint16_t) == 0);
	CHECK((size_t)&vector_uint16[1] % alignof(uint16_t) == 0);
	CHECK((size_t)&vector_uint16[2] % alignof(uint16_t) == 0);
	CHECK((size_t)&vector_uint16[3] % alignof(uint16_t) == 0);

	FixedArray<uint32_t, 4> vector_uint32;
	CHECK((size_t)&vector_uint32[0] % alignof(uint32_t) == 0);
	CHECK((size_t)&vector_uint32[1] % alignof(uint32_t) == 0);
	CHECK((size_t)&vector_uint32[2] % alignof(uint32_t) == 0);
	CHECK((size_t)&vector_uint32[3] % alignof(uint32_t) == 0);

	FixedArray<uint64_t, 4> vector_uint64;
	CHECK((size_t)&vector_uint64[0] % alignof(uint64_t) == 0);
	CHECK((size_t)&vector_uint64[1] % alignof(uint64_t) == 0);
	CHECK((size_t)&vector_uint64[2] % alignof(uint64_t) == 0);
	CHECK((size_t)&vector_uint64[3] % alignof(uint64_t) == 0);
}

} //namespace TestFixedArray
