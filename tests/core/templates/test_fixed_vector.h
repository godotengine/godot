/**************************************************************************/
/*  test_fixed_vector.h                                                   */
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

#include "core/templates/fixed_vector.h"

#include "tests/test_macros.h"

namespace TestFixedVector {

struct MoveOnly {
	bool is_alive = true;

	MoveOnly() = default;
	MoveOnly(const MoveOnly &p) = delete;
	MoveOnly &operator=(const MoveOnly &p) = delete;

	MoveOnly(MoveOnly &&p) {
		is_alive = p.is_alive;
		p.is_alive = false;
	}
	MoveOnly &operator=(MoveOnly &&p) {
		if (&p == this) {
			return *this;
		}
		is_alive = p.is_alive;
		p.is_alive = false;
		return *this;
	}
};

static_assert(!std::is_trivially_copyable_v<MoveOnly>);
static_assert(!std::is_trivially_constructible_v<MoveOnly>);

TEST_CASE("[FixedVector] Basic Checks") {
	FixedVector<uint16_t, 1> vector;
	CHECK_EQ(vector.capacity(), 1);

	CHECK_EQ(vector.size(), 0);
	CHECK(vector.is_empty());
	CHECK(!vector.is_full());

	vector.push_back(5);
	CHECK_EQ(vector.size(), 1);
	CHECK_EQ(vector[0], 5);
	CHECK_EQ(vector.ptr()[0], 5);
	CHECK(!vector.is_empty());
	CHECK(vector.is_full());

	vector.pop_back();
	CHECK_EQ(vector.size(), 0);
	CHECK(vector.is_empty());
	CHECK(!vector.is_full());

	FixedVector<uint16_t, 2> vector1 = { 1, 2 };
	CHECK_EQ(vector1.capacity(), 2);
	CHECK_EQ(vector1.size(), 2);
	CHECK_EQ(vector1[0], 1);
	CHECK_EQ(vector1[1], 2);

	FixedVector<Variant, 3> vector_variant;
	CHECK_EQ(vector_variant.size(), 0);
	CHECK_EQ(vector_variant.capacity(), 3);
	vector_variant.resize_initialized(3);
	vector_variant[0] = "Test";
	vector_variant[1] = 1;
	CHECK_EQ(vector_variant.capacity(), 3);
	CHECK_EQ(vector_variant.size(), 3);
	CHECK_EQ(vector_variant[0], "Test");
	CHECK_EQ(vector_variant[1], Variant(1));
	CHECK_EQ(vector_variant[2].get_type(), Variant::NIL);

	// Test that move-only types are transferred.
	FixedVector<MoveOnly, 1> a;
	a.push_back(MoveOnly());
	CHECK_EQ(a.size(), 1);
	FixedVector<MoveOnly, 1> b(std::move(a));
	CHECK_EQ(a.size(), 0);
	CHECK_EQ(b.size(), 1);
	a = std::move(b);
	CHECK_EQ(a.size(), 1);
	CHECK_EQ(b.size(), 0);
}

TEST_CASE("[FixedVector] Alignment Checks") {
	FixedVector<uint16_t, 4> vector_uint16;
	vector_uint16.resize_uninitialized(4);
	CHECK((size_t)&vector_uint16[0] % alignof(uint16_t) == 0);
	CHECK((size_t)&vector_uint16[1] % alignof(uint16_t) == 0);
	CHECK((size_t)&vector_uint16[2] % alignof(uint16_t) == 0);
	CHECK((size_t)&vector_uint16[3] % alignof(uint16_t) == 0);

	FixedVector<uint32_t, 4> vector_uint32;
	vector_uint32.resize_uninitialized(4);
	CHECK((size_t)&vector_uint32[0] % alignof(uint32_t) == 0);
	CHECK((size_t)&vector_uint32[1] % alignof(uint32_t) == 0);
	CHECK((size_t)&vector_uint32[2] % alignof(uint32_t) == 0);
	CHECK((size_t)&vector_uint32[3] % alignof(uint32_t) == 0);

	FixedVector<uint64_t, 4> vector_uint64;
	vector_uint64.resize_uninitialized(4);
	CHECK((size_t)&vector_uint64[0] % alignof(uint64_t) == 0);
	CHECK((size_t)&vector_uint64[1] % alignof(uint64_t) == 0);
	CHECK((size_t)&vector_uint64[2] % alignof(uint64_t) == 0);
	CHECK((size_t)&vector_uint64[3] % alignof(uint64_t) == 0);
}

} //namespace TestFixedVector
