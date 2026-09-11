/**************************************************************************/
/*  test_bit_field.cpp                                                    */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_bit_field)

#include "core/templates/bit_field.h"

namespace TestBitField {

enum TestFlags {
	FLAG_NONE = 0,
	FLAG_A = 1,
	FLAG_B = 2,
	FLAG_C = 4,
	FLAG_AB = FLAG_A | FLAG_B,
	FLAG_ALL = FLAG_A | FLAG_B | FLAG_C,
};

TEST_CASE("[BitField] Default construction is empty") {
	BitField<TestFlags> bf;
	CHECK_MESSAGE(
			bf.is_empty(),
			"Default-constructed BitField should be empty.");
}

TEST_CASE("[BitField] Construction from enum value") {
	BitField<TestFlags> bf(FLAG_A);
	CHECK_MESSAGE(
			!bf.is_empty(),
			"BitField constructed with a flag should not be empty.");
	CHECK_MESSAGE(
			bf.has_flag(FLAG_A),
			"BitField should have the flag it was constructed with.");
}

TEST_CASE("[BitField] Construction from integer") {
	BitField<TestFlags> bf(3); // FLAG_A | FLAG_B
	CHECK_MESSAGE(
			bf.has_flag(FLAG_A),
			"BitField constructed with integer 3 should have FLAG_A.");
	CHECK_MESSAGE(
			bf.has_flag(FLAG_B),
			"BitField constructed with integer 3 should have FLAG_B.");
	CHECK_MESSAGE(
			!bf.has_flag(FLAG_C),
			"BitField constructed with integer 3 should not have FLAG_C.");
}

TEST_CASE("[BitField] Set single flag") {
	BitField<TestFlags> bf;
	bf.set_flag(FLAG_B);
	CHECK_MESSAGE(
			bf.has_flag(FLAG_B),
			"FLAG_B should be set after set_flag.");
	CHECK_MESSAGE(
			!bf.has_flag(FLAG_A),
			"FLAG_A should not be set.");
	CHECK_MESSAGE(
			!bf.has_flag(FLAG_C),
			"FLAG_C should not be set.");
}

TEST_CASE("[BitField] Set multiple flags") {
	BitField<TestFlags> bf;
	bf.set_flag(FLAG_A);
	bf.set_flag(FLAG_C);
	CHECK_MESSAGE(
			bf.has_flag(FLAG_A),
			"FLAG_A should be set.");
	CHECK_MESSAGE(
			bf.has_flag(FLAG_C),
			"FLAG_C should be set.");
	CHECK_MESSAGE(
			!bf.has_flag(FLAG_B),
			"FLAG_B should not be set.");
}

TEST_CASE("[BitField] Set same flag twice is idempotent") {
	BitField<TestFlags> bf;
	bf.set_flag(FLAG_A);
	bf.set_flag(FLAG_A);
	CHECK_MESSAGE(
			bf.has_flag(FLAG_A),
			"FLAG_A should still be set after setting it twice.");
}

TEST_CASE("[BitField] Clear a set flag") {
	BitField<TestFlags> bf(FLAG_AB);
	bf.clear_flag(FLAG_A);
	CHECK_MESSAGE(
			!bf.has_flag(FLAG_A),
			"FLAG_A should be cleared.");
	CHECK_MESSAGE(
			bf.has_flag(FLAG_B),
			"FLAG_B should still be set.");
}

TEST_CASE("[BitField] Clear unset flag is no-op") {
	BitField<TestFlags> bf(FLAG_A);
	bf.clear_flag(FLAG_B);
	CHECK_MESSAGE(
			bf.has_flag(FLAG_A),
			"FLAG_A should still be set after clearing an unset flag.");
	CHECK_MESSAGE(
			!bf.has_flag(FLAG_B),
			"FLAG_B should remain unset.");
}

TEST_CASE("[BitField] Clear all") {
	BitField<TestFlags> bf(FLAG_ALL);
	CHECK_MESSAGE(
			!bf.is_empty(),
			"BitField with all flags should not be empty.");
	bf.clear();
	CHECK_MESSAGE(
			bf.is_empty(),
			"BitField should be empty after clear().");
}

TEST_CASE("[BitField] get_combined (OR)") {
	BitField<TestFlags> a(FLAG_A);
	BitField<TestFlags> b(FLAG_B);
	BitField<TestFlags> combined = a.get_combined(b);
	CHECK_MESSAGE(
			combined.has_flag(FLAG_A),
			"Combined should have FLAG_A.");
	CHECK_MESSAGE(
			combined.has_flag(FLAG_B),
			"Combined should have FLAG_B.");
	CHECK_MESSAGE(
			!combined.has_flag(FLAG_C),
			"Combined should not have FLAG_C.");
}

TEST_CASE("[BitField] get_shared (AND)") {
	BitField<TestFlags> a(FLAG_AB);
	BitField<TestFlags> b(FLAG_A);
	BitField<TestFlags> shared = a.get_shared(b);
	CHECK_MESSAGE(
			shared.has_flag(FLAG_A),
			"Shared should have FLAG_A (common to both).");
	CHECK_MESSAGE(
			!shared.has_flag(FLAG_B),
			"Shared should not have FLAG_B (only in a).");
}

TEST_CASE("[BitField] get_different (XOR)") {
	BitField<TestFlags> a(FLAG_AB);
	BitField<TestFlags> b(FLAG_A);
	BitField<TestFlags> diff = a.get_different(b);
	CHECK_MESSAGE(
			!diff.has_flag(FLAG_A),
			"XOR should not have FLAG_A (common to both).");
	CHECK_MESSAGE(
			diff.has_flag(FLAG_B),
			"XOR should have FLAG_B (only in a).");
}

TEST_CASE("[BitField] get_combined with empty") {
	BitField<TestFlags> a(FLAG_A);
	BitField<TestFlags> empty;
	BitField<TestFlags> combined = a.get_combined(empty);
	CHECK_MESSAGE(
			combined.has_flag(FLAG_A),
			"Combining with empty should preserve FLAG_A.");
	CHECK_MESSAGE(
			!combined.has_flag(FLAG_B),
			"Combining with empty should not introduce other flags.");
}

TEST_CASE("[BitField] get_shared with empty") {
	BitField<TestFlags> a(FLAG_A);
	BitField<TestFlags> empty;
	BitField<TestFlags> shared = a.get_shared(empty);
	CHECK_MESSAGE(
			shared.is_empty(),
			"Intersection with empty should be empty.");
}

TEST_CASE("[BitField] Cast to enum type") {
	BitField<TestFlags> bf(FLAG_AB);
	TestFlags result = static_cast<TestFlags>(bf);
	CHECK_MESSAGE(
			result == FLAG_AB,
			"Cast to enum should return the combined flag value.");
}

TEST_CASE("[BitField] Explicit cast to integer") {
	BitField<TestFlags> bf(FLAG_ALL);
	uint64_t result = static_cast<uint64_t>(bf);
	CHECK_MESSAGE(
			result == 7,
			"Explicit cast to uint64_t should return 7 (1|2|4).");
}

} // namespace TestBitField
