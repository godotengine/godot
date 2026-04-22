/**************************************************************************/
/*  test_swiss_table_simd.cpp                                             */
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

TEST_FORCE_LINK(test_swiss_table_simd)

#include "core/templates/swiss_table_simd.h"

namespace TestSwissTableSIMD {

// Reference: brute-force compute the expected match bitmask, where bit i is
// set if buf[i] satisfies the predicate.
static uint32_t reference_match(const uint8_t *buf, uint32_t width, uint8_t needle) {
	uint32_t out = 0;
	for (uint32_t i = 0; i < width; i++) {
		if (buf[i] == needle) {
			out |= (1u << i);
		}
	}
	return out;
}

static uint32_t reference_match_empty(const uint8_t *buf, uint32_t width) {
	return reference_match(buf, width, SwissTable::kEmpty);
}

static uint32_t reference_match_empty_or_deleted(const uint8_t *buf, uint32_t width) {
	uint32_t out = 0;
	for (uint32_t i = 0; i < width; i++) {
		if ((buf[i] & 0x80) != 0) {
			out |= (1u << i);
		}
	}
	return out;
}

template <typename Group>
static uint32_t mask_to_bits(typename Group::Mask m) {
	uint32_t out = 0;
	auto it = m.iter();
	while (it.has_next()) {
		out |= (1u << it.next());
	}
	return out;
}

template <typename Group>
static void check_group_against_reference(const uint8_t *buf, uint8_t needle) {
	Group g(buf);
	CHECK_EQ(mask_to_bits<Group>(g.match(needle)), reference_match(buf, Group::kWidth, needle));
	CHECK_EQ(mask_to_bits<Group>(g.match_empty()), reference_match_empty(buf, Group::kWidth));
	CHECK_EQ(mask_to_bits<Group>(g.match_empty_or_deleted()), reference_match_empty_or_deleted(buf, Group::kWidth));
}

static void fill_pattern(uint8_t *buf, uint32_t width, uint32_t seed) {
	uint32_t s = seed;
	for (uint32_t i = 0; i < width; i++) {
		s = s * 1664525u + 1013904223u;
		uint32_t bucket = s & 7;
		if (bucket == 0) {
			buf[i] = SwissTable::kEmpty;
		} else if (bucket == 1) {
			buf[i] = SwissTable::kDeleted;
		} else {
			buf[i] = static_cast<uint8_t>((s >> 24) & 0x7F);
		}
	}
}

template <typename Group>
static void exercise_group() {
	alignas(16) uint8_t buf[16] = {};

	// All empty.
	memset(buf, SwissTable::kEmpty, sizeof(buf));
	check_group_against_reference<Group>(buf, 0x00);
	check_group_against_reference<Group>(buf, SwissTable::kEmpty);

	// All same fingerprint.
	memset(buf, 0x42, sizeof(buf));
	check_group_against_reference<Group>(buf, 0x42);
	check_group_against_reference<Group>(buf, 0x43);

	// Patterned mix of empty / deleted / full.
	const uint32_t seeds[] = { 0x12345678u, 0xdeadbeefu, 0xfeedfaceu, 0xcafebabeu, 0x00000001u, 0xffffffffu };
	for (uint32_t seed : seeds) {
		fill_pattern(buf, Group::kWidth, seed);
		uint8_t needles[] = { 0x00, 0x42, 0x7F, SwissTable::kEmpty, SwissTable::kDeleted, buf[0], buf[Group::kWidth / 2] };
		for (uint8_t needle : needles) {
			check_group_against_reference<Group>(buf, needle);
		}
	}
}

TEST_CASE("[SwissTable][SIMD] SWAR group matches reference") {
	exercise_group<SwissTable::GroupSWAR>();
}

#if defined(SWISS_TABLE_HAS_SSE2)
TEST_CASE("[SwissTable][SIMD] SSE2 group matches reference") {
	exercise_group<SwissTable::GroupSSE2>();
}
#endif

#if defined(SWISS_TABLE_HAS_NEON)
TEST_CASE("[SwissTable][SIMD] NEON group matches reference") {
	exercise_group<SwissTable::GroupNEON>();
}
#endif

#if defined(SWISS_TABLE_HAS_WASM_SIMD)
TEST_CASE("[SwissTable][SIMD] WASM group matches reference") {
	exercise_group<SwissTable::GroupWASM>();
}
#endif

TEST_CASE("[SwissTable] H1/H2 derivation") {
	// h2 should always have the top bit cleared so it never collides with kEmpty/kDeleted.
	for (uint32_t hash = 0; hash < 1024; hash++) {
		uint8_t h2 = SwissTable::h2(hash);
		CHECK((h2 & 0x80) == 0);
		CHECK(h2 != SwissTable::kEmpty);
		CHECK(h2 != SwissTable::kDeleted);
	}
	// Different high bits should give different fingerprints.
	CHECK_NE(SwissTable::h2(0x00000000u), SwissTable::h2(0x80000000u));
	CHECK_NE(SwissTable::h2(0xFE000000u), SwissTable::h2(0x00000000u));
}

TEST_CASE("[SwissTable] capacity_to_growth") {
	for (uint32_t cap : { 8u, 16u, 32u, 64u, 128u, 256u, 1024u }) {
		CHECK_EQ(SwissTable::capacity_to_growth(cap), cap - cap / 8);
	}
}

} // namespace TestSwissTableSIMD
