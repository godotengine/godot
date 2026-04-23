/**************************************************************************/
/*  swiss_table_simd.h                                                    */
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

#include "core/typedefs.h"

#include <stdint.h>
#include <string.h>

/**
 * SwissTable-style control-byte group scanning primitives, shared by HashMap
 * and AHashMap.
 *
 * Control bytes store empty, deleted, or a 7-bit hash fingerprint. Probing
 * compares a group of control bytes against h2 and only does a full key
 * comparison on matching slots.
 */

namespace SwissTable {

inline constexpr uint8_t kEmpty = 0x80;
inline constexpr uint8_t kDeleted = 0xFE;
inline constexpr uint8_t kSentinel = 0xFF; // Reserved, currently unused.

static _FORCE_INLINE_ bool is_empty(uint8_t ctrl) {
	return ctrl == kEmpty;
}

static _FORCE_INLINE_ bool is_deleted(uint8_t ctrl) {
	return ctrl == kDeleted;
}

static _FORCE_INLINE_ bool is_empty_or_deleted(uint8_t ctrl) {
	return (ctrl & 0x80) != 0;
}

static _FORCE_INLINE_ bool is_full(uint8_t ctrl) {
	return (ctrl & 0x80) == 0;
}

// Finalizer applied before splitting a 32-bit hash into h1 and h2.
static _FORCE_INLINE_ uint32_t mix(uint32_t hash) {
	hash ^= hash >> 16;
	hash *= 0x7feb352dU;
	hash ^= hash >> 15;
	hash *= 0x846ca68bU;
	hash ^= hash >> 16;
	return hash;
}

// Derive the 7-bit fingerprint stored in the control byte.
static _FORCE_INLINE_ uint8_t h2(uint32_t hash) {
	return static_cast<uint8_t>((hash >> 25) & 0x7F);
}

// Starting group for the probe sequence.
static _FORCE_INLINE_ uint32_t h1(uint32_t hash) {
	return hash >> 7;
}

// Iterate set bits in a backend-specific SIMD match bitmask.
template <uint32_t Width, uint32_t Shift>
struct BitMask {
	uint64_t mask;

	static constexpr uint64_t kMatchBits = (1ULL << (1u << Shift)) - 1;

	_FORCE_INLINE_ explicit operator bool() const { return mask != 0; }

	static _FORCE_INLINE_ uint32_t ctz64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
		return static_cast<uint32_t>(__builtin_ctzll(v));
#elif defined(_MSC_VER)
		unsigned long idx;
		_BitScanForward64(&idx, v);
		return static_cast<uint32_t>(idx);
#else
		uint32_t result = 0;
		while ((v & 1) == 0) {
			v >>= 1;
			++result;
		}
		return result;
#endif
	}

	static _FORCE_INLINE_ uint32_t clz64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
		return static_cast<uint32_t>(__builtin_clzll(v));
#elif defined(_MSC_VER)
		unsigned long idx;
		_BitScanReverse64(&idx, v);
		return static_cast<uint32_t>(63 - idx);
#else
		uint32_t result = 0;
		while ((v & (1ULL << 63)) == 0) {
			v <<= 1;
			++result;
		}
		return result;
#endif
	}

	_FORCE_INLINE_ uint32_t lowest_set_bit() const {
		return ctz64(mask) >> Shift;
	}

	_FORCE_INLINE_ uint32_t highest_set_bit() const {
		return (63u - clz64(mask)) >> Shift;
	}

	struct Iterator {
		uint64_t m;
		_FORCE_INLINE_ Iterator(uint64_t p_m) :
				m(p_m) {}
		_FORCE_INLINE_ bool has_next() const { return m != 0; }
		_FORCE_INLINE_ uint32_t next() {
			const uint32_t bit = ctz64(m);
			const uint32_t idx = bit >> Shift;
			const uint32_t base = bit & ~((1u << Shift) - 1u);
			m &= ~(kMatchBits << base);
			return idx;
		}
	};

	_FORCE_INLINE_ Iterator iter() const { return Iterator(mask); }

	static constexpr uint32_t width() { return Width; }
};

} // namespace SwissTable

// SSE2 implementation.

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86) && defined(_M_IX86_FP) && (_M_IX86_FP >= 2))
#define SWISS_TABLE_HAS_SSE2 1
#include <emmintrin.h>

namespace SwissTable {

struct GroupSSE2 {
	static constexpr uint32_t kWidth = 16;
	using Mask = BitMask<kWidth, 0>;

	__m128i ctrl;

	_FORCE_INLINE_ explicit GroupSSE2(const uint8_t *p_ctrl) {
		ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p_ctrl));
	}

	_FORCE_INLINE_ Mask match(uint8_t p_h2) const {
		__m128i needle = _mm_set1_epi8(static_cast<char>(p_h2));
		__m128i eq = _mm_cmpeq_epi8(ctrl, needle);
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(eq))) };
	}

	_FORCE_INLINE_ Mask match_empty() const {
		__m128i needle = _mm_set1_epi8(static_cast<char>(kEmpty));
		__m128i eq = _mm_cmpeq_epi8(ctrl, needle);
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(eq))) };
	}

	_FORCE_INLINE_ Mask match_empty_or_deleted() const {
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(ctrl))) };
	}
};

} // namespace SwissTable
#endif

// NEON implementation.

#if (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)) && !defined(SWISS_TABLE_HAS_SSE2)
#define SWISS_TABLE_HAS_NEON 1
#include <arm_neon.h>

namespace SwissTable {

struct GroupNEON {
	static constexpr uint32_t kWidth = 8;
	// Each matched byte produces 4 bits in the mask; shift right by 2 to get
	// the slot index from a trailing-zero count.
	using Mask = BitMask<kWidth, 2>;

	uint8x8_t ctrl;

	_FORCE_INLINE_ explicit GroupNEON(const uint8_t *p_ctrl) {
		ctrl = vld1_u8(p_ctrl);
	}

	// Convert an 8-byte vector of all-ones-or-all-zeros bytes into a 32-bit
	// mask with 4 bits per matched byte.
	static _FORCE_INLINE_ uint64_t to_bitmask(uint8x8_t v) {
		// vshrn_n_u16 with shift 4 on a 16x4 reinterpretation packs each
		// nibble; the resulting 32-bit value has 4 bits set per matched byte.
		uint16x8_t v16 = vreinterpretq_u16_u8(vcombine_u8(v, vdup_n_u8(0)));
		uint8x8_t packed = vshrn_n_u16(v16, 4);
		return static_cast<uint64_t>(vget_lane_u64(vreinterpret_u64_u8(packed), 0));
	}

	_FORCE_INLINE_ Mask match(uint8_t p_h2) const {
		uint8x8_t needle = vdup_n_u8(p_h2);
		uint8x8_t eq = vceq_u8(ctrl, needle);
		return Mask{ to_bitmask(eq) };
	}

	_FORCE_INLINE_ Mask match_empty() const {
		uint8x8_t needle = vdup_n_u8(kEmpty);
		uint8x8_t eq = vceq_u8(ctrl, needle);
		return Mask{ to_bitmask(eq) };
	}

	_FORCE_INLINE_ Mask match_empty_or_deleted() const {
		// Top bit set -> empty or deleted. Test with sign extension.
		uint8x8_t signbit = vdup_n_u8(0x80);
		uint8x8_t masked = vand_u8(ctrl, signbit);
		uint8x8_t eq = vceq_u8(masked, signbit);
		return Mask{ to_bitmask(eq) };
	}
};

} // namespace SwissTable
#endif

// =============================================================================
// WASM SIMD128 implementation
// =============================================================================

#if defined(__wasm_simd128__) && !defined(SWISS_TABLE_HAS_SSE2) && !defined(SWISS_TABLE_HAS_NEON)
#define SWISS_TABLE_HAS_WASM_SIMD 1
#include <wasm_simd128.h>

namespace SwissTable {

struct GroupWASM {
	static constexpr uint32_t kWidth = 16;
	using Mask = BitMask<kWidth, 0>;

	v128_t ctrl;

	_FORCE_INLINE_ explicit GroupWASM(const uint8_t *p_ctrl) {
		ctrl = wasm_v128_load(p_ctrl);
	}

	_FORCE_INLINE_ Mask match(uint8_t p_h2) const {
		v128_t needle = wasm_i8x16_splat(static_cast<int8_t>(p_h2));
		v128_t eq = wasm_i8x16_eq(ctrl, needle);
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(wasm_i8x16_bitmask(eq))) };
	}

	_FORCE_INLINE_ Mask match_empty() const {
		v128_t needle = wasm_i8x16_splat(static_cast<int8_t>(kEmpty));
		v128_t eq = wasm_i8x16_eq(ctrl, needle);
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(wasm_i8x16_bitmask(eq))) };
	}

	_FORCE_INLINE_ Mask match_empty_or_deleted() const {
		// High bit of each byte -> empty or deleted.
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(wasm_i8x16_bitmask(ctrl))) };
	}
};

} // namespace SwissTable
#endif

// =============================================================================
// SWAR scalar fallback (always available)
// =============================================================================

namespace SwissTable {

struct GroupSWAR {
	static constexpr uint32_t kWidth = 8;
	// Each matched byte produces 8 bits in the mask; shift right by 3 to get
	// the slot index from a trailing-zero count.
	using Mask = BitMask<kWidth, 3>;

	uint64_t ctrl;

	_FORCE_INLINE_ explicit GroupSWAR(const uint8_t *p_ctrl) {
		memcpy(&ctrl, p_ctrl, sizeof(ctrl));
	}

	// Standard "find equal byte" SWAR trick:
	//   v = ctrl XOR broadcast(p_h2)
	//   matches are bytes equal to 0
	//   has_zero(v) = (v - 0x01010101...) AND ~v AND 0x80808080...
	_FORCE_INLINE_ Mask match(uint8_t p_h2) const {
		uint64_t broadcast = 0x0101010101010101ULL * static_cast<uint64_t>(p_h2);
		uint64_t v = ctrl ^ broadcast;
		uint64_t hi = (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;
		return Mask{ hi };
	}

	_FORCE_INLINE_ Mask match_empty() const {
		// kEmpty == 0x80. A byte equals 0x80 iff it has only the high bit set:
		// (b & 0x80) != 0 AND (b & 0x7F) == 0.
		// We compute this as (ctrl & ((~ctrl << 1) | ~0x7F-line)). Easier:
		// match against the broadcast of 0x80 using the generic trick.
		return match(kEmpty);
	}

	_FORCE_INLINE_ Mask match_empty_or_deleted() const {
		// Any byte with the high bit set -> kEmpty or kDeleted.
		return Mask{ ctrl & 0x8080808080808080ULL };
	}
};

} // namespace SwissTable

// =============================================================================
// Default Group selection
// =============================================================================

namespace SwissTable {

#if defined(SWISS_TABLE_HAS_SSE2)
using Group = GroupSSE2;
#elif defined(SWISS_TABLE_HAS_NEON)
using Group = GroupNEON;
#elif defined(SWISS_TABLE_HAS_WASM_SIMD)
using Group = GroupWASM;
#else
using Group = GroupSWAR;
#endif

inline constexpr uint32_t kGroupWidth = Group::kWidth;

// Round capacity up so it is large enough to host (size + 1) entries with at
// most 7/8 occupancy. Capacities are always powers of two with a minimum of
// kGroupWidth.
static _FORCE_INLINE_ uint32_t capacity_to_growth(uint32_t capacity) {
	// 7/8 of capacity, rounded down. capacity is always a power of two >= 8.
	return capacity - (capacity / 8);
}

static _FORCE_INLINE_ uint32_t growth_to_lower_bound_capacity(uint32_t growth) {
	// We want capacity >= growth * 8 / 7. Round up.
	if (growth == 0) {
		return 0;
	}
	return growth + ((growth - 1) / 7) + 1;
}

} // namespace SwissTable
