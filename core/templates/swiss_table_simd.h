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
 * Each control byte describes one slot in the index table:
 *   - kEmpty   (0x80) -- never used or rehashed-out.
 *   - kDeleted (0xFE) -- tombstone left by erase, may still be probed past.
 *   - 0..0x7F (kFull) -- 7-bit fingerprint (h2) of the contained key's hash.
 *
 * H1 (top bits) selects the starting probe group; H2 (low 7 bits, top bit
 * cleared) is stored in the control byte. Lookup loads a group of bytes,
 * SIMD-compares them against H2, and only follows up with a full key compare
 * on slots whose fingerprint matches.
 *
 * Group widths:
 *   - SSE2 / WASM SIMD128: 16 bytes
 *   - NEON: 8 bytes (we use the vshrn-pack trick to produce a 64-bit mask
 *     where each match byte is 0xF, and then condense to a 32-bit bitmask
 *     with one set bit per match)
 *   - SWAR scalar fallback: 8 bytes
 *
 * The Match return type (BitMask) abstracts the per-platform mask shape so
 * that callers can iterate set bits uniformly via lowest_set_bit() / next().
 */

namespace SwissTable {

inline constexpr uint8_t kEmpty = 0x80;
inline constexpr uint8_t kDeleted = 0xFE;
inline constexpr uint8_t kSentinel = 0xFF; // Reserved, currently unused.

// True if the control byte represents an empty slot.
static _FORCE_INLINE_ bool is_empty(uint8_t ctrl) {
	return ctrl == kEmpty;
}

// True if the control byte represents a tombstone.
static _FORCE_INLINE_ bool is_deleted(uint8_t ctrl) {
	return ctrl == kDeleted;
}

// True if the control byte represents either empty or a tombstone.
static _FORCE_INLINE_ bool is_empty_or_deleted(uint8_t ctrl) {
	return (ctrl & 0x80) != 0;
}

// True if the control byte represents a full slot (top bit cleared).
static _FORCE_INLINE_ bool is_full(uint8_t ctrl) {
	return (ctrl & 0x80) == 0;
}

// Bit avalanche / "finalizer" applied to user hashes before splitting them
// into H1 (group selector) and H2 (fingerprint).
//
// SwissTable selects buckets and fingerprints from disjoint bit ranges of a
// 32-bit hash. Several Godot keys use hash functions whose bits are not well
// mixed -- DJB2 String hashing in particular leaves the high bits highly
// correlated when keys share a common prefix (e.g. "key_0".."key_N"). That
// causes both probe-chain pile-ups (poor H1) and frequent fingerprint
// collisions (poor H2), which in turn force cache-missing Element derefs to
// do real key compares.
//
// We mix once at the public boundary (just before h1/h2) so callers don't
// have to know about it. This is the splitmix32-style finalizer; cheap (a
// few cycles) and breaks low-entropy correlations across all 32 bits.
static _FORCE_INLINE_ uint32_t mix(uint32_t hash) {
	hash ^= hash >> 16;
	hash *= 0x7feb352dU;
	hash ^= hash >> 15;
	hash *= 0x846ca68bU;
	hash ^= hash >> 16;
	return hash;
}

// Derive the 7-bit fingerprint to store in a control byte from a 32-bit hash.
// We use the top 7 bits of the 32-bit hash to spread the fingerprint across
// keys whose H1 (low bits) collide. Top bit is masked off so the value never
// collides with kEmpty/kDeleted.
static _FORCE_INLINE_ uint8_t h2(uint32_t hash) {
	return static_cast<uint8_t>((hash >> 25) & 0x7F);
}

// "Probe sequence" group index. The fast path uses the low bits of the hash
// to select the starting group; the caller is responsible for masking by
// (capacity - 1). See the comment in BitMask for how H1 is consumed.
static _FORCE_INLINE_ uint32_t h1(uint32_t hash) {
	return hash >> 7;
}

// Iterate set bits in a SIMD match bitmask. Each call to next() returns the
// slot index of the next match (0..Width-1). The mask layout differs per
// backend:
//   - SSE2 / WASM SIMD128: 1 bit per matched byte (Shift = 0).
//   - NEON: 4 bits per matched byte (Shift = 2).
//   - SWAR: 8 bits per matched byte (Shift = 3).
//
// `next()` clears ALL bits belonging to the lowest match before returning, so
// that the iteration step correctly advances past the per-match bit group.
template <uint32_t Width, uint32_t Shift>
struct BitMask {
	uint64_t mask;

	// Bits belonging to one match. 1 for SSE2/WASM, 0xF for NEON, 0xFF for SWAR.
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

	_FORCE_INLINE_ uint32_t lowest_set_bit() const {
		return ctz64(mask) >> Shift;
	}

	struct Iterator {
		uint64_t m;
		_FORCE_INLINE_ Iterator(uint64_t p_m) :
				m(p_m) {}
		_FORCE_INLINE_ bool has_next() const { return m != 0; }
		_FORCE_INLINE_ uint32_t next() {
			const uint32_t bit = ctz64(m);
			const uint32_t idx = bit >> Shift;
			// Clear all bits belonging to this match (one bit, one nibble, or
			// one byte depending on Shift). The base aligns to the per-match
			// stride (1, 4, or 8 bits).
			const uint32_t base = bit & ~((1u << Shift) - 1u);
			m &= ~(kMatchBits << base);
			return idx;
		}
	};

	_FORCE_INLINE_ Iterator iter() const { return Iterator(mask); }

	static constexpr uint32_t width() { return Width; }
};

} // namespace SwissTable

// =============================================================================
// SSE2 implementation (x86_64 always-on, x86 with -msse2)
// =============================================================================

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86) && defined(_M_IX86_FP) && (_M_IX86_FP >= 2))
#define SWISS_TABLE_HAS_SSE2 1
#include <emmintrin.h>

namespace SwissTable {

struct GroupSSE2 {
	static constexpr uint32_t kWidth = 16;
	using Mask = BitMask<kWidth, 0>;

	__m128i ctrl;

	_FORCE_INLINE_ explicit GroupSSE2(const uint8_t *p_ctrl) {
		// Use unaligned load -- callers may probe across group boundaries.
		ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p_ctrl));
	}

	// Slots whose control byte equals h2.
	_FORCE_INLINE_ Mask match(uint8_t p_h2) const {
		__m128i needle = _mm_set1_epi8(static_cast<char>(p_h2));
		__m128i eq = _mm_cmpeq_epi8(ctrl, needle);
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(eq))) };
	}

	// Slots whose control byte is kEmpty.
	_FORCE_INLINE_ Mask match_empty() const {
		__m128i needle = _mm_set1_epi8(static_cast<char>(kEmpty));
		__m128i eq = _mm_cmpeq_epi8(ctrl, needle);
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(eq))) };
	}

	// Slots whose control byte has the high bit set (empty OR deleted).
	_FORCE_INLINE_ Mask match_empty_or_deleted() const {
		// (ctrl & 0x80) != 0 is true for kEmpty and kDeleted.
		// _mm_movemask_epi8 returns the high bit of each byte directly.
		return Mask{ static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(ctrl))) };
	}
};

} // namespace SwissTable
#endif

// =============================================================================
// NEON implementation (ARM64 / ARMv7 with NEON)
// =============================================================================

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
