/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkMathPriv_DEFINED
#define SkMathPriv_DEFINED

#include "include/core/SkMath.h"

/**
 *  Return the integer square root of value, with a bias of bitBias
 */
int32_t SkSqrtBits(int32_t value, int bitBias);

/** Return the integer square root of n, treated as a SkFixed (16.16)
 */
static inline int32_t SkSqrt32(int32_t n) { return SkSqrtBits(n, 15); }

/**
 *  Returns (value < 0 ? 0 : value) efficiently (i.e. no compares or branches)
 */
static inline int SkClampPos(int value) {
    return value & ~(value >> 31);
}

/**
 * Stores numer/denom and numer%denom into div and mod respectively.
 */
template <typename In, typename Out>
inline void SkTDivMod(In numer, In denom, Out* div, Out* mod) {
#ifdef SK_CPU_ARM32
    // If we wrote this as in the else branch, GCC won't fuse the two into one
    // divmod call, but rather a div call followed by a divmod.  Silly!  This
    // version is just as fast as calling __aeabi_[u]idivmod manually, but with
    // prettier code.
    //
    // This benches as around 2x faster than the code in the else branch.
    const In d = numer/denom;
    *div = static_cast<Out>(d);
    *mod = static_cast<Out>(numer-d*denom);
#else
    // On x86 this will just be a single idiv.
    *div = static_cast<Out>(numer/denom);
    *mod = static_cast<Out>(numer%denom);
#endif
}

/** Returns -1 if n < 0, else returns 0
 */
#define SkExtractSign(n)    ((int32_t)(n) >> 31)

/** If sign == -1, returns -n, else sign must be 0, and returns n.
 Typically used in conjunction with SkExtractSign().
 */
static inline int32_t SkApplySign(int32_t n, int32_t sign) {
    SkASSERT(sign == 0 || sign == -1);
    return (n ^ sign) - sign;
}

/** Return x with the sign of y */
static inline int32_t SkCopySign32(int32_t x, int32_t y) {
    return SkApplySign(x, SkExtractSign(x ^ y));
}

/** Given a positive value and a positive max, return the value
 pinned against max.
 Note: only works as long as max - value doesn't wrap around
 @return max if value >= max, else value
 */
static inline unsigned SkClampUMax(unsigned value, unsigned max) {
    if (value > max) {
        value = max;
    }
    return value;
}

// If a signed int holds min_int (e.g. 0x80000000) it is undefined what happens when
// we negate it (even though we *know* we're 2's complement and we'll get the same
// value back). So we create this helper function that casts to size_t (unsigned) first,
// to avoid the complaint.
static inline size_t sk_negate_to_size_t(int32_t value) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4146)  // Thanks MSVC, we know what we're negating an unsigned
#endif
    return -static_cast<size_t>(value);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

///////////////////////////////////////////////////////////////////////////////

/** Return a*b/255, truncating away any fractional bits. Only valid if both
 a and b are 0..255
 */
static inline U8CPU SkMulDiv255Trunc(U8CPU a, U8CPU b) {
    SkASSERT((uint8_t)a == a);
    SkASSERT((uint8_t)b == b);
    unsigned prod = a*b + 1;
    return (prod + (prod >> 8)) >> 8;
}

/** Return (a*b)/255, taking the ceiling of any fractional bits. Only valid if
 both a and b are 0..255. The expected result equals (a * b + 254) / 255.
 */
static inline U8CPU SkMulDiv255Ceiling(U8CPU a, U8CPU b) {
    SkASSERT((uint8_t)a == a);
    SkASSERT((uint8_t)b == b);
    unsigned prod = a*b + 255;
    return (prod + (prod >> 8)) >> 8;
}

/** Just the rounding step in SkDiv255Round: round(value / 255)
 */
static inline unsigned SkDiv255Round(unsigned prod) {
    prod += 128;
    return (prod + (prod >> 8)) >> 8;
}

/**
 * Swap byte order of a 4-byte value, e.g. 0xaarrggbb -> 0xbbggrraa.
 */
#if defined(_MSC_VER)
    #include <stdlib.h>
    static inline uint32_t SkBSwap32(uint32_t v) { return _byteswap_ulong(v); }
#else
    static inline uint32_t SkBSwap32(uint32_t v) { return __builtin_bswap32(v); }
#endif

//! Returns the number of leading zero bits (0...32)
// From Hacker's Delight 2nd Edition
constexpr int SkCLZ_portable(uint32_t x) {
    int n = 32;
    uint32_t y = x >> 16; if (y != 0) {n -= 16; x = y;}
             y = x >>  8; if (y != 0) {n -=  8; x = y;}
             y = x >>  4; if (y != 0) {n -=  4; x = y;}
             y = x >>  2; if (y != 0) {n -=  2; x = y;}
             y = x >>  1; if (y != 0) {return n - 2;}
    return n - x;
}

static_assert(32 == SkCLZ_portable(0));
static_assert(31 == SkCLZ_portable(1));
static_assert( 1 == SkCLZ_portable(1 << 30));
static_assert( 1 == SkCLZ_portable((1 << 30) | (1 << 24) | 1));
static_assert( 0 == SkCLZ_portable(~0U));

#if defined(SK_BUILD_FOR_WIN)
    #include <intrin.h>

    static inline int SkCLZ(uint32_t mask) {
        if (mask) {
            unsigned long index = 0;
            _BitScanReverse(&index, mask);
            // Suppress this bogus /analyze warning. The check for non-zero
            // guarantees that _BitScanReverse will succeed.
            #pragma warning(suppress : 6102) // Using 'index' from failed function call
            return index ^ 0x1F;
        } else {
            return 32;
        }
    }
#elif defined(SK_CPU_ARM32) || defined(__GNUC__) || defined(__clang__)
    static inline int SkCLZ(uint32_t mask) {
        // __builtin_clz(0) is undefined, so we have to detect that case.
        return mask ? __builtin_clz(mask) : 32;
    }
#else
    static inline int SkCLZ(uint32_t mask) {
        return SkCLZ_portable(mask);
    }
#endif

//! Returns the number of trailing zero bits (0...32)
// From Hacker's Delight 2nd Edition
constexpr int SkCTZ_portable(uint32_t x) {
    return 32 - SkCLZ_portable(~x & (x - 1));
}

static_assert(32 == SkCTZ_portable(0));
static_assert( 0 == SkCTZ_portable(1));
static_assert(30 == SkCTZ_portable(1 << 30));
static_assert( 2 == SkCTZ_portable((1 << 30) | (1 << 24) | (1 << 2)));
static_assert( 0 == SkCTZ_portable(~0U));

#if defined(SK_BUILD_FOR_WIN)
    #include <intrin.h>

    static inline int SkCTZ(uint32_t mask) {
        if (mask) {
            unsigned long index = 0;
            _BitScanForward(&index, mask);
            // Suppress this bogus /analyze warning. The check for non-zero
            // guarantees that _BitScanReverse will succeed.
            #pragma warning(suppress : 6102) // Using 'index' from failed function call
            return index;
        } else {
            return 32;
        }
    }
#elif defined(SK_CPU_ARM32) || defined(__GNUC__) || defined(__clang__)
    static inline int SkCTZ(uint32_t mask) {
        // __builtin_ctz(0) is undefined, so we have to detect that case.
        return mask ? __builtin_ctz(mask) : 32;
    }
#else
    static inline int SkCTZ(uint32_t mask) {
        return SkCTZ_portable(mask);
    }
#endif

/**
 *  Returns the log2 of the specified value, were that value to be rounded up
 *  to the next power of 2. It is undefined to pass 0. Examples:
 *  SkNextLog2(1) -> 0
 *  SkNextLog2(2) -> 1
 *  SkNextLog2(3) -> 2
 *  SkNextLog2(4) -> 2
 *  SkNextLog2(5) -> 3
 */
static inline int SkNextLog2(uint32_t value) {
    SkASSERT(value != 0);
    return 32 - SkCLZ(value - 1);
}

constexpr int SkNextLog2_portable(uint32_t value) {
    SkASSERT(value != 0);
    return 32 - SkCLZ_portable(value - 1);
}

/**
*  Returns the log2 of the specified value, were that value to be rounded down
*  to the previous power of 2. It is undefined to pass 0. Examples:
*  SkPrevLog2(1) -> 0
*  SkPrevLog2(2) -> 1
*  SkPrevLog2(3) -> 1
*  SkPrevLog2(4) -> 2
*  SkPrevLog2(5) -> 2
*/
static inline int SkPrevLog2(uint32_t value) {
    SkASSERT(value != 0);
    return 32 - SkCLZ(value >> 1);
}

constexpr int SkPrevLog2_portable(uint32_t value) {
    SkASSERT(value != 0);
    return 32 - SkCLZ_portable(value >> 1);
}

/**
 *  Returns the smallest power-of-2 that is >= the specified value. If value
 *  is already a power of 2, then it is returned unchanged. It is undefined
 *  if value is <= 0.
 */
static inline int SkNextPow2(int value) {
    SkASSERT(value > 0);
    return 1 << SkNextLog2(value);
}

constexpr int SkNextPow2_portable(int value) {
    SkASSERT(value > 0);
    return 1 << SkNextLog2_portable(value);
}

/**
*  Returns the largest power-of-2 that is <= the specified value. If value
*  is already a power of 2, then it is returned unchanged. It is undefined
*  if value is <= 0.
*/
static inline int SkPrevPow2(int value) {
    SkASSERT(value > 0);
    return 1 << SkPrevLog2(value);
}

constexpr int SkPrevPow2_portable(int value) {
    SkASSERT(value > 0);
    return 1 << SkPrevLog2_portable(value);
}

///////////////////////////////////////////////////////////////////////////////

/**
 *  Return the smallest power-of-2 >= n.
 */
static inline uint32_t GrNextPow2(uint32_t n) {
    return n ? (1 << (32 - SkCLZ(n - 1))) : 1;
}

/**
 * Returns the next power of 2 >= n or n if the next power of 2 can't be represented by size_t.
 */
static inline size_t GrNextSizePow2(size_t n) {
    constexpr int kNumSizeTBits = 8 * sizeof(size_t);
    constexpr size_t kHighBitSet = size_t(1) << (kNumSizeTBits - 1);

    if (!n) {
        return 1;
    } else if (n >= kHighBitSet) {
        return n;
    }

    n--;
    uint32_t shift = 1;
    while (shift < kNumSizeTBits) {
        n |= n >> shift;
        shift <<= 1;
    }
    return n + 1;
}

// conservative check. will return false for very large values that "could" fit
template <typename T> static inline bool SkFitsInFixed(T x) {
    return SkTAbs(x) <= 32767.0f;
}

#endif
