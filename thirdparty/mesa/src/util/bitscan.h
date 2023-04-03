/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


#ifndef BITSCAN_H
#define BITSCAN_H

#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__POPCNT__)
#include <popcntintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Find first bit set in word.  Least significant bit is 1.
 * Return 0 if no bits set.
 */
#ifdef HAVE___BUILTIN_FFS
#define ffs __builtin_ffs
#elif defined(_MSC_VER) && (_M_IX86 || _M_ARM || _M_AMD64 || _M_IA64)
static inline
int ffs(int i)
{
   unsigned long index;
   if (_BitScanForward(&index, i))
      return index + 1;
   else
      return 0;
}
#else
extern
int ffs(int i);
#endif

#ifdef HAVE___BUILTIN_FFSLL
#define ffsll __builtin_ffsll
#elif defined(_MSC_VER) && (_M_AMD64 || _M_ARM64 || _M_IA64)
static inline int
ffsll(long long int i)
{
   unsigned long index;
   if (_BitScanForward64(&index, i))
      return index + 1;
   else
      return 0;
}
#else
extern int
ffsll(long long int val);
#endif


/* Destructively loop over all of the bits in a mask as in:
 *
 * while (mymask) {
 *   int i = u_bit_scan(&mymask);
 *   ... process element i
 * }
 *
 */
static inline int
u_bit_scan(unsigned *mask)
{
   const int i = ffs(*mask) - 1;
   *mask ^= (1u << i);
   return i;
}

#define u_foreach_bit(b, dword)                          \
   for (uint32_t __dword = (dword), b;                     \
        ((b) = ffs(__dword) - 1, __dword);      \
        __dword &= ~(1 << (b)))

static inline int
u_bit_scan64(uint64_t *mask)
{
   const int i = ffsll(*mask) - 1;
   *mask ^= (((uint64_t)1) << i);
   return i;
}

#define u_foreach_bit64(b, dword)                          \
   for (uint64_t __dword = (dword), b;                     \
        ((b) = ffsll(__dword) - 1, __dword);      \
        __dword &= ~(1ull << (b)))

/* Determine if an unsigned value is a power of two.
 *
 * \note
 * Zero is treated as a power of two.
 */
static inline bool
util_is_power_of_two_or_zero(unsigned v)
{
   return (v & (v - 1)) == 0;
}

/* Determine if an uint64_t value is a power of two.
 *
 * \note
 * Zero is treated as a power of two.
 */
static inline bool
util_is_power_of_two_or_zero64(uint64_t v)
{
   return (v & (v - 1)) == 0;
}

/* Determine if an unsigned value is a power of two.
 *
 * \note
 * Zero is \b not treated as a power of two.
 */
static inline bool
util_is_power_of_two_nonzero(unsigned v)
{
   /* __POPCNT__ is different from HAVE___BUILTIN_POPCOUNT.  The latter
    * indicates the existence of the __builtin_popcount function.  The former
    * indicates that _mm_popcnt_u32 exists and is a native instruction.
    *
    * The other alternative is to use SSE 4.2 compile-time flags.  This has
    * two drawbacks.  First, there is currently no build infrastructure for
    * SSE 4.2 (only 4.1), so that would have to be added.  Second, some AMD
    * CPUs support POPCNT but not SSE 4.2 (e.g., Barcelona).
    */
#ifdef __POPCNT__
   return _mm_popcnt_u32(v) == 1;
#else
   return v != 0 && (v & (v - 1)) == 0;
#endif
}

/* For looping over a bitmask when you want to loop over consecutive bits
 * manually, for example:
 *
 * while (mask) {
 *    int start, count, i;
 *
 *    u_bit_scan_consecutive_range(&mask, &start, &count);
 *
 *    for (i = 0; i < count; i++)
 *       ... process element (start+i)
 * }
 */
static inline void
u_bit_scan_consecutive_range(unsigned *mask, int *start, int *count)
{
   if (*mask == 0xffffffff) {
      *start = 0;
      *count = 32;
      *mask = 0;
      return;
   }
   *start = ffs(*mask) - 1;
   *count = ffs(~(*mask >> *start)) - 1;
   *mask &= ~(((1u << *count) - 1) << *start);
}

static inline void
u_bit_scan_consecutive_range64(uint64_t *mask, int *start, int *count)
{
   if (*mask == ~0ull) {
      *start = 0;
      *count = 64;
      *mask = 0;
      return;
   }
   *start = ffsll(*mask) - 1;
   *count = ffsll(~(*mask >> *start)) - 1;
   *mask &= ~(((((uint64_t)1) << *count) - 1) << *start);
}


/**
 * Find last bit set in a word.  The least significant bit is 1.
 * Return 0 if no bits are set.
 * Essentially ffs() in the reverse direction.
 */
static inline unsigned
util_last_bit(unsigned u)
{
#if defined(HAVE___BUILTIN_CLZ)
   return u == 0 ? 0 : 32 - __builtin_clz(u);
#elif defined(_MSC_VER) && (_M_IX86 || _M_ARM || _M_AMD64 || _M_IA64)
   unsigned long index;
   if (_BitScanReverse(&index, u))
      return index + 1;
   else
      return 0;
#else
   unsigned r = 0;
   while (u) {
      r++;
      u >>= 1;
   }
   return r;
#endif
}

/**
 * Find last bit set in a word.  The least significant bit is 1.
 * Return 0 if no bits are set.
 * Essentially ffsll() in the reverse direction.
 */
static inline unsigned
util_last_bit64(uint64_t u)
{
#if defined(HAVE___BUILTIN_CLZLL)
   return u == 0 ? 0 : 64 - __builtin_clzll(u);
#elif defined(_MSC_VER) && (_M_AMD64 || _M_ARM64 || _M_IA64)
   unsigned long index;
   if (_BitScanReverse64(&index, u))
      return index + 1;
   else
      return 0;
#else
   unsigned r = 0;
   while (u) {
      r++;
      u >>= 1;
   }
   return r;
#endif
}

/**
 * Find last bit in a word that does not match the sign bit. The least
 * significant bit is 1.
 * Return 0 if no bits are set.
 */
static inline unsigned
util_last_bit_signed(int i)
{
   if (i >= 0)
      return util_last_bit(i);
   else
      return util_last_bit(~(unsigned)i);
}

/* Returns a bitfield in which the first count bits starting at start are
 * set.
 */
static inline unsigned
u_bit_consecutive(unsigned start, unsigned count)
{
   assert(start + count <= 32);
   if (count == 32)
      return ~0;
   return ((1u << count) - 1) << start;
}

static inline uint64_t
u_bit_consecutive64(unsigned start, unsigned count)
{
   assert(start + count <= 64);
   if (count == 64)
      return ~(uint64_t)0;
   return (((uint64_t)1 << count) - 1) << start;
}

/**
 * Return number of bits set in n.
 */
static inline unsigned
util_bitcount(unsigned n)
{
#if defined(HAVE___BUILTIN_POPCOUNT)
   return __builtin_popcount(n);
#else
   /* K&R classic bitcount.
    *
    * For each iteration, clear the LSB from the bitfield.
    * Requires only one iteration per set bit, instead of
    * one iteration per bit less than highest set bit.
    */
   unsigned bits;
   for (bits = 0; n; bits++) {
      n &= n - 1;
   }
   return bits;
#endif
}

/**
 * Return the number of bits set in n using the native popcnt instruction.
 * The caller is responsible for ensuring that popcnt is supported by the CPU.
 *
 * gcc doesn't use it if -mpopcnt or -march= that has popcnt is missing.
 *
 */
static inline unsigned
util_popcnt_inline_asm(unsigned n)
{
#if defined(USE_X86_64_ASM) || defined(USE_X86_ASM)
   uint32_t out;
   __asm volatile("popcnt %1, %0" : "=r"(out) : "r"(n));
   return out;
#else
   /* We should never get here by accident, but I'm sure it'll happen. */
   return util_bitcount(n);
#endif
}

static inline unsigned
util_bitcount64(uint64_t n)
{
#ifdef HAVE___BUILTIN_POPCOUNTLL
   return __builtin_popcountll(n);
#else
   return util_bitcount(n) + util_bitcount(n >> 32);
#endif
}

/**
 * Widens the given bit mask by a multiplier, meaning that it will
 * replicate each bit by that amount.
 *
 * For example:
 * 0b101 widened by 2 will become: 0b110011
 *
 * This is typically used in shader I/O to transform a 64-bit
 * writemask to a 32-bit writemask.
 */
static inline uint32_t
util_widen_mask(uint32_t mask, unsigned multiplier)
{
   uint32_t new_mask = 0;
   u_foreach_bit(i, mask)
      new_mask |= ((1u << multiplier) - 1u) << (i * multiplier);
   return new_mask;
}

#ifdef __cplusplus
}

/* util_bitcount has large measurable overhead (~2%), so it's recommended to
 * use the POPCNT instruction via inline assembly if the CPU supports it.
 */
enum util_popcnt {
   POPCNT_NO,
   POPCNT_YES,
};

/* Convenient function to select popcnt through a C++ template argument.
 * This should be used as part of larger functions that are optimized
 * as a whole.
 */
template<util_popcnt POPCNT> inline unsigned
util_bitcount_fast(unsigned n)
{
   if (POPCNT == POPCNT_YES)
      return util_popcnt_inline_asm(n);
   else
      return util_bitcount(n);
}

#endif /* __cplusplus */

#endif /* BITSCAN_H */
