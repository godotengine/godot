/*
 * Copyright © 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef FAST_IDIV_BY_CONST_H
#define FAST_IDIV_BY_CONST_H

/* Imported from:
 *   https://raw.githubusercontent.com/ridiculousfish/libdivide/master/divide_by_constants_codegen_reference.c
 */

#include <inttypes.h>
#include <limits.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Computes "magic info" for performing signed division by a fixed integer D.
 * The type 'sint_t' is assumed to be defined as a signed integer type large
 * enough to hold both the dividend and the divisor.
 * Here >> is arithmetic (signed) shift, and >>> is logical shift.
 *
 * To emit code for n/d, rounding towards zero, use the following sequence:
 *
 *   m = compute_signed_magic_info(D)
 *   emit("result = (m.multiplier * n) >> SINT_BITS");
 *   if d > 0 and m.multiplier < 0: emit("result += n")
 *   if d < 0 and m.multiplier > 0: emit("result -= n")
 *   if m.post_shift > 0: emit("result >>= m.shift")
 *   emit("result += (result < 0)")
 *
 * The shifts by SINT_BITS may be "free" if the high half of the full multiply
 * is put in a separate register.
 *
 * The final add can of course be implemented via the sign bit, e.g.
 *    result += (result >>> (SINT_BITS - 1))
 * or
 *    result -= (result >> (SINT_BITS - 1))
 *
 * This code is heavily indebted to Hacker's Delight by Henry Warren.
 * See http://www.hackersdelight.org/HDcode/magic.c.txt
 * Used with permission from http://www.hackersdelight.org/permissions.htm
 */

struct util_fast_sdiv_info {
   int64_t multiplier; /* the "magic number" multiplier */
   unsigned shift; /* shift for the dividend after multiplying */
};

struct util_fast_sdiv_info
util_compute_fast_sdiv_info(int64_t D, unsigned SINT_BITS);

/* Computes "magic info" for performing unsigned division by a fixed positive
 * integer D.  UINT_BITS is the bit size at which the final "magic"
 * calculation will be performed; it is assumed to be large enough to hold
 * both the dividand and the divisor.  num_bits can be set appropriately if n
 * is known to be smaller than calc_bits; if this is not known then UINT_BITS
 * for num_bits.
 *
 * Assume we have a hardware register of width UINT_BITS, a known constant D
 * which is not zero and not a power of 2, and a variable n of width num_bits
 * (which may be up to UINT_BITS). To emit code for n/d, use one of the two
 * following sequences (here >>> refers to a logical bitshift):
 *
 *   m = compute_unsigned_magic_info(D, num_bits)
 *   if m.pre_shift > 0: emit("n >>>= m.pre_shift")
 *   if m.increment: emit("n = saturated_increment(n)")
 *   emit("result = (m.multiplier * n) >>> UINT_BITS")
 *   if m.post_shift > 0: emit("result >>>= m.post_shift")
 *
 * or
 *
 *   m = compute_unsigned_magic_info(D, num_bits)
 *   if m.pre_shift > 0: emit("n >>>= m.pre_shift")
 *   emit("result = m.multiplier * n")
 *   if m.increment: emit("result = result + m.multiplier")
 *   emit("result >>>= UINT_BITS")
 *   if m.post_shift > 0: emit("result >>>= m.post_shift")
 *
 * This second version works even if D is 1.  The shifts by UINT_BITS may be
 * "free" if the high half of the full multiply is put in a separate register.
 *
 * saturated_increment(n) means "increment n unless it would wrap to 0," i.e.
 *   if n == (1 << UINT_BITS)-1: result = n
 *   else: result = n+1
 * A common way to implement this is with the carry bit. For example, on x86:
 *   add 1
 *   sbb 0
 *
 * Some invariants:
 *   1: At least one of pre_shift and increment is zero
 *   2: multiplier is never zero
 *
 * This code incorporates the "round down" optimization per ridiculous_fish.
 */

struct util_fast_udiv_info {
   uint64_t multiplier; /* the "magic number" multiplier */
   unsigned pre_shift; /* shift for the dividend before multiplying */
   unsigned post_shift; /* shift for the dividend after multiplying */
   int increment; /* 0 or 1; if set then increment the numerator, using one of
                     the two strategies */
};

struct util_fast_udiv_info
util_compute_fast_udiv_info(uint64_t D, unsigned num_bits, unsigned UINT_BITS);

/* Below are possible options for dividing by a uniform in a shader where
 * the divisor is constant but not known at compile time.
 */

/* Full version. */
static inline uint32_t
util_fast_udiv32(uint32_t n, struct util_fast_udiv_info info)
{
   n = n >> info.pre_shift;
   /* If the divisor is not 1, you can instead use a 32-bit ADD that clamps
    * to UINT_MAX. Dividing by 1 needs the full 64-bit ADD.
    *
    * If you have unsigned 64-bit MAD with 32-bit inputs, you can do:
    *    increment = increment ? multiplier : 0; // on the CPU
    *    (n * multiplier + increment) // on the GPU using unsigned 64-bit MAD
    */
   n = (((uint64_t)n + info.increment) * info.multiplier) >> 32;
   n = n >> info.post_shift;
   return n;
}

/* A little more efficient version if n != UINT_MAX, i.e. no unsigned
 * wraparound in the computation.
 */
static inline uint32_t
util_fast_udiv32_nuw(uint32_t n, struct util_fast_udiv_info info)
{
   assert(n != UINT32_MAX);
   n = n >> info.pre_shift;
   n = n + info.increment;
   n = ((uint64_t)n * info.multiplier) >> 32;
   n = n >> info.post_shift;
   return n;
}

/* Even faster version but both operands must be 31-bit unsigned integers
 * and the divisor must be greater than 1.
 *
 * info must be computed with num_bits == 31.
 */
static inline uint32_t
util_fast_udiv32_u31_d_not_one(uint32_t n, struct util_fast_udiv_info info)
{
   assert(info.pre_shift == 0);
   assert(info.increment == 0);
   n = ((uint64_t)n * info.multiplier) >> 32;
   n = n >> info.post_shift;
   return n;
}

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* FAST_IDIV_BY_CONST_H */
