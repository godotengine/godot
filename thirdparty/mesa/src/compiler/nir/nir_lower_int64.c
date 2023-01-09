/*
 * Copyright © 2016 Intel Corporation
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

#include "nir.h"
#include "nir_builder.h"

#define COND_LOWER_OP(b, name, ...)                                   \
        (b->shader->options->lower_int64_options &                    \
         nir_lower_int64_op_to_options_mask(nir_op_##name)) ?         \
        lower_##name##64(b, __VA_ARGS__) : nir_##name(b, __VA_ARGS__)

#define COND_LOWER_CMP(b, name, ...)                                  \
        (b->shader->options->lower_int64_options &                    \
         nir_lower_int64_op_to_options_mask(nir_op_##name)) ?         \
        lower_int64_compare(b, nir_op_##name, __VA_ARGS__) :          \
        nir_##name(b, __VA_ARGS__)

#define COND_LOWER_CAST(b, name, ...)                                 \
        (b->shader->options->lower_int64_options &                    \
         nir_lower_int64_op_to_options_mask(nir_op_##name)) ?         \
        lower_##name(b, __VA_ARGS__) :                                \
        nir_##name(b, __VA_ARGS__)

static nir_ssa_def *
lower_b2i64(nir_builder *b, nir_ssa_def *x)
{
   return nir_pack_64_2x32_split(b, nir_b2i32(b, x), nir_imm_int(b, 0));
}

static nir_ssa_def *
lower_i2i8(nir_builder *b, nir_ssa_def *x)
{
   return nir_i2i8(b, nir_unpack_64_2x32_split_x(b, x));
}

static nir_ssa_def *
lower_i2i16(nir_builder *b, nir_ssa_def *x)
{
   return nir_i2i16(b, nir_unpack_64_2x32_split_x(b, x));
}


static nir_ssa_def *
lower_i2i32(nir_builder *b, nir_ssa_def *x)
{
   return nir_unpack_64_2x32_split_x(b, x);
}

static nir_ssa_def *
lower_i2i64(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *x32 = x->bit_size == 32 ? x : nir_i2i32(b, x);
   return nir_pack_64_2x32_split(b, x32, nir_ishr_imm(b, x32, 31));
}

static nir_ssa_def *
lower_u2u8(nir_builder *b, nir_ssa_def *x)
{
   return nir_u2u8(b, nir_unpack_64_2x32_split_x(b, x));
}

static nir_ssa_def *
lower_u2u16(nir_builder *b, nir_ssa_def *x)
{
   return nir_u2u16(b, nir_unpack_64_2x32_split_x(b, x));
}

static nir_ssa_def *
lower_u2u32(nir_builder *b, nir_ssa_def *x)
{
   return nir_unpack_64_2x32_split_x(b, x);
}

static nir_ssa_def *
lower_u2u64(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *x32 = x->bit_size == 32 ? x : nir_u2u32(b, x);
   return nir_pack_64_2x32_split(b, x32, nir_imm_int(b, 0));
}

static nir_ssa_def *
lower_bcsel64(nir_builder *b, nir_ssa_def *cond, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   return nir_pack_64_2x32_split(b, nir_bcsel(b, cond, x_lo, y_lo),
                                    nir_bcsel(b, cond, x_hi, y_hi));
}

static nir_ssa_def *
lower_inot64(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);

   return nir_pack_64_2x32_split(b, nir_inot(b, x_lo), nir_inot(b, x_hi));
}

static nir_ssa_def *
lower_iand64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   return nir_pack_64_2x32_split(b, nir_iand(b, x_lo, y_lo),
                                    nir_iand(b, x_hi, y_hi));
}

static nir_ssa_def *
lower_ior64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   return nir_pack_64_2x32_split(b, nir_ior(b, x_lo, y_lo),
                                    nir_ior(b, x_hi, y_hi));
}

static nir_ssa_def *
lower_ixor64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   return nir_pack_64_2x32_split(b, nir_ixor(b, x_lo, y_lo),
                                    nir_ixor(b, x_hi, y_hi));
}

static nir_ssa_def *
lower_ishl64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   /* Implemented as
    *
    * uint64_t lshift(uint64_t x, int c)
    * {
    *    c %= 64;
    *
    *    if (c == 0) return x;
    *
    *    uint32_t lo = LO(x), hi = HI(x);
    *
    *    if (c < 32) {
    *       uint32_t lo_shifted = lo << c;
    *       uint32_t hi_shifted = hi << c;
    *       uint32_t lo_shifted_hi = lo >> abs(32 - c);
    *       return pack_64(lo_shifted, hi_shifted | lo_shifted_hi);
    *    } else {
    *       uint32_t lo_shifted_hi = lo << abs(32 - c);
    *       return pack_64(0, lo_shifted_hi);
    *    }
    * }
    */
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   y = nir_iand_imm(b, y, 0x3f);

   nir_ssa_def *reverse_count = nir_iabs(b, nir_iadd(b, y, nir_imm_int(b, -32)));
   nir_ssa_def *lo_shifted = nir_ishl(b, x_lo, y);
   nir_ssa_def *hi_shifted = nir_ishl(b, x_hi, y);
   nir_ssa_def *lo_shifted_hi = nir_ushr(b, x_lo, reverse_count);

   nir_ssa_def *res_if_lt_32 =
      nir_pack_64_2x32_split(b, lo_shifted,
                                nir_ior(b, hi_shifted, lo_shifted_hi));
   nir_ssa_def *res_if_ge_32 =
      nir_pack_64_2x32_split(b, nir_imm_int(b, 0),
                                nir_ishl(b, x_lo, reverse_count));

   return nir_bcsel(b, nir_ieq_imm(b, y, 0), x,
                    nir_bcsel(b, nir_uge(b, y, nir_imm_int(b, 32)),
                                 res_if_ge_32, res_if_lt_32));
}

static nir_ssa_def *
lower_ishr64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   /* Implemented as
    *
    * uint64_t arshift(uint64_t x, int c)
    * {
    *    c %= 64;
    *
    *    if (c == 0) return x;
    *
    *    uint32_t lo = LO(x);
    *    int32_t  hi = HI(x);
    *
    *    if (c < 32) {
    *       uint32_t lo_shifted = lo >> c;
    *       uint32_t hi_shifted = hi >> c;
    *       uint32_t hi_shifted_lo = hi << abs(32 - c);
    *       return pack_64(hi_shifted, hi_shifted_lo | lo_shifted);
    *    } else {
    *       uint32_t hi_shifted = hi >> 31;
    *       uint32_t hi_shifted_lo = hi >> abs(32 - c);
    *       return pack_64(hi_shifted, hi_shifted_lo);
    *    }
    * }
    */
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   y = nir_iand_imm(b, y, 0x3f);

   nir_ssa_def *reverse_count = nir_iabs(b, nir_iadd(b, y, nir_imm_int(b, -32)));
   nir_ssa_def *lo_shifted = nir_ushr(b, x_lo, y);
   nir_ssa_def *hi_shifted = nir_ishr(b, x_hi, y);
   nir_ssa_def *hi_shifted_lo = nir_ishl(b, x_hi, reverse_count);

   nir_ssa_def *res_if_lt_32 =
      nir_pack_64_2x32_split(b, nir_ior(b, lo_shifted, hi_shifted_lo),
                                hi_shifted);
   nir_ssa_def *res_if_ge_32 =
      nir_pack_64_2x32_split(b, nir_ishr(b, x_hi, reverse_count),
                                nir_ishr(b, x_hi, nir_imm_int(b, 31)));

   return nir_bcsel(b, nir_ieq_imm(b, y, 0), x,
                    nir_bcsel(b, nir_uge(b, y, nir_imm_int(b, 32)),
                                 res_if_ge_32, res_if_lt_32));
}

static nir_ssa_def *
lower_ushr64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   /* Implemented as
    *
    * uint64_t rshift(uint64_t x, int c)
    * {
    *    c %= 64;
    *
    *    if (c == 0) return x;
    *
    *    uint32_t lo = LO(x), hi = HI(x);
    *
    *    if (c < 32) {
    *       uint32_t lo_shifted = lo >> c;
    *       uint32_t hi_shifted = hi >> c;
    *       uint32_t hi_shifted_lo = hi << abs(32 - c);
    *       return pack_64(hi_shifted, hi_shifted_lo | lo_shifted);
    *    } else {
    *       uint32_t hi_shifted_lo = hi >> abs(32 - c);
    *       return pack_64(0, hi_shifted_lo);
    *    }
    * }
    */

   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   y = nir_iand_imm(b, y, 0x3f);

   nir_ssa_def *reverse_count = nir_iabs(b, nir_iadd(b, y, nir_imm_int(b, -32)));
   nir_ssa_def *lo_shifted = nir_ushr(b, x_lo, y);
   nir_ssa_def *hi_shifted = nir_ushr(b, x_hi, y);
   nir_ssa_def *hi_shifted_lo = nir_ishl(b, x_hi, reverse_count);

   nir_ssa_def *res_if_lt_32 =
      nir_pack_64_2x32_split(b, nir_ior(b, lo_shifted, hi_shifted_lo),
                                hi_shifted);
   nir_ssa_def *res_if_ge_32 =
      nir_pack_64_2x32_split(b, nir_ushr(b, x_hi, reverse_count),
                                nir_imm_int(b, 0));

   return nir_bcsel(b, nir_ieq_imm(b, y, 0), x,
                    nir_bcsel(b, nir_uge(b, y, nir_imm_int(b, 32)),
                                 res_if_ge_32, res_if_lt_32));
}

static nir_ssa_def *
lower_iadd64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   nir_ssa_def *res_lo = nir_iadd(b, x_lo, y_lo);
   nir_ssa_def *carry = nir_b2i32(b, nir_ult(b, res_lo, x_lo));
   nir_ssa_def *res_hi = nir_iadd(b, carry, nir_iadd(b, x_hi, y_hi));

   return nir_pack_64_2x32_split(b, res_lo, res_hi);
}

static nir_ssa_def *
lower_isub64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   nir_ssa_def *res_lo = nir_isub(b, x_lo, y_lo);
   nir_ssa_def *borrow = nir_ineg(b, nir_b2i32(b, nir_ult(b, x_lo, y_lo)));
   nir_ssa_def *res_hi = nir_iadd(b, nir_isub(b, x_hi, y_hi), borrow);

   return nir_pack_64_2x32_split(b, res_lo, res_hi);
}

static nir_ssa_def *
lower_ineg64(nir_builder *b, nir_ssa_def *x)
{
   /* Since isub is the same number of instructions (with better dependencies)
    * as iadd, subtraction is actually more efficient for ineg than the usual
    * 2's complement "flip the bits and add one".
    */
   return lower_isub64(b, nir_imm_int64(b, 0), x);
}

static nir_ssa_def *
lower_iabs64(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *x_is_neg = nir_ilt(b, x_hi, nir_imm_int(b, 0));
   return nir_bcsel(b, x_is_neg, nir_ineg(b, x), x);
}

static nir_ssa_def *
lower_int64_compare(nir_builder *b, nir_op op, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   switch (op) {
   case nir_op_ieq:
      return nir_iand(b, nir_ieq(b, x_hi, y_hi), nir_ieq(b, x_lo, y_lo));
   case nir_op_ine:
      return nir_ior(b, nir_ine(b, x_hi, y_hi), nir_ine(b, x_lo, y_lo));
   case nir_op_ult:
      return nir_ior(b, nir_ult(b, x_hi, y_hi),
                        nir_iand(b, nir_ieq(b, x_hi, y_hi),
                                    nir_ult(b, x_lo, y_lo)));
   case nir_op_ilt:
      return nir_ior(b, nir_ilt(b, x_hi, y_hi),
                        nir_iand(b, nir_ieq(b, x_hi, y_hi),
                                    nir_ult(b, x_lo, y_lo)));
      break;
   case nir_op_uge:
      /* Lower as !(x < y) in the hopes of better CSE */
      return nir_inot(b, lower_int64_compare(b, nir_op_ult, x, y));
   case nir_op_ige:
      /* Lower as !(x < y) in the hopes of better CSE */
      return nir_inot(b, lower_int64_compare(b, nir_op_ilt, x, y));
   default:
      unreachable("Invalid comparison");
   }
}

static nir_ssa_def *
lower_umax64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   return nir_bcsel(b, lower_int64_compare(b, nir_op_ult, x, y), y, x);
}

static nir_ssa_def *
lower_imax64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   return nir_bcsel(b, lower_int64_compare(b, nir_op_ilt, x, y), y, x);
}

static nir_ssa_def *
lower_umin64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   return nir_bcsel(b, lower_int64_compare(b, nir_op_ult, x, y), x, y);
}

static nir_ssa_def *
lower_imin64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   return nir_bcsel(b, lower_int64_compare(b, nir_op_ilt, x, y), x, y);
}

static nir_ssa_def *
lower_mul_2x32_64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y,
                  bool sign_extend)
{
   nir_ssa_def *res_hi = sign_extend ? nir_imul_high(b, x, y)
                                     : nir_umul_high(b, x, y);

   return nir_pack_64_2x32_split(b, nir_imul(b, x, y), res_hi);
}

static nir_ssa_def *
lower_imul64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_ssa_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   nir_ssa_def *mul_lo = nir_umul_2x32_64(b, x_lo, y_lo);
   nir_ssa_def *res_hi = nir_iadd(b, nir_unpack_64_2x32_split_y(b, mul_lo),
                         nir_iadd(b, nir_imul(b, x_lo, y_hi),
                                     nir_imul(b, x_hi, y_lo)));

   return nir_pack_64_2x32_split(b, nir_unpack_64_2x32_split_x(b, mul_lo),
                                 res_hi);
}

static nir_ssa_def *
lower_mul_high64(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y,
                 bool sign_extend)
{
   nir_ssa_def *x32[4], *y32[4];
   x32[0] = nir_unpack_64_2x32_split_x(b, x);
   x32[1] = nir_unpack_64_2x32_split_y(b, x);
   if (sign_extend) {
      x32[2] = x32[3] = nir_ishr_imm(b, x32[1], 31);
   } else {
      x32[2] = x32[3] = nir_imm_int(b, 0);
   }

   y32[0] = nir_unpack_64_2x32_split_x(b, y);
   y32[1] = nir_unpack_64_2x32_split_y(b, y);
   if (sign_extend) {
      y32[2] = y32[3] = nir_ishr_imm(b, y32[1], 31);
   } else {
      y32[2] = y32[3] = nir_imm_int(b, 0);
   }

   nir_ssa_def *res[8] = { NULL, };

   /* Yes, the following generates a pile of code.  However, we throw res[0]
    * and res[1] away in the end and, if we're in the umul case, four of our
    * eight dword operands will be constant zero and opt_algebraic will clean
    * this up nicely.
    */
   for (unsigned i = 0; i < 4; i++) {
      nir_ssa_def *carry = NULL;
      for (unsigned j = 0; j < 4; j++) {
         /* The maximum values of x32[i] and y32[j] are UINT32_MAX so the
          * maximum value of tmp is UINT32_MAX * UINT32_MAX.  The maximum
          * value that will fit in tmp is
          *
          *    UINT64_MAX = UINT32_MAX << 32 + UINT32_MAX
          *               = UINT32_MAX * (UINT32_MAX + 1) + UINT32_MAX
          *               = UINT32_MAX * UINT32_MAX + 2 * UINT32_MAX
          *
          * so we're guaranteed that we can add in two more 32-bit values
          * without overflowing tmp.
          */
         nir_ssa_def *tmp = nir_umul_2x32_64(b, x32[i], y32[j]);

         if (res[i + j])
            tmp = nir_iadd(b, tmp, nir_u2u64(b, res[i + j]));
         if (carry)
            tmp = nir_iadd(b, tmp, carry);
         res[i + j] = nir_u2u32(b, tmp);
         carry = nir_ushr_imm(b, tmp, 32);
      }
      res[i + 4] = nir_u2u32(b, carry);
   }

   return nir_pack_64_2x32_split(b, res[2], res[3]);
}

static nir_ssa_def *
lower_isign64(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);

   nir_ssa_def *is_non_zero = nir_i2b(b, nir_ior(b, x_lo, x_hi));
   nir_ssa_def *res_hi = nir_ishr_imm(b, x_hi, 31);
   nir_ssa_def *res_lo = nir_ior(b, res_hi, nir_b2i32(b, is_non_zero));

   return nir_pack_64_2x32_split(b, res_lo, res_hi);
}

static void
lower_udiv64_mod64(nir_builder *b, nir_ssa_def *n, nir_ssa_def *d,
                   nir_ssa_def **q, nir_ssa_def **r)
{
   /* TODO: We should specially handle the case where the denominator is a
    * constant.  In that case, we should be able to reduce it to a multiply by
    * a constant, some shifts, and an add.
    */
   nir_ssa_def *n_lo = nir_unpack_64_2x32_split_x(b, n);
   nir_ssa_def *n_hi = nir_unpack_64_2x32_split_y(b, n);
   nir_ssa_def *d_lo = nir_unpack_64_2x32_split_x(b, d);
   nir_ssa_def *d_hi = nir_unpack_64_2x32_split_y(b, d);

   nir_ssa_def *q_lo = nir_imm_zero(b, n->num_components, 32);
   nir_ssa_def *q_hi = nir_imm_zero(b, n->num_components, 32);

   nir_ssa_def *n_hi_before_if = n_hi;
   nir_ssa_def *q_hi_before_if = q_hi;

   /* If the upper 32 bits of denom are non-zero, it is impossible for shifts
    * greater than 32 bits to occur.  If the upper 32 bits of the numerator
    * are zero, it is impossible for (denom << [63, 32]) <= numer unless
    * denom == 0.
    */
   nir_ssa_def *need_high_div =
      nir_iand(b, nir_ieq_imm(b, d_hi, 0), nir_uge(b, n_hi, d_lo));
   nir_push_if(b, nir_bany(b, need_high_div));
   {
      /* If we only have one component, then the bany above goes away and
       * this is always true within the if statement.
       */
      if (n->num_components == 1)
         need_high_div = nir_imm_true(b);

      nir_ssa_def *log2_d_lo = nir_ufind_msb(b, d_lo);

      for (int i = 31; i >= 0; i--) {
         /* if ((d.x << i) <= n.y) {
          *    n.y -= d.x << i;
          *    quot.y |= 1U << i;
          * }
          */
         nir_ssa_def *d_shift = nir_ishl(b, d_lo, nir_imm_int(b, i));
         nir_ssa_def *new_n_hi = nir_isub(b, n_hi, d_shift);
         nir_ssa_def *new_q_hi = nir_ior(b, q_hi, nir_imm_int(b, 1u << i));
         nir_ssa_def *cond = nir_iand(b, need_high_div,
                                         nir_uge(b, n_hi, d_shift));
         if (i != 0) {
            /* log2_d_lo is always <= 31, so we don't need to bother with it
             * in the last iteration.
             */
            cond = nir_iand(b, cond,
                               nir_ige(b, nir_imm_int(b, 31 - i), log2_d_lo));
         }
         n_hi = nir_bcsel(b, cond, new_n_hi, n_hi);
         q_hi = nir_bcsel(b, cond, new_q_hi, q_hi);
      }
   }
   nir_pop_if(b, NULL);
   n_hi = nir_if_phi(b, n_hi, n_hi_before_if);
   q_hi = nir_if_phi(b, q_hi, q_hi_before_if);

   nir_ssa_def *log2_denom = nir_ufind_msb(b, d_hi);

   n = nir_pack_64_2x32_split(b, n_lo, n_hi);
   d = nir_pack_64_2x32_split(b, d_lo, d_hi);
   for (int i = 31; i >= 0; i--) {
      /* if ((d64 << i) <= n64) {
       *    n64 -= d64 << i;
       *    quot.x |= 1U << i;
       * }
       */
      nir_ssa_def *d_shift = nir_ishl(b, d, nir_imm_int(b, i));
      nir_ssa_def *new_n = nir_isub(b, n, d_shift);
      nir_ssa_def *new_q_lo = nir_ior(b, q_lo, nir_imm_int(b, 1u << i));
      nir_ssa_def *cond = nir_uge(b, n, d_shift);
      if (i != 0) {
         /* log2_denom is always <= 31, so we don't need to bother with it
          * in the last iteration.
          */
         cond = nir_iand(b, cond,
                            nir_ige(b, nir_imm_int(b, 31 - i), log2_denom));
      }
      n = nir_bcsel(b, cond, new_n, n);
      q_lo = nir_bcsel(b, cond, new_q_lo, q_lo);
   }

   *q = nir_pack_64_2x32_split(b, q_lo, q_hi);
   *r = n;
}

static nir_ssa_def *
lower_udiv64(nir_builder *b, nir_ssa_def *n, nir_ssa_def *d)
{
   nir_ssa_def *q, *r;
   lower_udiv64_mod64(b, n, d, &q, &r);
   return q;
}

static nir_ssa_def *
lower_idiv64(nir_builder *b, nir_ssa_def *n, nir_ssa_def *d)
{
   nir_ssa_def *n_hi = nir_unpack_64_2x32_split_y(b, n);
   nir_ssa_def *d_hi = nir_unpack_64_2x32_split_y(b, d);

   nir_ssa_def *negate = nir_ine(b, nir_ilt(b, n_hi, nir_imm_int(b, 0)),
                                    nir_ilt(b, d_hi, nir_imm_int(b, 0)));
   nir_ssa_def *q, *r;
   lower_udiv64_mod64(b, nir_iabs(b, n), nir_iabs(b, d), &q, &r);
   return nir_bcsel(b, negate, nir_ineg(b, q), q);
}

static nir_ssa_def *
lower_umod64(nir_builder *b, nir_ssa_def *n, nir_ssa_def *d)
{
   nir_ssa_def *q, *r;
   lower_udiv64_mod64(b, n, d, &q, &r);
   return r;
}

static nir_ssa_def *
lower_imod64(nir_builder *b, nir_ssa_def *n, nir_ssa_def *d)
{
   nir_ssa_def *n_hi = nir_unpack_64_2x32_split_y(b, n);
   nir_ssa_def *d_hi = nir_unpack_64_2x32_split_y(b, d);
   nir_ssa_def *n_is_neg = nir_ilt(b, n_hi, nir_imm_int(b, 0));
   nir_ssa_def *d_is_neg = nir_ilt(b, d_hi, nir_imm_int(b, 0));

   nir_ssa_def *q, *r;
   lower_udiv64_mod64(b, nir_iabs(b, n), nir_iabs(b, d), &q, &r);

   nir_ssa_def *rem = nir_bcsel(b, n_is_neg, nir_ineg(b, r), r);

   return nir_bcsel(b, nir_ieq_imm(b, r, 0), nir_imm_int64(b, 0),
          nir_bcsel(b, nir_ieq(b, n_is_neg, d_is_neg), rem,
                       nir_iadd(b, rem, d)));
}

static nir_ssa_def *
lower_irem64(nir_builder *b, nir_ssa_def *n, nir_ssa_def *d)
{
   nir_ssa_def *n_hi = nir_unpack_64_2x32_split_y(b, n);
   nir_ssa_def *n_is_neg = nir_ilt(b, n_hi, nir_imm_int(b, 0));

   nir_ssa_def *q, *r;
   lower_udiv64_mod64(b, nir_iabs(b, n), nir_iabs(b, d), &q, &r);
   return nir_bcsel(b, n_is_neg, nir_ineg(b, r), r);
}

static nir_ssa_def *
lower_extract(nir_builder *b, nir_op op, nir_ssa_def *x, nir_ssa_def *c)
{
   assert(op == nir_op_extract_u8 || op == nir_op_extract_i8 ||
          op == nir_op_extract_u16 || op == nir_op_extract_i16);

   const int chunk = nir_src_as_uint(nir_src_for_ssa(c));
   const int chunk_bits =
      (op == nir_op_extract_u8 || op == nir_op_extract_i8) ? 8 : 16;
   const int num_chunks_in_32 = 32 / chunk_bits;

   nir_ssa_def *extract32;
   if (chunk < num_chunks_in_32) {
      extract32 = nir_build_alu(b, op, nir_unpack_64_2x32_split_x(b, x),
                                   nir_imm_int(b, chunk),
                                   NULL, NULL);
   } else {
      extract32 = nir_build_alu(b, op, nir_unpack_64_2x32_split_y(b, x),
                                   nir_imm_int(b, chunk - num_chunks_in_32),
                                   NULL, NULL);
   }

   if (op == nir_op_extract_i8 || op == nir_op_extract_i16)
      return lower_i2i64(b, extract32);
   else
      return lower_u2u64(b, extract32);
}

static nir_ssa_def *
lower_ufind_msb64(nir_builder *b, nir_ssa_def *x)
{

   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *lo_count = nir_ufind_msb(b, x_lo);
   nir_ssa_def *hi_count = nir_ufind_msb(b, x_hi);
   nir_ssa_def *valid_hi_bits = nir_ine(b, x_hi, nir_imm_int(b, 0));
   nir_ssa_def *hi_res = nir_iadd(b, nir_imm_intN_t(b, 32, 32), hi_count);
   return nir_bcsel(b, valid_hi_bits, hi_res, lo_count);
}

static nir_ssa_def *
lower_2f(nir_builder *b, nir_ssa_def *x, unsigned dest_bit_size,
         bool src_is_signed)
{
   nir_ssa_def *x_sign = NULL;

   if (src_is_signed) {
      x_sign = nir_bcsel(b, COND_LOWER_CMP(b, ilt, x, nir_imm_int64(b, 0)),
                         nir_imm_floatN_t(b, -1, dest_bit_size),
                         nir_imm_floatN_t(b, 1, dest_bit_size));
      x = COND_LOWER_OP(b, iabs, x);
   }

   nir_ssa_def *exp = COND_LOWER_OP(b, ufind_msb, x);
   unsigned significand_bits;

   switch (dest_bit_size) {
   case 64:
      significand_bits = 52;
      break;
   case 32:
      significand_bits = 23;
      break;
   case 16:
      significand_bits = 10;
      break;
   default:
      unreachable("Invalid dest_bit_size");
   }

   nir_ssa_def *discard =
      nir_imax(b, nir_isub(b, exp, nir_imm_int(b, significand_bits)),
                  nir_imm_int(b, 0));
   nir_ssa_def *significand = COND_LOWER_OP(b, ushr, x, discard);
   if (significand_bits < 32)
      significand = COND_LOWER_CAST(b, u2u32, significand);

   /* Round-to-nearest-even implementation:
    * - if the non-representable part of the significand is higher than half
    *   the minimum representable significand, we round-up
    * - if the non-representable part of the significand is equal to half the
    *   minimum representable significand and the representable part of the
    *   significand is odd, we round-up
    * - in any other case, we round-down
    */
   nir_ssa_def *lsb_mask = COND_LOWER_OP(b, ishl, nir_imm_int64(b, 1), discard);
   nir_ssa_def *rem_mask = COND_LOWER_OP(b, isub, lsb_mask, nir_imm_int64(b, 1));
   nir_ssa_def *half = COND_LOWER_OP(b, ishr, lsb_mask, nir_imm_int(b, 1));
   nir_ssa_def *rem = COND_LOWER_OP(b, iand, x, rem_mask);
   nir_ssa_def *halfway = nir_iand(b, COND_LOWER_CMP(b, ieq, rem, half),
                                   nir_ine(b, discard, nir_imm_int(b, 0)));
   nir_ssa_def *is_odd = COND_LOWER_CMP(b, ine, nir_imm_int64(b, 0),
                                         COND_LOWER_OP(b, iand, x, lsb_mask));
   nir_ssa_def *round_up = nir_ior(b, COND_LOWER_CMP(b, ilt, half, rem),
                                   nir_iand(b, halfway, is_odd));
   if (significand_bits >= 32)
      significand = COND_LOWER_OP(b, iadd, significand,
                                  COND_LOWER_CAST(b, b2i64, round_up));
   else
      significand = nir_iadd(b, significand, nir_b2i32(b, round_up));

   nir_ssa_def *res;

   if (dest_bit_size == 64) {
      /* Compute the left shift required to normalize the original
       * unrounded input manually.
       */
      nir_ssa_def *shift =
         nir_imax(b, nir_isub(b, nir_imm_int(b, significand_bits), exp),
                  nir_imm_int(b, 0));
      significand = COND_LOWER_OP(b, ishl, significand, shift);

      /* Check whether normalization led to overflow of the available
       * significand bits, which can only happen if round_up was true
       * above, in which case we need to add carry to the exponent and
       * discard an extra bit from the significand.  Note that we
       * don't need to repeat the round-up logic again, since the LSB
       * of the significand is guaranteed to be zero if there was
       * overflow.
       */
      nir_ssa_def *carry = nir_b2i32(
         b, nir_uge(b, nir_unpack_64_2x32_split_y(b, significand),
                    nir_imm_int(b, 1 << (significand_bits - 31))));
      significand = COND_LOWER_OP(b, ishr, significand, carry);
      exp = nir_iadd(b, exp, carry);

      /* Compute the biased exponent, taking care to handle a zero
       * input correctly, which would have caused exp to be negative.
       */
      nir_ssa_def *biased_exp = nir_bcsel(b, nir_ilt(b, exp, nir_imm_int(b, 0)),
                                          nir_imm_int(b, 0),
                                          nir_iadd(b, exp, nir_imm_int(b, 1023)));

      /* Pack the significand and exponent manually. */
      nir_ssa_def *lo = nir_unpack_64_2x32_split_x(b, significand);
      nir_ssa_def *hi = nir_bitfield_insert(
         b, nir_unpack_64_2x32_split_y(b, significand),
         biased_exp, nir_imm_int(b, 20), nir_imm_int(b, 11));

      res = nir_pack_64_2x32_split(b, lo, hi);

   } else if (dest_bit_size == 32) {
      res = nir_fmul(b, nir_u2f32(b, significand),
                     nir_fexp2(b, nir_u2f32(b, discard)));
   } else {
      res = nir_fmul(b, nir_u2f16(b, significand),
                     nir_fexp2(b, nir_u2f16(b, discard)));
   }

   if (src_is_signed)
      res = nir_fmul(b, res, x_sign);

   return res;
}

static nir_ssa_def *
lower_f2(nir_builder *b, nir_ssa_def *x, bool dst_is_signed)
{
   assert(x->bit_size == 16 || x->bit_size == 32 || x->bit_size == 64);
   nir_ssa_def *x_sign = NULL;

   if (dst_is_signed)
      x_sign = nir_fsign(b, x);

   x = nir_ftrunc(b, x);

   if (dst_is_signed)
      x = nir_fabs(b, x);

   nir_ssa_def *res;
   if (x->bit_size < 32) {
      res = nir_pack_64_2x32_split(b, nir_f2u32(b, x), nir_imm_int(b, 0));
   } else {
      nir_ssa_def *div = nir_imm_floatN_t(b, 1ULL << 32, x->bit_size);
      nir_ssa_def *res_hi = nir_f2u32(b, nir_fdiv(b, x, div));
      nir_ssa_def *res_lo = nir_f2u32(b, nir_frem(b, x, div));
      res = nir_pack_64_2x32_split(b, res_lo, res_hi);
   }

   if (dst_is_signed)
      res = nir_bcsel(b, nir_flt(b, x_sign, nir_imm_floatN_t(b, 0, x->bit_size)),
                      nir_ineg(b, res), res);

   return res;
}

static nir_ssa_def *
lower_bit_count64(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_ssa_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_ssa_def *lo_count = nir_bit_count(b, x_lo);
   nir_ssa_def *hi_count = nir_bit_count(b, x_hi);
   return nir_iadd(b, lo_count, hi_count);
}

nir_lower_int64_options
nir_lower_int64_op_to_options_mask(nir_op opcode)
{
   switch (opcode) {
   case nir_op_imul:
   case nir_op_amul:
      return nir_lower_imul64;
   case nir_op_imul_2x32_64:
   case nir_op_umul_2x32_64:
      return nir_lower_imul_2x32_64;
   case nir_op_imul_high:
   case nir_op_umul_high:
      return nir_lower_imul_high64;
   case nir_op_isign:
      return nir_lower_isign64;
   case nir_op_udiv:
   case nir_op_idiv:
   case nir_op_umod:
   case nir_op_imod:
   case nir_op_irem:
      return nir_lower_divmod64;
   case nir_op_b2i64:
   case nir_op_i2i8:
   case nir_op_i2i16:
   case nir_op_i2i32:
   case nir_op_i2i64:
   case nir_op_u2u8:
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_u2u64:
   case nir_op_i2f64:
   case nir_op_u2f64:
   case nir_op_i2f32:
   case nir_op_u2f32:
   case nir_op_i2f16:
   case nir_op_u2f16:
   case nir_op_f2i64:
   case nir_op_f2u64:
   case nir_op_bcsel:
      return nir_lower_mov64;
   case nir_op_ieq:
   case nir_op_ine:
   case nir_op_ult:
   case nir_op_ilt:
   case nir_op_uge:
   case nir_op_ige:
      return nir_lower_icmp64;
   case nir_op_iadd:
   case nir_op_isub:
      return nir_lower_iadd64;
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
      return nir_lower_minmax64;
   case nir_op_iabs:
      return nir_lower_iabs64;
   case nir_op_ineg:
      return nir_lower_ineg64;
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_inot:
      return nir_lower_logic64;
   case nir_op_ishl:
   case nir_op_ishr:
   case nir_op_ushr:
      return nir_lower_shift64;
   case nir_op_extract_u8:
   case nir_op_extract_i8:
   case nir_op_extract_u16:
   case nir_op_extract_i16:
      return nir_lower_extract64;
   case nir_op_ufind_msb:
      return nir_lower_ufind_msb64;
   case nir_op_bit_count:
      return nir_lower_bit_count64;
   default:
      return 0;
   }
}

static nir_ssa_def *
lower_int64_alu_instr(nir_builder *b, nir_alu_instr *alu)
{
   nir_ssa_def *src[4];
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
      src[i] = nir_ssa_for_alu_src(b, alu, i);

   switch (alu->op) {
   case nir_op_imul:
   case nir_op_amul:
      return lower_imul64(b, src[0], src[1]);
   case nir_op_imul_2x32_64:
      return lower_mul_2x32_64(b, src[0], src[1], true);
   case nir_op_umul_2x32_64:
      return lower_mul_2x32_64(b, src[0], src[1], false);
   case nir_op_imul_high:
      return lower_mul_high64(b, src[0], src[1], true);
   case nir_op_umul_high:
      return lower_mul_high64(b, src[0], src[1], false);
   case nir_op_isign:
      return lower_isign64(b, src[0]);
   case nir_op_udiv:
      return lower_udiv64(b, src[0], src[1]);
   case nir_op_idiv:
      return lower_idiv64(b, src[0], src[1]);
   case nir_op_umod:
      return lower_umod64(b, src[0], src[1]);
   case nir_op_imod:
      return lower_imod64(b, src[0], src[1]);
   case nir_op_irem:
      return lower_irem64(b, src[0], src[1]);
   case nir_op_b2i64:
      return lower_b2i64(b, src[0]);
   case nir_op_i2i8:
      return lower_i2i8(b, src[0]);
   case nir_op_i2i16:
      return lower_i2i16(b, src[0]);
   case nir_op_i2i32:
      return lower_i2i32(b, src[0]);
   case nir_op_i2i64:
      return lower_i2i64(b, src[0]);
   case nir_op_u2u8:
      return lower_u2u8(b, src[0]);
   case nir_op_u2u16:
      return lower_u2u16(b, src[0]);
   case nir_op_u2u32:
      return lower_u2u32(b, src[0]);
   case nir_op_u2u64:
      return lower_u2u64(b, src[0]);
   case nir_op_bcsel:
      return lower_bcsel64(b, src[0], src[1], src[2]);
   case nir_op_ieq:
   case nir_op_ine:
   case nir_op_ult:
   case nir_op_ilt:
   case nir_op_uge:
   case nir_op_ige:
      return lower_int64_compare(b, alu->op, src[0], src[1]);
   case nir_op_iadd:
      return lower_iadd64(b, src[0], src[1]);
   case nir_op_isub:
      return lower_isub64(b, src[0], src[1]);
   case nir_op_imin:
      return lower_imin64(b, src[0], src[1]);
   case nir_op_imax:
      return lower_imax64(b, src[0], src[1]);
   case nir_op_umin:
      return lower_umin64(b, src[0], src[1]);
   case nir_op_umax:
      return lower_umax64(b, src[0], src[1]);
   case nir_op_iabs:
      return lower_iabs64(b, src[0]);
   case nir_op_ineg:
      return lower_ineg64(b, src[0]);
   case nir_op_iand:
      return lower_iand64(b, src[0], src[1]);
   case nir_op_ior:
      return lower_ior64(b, src[0], src[1]);
   case nir_op_ixor:
      return lower_ixor64(b, src[0], src[1]);
   case nir_op_inot:
      return lower_inot64(b, src[0]);
   case nir_op_ishl:
      return lower_ishl64(b, src[0], src[1]);
   case nir_op_ishr:
      return lower_ishr64(b, src[0], src[1]);
   case nir_op_ushr:
      return lower_ushr64(b, src[0], src[1]);
   case nir_op_extract_u8:
   case nir_op_extract_i8:
   case nir_op_extract_u16:
   case nir_op_extract_i16:
      return lower_extract(b, alu->op, src[0], src[1]);
   case nir_op_ufind_msb:
      return lower_ufind_msb64(b, src[0]);
   case nir_op_bit_count:
      return lower_bit_count64(b, src[0]);
   case nir_op_i2f64:
   case nir_op_i2f32:
   case nir_op_i2f16:
      return lower_2f(b, src[0], nir_dest_bit_size(alu->dest.dest), true);
   case nir_op_u2f64:
   case nir_op_u2f32:
   case nir_op_u2f16:
      return lower_2f(b, src[0], nir_dest_bit_size(alu->dest.dest), false);
   case nir_op_f2i64:
   case nir_op_f2u64:
      return lower_f2(b, src[0], alu->op == nir_op_f2i64);
   default:
      unreachable("Invalid ALU opcode to lower");
   }
}

static bool
should_lower_int64_alu_instr(const nir_alu_instr *alu,
                             const nir_shader_compiler_options *options)
{
   switch (alu->op) {
   case nir_op_i2i8:
   case nir_op_i2i16:
   case nir_op_i2i32:
   case nir_op_u2u8:
   case nir_op_u2u16:
   case nir_op_u2u32:
      assert(alu->src[0].src.is_ssa);
      if (alu->src[0].src.ssa->bit_size != 64)
         return false;
      break;
   case nir_op_bcsel:
      assert(alu->src[1].src.is_ssa);
      assert(alu->src[2].src.is_ssa);
      assert(alu->src[1].src.ssa->bit_size ==
             alu->src[2].src.ssa->bit_size);
      if (alu->src[1].src.ssa->bit_size != 64)
         return false;
      break;
   case nir_op_ieq:
   case nir_op_ine:
   case nir_op_ult:
   case nir_op_ilt:
   case nir_op_uge:
   case nir_op_ige:
      assert(alu->src[0].src.is_ssa);
      assert(alu->src[1].src.is_ssa);
      assert(alu->src[0].src.ssa->bit_size ==
             alu->src[1].src.ssa->bit_size);
      if (alu->src[0].src.ssa->bit_size != 64)
         return false;
      break;
   case nir_op_ufind_msb:
   case nir_op_bit_count:
      assert(alu->src[0].src.is_ssa);
      if (alu->src[0].src.ssa->bit_size != 64)
         return false;
      break;
   case nir_op_amul:
      assert(alu->dest.dest.is_ssa);
      if (options->has_imul24)
         return false;
      if (alu->dest.dest.ssa.bit_size != 64)
         return false;
      break;
   case nir_op_i2f64:
   case nir_op_u2f64:
   case nir_op_i2f32:
   case nir_op_u2f32:
   case nir_op_i2f16:
   case nir_op_u2f16:
      assert(alu->src[0].src.is_ssa);
      if (alu->src[0].src.ssa->bit_size != 64)
         return false;
      break;
   case nir_op_f2u64:
   case nir_op_f2i64:
      FALLTHROUGH;
   default:
      assert(alu->dest.dest.is_ssa);
      if (alu->dest.dest.ssa.bit_size != 64)
         return false;
      break;
   }

   unsigned mask = nir_lower_int64_op_to_options_mask(alu->op);
   return (options->lower_int64_options & mask) != 0;
}

static nir_ssa_def *
split_64bit_subgroup_op(nir_builder *b, const nir_intrinsic_instr *intrin)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intrin->intrinsic];

   /* This works on subgroup ops with a single 64-bit source which can be
    * trivially lowered by doing the exact same op on both halves.
    */
   assert(intrin->src[0].is_ssa && intrin->src[0].ssa->bit_size == 64);
   nir_ssa_def *split_src0[2] = {
      nir_unpack_64_2x32_split_x(b, intrin->src[0].ssa),
      nir_unpack_64_2x32_split_y(b, intrin->src[0].ssa),
   };

   assert(info->has_dest && intrin->dest.is_ssa &&
          intrin->dest.ssa.bit_size == 64);

   nir_ssa_def *res[2];
   for (unsigned i = 0; i < 2; i++) {
      nir_intrinsic_instr *split =
         nir_intrinsic_instr_create(b->shader, intrin->intrinsic);
      split->num_components = intrin->num_components;
      split->src[0] = nir_src_for_ssa(split_src0[i]);

      /* Other sources must be less than 64 bits and get copied directly */
      for (unsigned j = 1; j < info->num_srcs; j++) {
         assert(intrin->src[j].is_ssa && intrin->src[j].ssa->bit_size < 64);
         split->src[j] = nir_src_for_ssa(intrin->src[j].ssa);
      }

      /* Copy const indices, if any */
      memcpy(split->const_index, intrin->const_index,
             sizeof(intrin->const_index));

      nir_ssa_dest_init(&split->instr, &split->dest,
                        intrin->dest.ssa.num_components, 32, NULL);
      nir_builder_instr_insert(b, &split->instr);

      res[i] = &split->dest.ssa;
   }

   return nir_pack_64_2x32_split(b, res[0], res[1]);
}

static nir_ssa_def *
build_vote_ieq(nir_builder *b, nir_ssa_def *x)
{
   nir_intrinsic_instr *vote =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_vote_ieq);
   vote->src[0] = nir_src_for_ssa(x);
   vote->num_components = x->num_components;
   nir_ssa_dest_init(&vote->instr, &vote->dest, 1, 1, NULL);
   nir_builder_instr_insert(b, &vote->instr);
   return &vote->dest.ssa;
}

static nir_ssa_def *
lower_vote_ieq(nir_builder *b, nir_ssa_def *x)
{
   return nir_iand(b, build_vote_ieq(b, nir_unpack_64_2x32_split_x(b, x)),
                      build_vote_ieq(b, nir_unpack_64_2x32_split_y(b, x)));
}

static nir_ssa_def *
build_scan_intrinsic(nir_builder *b, nir_intrinsic_op scan_op,
                     nir_op reduction_op, unsigned cluster_size,
                     nir_ssa_def *val)
{
   nir_intrinsic_instr *scan =
      nir_intrinsic_instr_create(b->shader, scan_op);
   scan->num_components = val->num_components;
   scan->src[0] = nir_src_for_ssa(val);
   nir_intrinsic_set_reduction_op(scan, reduction_op);
   if (scan_op == nir_intrinsic_reduce)
      nir_intrinsic_set_cluster_size(scan, cluster_size);
   nir_ssa_dest_init(&scan->instr, &scan->dest,
                     val->num_components, val->bit_size, NULL);
   nir_builder_instr_insert(b, &scan->instr);
   return &scan->dest.ssa;
}

static nir_ssa_def *
lower_scan_iadd64(nir_builder *b, const nir_intrinsic_instr *intrin)
{
   unsigned cluster_size =
      intrin->intrinsic == nir_intrinsic_reduce ?
      nir_intrinsic_cluster_size(intrin) : 0;

   /* Split it into three chunks of no more than 24 bits each.  With 8 bits
    * of headroom, we're guaranteed that there will never be overflow in the
    * individual subgroup operations.  (Assuming, of course, a subgroup size
    * no larger than 256 which seems reasonable.)  We can then scan on each of
    * the chunks and add them back together at the end.
    */
   assert(intrin->src[0].is_ssa);
   nir_ssa_def *x = intrin->src[0].ssa;
   nir_ssa_def *x_low =
      nir_u2u32(b, nir_iand_imm(b, x, 0xffffff));
   nir_ssa_def *x_mid =
      nir_u2u32(b, nir_iand_imm(b, nir_ushr(b, x, nir_imm_int(b, 24)),
                                   0xffffff));
   nir_ssa_def *x_hi =
      nir_u2u32(b, nir_ushr(b, x, nir_imm_int(b, 48)));

   nir_ssa_def *scan_low =
      build_scan_intrinsic(b, intrin->intrinsic, nir_op_iadd,
                              cluster_size, x_low);
   nir_ssa_def *scan_mid =
      build_scan_intrinsic(b, intrin->intrinsic, nir_op_iadd,
                              cluster_size, x_mid);
   nir_ssa_def *scan_hi =
      build_scan_intrinsic(b, intrin->intrinsic, nir_op_iadd,
                              cluster_size, x_hi);

   scan_low = nir_u2u64(b, scan_low);
   scan_mid = nir_ishl(b, nir_u2u64(b, scan_mid), nir_imm_int(b, 24));
   scan_hi = nir_ishl(b, nir_u2u64(b, scan_hi), nir_imm_int(b, 48));

   return nir_iadd(b, scan_hi, nir_iadd(b, scan_mid, scan_low));
}

static bool
should_lower_int64_intrinsic(const nir_intrinsic_instr *intrin,
                             const nir_shader_compiler_options *options)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
      assert(intrin->dest.is_ssa);
      return intrin->dest.ssa.bit_size == 64 &&
             (options->lower_int64_options & nir_lower_subgroup_shuffle64);

   case nir_intrinsic_vote_ieq:
      assert(intrin->src[0].is_ssa);
      return intrin->src[0].ssa->bit_size == 64 &&
             (options->lower_int64_options & nir_lower_vote_ieq64);

   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      assert(intrin->dest.is_ssa);
      if (intrin->dest.ssa.bit_size != 64)
         return false;

      switch (nir_intrinsic_reduction_op(intrin)) {
      case nir_op_iadd:
         return options->lower_int64_options & nir_lower_scan_reduce_iadd64;
      case nir_op_iand:
      case nir_op_ior:
      case nir_op_ixor:
         return options->lower_int64_options & nir_lower_scan_reduce_bitwise64;
      default:
         return false;
      }
      break;

   default:
      return false;
   }
}

static nir_ssa_def *
lower_int64_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
      return split_64bit_subgroup_op(b, intrin);

   case nir_intrinsic_vote_ieq:
      assert(intrin->src[0].is_ssa);
      return lower_vote_ieq(b, intrin->src[0].ssa);

   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      switch (nir_intrinsic_reduction_op(intrin)) {
      case nir_op_iadd:
         return lower_scan_iadd64(b, intrin);
      case nir_op_iand:
      case nir_op_ior:
      case nir_op_ixor:
         return split_64bit_subgroup_op(b, intrin);
      default:
         unreachable("Unsupported subgroup scan/reduce op");
      }
      break;

   default:
      unreachable("Unsupported intrinsic");
   }
}

static bool
should_lower_int64_instr(const nir_instr *instr, const void *_options)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return should_lower_int64_alu_instr(nir_instr_as_alu(instr), _options);
   case nir_instr_type_intrinsic:
      return should_lower_int64_intrinsic(nir_instr_as_intrinsic(instr),
                                          _options);
   default:
      return false;
   }
}

static nir_ssa_def *
lower_int64_instr(nir_builder *b, nir_instr *instr, void *_options)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return lower_int64_alu_instr(b, nir_instr_as_alu(instr));
   case nir_instr_type_intrinsic:
      return lower_int64_intrinsic(b, nir_instr_as_intrinsic(instr));
   default:
      return NULL;
   }
}

bool
nir_lower_int64(nir_shader *shader)
{
   return nir_shader_lower_instructions(shader, should_lower_int64_instr,
                                        lower_int64_instr,
                                        (void *)shader->options);
}
