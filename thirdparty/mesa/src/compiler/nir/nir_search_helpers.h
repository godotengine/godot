/*
 * Copyright © 2016 Red Hat
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
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#ifndef _NIR_SEARCH_HELPERS_
#define _NIR_SEARCH_HELPERS_

#include "nir.h"
#include "util/bitscan.h"
#include "nir_range_analysis.h"
#include <math.h>

static inline bool
is_pos_power_of_two(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                    unsigned src, unsigned num_components,
                    const uint8_t *swizzle)
{
   /* only constant srcs: */
   if (!nir_src_is_const(instr->src[src].src))
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      nir_alu_type type = nir_op_infos[instr->op].input_types[src];
      switch (nir_alu_type_get_base_type(type)) {
      case nir_type_int: {
         int64_t val = nir_src_comp_as_int(instr->src[src].src, swizzle[i]);
         if (val <= 0 || !util_is_power_of_two_or_zero64(val))
            return false;
         break;
      }
      case nir_type_uint: {
         uint64_t val = nir_src_comp_as_uint(instr->src[src].src, swizzle[i]);
         if (val == 0 || !util_is_power_of_two_or_zero64(val))
            return false;
         break;
      }
      default:
         return false;
      }
   }

   return true;
}

static inline bool
is_neg_power_of_two(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                    unsigned src, unsigned num_components,
                    const uint8_t *swizzle)
{
   /* only constant srcs: */
   if (!nir_src_is_const(instr->src[src].src))
      return false;

   int64_t int_min = u_intN_min(instr->src[src].src.ssa->bit_size);

   for (unsigned i = 0; i < num_components; i++) {
      nir_alu_type type = nir_op_infos[instr->op].input_types[src];
      switch (nir_alu_type_get_base_type(type)) {
      case nir_type_int: {
         int64_t val = nir_src_comp_as_int(instr->src[src].src, swizzle[i]);
         /* "int_min" is a power-of-two, but negation can cause overflow. */
         if (val == int_min || val >= 0 || !util_is_power_of_two_or_zero64(-val))
            return false;
         break;
      }
      default:
         return false;
      }
   }

   return true;
}

static inline bool
is_bitcount2(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
             unsigned src, unsigned num_components,
             const uint8_t *swizzle)
{
   /* only constant srcs: */
   if (!nir_src_is_const(instr->src[src].src))
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      uint64_t val = nir_src_comp_as_uint(instr->src[src].src, swizzle[i]);
      if (util_bitcount64(val) != 2)
         return false;
   }

   return true;
}

#define MULTIPLE(test)                                                  \
static inline bool                                                      \
is_unsigned_multiple_of_ ## test(UNUSED struct hash_table *ht,          \
                                 const nir_alu_instr *instr,            \
                                 unsigned src, unsigned num_components, \
                                 const uint8_t *swizzle)                \
{                                                                       \
   /* only constant srcs: */                                            \
   if (!nir_src_is_const(instr->src[src].src))                          \
      return false;                                                     \
                                                                        \
   for (unsigned i = 0; i < num_components; i++) {                      \
      uint64_t val = nir_src_comp_as_uint(instr->src[src].src, swizzle[i]); \
      if (val % test != 0)                                              \
         return false;                                                  \
   }                                                                    \
                                                                        \
   return true;                                                         \
}

MULTIPLE(2)
MULTIPLE(4)
MULTIPLE(8)
MULTIPLE(16)
MULTIPLE(32)
MULTIPLE(64)

static inline bool
is_zero_to_one(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
               unsigned src, unsigned num_components,
               const uint8_t *swizzle)
{
   /* only constant srcs: */
   if (!nir_src_is_const(instr->src[src].src))
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      nir_alu_type type = nir_op_infos[instr->op].input_types[src];
      switch (nir_alu_type_get_base_type(type)) {
      case nir_type_float: {
         double val = nir_src_comp_as_float(instr->src[src].src, swizzle[i]);
         if (isnan(val) || val < 0.0f || val > 1.0f)
            return false;
         break;
      }
      default:
         return false;
      }
   }

   return true;
}

/**
 * Exclusive compare with (0, 1).
 *
 * This differs from \c is_zero_to_one because that function tests 0 <= src <=
 * 1 while this function tests 0 < src < 1.
 */
static inline bool
is_gt_0_and_lt_1(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                 unsigned src, unsigned num_components,
                 const uint8_t *swizzle)
{
   /* only constant srcs: */
   if (!nir_src_is_const(instr->src[src].src))
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      nir_alu_type type = nir_op_infos[instr->op].input_types[src];
      switch (nir_alu_type_get_base_type(type)) {
      case nir_type_float: {
         double val = nir_src_comp_as_float(instr->src[src].src, swizzle[i]);
         if (isnan(val) || val <= 0.0f || val >= 1.0f)
            return false;
         break;
      }
      default:
         return false;
      }
   }

   return true;
}

static inline bool
is_not_const_zero(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                  unsigned src, unsigned num_components,
                  const uint8_t *swizzle)
{
   if (nir_src_as_const_value(instr->src[src].src) == NULL)
      return true;

   for (unsigned i = 0; i < num_components; i++) {
      nir_alu_type type = nir_op_infos[instr->op].input_types[src];
      switch (nir_alu_type_get_base_type(type)) {
      case nir_type_float:
         if (nir_src_comp_as_float(instr->src[src].src, swizzle[i]) == 0.0)
            return false;
         break;
      case nir_type_bool:
      case nir_type_int:
      case nir_type_uint:
         if (nir_src_comp_as_uint(instr->src[src].src, swizzle[i]) == 0)
            return false;
         break;
      default:
         return false;
      }
   }

   return true;
}

/** Is value unsigned less than the limit? */
static inline bool
is_ult(const nir_alu_instr *instr, unsigned src, unsigned num_components, const uint8_t *swizzle,
       uint64_t limit)
{
   /* only constant srcs: */
   if (!nir_src_is_const(instr->src[src].src))
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      const uint64_t val =
         nir_src_comp_as_uint(instr->src[src].src, swizzle[i]);

      if (val >= limit)
         return false;
   }

   return true;
}

/** Is value unsigned less than 32? */
static inline bool
is_ult_32(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
          unsigned src, unsigned num_components,
          const uint8_t *swizzle)
{
   return is_ult(instr, src, num_components, swizzle, 32);
}

/** Is value unsigned less than 0xfffc07fc? */
static inline bool
is_ult_0xfffc07fc(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                  unsigned src, unsigned num_components,
                  const uint8_t *swizzle)
{
   return is_ult(instr, src, num_components, swizzle, 0xfffc07fcU);
}

/** Is the first 5 bits of value unsigned greater than or equal 2? */
static inline bool
is_first_5_bits_uge_2(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                      unsigned src, unsigned num_components,
                      const uint8_t *swizzle)
{
   /* only constant srcs: */
   if (!nir_src_is_const(instr->src[src].src))
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      const unsigned val =
         nir_src_comp_as_uint(instr->src[src].src, swizzle[i]);

      if ((val & 0x1f) < 2)
         return false;
   }

   return true;
}

static inline bool
is_not_const(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
             unsigned src, UNUSED unsigned num_components,
             UNUSED const uint8_t *swizzle)
{
   return !nir_src_is_const(instr->src[src].src);
}

static inline bool
is_not_fmul(struct hash_table *ht, const nir_alu_instr *instr, unsigned src,
            UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   nir_alu_instr *src_alu =
      nir_src_as_alu_instr(instr->src[src].src);

   if (src_alu == NULL)
      return true;

   if (src_alu->op == nir_op_fneg)
      return is_not_fmul(ht, src_alu, 0, 0, NULL);

   return src_alu->op != nir_op_fmul && src_alu->op != nir_op_fmulz;
}

static inline bool
is_fmul(struct hash_table *ht, const nir_alu_instr *instr, unsigned src,
        UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   nir_alu_instr *src_alu =
      nir_src_as_alu_instr(instr->src[src].src);

   if (src_alu == NULL)
      return false;

   if (src_alu->op == nir_op_fneg)
      return is_fmul(ht, src_alu, 0, 0, NULL);

   return src_alu->op == nir_op_fmul || src_alu->op == nir_op_fmulz;
}

static inline bool
is_fsign(const nir_alu_instr *instr, unsigned src,
         UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   nir_alu_instr *src_alu =
      nir_src_as_alu_instr(instr->src[src].src);

   if (src_alu == NULL)
      return false;

   if (src_alu->op == nir_op_fneg)
      src_alu = nir_src_as_alu_instr(src_alu->src[0].src);

   return src_alu != NULL && src_alu->op == nir_op_fsign;
}

static inline bool
is_not_const_and_not_fsign(struct hash_table *ht, const nir_alu_instr *instr,
                           unsigned src, unsigned num_components,
                           const uint8_t *swizzle)
{
   return is_not_const(ht, instr, src, num_components, swizzle) &&
          !is_fsign(instr, src, num_components, swizzle);
}

static inline bool
is_used_once(const nir_alu_instr *instr)
{
   bool zero_if_use = list_is_empty(&instr->dest.dest.ssa.if_uses);
   bool zero_use = list_is_empty(&instr->dest.dest.ssa.uses);

   if (zero_if_use && zero_use)
      return false;

   if (!zero_if_use && list_is_singular(&instr->dest.dest.ssa.uses))
     return false;

   if (!zero_use && list_is_singular(&instr->dest.dest.ssa.if_uses))
     return false;

   if (!list_is_singular(&instr->dest.dest.ssa.if_uses) &&
       !list_is_singular(&instr->dest.dest.ssa.uses))
      return false;

   return true;
}

static inline bool
is_used_by_if(const nir_alu_instr *instr)
{
   return !list_is_empty(&instr->dest.dest.ssa.if_uses);
}

static inline bool
is_not_used_by_if(const nir_alu_instr *instr)
{
   return list_is_empty(&instr->dest.dest.ssa.if_uses);
}

static inline bool
is_used_by_non_fsat(const nir_alu_instr *instr)
{
   nir_foreach_use(src, &instr->dest.dest.ssa) {
      const nir_instr *const user_instr = src->parent_instr;

      if (user_instr->type != nir_instr_type_alu)
         return true;

      const nir_alu_instr *const user_alu = nir_instr_as_alu(user_instr);

      assert(instr != user_alu);
      if (user_alu->op != nir_op_fsat)
         return true;
   }

   return false;
}

static inline bool
is_only_used_as_float(const nir_alu_instr *instr)
{
   nir_foreach_use(src, &instr->dest.dest.ssa) {
      const nir_instr *const user_instr = src->parent_instr;
      if (user_instr->type != nir_instr_type_alu)
         return false;

      const nir_alu_instr *const user_alu = nir_instr_as_alu(user_instr);
      assert(instr != user_alu);

      unsigned index = (nir_alu_src*)container_of(src, nir_alu_src, src) - user_alu->src;
      nir_alu_type type = nir_op_infos[user_alu->op].input_types[index];
      if (nir_alu_type_get_base_type(type) != nir_type_float)
         return false;
   }

   return true;
}

static inline bool
is_only_used_by_fadd(const nir_alu_instr *instr)
{
   nir_foreach_use(src, &instr->dest.dest.ssa) {
      const nir_instr *const user_instr = src->parent_instr;
      if (user_instr->type != nir_instr_type_alu)
         return false;

      const nir_alu_instr *const user_alu = nir_instr_as_alu(user_instr);
      assert(instr != user_alu);

      if (user_alu->op == nir_op_fneg || user_alu->op == nir_op_fabs) {
         if (!is_only_used_by_fadd(user_alu))
            return false;
      } else if (user_alu->op != nir_op_fadd) {
         return false;
      }
   }

   return true;
}

static inline bool
only_lower_8_bits_used(const nir_alu_instr *instr)
{
   return (nir_ssa_def_bits_used(&instr->dest.dest.ssa) & ~0xffull) == 0;
}

static inline bool
only_lower_16_bits_used(const nir_alu_instr *instr)
{
   return (nir_ssa_def_bits_used(&instr->dest.dest.ssa) & ~0xffffull) == 0;
}

/**
 * Returns true if a NIR ALU src represents a constant integer
 * of either 32 or 64 bits, and the higher word (bit-size / 2)
 * of all its components is zero.
 */
static inline bool
is_upper_half_zero(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                   unsigned src, unsigned num_components,
                   const uint8_t *swizzle)
{
   if (nir_src_as_const_value(instr->src[src].src) == NULL)
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      unsigned half_bit_size = nir_src_bit_size(instr->src[src].src) / 2;
      uint64_t high_bits = u_bit_consecutive64(half_bit_size, half_bit_size);
      if ((nir_src_comp_as_uint(instr->src[src].src,
                                swizzle[i]) & high_bits) != 0) {
         return false;
      }
   }

   return true;
}

/**
 * Returns true if a NIR ALU src represents a constant integer
 * of either 32 or 64 bits, and the lower word (bit-size / 2)
 * of all its components is zero.
 */
static inline bool
is_lower_half_zero(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                   unsigned src, unsigned num_components,
                   const uint8_t *swizzle)
{
   if (nir_src_as_const_value(instr->src[src].src) == NULL)
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      uint64_t low_bits = u_bit_consecutive64(0, nir_src_bit_size(instr->src[src].src) / 2);
      if ((nir_src_comp_as_uint(instr->src[src].src, swizzle[i]) & low_bits) != 0)
         return false;
   }

   return true;
}

static inline bool
is_upper_half_negative_one(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                           unsigned src, unsigned num_components,
                           const uint8_t *swizzle)
{
   if (nir_src_as_const_value(instr->src[src].src) == NULL)
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      unsigned half_bit_size = nir_src_bit_size(instr->src[src].src) / 2;
      uint64_t high_bits = u_bit_consecutive64(half_bit_size, half_bit_size);
      if ((nir_src_comp_as_uint(instr->src[src].src,
                                swizzle[i]) & high_bits) != high_bits) {
         return false;
      }
   }

   return true;
}

static inline bool
is_lower_half_negative_one(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                           unsigned src, unsigned num_components,
                           const uint8_t *swizzle)
{
   if (nir_src_as_const_value(instr->src[src].src) == NULL)
      return false;

   for (unsigned i = 0; i < num_components; i++) {
      uint64_t low_bits = u_bit_consecutive64(0, nir_src_bit_size(instr->src[src].src) / 2);
      if ((nir_src_comp_as_uint(instr->src[src].src, swizzle[i]) & low_bits) != low_bits)
         return false;
   }

   return true;
}

static inline bool
no_signed_wrap(const nir_alu_instr *instr)
{
   return instr->no_signed_wrap;
}

static inline bool
no_unsigned_wrap(const nir_alu_instr *instr)
{
   return instr->no_unsigned_wrap;
}

static inline bool
is_integral(struct hash_table *ht, const nir_alu_instr *instr, unsigned src,
            UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range r = nir_analyze_range(ht, instr, src);

   return r.is_integral;
}

/**
 * Is the value finite?
 */
static inline bool
is_finite(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
          unsigned src, UNUSED unsigned num_components,
          UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);

   return v.is_finite;
}

static inline bool
is_finite_not_zero(UNUSED struct hash_table *ht, const nir_alu_instr *instr,
                   unsigned src, UNUSED unsigned num_components,
                   UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);

   return v.is_finite &&
          (v.range == lt_zero || v.range == gt_zero || v.range == ne_zero);
}


#define RELATION(r)                                                     \
static inline bool                                                      \
is_ ## r (struct hash_table *ht, const nir_alu_instr *instr,            \
          unsigned src, UNUSED unsigned num_components,                 \
          UNUSED const uint8_t *swizzle)                                \
{                                                                       \
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);  \
   return v.range == r;                                                 \
}                                                                       \
                                                                        \
static inline bool                                                      \
is_a_number_ ## r (struct hash_table *ht, const nir_alu_instr *instr,   \
                   unsigned src, UNUSED unsigned num_components,        \
                   UNUSED const uint8_t *swizzle)                       \
{                                                                       \
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src); \
   return v.is_a_number && v.range == r;                                \
}

RELATION(lt_zero)
RELATION(le_zero)
RELATION(gt_zero)
RELATION(ge_zero)
RELATION(ne_zero)

static inline bool
is_not_negative(struct hash_table *ht, const nir_alu_instr *instr, unsigned src,
                UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);
   return v.range == ge_zero || v.range == gt_zero || v.range == eq_zero;
}

static inline bool
is_a_number_not_negative(struct hash_table *ht, const nir_alu_instr *instr,
                         unsigned src, UNUSED unsigned num_components,
                         UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);
   return v.is_a_number &&
          (v.range == ge_zero || v.range == gt_zero || v.range == eq_zero);
}


static inline bool
is_not_positive(struct hash_table *ht, const nir_alu_instr *instr, unsigned src,
                UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);
   return v.range == le_zero || v.range == lt_zero || v.range == eq_zero;
}

static inline bool
is_a_number_not_positive(struct hash_table *ht, const nir_alu_instr *instr,
                         unsigned src, UNUSED unsigned num_components,
                         UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);
   return v.is_a_number &&
          (v.range == le_zero || v.range == lt_zero || v.range == eq_zero);
}

static inline bool
is_not_zero(struct hash_table *ht, const nir_alu_instr *instr, unsigned src,
            UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);
   return v.range == lt_zero || v.range == gt_zero || v.range == ne_zero;
}

static inline bool
is_a_number_not_zero(struct hash_table *ht, const nir_alu_instr *instr,
                     unsigned src, UNUSED unsigned num_components,
                     UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);
   return v.is_a_number &&
          (v.range == lt_zero || v.range == gt_zero || v.range == ne_zero);
}

static inline bool
is_a_number(struct hash_table *ht, const nir_alu_instr *instr, unsigned src,
            UNUSED unsigned num_components, UNUSED const uint8_t *swizzle)
{
   const struct ssa_result_range v = nir_analyze_range(ht, instr, src);
   return v.is_a_number;
}

#endif /* _NIR_SEARCH_ */
