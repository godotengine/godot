/*
 * Copyright © 2018 Intel Corporation
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
#include <math.h>
#include "nir.h"
#include "nir_builder.h"
#include "util/u_vector.h"

/**
 * Lower flrp instructions.
 *
 * Unlike the lowerings that are possible in nir_opt_algrbraic, this pass can
 * examine more global information to determine a possibly more efficient
 * lowering for each flrp.
 */

static void
append_flrp_to_dead_list(struct u_vector *dead_flrp, struct nir_alu_instr *alu)
{
   struct nir_alu_instr **tail = u_vector_add(dead_flrp);
   *tail = alu;
}

/**
 * Replace flrp(a, b, c) with ffma(b, c, ffma(-a, c, a)).
 */
static void
replace_with_strict_ffma(struct nir_builder *bld, struct u_vector *dead_flrp,
                         struct nir_alu_instr *alu)
{
   nir_ssa_def *const a = nir_ssa_for_alu_src(bld, alu, 0);
   nir_ssa_def *const b = nir_ssa_for_alu_src(bld, alu, 1);
   nir_ssa_def *const c = nir_ssa_for_alu_src(bld, alu, 2);

   nir_ssa_def *const neg_a = nir_fneg(bld, a);
   nir_instr_as_alu(neg_a->parent_instr)->exact = alu->exact;

   nir_ssa_def *const inner_ffma = nir_ffma(bld, neg_a, c, a);
   nir_instr_as_alu(inner_ffma->parent_instr)->exact = alu->exact;

   nir_ssa_def *const outer_ffma = nir_ffma(bld, b, c, inner_ffma);
   nir_instr_as_alu(outer_ffma->parent_instr)->exact = alu->exact;

   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, outer_ffma);

   /* DO NOT REMOVE the original flrp yet.  Many of the lowering choices are
    * based on other uses of the sources.  Removing the flrp may cause the
    * last flrp in a sequence to make a different, incorrect choice.
    */
   append_flrp_to_dead_list(dead_flrp, alu);
}

/**
 * Replace flrp(a, b, c) with ffma(a, (1 - c), bc)
 */
static void
replace_with_single_ffma(struct nir_builder *bld, struct u_vector *dead_flrp,
                         struct nir_alu_instr *alu)
{
   nir_ssa_def *const a = nir_ssa_for_alu_src(bld, alu, 0);
   nir_ssa_def *const b = nir_ssa_for_alu_src(bld, alu, 1);
   nir_ssa_def *const c = nir_ssa_for_alu_src(bld, alu, 2);

   nir_ssa_def *const neg_c = nir_fneg(bld, c);
   nir_instr_as_alu(neg_c->parent_instr)->exact = alu->exact;

   nir_ssa_def *const one_minus_c =
      nir_fadd(bld, nir_imm_floatN_t(bld, 1.0f, c->bit_size), neg_c);
   nir_instr_as_alu(one_minus_c->parent_instr)->exact = alu->exact;

   nir_ssa_def *const b_times_c = nir_fmul(bld, b, c);
   nir_instr_as_alu(b_times_c->parent_instr)->exact = alu->exact;

   nir_ssa_def *const final_ffma = nir_ffma(bld, a, one_minus_c, b_times_c);
   nir_instr_as_alu(final_ffma->parent_instr)->exact = alu->exact;

   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, final_ffma);

   /* DO NOT REMOVE the original flrp yet.  Many of the lowering choices are
    * based on other uses of the sources.  Removing the flrp may cause the
    * last flrp in a sequence to make a different, incorrect choice.
    */
   append_flrp_to_dead_list(dead_flrp, alu);
}

/**
 * Replace flrp(a, b, c) with a(1-c) + bc.
 */
static void
replace_with_strict(struct nir_builder *bld, struct u_vector *dead_flrp,
                    struct nir_alu_instr *alu)
{
   nir_ssa_def *const a = nir_ssa_for_alu_src(bld, alu, 0);
   nir_ssa_def *const b = nir_ssa_for_alu_src(bld, alu, 1);
   nir_ssa_def *const c = nir_ssa_for_alu_src(bld, alu, 2);

   nir_ssa_def *const neg_c = nir_fneg(bld, c);
   nir_instr_as_alu(neg_c->parent_instr)->exact = alu->exact;

   nir_ssa_def *const one_minus_c =
      nir_fadd(bld, nir_imm_floatN_t(bld, 1.0f, c->bit_size), neg_c);
   nir_instr_as_alu(one_minus_c->parent_instr)->exact = alu->exact;

   nir_ssa_def *const first_product = nir_fmul(bld, a, one_minus_c);
   nir_instr_as_alu(first_product->parent_instr)->exact = alu->exact;

   nir_ssa_def *const second_product = nir_fmul(bld, b, c);
   nir_instr_as_alu(second_product->parent_instr)->exact = alu->exact;

   nir_ssa_def *const sum = nir_fadd(bld, first_product, second_product);
   nir_instr_as_alu(sum->parent_instr)->exact = alu->exact;

   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, sum);

   /* DO NOT REMOVE the original flrp yet.  Many of the lowering choices are
    * based on other uses of the sources.  Removing the flrp may cause the
    * last flrp in a sequence to make a different, incorrect choice.
    */
   append_flrp_to_dead_list(dead_flrp, alu);
}

/**
 * Replace flrp(a, b, c) with a + c(b-a).
 */
static void
replace_with_fast(struct nir_builder *bld, struct u_vector *dead_flrp,
                  struct nir_alu_instr *alu)
{
   nir_ssa_def *const a = nir_ssa_for_alu_src(bld, alu, 0);
   nir_ssa_def *const b = nir_ssa_for_alu_src(bld, alu, 1);
   nir_ssa_def *const c = nir_ssa_for_alu_src(bld, alu, 2);

   nir_ssa_def *const neg_a = nir_fneg(bld, a);
   nir_instr_as_alu(neg_a->parent_instr)->exact = alu->exact;

   nir_ssa_def *const b_minus_a = nir_fadd(bld, b, neg_a);
   nir_instr_as_alu(b_minus_a->parent_instr)->exact = alu->exact;

   nir_ssa_def *const product = nir_fmul(bld, c, b_minus_a);
   nir_instr_as_alu(product->parent_instr)->exact = alu->exact;

   nir_ssa_def *const sum = nir_fadd(bld, a, product);
   nir_instr_as_alu(sum->parent_instr)->exact = alu->exact;

   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, sum);

   /* DO NOT REMOVE the original flrp yet.  Many of the lowering choices are
    * based on other uses of the sources.  Removing the flrp may cause the
    * last flrp in a sequence to make a different, incorrect choice.
    */
   append_flrp_to_dead_list(dead_flrp, alu);
}

/**
 * Replace flrp(a, b, c) with (b*c ± c) + a => b*c + (a ± c)
 *
 * \note: This only works if a = ±1.
 */
static void
replace_with_expanded_ffma_and_add(struct nir_builder *bld,
                                   struct u_vector *dead_flrp,
                                   struct nir_alu_instr *alu, bool subtract_c)
{
   nir_ssa_def *const a = nir_ssa_for_alu_src(bld, alu, 0);
   nir_ssa_def *const b = nir_ssa_for_alu_src(bld, alu, 1);
   nir_ssa_def *const c = nir_ssa_for_alu_src(bld, alu, 2);

   nir_ssa_def *const b_times_c = nir_fmul(bld, b, c);
   nir_instr_as_alu(b_times_c->parent_instr)->exact = alu->exact;

   nir_ssa_def *inner_sum;

   if (subtract_c) {
      nir_ssa_def *const neg_c = nir_fneg(bld, c);
      nir_instr_as_alu(neg_c->parent_instr)->exact = alu->exact;

      inner_sum = nir_fadd(bld, a, neg_c);
   } else {
      inner_sum = nir_fadd(bld, a, c);
   }

   nir_instr_as_alu(inner_sum->parent_instr)->exact = alu->exact;

   nir_ssa_def *const outer_sum = nir_fadd(bld, inner_sum, b_times_c);
   nir_instr_as_alu(outer_sum->parent_instr)->exact = alu->exact;

   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, outer_sum);

   /* DO NOT REMOVE the original flrp yet.  Many of the lowering choices are
    * based on other uses of the sources.  Removing the flrp may cause the
    * last flrp in a sequence to make a different, incorrect choice.
    */
   append_flrp_to_dead_list(dead_flrp, alu);
}

/**
 * Determines whether a swizzled source is constant w/ all components the same.
 *
 * The value of the constant is stored in \c result.
 *
 * \return
 * True if all components of the swizzled source are the same constant.
 * Otherwise false is returned.
 */
static bool
all_same_constant(const nir_alu_instr *instr, unsigned src, double *result)
{
   nir_const_value *val = nir_src_as_const_value(instr->src[src].src);

   if (!val)
      return false;

   const uint8_t *const swizzle = instr->src[src].swizzle;
   const unsigned num_components = nir_dest_num_components(instr->dest.dest);

   if (instr->dest.dest.ssa.bit_size == 32) {
      const float first = val[swizzle[0]].f32;

      for (unsigned i = 1; i < num_components; i++) {
         if (val[swizzle[i]].f32 != first)
            return false;
      }

      *result = first;
   } else {
      const double first = val[swizzle[0]].f64;

      for (unsigned i = 1; i < num_components; i++) {
         if (val[swizzle[i]].f64 != first)
            return false;
      }

      *result = first;
   }

   return true;
}

static bool
sources_are_constants_with_similar_magnitudes(const nir_alu_instr *instr)
{
   nir_const_value *val0 = nir_src_as_const_value(instr->src[0].src);
   nir_const_value *val1 = nir_src_as_const_value(instr->src[1].src);

   if (val0 == NULL || val1 == NULL)
      return false;

   const uint8_t *const swizzle0 = instr->src[0].swizzle;
   const uint8_t *const swizzle1 = instr->src[1].swizzle;
   const unsigned num_components = nir_dest_num_components(instr->dest.dest);

   if (instr->dest.dest.ssa.bit_size == 32) {
      for (unsigned i = 0; i < num_components; i++) {
         int exp0;
         int exp1;

         frexpf(val0[swizzle0[i]].f32, &exp0);
         frexpf(val1[swizzle1[i]].f32, &exp1);

         /* If the difference between exponents is >= 24, then A+B will always
          * have the value whichever between A and B has the largest absolute
          * value.  So, [0, 23] is the valid range.  The smaller the limit
          * value, the more precision will be maintained at a potential
          * performance cost.  Somewhat arbitrarilly split the range in half.
          */
         if (abs(exp0 - exp1) > (23 / 2))
            return false;
      }
   } else {
      for (unsigned i = 0; i < num_components; i++) {
         int exp0;
         int exp1;

         frexp(val0[swizzle0[i]].f64, &exp0);
         frexp(val1[swizzle1[i]].f64, &exp1);

         /* If the difference between exponents is >= 53, then A+B will always
          * have the value whichever between A and B has the largest absolute
          * value.  So, [0, 52] is the valid range.  The smaller the limit
          * value, the more precision will be maintained at a potential
          * performance cost.  Somewhat arbitrarilly split the range in half.
          */
         if (abs(exp0 - exp1) > (52 / 2))
            return false;
      }
   }

   return true;
}

/**
 * Counts of similar types of nir_op_flrp instructions
 *
 * If a similar instruction fits into more than one category, it will only be
 * counted once.  The assumption is that no other instruction will have all
 * sources the same, or CSE would have removed one of the instructions.
 */
struct similar_flrp_stats {
   unsigned src2;
   unsigned src0_and_src2;
   unsigned src1_and_src2;
};

/**
 * Collection counts of similar FLRP instructions.
 *
 * This function only cares about similar instructions that have src2 in
 * common.
 */
static void
get_similar_flrp_stats(nir_alu_instr *alu, struct similar_flrp_stats *st)
{
   memset(st, 0, sizeof(*st));

   nir_foreach_use(other_use, alu->src[2].src.ssa) {
      /* Is the use also a flrp? */
      nir_instr *const other_instr = other_use->parent_instr;
      if (other_instr->type != nir_instr_type_alu)
         continue;

      /* Eh-hem... don't match the instruction with itself. */
      if (other_instr == &alu->instr)
         continue;

      nir_alu_instr *const other_alu = nir_instr_as_alu(other_instr);
      if (other_alu->op != nir_op_flrp)
         continue;

      /* Does the other flrp use source 2 from the first flrp as its source 2
       * as well?
       */
      if (!nir_alu_srcs_equal(alu, other_alu, 2, 2))
         continue;

      if (nir_alu_srcs_equal(alu, other_alu, 0, 0))
         st->src0_and_src2++;
      else if (nir_alu_srcs_equal(alu, other_alu, 1, 1))
         st->src1_and_src2++;
      else
         st->src2++;
   }
}

static void
convert_flrp_instruction(nir_builder *bld,
                         struct u_vector *dead_flrp,
                         nir_alu_instr *alu,
                         bool always_precise)
{
   bool have_ffma = false;
   unsigned bit_size = nir_dest_bit_size(alu->dest.dest);

   if (bit_size == 16)
      have_ffma = !bld->shader->options->lower_ffma16;
   else if (bit_size == 32)
      have_ffma = !bld->shader->options->lower_ffma32;
   else if (bit_size == 64)
      have_ffma = !bld->shader->options->lower_ffma64;
   else
      unreachable("invalid bit_size");

   bld->cursor = nir_before_instr(&alu->instr);

   /* There are two methods to implement flrp(x, y, t).  The strictly correct
    * implementation according to the GLSL spec is:
    *
    *    x(1 - t) + yt
    *
    * This can also be implemented using two chained FMAs
    *
    *    fma(y, t, fma(-x, t, x))
    *
    * This method, using either formulation, has better precision when the
    * difference between x and y is very large.  It guarantess that flrp(x, y,
    * 1) = y.  For example, flrp(1e38, 1.0, 1.0) is 1.0.  This is correct.
    *
    * The other possible implementation is:
    *
    *    x + t(y - x)
    *
    * This can also be formuated as an FMA:
    *
    *    fma(y - x, t, x)
    *
    * For this implementation, flrp(1e38, 1.0, 1.0) is 0.0.  Since 1.0 was
    * expected, that's a pretty significant error.
    *
    * The choice made for lowering depends on a number of factors.
    *
    * - If the flrp is marked precise and FMA is supported:
    *
    *        fma(y, t, fma(-x, t, x))
    *
    *   This is strictly correct (maybe?), and the cost is two FMA
    *   instructions.  It at least maintains the flrp(x, y, 1.0) == y
    *   condition.
    *
    * - If the flrp is marked precise and FMA is not supported:
    *
    *        x(1 - t) + yt
    *
    *   This is strictly correct, and the cost is 4 instructions.  If FMA is
    *   supported, this may or may not be reduced to 3 instructions (a
    *   subtract, a multiply, and an FMA)... but in that case the other
    *   formulation should have been used.
    */
   if (alu->exact) {
      if (have_ffma)
         replace_with_strict_ffma(bld, dead_flrp, alu);
      else
         replace_with_strict(bld, dead_flrp, alu);

      return;
   }

   /*
    * - If x and y are both immediates and the relative magnitude of the
    *   values is similar (such that x-y does not lose too much precision):
    *
    *        x + t(x - y)
    *
    *   We rely on constant folding to eliminate x-y, and we rely on
    *   nir_opt_algebraic to possibly generate an FMA.  The cost is either one
    *   FMA or two instructions.
    */
   if (sources_are_constants_with_similar_magnitudes(alu)) {
      replace_with_fast(bld, dead_flrp, alu);
      return;
   }

   /*
    * - If x = 1:
    *
    *        (yt + -t) + 1
    *
    * - If x = -1:
    *
    *        (yt + t) - 1
    *
    *   In both cases, x is used in place of ±1 for simplicity.  Both forms
    *   lend to ffma generation on platforms that support ffma.
    */
   double src0_as_constant;
   if (all_same_constant(alu, 0, &src0_as_constant)) {
      if (src0_as_constant == 1.0) {
         replace_with_expanded_ffma_and_add(bld, dead_flrp, alu,
                                            true /* subtract t */);
         return;
      } else if (src0_as_constant == -1.0) {
         replace_with_expanded_ffma_and_add(bld, dead_flrp, alu,
                                            false /* add t */);
         return;
      }
   }

   /*
    * - If y = ±1:
    *
    *        x(1 - t) + yt
    *
    *   In this case either the multiply in yt will be eliminated by
    *   nir_opt_algebraic.  If FMA is supported, this results in fma(x, (1 -
    *   t), ±t) for two instructions.  If FMA is not supported, then the cost
    *   is 3 instructions.  We rely on nir_opt_algebraic to generate the FMA
    *   instructions as well.
    *
    *   Another possible replacement is
    *
    *        -xt + x ± t
    *
    *   Some groupings of this may be better on some platforms in some
    *   circumstances, bit it is probably dependent on scheduling.  Futher
    *   investigation may be required.
    */
   double src1_as_constant;
   if ((all_same_constant(alu, 1, &src1_as_constant) &&
        (src1_as_constant == -1.0 || src1_as_constant == 1.0))) {
      replace_with_strict(bld, dead_flrp, alu);
      return;
   }

   if (have_ffma) {
      if (always_precise) {
         replace_with_strict_ffma(bld, dead_flrp, alu);
         return;
      }

      /*
       * - If FMA is supported and other flrp(x, _, t) exists:
       *
       *        fma(y, t, fma(-x, t, x))
       *
       *   The hope is that the inner FMA calculation will be shared with the
       *   other lowered flrp.  This results in two FMA instructions for the
       *   first flrp and one FMA instruction for each additional flrp.  It
       *   also means that the live range for x might be complete after the
       *   inner ffma instead of after the last flrp.
       */
      struct similar_flrp_stats st;

      get_similar_flrp_stats(alu, &st);
      if (st.src0_and_src2 > 0) {
         replace_with_strict_ffma(bld, dead_flrp, alu);
         return;
      }

      /*
       * - If FMA is supported and another flrp(_, y, t) exists:
       *
       *        fma(x, (1 - t), yt)
       *
       *   The hope is that the (1 - t) and the yt will be shared with the
       *   other lowered flrp.  This results in 3 insructions for the first
       *   flrp and 1 for each additional flrp.
       */
      if (st.src1_and_src2 > 0) {
         replace_with_single_ffma(bld, dead_flrp, alu);
         return;
      }
   } else {
      if (always_precise) {
         replace_with_strict(bld, dead_flrp, alu);
         return;
      }

      /*
       * - If FMA is not supported and another flrp(x, _, t) exists:
       *
       *        x(1 - t) + yt
       *
       *   The hope is that the x(1 - t) will be shared with the other lowered
       *   flrp.  This results in 4 insructions for the first flrp and 2 for
       *   each additional flrp.
       *
       * - If FMA is not supported and another flrp(_, y, t) exists:
       *
       *        x(1 - t) + yt
       *
       *   The hope is that the (1 - t) and the yt will be shared with the
       *   other lowered flrp.  This results in 4 insructions for the first
       *   flrp and 2 for each additional flrp.
       */
      struct similar_flrp_stats st;

      get_similar_flrp_stats(alu, &st);
      if (st.src0_and_src2 > 0 || st.src1_and_src2 > 0) {
         replace_with_strict(bld, dead_flrp, alu);
         return;
      }
   }

   /*
    * - If t is constant:
    *
    *        x(1 - t) + yt
    *
    *   The cost is three instructions without FMA or two instructions with
    *   FMA.  This is the same cost as the imprecise lowering, but it gives
    *   the instruction scheduler a little more freedom.
    *
    *   There is no need to handle t = 0.5 specially.  nir_opt_algebraic
    *   already has optimizations to convert 0.5x + 0.5y to 0.5(x + y).
    */
   if (alu->src[2].src.ssa->parent_instr->type == nir_instr_type_load_const) {
      replace_with_strict(bld, dead_flrp, alu);
      return;
   }

   /*
    * - Otherwise
    *
    *        x + t(x - y)
    */
   replace_with_fast(bld, dead_flrp, alu);
}

static void
lower_flrp_impl(nir_function_impl *impl,
                struct u_vector *dead_flrp,
                unsigned lowering_mask,
                bool always_precise)
{
   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *const alu = nir_instr_as_alu(instr);

            if (alu->op == nir_op_flrp &&
                (alu->dest.dest.ssa.bit_size & lowering_mask)) {
               convert_flrp_instruction(&b, dead_flrp, alu, always_precise);
            }
         }
      }
   }

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);
}

/**
 * \param lowering_mask - Bitwise-or of the bit sizes that need to be lowered
 *                        (e.g., 16 | 64 if only 16-bit and 64-bit flrp need
 *                        lowering).
 * \param always_precise - Always require precise lowering for flrp.  This
 *                        will always lower flrp to (a * (1 - c)) + (b * c).
 * \param have_ffma - Set to true if the GPU has an FFMA instruction that
 *                    should be used.
 */
bool
nir_lower_flrp(nir_shader *shader,
               unsigned lowering_mask,
               bool always_precise)
{
   struct u_vector dead_flrp;

   if (!u_vector_init_pow2(&dead_flrp, 8, sizeof(struct nir_alu_instr *)))
      return false;

   nir_foreach_function(function, shader) {
      if (function->impl) {
         lower_flrp_impl(function->impl, &dead_flrp, lowering_mask,
                         always_precise);
      }
   }

   /* Progress was made if the dead list is not empty.  Remove all the
    * instructions from the dead list.
    */
   const bool progress = u_vector_length(&dead_flrp) != 0;

   struct nir_alu_instr **instr;
   u_vector_foreach(instr, &dead_flrp)
      nir_instr_remove(&(*instr)->instr);

   u_vector_finish(&dead_flrp);

   return progress;
}
