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

#include "nir.h"
#include "nir_builder.h"
#include "util/fast_idiv_by_const.h"
#include "util/u_math.h"

static nir_ssa_def *
build_udiv(nir_builder *b, nir_ssa_def *n, uint64_t d)
{
   if (d == 0) {
      return nir_imm_intN_t(b, 0, n->bit_size);
   } else if (util_is_power_of_two_or_zero64(d)) {
      return nir_ushr_imm(b, n, util_logbase2_64(d));
   } else {
      struct util_fast_udiv_info m =
         util_compute_fast_udiv_info(d, n->bit_size, n->bit_size);

      if (m.pre_shift)
         n = nir_ushr_imm(b, n, m.pre_shift);
      if (m.increment)
         n = nir_uadd_sat(b, n, nir_imm_intN_t(b, m.increment, n->bit_size));
      n = nir_umul_high(b, n, nir_imm_intN_t(b, m.multiplier, n->bit_size));
      if (m.post_shift)
         n = nir_ushr_imm(b, n, m.post_shift);

      return n;
   }
}

static nir_ssa_def *
build_umod(nir_builder *b, nir_ssa_def *n, uint64_t d)
{
   if (d == 0) {
      return nir_imm_intN_t(b, 0, n->bit_size);
   } else if (util_is_power_of_two_or_zero64(d)) {
      return nir_iand(b, n, nir_imm_intN_t(b, d - 1, n->bit_size));
   } else {
      return nir_isub(b, n, nir_imul(b, build_udiv(b, n, d),
                                        nir_imm_intN_t(b, d, n->bit_size)));
   }
}

static nir_ssa_def *
build_idiv(nir_builder *b, nir_ssa_def *n, int64_t d)
{
   int64_t int_min = u_intN_min(n->bit_size);
   if (d == int_min)
      return nir_b2iN(b, nir_ieq_imm(b, n, int_min), n->bit_size);

   uint64_t abs_d = d < 0 ? -d : d;

   if (d == 0) {
      return nir_imm_intN_t(b, 0, n->bit_size);
   } else if (d == 1) {
      return n;
   } else if (d == -1) {
      return nir_ineg(b, n);
   } else if (util_is_power_of_two_or_zero64(abs_d)) {
      nir_ssa_def *uq = nir_ushr_imm(b, nir_iabs(b, n), util_logbase2_64(abs_d));
      nir_ssa_def *n_neg = nir_ilt(b, n, nir_imm_intN_t(b, 0, n->bit_size));
      nir_ssa_def *neg = d < 0 ? nir_inot(b, n_neg) : n_neg;
      return nir_bcsel(b, neg, nir_ineg(b, uq), uq);
   } else {
      struct util_fast_sdiv_info m =
         util_compute_fast_sdiv_info(d, n->bit_size);

      nir_ssa_def *res =
         nir_imul_high(b, n, nir_imm_intN_t(b, m.multiplier, n->bit_size));
      if (d > 0 && m.multiplier < 0)
         res = nir_iadd(b, res, n);
      if (d < 0 && m.multiplier > 0)
         res = nir_isub(b, res, n);
      if (m.shift)
         res = nir_ishr_imm(b, res, m.shift);
      res = nir_iadd(b, res, nir_ushr_imm(b, res, n->bit_size - 1));

      return res;
   }
}

static nir_ssa_def *
build_irem(nir_builder *b, nir_ssa_def *n, int64_t d)
{
   int64_t int_min = u_intN_min(n->bit_size);
   if (d == 0) {
      return nir_imm_intN_t(b, 0, n->bit_size);
   } else if (d == int_min) {
      return nir_bcsel(b, nir_ieq_imm(b, n, int_min), nir_imm_intN_t(b, 0, n->bit_size), n);
   } else {
      d = d < 0 ? -d : d;
      if (util_is_power_of_two_or_zero64(d)) {
         nir_ssa_def *tmp = nir_bcsel(b, nir_ilt(b, n, nir_imm_intN_t(b, 0, n->bit_size)),
                                      nir_iadd_imm(b, n, d - 1), n);
         return nir_isub(b, n, nir_iand_imm(b, tmp, -d));
      } else {
         return nir_isub(b, n, nir_imul(b, build_idiv(b, n, d),
                                        nir_imm_intN_t(b, d, n->bit_size)));
      }
   }
}

static nir_ssa_def *
build_imod(nir_builder *b, nir_ssa_def *n, int64_t d)
{
   int64_t int_min = u_intN_min(n->bit_size);
   if (d == 0) {
      return nir_imm_intN_t(b, 0, n->bit_size);
   } else if (d == int_min) {
      nir_ssa_def *int_min_def = nir_imm_intN_t(b, int_min, n->bit_size);
      nir_ssa_def *is_neg_not_int_min = nir_ult(b, int_min_def, n);
      nir_ssa_def *is_zero = nir_ieq_imm(b, n, 0);
      return nir_bcsel(b, nir_ior(b, is_neg_not_int_min, is_zero), n, nir_iadd(b, int_min_def, n));
   } else if (d > 0 && util_is_power_of_two_or_zero64(d)) {
      return nir_iand(b, n, nir_imm_intN_t(b, d - 1, n->bit_size));
   } else if (d < 0 && util_is_power_of_two_or_zero64(-d)) {
      nir_ssa_def *d_def = nir_imm_intN_t(b, d, n->bit_size);
      nir_ssa_def *res = nir_ior(b, n, d_def);
      return nir_bcsel(b, nir_ieq(b, res, d_def), nir_imm_intN_t(b, 0, n->bit_size), res);
   } else {
      nir_ssa_def *rem = build_irem(b, n, d);
      nir_ssa_def *zero = nir_imm_intN_t(b, 0, n->bit_size);
      nir_ssa_def *sign_same = d < 0 ? nir_ilt(b, n, zero) : nir_ige(b, n, zero);
      nir_ssa_def *rem_zero = nir_ieq(b, rem, zero);
      return nir_bcsel(b, nir_ior(b, rem_zero, sign_same), rem, nir_iadd_imm(b, rem, d));
   }
}

static bool
nir_opt_idiv_const_instr(nir_builder *b, nir_instr *instr, void *user_data)
{
   unsigned *min_bit_size = user_data;

   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_udiv &&
       alu->op != nir_op_idiv &&
       alu->op != nir_op_umod &&
       alu->op != nir_op_imod &&
       alu->op != nir_op_irem)
      return false;

   assert(alu->dest.dest.is_ssa);
   assert(alu->src[0].src.is_ssa && alu->src[1].src.is_ssa);

   if (alu->dest.dest.ssa.bit_size < *min_bit_size)
      return false;

   if (!nir_src_is_const(alu->src[1].src))
      return false;

   unsigned bit_size = alu->src[1].src.ssa->bit_size;

   b->cursor = nir_before_instr(&alu->instr);

   nir_ssa_def *q[NIR_MAX_VEC_COMPONENTS];
   for (unsigned comp = 0; comp < alu->dest.dest.ssa.num_components; comp++) {
      /* Get the numerator for the channel */
      nir_ssa_def *n = nir_channel(b, alu->src[0].src.ssa,
                                   alu->src[0].swizzle[comp]);

      /* Get the denominator for the channel */
      int64_t d = nir_src_comp_as_int(alu->src[1].src,
                                      alu->src[1].swizzle[comp]);

      nir_alu_type d_type = nir_op_infos[alu->op].input_types[1];
      if (nir_alu_type_get_base_type(d_type) == nir_type_uint) {
         /* The code above sign-extended.  If we're lowering an unsigned op,
          * we need to mask it off to the correct number of bits so that a
          * cast to uint64_t will do the right thing.
          */
         if (bit_size < 64)
            d &= (1ull << bit_size) - 1;
      }

      switch (alu->op) {
      case nir_op_udiv:
         q[comp] = build_udiv(b, n, d);
         break;
      case nir_op_idiv:
         q[comp] = build_idiv(b, n, d);
         break;
      case nir_op_umod:
         q[comp] = build_umod(b, n, d);
         break;
      case nir_op_imod:
         q[comp] = build_imod(b, n, d);
         break;
      case nir_op_irem:
         q[comp] = build_irem(b, n, d);
         break;
      default:
         unreachable("Unknown integer division op");
      }
   }

   nir_ssa_def *qvec = nir_vec(b, q, alu->dest.dest.ssa.num_components);
   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, qvec);
   nir_instr_remove(&alu->instr);

   return true;
}

bool
nir_opt_idiv_const(nir_shader *shader, unsigned min_bit_size)
{
   return nir_shader_instructions_pass(shader, nir_opt_idiv_const_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       &min_bit_size);
}
