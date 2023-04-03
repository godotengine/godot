/*
 * Copyright © 2015 Red Hat
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

#include "nir.h"
#include "nir_builder.h"

/* ported from LLVM's AMDGPUTargetLowering::LowerUDIVREM */
static nir_ssa_def *
emit_udiv(nir_builder *bld, nir_ssa_def *numer, nir_ssa_def *denom, bool modulo)
{
   nir_ssa_def *rcp = nir_frcp(bld, nir_u2f32(bld, denom));
   rcp = nir_f2u32(bld, nir_fmul_imm(bld, rcp, 4294966784.0));

   nir_ssa_def *neg_rcp_times_denom =
      nir_imul(bld, rcp, nir_ineg(bld, denom));
   rcp = nir_iadd(bld, rcp, nir_umul_high(bld, rcp, neg_rcp_times_denom));

   /* Get initial estimate for quotient/remainder, then refine the estimate
    * in two iterations after */
   nir_ssa_def *quotient = nir_umul_high(bld, numer, rcp);
   nir_ssa_def *num_s_remainder = nir_imul(bld, quotient, denom);
   nir_ssa_def *remainder = nir_isub(bld, numer, num_s_remainder);

   /* First refinement step */
   nir_ssa_def *remainder_ge_den = nir_uge(bld, remainder, denom);
   if (!modulo) {
      quotient = nir_bcsel(bld, remainder_ge_den,
                           nir_iadd_imm(bld, quotient, 1), quotient);
   }
   remainder = nir_bcsel(bld, remainder_ge_den,
                         nir_isub(bld, remainder, denom), remainder);

   /* Second refinement step */
   remainder_ge_den = nir_uge(bld, remainder, denom);
   if (modulo) {
      return nir_bcsel(bld, remainder_ge_den, nir_isub(bld, remainder, denom),
                       remainder);
   } else {
      return nir_bcsel(bld, remainder_ge_den, nir_iadd_imm(bld, quotient, 1),
                       quotient);
   }
}

/* ported from LLVM's AMDGPUTargetLowering::LowerSDIVREM */
static nir_ssa_def *
emit_idiv(nir_builder *bld, nir_ssa_def *numer, nir_ssa_def *denom, nir_op op)
{
   nir_ssa_def *lh_sign = nir_ilt(bld, numer, nir_imm_int(bld, 0));
   nir_ssa_def *rh_sign = nir_ilt(bld, denom, nir_imm_int(bld, 0));

   nir_ssa_def *lhs = nir_iabs(bld, numer);
   nir_ssa_def *rhs = nir_iabs(bld, denom);

   if (op == nir_op_idiv) {
      nir_ssa_def *d_sign = nir_ixor(bld, lh_sign, rh_sign);
      nir_ssa_def *res = emit_udiv(bld, lhs, rhs, false);
      return nir_bcsel(bld, d_sign, nir_ineg(bld, res), res);
   } else {
      nir_ssa_def *res = emit_udiv(bld, lhs, rhs, true);
      res = nir_bcsel(bld, lh_sign, nir_ineg(bld, res), res);
      if (op == nir_op_imod) {
         nir_ssa_def *cond = nir_ieq_imm(bld, res, 0);
         cond = nir_ior(bld, nir_ieq(bld, lh_sign, rh_sign), cond);
         res = nir_bcsel(bld, cond, res, nir_iadd(bld, res, denom));
      }
      return res;
   }
}

static nir_ssa_def *
convert_instr_small(nir_builder *b, nir_op op,
      nir_ssa_def *numer, nir_ssa_def *denom,
      const nir_lower_idiv_options *options)
{
   unsigned sz = numer->bit_size;
   nir_alu_type int_type = nir_op_infos[op].output_type | sz;
   nir_alu_type float_type = nir_type_float | (options->allow_fp16 ? sz * 2 : 32);

   nir_ssa_def *p = nir_type_convert(b, numer, int_type, float_type, nir_rounding_mode_undef);
   nir_ssa_def *q = nir_type_convert(b, denom, int_type, float_type, nir_rounding_mode_undef);

   /* Take 1/q but offset mantissa by 1 to correct for rounding. This is
    * needed for correct results and has been checked exhaustively for
    * all pairs of 16-bit integers */
   nir_ssa_def *rcp = nir_iadd_imm(b, nir_frcp(b, q), 1);

   /* Divide by multiplying by adjusted reciprocal */
   nir_ssa_def *res = nir_fmul(b, p, rcp);

   /* Convert back to integer space with rounding inferred by type */
   res = nir_type_convert(b, res, float_type, int_type, nir_rounding_mode_undef);

   /* Get remainder given the quotient */
   if (op == nir_op_umod || op == nir_op_imod || op == nir_op_irem)
      res = nir_isub(b, numer, nir_imul(b, denom, res));

   /* Adjust for sign, see constant folding definition */
   if (op == nir_op_imod) {
      nir_ssa_def *zero = nir_imm_zero(b, 1, sz);
      nir_ssa_def *diff_sign =
               nir_ine(b, nir_ige(b, numer, zero), nir_ige(b, denom, zero));

      nir_ssa_def *adjust = nir_iand(b, diff_sign, nir_ine(b, res, zero));
      res = nir_iadd(b, res, nir_bcsel(b, adjust, denom, zero));
   }

   return res;
}

static nir_ssa_def *
lower_idiv(nir_builder *b, nir_instr *instr, void *_data)
{
   const nir_lower_idiv_options *options = _data;
   nir_alu_instr *alu = nir_instr_as_alu(instr);

   nir_ssa_def *numer = nir_ssa_for_alu_src(b, alu, 0);
   nir_ssa_def *denom = nir_ssa_for_alu_src(b, alu, 1);

   b->exact = true;

   if (numer->bit_size < 32)
      return convert_instr_small(b, alu->op, numer, denom, options);
   else if (alu->op == nir_op_udiv || alu->op == nir_op_umod)
      return emit_udiv(b, numer, denom, alu->op == nir_op_umod);
   else
      return emit_idiv(b, numer, denom, alu->op);
}

static bool
inst_is_idiv(const nir_instr *instr, UNUSED const void *_state)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);

   if (alu->dest.dest.ssa.bit_size > 32)
      return false;

   switch (alu->op) {
   case nir_op_idiv:
   case nir_op_udiv:
   case nir_op_imod:
   case nir_op_umod:
   case nir_op_irem:
      return true;
   default:
      return false;
   }
}

bool
nir_lower_idiv(nir_shader *shader, const nir_lower_idiv_options *options)
{
   return nir_shader_lower_instructions(shader,
         inst_is_idiv,
         lower_idiv,
         (void *)options);
}
