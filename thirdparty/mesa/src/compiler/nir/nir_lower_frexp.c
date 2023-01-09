/*
 * Copyright © 2015 Intel Corporation
 * Copyright © 2019 Valve Corporation
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
 *    Jason Ekstrand (jason@jlekstrand.net)
 *    Samuel Pitoiset (samuel.pitoiset@gmail.com>
 */

#include "nir.h"
#include "nir_builder.h"

static nir_ssa_def *
lower_frexp_sig(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *abs_x = nir_fabs(b, x);
   nir_ssa_def *zero = nir_imm_floatN_t(b, 0, x->bit_size);
   nir_ssa_def *sign_mantissa_mask, *exponent_value;

   switch (x->bit_size) {
   case 16:
      /* Half-precision floating-point values are stored as
       *   1 sign bit;
       *   5 exponent bits;
       *   10 mantissa bits.
       *
       * An exponent shift of 10 will shift the mantissa out, leaving only the
       * exponent and sign bit (which itself may be zero, if the absolute value
       * was taken before the bitcast and shift).
       */
      sign_mantissa_mask = nir_imm_intN_t(b, 0x83ffu, 16);
      /* Exponent of floating-point values in the range [0.5, 1.0). */
      exponent_value = nir_imm_intN_t(b, 0x3800u, 16);
      break;
   case 32:
      /* Single-precision floating-point values are stored as
       *   1 sign bit;
       *   8 exponent bits;
       *   23 mantissa bits.
       *
       * An exponent shift of 23 will shift the mantissa out, leaving only the
       * exponent and sign bit (which itself may be zero, if the absolute value
       * was taken before the bitcast and shift.
       */
      sign_mantissa_mask = nir_imm_int(b, 0x807fffffu);
      /* Exponent of floating-point values in the range [0.5, 1.0). */
      exponent_value = nir_imm_int(b, 0x3f000000u);
      break;
   case 64:
      /* Double-precision floating-point values are stored as
       *   1 sign bit;
       *   11 exponent bits;
       *   52 mantissa bits.
       *
       * An exponent shift of 20 will shift the remaining mantissa bits out,
       * leaving only the exponent and sign bit (which itself may be zero, if
       * the absolute value was taken before the bitcast and shift.
       */
      sign_mantissa_mask = nir_imm_int(b, 0x800fffffu);
      /* Exponent of floating-point values in the range [0.5, 1.0). */
      exponent_value = nir_imm_int(b, 0x3fe00000u);
      break;
   default:
      unreachable("Invalid bitsize");
   }

   if (x->bit_size == 64) {
      /* We only need to deal with the exponent so first we extract the upper
       * 32 bits using nir_unpack_64_2x32_split_y.
       */
      nir_ssa_def *upper_x = nir_unpack_64_2x32_split_y(b, x);

      /* If x is ±0, ±Inf, or NaN, return x unmodified. */
      nir_ssa_def *new_upper =
         nir_bcsel(b,
                   nir_iand(b,
                            nir_flt(b, zero, abs_x),
                            nir_fisfinite(b, x)),
                   nir_ior(b,
                           nir_iand(b, upper_x, sign_mantissa_mask),
                           exponent_value),
                   upper_x);

      nir_ssa_def *lower_x = nir_unpack_64_2x32_split_x(b, x);

      return nir_pack_64_2x32_split(b, lower_x, new_upper);
   } else {
      /* If x is ±0, ±Inf, or NaN, return x unmodified. */
      return nir_bcsel(b,
                       nir_iand(b,
                                nir_flt(b, zero, abs_x),
                                nir_fisfinite(b, x)),
                       nir_ior(b,
                               nir_iand(b, x, sign_mantissa_mask),
                               exponent_value),
                       x);
   }
}

static nir_ssa_def *
lower_frexp_exp(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *abs_x = nir_fabs(b, x);
   nir_ssa_def *zero = nir_imm_floatN_t(b, 0, x->bit_size);
   nir_ssa_def *is_not_zero = nir_fneu(b, abs_x, zero);
   nir_ssa_def *exponent;

   switch (x->bit_size) {
   case 16: {
      nir_ssa_def *exponent_shift = nir_imm_int(b, 10);
      nir_ssa_def *exponent_bias = nir_imm_intN_t(b, -14, 16);

      /* Significand return must be of the same type as the input, but the
       * exponent must be a 32-bit integer.
       */
      exponent = nir_i2i32(b, nir_iadd(b, nir_ushr(b, abs_x, exponent_shift),
                              nir_bcsel(b, is_not_zero, exponent_bias, zero)));
      break;
   }
   case 32: {
      nir_ssa_def *exponent_shift = nir_imm_int(b, 23);
      nir_ssa_def *exponent_bias = nir_imm_int(b, -126);

      exponent = nir_iadd(b, nir_ushr(b, abs_x, exponent_shift),
                             nir_bcsel(b, is_not_zero, exponent_bias, zero));
      break;
   }
   case 64: {
      nir_ssa_def *exponent_shift = nir_imm_int(b, 20);
      nir_ssa_def *exponent_bias = nir_imm_int(b, -1022);

      nir_ssa_def *zero32 = nir_imm_int(b, 0);
      nir_ssa_def *abs_upper_x = nir_unpack_64_2x32_split_y(b, abs_x);

      exponent = nir_iadd(b, nir_ushr(b, abs_upper_x, exponent_shift),
                             nir_bcsel(b, is_not_zero, exponent_bias, zero32));
      break;
   }
   default:
      unreachable("Invalid bitsize");
   }

   return exponent;
}

static bool
lower_frexp_instr(nir_builder *b, nir_instr *instr, UNUSED void *cb_data)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu_instr = nir_instr_as_alu(instr);
   nir_ssa_def *lower;

   b->cursor = nir_before_instr(instr);

   switch (alu_instr->op) {
   case nir_op_frexp_sig:
      lower = lower_frexp_sig(b, nir_ssa_for_alu_src(b, alu_instr, 0));
      break;
   case nir_op_frexp_exp:
      lower = lower_frexp_exp(b, nir_ssa_for_alu_src(b, alu_instr, 0));
      break;
   default:
      return false;
   }

   nir_ssa_def_rewrite_uses(&alu_instr->dest.dest.ssa, lower);
   nir_instr_remove(instr);
   return true;
}

bool
nir_lower_frexp(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_frexp_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
