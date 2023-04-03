/*
 * Copyright © 2010 Intel Corporation
 * Copyright © 2018 Broadcom
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"

/** nir_lower_alu.c
 *
 * NIR's home for miscellaneous ALU operation lowering implementations.
 *
 * Most NIR ALU lowering occurs in nir_opt_algebraic.py, since it's generally
 * easy to write them there.  However, if terms appear multiple times in the
 * lowered code, it can get very verbose and cause a lot of work for CSE, so
 * it may end up being easier to write out in C code.
 *
 * The shader must be in SSA for this pass.
 */

#define LOWER_MUL_HIGH (1 << 0)

static bool
lower_alu_instr(nir_builder *b, nir_instr *instr_, UNUSED void *cb_data)
{
   if (instr_->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *instr = nir_instr_as_alu(instr_);

   nir_ssa_def *lowered = NULL;

   assert(instr->dest.dest.is_ssa);

   b->cursor = nir_before_instr(&instr->instr);
   b->exact = instr->exact;

   switch (instr->op) {
   case nir_op_bitfield_reverse:
      if (b->shader->options->lower_bitfield_reverse) {
         /* For more details, see:
          *
          * http://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
          */
         nir_ssa_def *c1 = nir_imm_int(b, 1);
         nir_ssa_def *c2 = nir_imm_int(b, 2);
         nir_ssa_def *c4 = nir_imm_int(b, 4);
         nir_ssa_def *c8 = nir_imm_int(b, 8);
         nir_ssa_def *c16 = nir_imm_int(b, 16);
         nir_ssa_def *c33333333 = nir_imm_int(b, 0x33333333);
         nir_ssa_def *c55555555 = nir_imm_int(b, 0x55555555);
         nir_ssa_def *c0f0f0f0f = nir_imm_int(b, 0x0f0f0f0f);
         nir_ssa_def *c00ff00ff = nir_imm_int(b, 0x00ff00ff);

         lowered = nir_ssa_for_alu_src(b, instr, 0);

         /* Swap odd and even bits. */
         lowered = nir_ior(b,
                           nir_iand(b, nir_ushr(b, lowered, c1), c55555555),
                           nir_ishl(b, nir_iand(b, lowered, c55555555), c1));

         /* Swap consecutive pairs. */
         lowered = nir_ior(b,
                           nir_iand(b, nir_ushr(b, lowered, c2), c33333333),
                           nir_ishl(b, nir_iand(b, lowered, c33333333), c2));

         /* Swap nibbles. */
         lowered = nir_ior(b,
                           nir_iand(b, nir_ushr(b, lowered, c4), c0f0f0f0f),
                           nir_ishl(b, nir_iand(b, lowered, c0f0f0f0f), c4));

         /* Swap bytes. */
         lowered = nir_ior(b,
                           nir_iand(b, nir_ushr(b, lowered, c8), c00ff00ff),
                           nir_ishl(b, nir_iand(b, lowered, c00ff00ff), c8));

         lowered = nir_ior(b,
                           nir_ushr(b, lowered, c16),
                           nir_ishl(b, lowered, c16));
      }
      break;

   case nir_op_bit_count:
      if (b->shader->options->lower_bit_count) {
         /* For more details, see:
          *
          * http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
          */
         nir_ssa_def *c1 = nir_imm_int(b, 1);
         nir_ssa_def *c2 = nir_imm_int(b, 2);
         nir_ssa_def *c4 = nir_imm_int(b, 4);
         nir_ssa_def *c24 = nir_imm_int(b, 24);
         nir_ssa_def *c33333333 = nir_imm_int(b, 0x33333333);
         nir_ssa_def *c55555555 = nir_imm_int(b, 0x55555555);
         nir_ssa_def *c0f0f0f0f = nir_imm_int(b, 0x0f0f0f0f);
         nir_ssa_def *c01010101 = nir_imm_int(b, 0x01010101);

         lowered = nir_ssa_for_alu_src(b, instr, 0);

         lowered = nir_isub(b, lowered,
                            nir_iand(b, nir_ushr(b, lowered, c1), c55555555));

         lowered = nir_iadd(b,
                            nir_iand(b, lowered, c33333333),
                            nir_iand(b, nir_ushr(b, lowered, c2), c33333333));

         lowered = nir_ushr(b,
                            nir_imul(b,
                                     nir_iand(b,
                                              nir_iadd(b,
                                                       lowered,
                                                       nir_ushr(b, lowered, c4)),
                                              c0f0f0f0f),
                                     c01010101),
                            c24);
      }
      break;

   case nir_op_imul_high:
   case nir_op_umul_high:
      if (b->shader->options->lower_mul_high) {
         nir_ssa_def *src0 = nir_ssa_for_alu_src(b, instr, 0);
         nir_ssa_def *src1 = nir_ssa_for_alu_src(b, instr, 1);
         if (src0->bit_size < 32) {
            /* Just do the math in 32-bit space and shift the result */
            nir_alu_type base_type = nir_op_infos[instr->op].output_type;

            nir_ssa_def *src0_32 = nir_type_convert(b, src0, base_type, base_type | 32, nir_rounding_mode_undef);
            nir_ssa_def *src1_32 = nir_type_convert(b, src1, base_type, base_type | 32, nir_rounding_mode_undef);
            nir_ssa_def *dest_32 = nir_imul(b, src0_32, src1_32);
            nir_ssa_def *dest_shifted = nir_ishr(b, dest_32, nir_imm_int(b, src0->bit_size));
            lowered = nir_type_convert(b, dest_shifted, base_type, base_type | src0->bit_size, nir_rounding_mode_undef);
         } else {
            nir_ssa_def *cshift = nir_imm_int(b, src0->bit_size / 2);
            nir_ssa_def *cmask = nir_imm_intN_t(b, (1ull << (src0->bit_size / 2)) - 1, src0->bit_size);
            nir_ssa_def *different_signs = NULL;
            if (instr->op == nir_op_imul_high) {
               nir_ssa_def *c0 = nir_imm_intN_t(b, 0, src0->bit_size);
               different_signs = nir_ixor(b,
                                          nir_ilt(b, src0, c0),
                                          nir_ilt(b, src1, c0));
               src0 = nir_iabs(b, src0);
               src1 = nir_iabs(b, src1);
            }

            /*   ABCD
             * * EFGH
             * ======
             * (GH * CD) + (GH * AB) << 16 + (EF * CD) << 16 + (EF * AB) << 32
             *
             * Start by splitting into the 4 multiplies.
             */
            nir_ssa_def *src0l = nir_iand(b, src0, cmask);
            nir_ssa_def *src1l = nir_iand(b, src1, cmask);
            nir_ssa_def *src0h = nir_ushr(b, src0, cshift);
            nir_ssa_def *src1h = nir_ushr(b, src1, cshift);

            nir_ssa_def *lo = nir_imul(b, src0l, src1l);
            nir_ssa_def *m1 = nir_imul(b, src0l, src1h);
            nir_ssa_def *m2 = nir_imul(b, src0h, src1l);
            nir_ssa_def *hi = nir_imul(b, src0h, src1h);

            nir_ssa_def *tmp;

            tmp = nir_ishl(b, m1, cshift);
            hi = nir_iadd(b, hi, nir_uadd_carry(b, lo, tmp));
            lo = nir_iadd(b, lo, tmp);
            hi = nir_iadd(b, hi, nir_ushr(b, m1, cshift));

            tmp = nir_ishl(b, m2, cshift);
            hi = nir_iadd(b, hi, nir_uadd_carry(b, lo, tmp));
            lo = nir_iadd(b, lo, tmp);
            hi = nir_iadd(b, hi, nir_ushr(b, m2, cshift));

            if (instr->op == nir_op_imul_high) {
               /* For channels where different_signs is set we have to perform a
                * 64-bit negation.  This is *not* the same as just negating the
                * high 32-bits.  Consider -3 * 2.  The high 32-bits is 0, but the
                * desired result is -1, not -0!  Recall -x == ~x + 1.
                */
               nir_ssa_def *c1 = nir_imm_intN_t(b, 1, src0->bit_size);
               hi = nir_bcsel(b, different_signs,
                              nir_iadd(b,
                                       nir_inot(b, hi),
                                       nir_uadd_carry(b, nir_inot(b, lo), c1)),
                              hi);
            }

            lowered = hi;
         }
      }
      break;

   default:
      break;
   }

   if (lowered) {
      nir_ssa_def_rewrite_uses(&instr->dest.dest.ssa, lowered);
      nir_instr_remove(&instr->instr);
      return true;
   } else {
      return false;
   }
}

bool
nir_lower_alu(nir_shader *shader)
{
   if (!shader->options->lower_bitfield_reverse &&
       !shader->options->lower_mul_high)
      return false;

   return nir_shader_instructions_pass(shader, lower_alu_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
