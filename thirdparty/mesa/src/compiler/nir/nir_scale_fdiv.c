/*
 * Copyright © 2020 Microsoft Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"

static bool
nir_scale_fdiv_instr(nir_builder *b, nir_instr *instr, UNUSED void *_data)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_fdiv || alu->src[0].src.ssa->bit_size != 32)
      return false;

   b->cursor = nir_before_instr(&alu->instr);

   nir_ssa_def *orig_a = nir_ssa_for_alu_src(b, alu, 0);
   nir_ssa_def *orig_b = nir_ssa_for_alu_src(b, alu, 1);
   nir_ssa_def *fabs = nir_fabs(b, orig_b);
   nir_ssa_def *big = nir_flt(b, nir_imm_int(b, 0x7e800000), fabs);
   nir_ssa_def *small = nir_flt(b, fabs, nir_imm_int(b, 0x00800000));

   nir_ssa_def *scaled_down_a = nir_fmul_imm(b, orig_a, 0.25);
   nir_ssa_def *scaled_down_b = nir_fmul_imm(b, orig_b, 0.25);
   nir_ssa_def *scaled_up_a = nir_fmul_imm(b, orig_a, 16777216.0);
   nir_ssa_def *scaled_up_b = nir_fmul_imm(b, orig_b, 16777216.0);

   nir_ssa_def *final_a =
      nir_bcsel(b, big, scaled_down_a,
     (nir_bcsel(b, small, scaled_up_a, orig_a)));
   nir_ssa_def *final_b =
      nir_bcsel(b, big, scaled_down_b,
     (nir_bcsel(b, small, scaled_up_b, orig_b)));

   nir_ssa_def *new_div = nir_fdiv(b, final_a, final_b);
   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, new_div);

   return true;
}

/** Scale both sides of an fdiv if needed to prevent denorm flushing
 *
 * This may be needed to satisfy the precision requirements of OpenCL.  When
 * fdiv is lowered to frcp+fmul, denorm flushing may cause the frcp to return
 * zero even for finite floats.  This multiplies both sides of an fdiv by a
 * constant, if needed, to prevent such flushing.
 */
bool
nir_scale_fdiv(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, nir_scale_fdiv_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
