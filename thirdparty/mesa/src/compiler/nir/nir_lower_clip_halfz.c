/*
 * Copyright 2018-2019 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "nir_builder.h"

static bool
lower_pos_write(nir_builder *b, nir_instr *instr, UNUSED void *cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_POS)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *pos = nir_ssa_for_src(b, intr->src[1], 4);
   nir_ssa_def *def = nir_vec4(b,
                               nir_channel(b, pos, 0),
                               nir_channel(b, pos, 1),
                               nir_fmul_imm(b,
                                            nir_fadd(b,
                                                     nir_channel(b, pos, 2),
                                                     nir_channel(b, pos, 3)),
                                            0.5),
                               nir_channel(b, pos, 3));
   nir_instr_rewrite_src(&intr->instr, intr->src + 1, nir_src_for_ssa(def));
   return true;
}

void
nir_lower_clip_halfz(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_VERTEX &&
       shader->info.stage != MESA_SHADER_GEOMETRY &&
       shader->info.stage != MESA_SHADER_TESS_EVAL)
      return;

   nir_shader_instructions_pass(shader, lower_pos_write,
                                nir_metadata_block_index |
                                nir_metadata_dominance,
                                NULL);
}
