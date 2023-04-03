/*
 * Copyright © 2021 Intel Corporation
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

static bool
lower_single_sampled_instr(nir_builder *b,
                           nir_instr *instr,
                           UNUSED void *cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   nir_ssa_def *lowered;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_sample_id:
      b->cursor = nir_before_instr(instr);
      lowered = nir_imm_int(b, 0);
      break;

   case nir_intrinsic_load_sample_pos:
      b->cursor = nir_before_instr(instr);
      lowered = nir_imm_vec2(b, 0.5, 0.5);
      break;

   case nir_intrinsic_load_sample_mask_in:
      /* Don't lower to helper invocations if helper invocations are going
       * to be lowered right back to sample mask.
       */
      if (b->shader->options->lower_helper_invocation)
         return false;

      b->cursor = nir_before_instr(instr);
      lowered = nir_b2i32(b, nir_inot(b, nir_load_helper_invocation(b, 1)));
      break;

   case nir_intrinsic_interp_deref_at_centroid:
   case nir_intrinsic_interp_deref_at_sample:
      b->cursor = nir_before_instr(instr);
      assert(intrin->src[0].is_ssa);
      lowered = nir_load_deref(b, nir_src_as_deref(intrin->src[0]));
      break;

   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_at_sample:
      b->cursor = nir_before_instr(instr);
      lowered = nir_load_barycentric(b, nir_intrinsic_load_barycentric_pixel,
                                        nir_intrinsic_interp_mode(intrin));

      if (nir_intrinsic_interp_mode(intrin) == INTERP_MODE_NOPERSPECTIVE) {
         BITSET_SET(b->shader->info.system_values_read,
                    SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL);
      } else {
         BITSET_SET(b->shader->info.system_values_read,
                    SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL);
      }
      break;

   default:
      return false;
   }

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, lowered);
   nir_instr_remove(instr);
   return true;
}

/* Assume the fragment shader is single-sampled and lower accordingly
 *
 * This drops sample/centroid qualifiers from all input variables, forces
 * barycentrics to pixel, and constant-folds various built-ins.
 */
bool
nir_lower_single_sampled(nir_shader *shader)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);

   bool progress = false;
   nir_foreach_shader_in_variable(var, shader) {
      if (var->data.sample) {
         var->data.sample = false;
         progress = true;
      }
      if (var->data.centroid) {
         var->data.centroid = false;
         progress = true;
      }
   }

   /* We're going to get rid of any uses of these */
   BITSET_CLEAR(shader->info.system_values_read,
                SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE);
   BITSET_CLEAR(shader->info.system_values_read,
                SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID);
   BITSET_CLEAR(shader->info.system_values_read,
                SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE);
   BITSET_CLEAR(shader->info.system_values_read,
                SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID);

   return nir_shader_instructions_pass(shader, lower_single_sampled_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL) || progress;
}
