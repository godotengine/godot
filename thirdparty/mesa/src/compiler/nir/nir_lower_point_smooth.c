/*
 * Copyright © 2022 Advanced Micro Devices, Inc.
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
#include "nir_builtin_builder.h"

/**
 * This NIR lowers pass for point smoothing by modifying the alpha value of
 * fragment outputs using the distance from the center of the point.
 * Anti-aliased points get rounded with respect to their radius.
 */

static bool
lower_point_smooth(nir_builder *b, nir_instr *instr, UNUSED void *_state)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   int location = nir_intrinsic_io_semantics(intr).location;
   if ((location != FRAG_RESULT_COLOR && location < FRAG_RESULT_DATA0) ||
       nir_intrinsic_src_type(intr) != nir_type_float32)
      return false;

   assert(intr->src[0].is_ssa);
   assert(intr->num_components == 4);

   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *coord = nir_build_load_point_coord_maybe_flipped(b);

   /* point_size = 1.0 / dFdx(gl_PointCoord.x); */
   nir_ssa_def *point_size = nir_frcp(b, nir_fddx(b, nir_channel(b, coord, 0)));

   /* radius = point_size * 0.5 */
   nir_ssa_def *radius = nir_fmul_imm(b, point_size, 0.5);;

   /**
    * Compute the distance of point from centre
    * distance = √ (x - 0.5)^2 + (y - 0.5)^2
    */
   nir_ssa_def *distance = nir_fast_distance(b, coord,
                                             nir_imm_vec2(b, 0.5, 0.5));
   distance = nir_fmul(b, distance, point_size);

   /* alpha = min(max(radius - distance, 0.0), 1.0) */
   nir_ssa_def *coverage = nir_fsat(b, nir_fsub(b, radius, distance));

   /* Discard fragments that are not covered by the point */
   nir_discard_if(b, nir_feq(b, nir_imm_float(b, 0.0f), coverage));

   /* Write out the fragment color*vec4(1, 1, 1, coverage)*/
   nir_ssa_def *one = nir_imm_float(b, 1.0f);
   nir_ssa_def *new_val = nir_fmul(b, nir_vec4(b, one, one, one, coverage),
                                   intr->src[0].ssa);
   nir_instr_rewrite_src(instr, &intr->src[0], nir_src_for_ssa(new_val));

   return true;
}

bool
nir_lower_point_smooth(nir_shader *shader)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   return nir_shader_instructions_pass(shader, lower_point_smooth,
                                       nir_metadata_loop_analysis |
                                       nir_metadata_block_index |
                                       nir_metadata_dominance, NULL);
}
