/*
 * Copyright © 2019 Google, Inc.
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

/*
 * This lower pass lowers load_interpolated_input for various interpolation
 * modes (as configured via nir_lower_interpolation_options bitmask) into
 * load_attribute_deltas plus alu instructions:
 *
 *    vec3 ad = load_attribute_deltas(varying_slot)
 *    float result = ad.x + ad.y * j + ad.z * i
 *
 */

#include "nir.h"
#include "nir_builder.h"

static bool
nir_lower_interpolation_instr(nir_builder *b, nir_instr *instr, void *cb_data)
{
   nir_lower_interpolation_options options =
         *(nir_lower_interpolation_options *)cb_data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;

   assert(intr->dest.is_ssa);
   assert(intr->src[0].is_ssa);
   assert(intr->src[1].is_ssa);

   nir_intrinsic_instr *bary_intrinsic =
      nir_instr_as_intrinsic(intr->src[0].ssa->parent_instr);

   /* Leave VARYING_SLOT_POS alone */
   if (nir_intrinsic_base(intr) == VARYING_SLOT_POS)
      return false;

   const enum glsl_interp_mode interp_mode =
      nir_intrinsic_interp_mode(bary_intrinsic);

   /* We need actual interpolation modes by the time we get here */
   assert(interp_mode != INTERP_MODE_NONE);

   /* Only lower for inputs that need interpolation */
   if (interp_mode != INTERP_MODE_SMOOTH &&
       interp_mode != INTERP_MODE_NOPERSPECTIVE)
      return false;

   nir_intrinsic_op op = bary_intrinsic->intrinsic;

   switch (op) {
   case nir_intrinsic_load_barycentric_at_sample:
      if (options & nir_lower_interpolation_at_sample)
         break;
      return false;
   case nir_intrinsic_load_barycentric_at_offset:
      if (options & nir_lower_interpolation_at_offset)
         break;
      return false;
   case nir_intrinsic_load_barycentric_centroid:
      if (options & nir_lower_interpolation_centroid)
         break;
      return false;
   case nir_intrinsic_load_barycentric_pixel:
      if (options & nir_lower_interpolation_pixel)
         break;
      return false;
   case nir_intrinsic_load_barycentric_sample:
      if (options & nir_lower_interpolation_sample)
         break;
      return false;
   default:
      return false;
   }

   b->cursor = nir_before_instr(instr);

   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (int i = 0; i < intr->num_components; i++) {
      nir_ssa_def *iid =
         nir_load_fs_input_interp_deltas(b, 32, intr->src[1].ssa,
                                         .base = nir_intrinsic_base(intr),
                                         .component = (nir_intrinsic_component(intr) + i),
                                         .io_semantics = nir_intrinsic_io_semantics(intr));

      nir_ssa_def *bary = intr->src[0].ssa;
      nir_ssa_def *val;

      val = nir_ffma(b, nir_channel(b, bary, 1),
                        nir_channel(b, iid, 1),
                        nir_channel(b, iid, 0));
      val = nir_ffma(b, nir_channel(b, bary, 0),
                        nir_channel(b, iid, 2),
                        val);

      comps[i] = val;
   }
   nir_ssa_def *vec = nir_vec(b, comps, intr->num_components);
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, vec);

   return true;
}

bool
nir_lower_interpolation(nir_shader *shader, nir_lower_interpolation_options options)
{
   return nir_shader_instructions_pass(shader, nir_lower_interpolation_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       &options);
}
