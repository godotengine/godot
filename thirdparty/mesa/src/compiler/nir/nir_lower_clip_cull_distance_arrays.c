/*
 * Copyright © 2015 Intel Corporation
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

/**
 * @file
 *
 * This pass combines clip and cull distance arrays in separate locations and
 * colocates them both in VARYING_SLOT_CLIP_DIST0.  It does so by maintaining
 * two arrays but making them compact and using location_frac to stack them on
 * top of each other.
 */

/**
 * Get the length of the clip/cull distance array, looking past
 * any interface block arrays.
 */
static unsigned
get_unwrapped_array_length(nir_shader *nir, nir_variable *var)
{
   if (!var)
      return 0;

   /* Unwrap GS input and TCS input/output interfaces.  We want the
    * underlying clip/cull distance array length, not the per-vertex
    * array length.
    */
   const struct glsl_type *type = var->type;
   if (nir_is_arrayed_io(var, nir->info.stage))
      type = glsl_get_array_element(type);

   if (var->data.per_view) {
      assert(glsl_type_is_array(type));
      type = glsl_get_array_element(type);
   }

   assert(glsl_type_is_array(type));

   return glsl_get_length(type);
}

static bool
combine_clip_cull(nir_shader *nir,
                  nir_variable_mode mode,
                  bool store_info)
{
   nir_variable *cull = NULL;
   nir_variable *clip = NULL;

   nir_foreach_variable_with_modes(var, nir, mode) {
      if (var->data.location == VARYING_SLOT_CLIP_DIST0)
         clip = var;

      if (var->data.location == VARYING_SLOT_CULL_DIST0)
         cull = var;
   }

   if (!cull && !clip) {
      /* If this is run after optimizations and the variables have been
       * eliminated, we should update the shader info, because no other
       * place does that.
       */
      if (store_info) {
         nir->info.clip_distance_array_size = 0;
         nir->info.cull_distance_array_size = 0;
      }
      return false;
   }

   if (!cull && clip) {
      /* The GLSL IR lowering pass must have converted these to vectors */
      if (!clip->data.compact)
         return false;

      /* If this pass has already run, don't repeat.  We would think that
       * the combined clip/cull distance array was clip-only and mess up.
       */
      if (clip->data.how_declared == nir_var_hidden)
         return false;
   }

   const unsigned clip_array_size = get_unwrapped_array_length(nir, clip);
   const unsigned cull_array_size = get_unwrapped_array_length(nir, cull);

   if (store_info) {
      nir->info.clip_distance_array_size = clip_array_size;
      nir->info.cull_distance_array_size = cull_array_size;
   }

   if (clip) {
      assert(clip->data.compact);
      clip->data.how_declared = nir_var_hidden;
   }

   if (cull) {
      assert(cull->data.compact);
      cull->data.how_declared = nir_var_hidden;
      cull->data.location = VARYING_SLOT_CLIP_DIST0 + clip_array_size / 4;
      cull->data.location_frac = clip_array_size % 4;
   }

   return true;
}

bool
nir_lower_clip_cull_distance_arrays(nir_shader *nir)
{
   bool progress = false;

   if (nir->info.stage <= MESA_SHADER_GEOMETRY ||
       nir->info.stage == MESA_SHADER_MESH)
      progress |= combine_clip_cull(nir, nir_var_shader_out, true);

   if (nir->info.stage > MESA_SHADER_VERTEX &&
       nir->info.stage <= MESA_SHADER_FRAGMENT) {
      progress |= combine_clip_cull(nir, nir_var_shader_in,
                                    nir->info.stage == MESA_SHADER_FRAGMENT);
   }

   nir_foreach_function(function, nir) {
      if (!function->impl)
         continue;

      if (progress) {
         nir_metadata_preserve(function->impl,
                               nir_metadata_block_index |
                               nir_metadata_dominance |
                               nir_metadata_live_ssa_defs |
                               nir_metadata_loop_analysis);
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}
