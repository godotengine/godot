/*
 * Copyright © 2020 Google LLC
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

/**
 * @file
 *
 * Removes unused trailing components from store data.
 *
 */

#include "nir.h"
#include "nir_builder.h"

static bool
opt_shrink_vectors_image_store(nir_builder *b, nir_intrinsic_instr *instr)
{
   enum pipe_format format;
   if (instr->intrinsic == nir_intrinsic_image_deref_store) {
      nir_deref_instr *deref = nir_src_as_deref(instr->src[0]);
      format = nir_deref_instr_get_variable(deref)->data.image.format;
   } else {
      format = nir_intrinsic_format(instr);
   }
   if (format == PIPE_FORMAT_NONE)
      return false;

   unsigned components = util_format_get_nr_components(format);
   if (components >= instr->num_components)
      return false;

   nir_ssa_def *data = nir_trim_vector(b, instr->src[3].ssa, components);
   nir_instr_rewrite_src(&instr->instr, &instr->src[3], nir_src_for_ssa(data));
   instr->num_components = components;

   return true;
}

static bool
opt_shrink_store_instr(nir_builder *b, nir_intrinsic_instr *instr, bool shrink_image_store)
{
   b->cursor = nir_before_instr(&instr->instr);

   switch (instr->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_scratch:
      break;
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_store:
      return shrink_image_store && opt_shrink_vectors_image_store(b, instr);
   default:
      return false;
   }

   /* Must be a vectorized intrinsic that we can resize. */
   assert(instr->num_components != 0);

   /* Trim the num_components stored according to the write mask. */
   unsigned write_mask = nir_intrinsic_write_mask(instr);
   unsigned last_bit = util_last_bit(write_mask);
   if (last_bit < instr->num_components && instr->src[0].is_ssa) {
      nir_ssa_def *def = nir_trim_vector(b, instr->src[0].ssa, last_bit);
      nir_instr_rewrite_src(&instr->instr,
                            &instr->src[0],
                            nir_src_for_ssa(def));
      instr->num_components = last_bit;

      return true;
   }

   return false;
}

bool
nir_opt_shrink_stores(nir_shader *shader, bool shrink_image_store)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_builder b;
      nir_builder_init(&b, function->impl);

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            progress |= opt_shrink_store_instr(&b, intrin, shrink_image_store);
         }
      }

      if (progress) {
         nir_metadata_preserve(function->impl,
                               nir_metadata_block_index |
                               nir_metadata_dominance);
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}
