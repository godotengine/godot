/*
 * Copyright © 2022 Collabora Ltd.
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
is_ubo_intrinsic(nir_intrinsic_instr *intrin)
{
   return intrin->intrinsic == nir_intrinsic_load_ubo;
}

static bool
is_ssbo_intrinsic(nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
   case nir_intrinsic_ssbo_atomic_fadd:
   case nir_intrinsic_ssbo_atomic_fmin:
   case nir_intrinsic_ssbo_atomic_fmax:
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
      return true;

   default:
      return false;
   }
}

static bool
is_image_intrinsic(nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_atomic_add:
   case nir_intrinsic_image_atomic_imin:
   case nir_intrinsic_image_atomic_umin:
   case nir_intrinsic_image_atomic_imax:
   case nir_intrinsic_image_atomic_umax:
   case nir_intrinsic_image_atomic_and:
   case nir_intrinsic_image_atomic_or:
   case nir_intrinsic_image_atomic_xor:
   case nir_intrinsic_image_atomic_exchange:
   case nir_intrinsic_image_atomic_comp_swap:
   case nir_intrinsic_image_atomic_fadd:
   case nir_intrinsic_image_atomic_fmin:
   case nir_intrinsic_image_atomic_fmax:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_samples:
   case nir_intrinsic_image_fragment_mask_load_amd:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic_add:
   case nir_intrinsic_bindless_image_atomic_imin:
   case nir_intrinsic_bindless_image_atomic_umin:
   case nir_intrinsic_bindless_image_atomic_imax:
   case nir_intrinsic_bindless_image_atomic_umax:
   case nir_intrinsic_bindless_image_atomic_and:
   case nir_intrinsic_bindless_image_atomic_or:
   case nir_intrinsic_bindless_image_atomic_xor:
   case nir_intrinsic_bindless_image_atomic_exchange:
   case nir_intrinsic_bindless_image_atomic_comp_swap:
   case nir_intrinsic_bindless_image_atomic_fadd:
   case nir_intrinsic_bindless_image_atomic_fmin:
   case nir_intrinsic_bindless_image_atomic_fmax:
   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_bindless_image_samples:
   case nir_intrinsic_bindless_image_fragment_mask_load_amd:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic_add:
   case nir_intrinsic_image_deref_atomic_umin:
   case nir_intrinsic_image_deref_atomic_imin:
   case nir_intrinsic_image_deref_atomic_umax:
   case nir_intrinsic_image_deref_atomic_imax:
   case nir_intrinsic_image_deref_atomic_and:
   case nir_intrinsic_image_deref_atomic_or:
   case nir_intrinsic_image_deref_atomic_xor:
   case nir_intrinsic_image_deref_atomic_exchange:
   case nir_intrinsic_image_deref_atomic_comp_swap:
   case nir_intrinsic_image_deref_atomic_fadd:
   case nir_intrinsic_image_deref_atomic_fmin:
   case nir_intrinsic_image_deref_atomic_fmax:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_fragment_mask_load_amd:
      return true;

   default:
      return false;
   }
}

static bool
has_non_uniform_tex_access(nir_tex_instr *tex)
{
   return tex->texture_non_uniform || tex->sampler_non_uniform;
}

static bool
has_non_uniform_access_intrin(nir_intrinsic_instr *intrin)
{
   return (nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM) != 0;
}

static bool
nir_has_non_uniform_access_impl(nir_function_impl *impl, enum nir_lower_non_uniform_access_type types)
{
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_tex: {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if ((types & nir_lower_non_uniform_texture_access) &&
                 has_non_uniform_tex_access(tex))
               return true;
            break;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (is_ubo_intrinsic(intrin)) {
               if ((types & nir_lower_non_uniform_ubo_access) &&
                    has_non_uniform_access_intrin(intrin))
                  return true;
            } else if (is_ssbo_intrinsic(intrin)) {
               if ((types & nir_lower_non_uniform_ssbo_access) &&
                    has_non_uniform_access_intrin(intrin))
                  return true;
            } else if (is_image_intrinsic(intrin)) {
               if ((types & nir_lower_non_uniform_image_access) &&
                    has_non_uniform_access_intrin(intrin))
                  return true;
            } else {
               /* Nothing to do */
            }
            break;
         }

         default:
            /* Nothing to do */
            break;
         }
      }
   }

   return false;
}

bool
nir_has_non_uniform_access(nir_shader *shader, enum nir_lower_non_uniform_access_type types)
{
   nir_foreach_function(function, shader) {
      if (function->impl && nir_has_non_uniform_access_impl(function->impl, types))
         return true;
   }

   return false;
}

static bool
opt_non_uniform_tex_access(nir_tex_instr *tex)
{
   if (!has_non_uniform_tex_access(tex))
      return false;

   bool progress = false;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_offset:
      case nir_tex_src_texture_handle:
      case nir_tex_src_texture_deref:
         if (tex->texture_non_uniform && !tex->src[i].src.ssa->divergent) {
            tex->texture_non_uniform = false;
            progress = true;
         }
         break;

      case nir_tex_src_sampler_offset:
      case nir_tex_src_sampler_handle:
      case nir_tex_src_sampler_deref:
         if (tex->sampler_non_uniform && !tex->src[i].src.ssa->divergent) {
            tex->sampler_non_uniform = false;
            progress = true;
         }
         break;

      default:
         break;
      }
   }

   return progress;
}

static bool
opt_non_uniform_access_intrin(nir_intrinsic_instr *intrin, unsigned handle_src)
{
   if (!has_non_uniform_access_intrin(intrin))
      return false;

   if (intrin->src[handle_src].ssa->divergent)
      return false;

   nir_intrinsic_set_access(intrin, nir_intrinsic_access(intrin) & ~ACCESS_NON_UNIFORM);

   return true;
}

static bool
nir_opt_non_uniform_access_instr(nir_builder *b, nir_instr *instr, UNUSED void *user_data)
{
   switch (instr->type) {
   case nir_instr_type_tex:
      return opt_non_uniform_tex_access(nir_instr_as_tex(instr));

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (is_ubo_intrinsic(intrin) || is_ssbo_intrinsic(intrin) || is_image_intrinsic(intrin)) {
         unsigned handle_src = 0;
         /* SSBO Stores put the index in the second source */
         if (intrin->intrinsic == nir_intrinsic_store_ssbo)
            handle_src = 1;
         return opt_non_uniform_access_intrin(intrin, handle_src);
      }
      break;
   }

   default:
      /* Nothing to do */
      break;
   }

   return false;
}

bool
nir_opt_non_uniform_access(nir_shader *shader)
{
   NIR_PASS(_, shader, nir_convert_to_lcssa, true, true);
   nir_divergence_analysis(shader);

   bool progress = nir_shader_instructions_pass(shader,
                                                nir_opt_non_uniform_access_instr,
                                                nir_metadata_all, NULL);

   NIR_PASS(_, shader, nir_opt_remove_phis); /* cleanup LCSSA phis */

   return progress;
}
