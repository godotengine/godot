/*
 * Copyright © 2020 Intel Corporation
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

#include "nir_builder.h"

static bool
lower_vec3_to_vec4_impl(nir_function_impl *impl, nir_variable_mode modes)
{
   bool progress = false;

   if (modes & nir_var_function_temp) {
      nir_foreach_function_temp_variable(var, impl) {
         const struct glsl_type *vec4_type =
            glsl_type_replace_vec3_with_vec4(var->type);
         if (var->type != vec4_type) {
            var->type = vec4_type;
            progress = true;
         }
      }
   }

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (!nir_deref_mode_is_in_set(deref, modes))
               continue;

            const struct glsl_type *vec4_type =
               glsl_type_replace_vec3_with_vec4(deref->type);
            if (deref->type != vec4_type) {
               deref->type = vec4_type;
               progress = true;
            }
            break;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_load_deref: {
               if (intrin->num_components != 3)
                  break;

               nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
               if (!nir_deref_mode_is_in_set(deref, modes))
                  break;

               assert(intrin->dest.is_ssa);
               intrin->num_components = 4;
               intrin->dest.ssa.num_components = 4;

               b.cursor = nir_after_instr(&intrin->instr);
               nir_ssa_def *vec3 = nir_channels(&b, &intrin->dest.ssa, 0x7);
               nir_ssa_def_rewrite_uses_after(&intrin->dest.ssa,
                                              vec3,
                                              vec3->parent_instr);
               progress = true;
               break;
            }

            case nir_intrinsic_store_deref: {
               if (intrin->num_components != 3)
                  break;

               nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
               if (!nir_deref_mode_is_in_set(deref, modes))
                  break;

               assert(intrin->src[1].is_ssa);
               nir_ssa_def *data = intrin->src[1].ssa;

               b.cursor = nir_before_instr(&intrin->instr);
               unsigned swiz[] = { 0, 1, 2, 2 };
               data = nir_swizzle(&b, data, swiz, 4);

               intrin->num_components = 4;
               nir_instr_rewrite_src(&intrin->instr, &intrin->src[1],
                                     nir_src_for_ssa(data));
               progress = true;
               break;
            }

            case nir_intrinsic_copy_deref: {
               nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);
               nir_deref_instr *src = nir_src_as_deref(intrin->src[0]);
               /* If we convert once side of a copy and not the other, that
                * would be very bad.
                */
               if (nir_deref_mode_may_be(dst, modes) ||
                   nir_deref_mode_may_be(src, modes)) {
                  assert(nir_deref_mode_must_be(dst, modes));
                  assert(nir_deref_mode_must_be(src, modes));
               }
               break;
            }

            default:
               break;
            }
            break;
         }

         default:
            break;
         }
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_lower_vec3_to_vec4(nir_shader *shader, nir_variable_mode modes)
{
   bool progress = false;

   if (modes & ~nir_var_function_temp) {
      nir_foreach_variable_in_shader(var, shader) {
         if (!(var->data.mode & modes))
            continue;

         const struct glsl_type *vec4_type =
            glsl_type_replace_vec3_with_vec4(var->type);
         if (var->type != vec4_type) {
            var->type = vec4_type;
            progress = true;
         }
      }
   }

   nir_foreach_function(function, shader) {
      if (function->impl && lower_vec3_to_vec4_impl(function->impl, modes))
         progress = true;
   }

   return progress;
}
