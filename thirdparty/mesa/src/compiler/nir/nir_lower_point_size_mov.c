/*
 * Copyright © 2019 Collabora Ltd
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

/** nir_lower_point_size_mov.c
 *
 * This pass lowers glPointSize into gl_PointSize, by adding a uniform
 * and a move from that uniform to VARYING_SLOT_PSIZ. This is useful for
 * OpenGL ES level hardware that lack constant point-size hardware state.
 */

static bool
lower_impl(nir_function_impl *impl,
           const gl_state_index16 *pointsize_state_tokens,
           nir_variable *out)
{
   nir_shader *shader = impl->function->shader;
   nir_builder b;
   nir_variable *in, *new_out = NULL;

   nir_builder_init(&b, impl);

   in = nir_variable_create(shader, nir_var_uniform,
                            glsl_vec4_type(), "gl_PointSizeClampedMESA");
   in->num_state_slots = 1;
   in->state_slots = ralloc_array(in, nir_state_slot, 1);
   in->state_slots[0].swizzle = BITFIELD_MASK(4);
   memcpy(in->state_slots[0].tokens,
         pointsize_state_tokens,
         sizeof(in->state_slots[0].tokens));

   /* the existing output can't be removed in order to avoid breaking xfb.
    * drivers must check var->data.explicit_location to find the original output
    * and only emit that one for xfb
    */
   if (!out || out->data.explicit_location) {
      new_out = nir_variable_create(shader, nir_var_shader_out,
                                    glsl_float_type(), "gl_PointSizeMESA");
      new_out->data.location = VARYING_SLOT_PSIZ;
   }


   if (!out) {
      b.cursor = nir_before_cf_list(&impl->body);
      nir_ssa_def *load = nir_load_var(&b, in);
      load = nir_fclamp(&b, nir_channel(&b, load, 0), nir_channel(&b, load, 1), nir_channel(&b, load, 2));
      nir_store_var(&b, new_out, load, 0x1);
   } else {
      bool found = false;
      nir_foreach_block_safe(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_deref) {
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (var == out) {
                     b.cursor = nir_after_instr(instr);
                     nir_ssa_def *load = nir_load_var(&b, in);
                     load = nir_fclamp(&b, nir_channel(&b, load, 0), nir_channel(&b, load, 1), nir_channel(&b, load, 2));
                     nir_store_var(&b, new_out ? new_out : out, load, 0x1);
                     found = true;
                  }
               }
            }
         }
      }
      if (!found) {
         b.cursor = nir_before_cf_list(&impl->body);
         nir_ssa_def *load = nir_load_var(&b, in);
         load = nir_fclamp(&b, nir_channel(&b, load, 0), nir_channel(&b, load, 1), nir_channel(&b, load, 2));
         nir_store_var(&b, new_out, load, 0x1);
      }
   }

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);
   return true;
}

void
nir_lower_point_size_mov(nir_shader *shader,
                         const gl_state_index16 *pointsize_state_tokens)
{
   assert(shader->info.stage != MESA_SHADER_FRAGMENT &&
          shader->info.stage != MESA_SHADER_COMPUTE);

   nir_variable *out =
      nir_find_variable_with_location(shader, nir_var_shader_out,
                                      VARYING_SLOT_PSIZ);

   lower_impl(nir_shader_get_entrypoint(shader), pointsize_state_tokens,
              out);
}
