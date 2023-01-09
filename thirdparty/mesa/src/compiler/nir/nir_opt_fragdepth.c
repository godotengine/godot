/*
 * Copyright © 2021 Advanced Micro Devices, Inc.
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
 * This pass removes no-op assignment to gl_FragDepth.
 *
 * gl_FragDepth implicit value is gl_FragCoord.z, so if a shader only assign
 * this value to gl_FragDepth, the store instruction is removed.
 */

static bool
ssa_def_is_source_depth(nir_ssa_def *def)
{
   nir_ssa_scalar scalar = nir_ssa_scalar_resolved(def, 0);
   nir_instr *instr = scalar.def->parent_instr;
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   if (intrin->intrinsic != nir_intrinsic_load_frag_coord)
      return false;

   /* Depth is gl_FragCoord.z */
   return scalar.comp == 2;
}

bool
nir_opt_fragdepth(nir_shader *shader)
{
   bool progress = false;
   nir_intrinsic_instr *store_intrin = NULL;

   if (shader->info.stage != MESA_SHADER_FRAGMENT)
      goto end;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_shader_out))
            continue;

         nir_variable *var = nir_deref_instr_get_variable(deref);
         if (var->data.location != FRAG_RESULT_DEPTH)
            continue;

         /* We found a write to gl_FragDepth */
         if (store_intrin) {
            /* This isn't the only write: give up on this optimization */
            goto end;
         } else {
            if (ssa_def_is_source_depth(intrin->src[1].ssa)) {
               /* We're writing gl_FragCoord.z in gl_FragDepth: remember
                * intrin so we can try to remove it later. */
               store_intrin = intrin;
            } else {
               /* We're writing something else: give up. */
               goto end;
            }
         }
      }
   }

   if (store_intrin) {
      /* Found a single store to gl_FragDepth, and it writes gl_FragCoord.z to it.
       * Remove it since that's the implicit value of gl_FragDepth.
       */
      nir_instr_remove(&store_intrin->instr);

      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance |
                                  nir_metadata_loop_analysis |
                                  nir_metadata_instr_index);
      progress = true;
   }

end:
   if (!progress)
      nir_shader_preserve_all_metadata(shader);
   return progress;
}
