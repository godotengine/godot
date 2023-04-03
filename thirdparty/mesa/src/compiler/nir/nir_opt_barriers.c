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

#include "nir.h"

static bool
nir_opt_combine_memory_barriers_impl(
   nir_function_impl *impl, nir_combine_memory_barrier_cb combine_cb, void *data)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_intrinsic_instr *prev = NULL;

      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) {
            prev = NULL;
            continue;
         }

         nir_intrinsic_instr *current = nir_instr_as_intrinsic(instr);
         if (current->intrinsic != nir_intrinsic_scoped_barrier ||
             nir_intrinsic_execution_scope(current) != NIR_SCOPE_NONE) {
            prev = NULL;
            continue;
         }

         if (prev && combine_cb(prev, current, data)) {
            nir_instr_remove(&current->instr);
            progress = true;
         } else {
            prev = current;
         }
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance |
                                  nir_metadata_live_ssa_defs);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

/* Combine adjacent scoped memory barriers. */
bool
nir_opt_combine_memory_barriers(
   nir_shader *shader, nir_combine_memory_barrier_cb combine_cb, void *data)
{
   assert(combine_cb);

   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl &&
          nir_opt_combine_memory_barriers_impl(function->impl, combine_cb, data)) {
         progress = true;
      }
   }

   return progress;
}
