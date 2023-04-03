/*
 * Copyright © 2015 Broadcom
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

#include "util/macros.h"
#include "nir.h"
#include "nir_builder.h"

/** @file nir_lower_load_const_to_scalar.c
 *
 * Replaces vector nir_load_const instructions with a series of loads and a
 * vec[234] to reconstruct the original vector (on the assumption that
 * nir_lower_alu_to_scalar() will then be used to split it up).
 *
 * This gives NIR a chance to CSE more operations on a scalar shader, when the
 * same value was used in different vector contant loads.
 */

static bool
lower_load_const_instr_scalar(nir_load_const_instr *lower)
{
   if (lower->def.num_components == 1)
      return false;

   nir_builder b;
   nir_builder_init(&b, nir_cf_node_get_function(&lower->instr.block->cf_node));
   b.cursor = nir_before_instr(&lower->instr);

   /* Emit the individual loads. */
   nir_ssa_def *loads[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < lower->def.num_components; i++) {
      nir_load_const_instr *load_comp =
         nir_load_const_instr_create(b.shader, 1, lower->def.bit_size);
      load_comp->value[0] = lower->value[i];
      nir_builder_instr_insert(&b, &load_comp->instr);
      loads[i] = &load_comp->def;
   }

   /* Batch things back together into a vector. */
   nir_ssa_def *vec = nir_vec(&b, loads, lower->def.num_components);

   /* Replace the old load with a reference to our reconstructed vector. */
   nir_ssa_def_rewrite_uses(&lower->def, vec);
   nir_instr_remove(&lower->instr);
   return true;
}

static bool
nir_lower_load_const_to_scalar_impl(nir_function_impl *impl)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_load_const)
            progress |=
               lower_load_const_instr_scalar(nir_instr_as_load_const(instr));
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
nir_lower_load_const_to_scalar(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_lower_load_const_to_scalar_impl(function->impl);
   }

   return progress;
}
