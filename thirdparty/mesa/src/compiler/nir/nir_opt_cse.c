/*
 * Copyright © 2014 Intel Corporation
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
 *
 * Authors:
 *    Jason Ekstrand (jason@jlekstrand.net)
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#include "nir_instr_set.h"

/*
 * Implements common subexpression elimination
 */

static bool
dominates(const nir_instr *old_instr, const nir_instr *new_instr)
{
   return nir_block_dominates(old_instr->block, new_instr->block);
}

static bool
nir_opt_cse_impl(nir_function_impl *impl)
{
   struct set *instr_set = nir_instr_set_create(NULL);

   _mesa_set_resize(instr_set, impl->ssa_alloc);

   nir_metadata_require(impl, nir_metadata_dominance);

   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block)
         progress |= nir_instr_set_add_or_rewrite(instr_set, instr, dominates);
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   nir_instr_set_destroy(instr_set);
   return progress;
}

bool
nir_opt_cse(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_cse_impl(function->impl);
   }

   return progress;
}

