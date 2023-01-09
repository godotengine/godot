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
 */

#include "nir.h"

/*
 * Handles management of the metadata.
 */

void
nir_metadata_require(nir_function_impl *impl, nir_metadata required, ...)
{
#define NEEDS_UPDATE(X) ((required & ~impl->valid_metadata) & (X))

   if (NEEDS_UPDATE(nir_metadata_block_index))
      nir_index_blocks(impl);
   if (NEEDS_UPDATE(nir_metadata_instr_index))
      nir_index_instrs(impl);
   if (NEEDS_UPDATE(nir_metadata_dominance))
      nir_calc_dominance_impl(impl);
   if (NEEDS_UPDATE(nir_metadata_live_ssa_defs))
      nir_live_ssa_defs_impl(impl);
   if (NEEDS_UPDATE(nir_metadata_loop_analysis)) {
      va_list ap;
      va_start(ap, required);
      /* !! Warning !! Do not move these va_arg() call directly to
       * nir_loop_analyze_impl() as parameters because the execution order will
       * become undefined.
       */
      nir_variable_mode mode = va_arg(ap, nir_variable_mode);
      int force_unroll_sampler_indirect = va_arg(ap, int);
      nir_loop_analyze_impl(impl, mode, force_unroll_sampler_indirect);
      va_end(ap);
   }

#undef NEEDS_UPDATE

   impl->valid_metadata |= required;
}

void
nir_metadata_preserve(nir_function_impl *impl, nir_metadata preserved)
{
   impl->valid_metadata &= preserved;
}

void
nir_shader_preserve_all_metadata(nir_shader *shader)
{
   nir_foreach_function(function, shader) {
      if (function->impl)
         nir_metadata_preserve(function->impl, nir_metadata_all);
   }
}

#ifndef NDEBUG
/**
 * Make sure passes properly invalidate metadata (part 1).
 *
 * Call this before running a pass to set a bogus metadata flag, which will
 * only be preserved if the pass forgets to call nir_metadata_preserve().
 */
void
nir_metadata_set_validation_flag(nir_shader *shader)
{
   nir_foreach_function(function, shader) {
      if (function->impl) {
         function->impl->valid_metadata |= nir_metadata_not_properly_reset;
      }
   }
}

/**
 * Make sure passes properly invalidate metadata (part 2).
 *
 * Call this after a pass makes progress to verify that the bogus metadata set by
 * the earlier function was properly thrown away.  Note that passes may not call
 * nir_metadata_preserve() if they don't actually make any changes at all.
 */
void
nir_metadata_check_validation_flag(nir_shader *shader)
{
   nir_foreach_function(function, shader) {
      if (function->impl) {
         assert(!(function->impl->valid_metadata &
                  nir_metadata_not_properly_reset));
      }
   }
}
#endif
