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
 *
 */

#include "nir_worklist.h"

void
nir_block_worklist_add_all(nir_block_worklist *w, nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_block_worklist_push_tail(w, block);
   }
}

static bool
nir_instr_worklist_add_srcs_cb(nir_src *src, void *state)
{
   nir_instr_worklist *wl = state;

   if (src->is_ssa)
      nir_instr_worklist_push_tail(wl, src->ssa->parent_instr);

   return true;
}

void
nir_instr_worklist_add_ssa_srcs(nir_instr_worklist *wl, nir_instr *instr)
{
   nir_foreach_src(instr, nir_instr_worklist_add_srcs_cb, wl);
}
