/*
 * Copyright (C) 2019 Collabora, Ltd.
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
 * Authors (Collabora):
 *   Alyssa Rosenzweig <alyssa.rosenzweig@collabora.com>
 */

/**
 * @file
 *
 * Flushes undefined SSA values to a zero vector fo the appropriate component
 * count, to avoid undefined behaviour in the resulting shader. Not required
 * for conformance as use of uninitialized variables is explicitly left
 * undefined by the spec.  Works around buggy apps, however.
 *
 * Call immediately after nir_opt_undef. If called before, larger optimization
 * opportunities from the former pass will be missed. If called outside of an
 * optimization loop, constant propagation and algebraic optimizations won't be
 * able to kick in to reduce stuff consuming the zero.
 */

#include "nir_builder.h"

static bool
lower_undef_instr_to_zero(nir_builder *b, nir_instr *instr, UNUSED void *_state)
{
   if (instr->type != nir_instr_type_ssa_undef)
      return false;

   nir_ssa_undef_instr *und = nir_instr_as_ssa_undef(instr);
   b->cursor = nir_instr_remove(&und->instr);
   nir_ssa_def *zero = nir_imm_zero(b, und->def.num_components,
                                       und->def.bit_size);
   nir_ssa_def_rewrite_uses(&und->def, zero);
   return true;
}

bool
nir_lower_undef_to_zero(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_undef_instr_to_zero,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance, NULL);
}
