/*
 * Copyright © 2015 Red Hat
 * Copyright © 2016 Intel Corporation
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
#include "program/prog_instruction.h"

/**
 * This pass adds <0.5, 0.5> to all uses of gl_FragCoord.
 *
 * Run before nir_lower_io().
 *
 * For a more full featured pass, consider using nir_lower_wpos_ytransform(),
 * which can handle pixel center integer / half integer, and origin lower
 * left / upper left transformations.
 *
 * This simple pass is primarily intended for use by Vulkan drivers on
 * hardware which provides an integer pixel center.  Vulkan mandates that
 * the pixel center must be half-integer, and also that the coordinate
 * system's origin must be upper left.  This means that there's no need
 * for a uniform - we can always just add a constant. In the case that
 * sample shading is enabled, Vulkan expects FragCoord to include sample
 * positions.
 */

static void
update_fragcoord(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_ssa_def *wpos = &intr->dest.ssa;

   assert(intr->dest.is_ssa);

   b->cursor = nir_after_instr(&intr->instr);

   nir_ssa_def *spos = nir_load_sample_pos_or_center(b);

   wpos = nir_fadd(b, wpos,
                   nir_vec4(b,
                            nir_channel(b, spos, 0),
                            nir_channel(b, spos, 1),
                            nir_imm_float(b, 0.0f),
                            nir_imm_float(b, 0.0f)));

   nir_ssa_def_rewrite_uses_after(&intr->dest.ssa, wpos,
                                  wpos->parent_instr);
}

static bool
lower_wpos_center_block(nir_builder *b, nir_block *block)
{
   bool progress = false;

   nir_foreach_instr(instr, block) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_frag_coord) {
            update_fragcoord(b, intr);
            progress = true;
         }
      }
   }

   return progress;
}

bool
nir_lower_wpos_center(nir_shader *shader)
{
   bool progress = false;
   nir_builder b;

   assert(shader->info.stage == MESA_SHADER_FRAGMENT);

   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_builder_init(&b, function->impl);

         nir_foreach_block(block, function->impl) {
            progress = lower_wpos_center_block(&b, block) ||
                       progress;
         }
         nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                               nir_metadata_dominance);
      }
   }

   return progress;
}
