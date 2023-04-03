/*
 * Copyright 2018 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "nir.h"
#include "compiler/nir/nir_builder.h"

static bool
lower_discard_if_instr(nir_builder *b, nir_instr *instr_, void *cb_data)
{
   nir_lower_discard_if_options options = *(nir_lower_discard_if_options *)cb_data;

   if (instr_->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *instr = nir_instr_as_intrinsic(instr_);

   switch (instr->intrinsic) {
   case nir_intrinsic_discard_if:
      if (!(options & nir_lower_discard_if_to_cf))
         return false;
      break;
   case nir_intrinsic_demote_if:
      if (!(options & nir_lower_demote_if_to_cf))
         return false;
      break;
   case nir_intrinsic_terminate_if:
      if (!(options & nir_lower_terminate_if_to_cf))
         return false;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&instr->instr);

   nir_if *if_stmt = nir_push_if(b, nir_ssa_for_src(b, instr->src[0], 1));
   switch (instr->intrinsic) {
   case nir_intrinsic_discard_if:
      nir_discard(b);
      break;
   case nir_intrinsic_demote_if:
      nir_demote(b);
      break;
   case nir_intrinsic_terminate_if:
      nir_terminate(b);
      break;
   default:
      unreachable("bad intrinsic");
   }
   nir_pop_if(b, if_stmt);
   nir_instr_remove(&instr->instr);
   return true;

   /* a shader like this (shaders@glsl-fs-discard-04):

      uniform int j, k;

      void main()
      {
       for (int i = 0; i < j; i++) {
        if (i > k)
         continue;
        discard;
       }
       gl_FragColor = vec4(0.0, 1.0, 0.0, 0.0);
      }



      will generate nir like:

      loop   {
         //snip
         if   ssa_11   {
            block   block_5:
            /   preds:   block_4   /
            vec1   32   ssa_17   =   iadd   ssa_50,   ssa_31
            /   succs:   block_7   /
         }   else   {
            block   block_6:
            /   preds:   block_4   /
            intrinsic   discard   ()   () <-- not last instruction
            vec1   32   ssa_23   =   iadd   ssa_50,   ssa_31 <-- dead code loop itr increment
            /   succs:   block_7   /
         }
         //snip
      }

      which means that we can't assert like this:

      assert(instr->intrinsic != nir_intrinsic_discard ||
             nir_block_last_instr(instr->instr.block) == &instr->instr);


      and it's unnecessary anyway since later optimizations will dce the
      instructions following the discard
    */

   return false;
}

bool
nir_lower_discard_if(nir_shader *shader, nir_lower_discard_if_options options)
{
   return nir_shader_instructions_pass(shader,
                                       lower_discard_if_instr,
                                       nir_metadata_none,
                                       &options);
}
