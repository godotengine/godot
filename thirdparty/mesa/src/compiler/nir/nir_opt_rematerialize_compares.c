/*
 * Copyright © 2019 Intel Corporation
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
#include "nir/nir_builder.h"
#include "nir_constant_expressions.h"
#include "nir_control_flow.h"
#include "nir_loop_analyze.h"

static bool
is_two_src_comparison(const nir_alu_instr *instr)
{
   switch (instr->op) {
   case nir_op_flt:
   case nir_op_flt32:
   case nir_op_fge:
   case nir_op_fge32:
   case nir_op_feq:
   case nir_op_feq32:
   case nir_op_fneu:
   case nir_op_fneu32:
   case nir_op_ilt:
   case nir_op_ilt32:
   case nir_op_ult:
   case nir_op_ult32:
   case nir_op_ige:
   case nir_op_ige32:
   case nir_op_uge:
   case nir_op_uge32:
   case nir_op_ieq:
   case nir_op_ieq32:
   case nir_op_ine:
   case nir_op_ine32:
      return true;
   default:
      return false;
   }
}

static bool
all_srcs_are_ssa(const nir_alu_instr *instr)
{
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      if (!instr->src[i].src.is_ssa)
         return false;
   }

   return true;
}


static bool
all_uses_are_bcsel(const nir_alu_instr *instr)
{
   if (!instr->dest.dest.is_ssa)
      return false;

   nir_foreach_use(use, &instr->dest.dest.ssa) {
      if (use->parent_instr->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *const alu = nir_instr_as_alu(use->parent_instr);
      if (alu->op != nir_op_bcsel &&
          alu->op != nir_op_b32csel)
         return false;

      /* Not only must the result be used by a bcsel, but it must be used as
       * the first source (the condition).
       */
      if (alu->src[0].src.ssa != &instr->dest.dest.ssa)
         return false;
   }

   return true;
}

static bool
nir_opt_rematerialize_compares_impl(nir_shader *shader, nir_function_impl *impl)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;

         nir_alu_instr *const alu = nir_instr_as_alu(instr);
         if (!is_two_src_comparison(alu))
            continue;

         if (!all_srcs_are_ssa(alu))
            continue;

         if (!all_uses_are_bcsel(alu))
            continue;

         /* At this point it is known that alu is a comparison instruction
          * that is only used by nir_op_bcsel and possibly by if-statements
          * (though the latter has not been explicitly checked).
          *
          * Iterate through each use of the comparison.  For every use (or use
          * by an if-statement) that is in a different block, emit a copy of
          * the comparison.  Care must be taken here.  The original
          * instruction must be duplicated only once in each block because CSE
          * cannot be run after this pass.
          */
         nir_foreach_use_safe(use, &alu->dest.dest.ssa) {
            nir_instr *const use_instr = use->parent_instr;

            /* If the use is in the same block as the def, don't
             * rematerialize.
             */
            if (use_instr->block == alu->instr.block)
               continue;

            nir_alu_instr *clone = nir_alu_instr_clone(shader, alu);

            nir_instr_insert_before(use_instr, &clone->instr);

            nir_alu_instr *const use_alu = nir_instr_as_alu(use_instr);
            for (unsigned i = 0; i < nir_op_infos[use_alu->op].num_inputs; i++) {
               if (use_alu->src[i].src.ssa == &alu->dest.dest.ssa) {
                  nir_instr_rewrite_src(&use_alu->instr,
                                        &use_alu->src[i].src,
                                        nir_src_for_ssa(&clone->dest.dest.ssa));
                  progress = true;
               }
            }
         }

         nir_foreach_if_use_safe(use, &alu->dest.dest.ssa) {
            nir_if *const if_stmt = use->parent_if;

            nir_block *const prev_block =
               nir_cf_node_as_block(nir_cf_node_prev(&if_stmt->cf_node));

            /* If the compare is from the previous block, don't
             * rematerialize.
             */
            if (prev_block == alu->instr.block)
               continue;

            nir_alu_instr *clone = nir_alu_instr_clone(shader, alu);

            nir_instr_insert_after_block(prev_block, &clone->instr);

            nir_if_rewrite_condition(if_stmt,
                                     nir_src_for_ssa(&clone->dest.dest.ssa));
            progress = true;
         }
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
nir_opt_rematerialize_compares(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl == NULL)
         continue;

      progress = nir_opt_rematerialize_compares_impl(shader, function->impl)
         || progress;
   }

   return progress;
}
