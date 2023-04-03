/*
 * Copyright © 2016 Intel Corporation
 * Copyright © 2019 Valve Corporation
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

/**
 * \file nir_opt_move.c
 *
 * This pass can move various operations just before their first use inside the
 * same basic block. Usually this is to reduce register usage. It's probably
 * not a good idea to use this in an optimization loop.
 *
 * Moving comparisons is useful because many GPUs generate condition codes
 * for comparisons, and use predication for conditional selects and control
 * flow.  In a sequence such as:
 *
 *     vec1 32 ssa_1 = flt a b
 *     <some other operations>
 *     vec1 32 ssa_2 = bcsel ssa_1 c d
 *
 * the backend would likely do the comparison, producing condition codes,
 * then save those to a boolean value.  The intervening operations might
 * trash the condition codes.  Then, in order to do the bcsel, it would
 * need to re-populate the condition code register based on the boolean.
 *
 * By moving the comparison just before the bcsel, the condition codes could
 * be used directly.  This eliminates the need to reload them from the boolean
 * (generally eliminating an instruction).  It may also eliminate the need to
 * create a boolean value altogether (unless it's used elsewhere), which could
 * lower register pressure.
 */

static inline bool
src_is_ssa(nir_src *src, void *state)
{
   return src->is_ssa;
}

static inline bool
instr_reads_register(nir_instr *instr)
{
   return !nir_foreach_src(instr, src_is_ssa, NULL);
}

static bool
nir_opt_move_block(nir_block *block, nir_move_options options)
{
   bool progress = false;
   nir_instr *last_instr = nir_block_ends_in_jump(block) ?
                           nir_block_last_instr(block) : NULL;
   const nir_if *iff = nir_block_get_following_if(block);
   const nir_instr *if_cond_instr = iff ? iff->condition.parent_instr : NULL;

   /* Walk the instructions backwards.
    * The instructions get indexed while iterating.
    * For each instruction which can be moved, find the earliest user
    * and insert the instruction before it.
    * If multiple instructions have the same user,
    * the original order is kept.
    */
   unsigned index =  1;
   unsigned last_reg_def_index = 0;
   nir_foreach_instr_reverse_safe(instr, block) {
      instr->index = index++;

      /* Don't move register defs  */
      if (nir_instr_def_is_register(instr)) {
         last_reg_def_index = instr->index;
         continue;
      }

      /* Check if this instruction can be moved downwards */
      if (!nir_can_move_instr(instr, options))
         continue;

      /* Check all users in this block which is the first */
      const nir_ssa_def *def = nir_instr_ssa_def(instr);
      nir_instr *first_user = instr == if_cond_instr ? NULL : last_instr;
      nir_foreach_use(use, def) {
         nir_instr *parent = use->parent_instr;
         if (parent->type == nir_instr_type_phi || parent->block != block)
            continue;
         if (!first_user || parent->index > first_user->index)
            first_user = parent;
      }

      if (first_user) {
         /* Check predecessor instructions for the same index to keep the order */
         while (nir_instr_prev(first_user)->index == first_user->index)
            first_user = nir_instr_prev(first_user);

         /* check if the user is already the immediate successor */
         if (nir_instr_prev(first_user) == instr)
            continue;

         /* Don't move register reads past register defs  */
         if (first_user->index < last_reg_def_index &&
             instr_reads_register(instr)) {
            continue;
         }

         /* Insert the instruction before it's first user */
         exec_node_remove(&instr->node);
         instr->index = first_user->index;
         exec_node_insert_node_before(&first_user->node, &instr->node);
         progress = true;
         continue;
      }

      /* No user was found in this block:
       * This instruction will be moved to the end of the block.
       */
      assert(nir_block_last_instr(block)->type != nir_instr_type_jump);
      if (instr == nir_block_last_instr(block))
         continue;

      exec_node_remove(&instr->node);
      instr->index = 0;
      exec_list_push_tail(&block->instr_list, &instr->node);

      /* update last_instr */
      last_instr = instr;

      progress = true;
   }

   return progress;
}

bool
nir_opt_move(nir_shader *shader, nir_move_options options)
{
   bool progress = false;

   nir_foreach_function(func, shader) {
      if (!func->impl)
         continue;

      bool impl_progress = false;
      nir_foreach_block(block, func->impl) {
         if (nir_opt_move_block(block, options))
            impl_progress = true;
      }

      if (impl_progress) {
         nir_metadata_preserve(func->impl, nir_metadata_block_index |
                                           nir_metadata_dominance |
                                           nir_metadata_live_ssa_defs);
         progress = true;
      } else {
         nir_metadata_preserve(func->impl, nir_metadata_all);
      }
   }

   return progress;
}
