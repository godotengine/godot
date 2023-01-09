/*
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

static bool
instr_is_continue(nir_instr *instr)
{
   if (instr->type != nir_instr_type_jump)
      return false;

   return nir_instr_as_jump(instr)->type == nir_jump_continue;
}

static bool
lower_trivial_continues_block(nir_block *block, nir_loop *loop)
{
   bool progress = false;
   nir_instr *first_instr = nir_block_first_instr(block);
   if (!first_instr || instr_is_continue(first_instr)) {
      /* The block contains only a continue if anything */
      nir_cf_node *prev_node = nir_cf_node_prev(&block->cf_node);
      if (prev_node && prev_node->type == nir_cf_node_if) {
         nir_if *prev_if = nir_cf_node_as_if(prev_node);
         progress |= lower_trivial_continues_block(
            nir_if_last_then_block(prev_if), loop);
         progress |= lower_trivial_continues_block(
            nir_if_last_else_block(prev_if), loop);
      }

      /* The lower_phis_to_regs helper is smart enough to push phi sources as
       * far up if-ladders as it possibly can.  In other words, the recursive
       * calls above shouldn't be adding instructions to this block since it
       * follows an if statement.
       */
      first_instr = nir_block_first_instr(block);
      assert(!first_instr || instr_is_continue(first_instr));
   }

   nir_instr *last_instr = nir_block_last_instr(block);
   if (!last_instr || !instr_is_continue(last_instr))
      return progress;

   /* We're at the end of a block that goes straight through to the end of the
    * loop and continues to the top.  We can exit normally instead of jump.
    */
   nir_lower_phis_to_regs_block(nir_loop_first_block(loop));
   nir_instr_remove(last_instr);
   return true;
}

static bool
lower_trivial_continues_list(struct exec_list *cf_list,
                             bool list_ends_at_loop_tail,
                             nir_loop *loop)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, cf_node, node, cf_list) {
      bool at_loop_tail = list_ends_at_loop_tail &&
                          &cf_node->node == exec_list_get_tail(cf_list);
      switch (cf_node->type) {
      case nir_cf_node_block:
         break;

      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(cf_node);
         if (lower_trivial_continues_list(&nif->then_list, at_loop_tail, loop))
            progress = true;
         if (lower_trivial_continues_list(&nif->else_list, at_loop_tail, loop))
            progress = true;
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(cf_node);
         if (lower_trivial_continues_list(&loop->body, true, loop))
            progress = true;
         if (lower_trivial_continues_block(nir_loop_last_block(loop), loop))
            progress = true;
         break;
      }

      case nir_cf_node_function:
         unreachable("Invalid cf type");
      }
   }

   return progress;
}

/**
 * This simple pass gets rid of any "trivial" continues, i.e. those that are
 * at the tail of a loop where we can just delete the continue instruction and
 * control-flow will naturally and harmlessly take us back to the top of the
 * loop.
 */
bool
nir_opt_trivial_continues(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl == NULL)
         continue;

      /* First we run the simple pass to get rid of pesky continues */
      if (lower_trivial_continues_list(&function->impl->body, false, NULL)) {
         nir_metadata_preserve(function->impl, nir_metadata_none);

         /* If that made progress, we're no longer really in SSA form. */
         nir_lower_regs_to_ssa_impl(function->impl);
         progress = true;
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}
