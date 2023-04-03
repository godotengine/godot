/*
 * Copyright © 2014 Connor Abbott
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
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#include "nir.h"
#include "nir_control_flow.h"

/*
 * This file implements an optimization that deletes statically
 * unreachable/dead code. In NIR, one way this can happen is when an if
 * statement has a constant condition:
 *
 * if (true) {
 *    ...
 * }
 *
 * We delete the if statement and paste the contents of the always-executed
 * branch into the surrounding control flow, possibly removing more code if
 * the branch had a jump at the end.
 *
 * Another way is that control flow can end in a jump so that code after it
 * never gets executed. In particular, this can happen after optimizing
 * something like:
 *
 * if (true) {
 *    ...
 *    break;
 * }
 * ...
 *
 * We also consider the case where both branches of an if end in a jump, e.g.:
 *
 * if (...) {
 *    break;
 * } else {
 *    continue;
 * }
 * ...
 *
 * Finally, we also handle removing useless loops and ifs, i.e. loops and ifs
 * with no side effects and without any definitions that are used
 * elsewhere. This case is a little different from the first two in that the
 * code is actually run (it just never does anything), but there are similar
 * issues with needing to be careful with restarting after deleting the
 * cf_node (see dead_cf_list()) so this is a convenient place to remove them.
 */

static void
remove_after_cf_node(nir_cf_node *node)
{
   nir_cf_node *end = node;
   while (!nir_cf_node_is_last(end))
      end = nir_cf_node_next(end);

   nir_cf_list list;
   nir_cf_extract(&list, nir_after_cf_node(node), nir_after_cf_node(end));
   nir_cf_delete(&list);
}

static void
opt_constant_if(nir_if *if_stmt, bool condition)
{
   /* First, we need to remove any phi nodes after the if by rewriting uses to
    * point to the correct source.
    */
   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&if_stmt->cf_node));
   nir_block *last_block = condition ? nir_if_last_then_block(if_stmt)
                                     : nir_if_last_else_block(if_stmt);

   nir_foreach_instr_safe(instr, after) {
      if (instr->type != nir_instr_type_phi)
         break;

      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_ssa_def *def = NULL;
      nir_foreach_phi_src(phi_src, phi) {
         if (phi_src->pred != last_block)
            continue;

         assert(phi_src->src.is_ssa);
         def = phi_src->src.ssa;
      }

      assert(def);
      assert(phi->dest.is_ssa);
      nir_ssa_def_rewrite_uses(&phi->dest.ssa, def);
      nir_instr_remove(instr);
   }

   /* The control flow list we're about to paste in may include a jump at the
    * end, and in that case we have to delete the rest of the control flow
    * list after the if since it's unreachable and the validator will balk if
    * we don't.
    */

   if (!exec_list_is_empty(&last_block->instr_list)) {
      nir_instr *last_instr = nir_block_last_instr(last_block);
      if (last_instr->type == nir_instr_type_jump)
         remove_after_cf_node(&if_stmt->cf_node);
   }

   /* Finally, actually paste in the then or else branch and delete the if. */
   struct exec_list *cf_list = condition ? &if_stmt->then_list
                                         : &if_stmt->else_list;

   nir_cf_list list;
   nir_cf_list_extract(&list, cf_list);
   nir_cf_reinsert(&list, nir_after_cf_node(&if_stmt->cf_node));
   nir_cf_node_remove(&if_stmt->cf_node);
}

static bool
def_only_used_in_cf_node(nir_ssa_def *def, void *_node)
{
   nir_cf_node *node = _node;
   assert(node->type == nir_cf_node_loop || node->type == nir_cf_node_if);

   nir_block *before = nir_cf_node_as_block(nir_cf_node_prev(node));
   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(node));

   nir_foreach_use(use, def) {
      /* Because NIR is structured, we can easily determine whether or not a
       * value escapes a CF node by looking at the block indices of its uses
       * to see if they lie outside the bounds of the CF node.
       *
       * Note: Normally, the uses of a phi instruction are considered to be
       * used in the block that is the predecessor of the phi corresponding to
       * that use.  If we were computing liveness or something similar, that
       * would mean a special case here for phis.  However, we're trying here
       * to determine if the SSA def ever escapes the loop.  If it's used by a
       * phi that lives outside the loop then it doesn't matter if the
       * corresponding predecessor is inside the loop or not because the value
       * can go through the phi into the outside world and escape the loop.
       */
      if (use->parent_instr->block->index <= before->index ||
          use->parent_instr->block->index >= after->index)
         return false;
   }

   /* Same check for if-condition uses */
   nir_foreach_if_use(use, def) {
      nir_block *use_block =
         nir_cf_node_as_block(nir_cf_node_prev(&use->parent_if->cf_node));

      if (use_block->index <= before->index ||
          use_block->index >= after->index)
         return false;
   }

   return true;
}

/*
 * Test if a loop or if node is dead. Such nodes are dead if:
 *
 * 1) It has no side effects (i.e. intrinsics which could possibly affect the
 * state of the program aside from producing an SSA value, indicated by a lack
 * of NIR_INTRINSIC_CAN_ELIMINATE).
 *
 * 2) It has no phi instructions after it, since those indicate values inside
 * the node being used after the node.
 *
 * 3) None of the values defined inside the node is used outside the node,
 * i.e. none of the definitions that dominate the node exit are used outside.
 *
 * If those conditions hold, then the node is dead and can be deleted.
 */

static bool
node_is_dead(nir_cf_node *node)
{
   assert(node->type == nir_cf_node_loop || node->type == nir_cf_node_if);

   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(node));

   /* Quick check if there are any phis that follow this CF node.  If there
    * are, then we automatically know it isn't dead.
    */
   if (!exec_list_is_empty(&after->instr_list) &&
       nir_block_first_instr(after)->type == nir_instr_type_phi)
      return false;

   nir_function_impl *impl = nir_cf_node_get_function(node);
   nir_metadata_require(impl, nir_metadata_block_index);

   nir_foreach_block_in_cf_node(block, node) {
      bool inside_loop = node->type == nir_cf_node_loop;
      for (nir_cf_node *n = &block->cf_node;
           !inside_loop && n != node; n = n->parent) {
         if (n->type == nir_cf_node_loop)
            inside_loop = true;
      }

      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;

         /* Return and halt instructions can cause us to skip over other
          * side-effecting instructions after the loop, so consider them to
          * have side effects here.
          *
          * When the block is not inside a loop, break and continue might also
          * cause a skip.
          */
         if (instr->type == nir_instr_type_jump &&
             (!inside_loop ||
              nir_instr_as_jump(instr)->type == nir_jump_return ||
              nir_instr_as_jump(instr)->type == nir_jump_halt))
            return false;

         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (!(nir_intrinsic_infos[intrin->intrinsic].flags &
                NIR_INTRINSIC_CAN_ELIMINATE))
               return false;

            switch (intrin->intrinsic) {
            case nir_intrinsic_load_deref:
            case nir_intrinsic_load_ssbo:
            case nir_intrinsic_load_global:
               /* If there's a memory barrier after the loop, a load might be
                * required to happen before some other instruction after the
                * barrier, so it is not valid to eliminate it -- unless we
                * know we can reorder it.
                *
                * Consider only loads that the result can be affected by other
                * invocations.
                */
               if (intrin->intrinsic == nir_intrinsic_load_deref) {
                  nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
                  if (!nir_deref_mode_may_be(deref, nir_var_mem_ssbo |
                                                    nir_var_mem_shared |
                                                    nir_var_mem_global |
                                                    nir_var_shader_out))
                     break;
               }
               if (nir_intrinsic_access(intrin) & ACCESS_CAN_REORDER)
                  break;
               return false;

            case nir_intrinsic_load_shared:
            case nir_intrinsic_load_shared2_amd:
            case nir_intrinsic_load_output:
            case nir_intrinsic_load_per_vertex_output:
               /* Same as above loads. */
               return false;

            default:
               /* Do nothing. */
               break;
            }
         }

         if (!nir_foreach_ssa_def(instr, def_only_used_in_cf_node, node))
            return false;
      }
   }

   return true;
}

static bool
dead_cf_block(nir_block *block)
{
   nir_if *following_if = nir_block_get_following_if(block);
   if (following_if) {
      if (nir_src_is_const(following_if->condition)) {
         opt_constant_if(following_if, nir_src_as_bool(following_if->condition));
         return true;
      }

      if (node_is_dead(&following_if->cf_node)) {
         nir_cf_node_remove(&following_if->cf_node);
         return true;
      }
   }

   nir_loop *following_loop = nir_block_get_following_loop(block);
   if (!following_loop)
      return false;

   if (!node_is_dead(&following_loop->cf_node))
      return false;

   nir_cf_node_remove(&following_loop->cf_node);
   return true;
}

static bool
dead_cf_list(struct exec_list *list, bool *list_ends_in_jump)
{
   bool progress = false;
   *list_ends_in_jump = false;

   nir_cf_node *prev = NULL;

   foreach_list_typed(nir_cf_node, cur, node, list) {
      switch (cur->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(cur);
         if (dead_cf_block(block)) {
            /* We just deleted the if or loop after this block, so we may have
             * deleted the block before or after it -- which one is an
             * implementation detail. Therefore, to recover the place we were
             * at, we have to use the previous cf_node.
             */

            if (prev) {
               cur = nir_cf_node_next(prev);
            } else {
               cur = exec_node_data(nir_cf_node, exec_list_get_head(list),
                                    node);
            }

            block = nir_cf_node_as_block(cur);

            progress = true;
         }

         if (nir_block_ends_in_jump(block)) {
            *list_ends_in_jump = true;

            if (!exec_node_is_tail_sentinel(cur->node.next)) {
               remove_after_cf_node(cur);
               return true;
            }
         }

         break;
      }

      case nir_cf_node_if: {
         nir_if *if_stmt = nir_cf_node_as_if(cur);
         bool then_ends_in_jump, else_ends_in_jump;
         progress |= dead_cf_list(&if_stmt->then_list, &then_ends_in_jump);
         progress |= dead_cf_list(&if_stmt->else_list, &else_ends_in_jump);

         if (then_ends_in_jump && else_ends_in_jump) {
            *list_ends_in_jump = true;
            nir_block *next = nir_cf_node_as_block(nir_cf_node_next(cur));
            if (!exec_list_is_empty(&next->instr_list) ||
                !exec_node_is_tail_sentinel(next->cf_node.node.next)) {
               remove_after_cf_node(cur);
               return true;
            }
         }

         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(cur);
         bool dummy;
         progress |= dead_cf_list(&loop->body, &dummy);

         nir_block *next = nir_cf_node_as_block(nir_cf_node_next(cur));
         if (next->predecessors->entries == 0 &&
             (!exec_list_is_empty(&next->instr_list) ||
             !exec_node_is_tail_sentinel(next->cf_node.node.next))) {
            remove_after_cf_node(cur);
            return true;
         }
         break;
      }

      default:
         unreachable("unknown cf node type");
      }

      prev = cur;
   }

   return progress;
}

static bool
opt_dead_cf_impl(nir_function_impl *impl)
{
   bool dummy;
   bool progress = dead_cf_list(&impl->body, &dummy);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_none);

      /* The CF manipulation code called by this pass is smart enough to keep
       * from breaking any SSA use/def chains by replacing any uses of removed
       * instructions with SSA undefs.  However, it's not quite smart enough
       * to always preserve the dominance properties.  In particular, if you
       * remove the one break from a loop, stuff in the loop may still be used
       * outside the loop even though there's no path between the two.  We can
       * easily fix these issues by calling nir_repair_ssa which will ensure
       * that the dominance properties hold.
       */
      nir_repair_ssa_impl(impl);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_opt_dead_cf(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader)
      if (function->impl)
         progress |= opt_dead_cf_impl(function->impl);

   return progress;
}
