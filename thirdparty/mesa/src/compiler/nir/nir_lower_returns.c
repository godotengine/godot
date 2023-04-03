/*
 * Copyright © 2015 Intel Corporation
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
#include "nir_control_flow.h"

struct lower_returns_state {
   nir_builder builder;
   struct exec_list *cf_list;
   nir_loop *loop;
   nir_variable *return_flag;

   /* This indicates that we have a return which is predicated on some form of
    * control-flow.  Since whether or not the return happens can only be
    * determined dynamically at run-time, everything that occurs afterwards
    * needs to be predicated on the return flag variable.
    */
   bool has_predicated_return;

   bool removed_unreachable_code;
};

static bool lower_returns_in_cf_list(struct exec_list *cf_list,
                                     struct lower_returns_state *state);

static void
predicate_following(nir_cf_node *node, struct lower_returns_state *state)
{
   nir_builder *b = &state->builder;
   b->cursor = nir_after_cf_node_and_phis(node);

   if (!state->loop && nir_cursors_equal(b->cursor, nir_after_cf_list(state->cf_list)))
      return; /* Nothing to predicate */

   assert(state->return_flag);

   nir_if *if_stmt = nir_push_if(b, nir_load_var(b, state->return_flag));

   if (state->loop) {
      /* If we're inside of a loop, then all we need to do is insert a
       * conditional break.
       */
      nir_jump(b, nir_jump_break);

      nir_block *block = nir_cursor_current_block(b->cursor);
      nir_insert_phi_undef(block->successors[0], block);
   } else {
      /* Otherwise, we need to actually move everything into the else case
       * of the if statement.
       */
      nir_cf_list list;
      nir_cf_extract(&list, nir_after_cf_node(&if_stmt->cf_node),
                            nir_after_cf_list(state->cf_list));
      assert(!exec_list_is_empty(&list.list));
      nir_cf_reinsert(&list, nir_before_cf_list(&if_stmt->else_list));
   }

   nir_pop_if(b, NULL);
}

static bool
lower_returns_in_loop(nir_loop *loop, struct lower_returns_state *state)
{
   nir_loop *parent = state->loop;
   state->loop = loop;
   bool progress = lower_returns_in_cf_list(&loop->body, state);
   state->loop = parent;

   /* If the recursive call made progress, then there were returns inside
    * of the loop.  These would have been lowered to breaks with the return
    * flag set to true.  We need to predicate everything following the loop
    * on the return flag.
    */
   if (progress) {
      predicate_following(&loop->cf_node, state);
      state->has_predicated_return = true;
   }

   return progress;
}

static bool
lower_returns_in_if(nir_if *if_stmt, struct lower_returns_state *state)
{
   bool progress, then_progress, else_progress;

   bool has_predicated_return = state->has_predicated_return;
   state->has_predicated_return = false;

   then_progress = lower_returns_in_cf_list(&if_stmt->then_list, state);
   else_progress = lower_returns_in_cf_list(&if_stmt->else_list, state);
   progress = then_progress || else_progress;

   /* If either of the recursive calls made progress, then there were
    * returns inside of the body of the if.  If we're in a loop, then these
    * were lowered to breaks which automatically skip to the end of the
    * loop so we don't have to do anything.  If we're not in a loop, then
    * all we know is that the return flag is set appropriately and that the
    * recursive calls ensured that nothing gets executed *inside* the if
    * after a return.  In order to ensure nothing outside gets executed
    * after a return, we need to predicate everything following on the
    * return flag.
    */
   if (progress && !state->loop) {
      if (state->has_predicated_return) {
         predicate_following(&if_stmt->cf_node, state);
      } else {
         /* If there are no nested returns we can just add the instructions to
          * the end of the branch that doesn't have the return.
          */

         /* nir_cf_extract will not extract phis at the start of the block. In
          * this case we know that any phis will have to have a single
          * predecessor, so we can just replace the phi with its single source.
          */
         nir_block *succ_block = nir_after_cf_node(&if_stmt->cf_node).block;
         nir_opt_remove_phis_block(succ_block);
         assert(nir_block_first_instr(succ_block) == NULL ||
                nir_block_first_instr(succ_block)->type != nir_instr_type_phi);

         nir_cf_list list;
         nir_cf_extract(&list, nir_after_cf_node(&if_stmt->cf_node),
                        nir_after_cf_list(state->cf_list));

         if (then_progress && else_progress) {
            /* Both branches return so delete instructions following the if */
            nir_cf_delete(&list);
         } else if (then_progress) {
            nir_cf_reinsert(&list, nir_after_cf_list(&if_stmt->else_list));
         } else {
            nir_cf_reinsert(&list, nir_after_cf_list(&if_stmt->then_list));
         }
      }
   }

   state->has_predicated_return = progress || has_predicated_return;

   return progress;
}

static bool
lower_returns_in_block(nir_block *block, struct lower_returns_state *state)
{
   if (block->predecessors->entries == 0 &&
       block != nir_start_block(state->builder.impl)) {
      /* This block is unreachable.  Delete it and everything after it. */
      nir_cf_list list;
      nir_cf_extract(&list, nir_before_cf_node(&block->cf_node),
                            nir_after_cf_list(state->cf_list));

      if (exec_list_is_empty(&list.list)) {
         /* There's nothing here, which also means there's nothing in this
          * block so we have nothing to do.
          */
         return false;
      } else {
         state->removed_unreachable_code = true;
         nir_cf_delete(&list);
         return false;
      }
   }

   nir_instr *last_instr = nir_block_last_instr(block);
   if (last_instr == NULL)
      return false;

   if (last_instr->type != nir_instr_type_jump)
      return false;

   nir_jump_instr *jump = nir_instr_as_jump(last_instr);
   if (jump->type != nir_jump_return)
      return false;

   nir_instr_remove(&jump->instr);

   /* If this is a return in the last block of the function there is nothing
    * more to do once its removed.
    */
   if (block == nir_impl_last_block(state->builder.impl))
      return true;

   nir_builder *b = &state->builder;

   /* Set the return flag */
   if (state->return_flag == NULL) {
      state->return_flag =
         nir_local_variable_create(b->impl, glsl_bool_type(), "return");

      /* Initialize the variable to 0 */
      b->cursor = nir_before_cf_list(&b->impl->body);
      nir_store_var(b, state->return_flag, nir_imm_false(b), 1);
   }

   b->cursor = nir_after_block(block);
   nir_store_var(b, state->return_flag, nir_imm_true(b), 1);

   if (state->loop) {
      /* We're in a loop;  we need to break out of it. */
      nir_jump(b, nir_jump_break);

      nir_insert_phi_undef(block->successors[0], block);
   } else {
      /* Not in a loop;  we'll deal with predicating later*/
      assert(nir_cf_node_next(&block->cf_node) == NULL);
   }

   return true;
}

static bool
lower_returns_in_cf_list(struct exec_list *cf_list,
                         struct lower_returns_state *state)
{
   bool progress = false;

   struct exec_list *parent_list = state->cf_list;
   state->cf_list = cf_list;

   /* We iterate over the list backwards because any given lower call may
    * take everything following the given CF node and predicate it.  In
    * order to avoid recursion/iteration problems, we want everything after
    * a given node to already be lowered before this happens.
    */
   foreach_list_typed_reverse_safe(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block:
         if (lower_returns_in_block(nir_cf_node_as_block(node), state))
            progress = true;
         break;

      case nir_cf_node_if:
         if (lower_returns_in_if(nir_cf_node_as_if(node), state))
            progress = true;
         break;

      case nir_cf_node_loop:
         if (lower_returns_in_loop(nir_cf_node_as_loop(node), state))
            progress = true;
         break;

      default:
         unreachable("Invalid inner CF node type");
      }
   }

   state->cf_list = parent_list;

   return progress;
}

bool
nir_lower_returns_impl(nir_function_impl *impl)
{
   struct lower_returns_state state;

   state.cf_list = &impl->body;
   state.loop = NULL;
   state.return_flag = NULL;
   state.has_predicated_return = false;
   state.removed_unreachable_code = false;
   nir_builder_init(&state.builder, impl);

   bool progress = lower_returns_in_cf_list(&impl->body, &state);
   progress = progress || state.removed_unreachable_code;

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_none);
      nir_repair_ssa_impl(impl);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_lower_returns(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress = nir_lower_returns_impl(function->impl) || progress;
   }

   return progress;
}
