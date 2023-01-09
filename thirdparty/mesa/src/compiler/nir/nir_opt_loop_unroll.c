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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_control_flow.h"
#include "nir_loop_analyze.h"


/* This limit is chosen fairly arbitrarily.  GLSL IR max iteration is 32
 * instructions. (Multiply counting nodes and magic number 5.)  But there is
 * no 1:1 mapping between GLSL IR and NIR so 25 was picked because it seemed
 * to give about the same results. Around 5 instructions per node.  But some
 * loops that would unroll with GLSL IR fail to unroll if we set this to 25 so
 * we set it to 26.
 */
#define LOOP_UNROLL_LIMIT 26

/* Prepare this loop for unrolling by first converting to lcssa and then
 * converting the phis from the top level of the loop body to regs.
 * Partially converting out of SSA allows us to unroll the loop without having
 * to keep track of and update phis along the way which gets tricky and
 * doesn't add much value over converting to regs.
 *
 * The loop may have a jump instruction at the end of the loop which does
 * nothing.  Once we're out of SSA, we can safely delete it so we don't have
 * to deal with it later.
 */
static void
loop_prepare_for_unroll(nir_loop *loop)
{
   nir_rematerialize_derefs_in_use_blocks_impl(
      nir_cf_node_get_function(&loop->cf_node));

   nir_convert_loop_to_lcssa(loop);

   /* Lower phis at the top level of the loop body */
   foreach_list_typed_safe(nir_cf_node, node, node, &loop->body) {
      if (nir_cf_node_block == node->type) {
         nir_lower_phis_to_regs_block(nir_cf_node_as_block(node));
      }
   }

   /* Lower phis after the loop */
   nir_block *block_after_loop =
      nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));

   nir_lower_phis_to_regs_block(block_after_loop);

   /* Remove jump if it's the last instruction in the loop */
   nir_instr *last_instr = nir_block_last_instr(nir_loop_last_block(loop));
   if (last_instr && last_instr->type == nir_instr_type_jump) {
      nir_instr_remove(last_instr);
   }
}

static void
get_first_blocks_in_terminator(nir_loop_terminator *term,
                               nir_block **first_break_block,
                               nir_block **first_continue_block)
{
   if (term->continue_from_then) {
      *first_continue_block = nir_if_first_then_block(term->nif);
      *first_break_block = nir_if_first_else_block(term->nif);
   } else {
      *first_continue_block = nir_if_first_else_block(term->nif);
      *first_break_block = nir_if_first_then_block(term->nif);
   }
}

/**
 * Unroll a loop where we know exactly how many iterations there are and there
 * is only a single exit point.  Note here we can unroll loops with multiple
 * theoretical exits that only have a single terminating exit that we always
 * know is the "real" exit.
 *
 *     loop {
 *         ...instrs...
 *     }
 *
 * And the iteration count is 3, the output will be:
 *
 *     ...instrs... ...instrs... ...instrs...
 */
static void
simple_unroll(nir_loop *loop)
{
   nir_loop_terminator *limiting_term = loop->info->limiting_terminator;
   assert(nir_is_trivial_loop_if(limiting_term->nif,
                                 limiting_term->break_block));

   loop_prepare_for_unroll(loop);

   /* Skip over loop terminator and get the loop body. */
   list_for_each_entry(nir_loop_terminator, terminator,
                       &loop->info->loop_terminator_list,
                       loop_terminator_link) {

      /* Remove all but the limiting terminator as we know the other exit
       * conditions can never be met. Note we need to extract any instructions
       * in the continue from branch and insert then into the loop body before
       * removing it.
       */
      if (terminator->nif != limiting_term->nif) {
         nir_block *first_break_block;
         nir_block *first_continue_block;
         get_first_blocks_in_terminator(terminator, &first_break_block,
                                        &first_continue_block);

         assert(nir_is_trivial_loop_if(terminator->nif,
                                       terminator->break_block));

         nir_cf_list continue_from_lst;
         nir_cf_extract(&continue_from_lst,
                        nir_before_block(first_continue_block),
                        nir_after_block(terminator->continue_from_block));
         nir_cf_reinsert(&continue_from_lst,
                         nir_after_cf_node(&terminator->nif->cf_node));

         nir_cf_node_remove(&terminator->nif->cf_node);
      }
   }

   nir_block *first_break_block;
   nir_block *first_continue_block;
   get_first_blocks_in_terminator(limiting_term, &first_break_block,
                                  &first_continue_block);

   /* Pluck out the loop header */
   nir_block *header_blk = nir_loop_first_block(loop);
   nir_cf_list lp_header;
   nir_cf_extract(&lp_header, nir_before_block(header_blk),
                  nir_before_cf_node(&limiting_term->nif->cf_node));

   /* Add the continue from block of the limiting terminator to the loop body
    */
   nir_cf_list continue_from_lst;
   nir_cf_extract(&continue_from_lst, nir_before_block(first_continue_block),
                  nir_after_block(limiting_term->continue_from_block));
   nir_cf_reinsert(&continue_from_lst,
                   nir_after_cf_node(&limiting_term->nif->cf_node));

   /* Pluck out the loop body */
   nir_cf_list loop_body;
   nir_cf_extract(&loop_body, nir_after_cf_node(&limiting_term->nif->cf_node),
                  nir_after_block(nir_loop_last_block(loop)));

   struct hash_table *remap_table = _mesa_pointer_hash_table_create(NULL);

   /* Clone the loop header and insert before the loop */
   nir_cf_list_clone_and_reinsert(&lp_header, loop->cf_node.parent,
                                  nir_before_cf_node(&loop->cf_node),
                                  remap_table);

   for (unsigned i = 0; i < loop->info->max_trip_count; i++) {
      /* Clone loop body and insert before the loop */
      nir_cf_list_clone_and_reinsert(&loop_body, loop->cf_node.parent,
                                     nir_before_cf_node(&loop->cf_node),
                                     remap_table);

      /* Clone loop header and insert after loop body */
      nir_cf_list_clone_and_reinsert(&lp_header, loop->cf_node.parent,
                                     nir_before_cf_node(&loop->cf_node),
                                     remap_table);
   }

   /* Remove the break from the loop terminator and add instructions from
    * the break block after the unrolled loop.
    */
   nir_instr *break_instr = nir_block_last_instr(limiting_term->break_block);
   nir_instr_remove(break_instr);
   nir_cf_list break_list;
   nir_cf_extract(&break_list, nir_before_block(first_break_block),
                  nir_after_block(limiting_term->break_block));

   /* Clone so things get properly remapped */
   nir_cf_list_clone_and_reinsert(&break_list, loop->cf_node.parent,
                                  nir_before_cf_node(&loop->cf_node),
                                  remap_table);

   /* Remove the loop */
   nir_cf_node_remove(&loop->cf_node);

   /* Delete the original loop body, break block & header */
   nir_cf_delete(&lp_header);
   nir_cf_delete(&loop_body);
   nir_cf_delete(&break_list);

   _mesa_hash_table_destroy(remap_table, NULL);
}

static void
move_cf_list_into_loop_term(nir_cf_list *lst, nir_loop_terminator *term)
{
   /* Move the rest of the loop inside the continue-from-block */
   nir_cf_reinsert(lst, nir_after_block(term->continue_from_block));

   /* Remove the break */
   nir_instr_remove(nir_block_last_instr(term->break_block));
}

static nir_cursor
get_complex_unroll_insert_location(nir_cf_node *node, bool continue_from_then)
{
   if (node->type == nir_cf_node_loop) {
      return nir_before_cf_node(node);
   } else {
      nir_if *if_stmt = nir_cf_node_as_if(node);
      if (continue_from_then) {
         return nir_after_block(nir_if_last_then_block(if_stmt));
      } else {
         return nir_after_block(nir_if_last_else_block(if_stmt));
      }
   }
}

static nir_cf_node *
complex_unroll_loop_body(nir_loop *loop, nir_loop_terminator *unlimit_term,
                         nir_cf_list *lp_header, nir_cf_list *lp_body,
                         struct hash_table *remap_table,
                         unsigned num_times_to_clone)
{
   /* In the terminator that we have no trip count for move everything after
    * the terminator into the continue from branch.
    */
   nir_cf_list loop_end;
   nir_cf_extract(&loop_end, nir_after_cf_node(&unlimit_term->nif->cf_node),
                  nir_after_block(nir_loop_last_block(loop)));
   move_cf_list_into_loop_term(&loop_end, unlimit_term);

   /* Pluck out the loop body. */
   nir_cf_extract(lp_body, nir_before_block(nir_loop_first_block(loop)),
                  nir_after_block(nir_loop_last_block(loop)));

   /* Set unroll_loc to the loop as we will insert the unrolled loop before it
    */
   nir_cf_node *unroll_loc = &loop->cf_node;

   /* Temp list to store the cloned loop as we unroll */
   nir_cf_list unrolled_lp_body;

   for (unsigned i = 0; i < num_times_to_clone; i++) {

      nir_cursor cursor =
         get_complex_unroll_insert_location(unroll_loc,
                                            unlimit_term->continue_from_then);

      /* Clone loop header and insert in if branch */
      nir_cf_list_clone_and_reinsert(lp_header, loop->cf_node.parent,
                                     cursor, remap_table);

      cursor =
         get_complex_unroll_insert_location(unroll_loc,
                                            unlimit_term->continue_from_then);

      /* Clone loop body */
      nir_cf_list_clone(&unrolled_lp_body, lp_body, loop->cf_node.parent,
                        remap_table);

      unroll_loc = exec_node_data(nir_cf_node,
                                  exec_list_get_tail(&unrolled_lp_body.list),
                                  node);
      assert(unroll_loc->type == nir_cf_node_block &&
             exec_list_is_empty(&nir_cf_node_as_block(unroll_loc)->instr_list));

      /* Get the unrolled if node */
      unroll_loc = nir_cf_node_prev(unroll_loc);

      /* Insert unrolled loop body */
      nir_cf_reinsert(&unrolled_lp_body, cursor);
   }

   return unroll_loc;
}

/**
 * Unroll a loop with two exists when the trip count of one of the exits is
 * unknown.  If continue_from_then is true, the loop is repeated only when the
 * "then" branch of the if is taken; otherwise it is repeated only
 * when the "else" branch of the if is taken.
 *
 * For example, if the input is:
 *
 *      loop {
 *         ...phis/condition...
 *         if condition {
 *            ...then instructions...
 *         } else {
 *            ...continue instructions...
 *            break
 *         }
 *         ...body...
 *      }
 *
 * And the iteration count is 3, and unlimit_term->continue_from_then is true,
 * then the output will be:
 *
 *      ...condition...
 *      if condition {
 *         ...then instructions...
 *         ...body...
 *         if condition {
 *            ...then instructions...
 *            ...body...
 *            if condition {
 *               ...then instructions...
 *               ...body...
 *            } else {
 *               ...continue instructions...
 *            }
 *         } else {
 *            ...continue instructions...
 *         }
 *      } else {
 *         ...continue instructions...
 *      }
 */
static void
complex_unroll(nir_loop *loop, nir_loop_terminator *unlimit_term,
               bool limiting_term_second)
{
   assert(nir_is_trivial_loop_if(unlimit_term->nif,
                                 unlimit_term->break_block));

   nir_loop_terminator *limiting_term = loop->info->limiting_terminator;
   assert(nir_is_trivial_loop_if(limiting_term->nif,
                                 limiting_term->break_block));

   loop_prepare_for_unroll(loop);

   nir_block *header_blk = nir_loop_first_block(loop);

   nir_cf_list lp_header;
   nir_cf_list limit_break_list;
   unsigned num_times_to_clone;
   if (limiting_term_second) {
      /* Pluck out the loop header */
      nir_cf_extract(&lp_header, nir_before_block(header_blk),
                     nir_before_cf_node(&unlimit_term->nif->cf_node));

      /* We need some special handling when its the second terminator causing
       * us to exit the loop for example:
       *
       *   for (int i = 0; i < uniform_lp_count; i++) {
       *      colour = vec4(0.0, 1.0, 0.0, 1.0);
       *
       *      if (i == 1) {
       *         break;
       *      }
       *      ... any further code is unreachable after i == 1 ...
       *   }
       */
      nir_cf_list after_lt;
      nir_if *limit_if = limiting_term->nif;
      nir_cf_extract(&after_lt, nir_after_cf_node(&limit_if->cf_node),
                     nir_after_block(nir_loop_last_block(loop)));
      move_cf_list_into_loop_term(&after_lt, limiting_term);

      /* Because the trip count is the number of times we pass over the entire
       * loop before hitting a break when the second terminator is the
       * limiting terminator we can actually execute code inside the loop when
       * trip count == 0 e.g. the code above the break.  So we need to bump
       * the trip_count in order for the code below to clone anything.  When
       * trip count == 1 we execute the code above the break twice and the
       * code below it once so we need clone things twice and so on.
       */
      num_times_to_clone = loop->info->max_trip_count + 1;
   } else {
      /* Pluck out the loop header */
      nir_cf_extract(&lp_header, nir_before_block(header_blk),
                     nir_before_cf_node(&limiting_term->nif->cf_node));

      nir_block *first_break_block;
      nir_block *first_continue_block;
      get_first_blocks_in_terminator(limiting_term, &first_break_block,
                                     &first_continue_block);

      /* Remove the break then extract instructions from the break block so we
       * can insert them in the innermost else of the unrolled loop.
       */
      nir_instr *break_instr = nir_block_last_instr(limiting_term->break_block);
      nir_instr_remove(break_instr);
      nir_cf_extract(&limit_break_list, nir_before_block(first_break_block),
                     nir_after_block(limiting_term->break_block));

      nir_cf_list continue_list;
      nir_cf_extract(&continue_list, nir_before_block(first_continue_block),
                     nir_after_block(limiting_term->continue_from_block));

      nir_cf_reinsert(&continue_list,
                      nir_after_cf_node(&limiting_term->nif->cf_node));

      nir_cf_node_remove(&limiting_term->nif->cf_node);

      num_times_to_clone = loop->info->max_trip_count;
   }

   struct hash_table *remap_table = _mesa_pointer_hash_table_create(NULL);

   nir_cf_list lp_body;
   nir_cf_node *unroll_loc =
      complex_unroll_loop_body(loop, unlimit_term, &lp_header, &lp_body,
                               remap_table, num_times_to_clone);

   if (!limiting_term_second) {
      assert(unroll_loc->type == nir_cf_node_if);

      nir_cursor cursor =
         get_complex_unroll_insert_location(unroll_loc,
                                            unlimit_term->continue_from_then);

      /* Clone loop header and insert in if branch */
      nir_cf_list_clone_and_reinsert(&lp_header, loop->cf_node.parent,
                                     cursor, remap_table);

      cursor =
         get_complex_unroll_insert_location(unroll_loc,
                                            unlimit_term->continue_from_then);

      /* Clone so things get properly remapped, and insert break block from
       * the limiting terminator.
       */
      nir_cf_list_clone_and_reinsert(&limit_break_list, loop->cf_node.parent,
                                     cursor, remap_table);

      nir_cf_delete(&limit_break_list);
   }

   /* The loop has been unrolled so remove it. */
   nir_cf_node_remove(&loop->cf_node);

   /* Delete the original loop header and body */
   nir_cf_delete(&lp_header);
   nir_cf_delete(&lp_body);

   _mesa_hash_table_destroy(remap_table, NULL);
}

/**
 * Unroll loops where we only have a single terminator but the exact trip
 * count is unknown. For example:
 *
 *    for (int i = 0; i < imin(x, 4); i++)
 *       ...
 */
static void
complex_unroll_single_terminator(nir_loop *loop)
{
   assert(list_length(&loop->info->loop_terminator_list) == 1);
   assert(loop->info->limiting_terminator);
   assert(nir_is_trivial_loop_if(loop->info->limiting_terminator->nif,
                                 loop->info->limiting_terminator->break_block));

   nir_loop_terminator *terminator = loop->info->limiting_terminator;

   loop_prepare_for_unroll(loop);

   /* Pluck out the loop header */
   nir_cf_list lp_header;
   nir_cf_extract(&lp_header, nir_before_block(nir_loop_first_block(loop)),
                  nir_before_cf_node(&terminator->nif->cf_node));

   struct hash_table *remap_table =
      _mesa_hash_table_create(NULL, _mesa_hash_pointer,
                              _mesa_key_pointer_equal);

   /* We need to clone the loop one extra time in order to clone the lcssa
    * vars for the last iteration (they are inside the following ifs break
    * branch). We leave other passes to clean up this redundant if.
    */
   unsigned num_times_to_clone = loop->info->max_trip_count + 1;

   nir_cf_list lp_body;
   UNUSED nir_cf_node *unroll_loc =
      complex_unroll_loop_body(loop, terminator, &lp_header, &lp_body,
                               remap_table, num_times_to_clone);

   assert(unroll_loc->type == nir_cf_node_if);

   /* We need to clone the lcssa vars in order to insert them on both sides
    * of the if in the last iteration/if-statement. Otherwise the optimisation
    * passes will have trouble optimising the unrolled if ladder.
    */
   nir_cursor cursor =
      get_complex_unroll_insert_location(unroll_loc,
                                         terminator->continue_from_then);

   nir_if *if_stmt = nir_cf_node_as_if(unroll_loc);
   nir_cursor start_cursor;
   nir_cursor end_cursor;
   if (terminator->continue_from_then) {
      start_cursor = nir_before_block(nir_if_first_else_block(if_stmt));
      end_cursor = nir_after_block(nir_if_last_else_block(if_stmt));
   } else {
      start_cursor = nir_before_block(nir_if_first_then_block(if_stmt));
      end_cursor = nir_after_block(nir_if_last_then_block(if_stmt));
   }

   nir_cf_list lcssa_list;
   nir_cf_extract(&lcssa_list, start_cursor, end_cursor);

   /* Insert the cloned vars in the last continue branch */
   nir_cf_list_clone_and_reinsert(&lcssa_list, loop->cf_node.parent,
                                  cursor, remap_table);

   start_cursor = terminator->continue_from_then ?
      nir_before_block(nir_if_first_else_block(if_stmt)) :
      nir_before_block(nir_if_first_then_block(if_stmt));

   /* Reinsert the cloned vars back where they came from */
   nir_cf_reinsert(&lcssa_list, start_cursor);

   /* Delete the original loop header and body */
   nir_cf_delete(&lp_header);
   nir_cf_delete(&lp_body);

   /* The original loop has been replaced so remove it. */
   nir_cf_node_remove(&loop->cf_node);

   _mesa_hash_table_destroy(remap_table, NULL);
}

/* Unrolls the classic wrapper loops e.g
 *
 *    do {
 *        // ...
 *    } while (false)
 */
static bool
wrapper_unroll(nir_loop *loop)
{
   if (!list_is_empty(&loop->info->loop_terminator_list)) {

      /* Unrolling a loop with a large number of exits can result in a
       * large inrease in register pressure. For now we just skip
       * unrolling if we have more than 3 exits (not including the break
       * at the end of the loop).
       *
       * TODO: Most loops that fit this pattern are simply switch
       * statements that are converted to a loop to take advantage of
       * exiting jump instruction handling. In this case we could make
       * use of a binary seach pattern like we do in
       * nir_lower_indirect_derefs(), this should allow us to unroll the
       * loops in an optimal way and should also avoid some of the
       * register pressure that comes from simply nesting the
       * terminators one after the other.
       */
      if (list_length(&loop->info->loop_terminator_list) > 3)
         return false;

      loop_prepare_for_unroll(loop);

      nir_cursor loop_end = nir_after_block(nir_loop_last_block(loop));
      list_for_each_entry(nir_loop_terminator, terminator,
                          &loop->info->loop_terminator_list,
                          loop_terminator_link) {

         /* Remove break from the terminator */
         nir_instr *break_instr =
            nir_block_last_instr(terminator->break_block);
         nir_instr_remove(break_instr);

         /* Pluck out the loop body. */
         nir_cf_list loop_body;
         nir_cf_extract(&loop_body,
                        nir_after_cf_node(&terminator->nif->cf_node),
                        loop_end);

         /* Reinsert loop body into continue from block */
         nir_cf_reinsert(&loop_body,
                         nir_after_block(terminator->continue_from_block));

         loop_end = terminator->continue_from_then ?
           nir_after_block(nir_if_last_then_block(terminator->nif)) :
           nir_after_block(nir_if_last_else_block(terminator->nif));
      }
   } else {
      loop_prepare_for_unroll(loop);
   }

   /* Pluck out the loop body. */
   nir_cf_list loop_body;
   nir_cf_extract(&loop_body, nir_before_block(nir_loop_first_block(loop)),
                  nir_after_block(nir_loop_last_block(loop)));

   /* Reinsert loop body after the loop */
   nir_cf_reinsert(&loop_body, nir_after_cf_node(&loop->cf_node));

   /* The loop has been unrolled so remove it. */
   nir_cf_node_remove(&loop->cf_node);

   return true;
}

static bool
is_access_out_of_bounds(nir_loop_terminator *term, nir_deref_instr *deref,
                        unsigned trip_count)
{
   for (nir_deref_instr *d = deref; d; d = nir_deref_instr_parent(d)) {
      if (d->deref_type != nir_deref_type_array)
         continue;

      nir_alu_instr *alu = nir_instr_as_alu(term->conditional_instr);
      nir_src src = term->induction_rhs ? alu->src[1].src : alu->src[0].src;
      if (!nir_srcs_equal(d->arr.index, src))
         continue;

      nir_deref_instr *parent = nir_deref_instr_parent(d);
      assert(glsl_type_is_array(parent->type) ||
             glsl_type_is_matrix(parent->type) ||
             glsl_type_is_vector(parent->type));

      /* We have already unrolled the loop and the new one will be imbedded in
       * the innermost continue branch. So unless the array is greater than
       * the trip count any iteration over the loop will be an out of bounds
       * access of the array.
       */
      unsigned length = glsl_type_is_vector(parent->type) ?
                        glsl_get_vector_elements(parent->type) :
                        glsl_get_length(parent->type);
      return length <= trip_count;
   }

   return false;
}

/* If we know an array access is going to be out of bounds remove or replace
 * the access with an undef. This can later result in the entire loop being
 * removed by nir_opt_dead_cf().
 */
static void
remove_out_of_bounds_induction_use(nir_shader *shader, nir_loop *loop,
                                   nir_loop_terminator *term,
                                   nir_cf_list *lp_header,
                                   nir_cf_list *lp_body,
                                   unsigned trip_count)
{
   if (!loop->info->guessed_trip_count)
      return;

   /* Temporarily recreate the original loop so we can alter it */
   nir_cf_reinsert(lp_header, nir_after_block(nir_loop_last_block(loop)));
   nir_cf_reinsert(lp_body, nir_after_block(nir_loop_last_block(loop)));

   nir_builder b;
   nir_builder_init(&b, nir_cf_node_get_function(&loop->cf_node));

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         /* Check for arrays variably-indexed by a loop induction variable.
          * If this access is out of bounds remove the instruction or replace
          * its use with an undefined instruction.
          * If the loop is no longer useful we leave it for the appropriate
          * pass to clean it up for us.
          */
         if (intrin->intrinsic == nir_intrinsic_load_deref ||
             intrin->intrinsic == nir_intrinsic_store_deref ||
             intrin->intrinsic == nir_intrinsic_copy_deref) {

            if (is_access_out_of_bounds(term, nir_src_as_deref(intrin->src[0]),
                                        trip_count)) {
               if (intrin->intrinsic == nir_intrinsic_load_deref) {
                  nir_ssa_def *undef =
                     nir_ssa_undef(&b, intrin->dest.ssa.num_components,
                                   intrin->dest.ssa.bit_size);
                  nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                           undef);
               } else {
                  nir_instr_remove(instr);
                  continue;
               }
            }

            if (intrin->intrinsic == nir_intrinsic_copy_deref &&
                is_access_out_of_bounds(term, nir_src_as_deref(intrin->src[1]),
                                        trip_count)) {
               nir_instr_remove(instr);
            }
         }
      }
   }

   /* Now that we are done extract the loop header and body again */
   nir_cf_extract(lp_header, nir_before_block(nir_loop_first_block(loop)),
                  nir_before_cf_node(&term->nif->cf_node));
   nir_cf_extract(lp_body, nir_before_block(nir_loop_first_block(loop)),
                  nir_after_block(nir_loop_last_block(loop)));
}

/* Partially unrolls loops that don't have a known trip count.
 */
static void
partial_unroll(nir_shader *shader, nir_loop *loop, unsigned trip_count)
{
   assert(list_length(&loop->info->loop_terminator_list) == 1);

   nir_loop_terminator *terminator =
      list_first_entry(&loop->info->loop_terminator_list,
                        nir_loop_terminator, loop_terminator_link);

   assert(nir_is_trivial_loop_if(terminator->nif, terminator->break_block));

   loop_prepare_for_unroll(loop);

   /* Pluck out the loop header */
   nir_cf_list lp_header;
   nir_cf_extract(&lp_header, nir_before_block(nir_loop_first_block(loop)),
                  nir_before_cf_node(&terminator->nif->cf_node));

   struct hash_table *remap_table =
      _mesa_hash_table_create(NULL, _mesa_hash_pointer,
                              _mesa_key_pointer_equal);

   nir_cf_list lp_body;
   nir_cf_node *unroll_loc =
      complex_unroll_loop_body(loop, terminator, &lp_header, &lp_body,
                               remap_table, trip_count);

   /* Attempt to remove out of bounds array access */
   remove_out_of_bounds_induction_use(shader, loop, terminator, &lp_header,
                                      &lp_body, trip_count);

   nir_cursor cursor =
      get_complex_unroll_insert_location(unroll_loc,
                                         terminator->continue_from_then);

   /* Reinsert the loop in the innermost nested continue branch of the unrolled
    * loop.
    */
   nir_loop *new_loop = nir_loop_create(shader);
   nir_cf_node_insert(cursor, &new_loop->cf_node);
   new_loop->partially_unrolled = true;

   /* Clone loop header and insert into new loop */
   nir_cf_list_clone_and_reinsert(&lp_header, loop->cf_node.parent,
                                  nir_after_cf_list(&new_loop->body),
                                  remap_table);

   /* Clone loop body and insert into new loop */
   nir_cf_list_clone_and_reinsert(&lp_body, loop->cf_node.parent,
                                  nir_after_cf_list(&new_loop->body),
                                  remap_table);

   /* Insert break back into terminator */
   nir_jump_instr *brk = nir_jump_instr_create(shader, nir_jump_break);
   nir_if *nif = nir_block_get_following_if(nir_loop_first_block(new_loop));
   if (terminator->continue_from_then) {
      nir_instr_insert_after_block(nir_if_last_else_block(nif), &brk->instr);
   } else {
      nir_instr_insert_after_block(nir_if_last_then_block(nif), &brk->instr);
   }

   /* Delete the original loop header and body */
   nir_cf_delete(&lp_header);
   nir_cf_delete(&lp_body);

   /* The original loop has been replaced so remove it. */
   nir_cf_node_remove(&loop->cf_node);

   _mesa_hash_table_destroy(remap_table, NULL);
}

static bool
is_indirect_load(nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      if ((intrin->intrinsic == nir_intrinsic_load_ubo ||
           intrin->intrinsic == nir_intrinsic_load_ssbo) &&
          !nir_src_is_const(intrin->src[1])) {
         return true;
      }

      if (intrin->intrinsic == nir_intrinsic_load_global)
         return true;

      if (intrin->intrinsic == nir_intrinsic_load_deref ||
          intrin->intrinsic == nir_intrinsic_store_deref) {
         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         nir_variable_mode mem_modes = nir_var_mem_ssbo | nir_var_mem_ubo | nir_var_mem_global;
         if (!nir_deref_mode_may_be(deref, mem_modes))
            return false;
         while (deref) {
            if ((deref->deref_type == nir_deref_type_array ||
                 deref->deref_type == nir_deref_type_ptr_as_array) &&
                !nir_src_is_const(deref->arr.index)) {
               return true;
            }
            deref = nir_deref_instr_parent(deref);
         }
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      for (unsigned i = 0; i < tex->num_srcs; i++) {
         if (!nir_src_is_const(tex->src[i].src))
            return true;
      }
   }

   return false;
}

static bool
can_pipeline_loads(nir_loop *loop)
{
   if (!loop->info->exact_trip_count_known)
      return false;

   bool interesting_loads = false;

   foreach_list_typed(nir_cf_node, cf_node, node, &loop->body) {
      if (cf_node == &loop->info->limiting_terminator->nif->cf_node)
         continue;

      /* Control flow usually prevents useful scheduling */
      if (cf_node->type != nir_cf_node_block)
         return false;

      if (interesting_loads)
         continue;

      nir_block *block = nir_cf_node_as_block(cf_node);
      nir_foreach_instr(instr, block) {
         if (is_indirect_load(instr)) {
            interesting_loads = true;
            break;
         }
      }
   }

   return interesting_loads;
}

/*
 * Returns true if we should unroll the loop, otherwise false.
 */
static bool
check_unrolling_restrictions(nir_shader *shader, nir_loop *loop)
{
   if (loop->control == nir_loop_control_unroll)
      return true;

   if (loop->control == nir_loop_control_dont_unroll)
      return false;

   nir_loop_info *li = loop->info;
   unsigned max_iter = shader->options->max_unroll_iterations;
   /* Unroll much more aggressively if it can hide load latency. */
   if (shader->options->max_unroll_iterations_aggressive && can_pipeline_loads(loop))
      max_iter = shader->options->max_unroll_iterations_aggressive;
   /* Tune differently if the loop has double ops and soft fp64 is in use */
   else if (shader->options->max_unroll_iterations_fp64 && loop->info->has_soft_fp64)
      max_iter = shader->options->max_unroll_iterations_fp64;
   unsigned trip_count =
      li->max_trip_count ? li->max_trip_count : li->guessed_trip_count;

   if (li->force_unroll && !li->guessed_trip_count && trip_count <= max_iter)
      return true;

   unsigned cost_limit = max_iter * LOOP_UNROLL_LIMIT;
   unsigned cost = li->instr_cost * trip_count;

   if (cost <= cost_limit && trip_count <= max_iter)
      return true;

   return false;
}

static bool
process_loops(nir_shader *sh, nir_cf_node *cf_node, bool *has_nested_loop_out,
              bool *unrolled_this_block);

static bool
process_loops_in_block(nir_shader *sh, struct exec_list *block,
                       bool *has_nested_loop_out)
{
   /* We try to unroll as many loops in one pass as possible.
    * E.g. we can safely unroll both loops in this block:
    *
    *    if (...) {
    *       loop {...}
    *    }
    *
    *    if (...) {
    *       loop {...}
    *    }
    *
    * Unrolling one loop doesn't affect the other one.
    *
    * On the other hand for block with:
    *
    *    loop {...}
    *    ...
    *    loop {...}
    *
    * It is unsafe to unroll both loops in one pass without taking
    * complicating precautions, since the structure of the block would
    * change after unrolling the first loop. So in such a case we leave
    * the second loop for the next iteration of unrolling to handle.
    */

   bool progress = false;
   bool unrolled_this_block = false;

   foreach_list_typed(nir_cf_node, nested_node, node, block) {
      if (process_loops(sh, nested_node,
                        has_nested_loop_out, &unrolled_this_block)) {
         progress = true;

         /* If current node is unrolled we could not safely continue
          * our iteration since we don't know the next node
          * and it's hard to guarantee that we won't end up unrolling
          * inner loop of the currently unrolled one, if such exists.
          */
         if (unrolled_this_block) {
            break;
         }
      }
   }

   return progress;
}

static bool
process_loops(nir_shader *sh, nir_cf_node *cf_node, bool *has_nested_loop_out,
              bool *unrolled_this_block)
{
   bool progress = false;
   bool has_nested_loop = false;
   nir_loop *loop;

   switch (cf_node->type) {
   case nir_cf_node_block:
      return progress;
   case nir_cf_node_if: {
      nir_if *if_stmt = nir_cf_node_as_if(cf_node);
      progress |= process_loops_in_block(sh, &if_stmt->then_list,
                                         has_nested_loop_out);
      progress |= process_loops_in_block(sh, &if_stmt->else_list,
                                         has_nested_loop_out);
      return progress;
   }
   case nir_cf_node_loop: {
      loop = nir_cf_node_as_loop(cf_node);
      progress |= process_loops_in_block(sh, &loop->body, &has_nested_loop);

      break;
   }
   default:
      unreachable("unknown cf node type");
   }

   const bool unrolled_child_block = progress;

   /* Don't attempt to unroll a second inner loop in this pass, wait until the
    * next pass as we have altered the cf.
    */
   if (!progress && loop->control != nir_loop_control_dont_unroll) {

      /* Remove the conditional break statements associated with all terminators
       * that are associated with a fixed iteration count, except for the one
       * associated with the limiting terminator--that one needs to stay, since
       * it terminates the loop.
       */
      if (loop->info->limiting_terminator) {
         list_for_each_entry_safe(nir_loop_terminator, t,
                                  &loop->info->loop_terminator_list,
                                  loop_terminator_link) {
            if (t->exact_trip_count_unknown)
               continue;

            if (t != loop->info->limiting_terminator) {

               /* Only delete the if-statement if the continue block is empty.
                * We trust that nir_opt_if() does its job well enough to
                * remove all instructions from the continue block when possible.
                */
               nir_block *first_continue_from_blk = t->continue_from_then ?
                  nir_if_first_then_block(t->nif) :
                  nir_if_first_else_block(t->nif);

               if (!(nir_cf_node_is_last(&first_continue_from_blk->cf_node) &&
                     exec_list_is_empty(&first_continue_from_blk->instr_list)))
                  continue;

               /* Now delete the if */
               nir_cf_node_remove(&t->nif->cf_node);

               /* Also remove it from the terminator list */
               list_del(&t->loop_terminator_link);

               progress = true;
            }
         }
      }

      /* Check for the classic
       *
       *    do {
       *        // ...
       *    } while (false)
       *
       * that is used to wrap multi-line macros. GLSL IR also wraps switch
       * statements in a loop like this.
       */
      if (loop->info->limiting_terminator == NULL &&
          !loop->info->complex_loop) {

         nir_block *last_loop_blk = nir_loop_last_block(loop);
         if (nir_block_ends_in_break(last_loop_blk)) {
            progress = wrapper_unroll(loop);
            goto exit;
         }

         /* If we were able to guess the loop iteration based on array access
          * then do a partial unroll.
          */
         unsigned num_lt = list_length(&loop->info->loop_terminator_list);
         if (!has_nested_loop && num_lt == 1 && !loop->partially_unrolled &&
             loop->info->guessed_trip_count &&
             check_unrolling_restrictions(sh, loop)) {
            partial_unroll(sh, loop, loop->info->guessed_trip_count);
            progress = true;
         }
      }

      /* Intentionally don't consider exact_trip_count_known here.  When
       * max_trip_count is non-zero, it is the upper bound on the number of
       * times the loop will iterate, but the loop may iterate less.  For
       * example, the following loop will iterate 0 or 1 time:
       *
       *    for (i = 0; i < min(x, 1); i++) { ... }
       *
       * Trivial single-interation loops (e.g., do { ... } while (false)) and
       * trivial zero-iteration loops (e.g., while (false) { ... }) will have
       * already been handled.
       *
       * If the loop is known to execute at most once and meets the other
       * unrolling criteria, unroll it even if it has nested loops.
       *
       * It is unlikely that such loops exist in real shaders. GraphicsFuzz is
       * known to generate spurious loops that iterate exactly once.  It is
       * plausible that it could eventually start generating loops like the
       * example above, so it seems logical to defend against it now.
       */
      if (!loop->info->limiting_terminator ||
          (loop->info->max_trip_count != 1 && has_nested_loop))
         goto exit;

      if (!check_unrolling_restrictions(sh, loop))
         goto exit;

      if (loop->info->exact_trip_count_known) {
         simple_unroll(loop);
         progress = true;
      } else {
         /* Attempt to unroll loops with two terminators. */
         unsigned num_lt = list_length(&loop->info->loop_terminator_list);
         if (num_lt == 2 &&
             !loop->info->limiting_terminator->exact_trip_count_unknown) {
            bool limiting_term_second = true;
            nir_loop_terminator *terminator =
               list_first_entry(&loop->info->loop_terminator_list,
                                nir_loop_terminator, loop_terminator_link);


            if (terminator->nif == loop->info->limiting_terminator->nif) {
               limiting_term_second = false;
               terminator =
                  list_last_entry(&loop->info->loop_terminator_list,
                                  nir_loop_terminator, loop_terminator_link);
            }

            /* If the first terminator has a trip count of zero and is the
             * limiting terminator just do a simple unroll as the second
             * terminator can never be reached.
             */
            if (loop->info->max_trip_count == 0 && !limiting_term_second) {
               simple_unroll(loop);
            } else {
               complex_unroll(loop, terminator, limiting_term_second);
            }
            progress = true;
         }

         if (num_lt == 1) {
            assert(loop->info->limiting_terminator->exact_trip_count_unknown);
            complex_unroll_single_terminator(loop);
            progress = true;
         }
      }
   }

exit:
   *has_nested_loop_out = true;
   if (progress && !unrolled_child_block)
      *unrolled_this_block = true;

   return progress;
}

static bool
nir_opt_loop_unroll_impl(nir_function_impl *impl,
                         nir_variable_mode indirect_mask,
                         bool force_unroll_sampler_indirect)
{
   bool progress = false;
   nir_metadata_require(impl, nir_metadata_loop_analysis, indirect_mask,
                        (int) force_unroll_sampler_indirect);
   nir_metadata_require(impl, nir_metadata_block_index);

   bool has_nested_loop = false;
   progress |= process_loops_in_block(impl->function->shader, &impl->body,
                                      &has_nested_loop);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_none);
      nir_lower_regs_to_ssa_impl(impl);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

/**
 * indirect_mask specifies which type of indirectly accessed variables
 * should force loop unrolling.
 */
bool
nir_opt_loop_unroll(nir_shader *shader)
{
   bool progress = false;

   bool force_unroll_sampler_indirect = shader->options->force_indirect_unrolling_sampler;
   nir_variable_mode indirect_mask = shader->options->force_indirect_unrolling;
   nir_foreach_function(function, shader) {
      if (function->impl) {
         progress |= nir_opt_loop_unroll_impl(function->impl, indirect_mask,
                                              force_unroll_sampler_indirect);
      }
   }
   return progress;
}
