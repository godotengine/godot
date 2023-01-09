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
#include "nir/nir_builder.h"
#include "nir_constant_expressions.h"
#include "nir_control_flow.h"
#include "nir_loop_analyze.h"

static nir_ssa_def *clone_alu_and_replace_src_defs(nir_builder *b,
                                                   const nir_alu_instr *alu,
                                                   nir_ssa_def **src_defs);

/**
 * Gets the single block that jumps back to the loop header. Already assumes
 * there is exactly one such block.
 */
static nir_block*
find_continue_block(nir_loop *loop)
{
   nir_block *header_block = nir_loop_first_block(loop);
   nir_block *prev_block =
      nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));

   assert(header_block->predecessors->entries == 2);

   set_foreach(header_block->predecessors, pred_entry) {
      if (pred_entry->key != prev_block)
         return (nir_block*)pred_entry->key;
   }

   unreachable("Continue block not found!");
}

/**
 * Does a phi have one constant value from outside a loop and one from inside?
 */
static bool
phi_has_constant_from_outside_and_one_from_inside_loop(nir_phi_instr *phi,
                                                       const nir_block *entry_block,
                                                       bool *entry_val,
                                                       bool *continue_val)
{
   /* We already know we have exactly one continue */
   assert(exec_list_length(&phi->srcs) == 2);

   *entry_val = false;
   *continue_val = false;

    nir_foreach_phi_src(src, phi) {
       if (!nir_src_is_const(src->src))
         return false;

       if (src->pred != entry_block) {
          *continue_val = nir_src_as_bool(src->src);
       } else {
          *entry_val = nir_src_as_bool(src->src);
       }
    }

    return true;
}

/**
 * This optimization detects if statements at the tops of loops where the
 * condition is a phi node of two constants and moves half of the if to above
 * the loop and the other half of the if to the end of the loop.  A simple for
 * loop "for (int i = 0; i < 4; i++)", when run through the SPIR-V front-end,
 * ends up looking something like this:
 *
 * vec1 32 ssa_0 = load_const (0x00000000)
 * vec1 32 ssa_1 = load_const (0xffffffff)
 * loop {
 *    block block_1:
 *    vec1 32 ssa_2 = phi block_0: ssa_0, block_7: ssa_5
 *    vec1 32 ssa_3 = phi block_0: ssa_0, block_7: ssa_1
 *    if ssa_3 {
 *       block block_2:
 *       vec1 32 ssa_4 = load_const (0x00000001)
 *       vec1 32 ssa_5 = iadd ssa_2, ssa_4
 *    } else {
 *       block block_3:
 *    }
 *    block block_4:
 *    vec1 32 ssa_6 = load_const (0x00000004)
 *    vec1 32 ssa_7 = ilt ssa_5, ssa_6
 *    if ssa_7 {
 *       block block_5:
 *    } else {
 *       block block_6:
 *       break
 *    }
 *    block block_7:
 * }
 *
 * This turns it into something like this:
 *
 * // Stuff from block 1
 * // Stuff from block 3
 * loop {
 *    block block_1:
 *    vec1 32 ssa_2 = phi block_0: ssa_0, block_7: ssa_5
 *    vec1 32 ssa_6 = load_const (0x00000004)
 *    vec1 32 ssa_7 = ilt ssa_2, ssa_6
 *    if ssa_7 {
 *       block block_5:
 *    } else {
 *       block block_6:
 *       break
 *    }
 *    block block_7:
 *    // Stuff from block 1
 *    // Stuff from block 2
 *    vec1 32 ssa_4 = load_const (0x00000001)
 *    vec1 32 ssa_5 = iadd ssa_2, ssa_4
 * }
 */
static bool
opt_peel_loop_initial_if(nir_loop *loop)
{
   nir_block *header_block = nir_loop_first_block(loop);
   nir_block *const prev_block =
      nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));

   /* It would be insane if this were not true */
   assert(_mesa_set_search(header_block->predecessors, prev_block));

   /* The loop must have exactly one continue block which could be a block
    * ending in a continue instruction or the "natural" continue from the
    * last block in the loop back to the top.
    */
   if (header_block->predecessors->entries != 2)
      return false;

   nir_cf_node *if_node = nir_cf_node_next(&header_block->cf_node);
   if (!if_node || if_node->type != nir_cf_node_if)
      return false;

   nir_if *nif = nir_cf_node_as_if(if_node);
   if (!nif->condition.is_ssa)
      return false;

   nir_ssa_def *cond = nif->condition.ssa;
   if (cond->parent_instr->type != nir_instr_type_phi)
      return false;

   nir_phi_instr *cond_phi = nir_instr_as_phi(cond->parent_instr);
   if (cond->parent_instr->block != header_block)
      return false;

   bool entry_val = false, continue_val = false;
   if (!phi_has_constant_from_outside_and_one_from_inside_loop(cond_phi,
                                                               prev_block,
                                                               &entry_val,
                                                               &continue_val))
      return false;

   /* If they both execute or both don't execute, this is a job for
    * nir_dead_cf, not this pass.
    */
   if ((entry_val && continue_val) || (!entry_val && !continue_val))
      return false;

   struct exec_list *continue_list, *entry_list;
   if (continue_val) {
      continue_list = &nif->then_list;
      entry_list = &nif->else_list;
   } else {
      continue_list = &nif->else_list;
      entry_list = &nif->then_list;
   }

   /* We want to be moving the contents of entry_list to above the loop so it
    * can't contain any break or continue instructions.
    */
   foreach_list_typed(nir_cf_node, cf_node, node, entry_list) {
      nir_foreach_block_in_cf_node(block, cf_node) {
         nir_instr *last_instr = nir_block_last_instr(block);
         if (last_instr && last_instr->type == nir_instr_type_jump)
            return false;
      }
   }

   /* We're about to re-arrange a bunch of blocks so make sure that we don't
    * have deref uses which cross block boundaries.  We don't want a deref
    * accidentally ending up in a phi.
    */
   nir_rematerialize_derefs_in_use_blocks_impl(
      nir_cf_node_get_function(&loop->cf_node));

   /* Before we do anything, convert the loop to LCSSA.  We're about to
    * replace a bunch of SSA defs with registers and this will prevent any of
    * it from leaking outside the loop.
    */
   nir_convert_loop_to_lcssa(loop);

   nir_block *after_if_block =
      nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));

   /* Get rid of phis in the header block since we will be duplicating it */
   nir_lower_phis_to_regs_block(header_block);
   /* Get rid of phis after the if since dominance will change */
   nir_lower_phis_to_regs_block(after_if_block);

   /* Get rid of SSA defs in the pieces we're about to move around */
   nir_lower_ssa_defs_to_regs_block(header_block);
   nir_foreach_block_in_cf_node(block, &nif->cf_node)
      nir_lower_ssa_defs_to_regs_block(block);

   nir_cf_list header, tmp;
   nir_cf_extract(&header, nir_before_block(header_block),
                           nir_after_block(header_block));

   nir_cf_list_clone(&tmp, &header, &loop->cf_node, NULL);
   nir_cf_reinsert(&tmp, nir_before_cf_node(&loop->cf_node));
   nir_cf_extract(&tmp, nir_before_cf_list(entry_list),
                        nir_after_cf_list(entry_list));
   nir_cf_reinsert(&tmp, nir_before_cf_node(&loop->cf_node));

   nir_cf_reinsert(&header,
                   nir_after_block_before_jump(find_continue_block(loop)));

   bool continue_list_jumps =
      nir_block_ends_in_jump(exec_node_data(nir_block,
                                            exec_list_get_tail(continue_list),
                                            cf_node.node));

   nir_cf_extract(&tmp, nir_before_cf_list(continue_list),
                        nir_after_cf_list(continue_list));

   /* Get continue block again as the previous reinsert might have removed the
    * block.  Also, if both the continue list and the continue block ends in
    * jump instructions, removes the jump from the latter, as it will not be
    * executed if we insert the continue list before it. */

   nir_block *continue_block = find_continue_block(loop);

   if (continue_list_jumps) {
      nir_instr *last_instr = nir_block_last_instr(continue_block);
      if (last_instr && last_instr->type == nir_instr_type_jump)
         nir_instr_remove(last_instr);
   }

   nir_cf_reinsert(&tmp,
                   nir_after_block_before_jump(continue_block));

   nir_cf_node_remove(&nif->cf_node);

   return true;
}

static bool
alu_instr_is_comparison(const nir_alu_instr *alu)
{
   switch (alu->op) {
   case nir_op_flt32:
   case nir_op_fge32:
   case nir_op_feq32:
   case nir_op_fneu32:
   case nir_op_ilt32:
   case nir_op_ult32:
   case nir_op_ige32:
   case nir_op_uge32:
   case nir_op_ieq32:
   case nir_op_ine32:
      return true;
   default:
      return nir_alu_instr_is_comparison(alu);
   }
}

static bool
alu_instr_is_type_conversion(const nir_alu_instr *alu)
{
   return nir_op_infos[alu->op].num_inputs == 1 &&
          nir_op_infos[alu->op].output_type != nir_op_infos[alu->op].input_types[0];
}

static bool
is_trivial_bcsel(const nir_instr *instr, bool allow_non_phi_src)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *const bcsel = nir_instr_as_alu(instr);
   if (!nir_op_is_selection(bcsel->op))
      return false;

   for (unsigned i = 0; i < 3; i++) {
      if (!nir_alu_src_is_trivial_ssa(bcsel, i) ||
          bcsel->src[i].src.ssa->parent_instr->block != instr->block)
         return false;

      if (bcsel->src[i].src.ssa->parent_instr->type != nir_instr_type_phi) {
         /* opt_split_alu_of_phi() is able to peel that src from the loop */
         if (i == 0 || !allow_non_phi_src)
            return false;
         allow_non_phi_src = false;
      }
   }

   nir_foreach_phi_src(src, nir_instr_as_phi(bcsel->src[0].src.ssa->parent_instr)) {
      if (!nir_src_is_const(src->src))
         return false;
   }

   return true;
}

/**
 * Splits ALU instructions that have a source that is a phi node
 *
 * ALU instructions in the header block of a loop that meet the following
 * criteria can be split.
 *
 * - The loop has no continue instructions other than the "natural" continue
 *   at the bottom of the loop.
 *
 * - At least one source of the instruction is a phi node from the header block.
 *
 * - Any non-phi sources of the ALU instruction come from a block that
 *   dominates the block before the loop.  The most common failure mode for
 *   this check is sources that are generated in the loop header block.
 *
 * - The phi node selects a constant or undef from the block before the loop or
 *   the only ALU user is a trivial bcsel that gets removed by peeling the ALU
 *
 * The split process splits the original ALU instruction into two, one at the
 * bottom of the loop and one at the block before the loop. The instruction
 * before the loop computes the value on the first iteration, and the
 * instruction at the bottom computes the value on the second, third, and so
 * on. A new phi node is added to the header block that selects either the
 * instruction before the loop or the one at the end, and uses of the original
 * instruction are replaced by this phi.
 *
 * The splitting transforms a loop like:
 *
 *    vec1 32 ssa_8 = load_const (0x00000001)
 *    vec1 32 ssa_10 = load_const (0x00000000)
 *    // succs: block_1
 *    loop {
 *            block block_1:
 *            // preds: block_0 block_4
 *            vec1 32 ssa_11 = phi block_0: ssa_10, block_4: ssa_15
 *            vec1 32 ssa_12 = phi block_0: ssa_1, block_4: ssa_15
 *            vec1 32 ssa_13 = phi block_0: ssa_10, block_4: ssa_16
 *            vec1 32 ssa_14 = iadd ssa_11, ssa_8
 *            vec1 32 ssa_15 = b32csel ssa_13, ssa_14, ssa_12
 *            ...
 *            // succs: block_1
 *    }
 *
 * into:
 *
 *    vec1 32 ssa_8 = load_const (0x00000001)
 *    vec1 32 ssa_10 = load_const (0x00000000)
 *    vec1 32 ssa_22 = iadd ssa_10, ssa_8
 *    // succs: block_1
 *    loop {
 *            block block_1:
 *            // preds: block_0 block_4
 *            vec1 32 ssa_11 = phi block_0: ssa_10, block_4: ssa_15
 *            vec1 32 ssa_12 = phi block_0: ssa_1, block_4: ssa_15
 *            vec1 32 ssa_13 = phi block_0: ssa_10, block_4: ssa_16
 *            vec1 32 ssa_21 = phi block_0: ssa_22, block_4: ssa_20
 *            vec1 32 ssa_15 = b32csel ssa_13, ssa_21, ssa_12
 *            ...
 *            vec1 32 ssa_20 = iadd ssa_15, ssa_8
 *            // succs: block_1
 *    }
 */
static bool
opt_split_alu_of_phi(nir_builder *b, nir_loop *loop)
{
   bool progress = false;
   nir_block *header_block = nir_loop_first_block(loop);
   nir_block *const prev_block =
      nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));

   /* It would be insane if this were not true */
   assert(_mesa_set_search(header_block->predecessors, prev_block));

   /* The loop must have exactly one continue block which could be a block
    * ending in a continue instruction or the "natural" continue from the
    * last block in the loop back to the top.
    */
   if (header_block->predecessors->entries != 2)
      return false;

   nir_block *continue_block = find_continue_block(loop);
   if (continue_block == header_block)
      return false;

   nir_foreach_instr_safe(instr, header_block) {
      if (instr->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *const alu = nir_instr_as_alu(instr);

      /* nir_op_vec{2,3,4} and nir_op_mov are excluded because they can easily
       * lead to infinite optimization loops. Splitting comparisons can lead
       * to loop unrolling not recognizing loop termintators, and type
       * conversions also lead to regressions.
       */
      if (nir_op_is_vec(alu->op) ||
          alu_instr_is_comparison(alu) ||
          alu_instr_is_type_conversion(alu))
         continue;

      bool has_phi_src_from_prev_block = false;
      bool all_non_phi_exist_in_prev_block = true;
      bool is_prev_result_undef = true;
      bool is_prev_result_const = true;
      nir_ssa_def *prev_srcs[8];     // FINISHME: Array size?
      nir_ssa_def *continue_srcs[8]; // FINISHME: Array size?

      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         nir_instr *const src_instr = alu->src[i].src.ssa->parent_instr;

         /* If the source is a phi in the loop header block, then the
          * prev_srcs and continue_srcs will come from the different sources
          * of the phi.
          */
         if (src_instr->type == nir_instr_type_phi &&
             src_instr->block == header_block) {
            nir_phi_instr *const phi = nir_instr_as_phi(src_instr);

            /* Only strictly need to NULL out the pointers when the assertions
             * (below) are compiled in.  Debugging a NULL pointer deref in the
             * wild is easier than debugging a random pointer deref, so set
             * NULL unconditionally just to be safe.
             */
            prev_srcs[i] = NULL;
            continue_srcs[i] = NULL;

            nir_foreach_phi_src(src_of_phi, phi) {
               if (src_of_phi->pred == prev_block) {
                  if (src_of_phi->src.ssa->parent_instr->type !=
                      nir_instr_type_ssa_undef) {
                     is_prev_result_undef = false;
                  }

                  if (src_of_phi->src.ssa->parent_instr->type !=
                      nir_instr_type_load_const) {
                     is_prev_result_const = false;
                  }

                  prev_srcs[i] = src_of_phi->src.ssa;
                  has_phi_src_from_prev_block = true;
               } else
                  continue_srcs[i] = src_of_phi->src.ssa;
            }

            assert(prev_srcs[i] != NULL);
            assert(continue_srcs[i] != NULL);
         } else {
            /* If the source is not a phi (or a phi in a block other than the
             * loop header), then the value must exist in prev_block.
             */
            if (!nir_block_dominates(src_instr->block, prev_block)) {
               all_non_phi_exist_in_prev_block = false;
               break;
            }

            prev_srcs[i] = alu->src[i].src.ssa;
            continue_srcs[i] = alu->src[i].src.ssa;
         }
      }

      if (!has_phi_src_from_prev_block || !all_non_phi_exist_in_prev_block)
         continue;

      if (!is_prev_result_undef && !is_prev_result_const) {
         /* check if the only user is a trivial bcsel */
         if (!list_is_empty(&alu->dest.dest.ssa.if_uses) ||
             !list_is_singular(&alu->dest.dest.ssa.uses))
            continue;

         nir_src *use = list_first_entry(&alu->dest.dest.ssa.uses, nir_src, use_link);
         if (!is_trivial_bcsel(use->parent_instr, true))
            continue;
      }

      /* Split ALU of Phi */
      b->cursor = nir_after_block(prev_block);
      nir_ssa_def *prev_value = clone_alu_and_replace_src_defs(b, alu, prev_srcs);

      /* Make a copy of the original ALU instruction.  Replace the sources
       * of the new instruction that read a phi with an undef source from
       * prev_block with the non-undef source of that phi.
       *
       * Insert the new instruction at the end of the continue block.
       */
      b->cursor = nir_after_block_before_jump(continue_block);

      nir_ssa_def *const alu_copy =
         clone_alu_and_replace_src_defs(b, alu, continue_srcs);

      /* Make a new phi node that selects a value from prev_block and the
       * result of the new instruction from continue_block.
       */
      nir_phi_instr *const phi = nir_phi_instr_create(b->shader);
      nir_phi_instr_add_src(phi, prev_block, nir_src_for_ssa(prev_value));
      nir_phi_instr_add_src(phi, continue_block, nir_src_for_ssa(alu_copy));

      nir_ssa_dest_init(&phi->instr, &phi->dest,
                        alu_copy->num_components, alu_copy->bit_size, NULL);

      b->cursor = nir_after_phis(header_block);
      nir_builder_instr_insert(b, &phi->instr);

      /* Modify all readers of the original ALU instruction to read the
       * result of the phi.
       */
      nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa,
                               &phi->dest.ssa);

      /* Since the original ALU instruction no longer has any readers, just
       * remove it.
       */
      nir_instr_remove_v(&alu->instr);
      nir_instr_free(&alu->instr);

      progress = true;
   }

   return progress;
}

/**
 * Simplify a bcsel whose sources are all phi nodes from the loop header block
 *
 * bcsel instructions in a loop that meet the following criteria can be
 * converted to phi nodes:
 *
 * - The loop has no continue instructions other than the "natural" continue
 *   at the bottom of the loop.
 *
 * - All of the sources of the bcsel are phi nodes in the header block of the
 *   loop.
 *
 * - The phi node representing the condition of the bcsel instruction chooses
 *   only constant values.
 *
 * The contant value from the condition will select one of the other sources
 * when entered from outside the loop and the remaining source when entered
 * from the continue block.  Since each of these sources is also a phi node in
 * the header block, the value of the phi node can be "evaluated."  These
 * evaluated phi nodes provide the sources for a new phi node.  All users of
 * the bcsel result are updated to use the phi node result.
 *
 * The replacement transforms loops like:
 *
 *    vec1 32 ssa_7 = undefined
 *    vec1 32 ssa_8 = load_const (0x00000001)
 *    vec1 32 ssa_9 = load_const (0x000000c8)
 *    vec1 32 ssa_10 = load_const (0x00000000)
 *    // succs: block_1
 *    loop {
 *            block block_1:
 *            // preds: block_0 block_4
 *            vec1 32 ssa_11 = phi block_0: ssa_1, block_4: ssa_14
 *            vec1 32 ssa_12 = phi block_0: ssa_10, block_4: ssa_15
 *            vec1 32 ssa_13 = phi block_0: ssa_7, block_4: ssa_25
 *            vec1 32 ssa_14 = b32csel ssa_12, ssa_13, ssa_11
 *            vec1 32 ssa_16 = ige32 ssa_14, ssa_9
 *            ...
 *            vec1 32 ssa_15 = load_const (0xffffffff)
 *            ...
 *            vec1 32 ssa_25 = iadd ssa_14, ssa_8
 *            // succs: block_1
 *    }
 *
 * into:
 *
 *    vec1 32 ssa_7 = undefined
 *    vec1 32 ssa_8 = load_const (0x00000001)
 *    vec1 32 ssa_9 = load_const (0x000000c8)
 *    vec1 32 ssa_10 = load_const (0x00000000)
 *    // succs: block_1
 *    loop {
 *            block block_1:
 *            // preds: block_0 block_4
 *            vec1 32 ssa_11 = phi block_0: ssa_1, block_4: ssa_14
 *            vec1 32 ssa_12 = phi block_0: ssa_10, block_4: ssa_15
 *            vec1 32 ssa_13 = phi block_0: ssa_7, block_4: ssa_25
 *            vec1 32 sss_26 = phi block_0: ssa_1, block_4: ssa_25
 *            vec1 32 ssa_16 = ige32 ssa_26, ssa_9
 *            ...
 *            vec1 32 ssa_15 = load_const (0xffffffff)
 *            ...
 *            vec1 32 ssa_25 = iadd ssa_26, ssa_8
 *            // succs: block_1
 *    }
 *
 * \note
 * It may be possible modify this function to not require a phi node as the
 * source of the bcsel that is selected when entering from outside the loop.
 * The only restriction is that the source must be geneated outside the loop
 * (since it will become the source of a phi node in the header block of the
 * loop).
 */
static bool
opt_simplify_bcsel_of_phi(nir_builder *b, nir_loop *loop)
{
   bool progress = false;
   nir_block *header_block = nir_loop_first_block(loop);
   nir_block *const prev_block =
      nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));

   /* It would be insane if this were not true */
   assert(_mesa_set_search(header_block->predecessors, prev_block));

   /* The loop must have exactly one continue block which could be a block
    * ending in a continue instruction or the "natural" continue from the
    * last block in the loop back to the top.
    */
   if (header_block->predecessors->entries != 2)
      return false;

   /* We can move any bcsel that can guaranteed to execut on every iteration
    * of a loop.  For now this is accomplished by only taking bcsels from the
    * header_block.  In the future, this could be expanced to include any
    * bcsel that must come before any break.
    *
    * For more details, see
    * https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/170#note_110305
    */
   nir_foreach_instr_safe(instr, header_block) {
      if (!is_trivial_bcsel(instr, false))
         continue;

      nir_alu_instr *const bcsel = nir_instr_as_alu(instr);
      nir_phi_instr *const cond_phi =
         nir_instr_as_phi(bcsel->src[0].src.ssa->parent_instr);

      bool entry_val = false, continue_val = false;
      if (!phi_has_constant_from_outside_and_one_from_inside_loop(cond_phi,
                                                                  prev_block,
                                                                  &entry_val,
                                                                  &continue_val))
         continue;

      /* If they both execute or both don't execute, this is a job for
       * nir_dead_cf, not this pass.
       */
      if ((entry_val && continue_val) || (!entry_val && !continue_val))
         continue;

      const unsigned entry_src = entry_val ? 1 : 2;
      const unsigned continue_src = entry_val ? 2 : 1;

      /* Create a new phi node that selects the value for prev_block from
       * the bcsel source that is selected by entry_val and the value for
       * continue_block from the other bcsel source.  Both sources have
       * already been verified to be phi nodes.
       */
      nir_block *continue_block = find_continue_block(loop);
      nir_phi_instr *const phi = nir_phi_instr_create(b->shader);
      nir_phi_instr_add_src(phi, prev_block,
                            nir_phi_get_src_from_block(nir_instr_as_phi(bcsel->src[entry_src].src.ssa->parent_instr),
                                                       prev_block)->src);

      nir_phi_instr_add_src(phi, continue_block,
                            nir_phi_get_src_from_block(nir_instr_as_phi(bcsel->src[continue_src].src.ssa->parent_instr),
                                    continue_block)->src);

      nir_ssa_dest_init(&phi->instr,
                        &phi->dest,
                        nir_dest_num_components(bcsel->dest.dest),
                        nir_dest_bit_size(bcsel->dest.dest),
                        NULL);

      b->cursor = nir_after_phis(header_block);
      nir_builder_instr_insert(b, &phi->instr);

      /* Modify all readers of the bcsel instruction to read the result of
       * the phi.
       */
      nir_ssa_def_rewrite_uses(&bcsel->dest.dest.ssa,
                               &phi->dest.ssa);

      /* Since the original bcsel instruction no longer has any readers,
       * just remove it.
       */
      nir_instr_remove_v(&bcsel->instr);
      nir_instr_free(&bcsel->instr);

      progress = true;
   }

   return progress;
}

static bool
is_block_empty(nir_block *block)
{
   return nir_cf_node_is_last(&block->cf_node) &&
          exec_list_is_empty(&block->instr_list);
}

static bool
nir_block_ends_in_continue(nir_block *block)
{
   if (exec_list_is_empty(&block->instr_list))
      return false;

   nir_instr *instr = nir_block_last_instr(block);
   return instr->type == nir_instr_type_jump &&
      nir_instr_as_jump(instr)->type == nir_jump_continue;
}

/**
 * This optimization turns:
 *
 *     loop {
 *        ...
 *        if (cond) {
 *           do_work_1();
 *           continue;
 *        } else {
 *        }
 *        do_work_2();
 *     }
 *
 * into:
 *
 *     loop {
 *        ...
 *        if (cond) {
 *           do_work_1();
 *           continue;
 *        } else {
 *           do_work_2();
 *        }
 *     }
 *
 * The continue should then be removed by nir_opt_trivial_continues() and the
 * loop can potentially be unrolled.
 *
 * Note: Unless the function param aggressive_last_continue==true do_work_2()
 * is only ever blocks and nested loops. We avoid nesting other if-statments
 * in the branch as this can result in increased register pressure, and in
 * the i965 driver it causes a large amount of spilling in shader-db.
 * For RADV however nesting these if-statements allows further continues to be
 * remove and provides a significant FPS boost in Doom, which is why we have
 * opted for this special bool to enable more aggresive optimisations.
 * TODO: The GCM pass solves most of the spilling regressions in i965, if it
 * is ever enabled we should consider removing the aggressive_last_continue
 * param.
 */
static bool
opt_if_loop_last_continue(nir_loop *loop, bool aggressive_last_continue)
{
   nir_if *nif = NULL;
   bool then_ends_in_continue = false;
   bool else_ends_in_continue = false;

   /* Scan the control flow of the loop from the last to the first node
    * looking for an if-statement we can optimise.
    */
   nir_block *last_block = nir_loop_last_block(loop);
   nir_cf_node *if_node = nir_cf_node_prev(&last_block->cf_node);
   while (if_node) {
      if (if_node->type == nir_cf_node_if) {
         nif = nir_cf_node_as_if(if_node);
         nir_block *then_block = nir_if_last_then_block(nif);
         nir_block *else_block = nir_if_last_else_block(nif);

         then_ends_in_continue = nir_block_ends_in_continue(then_block);
         else_ends_in_continue = nir_block_ends_in_continue(else_block);

         /* If both branches end in a jump do nothing, this should be handled
          * by nir_opt_dead_cf().
          */
         if ((then_ends_in_continue || nir_block_ends_in_break(then_block)) &&
             (else_ends_in_continue || nir_block_ends_in_break(else_block)))
            return false;

         /* If continue found stop scanning and attempt optimisation, or
          */
         if (then_ends_in_continue || else_ends_in_continue ||
             !aggressive_last_continue)
            break;
      }

      if_node = nir_cf_node_prev(if_node);
   }

   /* If we didn't find an if to optimise return */
   if (!nif || (!then_ends_in_continue && !else_ends_in_continue))
      return false;

   /* If there is nothing after the if-statement we bail */
   if (&nif->cf_node == nir_cf_node_prev(&last_block->cf_node) &&
       exec_list_is_empty(&last_block->instr_list))
      return false;

   /* If there are single-source phis in the last block,
    * get rid of them first
    */
   nir_opt_remove_phis_block(last_block);

   /* Move the last block of the loop inside the last if-statement */
   nir_cf_list tmp;
   nir_cf_extract(&tmp, nir_after_cf_node(if_node),
                        nir_after_block(last_block));
   if (then_ends_in_continue)
      nir_cf_reinsert(&tmp, nir_after_cf_list(&nif->else_list));
   else
      nir_cf_reinsert(&tmp, nir_after_cf_list(&nif->then_list));

   /* In order to avoid running nir_lower_regs_to_ssa_impl() every time an if
    * opt makes progress we leave nir_opt_trivial_continues() to remove the
    * continue now that the end of the loop has been simplified.
    */

   return true;
}

/* Walk all the phis in the block immediately following the if statement and
 * swap the blocks.
 */
static void
rewrite_phi_predecessor_blocks(nir_if *nif,
                               nir_block *old_then_block,
                               nir_block *old_else_block,
                               nir_block *new_then_block,
                               nir_block *new_else_block)
{
   nir_block *after_if_block =
      nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));

   nir_foreach_instr(instr, after_if_block) {
      if (instr->type != nir_instr_type_phi)
         continue;

      nir_phi_instr *phi = nir_instr_as_phi(instr);

      nir_foreach_phi_src(src, phi) {
         if (src->pred == old_then_block) {
            src->pred = new_then_block;
         } else if (src->pred == old_else_block) {
            src->pred = new_else_block;
         }
      }
   }
}

/**
 * This optimization turns:
 *
 *     if (cond) {
 *     } else {
 *         do_work();
 *     }
 *
 * into:
 *
 *     if (!cond) {
 *         do_work();
 *     } else {
 *     }
 */
static bool
opt_if_simplification(nir_builder *b, nir_if *nif)
{
   /* Only simplify if the then block is empty and the else block is not. */
   if (!is_block_empty(nir_if_first_then_block(nif)) ||
       is_block_empty(nir_if_first_else_block(nif)))
      return false;

   /* Make sure the condition is a comparison operation. */
   nir_instr *src_instr = nif->condition.ssa->parent_instr;
   if (src_instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu_instr = nir_instr_as_alu(src_instr);
   if (!nir_alu_instr_is_comparison(alu_instr))
      return false;

   /* Insert the inverted instruction and rewrite the condition. */
   b->cursor = nir_after_instr(&alu_instr->instr);

   nir_ssa_def *new_condition =
      nir_inot(b, &alu_instr->dest.dest.ssa);

   nir_if_rewrite_condition(nif, nir_src_for_ssa(new_condition));

   /* Grab pointers to the last then/else blocks for fixing up the phis. */
   nir_block *then_block = nir_if_last_then_block(nif);
   nir_block *else_block = nir_if_last_else_block(nif);

   if (nir_block_ends_in_jump(else_block)) {
      /* Even though this if statement has a jump on one side, we may still have
       * phis afterwards.  Single-source phis can be produced by loop unrolling
       * or dead control-flow passes and are perfectly legal.  Run a quick phi
       * removal on the block after the if to clean up any such phis.
       */
      nir_block *const next_block =
         nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
      nir_opt_remove_phis_block(next_block);
   }

   rewrite_phi_predecessor_blocks(nif, then_block, else_block, else_block,
                                  then_block);

   /* Finally, move the else block to the then block. */
   nir_cf_list tmp;
   nir_cf_extract(&tmp, nir_before_cf_list(&nif->else_list),
                        nir_after_cf_list(&nif->else_list));
   nir_cf_reinsert(&tmp, nir_before_cf_list(&nif->then_list));

   return true;
}

/* Find phi statements after an if that choose between true and false, and
 * replace them with the if statement's condition (or an inot of it).
 */
static bool
opt_if_phi_is_condition(nir_builder *b, nir_if *nif)
{
   /* Grab pointers to the last then/else blocks for looking in the phis. */
   nir_block *then_block = nir_if_last_then_block(nif);
   ASSERTED nir_block *else_block = nir_if_last_else_block(nif);
   nir_ssa_def *cond = nif->condition.ssa;
   bool progress = false;

   nir_block *after_if_block = nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
   nir_foreach_instr_safe(instr, after_if_block) {
      if (instr->type != nir_instr_type_phi)
         break;

      nir_phi_instr *phi = nir_instr_as_phi(instr);
      if (phi->dest.ssa.bit_size != cond->bit_size ||
          phi->dest.ssa.num_components != 1)
         continue;

      enum opt_bool {
         T, F, UNKNOWN
      } then_val = UNKNOWN, else_val = UNKNOWN;

      nir_foreach_phi_src(src, phi) {
         assert(src->pred == then_block || src->pred == else_block);
         enum opt_bool *pred_val = src->pred == then_block ? &then_val : &else_val;

         nir_ssa_scalar val = nir_ssa_scalar_resolved(src->src.ssa, 0);
         if (!nir_ssa_scalar_is_const(val))
            break;

         if (nir_ssa_scalar_as_int(val) == -1)
            *pred_val = T;
         else if (nir_ssa_scalar_as_uint(val) == 0)
            *pred_val = F;
         else
            break;
      }
      if (then_val == T && else_val == F) {
         nir_ssa_def_rewrite_uses(&phi->dest.ssa, cond);
         progress = true;
      } else if (then_val == F && else_val == T) {
         b->cursor = nir_before_cf_node(&nif->cf_node);
         nir_ssa_def_rewrite_uses(&phi->dest.ssa, nir_inot(b, cond));
         progress = true;
      }
   }

   return progress;
}

/**
 * This optimization tries to merge two break statements into a single break.
 * For this purpose, it checks if both branch legs end in a break or
 * if one branch leg ends in a break, and the other one does so after the
 * branch.
 *
 * This optimization turns
 *
 *     loop {
 *        ...
 *        if (cond) {
 *           do_work_1();
 *           break;
 *        } else {
 *           do_work_2();
 *           break;
 *        }
 *     }
 *
 * into:
 *
 *     loop {
 *        ...
 *        if (cond) {
 *           do_work_1();
 *        } else {
 *           do_work_2();
 *        }
 *        break;
 *     }
 *
 * but also situations like
 *
 *     loop {
 *        ...
 *        if (cond1) {
 *           if (cond2) {
 *              do_work_1();
 *              break;
 *           } else {
 *              do_work_2();
 *           }
 *           do_work_3();
 *           break;
 *        } else {
 *           ...
 *        }
 *     }
 *
 *  into:
 *
 *     loop {
 *        ...
 *        if (cond1) {
 *           if (cond2) {
 *              do_work_1();
 *           } else {
 *              do_work_2();
 *              do_work_3();
 *           }
 *           break;
 *        } else {
 *           ...
 *        }
 *     }
 */
static bool
opt_merge_breaks(nir_if *nif)
{
   nir_block *last_then = nir_if_last_then_block(nif);
   nir_block *last_else = nir_if_last_else_block(nif);
   bool then_break = nir_block_ends_in_break(last_then);
   bool else_break = nir_block_ends_in_break(last_else);

   /* If both branch legs end in a break, merge the break after the branch */
   if (then_break && else_break) {
      nir_block *after_if = nir_cf_node_cf_tree_next(&nif->cf_node);
      /* Make sure that the successor is empty.
       * If not we let nir_opt_dead_cf() clean it up first.
       */
      if (!is_block_empty(after_if))
         return false;

      nir_lower_phis_to_regs_block(last_then->successors[0]);
      nir_instr_remove_v(nir_block_last_instr(last_then));
      nir_instr *jump = nir_block_last_instr(last_else);
      nir_instr_remove_v(jump);
      nir_instr_insert(nir_after_block(after_if), jump);
      return true;
    }

   /* Single break: If there's a break after the branch and the non-breaking
    * side of the if falls through to it, then hoist that code after up into
    * the if and leave just a single break there.
    */
   if (then_break || else_break) {

      /* At least one branch leg must fall-through */
      if (nir_block_ends_in_jump(last_then) && nir_block_ends_in_jump(last_else))
         return false;

      /* Check if there is a single break after the IF */
      nir_cf_node *first = nir_cf_node_next(&nif->cf_node);
      nir_cf_node *last = first;
      while (!nir_cf_node_is_last(last)) {
         if (contains_other_jump (last, NULL))
            return false;
         last = nir_cf_node_next(last);
      }

      assert(last->type == nir_cf_node_block);
      if (!nir_block_ends_in_break(nir_cf_node_as_block(last)))
         return false;

      /* Hoist the code from after the IF into the falling-through branch leg */
      nir_opt_remove_phis_block(nir_cf_node_as_block(first));
      nir_block *break_block = then_break ? last_then : last_else;
      nir_lower_phis_to_regs_block(break_block->successors[0]);

      nir_cf_list tmp;
      nir_cf_extract(&tmp, nir_before_cf_node(first),
                           nir_after_block_before_jump(nir_cf_node_as_block(last)));
      if (then_break)
         nir_cf_reinsert(&tmp, nir_after_block(last_else));
      else
         nir_cf_reinsert(&tmp, nir_after_block(last_then));

      nir_instr_remove_v(nir_block_last_instr(break_block));
      return true;
   }

   return false;
}

/**
 * This optimization simplifies potential loop terminators which then allows
 * other passes such as opt_if_simplification() and loop unrolling to progress
 * further:
 *
 *     if (cond) {
 *        ... then block instructions ...
 *     } else {
 *         ...
 *        break;
 *     }
 *
 * into:
 *
 *     if (cond) {
 *     } else {
 *         ...
 *        break;
 *     }
 *     ... then block instructions ...
 */
static bool
opt_if_loop_terminator(nir_if *nif)
{
   nir_block *break_blk = NULL;
   nir_block *continue_from_blk = NULL;
   bool continue_from_then = true;

   nir_block *last_then = nir_if_last_then_block(nif);
   nir_block *last_else = nir_if_last_else_block(nif);

   if (nir_block_ends_in_break(last_then)) {
      break_blk = last_then;
      continue_from_blk = last_else;
      continue_from_then = false;
   } else if (nir_block_ends_in_break(last_else)) {
      break_blk = last_else;
      continue_from_blk = last_then;
   }

   /* Continue if the if-statement contained no jumps at all */
   if (!break_blk)
      return false;

   /* If the continue from block is empty then return as there is nothing to
    * move.
    */
   nir_block *first_continue_from_blk = continue_from_then ?
      nir_if_first_then_block(nif) :
      nir_if_first_else_block(nif);
   if (is_block_empty(first_continue_from_blk))
      return false;

   if (nir_block_ends_in_jump(continue_from_blk))
      return false;

   /* Even though this if statement has a jump on one side, we may still have
    * phis afterwards.  Single-source phis can be produced by loop unrolling
    * or dead control-flow passes and are perfectly legal.  Run a quick phi
    * removal on the block after the if to clean up any such phis.
    */
   nir_opt_remove_phis_block(nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node)));

   /* Finally, move the continue from branch after the if-statement. */
   nir_cf_list tmp;
   nir_cf_extract(&tmp, nir_before_block(first_continue_from_blk),
                        nir_after_block(continue_from_blk));
   nir_cf_reinsert(&tmp, nir_after_cf_node(&nif->cf_node));

   return true;
}

static bool
evaluate_if_condition(nir_if *nif, nir_cursor cursor, bool *value)
{
   nir_block *use_block = nir_cursor_current_block(cursor);
   if (nir_block_dominates(nir_if_first_then_block(nif), use_block)) {
      *value = true;
      return true;
   } else if (nir_block_dominates(nir_if_first_else_block(nif), use_block)) {
      *value = false;
      return true;
   } else {
      return false;
   }
}

static nir_ssa_def *
clone_alu_and_replace_src_defs(nir_builder *b, const nir_alu_instr *alu,
                               nir_ssa_def **src_defs)
{
   nir_alu_instr *nalu = nir_alu_instr_create(b->shader, alu->op);
   nalu->exact = alu->exact;

   nir_ssa_dest_init(&nalu->instr, &nalu->dest.dest,
                     alu->dest.dest.ssa.num_components,
                     alu->dest.dest.ssa.bit_size, NULL);

   nalu->dest.saturate = alu->dest.saturate;
   nalu->dest.write_mask = alu->dest.write_mask;

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      assert(alu->src[i].src.is_ssa);
      nalu->src[i].src = nir_src_for_ssa(src_defs[i]);
      nalu->src[i].negate = alu->src[i].negate;
      nalu->src[i].abs = alu->src[i].abs;
      memcpy(nalu->src[i].swizzle, alu->src[i].swizzle,
             sizeof(nalu->src[i].swizzle));
   }

   nir_builder_instr_insert(b, &nalu->instr);

   return &nalu->dest.dest.ssa;;
}

/*
 * This propagates if condition evaluation down the chain of some alu
 * instructions. For example by checking the use of some of the following alu
 * instruction we can eventually replace ssa_107 with NIR_TRUE.
 *
 *   loop {
 *      block block_1:
 *      vec1 32 ssa_85 = load_const (0x00000002)
 *      vec1 32 ssa_86 = ieq ssa_48, ssa_85
 *      vec1 32 ssa_87 = load_const (0x00000001)
 *      vec1 32 ssa_88 = ieq ssa_48, ssa_87
 *      vec1 32 ssa_89 = ior ssa_86, ssa_88
 *      vec1 32 ssa_90 = ieq ssa_48, ssa_0
 *      vec1 32 ssa_91 = ior ssa_89, ssa_90
 *      if ssa_86 {
 *         block block_2:
 *             ...
 *            break
 *      } else {
 *            block block_3:
 *      }
 *      block block_4:
 *      if ssa_88 {
 *            block block_5:
 *             ...
 *            break
 *      } else {
 *            block block_6:
 *      }
 *      block block_7:
 *      if ssa_90 {
 *            block block_8:
 *             ...
 *            break
 *      } else {
 *            block block_9:
 *      }
 *      block block_10:
 *      vec1 32 ssa_107 = inot ssa_91
 *      if ssa_107 {
 *            block block_11:
 *            break
 *      } else {
 *            block block_12:
 *      }
 *   }
 */
static bool
propagate_condition_eval(nir_builder *b, nir_if *nif, nir_src *use_src,
                         nir_src *alu_use, nir_alu_instr *alu,
                         bool is_if_condition)
{
   bool bool_value;
   b->cursor = nir_before_src(alu_use, is_if_condition);
   if (!evaluate_if_condition(nif, b->cursor, &bool_value))
      return false;

   nir_ssa_def *def[NIR_MAX_VEC_COMPONENTS] = {0};
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      if (alu->src[i].src.ssa == use_src->ssa) {
         def[i] = nir_imm_bool(b, bool_value);
      } else {
         def[i] = alu->src[i].src.ssa;
      }
   }

   nir_ssa_def *nalu = clone_alu_and_replace_src_defs(b, alu, def);

   /* Rewrite use to use new alu instruction */
   nir_src new_src = nir_src_for_ssa(nalu);

   if (is_if_condition)
      nir_if_rewrite_condition(alu_use->parent_if, new_src);
   else
      nir_instr_rewrite_src(alu_use->parent_instr, alu_use, new_src);

   return true;
}

static bool
can_propagate_through_alu(nir_src *src)
{
   if (src->parent_instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(src->parent_instr);
   switch (alu->op) {
      case nir_op_ior:
      case nir_op_iand:
      case nir_op_inot:
      case nir_op_b2i32:
         return true;
      case nir_op_bcsel:
         return src == &alu->src[0].src;
      default:
         return false;
   }
}

static bool
evaluate_condition_use(nir_builder *b, nir_if *nif, nir_src *use_src,
                       bool is_if_condition)
{
   bool progress = false;

   b->cursor = nir_before_src(use_src, is_if_condition);

   bool bool_value;
   if (evaluate_if_condition(nif, b->cursor, &bool_value)) {
      /* Rewrite use to use const */
      nir_src imm_src = nir_src_for_ssa(nir_imm_bool(b, bool_value));
      if (is_if_condition)
         nir_if_rewrite_condition(use_src->parent_if, imm_src);
      else
         nir_instr_rewrite_src(use_src->parent_instr, use_src, imm_src);

      progress = true;
   }

   if (!is_if_condition && can_propagate_through_alu(use_src)) {
      nir_alu_instr *alu = nir_instr_as_alu(use_src->parent_instr);

      nir_foreach_use_safe(alu_use, &alu->dest.dest.ssa) {
         progress |= propagate_condition_eval(b, nif, use_src, alu_use, alu,
                                              false);
      }

      nir_foreach_if_use_safe(alu_use, &alu->dest.dest.ssa) {
         progress |= propagate_condition_eval(b, nif, use_src, alu_use, alu,
                                              true);
      }
   }

   return progress;
}

static bool
opt_if_evaluate_condition_use(nir_builder *b, nir_if *nif)
{
   bool progress = false;

   /* Evaluate any uses of the if condition inside the if branches */
   assert(nif->condition.is_ssa);
   nir_foreach_use_safe(use_src, nif->condition.ssa) {
      progress |= evaluate_condition_use(b, nif, use_src, false);
   }

   nir_foreach_if_use_safe(use_src, nif->condition.ssa) {
      if (use_src->parent_if != nif)
         progress |= evaluate_condition_use(b, nif, use_src, true);
   }

   return progress;
}

static bool
rewrite_comp_uses_within_if(nir_builder *b, nir_if *nif, bool invert,
                            nir_ssa_scalar scalar, nir_ssa_scalar new_scalar)
{
   bool progress = false;

   nir_block *first = invert ? nir_if_first_else_block(nif) : nir_if_first_then_block(nif);
   nir_block *last = invert ? nir_if_last_else_block(nif) : nir_if_last_then_block(nif);

   nir_ssa_def *new_ssa = NULL;
   nir_foreach_use_safe(use, scalar.def) {
      if (use->parent_instr->block->index < first->index ||
          use->parent_instr->block->index > last->index)
         continue;

      /* Only rewrite users which use only the new component. This is to avoid a
       * situation where copy propagation will undo the rewrite and we risk an infinite
       * loop.
       *
       * We could rewrite users which use a mix of the old and new components, but if
       * nir_src_components_read() is incomplete, then we risk the new component actually being
       * unused and some optimization later undoing the rewrite.
       */
      if (nir_src_components_read(use) != BITFIELD64_BIT(scalar.comp))
         continue;

      if (!new_ssa) {
         b->cursor = nir_before_cf_node(&nif->cf_node);
         new_ssa = nir_channel(b, new_scalar.def, new_scalar.comp);
         if (scalar.def->num_components > 1) {
            nir_ssa_def *vec = nir_ssa_undef(b, scalar.def->num_components, scalar.def->bit_size);
            new_ssa = nir_vector_insert_imm(b, vec, new_ssa, scalar.comp);
         }
      }

      nir_instr_rewrite_src_ssa(use->parent_instr, use, new_ssa);
      progress = true;
   }

   return progress;
}

/*
 * This optimization turns:
 *
 *     if (a == (b=readfirstlane(a)))
 *        use(a)
 *     if (c == (d=load_const))
 *        use(c)
 *
 * into:
 *
 *     if (a == (b=readfirstlane(a)))
 *        use(b)
 *     if (c == (d=load_const))
 *        use(d)
*/
static bool
opt_if_rewrite_uniform_uses(nir_builder *b, nir_if *nif, nir_ssa_scalar cond, bool accept_ine)
{
   bool progress = false;

   if (!nir_ssa_scalar_is_alu(cond))
      return false;

   nir_op op = nir_ssa_scalar_alu_op(cond);
   if (op == nir_op_iand) {
      progress |= opt_if_rewrite_uniform_uses(b, nif, nir_ssa_scalar_chase_alu_src(cond, 0), false);
      progress |= opt_if_rewrite_uniform_uses(b, nif, nir_ssa_scalar_chase_alu_src(cond, 1), false);
      return progress;
   }

   if (op != nir_op_ieq && (op != nir_op_ine || !accept_ine))
      return false;

   for (unsigned i = 0; i < 2; i++) {
      nir_ssa_scalar src_uni = nir_ssa_scalar_chase_alu_src(cond, i);
      nir_ssa_scalar src_div = nir_ssa_scalar_chase_alu_src(cond, !i);

      if (src_uni.def->parent_instr->type == nir_instr_type_load_const && src_div.def != src_uni.def)
         return rewrite_comp_uses_within_if(b, nif, op == nir_op_ine, src_div, src_uni);

      if (src_uni.def->parent_instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(src_uni.def->parent_instr);
      if (intrin->intrinsic != nir_intrinsic_read_first_invocation &&
          (intrin->intrinsic != nir_intrinsic_reduce || nir_intrinsic_cluster_size(intrin)))
         continue;

      nir_ssa_scalar intrin_src = {intrin->src[0].ssa, src_uni.comp};
      nir_ssa_scalar resolved_intrin_src = nir_ssa_scalar_resolved(intrin_src.def, intrin_src.comp);

      if (resolved_intrin_src.comp != src_div.comp || resolved_intrin_src.def != src_div.def)
         continue;

      progress |= rewrite_comp_uses_within_if(b, nif, op == nir_op_ine, resolved_intrin_src, src_uni);
      if (intrin_src.comp != resolved_intrin_src.comp || intrin_src.def != resolved_intrin_src.def)
         progress |= rewrite_comp_uses_within_if(b, nif, op == nir_op_ine, intrin_src, src_uni);

      return progress;
   }

   return false;
}

static void
simple_merge_if(nir_if *dest_if, nir_if *src_if, bool dest_if_then,
                bool src_if_then)
{
   /* Now merge the if branch */
   nir_block *dest_blk = dest_if_then ? nir_if_last_then_block(dest_if)
                                      : nir_if_last_else_block(dest_if);

   struct exec_list *list = src_if_then ? &src_if->then_list
                                        : &src_if->else_list;

   nir_cf_list if_cf_list;
   nir_cf_extract(&if_cf_list, nir_before_cf_list(list),
                  nir_after_cf_list(list));
   nir_cf_reinsert(&if_cf_list, nir_after_block(dest_blk));
}

static bool
opt_if_merge(nir_if *nif)
{
   bool progress = false;

   nir_block *next_blk = nir_cf_node_cf_tree_next(&nif->cf_node);
   if (!next_blk || !nif->condition.is_ssa)
      return false;

   nir_if *next_if = nir_block_get_following_if(next_blk);
   if (!next_if || !next_if->condition.is_ssa)
      return false;

   /* Here we merge two consecutive ifs that have the same condition e.g:
    *
    *   if ssa_12 {
    *      ...
    *   } else {
    *      ...
    *   }
    *   if ssa_12 {
    *      ...
    *   } else {
    *      ...
    *   }
    *
    * Note: This only merges if-statements when the block between them is
    * empty. The reason we don't try to merge ifs that just have phis between
    * them is because this can result in increased register pressure. For
    * example when merging if ladders created by indirect indexing.
    */
   if (nif->condition.ssa == next_if->condition.ssa &&
       exec_list_is_empty(&next_blk->instr_list)) {

      /* This optimization isn't made to work in this case and
       * opt_if_evaluate_condition_use will optimize it later.
       */
      if (nir_block_ends_in_jump(nir_if_last_then_block(nif)) ||
          nir_block_ends_in_jump(nir_if_last_else_block(nif)))
         return false;

      simple_merge_if(nif, next_if, true, true);
      simple_merge_if(nif, next_if, false, false);

      nir_block *new_then_block = nir_if_last_then_block(nif);
      nir_block *new_else_block = nir_if_last_else_block(nif);

      nir_block *old_then_block = nir_if_last_then_block(next_if);
      nir_block *old_else_block = nir_if_last_else_block(next_if);

      /* Rewrite the predecessor block for any phis following the second
       * if-statement.
       */
      rewrite_phi_predecessor_blocks(next_if, old_then_block,
                                     old_else_block,
                                     new_then_block,
                                     new_else_block);

      /* Move phis after merged if to avoid them being deleted when we remove
       * the merged if-statement.
       */
      nir_block *after_next_if_block =
         nir_cf_node_as_block(nir_cf_node_next(&next_if->cf_node));

      nir_foreach_instr_safe(instr, after_next_if_block) {
         if (instr->type != nir_instr_type_phi)
            break;

         exec_node_remove(&instr->node);
         exec_list_push_tail(&next_blk->instr_list, &instr->node);
         instr->block = next_blk;
      }

      nir_cf_node_remove(&next_if->cf_node);

      progress = true;
   }

   return progress;
}

static bool
opt_if_cf_list(nir_builder *b, struct exec_list *cf_list,
               nir_opt_if_options options)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, cf_node, node, cf_list) {
      switch (cf_node->type) {
      case nir_cf_node_block:
         break;

      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(cf_node);
         progress |= opt_if_cf_list(b, &nif->then_list,
                                    options);
         progress |= opt_if_cf_list(b, &nif->else_list,
                                    options);
         progress |= opt_if_loop_terminator(nif);
         progress |= opt_if_merge(nif);
         progress |= opt_if_simplification(b, nif);
         if (options & nir_opt_if_optimize_phi_true_false)
            progress |= opt_if_phi_is_condition(b, nif);
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(cf_node);
         progress |= opt_if_cf_list(b, &loop->body,
                                    options);
         progress |= opt_simplify_bcsel_of_phi(b, loop);
         progress |= opt_if_loop_last_continue(loop,
                                               options & nir_opt_if_aggressive_last_continue);
         break;
      }

      case nir_cf_node_function:
         unreachable("Invalid cf type");
      }
   }

   return progress;
}

/**
 * Optimizations which can create registers are done after other optimizations
 * which require SSA.
 */
static bool
opt_if_regs_cf_list(struct exec_list *cf_list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, cf_node, node, cf_list) {
      switch (cf_node->type) {
      case nir_cf_node_block:
         break;

      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(cf_node);
         progress |= opt_if_regs_cf_list(&nif->then_list);
         progress |= opt_if_regs_cf_list(&nif->else_list);
         if (opt_merge_breaks(nif)) {
            /* This optimization might move blocks
             * from after the NIF into the NIF */
            progress = true;
            opt_if_regs_cf_list(&nif->then_list);
            opt_if_regs_cf_list(&nif->else_list);
         }
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(cf_node);
         progress |= opt_if_regs_cf_list(&loop->body);
         progress |= opt_peel_loop_initial_if(loop);
         break;
      }

      case nir_cf_node_function:
         unreachable("Invalid cf type");
      }
   }

   return progress;
}

/**
 * These optimisations depend on nir_metadata_block_index and therefore must
 * not do anything to cause the metadata to become invalid.
 */
static bool
opt_if_safe_cf_list(nir_builder *b, struct exec_list *cf_list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, cf_node, node, cf_list) {
      switch (cf_node->type) {
      case nir_cf_node_block:
         break;

      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(cf_node);
         progress |= opt_if_safe_cf_list(b, &nif->then_list);
         progress |= opt_if_safe_cf_list(b, &nif->else_list);
         progress |= opt_if_evaluate_condition_use(b, nif);
         nir_ssa_scalar cond = nir_ssa_scalar_resolved(nif->condition.ssa, 0);
         progress |= opt_if_rewrite_uniform_uses(b, nif, cond, true);
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(cf_node);
         progress |= opt_if_safe_cf_list(b, &loop->body);
         progress |= opt_split_alu_of_phi(b, loop);
         break;
      }

      case nir_cf_node_function:
         unreachable("Invalid cf type");
      }
   }

   return progress;
}

bool
nir_opt_if(nir_shader *shader, nir_opt_if_options options)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl == NULL)
         continue;

      nir_builder b;
      nir_builder_init(&b, function->impl);

      nir_metadata_require(function->impl, nir_metadata_block_index |
                           nir_metadata_dominance);
      progress = opt_if_safe_cf_list(&b, &function->impl->body);
      nir_metadata_preserve(function->impl, nir_metadata_block_index |
                            nir_metadata_dominance);

      bool preserve = true;

      if (opt_if_cf_list(&b, &function->impl->body, options)) {
         preserve = false;
         progress = true;
      }

      if (opt_if_regs_cf_list(&function->impl->body)) {
         preserve = false;
         progress = true;

         /* If that made progress, we're no longer really in SSA form.  We
          * need to convert registers back into SSA defs and clean up SSA defs
          * that don't dominate their uses.
          */
         nir_lower_regs_to_ssa_impl(function->impl);
      }

      if (preserve) {
         nir_metadata_preserve(function->impl, nir_metadata_none);
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}
