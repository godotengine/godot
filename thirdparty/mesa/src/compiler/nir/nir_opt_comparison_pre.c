/*
 * Copyright © 2018 Intel Corporation
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

#include "nir_instr_set.h"
#include "nir_search_helpers.h"
#include "nir_builder.h"
#include "util/u_vector.h"

/* Partial redundancy elimination of compares
 *
 * Seaches for comparisons of the form 'a cmp b' that dominate arithmetic
 * instructions like 'b - a'.  The comparison is replaced by the arithmetic
 * instruction, and the result is compared with zero.  For example,
 *
 *       vec1 32 ssa_111 = flt 0.37, ssa_110.w
 *       if ssa_111 {
 *               block block_1:
 *              vec1 32 ssa_112 = fadd ssa_110.w, -0.37
 *              ...
 *
 * becomes
 *
 *       vec1 32 ssa_111 = fadd ssa_110.w, -0.37
 *       vec1 32 ssa_112 = flt 0.0, ssa_111
 *       if ssa_112 {
 *               block block_1:
 *              ...
 */

struct block_queue {
   /**
    * Stack of blocks from the current location in the CFG to the entry point
    * of the function.
    *
    * This is sort of a poor man's dominator tree.
    */
   struct exec_list blocks;

   /** List of freed block_instructions structures that can be reused. */
   struct exec_list reusable_blocks;
};

struct block_instructions {
   struct exec_node node;

   /**
    * Set of comparison instructions from the block that are candidates for
    * being replaced by add instructions.
    */
   struct u_vector instructions;
};

static void
block_queue_init(struct block_queue *bq)
{
   exec_list_make_empty(&bq->blocks);
   exec_list_make_empty(&bq->reusable_blocks);
}

static void
block_queue_finish(struct block_queue *bq)
{
   struct block_instructions *n;

   while ((n = (struct block_instructions *) exec_list_pop_head(&bq->blocks)) != NULL) {
      u_vector_finish(&n->instructions);
      free(n);
   }

   while ((n = (struct block_instructions *) exec_list_pop_head(&bq->reusable_blocks)) != NULL) {
      free(n);
   }
}

static struct block_instructions *
push_block(struct block_queue *bq)
{
   struct block_instructions *bi =
      (struct block_instructions *) exec_list_pop_head(&bq->reusable_blocks);

   if (bi == NULL) {
      bi = calloc(1, sizeof(struct block_instructions));

      if (bi == NULL)
         return NULL;
   }

   if (!u_vector_init_pow2(&bi->instructions, 8, sizeof(nir_alu_instr *))) {
      free(bi);
      return NULL;
   }

   exec_list_push_tail(&bq->blocks, &bi->node);

   return bi;
}

static void
pop_block(struct block_queue *bq, struct block_instructions *bi)
{
   u_vector_finish(&bi->instructions);
   exec_node_remove(&bi->node);
   exec_list_push_head(&bq->reusable_blocks, &bi->node);
}

static void
add_instruction_for_block(struct block_instructions *bi,
                          nir_alu_instr *alu)
{
   nir_alu_instr **data =
      u_vector_add(&bi->instructions);

   *data = alu;
}

/**
 * Determine if the ALU instruction is used by an if-condition or used by a
 * logic-not that is used by an if-condition.
 */
static bool
is_compatible_condition(const nir_alu_instr *instr)
{
   if (is_used_by_if(instr))
      return true;

   nir_foreach_use(src, &instr->dest.dest.ssa) {
      const nir_instr *const user_instr = src->parent_instr;

      if (user_instr->type != nir_instr_type_alu)
         continue;

      const nir_alu_instr *const user_alu = nir_instr_as_alu(user_instr);

      if (user_alu->op != nir_op_inot)
         continue;

      if (is_used_by_if(user_alu))
         return true;
   }

   return false;
}

static void
rewrite_compare_instruction(nir_builder *bld, nir_alu_instr *orig_cmp,
                            nir_alu_instr *orig_add, bool zero_on_left)
{
   bld->cursor = nir_before_instr(&orig_cmp->instr);

   /* This is somewhat tricky.  The compare instruction may be something like
    * (fcmp, a, b) while the add instruction is something like (fadd, fneg(a),
    * b).  This is problematic because the SSA value for the fneg(a) may not
    * exist yet at the compare instruction.
    *
    * We fabricate the operands of the new add.  This is done using
    * information provided by zero_on_left.  If zero_on_left is true, we know
    * the resulting compare instruction is (fcmp, 0.0, (fadd, x, y)).  If the
    * original compare instruction was (fcmp, a, b), x = b and y = -a.  If
    * zero_on_left is false, the resulting compare instruction is (fcmp,
    * (fadd, x, y), 0.0) and x = a and y = -b.
    */
   nir_ssa_def *const a = nir_ssa_for_alu_src(bld, orig_cmp, 0);
   nir_ssa_def *const b = nir_ssa_for_alu_src(bld, orig_cmp, 1);

   nir_ssa_def *const fadd = zero_on_left
      ? nir_fadd(bld, b, nir_fneg(bld, a))
      : nir_fadd(bld, a, nir_fneg(bld, b));

   nir_ssa_def *const zero =
      nir_imm_floatN_t(bld, 0.0, orig_add->dest.dest.ssa.bit_size);

   nir_ssa_def *const cmp = zero_on_left
      ? nir_build_alu(bld, orig_cmp->op, zero, fadd, NULL, NULL)
      : nir_build_alu(bld, orig_cmp->op, fadd, zero, NULL, NULL);

   /* Generating extra moves of the results is the easy way to make sure the
    * writemasks match the original instructions.  Later optimization passes
    * will clean these up.  This is similar to nir_replace_instr (in
    * nir_search.c).
    */
   nir_alu_instr *mov_add = nir_alu_instr_create(bld->shader, nir_op_mov);
   mov_add->dest.write_mask = orig_add->dest.write_mask;
   nir_ssa_dest_init(&mov_add->instr, &mov_add->dest.dest,
                     orig_add->dest.dest.ssa.num_components,
                     orig_add->dest.dest.ssa.bit_size, NULL);
   mov_add->src[0].src = nir_src_for_ssa(fadd);

   nir_builder_instr_insert(bld, &mov_add->instr);

   nir_alu_instr *mov_cmp = nir_alu_instr_create(bld->shader, nir_op_mov);
   mov_cmp->dest.write_mask = orig_cmp->dest.write_mask;
   nir_ssa_dest_init(&mov_cmp->instr, &mov_cmp->dest.dest,
                     orig_cmp->dest.dest.ssa.num_components,
                     orig_cmp->dest.dest.ssa.bit_size, NULL);
   mov_cmp->src[0].src = nir_src_for_ssa(cmp);

   nir_builder_instr_insert(bld, &mov_cmp->instr);

   nir_ssa_def_rewrite_uses(&orig_cmp->dest.dest.ssa,
                            &mov_cmp->dest.dest.ssa);
   nir_ssa_def_rewrite_uses(&orig_add->dest.dest.ssa,
                            &mov_add->dest.dest.ssa);

   /* We know these have no more uses because we just rewrote them all, so we
    * can remove them.
    */
   nir_instr_remove(&orig_cmp->instr);
   nir_instr_remove(&orig_add->instr);
}

static bool
comparison_pre_block(nir_block *block, struct block_queue *bq, nir_builder *bld)
{
   bool progress = false;

   struct block_instructions *bi = push_block(bq);
   if (bi == NULL)
      return false;

   /* Starting with the current block, examine each instruction.  If the
    * instruction is a comparison that matches the '±a cmp ±b' pattern, add it
    * to the block_instructions::instructions set.  If the instruction is an
    * add instruction, walk up the block queue looking at the stored
    * instructions.  If a matching comparison is found, move the addition and
    * replace the comparison with a different comparison based on the result
    * of the addition.  All of the blocks in the queue are guaranteed to be
    * dominators of the current block.
    *
    * After processing the current block, recurse into the blocks dominated by
    * the current block.
    */
   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *const alu = nir_instr_as_alu(instr);

      if (alu->dest.dest.ssa.num_components != 1)
         continue;

      if (alu->dest.saturate)
         continue;

      static const uint8_t swizzle[NIR_MAX_VEC_COMPONENTS] = {0};

      switch (alu->op) {
      case nir_op_fadd: {
         /* If the instruction is fadd, check it against comparison
          * instructions that dominate it.
          */
         struct block_instructions *b =
            (struct block_instructions *) exec_list_get_head_raw(&bq->blocks);

         while (b->node.next != NULL) {
            nir_alu_instr **a;
            bool rewrote_compare = false;

            u_vector_foreach(a, &b->instructions) {
               nir_alu_instr *const cmp = *a;

               if (cmp == NULL)
                  continue;

               /* The operands of both instructions are, with some liberty,
                * commutative.  Check all four permutations.  The third and
                * fourth permutations are negations of the first two.
                */
               if ((nir_alu_srcs_equal(cmp, alu, 0, 0) &&
                    nir_alu_srcs_negative_equal(cmp, alu, 1, 1)) ||
                   (nir_alu_srcs_equal(cmp, alu, 0, 1) &&
                    nir_alu_srcs_negative_equal(cmp, alu, 1, 0))) {
                  /* These are the cases where (A cmp B) matches either (A +
                   * -B) or (-B + A)
                   *
                   *    A cmp B <=> A + -B cmp 0
                   */
                  rewrite_compare_instruction(bld, cmp, alu, false);

                  *a = NULL;
                  rewrote_compare = true;
                  break;
               } else if ((nir_alu_srcs_equal(cmp, alu, 1, 0) &&
                           nir_alu_srcs_negative_equal(cmp, alu, 0, 1)) ||
                          (nir_alu_srcs_equal(cmp, alu, 1, 1) &&
                           nir_alu_srcs_negative_equal(cmp, alu, 0, 0))) {
                  /* This is the case where (A cmp B) matches (B + -A) or (-A
                   * + B).
                   *
                   *    A cmp B <=> 0 cmp B + -A
                   */
                  rewrite_compare_instruction(bld, cmp, alu, true);

                  *a = NULL;
                  rewrote_compare = true;
                  break;
               }
            }

            /* Bail after a compare in the most dominating block is found.
             * This is necessary because 'alu' has been removed from the
             * instruction stream.  Should there be a matching compare in
             * another block, calling rewrite_compare_instruction again will
             * try to operate on a node that is not in the list as if it were
             * in the list.
             *
             * FINISHME: There may be opportunity for additional optimization
             * here.  I discovered this problem due to a shader in Guacamelee.
             * It may be possible to rewrite the matching compares that are
             * encountered later to reuse the result from the compare that was
             * first rewritten.  It's also possible that this is just taken
             * care of by calling the optimization pass repeatedly.
             */
            if (rewrote_compare) {
               progress = true;
               break;
            }

            b = (struct block_instructions *) b->node.next;
         }

         break;
      }

      case nir_op_flt:
      case nir_op_fge:
      case nir_op_fneu:
      case nir_op_feq:
         /* If the instruction is a comparison that is used by an if-statement
          * and neither operand is immediate value 0, add it to the set.
          */
         if (is_compatible_condition(alu) &&
             is_not_const_zero(NULL, alu, 0, 1, swizzle) &&
             is_not_const_zero(NULL, alu, 1, 1, swizzle))
            add_instruction_for_block(bi, alu);

         break;

      default:
         break;
      }
   }

   for (unsigned i = 0; i < block->num_dom_children; i++) {
      nir_block *child = block->dom_children[i];

      if (comparison_pre_block(child, bq, bld))
         progress = true;
   }

   pop_block(bq, bi);

   return progress;
}

bool
nir_opt_comparison_pre_impl(nir_function_impl *impl)
{
   struct block_queue bq;
   nir_builder bld;

   block_queue_init(&bq);
   nir_builder_init(&bld, impl);

   nir_metadata_require(impl, nir_metadata_dominance);

   const bool progress =
      comparison_pre_block(nir_start_block(impl), &bq, &bld);

   block_queue_finish(&bq);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_opt_comparison_pre(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_comparison_pre_impl(function->impl);
   }

   return progress;
}
