/*
 * Copyright © 2014 Intel Corporation
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

/*
 * Implements the algorithms for computing the dominance tree and the
 * dominance frontier from "A Simple, Fast Dominance Algorithm" by Cooper,
 * Harvey, and Kennedy.
 */

static bool
init_block(nir_block *block, nir_function_impl *impl)
{
   if (block == nir_start_block(impl))
      block->imm_dom = block;
   else
      block->imm_dom = NULL;
   block->num_dom_children = 0;

   /* See nir_block_dominates */
   block->dom_pre_index = UINT32_MAX;
   block->dom_post_index = 0;

   _mesa_set_clear(block->dom_frontier, NULL);

   return true;
}

static nir_block *
intersect(nir_block *b1, nir_block *b2)
{
   while (b1 != b2) {
      /*
       * Note, the comparisons here are the opposite of what the paper says
       * because we index blocks from beginning -> end (i.e. reverse
       * post-order) instead of post-order like they assume.
       */
      while (b1->index > b2->index)
         b1 = b1->imm_dom;
      while (b2->index > b1->index)
         b2 = b2->imm_dom;
   }

   return b1;
}

static bool
calc_dominance(nir_block *block)
{
   nir_block *new_idom = NULL;
   set_foreach(block->predecessors, entry) {
      nir_block *pred = (nir_block *) entry->key;

      if (pred->imm_dom) {
         if (new_idom)
            new_idom = intersect(pred, new_idom);
         else
            new_idom = pred;
      }
   }

   if (block->imm_dom != new_idom) {
      block->imm_dom = new_idom;
      return true;
   }

   return false;
}

static bool
calc_dom_frontier(nir_block *block)
{
   if (block->predecessors->entries > 1) {
      set_foreach(block->predecessors, entry) {
         nir_block *runner = (nir_block *) entry->key;

         /* Skip unreachable predecessors */
         if (runner->imm_dom == NULL)
            continue;

         while (runner != block->imm_dom) {
            _mesa_set_add(runner->dom_frontier, block);
            runner = runner->imm_dom;
         }
      }
   }

   return true;
}

/*
 * Compute each node's children in the dominance tree from the immediate
 * dominator information. We do this in three stages:
 *
 * 1. Calculate the number of children each node has
 * 2. Allocate arrays, setting the number of children to 0 again
 * 3. For each node, add itself to its parent's list of children, using
 *    num_dom_children as an index - at the end of this step, num_dom_children
 *    for each node will be the same as it was at the end of step #1.
 */

static void
calc_dom_children(nir_function_impl* impl)
{
   void *mem_ctx = ralloc_parent(impl);

   nir_foreach_block_unstructured(block, impl) {
      if (block->imm_dom)
         block->imm_dom->num_dom_children++;
   }

   nir_foreach_block_unstructured(block, impl) {
      block->dom_children = ralloc_array(mem_ctx, nir_block *,
                                         block->num_dom_children);
      block->num_dom_children = 0;
   }

   nir_foreach_block_unstructured(block, impl) {
      if (block->imm_dom) {
         block->imm_dom->dom_children[block->imm_dom->num_dom_children++]
            = block;
      }
   }
}

static void
calc_dfs_indicies(nir_block *block, uint32_t *index)
{
   /* UINT32_MAX has special meaning. See nir_block_dominates. */
   assert(*index < UINT32_MAX - 2);

   block->dom_pre_index = (*index)++;

   for (unsigned i = 0; i < block->num_dom_children; i++)
      calc_dfs_indicies(block->dom_children[i], index);

   block->dom_post_index = (*index)++;
}

void
nir_calc_dominance_impl(nir_function_impl *impl)
{
   if (impl->valid_metadata & nir_metadata_dominance)
      return;

   nir_metadata_require(impl, nir_metadata_block_index);


   nir_foreach_block_unstructured(block, impl) {
      init_block(block, impl);
   }

   bool progress = true;
   while (progress) {
      progress = false;
      nir_foreach_block_unstructured(block, impl) {
         if (block != nir_start_block(impl))
            progress |= calc_dominance(block);
      }
   }

   nir_foreach_block_unstructured(block, impl) {
      calc_dom_frontier(block);
   }

   nir_block *start_block = nir_start_block(impl);
   start_block->imm_dom = NULL;

   calc_dom_children(impl);

   uint32_t dfs_index = 1;
   calc_dfs_indicies(start_block, &dfs_index);
}

void
nir_calc_dominance(nir_shader *shader)
{
   nir_foreach_function(function, shader) {
      if (function->impl)
         nir_calc_dominance_impl(function->impl);
   }
}

static nir_block *
block_return_if_reachable(nir_block *b)
{
   return (b && nir_block_is_reachable(b)) ? b : NULL;
}

/**
 * Computes the least common ancestor of two blocks.  If one of the blocks
 * is null or unreachable, the other block is returned or NULL if it's
 * unreachable.
 */
nir_block *
nir_dominance_lca(nir_block *b1, nir_block *b2)
{
   if (b1 == NULL || !nir_block_is_reachable(b1))
      return block_return_if_reachable(b2);

   if (b2 == NULL || !nir_block_is_reachable(b2))
      return block_return_if_reachable(b1);

   assert(nir_cf_node_get_function(&b1->cf_node) ==
          nir_cf_node_get_function(&b2->cf_node));

   assert(nir_cf_node_get_function(&b1->cf_node)->valid_metadata &
          nir_metadata_dominance);

   return intersect(b1, b2);
}

/**
 * Returns true if parent dominates child according to the following
 * definition:
 *
 *    "The block A dominates the block B if every path from the start block
 *    to block B passes through A."
 *
 * This means, in particular, that any unreachable block is dominated by every
 * other block and an unreachable block does not dominate anything except
 * another unreachable block.
 */
bool
nir_block_dominates(nir_block *parent, nir_block *child)
{
   assert(nir_cf_node_get_function(&parent->cf_node) ==
          nir_cf_node_get_function(&child->cf_node));

   assert(nir_cf_node_get_function(&parent->cf_node)->valid_metadata &
          nir_metadata_dominance);

   /* If a block is unreachable, then nir_block::dom_pre_index == UINT32_MAX
    * and nir_block::dom_post_index == 0.  This allows us to trivially handle
    * unreachable blocks here with zero extra work.
    */
   return child->dom_pre_index >= parent->dom_pre_index &&
          child->dom_post_index <= parent->dom_post_index;
}

bool
nir_block_is_unreachable(nir_block *block)
{
   assert(nir_cf_node_get_function(&block->cf_node)->valid_metadata &
          nir_metadata_dominance);
   assert(nir_cf_node_get_function(&block->cf_node)->valid_metadata &
          nir_metadata_block_index);

   /* Unreachable blocks have no dominator.  The only reachable block with no
    * dominator is the start block which has index 0.
    */
   return block->index > 0 && block->imm_dom == NULL;
}

void
nir_dump_dom_tree_impl(nir_function_impl *impl, FILE *fp)
{
   fprintf(fp, "digraph doms_%s {\n", impl->function->name);

   nir_foreach_block_unstructured(block, impl) {
      if (block->imm_dom)
         fprintf(fp, "\t%u -> %u\n", block->imm_dom->index, block->index);
   }

   fprintf(fp, "}\n\n");
}

void
nir_dump_dom_tree(nir_shader *shader, FILE *fp)
{
   nir_foreach_function(function, shader) {
      if (function->impl)
         nir_dump_dom_tree_impl(function->impl, fp);
   }
}

void
nir_dump_dom_frontier_impl(nir_function_impl *impl, FILE *fp)
{
   nir_foreach_block_unstructured(block, impl) {
      fprintf(fp, "DF(%u) = {", block->index);
      set_foreach(block->dom_frontier, entry) {
         nir_block *df = (nir_block *) entry->key;
         fprintf(fp, "%u, ", df->index);
      }
      fprintf(fp, "}\n");
   }
}

void
nir_dump_dom_frontier(nir_shader *shader, FILE *fp)
{
   nir_foreach_function(function, shader) {
      if (function->impl)
         nir_dump_dom_frontier_impl(function->impl, fp);
   }
}

void
nir_dump_cfg_impl(nir_function_impl *impl, FILE *fp)
{
   fprintf(fp, "digraph cfg_%s {\n", impl->function->name);

   nir_foreach_block_unstructured(block, impl) {
      if (block->successors[0])
         fprintf(fp, "\t%u -> %u\n", block->index, block->successors[0]->index);
      if (block->successors[1])
         fprintf(fp, "\t%u -> %u\n", block->index, block->successors[1]->index);
   }

   fprintf(fp, "}\n\n");
}

void
nir_dump_cfg(nir_shader *shader, FILE *fp)
{
   nir_foreach_function(function, shader) {
      if (function->impl)
         nir_dump_cfg_impl(function->impl, fp);
   }
}
