/*
 * Copyright © 2015 Connor Abbott
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
#include "nir_builder.h"

static nir_alu_instr *
get_parent_mov(nir_ssa_def *ssa)
{
   if (ssa->parent_instr->type != nir_instr_type_alu)
      return NULL;

   nir_alu_instr *alu = nir_instr_as_alu(ssa->parent_instr);
   return (alu->op == nir_op_mov) ? alu : NULL;
}

static bool
matching_mov(nir_alu_instr *mov1, nir_ssa_def *ssa)
{
   if (!mov1)
      return false;

   nir_alu_instr *mov2 = get_parent_mov(ssa);
   return mov2 && nir_alu_srcs_equal(mov1, mov2, 0, 0);
}

/*
 * This is a pass for removing phi nodes that look like:
 * a = phi(b, b, b, ...)
 *
 * Note that we can't always ignore undef sources here, or else we may create a
 * situation where the definition of b isn't dominated by its uses. We're
 * allowed to do this since the definition of b must dominate all of the
 * phi node's predecessors, which means it must dominate the phi node as well
 * as all of the phi node's uses. In essence, the phi node acts as a copy
 * instruction. b can't be another phi node in the same block, since the only
 * time when phi nodes can source other phi nodes defined in the same block is
 * at the loop header, and in that case one of the sources of the phi has to
 * be from before the loop and that source can't be b.
 */

static bool
remove_phis_block(nir_block *block, nir_builder *b)
{
   bool progress = false;

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_phi)
         break;

      nir_phi_instr *phi = nir_instr_as_phi(instr);

      nir_ssa_def *def = NULL;
      nir_alu_instr *mov = NULL;
      bool srcs_same = true;

      nir_foreach_phi_src(src, phi) {
         assert(src->src.is_ssa);

         /* For phi nodes at the beginning of loops, we may encounter some
          * sources from backedges that point back to the destination of the
          * same phi, i.e. something like:
          *
          * a = phi(a, b, ...)
          *
          * We can safely ignore these sources, since if all of the normal
          * sources point to the same definition, then that definition must
          * still dominate the phi node, and the phi will still always take
          * the value of that definition.
          */
         if (src->src.ssa == &phi->dest.ssa)
            continue;
         
         if (def == NULL) {
            def  = src->src.ssa;
            mov = get_parent_mov(def);
         } else if (nir_src_is_undef(src->src) &&
                    nir_block_dominates(def->parent_instr->block, src->pred)) {
            /* Ignore this undef source. */
         } else {
            if (src->src.ssa != def && !matching_mov(mov, src->src.ssa)) {
               srcs_same = false;
               break;
            }
         }
      }

      if (!srcs_same)
         continue;

      if (!def) {
         /* In this case, the phi had no sources. So turn it into an undef. */

         b->cursor = nir_after_phis(block);
         def = nir_ssa_undef(b, phi->dest.ssa.num_components,
                             phi->dest.ssa.bit_size);
      } else if (mov) {
         /* If the sources were all movs from the same source with the same
          * swizzle, then we can't just pick a random move because it may not
          * dominate the phi node. Instead, we need to emit our own move after
          * the phi which uses the shared source, and rewrite uses of the phi
          * to use the move instead. This is ok, because while the movs may
          * not all dominate the phi node, their shared source does.
          */

         b->cursor = nir_after_phis(block);
         def = nir_mov_alu(b, mov->src[0], def->num_components);
      }

      assert(phi->dest.is_ssa);
      nir_ssa_def_rewrite_uses(&phi->dest.ssa, def);
      nir_instr_remove(instr);

      progress = true;
   }

   return progress;
}

bool
nir_opt_remove_phis_block(nir_block *block)
{
   nir_builder b;
   nir_builder_init(&b, nir_cf_node_get_function(&block->cf_node));
   return remove_phis_block(block, &b);
}

static bool
nir_opt_remove_phis_impl(nir_function_impl *impl)
{
   bool progress = false;
   nir_builder bld;
   nir_builder_init(&bld, impl);

   nir_metadata_require(impl, nir_metadata_dominance);

   nir_foreach_block(block, impl) {
      progress |= remove_phis_block(block, &bld);
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
nir_opt_remove_phis(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader)
      if (function->impl)
         progress = nir_opt_remove_phis_impl(function->impl) || progress;

   return progress;
}

