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
#include "nir_builder.h"

/**
 * SSA-based copy propagation
 */

static bool
is_swizzleless_move(nir_alu_instr *instr)
{
   unsigned num_comp = instr->dest.dest.ssa.num_components;

   if (instr->src[0].src.ssa->num_components != num_comp)
      return false;

   if (instr->op == nir_op_mov) {
      for (unsigned i = 0; i < num_comp; i++) {
         if (instr->src[0].swizzle[i] != i)
            return false;
      }
   } else {
      for (unsigned i = 0; i < num_comp; i++) {
         if (instr->src[i].swizzle[0] != i ||
             instr->src[i].src.ssa != instr->src[0].src.ssa)
            return false;
      }
   }

   return true;
}

static bool
rewrite_to_vec(nir_function_impl *impl, nir_alu_instr *mov, nir_alu_instr *vec)
{
   if (mov->op != nir_op_mov)
      return false;

   nir_builder b;
   nir_builder_init(&b, impl);
   b.cursor = nir_after_instr(&mov->instr);

   unsigned num_comp = mov->dest.dest.ssa.num_components;
   nir_alu_instr *new_vec = nir_alu_instr_create(b.shader, nir_op_vec(num_comp));
   for (unsigned i = 0; i < num_comp; i++)
      new_vec->src[i] = vec->src[mov->src[0].swizzle[i]];

   nir_ssa_def *new = nir_builder_alu_instr_finish_and_insert(&b, new_vec);
   nir_ssa_def_rewrite_uses(&mov->dest.dest.ssa, new);

   /* If we remove "mov" and it's the next instruction in the
    * nir_foreach_instr_safe() loop, then we would end copy-propagation early. */

   return true;
}

static bool
copy_propagate_alu(nir_function_impl *impl, nir_alu_src *src, nir_alu_instr *copy)
{
   nir_ssa_def *def = NULL;
   nir_alu_instr *user = nir_instr_as_alu(src->src.parent_instr);
   unsigned src_idx = src - user->src;
   assert(src_idx < nir_op_infos[user->op].num_inputs);
   unsigned num_comp = nir_ssa_alu_instr_src_components(user, src_idx);

   if (copy->op == nir_op_mov) {
      def = copy->src[0].src.ssa;

      for (unsigned i = 0; i < num_comp; i++)
         src->swizzle[i] = copy->src[0].swizzle[src->swizzle[i]];
   } else {
      def = copy->src[src->swizzle[0]].src.ssa;

      for (unsigned i = 1; i < num_comp; i++) {
         if (copy->src[src->swizzle[i]].src.ssa != def)
            return rewrite_to_vec(impl, user, copy);
      }

      for (unsigned i = 0; i < num_comp; i++)
         src->swizzle[i] = copy->src[src->swizzle[i]].swizzle[0];
   }

   nir_instr_rewrite_src_ssa(src->src.parent_instr, &src->src, def);

   return true;
}

static bool
copy_propagate(nir_src *src, nir_alu_instr *copy)
{
   if (!is_swizzleless_move(copy))
      return false;

   nir_instr_rewrite_src_ssa(src->parent_instr, src, copy->src[0].src.ssa);

   return true;
}

static bool
copy_propagate_if(nir_src *src, nir_alu_instr *copy)
{
   if (!is_swizzleless_move(copy))
      return false;

   nir_if_rewrite_condition_ssa(src->parent_if, src, copy->src[0].src.ssa);

   return true;
}

static bool
copy_prop_instr(nir_function_impl *impl, nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *mov = nir_instr_as_alu(instr);

   if (!nir_alu_instr_is_copy(mov))
      return false;

   bool progress = false;

   nir_foreach_use_safe(src, &mov->dest.dest.ssa) {
      if (src->parent_instr->type == nir_instr_type_alu)
         progress |= copy_propagate_alu(impl, container_of(src, nir_alu_src, src), mov);
      else
         progress |= copy_propagate(src, mov);
   }

   nir_foreach_if_use_safe(src, &mov->dest.dest.ssa)
      progress |= copy_propagate_if(src, mov);

   if (progress && nir_ssa_def_is_unused(&mov->dest.dest.ssa))
      nir_instr_remove(&mov->instr);

   return progress;
}

bool
nir_copy_prop_impl(nir_function_impl *impl)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         progress |= copy_prop_instr(impl, instr);
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
nir_copy_prop(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl && nir_copy_prop_impl(function->impl))
         progress = true;
   }

   return progress;
}
