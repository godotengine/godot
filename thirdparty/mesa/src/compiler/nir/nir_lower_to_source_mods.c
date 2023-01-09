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
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#include "nir.h"

/*
 * This pass lowers the neg, abs, and sat operations to source modifiers on
 * ALU operations to make things nicer for the backend.  It's just much
 * easier to not have them when we're doing optimizations.
 */

static void
alu_src_consume_abs(nir_alu_src *src)
{
   src->abs = true;
}

static void
alu_src_consume_negate(nir_alu_src *src)
{
   /* If abs is set on the source, the negate goes away */
   if (!src->abs)
      src->negate = !src->negate;
}

static bool
nir_lower_to_source_mods_block(nir_block *block,
                               nir_lower_to_source_mods_flags options)
{
   bool progress = false;

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *alu = nir_instr_as_alu(instr);

      bool lower_abs = (nir_op_infos[alu->op].num_inputs < 3) ||
            (options & nir_lower_triop_abs);

      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (!alu->src[i].src.is_ssa)
            continue;

         if (alu->src[i].src.ssa->parent_instr->type != nir_instr_type_alu)
            continue;

         nir_alu_instr *parent = nir_instr_as_alu(alu->src[i].src.ssa->parent_instr);

         if (parent->dest.saturate)
            continue;

         switch (nir_alu_type_get_base_type(nir_op_infos[alu->op].input_types[i])) {
         case nir_type_float:
            if (!(options & nir_lower_float_source_mods))
               continue;
            if (!(parent->op == nir_op_fabs && (options & nir_lower_fabs_source_mods)) &&
                !(parent->op == nir_op_fneg && (options & nir_lower_fneg_source_mods))) {
               continue;
            }
            break;
         case nir_type_int:
            if (!(options & nir_lower_int_source_mods))
               continue;
            if (parent->op != nir_op_iabs && parent->op != nir_op_ineg)
               continue;
            break;
         default:
            continue;
         }

         if (nir_src_bit_size(alu->src[i].src) == 64 &&
             !(options & nir_lower_64bit_source_mods)) {
            continue;
         }

         /* We can only do a rewrite if the source we are copying is SSA.
          * Otherwise, moving the read might invalidly reorder reads/writes
          * on a register.
          */
         if (!parent->src[0].src.is_ssa)
            continue;

         if (!lower_abs && (parent->op == nir_op_fabs ||
                            parent->op == nir_op_iabs ||
                            parent->src[0].abs))
            continue;

         nir_instr_rewrite_src(instr, &alu->src[i].src, parent->src[0].src);

         /* Apply any modifiers that come from the parent opcode */
         if (parent->op == nir_op_fneg || parent->op == nir_op_ineg)
            alu_src_consume_negate(&alu->src[i]);
         if (parent->op == nir_op_fabs || parent->op == nir_op_iabs)
            alu_src_consume_abs(&alu->src[i]);

         /* Apply modifiers from the parent source */
         if (parent->src[0].negate)
            alu_src_consume_negate(&alu->src[i]);
         if (parent->src[0].abs)
            alu_src_consume_abs(&alu->src[i]);

         for (int j = 0; j < 4; ++j) {
            if (!nir_alu_instr_channel_used(alu, i, j))
               continue;
            alu->src[i].swizzle[j] = parent->src[0].swizzle[alu->src[i].swizzle[j]];
         }

         if (nir_ssa_def_is_unused(&parent->dest.dest.ssa))
            nir_instr_remove(&parent->instr);

         progress = true;
      }

      /* We've covered sources.  Now we're going to try and saturate the
       * destination if we can.
       */

      if (!alu->dest.dest.is_ssa)
         continue;

      if (nir_dest_bit_size(alu->dest.dest) == 64 &&
          !(options & nir_lower_64bit_source_mods)) {
         continue;
      }

      /* We can only saturate float destinations */
      if (nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) !=
          nir_type_float)
         continue;

      if (!(options & nir_lower_float_source_mods))
         continue;

      if (!list_is_empty(&alu->dest.dest.ssa.if_uses))
         continue;

      bool all_children_are_sat = true;
      nir_foreach_use(child_src, &alu->dest.dest.ssa) {
         assert(child_src->is_ssa);
         nir_instr *child = child_src->parent_instr;
         if (child->type != nir_instr_type_alu) {
            all_children_are_sat = false;
            continue;
         }

         nir_alu_instr *child_alu = nir_instr_as_alu(child);
         if (child_alu->src[0].negate || child_alu->src[0].abs) {
            all_children_are_sat = false;
            continue;
         }

         if (child_alu->op != nir_op_fsat) {
            all_children_are_sat = false;
            continue;
         }
      }

      if (!all_children_are_sat)
         continue;

      alu->dest.saturate = true;
      progress = true;

      nir_foreach_use(child_src, &alu->dest.dest.ssa) {
         assert(child_src->is_ssa);
         nir_alu_instr *child_alu = nir_instr_as_alu(child_src->parent_instr);

         child_alu->op = nir_op_mov;
         child_alu->dest.saturate = false;
         /* We could propagate the dest of our instruction to the
          * destinations of the uses here.  However, one quick round of
          * copy propagation will clean that all up and then we don't have
          * the complexity.
          */
      }
   }

   return progress;
}

static bool
nir_lower_to_source_mods_impl(nir_function_impl *impl,
                              nir_lower_to_source_mods_flags options)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      progress |= nir_lower_to_source_mods_block(block, options);
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);

   return progress;
}

bool
nir_lower_to_source_mods(nir_shader *shader,
                         nir_lower_to_source_mods_flags options)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl) {
         progress |= nir_lower_to_source_mods_impl(function->impl, options);
      }
   }

   return progress;
}
