/*
 * Copyright © 2015 Broadcom
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

/** @file nir_opt_undef.c
 *
 * Handles optimization of operations involving ssa_undef.
 */

/**
 * Turn conditional selects between an undef and some other value into a move
 * of that other value (on the assumption that the condition's going to be
 * choosing the defined value).  This reduces work after if flattening when
 * each side of the if is defining a variable.
 */
static bool
opt_undef_csel(nir_alu_instr *instr)
{
   if (!nir_op_is_selection(instr->op))
      return false;

   assert(instr->dest.dest.is_ssa);

   for (int i = 1; i <= 2; i++) {
      if (!instr->src[i].src.is_ssa)
         continue;

      nir_instr *parent = instr->src[i].src.ssa->parent_instr;
      if (parent->type != nir_instr_type_ssa_undef)
         continue;

      /* We can't just use nir_alu_src_copy, because we need the def/use
       * updated.
       */
      nir_instr_rewrite_src(&instr->instr, &instr->src[0].src,
                            instr->src[i == 1 ? 2 : 1].src);
      nir_alu_src_copy(&instr->src[0], &instr->src[i == 1 ? 2 : 1],
                       instr);

      nir_src empty_src;
      memset(&empty_src, 0, sizeof(empty_src));
      nir_instr_rewrite_src(&instr->instr, &instr->src[1].src, empty_src);
      nir_instr_rewrite_src(&instr->instr, &instr->src[2].src, empty_src);
      instr->op = nir_op_mov;

      return true;
   }

   return false;
}

/**
 * Replace vecN(undef, undef, ...) with a single undef.
 */
static bool
opt_undef_vecN(nir_builder *b, nir_alu_instr *alu)
{
   if (!nir_op_is_vec(alu->op))
      return false;

   assert(alu->dest.dest.is_ssa);

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      if (!alu->src[i].src.is_ssa ||
          alu->src[i].src.ssa->parent_instr->type != nir_instr_type_ssa_undef)
         return false;
   }

   b->cursor = nir_before_instr(&alu->instr);
   nir_ssa_def *undef = nir_ssa_undef(b, alu->dest.dest.ssa.num_components,
                                      nir_dest_bit_size(alu->dest.dest));
   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, undef);

   return true;
}

static uint32_t
nir_get_undef_mask(nir_ssa_def *def)
{
   nir_instr *instr = def->parent_instr;

   if (instr->type == nir_instr_type_ssa_undef)
      return BITSET_MASK(def->num_components);

   if (instr->type != nir_instr_type_alu)
      return 0;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   unsigned undef = 0;

   if (nir_op_is_vec(alu->op)) {
      for (int i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (alu->src[i].src.is_ssa &&
             alu->src[i].src.ssa->parent_instr->type ==
             nir_instr_type_ssa_undef) {
            undef |= BITSET_MASK(nir_ssa_alu_instr_src_components(alu, i)) << i;
         }
      }
   }

   return undef;
}

/**
 * Remove any store intrinsic writemask channels whose value is undefined (the
 * existing value is a fine representation of "undefined").
 */
static bool
opt_undef_store(nir_intrinsic_instr *intrin)
{
   int arg_index;
   switch (intrin->intrinsic) {
   case nir_intrinsic_store_deref:
      arg_index = 1;
      break;
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_scratch:
      arg_index =  0;
      break;
   default:
      return false;
   }

   if (!intrin->src[arg_index].is_ssa)
      return false;

   nir_ssa_def *def = intrin->src[arg_index].ssa;

   unsigned write_mask = nir_intrinsic_write_mask(intrin);
   unsigned undef_mask = nir_get_undef_mask(def);

   if (!(write_mask & undef_mask))
      return false;

   write_mask &= ~undef_mask;
   if (!write_mask)
      nir_instr_remove(&intrin->instr);
   else
      nir_intrinsic_set_write_mask(intrin, write_mask);

   return true;
}

static bool
opt_undef_pack(nir_builder *b, nir_alu_instr *alu)
{
   switch (alu->op) {
   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_64_2x32_split_y:
   case nir_op_unpack_64_2x32:
      if (nir_src_is_undef(alu->src[0].src))
         break;
      return false;
   default:
      return false;
   }
   unsigned num_components = nir_dest_num_components(alu->dest.dest);
   b->cursor = nir_before_instr(&alu->instr);
   nir_ssa_def *def = nir_ssa_undef(b, num_components, 32);
   nir_ssa_def_rewrite_uses_after(&alu->dest.dest.ssa, def, &alu->instr);
   nir_instr_remove(&alu->instr);
   return true;
}

static bool
nir_opt_undef_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      return opt_undef_csel(alu) || opt_undef_vecN(b, alu) || opt_undef_pack(b, alu);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      return opt_undef_store(intrin);
   }

   return false;
}

bool
nir_opt_undef(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader,
                                       nir_opt_undef_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
