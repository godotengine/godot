/*
 * Copyright © 2021 Google, Inc.
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

/*
 * This pass tries to reduce the bitsize of phi instructions by either
 * moving narrowing conversions from the phi's consumers to the phi's
 * sources, if all the uses of the phi are equivalent narrowing
 * instructions.  In other words, convert:
 *
 *    vec1 32 ssa_124 = load_const (0x00000000)
 *    ...
 *    loop {
 *        ...
 *        vec1 32 ssa_155 = phi block_0: ssa_124, block_4: ssa_53
 *        vec1 16 ssa_8 = i2imp ssa_155
 *        ...
 *        vec1 32 ssa_53 = i2i32 ssa_52
 *    }
 *
 * into:
 *
 *    vec1 32 ssa_124 = load_const (0x00000000)
 *    vec1 16 ssa_156 = i2imp ssa_124
 *    ...
 *    loop {
 *        ...
 *        vec1 16 ssa_8 = phi block_0: ssa_156, block_4: ssa_157
 *        ...
 *        vec1 32 ssa_53 = i2i32 ssa_52
 *        vec1 16 ssa_157 = i2i16 ssa_53
 *    }
 *
 * Or failing that, tries to push widening conversion of phi srcs to
 * the phi def.  In this case, since load_const is frequently one
 * of the phi sources this pass checks if can be narrowed without a
 * loss of precision:
 *
 *    vec1 32 ssa_0 = load_const (0x00000000)
 *    ...
 *    loop {
 *        ...
 *        vec1 32 ssa_8 = phi block_0: ssa_0, block_4: ssa_19
 *        ...
 *        vec1 16 ssa_18 = iadd ssa_21, ssa_3
 *        vec1 32 ssa_19 = i2i32 ssa_18
 *    }
 *
 * into:
 *
 *    vec1 32 ssa_0 = load_const (0x00000000)
 *    vec1 16 ssa_22 = i2i16 ssa_0
 *    ...
 *    loop {
 *        ...
 *        vec1 16 ssa_8 = phi block_0: ssa_22, block_4: ssa_18
 *        vec1 32 ssa_23 = i2i32 ssa_8
 *        ...
 *        vec1 16 ssa_18 = iadd ssa_21, ssa_3
 *    }
 *
 * Note that either transformations can convert x2ymp  into x2y16, which
 * is normally done later in nir_opt_algebraic_late(), losing the option
 * to fold away sequences like (i2i32 (i2imp (x))), but algebraic opts
 * cannot see through phis.
 */

#define INVALID_OP nir_num_opcodes

/**
 * Get the corresponding exact conversion for a x2ymp conversion
 */
static nir_op
concrete_conversion(nir_op op)
{
   switch (op) {
   case nir_op_i2imp: return nir_op_i2i16;
   case nir_op_i2fmp: return nir_op_i2f16;
   case nir_op_u2fmp: return nir_op_u2f16;
   case nir_op_f2fmp: return nir_op_f2f16;
   case nir_op_f2imp: return nir_op_f2i16;
   case nir_op_f2ump: return nir_op_f2u16;
   default:           return op;
   }
}

static nir_op
narrowing_conversion_op(nir_instr *instr, nir_op current_op)
{
   if (instr->type != nir_instr_type_alu)
      return INVALID_OP;

   nir_op op = nir_instr_as_alu(instr)->op;
   switch (op) {
   case nir_op_i2imp:
   case nir_op_i2i16:
   case nir_op_i2fmp:
   case nir_op_i2f16:
   case nir_op_u2fmp:
   case nir_op_u2f16:
   case nir_op_f2fmp:
   case nir_op_f2f16:
   case nir_op_f2imp:
   case nir_op_f2i16:
   case nir_op_f2ump:
   case nir_op_f2u16:
   case nir_op_f2f16_rtne:
   case nir_op_f2f16_rtz:
      break;
   default:
      return INVALID_OP;
   }

   /* If we've already picked a conversion op from a previous phi use,
    * make sure it is compatible with the current use
    */
   if (current_op != INVALID_OP) {
      if (current_op != op) {
         /* If we have different conversions, but one can be converted
          * to the other, then let's do that:
          */
         if (concrete_conversion(current_op) == concrete_conversion(op)) {
            op = concrete_conversion(op);
         } else {
            return INVALID_OP;
         }
      }
   }

   return op;
}

static nir_op
widening_conversion_op(nir_instr *instr, unsigned *bit_size)
{
   if (instr->type != nir_instr_type_alu)
      return INVALID_OP;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   case nir_op_i2i32:
   case nir_op_i2f32:
   case nir_op_u2f32:
   case nir_op_f2f32:
   case nir_op_f2i32:
   case nir_op_f2u32:
      break;
   default:
      return INVALID_OP;
   }

   *bit_size = nir_src_bit_size(alu->src[0].src);

   /* We also need to check that the conversion's dest was actually
    * wider:
    */
   if (nir_dest_bit_size(alu->dest.dest) <= *bit_size)
      return INVALID_OP;

   return alu->op;
}

static nir_alu_type
op_to_type(nir_op op)
{
   return nir_alu_type_get_base_type(nir_op_infos[op].output_type);
}

/* Try to move narrowing instructions consuming the phi into the phi's
 * sources to reduce the phi's precision:
 */
static bool
try_move_narrowing_dst(nir_builder *b, nir_phi_instr *phi)
{
   nir_op op = INVALID_OP;

   assert(phi->dest.is_ssa);

   /* If the phi has already been narrowed, nothing more to do: */
   if (phi->dest.ssa.bit_size != 32)
      return false;

   /* Are the only uses of the phi conversion instructions, and
    * are they all the same conversion?
    */
   nir_foreach_use (use, &phi->dest.ssa) {
      op = narrowing_conversion_op(use->parent_instr, op);

      /* Not a (compatible) narrowing conversion: */
      if (op == INVALID_OP)
         return false;
   }

   /* an if_uses means the phi is used directly in a conditional, ie.
    * without a conversion
    */
   if (!list_is_empty(&phi->dest.ssa.if_uses))
      return false;

   /* If the phi has no uses, then nothing to do: */
   if (op == INVALID_OP)
      return false;

   /* construct replacement phi instruction: */
   nir_phi_instr *new_phi = nir_phi_instr_create(b->shader);
   nir_ssa_dest_init(&new_phi->instr, &new_phi->dest,
                     phi->dest.ssa.num_components,
                     nir_alu_type_get_type_size(nir_op_infos[op].output_type),
                     NULL);

   /* Push the conversion into the new phi sources: */
   nir_foreach_phi_src (src, phi) {
      assert(src->src.is_ssa);

      /* insert new conversion instr in block of original phi src: */
      b->cursor = nir_after_instr_and_phis(src->src.ssa->parent_instr);
      nir_ssa_def *old_src = src->src.ssa;
      nir_ssa_def *new_src = nir_build_alu(b, op, old_src, NULL, NULL, NULL);

      /* and add corresponding phi_src to the new_phi: */
      nir_phi_instr_add_src(new_phi, src->pred, nir_src_for_ssa(new_src));
   }

   /* And finally rewrite the original uses of the original phi uses to
    * directly use the new phi, skipping the conversion out of the orig
    * phi
    */
   nir_foreach_use (use, &phi->dest.ssa) {
      /* We've previously established that all the uses were alu
       * conversion ops.  Turn them into movs instead.
       */
      nir_alu_instr *alu = nir_instr_as_alu(use->parent_instr);
      alu->op = nir_op_mov;
   }
   nir_ssa_def_rewrite_uses(&phi->dest.ssa, &new_phi->dest.ssa);

   /* And finally insert the new phi after all sources are in place: */
   b->cursor = nir_after_instr(&phi->instr);
   nir_builder_instr_insert(b, &new_phi->instr);

   return true;
}

static bool
can_convert_load_const(nir_load_const_instr *lc, nir_op op)
{
   nir_alu_type type = op_to_type(op);

   /* Note that we only handle phi's with bit_size == 32: */
   assert(lc->def.bit_size == 32);

   for (unsigned i = 0; i < lc->def.num_components; i++) {
      switch (type) {
      case nir_type_int:
         if (lc->value[i].i32 != (int32_t)(int16_t)lc->value[i].i32)
            return false;
         break;
      case nir_type_uint:
         if (lc->value[i].u32 != (uint32_t)(uint16_t)lc->value[i].u32)
            return false;
         break;
      case nir_type_float:
         if (lc->value[i].f32 != _mesa_half_to_float(
               _mesa_float_to_half(lc->value[i].f32)))
            return false;
         break;
      default:
         unreachable("bad type");
         return false;
      }
   }

   return true;
}

/* Check all the phi sources to see if they are the same widening op, in
 * which case we can push the widening op to the other side of the phi
 */
static nir_op
find_widening_op(nir_phi_instr *phi, unsigned *bit_size)
{
   nir_op op = INVALID_OP;

   bool has_load_const = false;
   *bit_size = 0;

   nir_foreach_phi_src (src, phi) {
      assert(src->src.is_ssa);

      nir_instr *instr = src->src.ssa->parent_instr;
      if (instr->type == nir_instr_type_load_const) {
         has_load_const = true;
         continue;
      }

      unsigned src_bit_size;
      nir_op src_op = widening_conversion_op(instr, &src_bit_size);

      /* Not a widening conversion: */
      if (src_op == INVALID_OP)
         return INVALID_OP;

      /* If it is a widening conversion, it needs to be the same op as
       * other phi sources:
       */
      if ((op != INVALID_OP) && (op != src_op))
         return INVALID_OP;

      if (*bit_size && (*bit_size != src_bit_size))
         return INVALID_OP;

      op = src_op;
      *bit_size = src_bit_size;
   }

   if ((op == INVALID_OP) || !has_load_const)
      return op;

   /* If we could otherwise move widening sources, but load_const is
    * one of the phi sources (and does not have a widening conversion,
    * but could have a narrowing->widening sequence inserted without
    * loss of precision), then we could insert a narrowing->widening
    * sequence to make the rest of the transformation possible:
    */
   nir_foreach_phi_src (src, phi) {
      assert(src->src.is_ssa);

      nir_instr *instr = src->src.ssa->parent_instr;
      if (instr->type != nir_instr_type_load_const)
         continue;

      if (!can_convert_load_const(nir_instr_as_load_const(instr), op))
         return INVALID_OP;
   }

   return op;
}

/* Try to move widening conversions into the phi to the phi's output
 * to reduce the phi's precision:
 */
static bool
try_move_widening_src(nir_builder *b, nir_phi_instr *phi)
{
   assert(phi->dest.is_ssa);

   /* If the phi has already been narrowed, nothing more to do: */
   if (phi->dest.ssa.bit_size != 32)
      return false;

   unsigned bit_size;
   nir_op op = find_widening_op(phi, &bit_size);

   if (op == INVALID_OP)
      return false;

   /* construct replacement phi instruction: */
   nir_phi_instr *new_phi = nir_phi_instr_create(b->shader);
   nir_ssa_dest_init(&new_phi->instr, &new_phi->dest,
                     phi->dest.ssa.num_components,
                     bit_size, NULL);

   /* Remove the widening conversions from the phi sources: */
   nir_foreach_phi_src (src, phi) {
      assert(src->src.is_ssa);

      nir_instr *instr = src->src.ssa->parent_instr;
      nir_ssa_def *new_src;

      b->cursor = nir_after_instr(instr);

      if (instr->type == nir_instr_type_load_const) {
         /* if the src is a load_const, we've already verified that it
          * is safe to insert a narrowing conversion to make the rest
          * of this transformation legal:
          */
         nir_load_const_instr *lc = nir_instr_as_load_const(instr);

         if (op_to_type(op) == nir_type_float) {
            new_src = nir_f2f16(b, &lc->def);
         } else {
            new_src = nir_i2i16(b, &lc->def);
         }
      } else {
         /* at this point we know the sources source is a conversion: */
         nir_alu_instr *alu = nir_instr_as_alu(instr);

         /* The conversion we are stripping off could have had a swizzle,
          * so replace it with a mov if necessary:
          */
         unsigned num_comp = nir_dest_num_components(alu->dest.dest);
         new_src = nir_mov_alu(b, alu->src[0], num_comp);
      }

      /* add corresponding phi_src to the new_phi: */
      nir_phi_instr_add_src(new_phi, src->pred, nir_src_for_ssa(new_src));
   }

   /* And insert the new phi after all sources are in place: */
   b->cursor = nir_after_instr(&phi->instr);
   nir_builder_instr_insert(b, &new_phi->instr);

   /* And finally add back the widening conversion after the phi,
    * and re-write the original phi's uses
    */
   b->cursor = nir_after_instr_and_phis(&new_phi->instr);
   nir_ssa_def *def = nir_build_alu(b, op, &new_phi->dest.ssa, NULL, NULL, NULL);

   nir_ssa_def_rewrite_uses(&phi->dest.ssa, def);

   return true;
}

static bool
lower_phi(nir_builder *b, nir_phi_instr *phi)
{
   bool progress = try_move_narrowing_dst(b, phi);
   if (!progress)
      progress = try_move_widening_src(b, phi);
   return progress;
}

bool
nir_opt_phi_precision(nir_shader *shader)
{
   bool progress = false;

   /* If 8b or 16b bit_sizes are not used, no point to run this pass: */
   unsigned bit_sizes_used = shader->info.bit_sizes_float |
                             shader->info.bit_sizes_int;

   if (!bit_sizes_used) {
      nir_shader_gather_info(shader, nir_shader_get_entrypoint(shader));
      bit_sizes_used = shader->info.bit_sizes_float |
                       shader->info.bit_sizes_int;
   }

   if (!(bit_sizes_used & (8 | 16)))
      return false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_builder b;
      nir_builder_init(&b, function->impl);

      nir_foreach_block (block, function->impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_phi)
               break;

            progress |= lower_phi(&b, nir_instr_as_phi(instr));
         }
      }

      if (progress) {
         nir_metadata_preserve(function->impl,
                               nir_metadata_block_index |
                               nir_metadata_dominance);
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}

