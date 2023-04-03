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

#include "nir_builder.h"

/**
 * Some ALU operations may not be supported in hardware in specific bit-sizes.
 * This pass allows implementations to selectively lower such operations to
 * a bit-size that is supported natively and then converts the result back to
 * the original bit-size.
 */

static nir_ssa_def *convert_to_bit_size(nir_builder *bld, nir_ssa_def *src,
                                        nir_alu_type type, unsigned bit_size)
{
   /* create b2i32(a) instead of i2i32(b2i8(a))/i2i32(b2i16(a)) */
   nir_alu_instr *alu = nir_src_as_alu_instr(nir_src_for_ssa(src));
   if ((type & (nir_type_uint | nir_type_int)) && bit_size == 32 &&
       alu && (alu->op == nir_op_b2i8 || alu->op == nir_op_b2i16)) {
      nir_alu_instr *instr = nir_alu_instr_create(bld->shader, nir_op_b2i32);
      nir_alu_src_copy(&instr->src[0], &alu->src[0], instr);
      return nir_builder_alu_instr_finish_and_insert(bld, instr);
   }

   return nir_convert_to_bit_size(bld, src, type, bit_size);
}

static void
lower_alu_instr(nir_builder *bld, nir_alu_instr *alu, unsigned bit_size)
{
   const nir_op op = alu->op;
   unsigned dst_bit_size = alu->dest.dest.ssa.bit_size;

   bld->cursor = nir_before_instr(&alu->instr);

   /* Convert each source to the requested bit-size */
   nir_ssa_def *srcs[NIR_MAX_VEC_COMPONENTS] = { NULL };
   for (unsigned i = 0; i < nir_op_infos[op].num_inputs; i++) {
      nir_ssa_def *src = nir_ssa_for_alu_src(bld, alu, i);

      nir_alu_type type = nir_op_infos[op].input_types[i];
      if (nir_alu_type_get_type_size(type) == 0)
         src = convert_to_bit_size(bld, src, type, bit_size);

      if (i == 1 && (op == nir_op_ishl || op == nir_op_ishr || op == nir_op_ushr)) {
         assert(util_is_power_of_two_nonzero(dst_bit_size));
         src = nir_iand(bld, src, nir_imm_int(bld, dst_bit_size - 1));
      }

      srcs[i] = src;
   }

   /* Emit the lowered ALU instruction */
   nir_ssa_def *lowered_dst = NULL;
   if (op == nir_op_imul_high || op == nir_op_umul_high) {
      assert(dst_bit_size * 2 <= bit_size);
      lowered_dst = nir_imul(bld, srcs[0], srcs[1]);
      if (nir_op_infos[op].output_type & nir_type_uint)
         lowered_dst = nir_ushr_imm(bld, lowered_dst, dst_bit_size);
      else
         lowered_dst = nir_ishr_imm(bld, lowered_dst, dst_bit_size);
   } else if (op == nir_op_iadd_sat || op == nir_op_isub_sat || op == nir_op_uadd_sat ||
              op == nir_op_uadd_carry) {
      if (op == nir_op_isub_sat)
         lowered_dst = nir_isub(bld, srcs[0], srcs[1]);
      else
         lowered_dst = nir_iadd(bld, srcs[0], srcs[1]);

      /* The add_sat and sub_sat instructions need to clamp the result to the
       * range of the original type.
       */
      if (op == nir_op_iadd_sat || op == nir_op_isub_sat) {
         const int64_t int_max = u_intN_max(dst_bit_size);
         const int64_t int_min = u_intN_min(dst_bit_size);

         lowered_dst = nir_iclamp(bld, lowered_dst,
                                  nir_imm_intN_t(bld, int_min, bit_size),
                                  nir_imm_intN_t(bld, int_max, bit_size));
      } else if (op == nir_op_uadd_sat) {
         const uint64_t uint_max = u_uintN_max(dst_bit_size);

         lowered_dst = nir_umin(bld, lowered_dst,
                                nir_imm_intN_t(bld, uint_max, bit_size));
      } else {
         assert(op == nir_op_uadd_carry);
         lowered_dst = nir_ushr_imm(bld, lowered_dst, dst_bit_size);
      }
   } else {
      lowered_dst = nir_build_alu_src_arr(bld, op, srcs);
   }


   /* Convert result back to the original bit-size */
   if (nir_alu_type_get_type_size(nir_op_infos[op].output_type) == 0 &&
       dst_bit_size != bit_size) {
      nir_alu_type type = nir_op_infos[op].output_type;
      nir_ssa_def *dst = nir_convert_to_bit_size(bld, lowered_dst, type, dst_bit_size);
      nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, dst);
   } else {
      nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, lowered_dst);
   }
}

static void
lower_intrinsic_instr(nir_builder *b, nir_intrinsic_instr *intrin,
                      unsigned bit_size)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan: {
      assert(intrin->src[0].is_ssa && intrin->dest.is_ssa);
      const unsigned old_bit_size = intrin->dest.ssa.bit_size;
      assert(old_bit_size < bit_size);

      nir_alu_type type = nir_type_uint;
      if (nir_intrinsic_has_reduction_op(intrin))
         type = nir_op_infos[nir_intrinsic_reduction_op(intrin)].input_types[0];
      else if (intrin->intrinsic == nir_intrinsic_vote_feq)
         type = nir_type_float;

      b->cursor = nir_before_instr(&intrin->instr);
      nir_intrinsic_instr *new_intrin =
         nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intrin->instr));

      nir_ssa_def *new_src = nir_convert_to_bit_size(b, intrin->src[0].ssa,
                                                     type, bit_size);
      new_intrin->src[0] = nir_src_for_ssa(new_src);

      if (intrin->intrinsic == nir_intrinsic_vote_feq ||
          intrin->intrinsic == nir_intrinsic_vote_ieq) {
         /* These return a Boolean; it's always 1-bit */
         assert(new_intrin->dest.ssa.bit_size == 1);
      } else {
         /* These return the same bit size as the source; we need to adjust
          * the size and then we'll have to emit a down-cast.
          */
         assert(intrin->src[0].ssa->bit_size == intrin->dest.ssa.bit_size);
         new_intrin->dest.ssa.bit_size = bit_size;
      }

      nir_builder_instr_insert(b, &new_intrin->instr);

      nir_ssa_def *res = &new_intrin->dest.ssa;
      if (intrin->intrinsic == nir_intrinsic_exclusive_scan) {
         /* For exclusive scan, we have to be careful because the identity
          * value for the higher bit size may get added into the mix by
          * disabled channels.  For some cases (imin/imax in particular),
          * this value won't convert to the right identity value when we
          * down-cast so we have to clamp it.
          */
         switch (nir_intrinsic_reduction_op(intrin)) {
         case nir_op_imin: {
            int64_t int_max = (1ull << (old_bit_size - 1)) - 1;
            res = nir_imin(b, res, nir_imm_intN_t(b, int_max, bit_size));
            break;
         }
         case nir_op_imax: {
            int64_t int_min = -(int64_t)(1ull << (old_bit_size - 1));
            res = nir_imax(b, res, nir_imm_intN_t(b, int_min, bit_size));
            break;
         }
         default:
            break;
         }
      }

      if (intrin->intrinsic != nir_intrinsic_vote_feq &&
          intrin->intrinsic != nir_intrinsic_vote_ieq)
         res = nir_u2uN(b, res, old_bit_size);

      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, res);
      break;
   }

   default:
      unreachable("Unsupported instruction");
   }
}

static void
lower_phi_instr(nir_builder *b, nir_phi_instr *phi, unsigned bit_size,
                nir_phi_instr *last_phi)
{
   assert(phi->dest.is_ssa);
   unsigned old_bit_size = phi->dest.ssa.bit_size;
   assert(old_bit_size < bit_size);

   nir_foreach_phi_src(src, phi) {
      b->cursor = nir_after_block_before_jump(src->pred);
      assert(src->src.is_ssa);
      nir_ssa_def *new_src = nir_u2uN(b, src->src.ssa, bit_size);

      nir_instr_rewrite_src(&phi->instr, &src->src, nir_src_for_ssa(new_src));
   }

   phi->dest.ssa.bit_size = bit_size;

   b->cursor = nir_after_instr(&last_phi->instr);

   nir_ssa_def *new_dest = nir_u2uN(b, &phi->dest.ssa, old_bit_size);
   nir_ssa_def_rewrite_uses_after(&phi->dest.ssa, new_dest,
                                  new_dest->parent_instr);
}

static bool
lower_impl(nir_function_impl *impl,
           nir_lower_bit_size_callback callback,
           void *callback_data)
{
   nir_builder b;
   nir_builder_init(&b, impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      /* Stash this so we can rewrite phi destinations quickly. */
      nir_phi_instr *last_phi = nir_block_last_phi_instr(block);

      nir_foreach_instr_safe(instr, block) {
         unsigned lower_bit_size = callback(instr, callback_data);
         if (lower_bit_size == 0)
            continue;

         switch (instr->type) {
         case nir_instr_type_alu:
            lower_alu_instr(&b, nir_instr_as_alu(instr), lower_bit_size);
            break;

         case nir_instr_type_intrinsic:
            lower_intrinsic_instr(&b, nir_instr_as_intrinsic(instr),
                                  lower_bit_size);
            break;

         case nir_instr_type_phi:
            lower_phi_instr(&b, nir_instr_as_phi(instr),
                            lower_bit_size, last_phi);
            break;

         default:
            unreachable("Unsupported instruction type");
         }
         progress = true;
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
nir_lower_bit_size(nir_shader *shader,
                   nir_lower_bit_size_callback callback,
                   void *callback_data)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= lower_impl(function->impl, callback, callback_data);
   }

   return progress;
}

static void
split_phi(nir_builder *b, nir_phi_instr *phi)
{
   nir_phi_instr *lowered[2] = {
      nir_phi_instr_create(b->shader),
      nir_phi_instr_create(b->shader)
   };
   int num_components = phi->dest.ssa.num_components;
   assert(phi->dest.ssa.bit_size == 64);

   nir_foreach_phi_src(src, phi) {
      assert(num_components == src->src.ssa->num_components);

      b->cursor = nir_before_src(&src->src, false);

      nir_ssa_def *x = nir_unpack_64_2x32_split_x(b, src->src.ssa);
      nir_ssa_def *y = nir_unpack_64_2x32_split_y(b, src->src.ssa);

      nir_phi_instr_add_src(lowered[0], src->pred, nir_src_for_ssa(x));
      nir_phi_instr_add_src(lowered[1], src->pred, nir_src_for_ssa(y));
   }

   nir_ssa_dest_init(&lowered[0]->instr, &lowered[0]->dest,
                     num_components, 32, NULL);
   nir_ssa_dest_init(&lowered[1]->instr, &lowered[1]->dest,
                     num_components, 32, NULL);

   b->cursor = nir_before_instr(&phi->instr);
   nir_builder_instr_insert(b, &lowered[0]->instr);
   nir_builder_instr_insert(b, &lowered[1]->instr);

   b->cursor = nir_after_phis(nir_cursor_current_block(b->cursor));
   nir_ssa_def *merged = nir_pack_64_2x32_split(b, &lowered[0]->dest.ssa, &lowered[1]->dest.ssa);
   nir_ssa_def_rewrite_uses(&phi->dest.ssa, merged);
   nir_instr_remove(&phi->instr);
}

static bool
lower_64bit_phi_instr(nir_builder *b, nir_instr *instr, UNUSED void *cb_data)
{
   if (instr->type != nir_instr_type_phi)
      return false;

   nir_phi_instr *phi = nir_instr_as_phi(instr);
   assert(phi->dest.is_ssa);

   if (phi->dest.ssa.bit_size <= 32)
      return false;

   split_phi(b, phi);
   return true;
}

bool
nir_lower_64bit_phis(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_64bit_phi_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
