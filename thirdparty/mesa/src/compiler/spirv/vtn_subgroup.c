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

#include "vtn_private.h"

static struct vtn_ssa_value *
vtn_build_subgroup_instr(struct vtn_builder *b,
                         nir_intrinsic_op nir_op,
                         struct vtn_ssa_value *src0,
                         nir_ssa_def *index,
                         unsigned const_idx0,
                         unsigned const_idx1)
{
   /* Some of the subgroup operations take an index.  SPIR-V allows this to be
    * any integer type.  To make things simpler for drivers, we only support
    * 32-bit indices.
    */
   if (index && index->bit_size != 32)
      index = nir_u2u32(&b->nb, index);

   struct vtn_ssa_value *dst = vtn_create_ssa_value(b, src0->type);

   vtn_assert(dst->type == src0->type);
   if (!glsl_type_is_vector_or_scalar(dst->type)) {
      for (unsigned i = 0; i < glsl_get_length(dst->type); i++) {
         dst->elems[0] =
            vtn_build_subgroup_instr(b, nir_op, src0->elems[i], index,
                                     const_idx0, const_idx1);
      }
      return dst;
   }

   nir_intrinsic_instr *intrin =
      nir_intrinsic_instr_create(b->nb.shader, nir_op);
   nir_ssa_dest_init_for_type(&intrin->instr, &intrin->dest,
                              dst->type, NULL);
   intrin->num_components = intrin->dest.ssa.num_components;

   intrin->src[0] = nir_src_for_ssa(src0->def);
   if (index)
      intrin->src[1] = nir_src_for_ssa(index);

   intrin->const_index[0] = const_idx0;
   intrin->const_index[1] = const_idx1;

   nir_builder_instr_insert(&b->nb, &intrin->instr);

   dst->def = &intrin->dest.ssa;

   return dst;
}

void
vtn_handle_subgroup(struct vtn_builder *b, SpvOp opcode,
                    const uint32_t *w, unsigned count)
{
   struct vtn_type *dest_type = vtn_get_type(b, w[1]);

   switch (opcode) {
   case SpvOpGroupNonUniformElect: {
      vtn_fail_if(dest_type->type != glsl_bool_type(),
                  "OpGroupNonUniformElect must return a Bool");
      nir_intrinsic_instr *elect =
         nir_intrinsic_instr_create(b->nb.shader, nir_intrinsic_elect);
      nir_ssa_dest_init_for_type(&elect->instr, &elect->dest,
                                 dest_type->type, NULL);
      nir_builder_instr_insert(&b->nb, &elect->instr);
      vtn_push_nir_ssa(b, w[2], &elect->dest.ssa);
      break;
   }

   case SpvOpGroupNonUniformBallot:
   case SpvOpSubgroupBallotKHR: {
      bool has_scope = (opcode != SpvOpSubgroupBallotKHR);
      vtn_fail_if(dest_type->type != glsl_vector_type(GLSL_TYPE_UINT, 4),
                  "OpGroupNonUniformBallot must return a uvec4");
      nir_intrinsic_instr *ballot =
         nir_intrinsic_instr_create(b->nb.shader, nir_intrinsic_ballot);
      ballot->src[0] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[3 + has_scope]));
      nir_ssa_dest_init(&ballot->instr, &ballot->dest, 4, 32, NULL);
      ballot->num_components = 4;
      nir_builder_instr_insert(&b->nb, &ballot->instr);
      vtn_push_nir_ssa(b, w[2], &ballot->dest.ssa);
      break;
   }

   case SpvOpGroupNonUniformInverseBallot: {
      /* This one is just a BallotBitfieldExtract with subgroup invocation.
       * We could add a NIR intrinsic but it's easier to just lower it on the
       * spot.
       */
      nir_intrinsic_instr *intrin =
         nir_intrinsic_instr_create(b->nb.shader,
                                    nir_intrinsic_ballot_bitfield_extract);

      intrin->src[0] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[4]));
      intrin->src[1] = nir_src_for_ssa(nir_load_subgroup_invocation(&b->nb));

      nir_ssa_dest_init_for_type(&intrin->instr, &intrin->dest,
                                 dest_type->type, NULL);
      nir_builder_instr_insert(&b->nb, &intrin->instr);

      vtn_push_nir_ssa(b, w[2], &intrin->dest.ssa);
      break;
   }

   case SpvOpGroupNonUniformBallotBitExtract:
   case SpvOpGroupNonUniformBallotBitCount:
   case SpvOpGroupNonUniformBallotFindLSB:
   case SpvOpGroupNonUniformBallotFindMSB: {
      nir_ssa_def *src0, *src1 = NULL;
      nir_intrinsic_op op;
      switch (opcode) {
      case SpvOpGroupNonUniformBallotBitExtract:
         op = nir_intrinsic_ballot_bitfield_extract;
         src0 = vtn_get_nir_ssa(b, w[4]);
         src1 = vtn_get_nir_ssa(b, w[5]);
         break;
      case SpvOpGroupNonUniformBallotBitCount:
         switch ((SpvGroupOperation)w[4]) {
         case SpvGroupOperationReduce:
            op = nir_intrinsic_ballot_bit_count_reduce;
            break;
         case SpvGroupOperationInclusiveScan:
            op = nir_intrinsic_ballot_bit_count_inclusive;
            break;
         case SpvGroupOperationExclusiveScan:
            op = nir_intrinsic_ballot_bit_count_exclusive;
            break;
         default:
            unreachable("Invalid group operation");
         }
         src0 = vtn_get_nir_ssa(b, w[5]);
         break;
      case SpvOpGroupNonUniformBallotFindLSB:
         op = nir_intrinsic_ballot_find_lsb;
         src0 = vtn_get_nir_ssa(b, w[4]);
         break;
      case SpvOpGroupNonUniformBallotFindMSB:
         op = nir_intrinsic_ballot_find_msb;
         src0 = vtn_get_nir_ssa(b, w[4]);
         break;
      default:
         unreachable("Unhandled opcode");
      }

      nir_intrinsic_instr *intrin =
         nir_intrinsic_instr_create(b->nb.shader, op);

      intrin->src[0] = nir_src_for_ssa(src0);
      if (src1)
         intrin->src[1] = nir_src_for_ssa(src1);

      nir_ssa_dest_init_for_type(&intrin->instr, &intrin->dest,
                                 dest_type->type, NULL);
      nir_builder_instr_insert(&b->nb, &intrin->instr);

      vtn_push_nir_ssa(b, w[2], &intrin->dest.ssa);
      break;
   }

   case SpvOpGroupNonUniformBroadcastFirst:
   case SpvOpSubgroupFirstInvocationKHR: {
      bool has_scope = (opcode != SpvOpSubgroupFirstInvocationKHR);
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, nir_intrinsic_read_first_invocation,
                                  vtn_ssa_value(b, w[3 + has_scope]),
                                  NULL, 0, 0));
      break;
   }

   case SpvOpGroupNonUniformBroadcast:
   case SpvOpGroupBroadcast:
   case SpvOpSubgroupReadInvocationKHR: {
      bool has_scope = (opcode != SpvOpSubgroupReadInvocationKHR);
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, nir_intrinsic_read_invocation,
                                  vtn_ssa_value(b, w[3 + has_scope]),
                                  vtn_get_nir_ssa(b, w[4 + has_scope]), 0, 0));
      break;
   }

   case SpvOpGroupNonUniformAll:
   case SpvOpGroupNonUniformAny:
   case SpvOpGroupNonUniformAllEqual:
   case SpvOpGroupAll:
   case SpvOpGroupAny:
   case SpvOpSubgroupAllKHR:
   case SpvOpSubgroupAnyKHR:
   case SpvOpSubgroupAllEqualKHR: {
      vtn_fail_if(dest_type->type != glsl_bool_type(),
                  "OpGroupNonUniform(All|Any|AllEqual) must return a bool");
      nir_intrinsic_op op;
      switch (opcode) {
      case SpvOpGroupNonUniformAll:
      case SpvOpGroupAll:
      case SpvOpSubgroupAllKHR:
         op = nir_intrinsic_vote_all;
         break;
      case SpvOpGroupNonUniformAny:
      case SpvOpGroupAny:
      case SpvOpSubgroupAnyKHR:
         op = nir_intrinsic_vote_any;
         break;
      case SpvOpSubgroupAllEqualKHR:
         op = nir_intrinsic_vote_ieq;
         break;
      case SpvOpGroupNonUniformAllEqual:
         switch (glsl_get_base_type(vtn_ssa_value(b, w[4])->type)) {
         case GLSL_TYPE_FLOAT:
         case GLSL_TYPE_FLOAT16:
         case GLSL_TYPE_DOUBLE:
            op = nir_intrinsic_vote_feq;
            break;
         case GLSL_TYPE_UINT:
         case GLSL_TYPE_INT:
         case GLSL_TYPE_UINT8:
         case GLSL_TYPE_INT8:
         case GLSL_TYPE_UINT16:
         case GLSL_TYPE_INT16:
         case GLSL_TYPE_UINT64:
         case GLSL_TYPE_INT64:
         case GLSL_TYPE_BOOL:
            op = nir_intrinsic_vote_ieq;
            break;
         default:
            unreachable("Unhandled type");
         }
         break;
      default:
         unreachable("Unhandled opcode");
      }

      nir_ssa_def *src0;
      if (opcode == SpvOpGroupNonUniformAll || opcode == SpvOpGroupAll ||
          opcode == SpvOpGroupNonUniformAny || opcode == SpvOpGroupAny ||
          opcode == SpvOpGroupNonUniformAllEqual) {
         src0 = vtn_get_nir_ssa(b, w[4]);
      } else {
         src0 = vtn_get_nir_ssa(b, w[3]);
      }
      nir_intrinsic_instr *intrin =
         nir_intrinsic_instr_create(b->nb.shader, op);
      if (nir_intrinsic_infos[op].src_components[0] == 0)
         intrin->num_components = src0->num_components;
      intrin->src[0] = nir_src_for_ssa(src0);
      nir_ssa_dest_init_for_type(&intrin->instr, &intrin->dest,
                                 dest_type->type, NULL);
      nir_builder_instr_insert(&b->nb, &intrin->instr);

      vtn_push_nir_ssa(b, w[2], &intrin->dest.ssa);
      break;
   }

   case SpvOpGroupNonUniformShuffle:
   case SpvOpGroupNonUniformShuffleXor:
   case SpvOpGroupNonUniformShuffleUp:
   case SpvOpGroupNonUniformShuffleDown: {
      nir_intrinsic_op op;
      switch (opcode) {
      case SpvOpGroupNonUniformShuffle:
         op = nir_intrinsic_shuffle;
         break;
      case SpvOpGroupNonUniformShuffleXor:
         op = nir_intrinsic_shuffle_xor;
         break;
      case SpvOpGroupNonUniformShuffleUp:
         op = nir_intrinsic_shuffle_up;
         break;
      case SpvOpGroupNonUniformShuffleDown:
         op = nir_intrinsic_shuffle_down;
         break;
      default:
         unreachable("Invalid opcode");
      }
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[4]),
                                  vtn_get_nir_ssa(b, w[5]), 0, 0));
      break;
   }

   case SpvOpSubgroupShuffleINTEL:
   case SpvOpSubgroupShuffleXorINTEL: {
      nir_intrinsic_op op = opcode == SpvOpSubgroupShuffleINTEL ?
         nir_intrinsic_shuffle : nir_intrinsic_shuffle_xor;
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[3]),
                                  vtn_get_nir_ssa(b, w[4]), 0, 0));
      break;
   }

   case SpvOpSubgroupShuffleUpINTEL:
   case SpvOpSubgroupShuffleDownINTEL: {
      /* TODO: Move this lower on the compiler stack, where we can move the
       * current/other data to adjacent registers to avoid doing a shuffle
       * twice.
       */

      nir_builder *nb = &b->nb;
      nir_ssa_def *size = nir_load_subgroup_size(nb);
      nir_ssa_def *delta = vtn_get_nir_ssa(b, w[5]);

      /* Rewrite UP in terms of DOWN.
       *
       *   UP(a, b, delta) == DOWN(a, b, size - delta)
       */
      if (opcode == SpvOpSubgroupShuffleUpINTEL)
         delta = nir_isub(nb, size, delta);

      nir_ssa_def *index = nir_iadd(nb, nir_load_subgroup_invocation(nb), delta);
      struct vtn_ssa_value *current =
         vtn_build_subgroup_instr(b, nir_intrinsic_shuffle, vtn_ssa_value(b, w[3]),
                                  index, 0, 0);

      struct vtn_ssa_value *next =
         vtn_build_subgroup_instr(b, nir_intrinsic_shuffle, vtn_ssa_value(b, w[4]),
                                  nir_isub(nb, index, size), 0, 0);

      nir_ssa_def *cond = nir_ilt(nb, index, size);
      vtn_push_nir_ssa(b, w[2], nir_bcsel(nb, cond, current->def, next->def));

      break;
   }

   case SpvOpGroupNonUniformQuadBroadcast:
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, nir_intrinsic_quad_broadcast,
                                  vtn_ssa_value(b, w[4]),
                                  vtn_get_nir_ssa(b, w[5]), 0, 0));
      break;

   case SpvOpGroupNonUniformQuadSwap: {
      unsigned direction = vtn_constant_uint(b, w[5]);
      nir_intrinsic_op op;
      switch (direction) {
      case 0:
         op = nir_intrinsic_quad_swap_horizontal;
         break;
      case 1:
         op = nir_intrinsic_quad_swap_vertical;
         break;
      case 2:
         op = nir_intrinsic_quad_swap_diagonal;
         break;
      default:
         vtn_fail("Invalid constant value in OpGroupNonUniformQuadSwap");
      }
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[4]), NULL, 0, 0));
      break;
   }

   case SpvOpGroupNonUniformIAdd:
   case SpvOpGroupNonUniformFAdd:
   case SpvOpGroupNonUniformIMul:
   case SpvOpGroupNonUniformFMul:
   case SpvOpGroupNonUniformSMin:
   case SpvOpGroupNonUniformUMin:
   case SpvOpGroupNonUniformFMin:
   case SpvOpGroupNonUniformSMax:
   case SpvOpGroupNonUniformUMax:
   case SpvOpGroupNonUniformFMax:
   case SpvOpGroupNonUniformBitwiseAnd:
   case SpvOpGroupNonUniformBitwiseOr:
   case SpvOpGroupNonUniformBitwiseXor:
   case SpvOpGroupNonUniformLogicalAnd:
   case SpvOpGroupNonUniformLogicalOr:
   case SpvOpGroupNonUniformLogicalXor:
   case SpvOpGroupIAdd:
   case SpvOpGroupFAdd:
   case SpvOpGroupFMin:
   case SpvOpGroupUMin:
   case SpvOpGroupSMin:
   case SpvOpGroupFMax:
   case SpvOpGroupUMax:
   case SpvOpGroupSMax:
   case SpvOpGroupIAddNonUniformAMD:
   case SpvOpGroupFAddNonUniformAMD:
   case SpvOpGroupFMinNonUniformAMD:
   case SpvOpGroupUMinNonUniformAMD:
   case SpvOpGroupSMinNonUniformAMD:
   case SpvOpGroupFMaxNonUniformAMD:
   case SpvOpGroupUMaxNonUniformAMD:
   case SpvOpGroupSMaxNonUniformAMD: {
      nir_op reduction_op;
      switch (opcode) {
      case SpvOpGroupNonUniformIAdd:
      case SpvOpGroupIAdd:
      case SpvOpGroupIAddNonUniformAMD:
         reduction_op = nir_op_iadd;
         break;
      case SpvOpGroupNonUniformFAdd:
      case SpvOpGroupFAdd:
      case SpvOpGroupFAddNonUniformAMD:
         reduction_op = nir_op_fadd;
         break;
      case SpvOpGroupNonUniformIMul:
         reduction_op = nir_op_imul;
         break;
      case SpvOpGroupNonUniformFMul:
         reduction_op = nir_op_fmul;
         break;
      case SpvOpGroupNonUniformSMin:
      case SpvOpGroupSMin:
      case SpvOpGroupSMinNonUniformAMD:
         reduction_op = nir_op_imin;
         break;
      case SpvOpGroupNonUniformUMin:
      case SpvOpGroupUMin:
      case SpvOpGroupUMinNonUniformAMD:
         reduction_op = nir_op_umin;
         break;
      case SpvOpGroupNonUniformFMin:
      case SpvOpGroupFMin:
      case SpvOpGroupFMinNonUniformAMD:
         reduction_op = nir_op_fmin;
         break;
      case SpvOpGroupNonUniformSMax:
      case SpvOpGroupSMax:
      case SpvOpGroupSMaxNonUniformAMD:
         reduction_op = nir_op_imax;
         break;
      case SpvOpGroupNonUniformUMax:
      case SpvOpGroupUMax:
      case SpvOpGroupUMaxNonUniformAMD:
         reduction_op = nir_op_umax;
         break;
      case SpvOpGroupNonUniformFMax:
      case SpvOpGroupFMax:
      case SpvOpGroupFMaxNonUniformAMD:
         reduction_op = nir_op_fmax;
         break;
      case SpvOpGroupNonUniformBitwiseAnd:
      case SpvOpGroupNonUniformLogicalAnd:
         reduction_op = nir_op_iand;
         break;
      case SpvOpGroupNonUniformBitwiseOr:
      case SpvOpGroupNonUniformLogicalOr:
         reduction_op = nir_op_ior;
         break;
      case SpvOpGroupNonUniformBitwiseXor:
      case SpvOpGroupNonUniformLogicalXor:
         reduction_op = nir_op_ixor;
         break;
      default:
         unreachable("Invalid reduction operation");
      }

      nir_intrinsic_op op;
      unsigned cluster_size = 0;
      switch ((SpvGroupOperation)w[4]) {
      case SpvGroupOperationReduce:
         op = nir_intrinsic_reduce;
         break;
      case SpvGroupOperationInclusiveScan:
         op = nir_intrinsic_inclusive_scan;
         break;
      case SpvGroupOperationExclusiveScan:
         op = nir_intrinsic_exclusive_scan;
         break;
      case SpvGroupOperationClusteredReduce:
         op = nir_intrinsic_reduce;
         assert(count == 7);
         cluster_size = vtn_constant_uint(b, w[6]);
         break;
      default:
         unreachable("Invalid group operation");
      }

      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[5]), NULL,
                                  reduction_op, cluster_size));
      break;
   }

   default:
      unreachable("Invalid SPIR-V opcode");
   }
}
