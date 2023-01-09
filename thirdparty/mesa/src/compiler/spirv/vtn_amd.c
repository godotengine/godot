/*
 * Copyright © 2018 Valve Corporation
 * Copyright © 2017 Red Hat
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
 */

#include "vtn_private.h"
#include "GLSL.ext.AMD.h"

bool
vtn_handle_amd_gcn_shader_instruction(struct vtn_builder *b, SpvOp ext_opcode,
                                      const uint32_t *w, unsigned count)
{
   nir_ssa_def *def;
   switch ((enum GcnShaderAMD)ext_opcode) {
   case CubeFaceIndexAMD:
      def = nir_cube_face_index_amd(&b->nb, vtn_get_nir_ssa(b, w[5]));
      break;
   case CubeFaceCoordAMD:
      def = nir_cube_face_coord_amd(&b->nb, vtn_get_nir_ssa(b, w[5]));
      break;
   case TimeAMD: {
      def = nir_pack_64_2x32(&b->nb, nir_shader_clock(&b->nb, NIR_SCOPE_SUBGROUP));
      break;
   }
   default:
      unreachable("Invalid opcode");
   }

   vtn_push_nir_ssa(b, w[2], def);

   return true;
}

bool
vtn_handle_amd_shader_ballot_instruction(struct vtn_builder *b, SpvOp ext_opcode,
                                         const uint32_t *w, unsigned count)
{
   unsigned num_args;
   nir_intrinsic_op op;
   switch ((enum ShaderBallotAMD)ext_opcode) {
   case SwizzleInvocationsAMD:
      num_args = 1;
      op = nir_intrinsic_quad_swizzle_amd;
      break;
   case SwizzleInvocationsMaskedAMD:
      num_args = 1;
      op = nir_intrinsic_masked_swizzle_amd;
      break;
   case WriteInvocationAMD:
      num_args = 3;
      op = nir_intrinsic_write_invocation_amd;
      break;
   case MbcntAMD:
      num_args = 1;
      op = nir_intrinsic_mbcnt_amd;
      break;
   default:
      unreachable("Invalid opcode");
   }

   const struct glsl_type *dest_type = vtn_get_type(b, w[1])->type;
   nir_intrinsic_instr *intrin = nir_intrinsic_instr_create(b->nb.shader, op);
   nir_ssa_dest_init_for_type(&intrin->instr, &intrin->dest, dest_type, NULL);
   if (nir_intrinsic_infos[op].src_components[0] == 0)
      intrin->num_components = intrin->dest.ssa.num_components;

   for (unsigned i = 0; i < num_args; i++)
      intrin->src[i] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[i + 5]));

   if (intrin->intrinsic == nir_intrinsic_quad_swizzle_amd) {
      struct vtn_value *val = vtn_value(b, w[6], vtn_value_type_constant);
      unsigned mask = val->constant->values[0].u32 |
                      val->constant->values[1].u32 << 2 |
                      val->constant->values[2].u32 << 4 |
                      val->constant->values[3].u32 << 6;
      nir_intrinsic_set_swizzle_mask(intrin, mask);

   } else if (intrin->intrinsic == nir_intrinsic_masked_swizzle_amd) {
      struct vtn_value *val = vtn_value(b, w[6], vtn_value_type_constant);
      unsigned mask = val->constant->values[0].u32 |
                      val->constant->values[1].u32 << 5 |
                      val->constant->values[2].u32 << 10;
      nir_intrinsic_set_swizzle_mask(intrin, mask);
   } else if (intrin->intrinsic == nir_intrinsic_mbcnt_amd) {
      /* The v_mbcnt instruction has an additional source that is added to the result.
       * This is exposed by the NIR intrinsic but not by SPIR-V, so we add zero here.
       */
      intrin->src[1] = nir_src_for_ssa(nir_imm_int(&b->nb, 0));
   }

   nir_builder_instr_insert(&b->nb, &intrin->instr);
   vtn_push_nir_ssa(b, w[2], &intrin->dest.ssa);

   return true;
}

bool
vtn_handle_amd_shader_trinary_minmax_instruction(struct vtn_builder *b, SpvOp ext_opcode,
                                                 const uint32_t *w, unsigned count)
{
   struct nir_builder *nb = &b->nb;

   unsigned num_inputs = count - 5;
   assert(num_inputs == 3);
   nir_ssa_def *src[3] = { NULL, };
   for (unsigned i = 0; i < num_inputs; i++)
      src[i] = vtn_get_nir_ssa(b, w[i + 5]);

   /* place constants at src[1-2] for easier constant-folding */
   for (unsigned i = 1; i <= 2; i++) {
      if (nir_src_as_const_value(nir_src_for_ssa(src[0]))) {
         nir_ssa_def* tmp = src[i];
         src[i] = src[0];
         src[0] = tmp;
      }
   }
   nir_ssa_def *def;
   switch ((enum ShaderTrinaryMinMaxAMD)ext_opcode) {
   case FMin3AMD:
      def = nir_fmin(nb, src[0], nir_fmin(nb, src[1], src[2]));
      break;
   case UMin3AMD:
      def = nir_umin(nb, src[0], nir_umin(nb, src[1], src[2]));
      break;
   case SMin3AMD:
      def = nir_imin(nb, src[0], nir_imin(nb, src[1], src[2]));
      break;
   case FMax3AMD:
      def = nir_fmax(nb, src[0], nir_fmax(nb, src[1], src[2]));
      break;
   case UMax3AMD:
      def = nir_umax(nb, src[0], nir_umax(nb, src[1], src[2]));
      break;
   case SMax3AMD:
      def = nir_imax(nb, src[0], nir_imax(nb, src[1], src[2]));
      break;
   case FMid3AMD:
      def = nir_fmin(nb, nir_fmax(nb, src[0], nir_fmin(nb, src[1], src[2])),
                     nir_fmax(nb, src[1], src[2]));
      break;
   case UMid3AMD:
      def = nir_umin(nb, nir_umax(nb, src[0], nir_umin(nb, src[1], src[2])),
                     nir_umax(nb, src[1], src[2]));
      break;
   case SMid3AMD:
      def = nir_imin(nb, nir_imax(nb, src[0], nir_imin(nb, src[1], src[2])),
                     nir_imax(nb, src[1], src[2]));
      break;
   default:
      unreachable("unknown opcode\n");
      break;
   }

   vtn_push_nir_ssa(b, w[2], def);

   return true;
}

bool
vtn_handle_amd_shader_explicit_vertex_parameter_instruction(struct vtn_builder *b, SpvOp ext_opcode,
                                                            const uint32_t *w, unsigned count)
{
   nir_intrinsic_op op;
   switch ((enum ShaderExplicitVertexParameterAMD)ext_opcode) {
   case InterpolateAtVertexAMD:
      op = nir_intrinsic_interp_deref_at_vertex;
      break;
   default:
      unreachable("unknown opcode");
   }

   nir_intrinsic_instr *intrin = nir_intrinsic_instr_create(b->nb.shader, op);

   struct vtn_pointer *ptr =
      vtn_value(b, w[5], vtn_value_type_pointer)->pointer;
   nir_deref_instr *deref = vtn_pointer_to_deref(b, ptr);

   /* If the value we are interpolating has an index into a vector then
    * interpolate the vector and index the result of that instead. This is
    * necessary because the index will get generated as a series of nir_bcsel
    * instructions so it would no longer be an input variable.
    */
   const bool vec_array_deref = deref->deref_type == nir_deref_type_array &&
      glsl_type_is_vector(nir_deref_instr_parent(deref)->type);

   nir_deref_instr *vec_deref = NULL;
   if (vec_array_deref) {
      vec_deref = deref;
      deref = nir_deref_instr_parent(deref);
   }
   intrin->src[0] = nir_src_for_ssa(&deref->dest.ssa);
   intrin->src[1] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[6]));

   intrin->num_components = glsl_get_vector_elements(deref->type);
   nir_ssa_dest_init(&intrin->instr, &intrin->dest,
                     glsl_get_vector_elements(deref->type),
                     glsl_get_bit_size(deref->type), NULL);

   nir_builder_instr_insert(&b->nb, &intrin->instr);

   nir_ssa_def *def;
   if (vec_array_deref) {
      assert(vec_deref);
      def = nir_vector_extract(&b->nb, &intrin->dest.ssa,
                               vec_deref->arr.index.ssa);
   } else {
      def = &intrin->dest.ssa;
   }
   vtn_push_nir_ssa(b, w[2], def);

   return true;
}
