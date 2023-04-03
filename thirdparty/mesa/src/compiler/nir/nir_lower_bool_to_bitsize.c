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

#include "nir.h"
#include "nir_builder.h"

static bool
assert_ssa_def_is_not_1bit(nir_ssa_def *def, UNUSED void *unused)
{
   assert(def->bit_size > 1);
   return true;
}

static bool
rewrite_1bit_ssa_def_to_32bit(nir_ssa_def *def, void *_progress)
{
   bool *progress = _progress;
   if (def->bit_size == 1) {
      def->bit_size = 32;
      *progress = true;
   }
   return true;
}

static uint32_t
get_bool_convert_opcode(uint32_t dst_bit_size)
{
   switch (dst_bit_size) {
   case 32: return nir_op_i2i32;
   case 16: return nir_op_i2i16;
   case 8:  return nir_op_i2i8;
   default:
      unreachable("invalid boolean bit-size");
   }
}

static void
make_sources_canonical(nir_builder *b, nir_alu_instr *alu, uint32_t start_idx)
{
   /* TODO: for now we take the bit-size of the first source as the canonical
    * form but we could try to be smarter.
    */
   const nir_op_info *op_info = &nir_op_infos[alu->op];
   uint32_t bit_size = nir_src_bit_size(alu->src[start_idx].src);
   for (uint32_t i = start_idx + 1; i < op_info->num_inputs; i++) {
      if (nir_src_bit_size(alu->src[i].src) != bit_size) {
         b->cursor = nir_before_instr(&alu->instr);
         nir_op convert_op = get_bool_convert_opcode(bit_size);
         nir_ssa_def *new_src =
            nir_build_alu(b, convert_op, alu->src[i].src.ssa, NULL, NULL, NULL);
         /* Retain the write mask and swizzle of the original instruction so
          * that we don’t unnecessarily create a vectorized instruction.
          */
         nir_alu_instr *conv_instr =
            nir_instr_as_alu(nir_builder_last_instr(b));
         conv_instr->dest.write_mask = alu->dest.write_mask;
         conv_instr->dest.dest.ssa.num_components =
            alu->dest.dest.ssa.num_components;
         memcpy(conv_instr->src[0].swizzle,
                alu->src[i].swizzle,
                sizeof(conv_instr->src[0].swizzle));
         nir_instr_rewrite_src(&alu->instr,
                               &alu->src[i].src, nir_src_for_ssa(new_src));
         /* The swizzle will have been handled by the conversion instruction
          * so we can reset it back to the default
          */
         for (unsigned j = 0; j < NIR_MAX_VEC_COMPONENTS; j++)
            alu->src[i].swizzle[j] = j;
      }
   }
}

static bool
lower_alu_instr(nir_builder *b, nir_alu_instr *alu)
{
   const nir_op_info *op_info = &nir_op_infos[alu->op];

   /* For operations that can take multiple boolean sources we need to ensure
    * that all booleans have the same bit-size
    */
   switch (alu->op) {
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec5:
   case nir_op_vec8:
   case nir_op_vec16:
   case nir_op_inot:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
      if (nir_dest_bit_size(alu->dest.dest) > 1)
         return false; /* Not a boolean instruction */
      FALLTHROUGH;

   case nir_op_ball_fequal2:
   case nir_op_ball_fequal3:
   case nir_op_ball_fequal4:
   case nir_op_bany_fnequal2:
   case nir_op_bany_fnequal3:
   case nir_op_bany_fnequal4:
   case nir_op_ball_iequal2:
   case nir_op_ball_iequal3:
   case nir_op_ball_iequal4:
   case nir_op_bany_inequal2:
   case nir_op_bany_inequal3:
   case nir_op_bany_inequal4:
   case nir_op_ieq:
   case nir_op_ine:
      make_sources_canonical(b, alu, 0);
      break;

   case nir_op_bcsel:
      /* bcsel may be choosing between boolean sources too */
      if (nir_dest_bit_size(alu->dest.dest) == 1)
         make_sources_canonical(b, alu, 1);
      break;

   default:
      break;
   }

   /* Now that we have a canonical boolean bit-size, go on and rewrite the
    * instruction to match the canonical bit-size.
    */
   uint32_t bit_size = nir_src_bit_size(alu->src[0].src);
   assert(bit_size > 1);

   nir_op opcode = alu->op;
   switch (opcode) {
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec5:
   case nir_op_vec8:
   case nir_op_vec16:
   case nir_op_inot:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
      /* Nothing to do here, we do not specialize these opcodes by bit-size */
      break;

   case nir_op_f2b1:
      opcode = bit_size == 8 ? nir_op_f2b8 :
                               bit_size == 16 ? nir_op_f2b16 : nir_op_f2b32;
      break;

   case nir_op_b2b1:
      /* Since the canonical bit size is the size of the src, it's a no-op */
      opcode = nir_op_mov;
      break;

   case nir_op_b2b32:
      /* For up-converting booleans, sign-extend */
      opcode = nir_op_i2i32;
      break;

   case nir_op_flt:
      opcode = bit_size == 8 ? nir_op_flt8 :
                               bit_size == 16 ? nir_op_flt16 : nir_op_flt32;
      break;

   case nir_op_fge:
      opcode = bit_size == 8 ? nir_op_fge8 :
                               bit_size == 16 ? nir_op_fge16 : nir_op_fge32;
      break;

   case nir_op_feq:
      opcode = bit_size == 8 ? nir_op_feq8 :
                               bit_size == 16 ? nir_op_feq16 : nir_op_feq32;
      break;

   case nir_op_fneu:
      opcode = bit_size == 8 ? nir_op_fneu8 :
                               bit_size == 16 ? nir_op_fneu16 : nir_op_fneu32;
      break;

   case nir_op_ilt:
      opcode = bit_size == 8 ? nir_op_ilt8 :
                               bit_size == 16 ? nir_op_ilt16 : nir_op_ilt32;
      break;

   case nir_op_ige:
      opcode = bit_size == 8 ? nir_op_ige8 :
                               bit_size == 16 ? nir_op_ige16 : nir_op_ige32;
      break;

   case nir_op_ieq:
      opcode = bit_size == 8 ? nir_op_ieq8 :
                               bit_size == 16 ? nir_op_ieq16 : nir_op_ieq32;
      break;

   case nir_op_ine:
      opcode = bit_size == 8 ? nir_op_ine8 :
                               bit_size == 16 ? nir_op_ine16 : nir_op_ine32;
      break;

   case nir_op_ult:
      opcode = bit_size == 8 ? nir_op_ult8 :
                               bit_size == 16 ? nir_op_ult16 : nir_op_ult32;
      break;

   case nir_op_uge:
      opcode = bit_size == 8 ? nir_op_uge8 :
                               bit_size == 16 ? nir_op_uge16 : nir_op_uge32;
      break;

   case nir_op_ball_fequal2:
      opcode = bit_size == 8 ? nir_op_b8all_fequal2 :
                               bit_size == 16 ? nir_op_b16all_fequal2 :
                                                nir_op_b32all_fequal2;
      break;

   case nir_op_ball_fequal3:
      opcode = bit_size == 8 ? nir_op_b8all_fequal3 :
                               bit_size == 16 ? nir_op_b16all_fequal3 :
                                                nir_op_b32all_fequal3;
      break;

   case nir_op_ball_fequal4:
      opcode = bit_size == 8 ? nir_op_b8all_fequal4 :
                               bit_size == 16 ? nir_op_b16all_fequal4 :
                                                nir_op_b32all_fequal4;
      break;

   case nir_op_bany_fnequal2:
      opcode = bit_size == 8 ? nir_op_b8any_fnequal2 :
                               bit_size == 16 ? nir_op_b16any_fnequal2 :
                                                nir_op_b32any_fnequal2;
      break;

   case nir_op_bany_fnequal3:
      opcode = bit_size == 8 ? nir_op_b8any_fnequal3 :
                               bit_size == 16 ? nir_op_b16any_fnequal3 :
                                                nir_op_b32any_fnequal3;
      break;

   case nir_op_bany_fnequal4:
      opcode = bit_size == 8 ? nir_op_b8any_fnequal4 :
                               bit_size == 16 ? nir_op_b16any_fnequal4 :
                                                nir_op_b32any_fnequal4;
      break;

   case nir_op_ball_iequal2:
      opcode = bit_size == 8 ? nir_op_b8all_iequal2 :
                               bit_size == 16 ? nir_op_b16all_iequal2 :
                                                nir_op_b32all_iequal2;
      break;

   case nir_op_ball_iequal3:
      opcode = bit_size == 8 ? nir_op_b8all_iequal3 :
                               bit_size == 16 ? nir_op_b16all_iequal3 :
                                                nir_op_b32all_iequal3;
      break;

   case nir_op_ball_iequal4:
      opcode = bit_size == 8 ? nir_op_b8all_iequal4 :
                               bit_size == 16 ? nir_op_b16all_iequal4 :
                                                nir_op_b32all_iequal4;
      break;

   case nir_op_bany_inequal2:
      opcode = bit_size == 8 ? nir_op_b8any_inequal2 :
                               bit_size == 16 ? nir_op_b16any_inequal2 :
                                                nir_op_b32any_inequal2;
      break;

   case nir_op_bany_inequal3:
      opcode = bit_size == 8 ? nir_op_b8any_inequal3 :
                               bit_size == 16 ? nir_op_b16any_inequal3 :
                                                nir_op_b32any_inequal3;
      break;

   case nir_op_bany_inequal4:
      opcode = bit_size == 8 ? nir_op_b8any_inequal4 :
                               bit_size == 16 ? nir_op_b16any_inequal4 :
                                                nir_op_b32any_inequal4;
      break;

   case nir_op_bcsel:
      opcode = bit_size == 8 ? nir_op_b8csel :
                               bit_size == 16 ? nir_op_b16csel : nir_op_b32csel;

      /* The destination of the selection may have a different bit-size from
       * the bcsel condition.
       */
      bit_size = nir_src_bit_size(alu->src[1].src);
      break;

   default:
      assert(alu->dest.dest.ssa.bit_size > 1);
      for (unsigned i = 0; i < op_info->num_inputs; i++)
         assert(alu->src[i].src.ssa->bit_size > 1);
      return false;
   }

   alu->op = opcode;

   if (alu->dest.dest.ssa.bit_size == 1)
      alu->dest.dest.ssa.bit_size = bit_size;

   return true;
}

static bool
lower_load_const_instr(nir_load_const_instr *load)
{
   bool progress = false;

   if (load->def.bit_size > 1)
      return progress;

   /* TODO: It is not clear if there is any case in which we can ever hit
    * this path, so for now we just provide a 32-bit default.
    *
    * TODO2: after some changed on nir_const_value and other on upstream, we
    * removed the initialization of a general value like this:
    *   nir_const_value value = load->value
    *
    * to initialize per value component. Need to confirm if that is correct,
    * but look at the TOO before.
    */
   for (unsigned i = 0; i < load->def.num_components; i++) {
      load->value[i].u32 = load->value[i].b ? NIR_TRUE : NIR_FALSE;
      load->def.bit_size = 32;
      progress = true;
   }

   return progress;
}

static bool
lower_phi_instr(nir_builder *b, nir_phi_instr *phi)
{
   if (nir_dest_bit_size(phi->dest) != 1)
      return false;

   /* Ensure all phi sources have a canonical bit-size. We choose the
    * bit-size of the first phi source as the canonical form.
    *
    * TODO: maybe we can be smarter about how we choose the canonical form.
    */
   uint32_t dst_bit_size = 0;
   nir_foreach_phi_src(phi_src, phi) {
      uint32_t src_bit_size = nir_src_bit_size(phi_src->src);
      if (dst_bit_size == 0) {
         dst_bit_size = src_bit_size;
      } else if (src_bit_size != dst_bit_size) {
         assert(phi_src->src.is_ssa);
         b->cursor = nir_before_src(&phi_src->src, false);
         nir_op convert_op = get_bool_convert_opcode(dst_bit_size);
         nir_ssa_def *new_src =
            nir_build_alu(b, convert_op, phi_src->src.ssa, NULL, NULL, NULL);
         nir_instr_rewrite_src(&phi->instr, &phi_src->src,
                               nir_src_for_ssa(new_src));
      }
   }

   phi->dest.ssa.bit_size = dst_bit_size;

   return true;
}

static bool
lower_tex_instr(nir_tex_instr *tex)
{
   bool progress = false;
   rewrite_1bit_ssa_def_to_32bit(&tex->dest.ssa, &progress);
   if (tex->dest_type == nir_type_bool1) {
      tex->dest_type = nir_type_bool32;
      progress = true;
   }
   return progress;
}

static bool
nir_lower_bool_to_bitsize_instr(nir_builder *b,
                                nir_instr *instr,
                                UNUSED void *cb_data)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return lower_alu_instr(b, nir_instr_as_alu(instr));

   case nir_instr_type_load_const:
      return lower_load_const_instr(nir_instr_as_load_const(instr));

   case nir_instr_type_phi:
      return lower_phi_instr(b, nir_instr_as_phi(instr));

   case nir_instr_type_ssa_undef:
   case nir_instr_type_intrinsic: {
      bool progress = false;
      nir_foreach_ssa_def(instr, rewrite_1bit_ssa_def_to_32bit, &progress);
      return progress;
   }

   case nir_instr_type_tex:
      return lower_tex_instr(nir_instr_as_tex(instr));

   default:
      nir_foreach_ssa_def(instr, assert_ssa_def_is_not_1bit, NULL);
      return false;
   }
}

bool
nir_lower_bool_to_bitsize(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, nir_lower_bool_to_bitsize_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
