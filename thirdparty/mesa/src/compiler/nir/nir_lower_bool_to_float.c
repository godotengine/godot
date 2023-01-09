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

static bool
lower_alu_instr(nir_builder *b, nir_alu_instr *alu, bool has_fcsel_ne,
                bool has_fcsel_gt)
{
   const nir_op_info *op_info = &nir_op_infos[alu->op];

   b->cursor = nir_before_instr(&alu->instr);

   /* Replacement SSA value */
   nir_ssa_def *rep = NULL;
   switch (alu->op) {
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec5:
   case nir_op_vec8:
   case nir_op_vec16:
      if (alu->dest.dest.ssa.bit_size != 1)
         return false;
      /* These we expect to have booleans but the opcode doesn't change */
      break;

   case nir_op_b2f32: alu->op = nir_op_mov; break;
   case nir_op_b2i32: alu->op = nir_op_mov; break;
   case nir_op_f2b1:
      rep = nir_sne(b, nir_ssa_for_alu_src(b, alu, 0),
                       nir_imm_float(b, 0));
      break;
   case nir_op_b2b1: alu->op = nir_op_mov; break;

   case nir_op_flt: alu->op = nir_op_slt; break;
   case nir_op_fge: alu->op = nir_op_sge; break;
   case nir_op_feq: alu->op = nir_op_seq; break;
   case nir_op_fneu: alu->op = nir_op_sne; break;
   case nir_op_ilt: alu->op = nir_op_slt; break;
   case nir_op_ige: alu->op = nir_op_sge; break;
   case nir_op_ieq: alu->op = nir_op_seq; break;
   case nir_op_ine: alu->op = nir_op_sne; break;
   case nir_op_ult: alu->op = nir_op_slt; break;
   case nir_op_uge: alu->op = nir_op_sge; break;

   case nir_op_ball_fequal2:  alu->op = nir_op_fall_equal2; break;
   case nir_op_ball_fequal3:  alu->op = nir_op_fall_equal3; break;
   case nir_op_ball_fequal4:  alu->op = nir_op_fall_equal4; break;
   case nir_op_bany_fnequal2: alu->op = nir_op_fany_nequal2; break;
   case nir_op_bany_fnequal3: alu->op = nir_op_fany_nequal3; break;
   case nir_op_bany_fnequal4: alu->op = nir_op_fany_nequal4; break;
   case nir_op_ball_iequal2:  alu->op = nir_op_fall_equal2; break;
   case nir_op_ball_iequal3:  alu->op = nir_op_fall_equal3; break;
   case nir_op_ball_iequal4:  alu->op = nir_op_fall_equal4; break;
   case nir_op_bany_inequal2: alu->op = nir_op_fany_nequal2; break;
   case nir_op_bany_inequal3: alu->op = nir_op_fany_nequal3; break;
   case nir_op_bany_inequal4: alu->op = nir_op_fany_nequal4; break;

   case nir_op_bcsel:
      if (has_fcsel_gt)
         alu->op = nir_op_fcsel_gt;
      else if (has_fcsel_ne)
         alu->op = nir_op_fcsel;
      else {
         /* Only a few pre-VS 4.0 platforms (e.g., r300 vertex shaders) should
          * hit this path.
          */
         rep = nir_flrp(b,
                        nir_ssa_for_alu_src(b, alu, 2),
                        nir_ssa_for_alu_src(b, alu, 1),
                        nir_ssa_for_alu_src(b, alu, 0));
      }

      break;

   case nir_op_iand: alu->op = nir_op_fmul; break;
   case nir_op_ixor: alu->op = nir_op_sne; break;
   case nir_op_ior: alu->op = nir_op_fmax; break;

   case nir_op_inot:
      rep = nir_seq(b, nir_ssa_for_alu_src(b, alu, 0),
                       nir_imm_float(b, 0));
      break;

   default:
      assert(alu->dest.dest.ssa.bit_size > 1);
      for (unsigned i = 0; i < op_info->num_inputs; i++)
         assert(alu->src[i].src.ssa->bit_size > 1);
      return false;
   }

   if (rep) {
      /* We've emitted a replacement instruction */
      nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, rep);
      nir_instr_remove(&alu->instr);
   } else {
      if (alu->dest.dest.ssa.bit_size == 1)
         alu->dest.dest.ssa.bit_size = 32;
   }

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

struct lower_bool_to_float_data {
   bool has_fcsel_ne;
   bool has_fcsel_gt;
};

static bool
nir_lower_bool_to_float_instr(nir_builder *b,
                              nir_instr *instr,
                              void *cb_data)
{
   struct lower_bool_to_float_data *data = cb_data;

   switch (instr->type) {
   case nir_instr_type_alu:
      return lower_alu_instr(b, nir_instr_as_alu(instr),
                             data->has_fcsel_ne, data->has_fcsel_gt);

   case nir_instr_type_load_const: {
      nir_load_const_instr *load = nir_instr_as_load_const(instr);
      if (load->def.bit_size == 1) {
         nir_const_value *value = load->value;
         for (unsigned i = 0; i < load->def.num_components; i++)
            load->value[i].f32 = value[i].b ? 1.0 : 0.0;
         load->def.bit_size = 32;
         return true;
      }
      return false;
   }

   case nir_instr_type_intrinsic:
   case nir_instr_type_ssa_undef:
   case nir_instr_type_phi: {
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
nir_lower_bool_to_float(nir_shader *shader, bool has_fcsel_ne)
{
   struct lower_bool_to_float_data data = {
      .has_fcsel_ne = has_fcsel_ne,
      .has_fcsel_gt = shader->options->has_fused_comp_and_csel
   };

   return nir_shader_instructions_pass(shader, nir_lower_bool_to_float_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       &data);
}
