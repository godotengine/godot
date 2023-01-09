/*
 * Copyright © 2018 Intel Corporation
 * Copyright © 2019 Vasily Khoruzhick <anarsoul@gmail.com>
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
assert_ssa_def_is_not_int(nir_ssa_def *def, void *arg)
{
   ASSERTED BITSET_WORD *int_types = arg;
   assert(!BITSET_TEST(int_types, def->index));
   return true;
}

static bool
instr_has_only_trivial_swizzles(nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];

   for (unsigned i = 0; i < info->num_inputs; i++) {
      for (unsigned chan = 0; chan < alu->dest.dest.ssa.num_components; chan++) {
         if (alu->src[i].swizzle[chan] != chan)
            return false;
      }
   }
   return true;
}

/* Recognize the y = x - ffract(x) patterns from lowered ffloor.
 * It only works for the simple case when no swizzling is involved.
 */
static bool
check_for_lowered_ffloor(nir_alu_instr *fadd)
{
   if (!instr_has_only_trivial_swizzles(fadd))
      return false;

   nir_alu_instr *fneg = NULL;
   nir_src x;
   for (unsigned i = 0; i < 2; i++) {
      nir_alu_instr *fadd_src_alu = nir_src_as_alu_instr(fadd->src[i].src);
      if (fadd_src_alu && fadd_src_alu->op == nir_op_fneg) {
         fneg = fadd_src_alu;
         x = fadd->src[1 - i].src;
      }
   }

   if (!fneg || !instr_has_only_trivial_swizzles(fneg))
      return false;

   nir_alu_instr *ffract = nir_src_as_alu_instr(fneg->src[0].src);
   if (ffract && ffract->op == nir_op_ffract &&
       nir_srcs_equal(ffract->src[0].src, x) &&
       instr_has_only_trivial_swizzles(ffract))
      return true;

   return false;
}

static bool
lower_alu_instr(nir_builder *b, nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];

   bool is_bool_only = alu->dest.dest.ssa.bit_size == 1;
   for (unsigned i = 0; i < info->num_inputs; i++) {
      if (alu->src[i].src.ssa->bit_size != 1)
         is_bool_only = false;
   }

   if (is_bool_only) {
      /* avoid lowering integers ops are used for booleans (ieq,ine,etc) */
      return false;
   }

   b->cursor = nir_before_instr(&alu->instr);

   /* Replacement SSA value */
   nir_ssa_def *rep = NULL;
   switch (alu->op) {
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_bcsel:
      /* These we expect to have integers but the opcode doesn't change */
      break;

   case nir_op_b2i32: alu->op = nir_op_b2f32; break;
   case nir_op_i2f32: alu->op = nir_op_mov; break;
   case nir_op_u2f32: alu->op = nir_op_mov; break;

   case nir_op_f2i32: {
      alu->op = nir_op_ftrunc;

      /* If the source was already integer, then we did't need to truncate and
       * can switch it to a mov that can be copy-propagated away.
       */
      nir_alu_instr *src_alu = nir_src_as_alu_instr(alu->src[0].src);
      if (src_alu) {
         switch (src_alu->op) {
         /* Check for the y = x - ffract(x) patterns from lowered ffloor. */
         case nir_op_fadd:
            if (check_for_lowered_ffloor(src_alu))
               alu->op = nir_op_mov;
            break;
         case nir_op_fround_even:
         case nir_op_fceil:
         case nir_op_ftrunc:
         case nir_op_ffloor:
            alu->op = nir_op_mov;
            break;
         default:
            break;
         }
      }
      break;
   }

   case nir_op_f2u32: alu->op = nir_op_ffloor; break;

   case nir_op_ilt: alu->op = nir_op_flt; break;
   case nir_op_ige: alu->op = nir_op_fge; break;
   case nir_op_ieq: alu->op = nir_op_feq; break;
   case nir_op_ine: alu->op = nir_op_fneu; break;
   case nir_op_ult: alu->op = nir_op_flt; break;
   case nir_op_uge: alu->op = nir_op_fge; break;

   case nir_op_iadd: alu->op = nir_op_fadd; break;
   case nir_op_isub: alu->op = nir_op_fsub; break;
   case nir_op_imul: alu->op = nir_op_fmul; break;

   case nir_op_idiv: {
      nir_ssa_def *x = nir_ssa_for_alu_src(b, alu, 0);
      nir_ssa_def *y = nir_ssa_for_alu_src(b, alu, 1);

      /* Hand-lower fdiv, since lower_int_to_float is after nir_opt_algebraic. */
      if (b->shader->options->lower_fdiv) {
         rep = nir_ftrunc(b, nir_fmul(b, x, nir_frcp(b, y)));
      } else {
         rep = nir_ftrunc(b, nir_fdiv(b, x, y));
      }
      break;
   }

   case nir_op_iabs: alu->op = nir_op_fabs; break;
   case nir_op_ineg: alu->op = nir_op_fneg; break;
   case nir_op_imax: alu->op = nir_op_fmax; break;
   case nir_op_imin: alu->op = nir_op_fmin; break;
   case nir_op_umax: alu->op = nir_op_fmax; break;
   case nir_op_umin: alu->op = nir_op_fmin; break;

   case nir_op_ball_iequal2:  alu->op = nir_op_ball_fequal2; break;
   case nir_op_ball_iequal3:  alu->op = nir_op_ball_fequal3; break;
   case nir_op_ball_iequal4:  alu->op = nir_op_ball_fequal4; break;
   case nir_op_bany_inequal2: alu->op = nir_op_bany_fnequal2; break;
   case nir_op_bany_inequal3: alu->op = nir_op_bany_fnequal3; break;
   case nir_op_bany_inequal4: alu->op = nir_op_bany_fnequal4; break;

   case nir_op_i32csel_gt: alu->op = nir_op_fcsel_gt; break;
   case nir_op_i32csel_ge: alu->op = nir_op_fcsel_ge; break;

   default:
      assert(nir_alu_type_get_base_type(info->output_type) != nir_type_int &&
             nir_alu_type_get_base_type(info->output_type) != nir_type_uint);
      for (unsigned i = 0; i < info->num_inputs; i++) {
         assert(nir_alu_type_get_base_type(info->input_types[i]) != nir_type_int &&
                nir_alu_type_get_base_type(info->input_types[i]) != nir_type_uint);
      }
      return false;
   }

   if (rep) {
      /* We've emitted a replacement instruction */
      nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, rep);
      nir_instr_remove(&alu->instr);
   }

   return true;
}

static bool
nir_lower_int_to_float_impl(nir_function_impl *impl)
{
   bool progress = false;
   BITSET_WORD *float_types = NULL, *int_types = NULL;

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_index_ssa_defs(impl);
   float_types = calloc(BITSET_WORDS(impl->ssa_alloc),
                        sizeof(BITSET_WORD));
   int_types = calloc(BITSET_WORDS(impl->ssa_alloc),
                      sizeof(BITSET_WORD));
   nir_gather_ssa_types(impl, float_types, int_types);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
            progress |= lower_alu_instr(&b, nir_instr_as_alu(instr));
            break;

         case nir_instr_type_load_const: {
            nir_load_const_instr *load = nir_instr_as_load_const(instr);
            if (load->def.bit_size != 1 && BITSET_TEST(int_types, load->def.index)) {
               for (unsigned i = 0; i < load->def.num_components; i++)
                  load->value[i].f32 = load->value[i].i32;
            }
            break;
         }

         case nir_instr_type_intrinsic:
         case nir_instr_type_ssa_undef:
         case nir_instr_type_phi:
         case nir_instr_type_tex:
            break;

         default:
            nir_foreach_ssa_def(instr, assert_ssa_def_is_not_int, (void *)int_types);
            break;
         }
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   free(float_types);
   free(int_types);

   return progress;
}

bool
nir_lower_int_to_float(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl && nir_lower_int_to_float_impl(function->impl))
         progress = true;
   }

   return progress;
}
