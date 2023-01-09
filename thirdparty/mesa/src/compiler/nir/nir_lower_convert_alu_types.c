/*
 * Copyright © 2020 Intel Corporation
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
#include "nir_conversion_builder.h"

static bool
try_simplify_convert_intrin(nir_intrinsic_instr *conv)
{
   bool progress = false;

   nir_alu_type src_type = nir_intrinsic_src_type(conv);
   nir_alu_type dest_type = nir_intrinsic_dest_type(conv);

   nir_rounding_mode rounding = nir_intrinsic_rounding_mode(conv);
   nir_rounding_mode simple_rounding =
      nir_simplify_conversion_rounding(src_type, dest_type, rounding);
   if (rounding != simple_rounding) {
      nir_intrinsic_set_rounding_mode(conv, simple_rounding);
      progress = true;
   }

   if (nir_intrinsic_saturate(conv) &&
       nir_alu_type_range_contains_type_range(dest_type, src_type)) {
      nir_intrinsic_set_saturate(conv, false);
      progress = true;
   }

   return progress;
}

static void
lower_convert_alu_types_instr(nir_builder *b, nir_intrinsic_instr *conv)
{
   assert(conv->intrinsic == nir_intrinsic_convert_alu_types);
         assert(conv->src[0].is_ssa && conv->dest.is_ssa);

   b->cursor = nir_instr_remove(&conv->instr);
   nir_ssa_def *val =
      nir_convert_with_rounding(b, conv->src[0].ssa,
                                nir_intrinsic_src_type(conv),
                                nir_intrinsic_dest_type(conv),
                                nir_intrinsic_rounding_mode(conv),
                                nir_intrinsic_saturate(conv));
   nir_ssa_def_rewrite_uses(&conv->dest.ssa, val);
}

static bool
opt_simplify_convert_alu_types_impl(nir_function_impl *impl)
{
   bool progress = false;
   bool lowered_instr = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *conv = nir_instr_as_intrinsic(instr);
         if (conv->intrinsic != nir_intrinsic_convert_alu_types)
            continue;

         if (try_simplify_convert_intrin(conv))
            progress = true;

         if (nir_intrinsic_rounding_mode(conv) == nir_rounding_mode_undef &&
             !nir_intrinsic_saturate(conv)) {
            lower_convert_alu_types_instr(&b, conv);
            lowered_instr = true;
         }
      }
   }

   if (lowered_instr) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_opt_simplify_convert_alu_types(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(func, shader) {
      if (func->impl && opt_simplify_convert_alu_types_impl(func->impl))
         progress = true;
   }

   return progress;
}

static bool
lower_convert_alu_types_impl(nir_function_impl *impl,
                             bool (*should_lower)(nir_intrinsic_instr *))
{
   bool progress = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *conv = nir_instr_as_intrinsic(instr);
         if (conv->intrinsic != nir_intrinsic_convert_alu_types)
            continue;

         if (should_lower != NULL && !should_lower(conv))
            continue;

         lower_convert_alu_types_instr(&b, conv);
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
nir_lower_convert_alu_types(nir_shader *shader,
                            bool (*should_lower)(nir_intrinsic_instr *))
{
   bool progress = false;

   nir_foreach_function(func, shader) {
      if (func->impl && lower_convert_alu_types_impl(func->impl, should_lower))
         progress = true;
   }

   return progress;
}

static bool
is_constant(nir_intrinsic_instr *conv)
{
   assert(conv->intrinsic == nir_intrinsic_convert_alu_types);
   return nir_src_is_const(conv->src[0]);
}

bool
nir_lower_constant_convert_alu_types(nir_shader *shader)
{
   return nir_lower_convert_alu_types(shader, is_constant);
}

static bool
is_alu_conversion(const nir_instr *instr, UNUSED const void *_data)
{
   return instr->type == nir_instr_type_alu &&
          nir_op_infos[nir_instr_as_alu(instr)->op].is_conversion;
}

static nir_ssa_def *
lower_alu_conversion(nir_builder *b, nir_instr *instr, UNUSED void *_data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_ssa_def *src = nir_ssa_for_alu_src(b, alu, 0);
   nir_alu_type src_type = nir_op_infos[alu->op].input_types[0] | src->bit_size;
   nir_alu_type dst_type = nir_op_infos[alu->op].output_type;
   return nir_convert_alu_types(b, alu->dest.dest.ssa.bit_size, src,
                                .src_type = src_type, .dest_type = dst_type,
                                .rounding_mode = nir_rounding_mode_undef,
                                .saturate = false);
}

bool
nir_lower_alu_conversion_to_intrinsic(nir_shader *shader)
{
   return nir_shader_lower_instructions(shader, is_alu_conversion,
                                        lower_alu_conversion, NULL);
}
