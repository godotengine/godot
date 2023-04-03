/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "dxil_nir.h"

#include "nir_builder.h"
#include "nir_builtin_builder.h"

static enum pipe_format
get_input_target_format(nir_variable *var, const void *options)
{
   enum pipe_format *target_formats = (enum pipe_format *)options;
   return target_formats[var->data.driver_location];
}

static bool
lower_vs_vertex_conversion_filter(const nir_instr *instr, const void *options)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *inst = nir_instr_as_intrinsic(instr);
   if (inst->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(inst, 0);
   return (var->data.mode == nir_var_shader_in) &&
         (get_input_target_format(var, options) != PIPE_FORMAT_NONE);
}

typedef  nir_ssa_def *
(*shift_right_func)(nir_builder *build, nir_ssa_def *src0, nir_ssa_def *src1);

/* decoding the signed vs unsigned scaled format is handled
 * by applying the signed or unsigned shift right function
 * accordingly */
static nir_ssa_def *
from_10_10_10_2_scaled(nir_builder *b, nir_ssa_def *src,
                       nir_ssa_def *lshift, shift_right_func shr)
{
   nir_ssa_def *rshift = nir_imm_ivec4(b, 22, 22, 22, 30);
   return nir_i2f32(b, shr(b, nir_ishl(b, src, lshift), rshift));
}

static nir_ssa_def *
from_10_10_10_2_snorm(nir_builder *b, nir_ssa_def *src, nir_ssa_def *lshift)
{
   nir_ssa_def *split = from_10_10_10_2_scaled(b, src, lshift, nir_ishr);
   nir_ssa_def *scale_rgb = nir_imm_vec4(b,
                                         1.0f / 0x1ff,
                                         1.0f / 0x1ff,
                                         1.0f / 0x1ff,
                                         1.0f);
   return nir_fmul(b, split, scale_rgb);
}

static nir_ssa_def *
from_10_10_10_2_unorm(nir_builder *b, nir_ssa_def *src, nir_ssa_def *lshift)
{
   nir_ssa_def *split = from_10_10_10_2_scaled(b, src, lshift, nir_ushr);
   nir_ssa_def *scale_rgb = nir_imm_vec4(b,
                                         1.0f / 0x3ff,
                                         1.0f / 0x3ff,
                                         1.0f / 0x3ff,
                                         1.0f / 3.0f);
   return nir_fmul(b, split, scale_rgb);
}

inline static nir_ssa_def *
lshift_rgba(nir_builder *b)
{
   return nir_imm_ivec4(b, 22, 12, 2, 0);
}

inline static nir_ssa_def *
lshift_bgra(nir_builder *b)
{
   return nir_imm_ivec4(b, 2, 12, 22, 0);
}

static nir_ssa_def *
lower_vs_vertex_conversion_impl(nir_builder *b, nir_instr *instr, void *options)
{
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   enum pipe_format fmt = get_input_target_format(var, options);

   if (!util_format_has_alpha(fmt)) {
      /* these formats need the alpha channel replaced with 1: */
      assert(fmt == PIPE_FORMAT_R8G8B8_SINT ||
             fmt == PIPE_FORMAT_R8G8B8_UINT ||
             fmt == PIPE_FORMAT_R16G16B16_SINT ||
             fmt == PIPE_FORMAT_R16G16B16_UINT);
      if (intr->dest.ssa.num_components == 3)
         return NULL;
      return nir_vector_insert_imm(b, &intr->dest.ssa, nir_imm_int(b, 1), 3);
   } else {
      nir_ssa_def *src = nir_channel(b, &intr->dest.ssa, 0);

      switch (fmt) {
      case PIPE_FORMAT_R10G10B10A2_SNORM:
         return from_10_10_10_2_snorm(b, src, lshift_rgba(b));
      case PIPE_FORMAT_B10G10R10A2_SNORM:
         return from_10_10_10_2_snorm(b, src, lshift_bgra(b));
      case PIPE_FORMAT_B10G10R10A2_UNORM:
         return from_10_10_10_2_unorm(b, src, lshift_bgra(b));
      case PIPE_FORMAT_R10G10B10A2_SSCALED:
         return from_10_10_10_2_scaled(b, src, lshift_rgba(b), nir_ishr);
      case PIPE_FORMAT_B10G10R10A2_SSCALED:
         return from_10_10_10_2_scaled(b, src, lshift_bgra(b), nir_ishr);
      case PIPE_FORMAT_R10G10B10A2_USCALED:
         return from_10_10_10_2_scaled(b, src, lshift_rgba(b), nir_ushr);
      case PIPE_FORMAT_B10G10R10A2_USCALED:
         return from_10_10_10_2_scaled(b, src, lshift_bgra(b), nir_ushr);
      case PIPE_FORMAT_R8G8B8A8_USCALED:
      case PIPE_FORMAT_R16G16B16A16_USCALED:
         return nir_u2f32(b, &intr->dest.ssa);
      case PIPE_FORMAT_R8G8B8A8_SSCALED:
      case PIPE_FORMAT_R16G16B16A16_SSCALED:
         return nir_i2f32(b, &intr->dest.ssa);

      default:
         unreachable("Unsupported emulated vertex format");
      }
   }
}

/* Lower emulated vertex attribute input
 * The vertex attributes are passed as R32_UINT that needs to be converted
 * to one of the RGB10A2 formats that need to be emulated.
 *
 * @param target_formats contains the per attribute format to convert to
 * or PIPE_FORMAT_NONE if no conversion is needed
 */
bool
dxil_nir_lower_vs_vertex_conversion(nir_shader *s,
                                    enum pipe_format target_formats[])
{
   assert(s->info.stage == MESA_SHADER_VERTEX);

   bool result =
         nir_shader_lower_instructions(s,
                                       lower_vs_vertex_conversion_filter,
                                       lower_vs_vertex_conversion_impl,
                                       target_formats);
   return result;
}
