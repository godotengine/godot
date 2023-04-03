/*
 * Copyright © Microsoft Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir_builder.h"

/* The following float-to-half conversion routines are based on the "half" library:
 * https://sourceforge.net/projects/half/
 *
 * half - IEEE 754-based half-precision floating-point library.
 *
 * Copyright (c) 2012-2019 Christian Rau <rauy@users.sourceforge.net>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Version 2.1.0
 */
static bool
lower_fp16_casts_filter(const nir_instr *instr, const void *data)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_f2f16:
      case nir_op_f2f16_rtne:
      case nir_op_f2f16_rtz:
         return true;
      default:
         return false;
      }
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      return intrin->intrinsic == nir_intrinsic_convert_alu_types &&
         nir_intrinsic_dest_type(intrin) == nir_type_float16;
   }
   return false;
}

static nir_ssa_def *
half_rounded(nir_builder *b, nir_ssa_def *value, nir_ssa_def *guard, nir_ssa_def *sticky,
             nir_ssa_def *sign, nir_rounding_mode mode)
{
   switch (mode) {
   case nir_rounding_mode_rtne:
      return nir_iadd(b, value, nir_iand(b, guard, nir_ior(b, sticky, value)));
   case nir_rounding_mode_ru:
      sign = nir_ushr(b, sign, nir_imm_int(b, 31));
      return nir_iadd(b, value, nir_iand(b, nir_inot(b, sign),
                                            nir_ior(b, guard, sticky)));
   case nir_rounding_mode_rd:
      sign = nir_ushr(b, sign, nir_imm_int(b, 31));
      return nir_iadd(b, value, nir_iand(b, sign,
                                            nir_ior(b, guard, sticky)));
   default:
      return value;
   }
}

static nir_ssa_def *
float_to_half_impl(nir_builder *b, nir_ssa_def *src, nir_rounding_mode mode)
{
   nir_ssa_def *f32infinity = nir_imm_int(b, 255 << 23);
   nir_ssa_def *f16max = nir_imm_int(b, (127 + 16) << 23);

   if (src->bit_size == 64)
      src = nir_f2f32(b, src);
   nir_ssa_def *sign = nir_iand(b, src, nir_imm_int(b, 0x80000000));
   nir_ssa_def *one = nir_imm_int(b, 1);

   nir_ssa_def *abs = nir_iand(b, src, nir_imm_int(b, 0x7FFFFFFF));
   /* NaN or INF. For rtne, overflow also becomes INF, so combine the comparisons */
   nir_push_if(b, nir_ige(b, abs, mode == nir_rounding_mode_rtne ? f16max : f32infinity));
   nir_ssa_def *inf_nanfp16 = nir_bcsel(b,
                                    nir_ilt(b, f32infinity, abs),
                                    nir_imm_int(b, 0x7E00),
                                    nir_imm_int(b, 0x7C00));
   nir_push_else(b, NULL);

   nir_ssa_def *overflowed_fp16 = NULL;
   if (mode != nir_rounding_mode_rtne) {
      /* Handle overflow */
      nir_push_if(b, nir_ige(b, abs, f16max));
      switch (mode) {
      case nir_rounding_mode_rtz:
         overflowed_fp16 = nir_imm_int(b, 0x7BFF);
         break;
      case nir_rounding_mode_ru:
         /* Negative becomes max float, positive becomes inf */
         overflowed_fp16 = nir_bcsel(b, nir_i2b(b, sign), nir_imm_int(b, 0x7BFF), nir_imm_int(b, 0x7C00));
         break;
      case nir_rounding_mode_rd:
         /* Negative becomes inf, positive becomes max float */
         overflowed_fp16 = nir_bcsel(b, nir_i2b(b, sign), nir_imm_int(b, 0x7C00), nir_imm_int(b, 0x7BFF));
         break;
      default: unreachable("Should've been handled already");
      }
      nir_push_else(b, NULL);
   }

   nir_push_if(b, nir_ige(b, abs, nir_imm_int(b, 113 << 23)));

   /* FP16 will be normal */
   nir_ssa_def *zero = nir_imm_int(b, 0);
   nir_ssa_def *value = nir_ior(b,
                                nir_ishl(b,
                                         nir_isub(b,
                                                  nir_ushr(b, abs, nir_imm_int(b, 23)),
                                                  nir_imm_int(b, 112)),
                                         nir_imm_int(b, 10)),
                                nir_iand(b, nir_ushr(b, abs, nir_imm_int(b, 13)), nir_imm_int(b, 0x3FFF)));
   nir_ssa_def *guard = nir_iand(b, nir_ushr(b, abs, nir_imm_int(b, 12)), one);
   nir_ssa_def *sticky = nir_bcsel(b, nir_ine(b, nir_iand(b, abs, nir_imm_int(b, 0xFFF)), zero), one, zero);
   nir_ssa_def *normal_fp16 = half_rounded(b, value, guard, sticky, sign, mode);

   nir_push_else(b, NULL);
   nir_push_if(b, nir_ige(b, abs, nir_imm_int(b, 102 << 23)));

   /* FP16 will be denormal */
   nir_ssa_def *i = nir_isub(b, nir_imm_int(b, 125), nir_ushr(b, abs, nir_imm_int(b, 23)));
   nir_ssa_def *masked = nir_ior(b, nir_iand(b, abs, nir_imm_int(b, 0x7FFFFF)), nir_imm_int(b, 0x800000));
   value = nir_ushr(b, masked, nir_iadd(b, i, one));
   guard = nir_iand(b, nir_ushr(b, masked, i), one);
   sticky = nir_bcsel(b, nir_ine(b, nir_iand(b, masked, nir_isub(b, nir_ishl(b, one, i), one)), zero), one, zero);
   nir_ssa_def *denormal_fp16 = half_rounded(b, value, guard, sticky, sign, mode);

   nir_push_else(b, NULL);

   /* Handle underflow. Nonzero values need to shift up or down for round-up or round-down */
   nir_ssa_def *underflowed_fp16 = zero;
   if (mode == nir_rounding_mode_ru ||
       mode == nir_rounding_mode_rd) {
      nir_push_if(b, nir_i2b(b, abs));

      if (mode == nir_rounding_mode_ru)
         underflowed_fp16 = nir_bcsel(b, nir_i2b(b, sign), zero, one);
      else
         underflowed_fp16 = nir_bcsel(b, nir_i2b(b, sign), one, zero);

      nir_push_else(b, NULL);
      nir_pop_if(b, NULL);
      underflowed_fp16 = nir_if_phi(b, underflowed_fp16, zero);
   }

   nir_pop_if(b, NULL);
   nir_ssa_def *underflowed_or_denorm_fp16 = nir_if_phi(b, denormal_fp16, underflowed_fp16);

   nir_pop_if(b, NULL);
   nir_ssa_def *finite_fp16 = nir_if_phi(b, normal_fp16, underflowed_or_denorm_fp16);

   nir_ssa_def *finite_or_overflowed_fp16 = finite_fp16;
   if (mode != nir_rounding_mode_rtne) {
      nir_pop_if(b, NULL);
      finite_or_overflowed_fp16 = nir_if_phi(b, overflowed_fp16, finite_fp16);
   }

   nir_pop_if(b, NULL);
   nir_ssa_def *fp16 = nir_if_phi(b, inf_nanfp16, finite_or_overflowed_fp16);

   return nir_u2u16(b, nir_ior(b, fp16, nir_ushr(b, sign, nir_imm_int(b, 16))));
}

static nir_ssa_def *
lower_fp16_cast_impl(nir_builder *b, nir_instr *instr, void *data)
{
   nir_ssa_def *src, *dst;
   uint8_t *swizzle = NULL;
   nir_rounding_mode mode = nir_rounding_mode_rtne;

   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      src = alu->src[0].src.ssa;
      swizzle = alu->src[0].swizzle;
      dst = &alu->dest.dest.ssa;
      switch (alu->op) {
      case nir_op_f2f16:
      case nir_op_f2f16_rtne:
         break;
      case nir_op_f2f16_rtz:
         mode = nir_rounding_mode_rtz;
         break;
      default: unreachable("Should've been filtered");
      }
   } else {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      assert(nir_intrinsic_src_type(intrin) == nir_type_float32);
      src = intrin->src[0].ssa;
      dst = &intrin->dest.ssa;
      mode = nir_intrinsic_rounding_mode(intrin);
   }

   nir_ssa_def *rets[NIR_MAX_VEC_COMPONENTS] = { NULL };

   for (unsigned i = 0; i < dst->num_components; i++) {
      nir_ssa_def *comp = nir_channel(b, src, swizzle ? swizzle[i] : i);
      rets[i] = float_to_half_impl(b, comp, mode);
   }

   return nir_vec(b, rets, dst->num_components);
}

bool
nir_lower_fp16_casts(nir_shader *shader)
{
   return nir_shader_lower_instructions(shader,
                                        lower_fp16_casts_filter,
                                        lower_fp16_cast_impl,
                                        NULL);
}
