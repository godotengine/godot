/*
 * Copyright © 2015 Intel Corporation
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
 * Authors:
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#include <math.h>

#include "nir/nir_builtin_builder.h"

#include "vtn_private.h"
#include "GLSL.std.450.h"

#ifndef M_PIf
#define M_PIf   ((float) M_PI)
#endif
#ifndef M_PI_2f
#define M_PI_2f ((float) M_PI_2)
#endif
#ifndef M_PI_4f
#define M_PI_4f ((float) M_PI_4)
#endif

static nir_ssa_def *build_det(nir_builder *b, nir_ssa_def **col, unsigned cols);

/* Computes the determinate of the submatrix given by taking src and
 * removing the specified row and column.
 */
static nir_ssa_def *
build_mat_subdet(struct nir_builder *b, struct nir_ssa_def **src,
                 unsigned size, unsigned row, unsigned col)
{
   assert(row < size && col < size);
   if (size == 2) {
      return nir_channel(b, src[1 - col], 1 - row);
   } else {
      /* Swizzle to get all but the specified row */
      unsigned swiz[NIR_MAX_VEC_COMPONENTS] = {0};
      for (unsigned j = 0; j < 3; j++)
         swiz[j] = j + (j >= row);

      /* Grab all but the specified column */
      nir_ssa_def *subcol[3];
      for (unsigned j = 0; j < size; j++) {
         if (j != col) {
            subcol[j - (j > col)] = nir_swizzle(b, src[j], swiz, size - 1);
         }
      }

      return build_det(b, subcol, size - 1);
   }
}

static nir_ssa_def *
build_det(nir_builder *b, nir_ssa_def **col, unsigned size)
{
   assert(size <= 4);
   nir_ssa_def *subdet[4];
   for (unsigned i = 0; i < size; i++)
      subdet[i] = build_mat_subdet(b, col, size, i, 0);

   nir_ssa_def *prod = nir_fmul(b, col[0], nir_vec(b, subdet, size));

   nir_ssa_def *result = NULL;
   for (unsigned i = 0; i < size; i += 2) {
      nir_ssa_def *term;
      if (i + 1 < size) {
         term = nir_fsub(b, nir_channel(b, prod, i),
                            nir_channel(b, prod, i + 1));
      } else {
         term = nir_channel(b, prod, i);
      }

      result = result ? nir_fadd(b, result, term) : term;
   }

   return result;
}

static nir_ssa_def *
build_mat_det(struct vtn_builder *b, struct vtn_ssa_value *src)
{
   unsigned size = glsl_get_vector_elements(src->type);

   nir_ssa_def *cols[4];
   for (unsigned i = 0; i < size; i++)
      cols[i] = src->elems[i]->def;

   return build_det(&b->nb, cols, size);
}

static struct vtn_ssa_value *
matrix_inverse(struct vtn_builder *b, struct vtn_ssa_value *src)
{
   nir_ssa_def *adj_col[4];
   unsigned size = glsl_get_vector_elements(src->type);

   nir_ssa_def *cols[4];
   for (unsigned i = 0; i < size; i++)
      cols[i] = src->elems[i]->def;

   /* Build up an adjugate matrix */
   for (unsigned c = 0; c < size; c++) {
      nir_ssa_def *elem[4];
      for (unsigned r = 0; r < size; r++) {
         elem[r] = build_mat_subdet(&b->nb, cols, size, c, r);

         if ((r + c) % 2)
            elem[r] = nir_fneg(&b->nb, elem[r]);
      }

      adj_col[c] = nir_vec(&b->nb, elem, size);
   }

   nir_ssa_def *det_inv = nir_frcp(&b->nb, build_det(&b->nb, cols, size));

   struct vtn_ssa_value *val = vtn_create_ssa_value(b, src->type);
   for (unsigned i = 0; i < size; i++)
      val->elems[i]->def = nir_fmul(&b->nb, adj_col[i], det_inv);

   return val;
}

/**
 * Approximate asin(x) by the piecewise formula:
 * for |x| < 0.5, asin~(x) = x * (1 + x²(pS0 + x²(pS1 + x²*pS2)) / (1 + x²*qS1))
 * for |x| ≥ 0.5, asin~(x) = sign(x) * (π/2 - sqrt(1 - |x|) * (π/2 + |x|(π/4 - 1 + |x|(p0 + |x|p1))))
 *
 * The latter is correct to first order at x=0 and x=±1 regardless of the p
 * coefficients but can be made second-order correct at both ends by selecting
 * the fit coefficients appropriately.  Different p coefficients can be used
 * in the asin and acos implementation to minimize some relative error metric
 * in each case.
 */
static nir_ssa_def *
build_asin(nir_builder *b, nir_ssa_def *x, float p0, float p1, bool piecewise)
{
   if (x->bit_size == 16) {
      /* The polynomial approximation isn't precise enough to meet half-float
       * precision requirements. Alternatively, we could implement this using
       * the formula:
       *
       * asin(x) = atan2(x, sqrt(1 - x*x))
       *
       * But that is very expensive, so instead we just do the polynomial
       * approximation in 32-bit math and then we convert the result back to
       * 16-bit.
       */
      return nir_f2f16(b, build_asin(b, nir_f2f32(b, x), p0, p1, piecewise));
   }
   nir_ssa_def *one = nir_imm_floatN_t(b, 1.0f, x->bit_size);
   nir_ssa_def *half = nir_imm_floatN_t(b, 0.5f, x->bit_size);
   nir_ssa_def *abs_x = nir_fabs(b, x);

   nir_ssa_def *p0_plus_xp1 = nir_ffma_imm12(b, abs_x, p1, p0);

   nir_ssa_def *expr_tail =
      nir_ffma_imm2(b, abs_x,
                       nir_ffma_imm2(b, abs_x, p0_plus_xp1, M_PI_4f - 1.0f),
                       M_PI_2f);

   nir_ssa_def *result0 = nir_fmul(b, nir_fsign(b, x),
                      nir_a_minus_bc(b, nir_imm_floatN_t(b, M_PI_2f, x->bit_size),
                                        nir_fsqrt(b, nir_fsub(b, one, abs_x)),
                                        expr_tail));
   if (piecewise) {
      /* approximation for |x| < 0.5 */
      const float pS0 =  1.6666586697e-01f;
      const float pS1 = -4.2743422091e-02f;
      const float pS2 = -8.6563630030e-03f;
      const float qS1 = -7.0662963390e-01f;

      nir_ssa_def *x2 = nir_fmul(b, x, x);
      nir_ssa_def *p = nir_fmul(b,
                                x2,
                                nir_ffma_imm2(b, x2,
                                                 nir_ffma_imm12(b, x2, pS2, pS1),
                                                 pS0));

      nir_ssa_def *q = nir_ffma_imm1(b, x2, qS1, one);
      nir_ssa_def *result1 = nir_ffma(b, x, nir_fdiv(b, p, q), x);
      return nir_bcsel(b, nir_flt(b, abs_x, half), result1, result0);
   } else {
      return result0;
   }
}

static nir_op
vtn_nir_alu_op_for_spirv_glsl_opcode(struct vtn_builder *b,
                                     enum GLSLstd450 opcode,
                                     unsigned execution_mode,
                                     bool *exact)
{
   *exact = false;
   switch (opcode) {
   case GLSLstd450Round:         return nir_op_fround_even;
   case GLSLstd450RoundEven:     return nir_op_fround_even;
   case GLSLstd450Trunc:         return nir_op_ftrunc;
   case GLSLstd450FAbs:          return nir_op_fabs;
   case GLSLstd450SAbs:          return nir_op_iabs;
   case GLSLstd450FSign:         return nir_op_fsign;
   case GLSLstd450SSign:         return nir_op_isign;
   case GLSLstd450Floor:         return nir_op_ffloor;
   case GLSLstd450Ceil:          return nir_op_fceil;
   case GLSLstd450Fract:         return nir_op_ffract;
   case GLSLstd450Sin:           return nir_op_fsin;
   case GLSLstd450Cos:           return nir_op_fcos;
   case GLSLstd450Pow:           return nir_op_fpow;
   case GLSLstd450Exp2:          return nir_op_fexp2;
   case GLSLstd450Log2:          return nir_op_flog2;
   case GLSLstd450Sqrt:          return nir_op_fsqrt;
   case GLSLstd450InverseSqrt:   return nir_op_frsq;
   case GLSLstd450NMin:          *exact = true; return nir_op_fmin;
   case GLSLstd450FMin:          return nir_op_fmin;
   case GLSLstd450UMin:          return nir_op_umin;
   case GLSLstd450SMin:          return nir_op_imin;
   case GLSLstd450NMax:          *exact = true; return nir_op_fmax;
   case GLSLstd450FMax:          return nir_op_fmax;
   case GLSLstd450UMax:          return nir_op_umax;
   case GLSLstd450SMax:          return nir_op_imax;
   case GLSLstd450FMix:          return nir_op_flrp;
   case GLSLstd450Fma:           return nir_op_ffma;
   case GLSLstd450Ldexp:         return nir_op_ldexp;
   case GLSLstd450FindILsb:      return nir_op_find_lsb;
   case GLSLstd450FindSMsb:      return nir_op_ifind_msb;
   case GLSLstd450FindUMsb:      return nir_op_ufind_msb;

   /* Packing/Unpacking functions */
   case GLSLstd450PackSnorm4x8:     return nir_op_pack_snorm_4x8;
   case GLSLstd450PackUnorm4x8:     return nir_op_pack_unorm_4x8;
   case GLSLstd450PackSnorm2x16:    return nir_op_pack_snorm_2x16;
   case GLSLstd450PackUnorm2x16:    return nir_op_pack_unorm_2x16;
   case GLSLstd450PackHalf2x16:     return nir_op_pack_half_2x16;
   case GLSLstd450PackDouble2x32:   return nir_op_pack_64_2x32;
   case GLSLstd450UnpackSnorm4x8:   return nir_op_unpack_snorm_4x8;
   case GLSLstd450UnpackUnorm4x8:   return nir_op_unpack_unorm_4x8;
   case GLSLstd450UnpackSnorm2x16:  return nir_op_unpack_snorm_2x16;
   case GLSLstd450UnpackUnorm2x16:  return nir_op_unpack_unorm_2x16;
   case GLSLstd450UnpackHalf2x16:
      if (execution_mode & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16)
         return nir_op_unpack_half_2x16_flush_to_zero;
      else
         return nir_op_unpack_half_2x16;
   case GLSLstd450UnpackDouble2x32: return nir_op_unpack_64_2x32;

   default:
      vtn_fail("No NIR equivalent");
   }
}

#define NIR_IMM_FP(n, v) (nir_imm_floatN_t(n, v, src[0]->bit_size))

static void
handle_glsl450_alu(struct vtn_builder *b, enum GLSLstd450 entrypoint,
                   const uint32_t *w, unsigned count)
{
   struct nir_builder *nb = &b->nb;
   const struct glsl_type *dest_type = vtn_get_type(b, w[1])->type;
   struct vtn_value *dest_val = vtn_untyped_value(b, w[2]);

   bool mediump_16bit;
   switch (entrypoint) {
   case GLSLstd450PackSnorm4x8:
   case GLSLstd450PackUnorm4x8:
   case GLSLstd450PackSnorm2x16:
   case GLSLstd450PackUnorm2x16:
   case GLSLstd450PackHalf2x16:
   case GLSLstd450PackDouble2x32:
   case GLSLstd450UnpackSnorm4x8:
   case GLSLstd450UnpackUnorm4x8:
   case GLSLstd450UnpackSnorm2x16:
   case GLSLstd450UnpackUnorm2x16:
   case GLSLstd450UnpackHalf2x16:
   case GLSLstd450UnpackDouble2x32:
      /* Asking for relaxed precision snorm 4x8 pack results (for example)
       * doesn't even make sense.  The NIR opcodes have a fixed output size, so
       * no trying to reduce precision.
       */
      mediump_16bit = false;
      break;

   case GLSLstd450Frexp:
   case GLSLstd450FrexpStruct:
   case GLSLstd450Modf:
   case GLSLstd450ModfStruct:
      /* Not sure how to detect the ->elems[i] destinations on these in vtn_upconvert_value(). */
      mediump_16bit = false;
      break;

   default:
      mediump_16bit = b->options->mediump_16bit_alu && vtn_value_is_relaxed_precision(b, dest_val);
      break;
   }

   /* Collect the various SSA sources */
   unsigned num_inputs = count - 5;
   nir_ssa_def *src[3] = { NULL, };
   for (unsigned i = 0; i < num_inputs; i++) {
      /* These are handled specially below */
      if (vtn_untyped_value(b, w[i + 5])->value_type == vtn_value_type_pointer)
         continue;

      src[i] = vtn_get_nir_ssa(b, w[i + 5]);
      if (mediump_16bit) {
         struct vtn_ssa_value *vtn_src = vtn_ssa_value(b, w[i + 5]);
         src[i] = vtn_mediump_downconvert(b, glsl_get_base_type(vtn_src->type), src[i]);
      }
   }

   struct vtn_ssa_value *dest = vtn_create_ssa_value(b, dest_type);

   vtn_handle_no_contraction(b, vtn_untyped_value(b, w[2]));
   switch (entrypoint) {
   case GLSLstd450Radians:
      dest->def = nir_radians(nb, src[0]);
      break;
   case GLSLstd450Degrees:
      dest->def = nir_degrees(nb, src[0]);
      break;
   case GLSLstd450Tan:
      dest->def = nir_ftan(nb, src[0]);
      break;

   case GLSLstd450Modf: {
      nir_ssa_def *inf = nir_imm_floatN_t(&b->nb, INFINITY, src[0]->bit_size);
      nir_ssa_def *sign_bit =
         nir_imm_intN_t(&b->nb, (uint64_t)1 << (src[0]->bit_size - 1),
                        src[0]->bit_size);
      nir_ssa_def *sign = nir_fsign(nb, src[0]);
      nir_ssa_def *abs = nir_fabs(nb, src[0]);

      /* NaN input should produce a NaN results, and ±Inf input should provide
       * ±0 result.  The fmul(sign(x), ffract(x)) calculation will already
       * produce the expected NaN.  To get ±0, directly compare for equality
       * with Inf instead of using fisfinite (which is false for NaN).
       */
      dest->def = nir_bcsel(nb,
                            nir_ieq(nb, abs, inf),
                            nir_iand(nb, src[0], sign_bit),
                            nir_fmul(nb, sign, nir_ffract(nb, abs)));

      struct vtn_pointer *i_ptr = vtn_value(b, w[6], vtn_value_type_pointer)->pointer;
      struct vtn_ssa_value *whole = vtn_create_ssa_value(b, i_ptr->type->type);
      whole->def = nir_fmul(nb, sign, nir_ffloor(nb, abs));
      vtn_variable_store(b, whole, i_ptr, 0);
      break;
   }

   case GLSLstd450ModfStruct: {
      nir_ssa_def *inf = nir_imm_floatN_t(&b->nb, INFINITY, src[0]->bit_size);
      nir_ssa_def *sign_bit =
         nir_imm_intN_t(&b->nb, (uint64_t)1 << (src[0]->bit_size - 1),
                        src[0]->bit_size);
      nir_ssa_def *sign = nir_fsign(nb, src[0]);
      nir_ssa_def *abs = nir_fabs(nb, src[0]);
      vtn_assert(glsl_type_is_struct_or_ifc(dest_type));

      /* See GLSLstd450Modf for explanation of the Inf and NaN handling. */
      dest->elems[0]->def = nir_bcsel(nb,
                                      nir_ieq(nb, abs, inf),
                                      nir_iand(nb, src[0], sign_bit),
                                      nir_fmul(nb, sign, nir_ffract(nb, abs)));
      dest->elems[1]->def = nir_fmul(nb, sign, nir_ffloor(nb, abs));
      break;
   }

   case GLSLstd450Step: {
      /* The SPIR-V Extended Instructions for GLSL spec says:
       *
       *    Result is 0.0 if x < edge; otherwise result is 1.0.
       *
       * Here src[1] is x, and src[0] is edge.  The direct implementation is
       *
       *    bcsel(src[1] < src[0], 0.0, 1.0)
       *
       * This is effectively b2f(!(src1 < src0)).  Previously this was
       * implemented using sge(src1, src0), but that produces incorrect
       * results for NaN.  Instead, we use the identity b2f(!x) = 1 - b2f(x).
       */
      const bool exact = nb->exact;
      nb->exact = true;

      nir_ssa_def *cmp = nir_slt(nb, src[1], src[0]);

      nb->exact = exact;
      dest->def = nir_fsub(nb, nir_imm_floatN_t(nb, 1.0f, cmp->bit_size), cmp);
      break;
   }

   case GLSLstd450Length:
      dest->def = nir_fast_length(nb, src[0]);
      break;
   case GLSLstd450Distance:
      dest->def = nir_fast_distance(nb, src[0], src[1]);
      break;
   case GLSLstd450Normalize:
      dest->def = nir_fast_normalize(nb, src[0]);
      break;

   case GLSLstd450Exp:
      dest->def = nir_fexp(nb, src[0]);
      break;

   case GLSLstd450Log:
      dest->def = nir_flog(nb, src[0]);
      break;

   case GLSLstd450FClamp:
      dest->def = nir_fclamp(nb, src[0], src[1], src[2]);
      break;
   case GLSLstd450NClamp:
      nb->exact = true;
      dest->def = nir_fclamp(nb, src[0], src[1], src[2]);
      nb->exact = false;
      break;
   case GLSLstd450UClamp:
      dest->def = nir_uclamp(nb, src[0], src[1], src[2]);
      break;
   case GLSLstd450SClamp:
      dest->def = nir_iclamp(nb, src[0], src[1], src[2]);
      break;

   case GLSLstd450Cross: {
      dest->def = nir_cross3(nb, src[0], src[1]);
      break;
   }

   case GLSLstd450SmoothStep: {
      dest->def = nir_smoothstep(nb, src[0], src[1], src[2]);
      break;
   }

   case GLSLstd450FaceForward:
      dest->def =
         nir_bcsel(nb, nir_flt(nb, nir_fdot(nb, src[2], src[1]),
                                   NIR_IMM_FP(nb, 0.0)),
                       src[0], nir_fneg(nb, src[0]));
      break;

   case GLSLstd450Reflect:
      /* I - 2 * dot(N, I) * N */
      dest->def =
         nir_a_minus_bc(nb, src[0],
                            src[1],
                            nir_fmul(nb, nir_fdot(nb, src[0], src[1]),
                                         NIR_IMM_FP(nb, 2.0)));
      break;

   case GLSLstd450Refract: {
      nir_ssa_def *I = src[0];
      nir_ssa_def *N = src[1];
      nir_ssa_def *eta = src[2];
      nir_ssa_def *n_dot_i = nir_fdot(nb, N, I);
      nir_ssa_def *one = NIR_IMM_FP(nb, 1.0);
      nir_ssa_def *zero = NIR_IMM_FP(nb, 0.0);
      /* According to the SPIR-V and GLSL specs, eta is always a float
       * regardless of the type of the other operands. However in practice it
       * seems that if you try to pass it a float then glslang will just
       * promote it to a double and generate invalid SPIR-V. In order to
       * support a hypothetical fixed version of glslang we’ll promote eta to
       * double if the other operands are double also.
       */
      if (I->bit_size != eta->bit_size) {
         eta = nir_type_convert(nb, eta, nir_type_float,
                                nir_type_float | I->bit_size,
                                nir_rounding_mode_undef);
      }
      /* k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I)) */
      nir_ssa_def *k =
         nir_a_minus_bc(nb, one, eta,
                            nir_fmul(nb, eta, nir_a_minus_bc(nb, one, n_dot_i, n_dot_i)));
      nir_ssa_def *result =
         nir_a_minus_bc(nb, nir_fmul(nb, eta, I),
                            nir_ffma(nb, eta, n_dot_i, nir_fsqrt(nb, k)),
                            N);
      /* XXX: bcsel, or if statement? */
      dest->def = nir_bcsel(nb, nir_flt(nb, k, zero), zero, result);
      break;
   }

   case GLSLstd450Sinh:
      /* 0.5 * (e^x - e^(-x)) */
      dest->def =
         nir_fmul_imm(nb, nir_fsub(nb, nir_fexp(nb, src[0]),
                                       nir_fexp(nb, nir_fneg(nb, src[0]))),
                          0.5f);
      break;

   case GLSLstd450Cosh:
      /* 0.5 * (e^x + e^(-x)) */
      dest->def =
         nir_fmul_imm(nb, nir_fadd(nb, nir_fexp(nb, src[0]),
                                       nir_fexp(nb, nir_fneg(nb, src[0]))),
                          0.5f);
      break;

   case GLSLstd450Tanh: {
      /* tanh(x) := (e^x - e^(-x)) / (e^x + e^(-x))
       *
       * We clamp x to [-10, +10] to avoid precision problems.  When x > 10,
       * e^x dominates the sum, e^(-x) is lost and tanh(x) is 1.0 for 32 bit
       * floating point.
       *
       * For 16-bit precision this we clamp x to [-4.2, +4.2].
       */
      const uint32_t bit_size = src[0]->bit_size;
      const double clamped_x = bit_size > 16 ? 10.0 : 4.2;
      nir_ssa_def *x = nir_fclamp(nb, src[0],
                                  nir_imm_floatN_t(nb, -clamped_x, bit_size),
                                  nir_imm_floatN_t(nb, clamped_x, bit_size));

      /* The clamping will filter out NaN values causing an incorrect result.
       * The comparison is carefully structured to get NaN result for NaN and
       * get -0 for -0.
       *
       *    result = abs(s) > 0.0 ? ... : s;
       */
      const bool exact = nb->exact;

      nb->exact = true;
      nir_ssa_def *is_regular = nir_flt(nb,
                                        nir_imm_floatN_t(nb, 0, bit_size),
                                        nir_fabs(nb, src[0]));

      /* The extra 1.0*s ensures that subnormal inputs are flushed to zero
       * when that is selected by the shader.
       */
      nir_ssa_def *flushed = nir_fmul(nb,
                                      src[0],
                                      nir_imm_floatN_t(nb, 1.0, bit_size));
      nb->exact = exact;

      dest->def = nir_bcsel(nb,
                            is_regular,
                            nir_fdiv(nb, nir_fsub(nb, nir_fexp(nb, x),
                                                  nir_fexp(nb, nir_fneg(nb, x))),
                                     nir_fadd(nb, nir_fexp(nb, x),
                                              nir_fexp(nb, nir_fneg(nb, x)))),
                            flushed);
      break;
   }

   case GLSLstd450Asinh:
      dest->def = nir_fmul(nb, nir_fsign(nb, src[0]),
         nir_flog(nb, nir_fadd(nb, nir_fabs(nb, src[0]),
                      nir_fsqrt(nb, nir_ffma_imm2(nb, src[0], src[0], 1.0f)))));
      break;
   case GLSLstd450Acosh:
      dest->def = nir_flog(nb, nir_fadd(nb, src[0],
         nir_fsqrt(nb, nir_ffma_imm2(nb, src[0], src[0], -1.0f))));
      break;
   case GLSLstd450Atanh: {
      nir_ssa_def *one = nir_imm_floatN_t(nb, 1.0, src[0]->bit_size);
      dest->def =
         nir_fmul_imm(nb, nir_flog(nb, nir_fdiv(nb, nir_fadd(nb, src[0], one),
                                       nir_fsub(nb, one, src[0]))),
                          0.5f);
      break;
   }

   case GLSLstd450Asin:
      dest->def = build_asin(nb, src[0], 0.086566724, -0.03102955, true);
      break;

   case GLSLstd450Acos:
      dest->def =
         nir_fsub(nb, nir_imm_floatN_t(nb, M_PI_2f, src[0]->bit_size),
                      build_asin(nb, src[0], 0.08132463, -0.02363318, false));
      break;

   case GLSLstd450Atan:
      dest->def = nir_atan(nb, src[0]);
      break;

   case GLSLstd450Atan2:
      dest->def = nir_atan2(nb, src[0], src[1]);
      break;

   case GLSLstd450Frexp: {
      dest->def = nir_frexp_sig(nb, src[0]);

      struct vtn_pointer *i_ptr = vtn_value(b, w[6], vtn_value_type_pointer)->pointer;
      struct vtn_ssa_value *exp = vtn_create_ssa_value(b, i_ptr->type->type);
      exp->def = nir_frexp_exp(nb, src[0]);
      vtn_variable_store(b, exp, i_ptr, 0);
      break;
   }

   case GLSLstd450FrexpStruct: {
      vtn_assert(glsl_type_is_struct_or_ifc(dest_type));
      dest->elems[0]->def = nir_frexp_sig(nb, src[0]);
      dest->elems[1]->def = nir_frexp_exp(nb, src[0]);
      break;
   }

   default: {
      unsigned execution_mode =
         b->shader->info.float_controls_execution_mode;
      bool exact;
      nir_op op = vtn_nir_alu_op_for_spirv_glsl_opcode(b, entrypoint, execution_mode, &exact);
      /* don't override explicit decoration */
      b->nb.exact |= exact;
      dest->def = nir_build_alu(&b->nb, op, src[0], src[1], src[2], NULL);
      break;
   }
   }
   b->nb.exact = false;

   if (mediump_16bit)
      vtn_mediump_upconvert_value(b, dest);

   vtn_push_ssa_value(b, w[2], dest);
}

static void
handle_glsl450_interpolation(struct vtn_builder *b, enum GLSLstd450 opcode,
                             const uint32_t *w, unsigned count)
{
   nir_intrinsic_op op;
   switch (opcode) {
   case GLSLstd450InterpolateAtCentroid:
      op = nir_intrinsic_interp_deref_at_centroid;
      break;
   case GLSLstd450InterpolateAtSample:
      op = nir_intrinsic_interp_deref_at_sample;
      break;
   case GLSLstd450InterpolateAtOffset:
      op = nir_intrinsic_interp_deref_at_offset;
      break;
   default:
      vtn_fail("Invalid opcode");
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

   switch (opcode) {
   case GLSLstd450InterpolateAtCentroid:
      break;
   case GLSLstd450InterpolateAtSample:
   case GLSLstd450InterpolateAtOffset:
      intrin->src[1] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[6]));
      break;
   default:
      vtn_fail("Invalid opcode");
   }

   intrin->num_components = glsl_get_vector_elements(deref->type);
   nir_ssa_dest_init(&intrin->instr, &intrin->dest,
                     glsl_get_vector_elements(deref->type),
                     glsl_get_bit_size(deref->type), NULL);

   nir_builder_instr_insert(&b->nb, &intrin->instr);

   nir_ssa_def *def = &intrin->dest.ssa;
   if (vec_array_deref)
      def = nir_vector_extract(&b->nb, def, vec_deref->arr.index.ssa);

   vtn_push_nir_ssa(b, w[2], def);
}

bool
vtn_handle_glsl450_instruction(struct vtn_builder *b, SpvOp ext_opcode,
                               const uint32_t *w, unsigned count)
{
   switch ((enum GLSLstd450)ext_opcode) {
   case GLSLstd450Determinant: {
      vtn_push_nir_ssa(b, w[2], build_mat_det(b, vtn_ssa_value(b, w[5])));
      break;
   }

   case GLSLstd450MatrixInverse: {
      vtn_push_ssa_value(b, w[2], matrix_inverse(b, vtn_ssa_value(b, w[5])));
      break;
   }

   case GLSLstd450InterpolateAtCentroid:
   case GLSLstd450InterpolateAtSample:
   case GLSLstd450InterpolateAtOffset:
      handle_glsl450_interpolation(b, (enum GLSLstd450)ext_opcode, w, count);
      break;

   default:
      handle_glsl450_alu(b, (enum GLSLstd450)ext_opcode, w, count);
   }

   return true;
}
