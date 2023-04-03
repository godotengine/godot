/*
 * Copyright © 2018 Red Hat Inc.
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
 */

#include <math.h>

#include "nir.h"
#include "nir_builtin_builder.h"

nir_ssa_def*
nir_cross3(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   unsigned yzx[3] = { 1, 2, 0 };
   unsigned zxy[3] = { 2, 0, 1 };

   return nir_ffma(b, nir_swizzle(b, x, yzx, 3),
                      nir_swizzle(b, y, zxy, 3),
                      nir_fneg(b, nir_fmul(b, nir_swizzle(b, x, zxy, 3),
                                              nir_swizzle(b, y, yzx, 3))));
}

nir_ssa_def*
nir_cross4(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *cross = nir_cross3(b, x, y);

   return nir_vec4(b,
      nir_channel(b, cross, 0),
      nir_channel(b, cross, 1),
      nir_channel(b, cross, 2),
      nir_imm_intN_t(b, 0, cross->bit_size));
}

nir_ssa_def*
nir_fast_length(nir_builder *b, nir_ssa_def *vec)
{
   return nir_fsqrt(b, nir_fdot(b, vec, vec));
}

nir_ssa_def*
nir_nextafter(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *zero = nir_imm_intN_t(b, 0, x->bit_size);
   nir_ssa_def *one = nir_imm_intN_t(b, 1, x->bit_size);

   nir_ssa_def *condeq = nir_feq(b, x, y);
   nir_ssa_def *conddir = nir_flt(b, x, y);
   nir_ssa_def *condzero = nir_feq(b, x, zero);

   uint64_t sign_mask = 1ull << (x->bit_size - 1);
   uint64_t min_abs = 1;

   if (nir_is_denorm_flush_to_zero(b->shader->info.float_controls_execution_mode, x->bit_size)) {
      switch (x->bit_size) {
      case 16:
         min_abs = 1 << 10;
         break;
      case 32:
         min_abs = 1 << 23;
         break;
      case 64:
         min_abs = 1ULL << 52;
         break;
      }

      /* Flush denorm to zero to avoid returning a denorm when condeq is true. */
      x = nir_fmul(b, x, nir_imm_floatN_t(b, 1.0, x->bit_size));
   }

   /* beware of: +/-0.0 - 1 == NaN */
   nir_ssa_def *xn =
      nir_bcsel(b,
                condzero,
                nir_imm_intN_t(b, sign_mask | min_abs, x->bit_size),
                nir_isub(b, x, one));

   /* beware of -0.0 + 1 == -0x1p-149 */
   nir_ssa_def *xp = nir_bcsel(b, condzero,
                               nir_imm_intN_t(b, min_abs, x->bit_size),
                               nir_iadd(b, x, one));

   /* nextafter can be implemented by just +/- 1 on the int value */
   nir_ssa_def *res =
      nir_bcsel(b, nir_ixor(b, conddir, nir_flt(b, x, zero)), xp, xn);

   return nir_nan_check2(b, x, y, nir_bcsel(b, condeq, x, res));
}

nir_ssa_def*
nir_normalize(nir_builder *b, nir_ssa_def *vec)
{
   if (vec->num_components == 1)
      return nir_fsign(b, vec);

   nir_ssa_def *f0 = nir_imm_floatN_t(b, 0.0, vec->bit_size);
   nir_ssa_def *f1 = nir_imm_floatN_t(b, 1.0, vec->bit_size);
   nir_ssa_def *finf = nir_imm_floatN_t(b, INFINITY, vec->bit_size);

   /* scale the input to increase precision */
   nir_ssa_def *maxc = nir_fmax_abs_vec_comp(b, vec);
   nir_ssa_def *svec = nir_fdiv(b, vec, maxc);
   /* for inf */
   nir_ssa_def *finfvec = nir_copysign(b, nir_bcsel(b, nir_feq(b, vec, finf), f1, f0), f1);

   nir_ssa_def *temp = nir_bcsel(b, nir_feq(b, maxc, finf), finfvec, svec);
   nir_ssa_def *res = nir_fmul(b, temp, nir_frsq(b, nir_fdot(b, temp, temp)));

   return nir_bcsel(b, nir_feq(b, maxc, f0), vec, res);
}

nir_ssa_def*
nir_smoothstep(nir_builder *b, nir_ssa_def *edge0, nir_ssa_def *edge1, nir_ssa_def *x)
{
   nir_ssa_def *f2 = nir_imm_floatN_t(b, 2.0, x->bit_size);
   nir_ssa_def *f3 = nir_imm_floatN_t(b, 3.0, x->bit_size);

   /* t = clamp((x - edge0) / (edge1 - edge0), 0, 1) */
   nir_ssa_def *t =
      nir_fsat(b, nir_fdiv(b, nir_fsub(b, x, edge0),
                              nir_fsub(b, edge1, edge0)));

   /* result = t * t * (3 - 2 * t) */
   return nir_fmul(b, t, nir_fmul(b, t, nir_a_minus_bc(b, f3, f2, t)));
}

nir_ssa_def*
nir_upsample(nir_builder *b, nir_ssa_def *hi, nir_ssa_def *lo)
{
   assert(lo->num_components == hi->num_components);
   assert(lo->bit_size == hi->bit_size);

   nir_ssa_def *res[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < lo->num_components; ++i) {
      nir_ssa_def *vec = nir_vec2(b, nir_channel(b, lo, i), nir_channel(b, hi, i));
      res[i] = nir_pack_bits(b, vec, vec->bit_size * 2);
   }

   return nir_vec(b, res, lo->num_components);
}

/**
 * Compute xs[0] + xs[1] + xs[2] + ... using fadd.
 */
static nir_ssa_def *
build_fsum(nir_builder *b, nir_ssa_def **xs, int terms)
{
   nir_ssa_def *accum = xs[0];

   for (int i = 1; i < terms; i++)
      accum = nir_fadd(b, accum, xs[i]);

   return accum;
}

nir_ssa_def *
nir_atan(nir_builder *b, nir_ssa_def *y_over_x)
{
   const uint32_t bit_size = y_over_x->bit_size;

   nir_ssa_def *abs_y_over_x = nir_fabs(b, y_over_x);
   nir_ssa_def *one = nir_imm_floatN_t(b, 1.0f, bit_size);

   /*
    * range-reduction, first step:
    *
    *      / y_over_x         if |y_over_x| <= 1.0;
    * x = <
    *      \ 1.0 / y_over_x   otherwise
    */
   nir_ssa_def *x = nir_fdiv(b, nir_fmin(b, abs_y_over_x, one),
                                nir_fmax(b, abs_y_over_x, one));

   /*
    * approximate atan by evaluating polynomial:
    *
    * x   * 0.9999793128310355 - x^3  * 0.3326756418091246 +
    * x^5 * 0.1938924977115610 - x^7  * 0.1173503194786851 +
    * x^9 * 0.0536813784310406 - x^11 * 0.0121323213173444
    */
   nir_ssa_def *x_2  = nir_fmul(b, x,   x);
   nir_ssa_def *x_3  = nir_fmul(b, x_2, x);
   nir_ssa_def *x_5  = nir_fmul(b, x_3, x_2);
   nir_ssa_def *x_7  = nir_fmul(b, x_5, x_2);
   nir_ssa_def *x_9  = nir_fmul(b, x_7, x_2);
   nir_ssa_def *x_11 = nir_fmul(b, x_9, x_2);

   nir_ssa_def *polynomial_terms[] = {
      nir_fmul_imm(b, x,     0.9999793128310355f),
      nir_fmul_imm(b, x_3,  -0.3326756418091246f),
      nir_fmul_imm(b, x_5,   0.1938924977115610f),
      nir_fmul_imm(b, x_7,  -0.1173503194786851f),
      nir_fmul_imm(b, x_9,   0.0536813784310406f),
      nir_fmul_imm(b, x_11, -0.0121323213173444f),
   };

   nir_ssa_def *tmp =
      build_fsum(b, polynomial_terms, ARRAY_SIZE(polynomial_terms));

   /* range-reduction fixup */
   tmp = nir_ffma(b,
                  nir_b2fN(b, nir_flt(b, one, abs_y_over_x), bit_size),
                  nir_ffma_imm12(b, tmp, -2.0f, M_PI_2),
                  tmp);

   /* sign fixup */
   nir_ssa_def *result = nir_fmul(b, tmp, nir_fsign(b, y_over_x));

   /* The fmin and fmax above will filter out NaN values.  This leads to
    * non-NaN results for NaN inputs.  Work around this by doing
    *
    *    !isnan(y_over_x) ? ... : y_over_x;
    */
   if (b->exact ||
       nir_is_float_control_signed_zero_inf_nan_preserve(b->shader->info.float_controls_execution_mode, bit_size)) {
      const bool exact = b->exact;

      b->exact = true;
      nir_ssa_def *is_not_nan = nir_feq(b, y_over_x, y_over_x);
      b->exact = exact;

      /* The extra 1.0*y_over_x ensures that subnormal results are flushed to
       * zero.
       */
      result = nir_bcsel(b, is_not_nan, result, nir_fmul_imm(b, y_over_x, 1.0));
   }

   return result;
}

nir_ssa_def *
nir_atan2(nir_builder *b, nir_ssa_def *y, nir_ssa_def *x)
{
   assert(y->bit_size == x->bit_size);
   const uint32_t bit_size = x->bit_size;

   nir_ssa_def *zero = nir_imm_floatN_t(b, 0, bit_size);
   nir_ssa_def *one = nir_imm_floatN_t(b, 1, bit_size);

   /* If we're on the left half-plane rotate the coordinates π/2 clock-wise
    * for the y=0 discontinuity to end up aligned with the vertical
    * discontinuity of atan(s/t) along t=0.  This also makes sure that we
    * don't attempt to divide by zero along the vertical line, which may give
    * unspecified results on non-GLSL 4.1-capable hardware.
    */
   nir_ssa_def *flip = nir_fge(b, zero, x);
   nir_ssa_def *s = nir_bcsel(b, flip, nir_fabs(b, x), y);
   nir_ssa_def *t = nir_bcsel(b, flip, y, nir_fabs(b, x));

   /* If the magnitude of the denominator exceeds some huge value, scale down
    * the arguments in order to prevent the reciprocal operation from flushing
    * its result to zero, which would cause precision problems, and for s
    * infinite would cause us to return a NaN instead of the correct finite
    * value.
    *
    * If fmin and fmax are respectively the smallest and largest positive
    * normalized floating point values representable by the implementation,
    * the constants below should be in agreement with:
    *
    *    huge <= 1 / fmin
    *    scale <= 1 / fmin / fmax (for |t| >= huge)
    *
    * In addition scale should be a negative power of two in order to avoid
    * loss of precision.  The values chosen below should work for most usual
    * floating point representations with at least the dynamic range of ATI's
    * 24-bit representation.
    */
   const double huge_val = bit_size >= 32 ? 1e18 : 16384;
   nir_ssa_def *huge = nir_imm_floatN_t(b,  huge_val, bit_size);
   nir_ssa_def *scale = nir_bcsel(b, nir_fge(b, nir_fabs(b, t), huge),
                                  nir_imm_floatN_t(b, 0.25, bit_size), one);
   nir_ssa_def *rcp_scaled_t = nir_frcp(b, nir_fmul(b, t, scale));
   nir_ssa_def *s_over_t = nir_fmul(b, nir_fmul(b, s, scale), rcp_scaled_t);

   /* For |x| = |y| assume tan = 1 even if infinite (i.e. pretend momentarily
    * that ∞/∞ = 1) in order to comply with the rather artificial rules
    * inherited from IEEE 754-2008, namely:
    *
    *  "atan2(±∞, −∞) is ±3π/4
    *   atan2(±∞, +∞) is ±π/4"
    *
    * Note that this is inconsistent with the rules for the neighborhood of
    * zero that are based on iterated limits:
    *
    *  "atan2(±0, −0) is ±π
    *   atan2(±0, +0) is ±0"
    *
    * but GLSL specifically allows implementations to deviate from IEEE rules
    * at (0,0), so we take that license (i.e. pretend that 0/0 = 1 here as
    * well).
    */
   nir_ssa_def *tan = nir_bcsel(b, nir_feq(b, nir_fabs(b, x), nir_fabs(b, y)),
                                one, nir_fabs(b, s_over_t));

   /* Calculate the arctangent and fix up the result if we had flipped the
    * coordinate system.
    */
   nir_ssa_def *arc =
      nir_ffma_imm1(b, nir_b2fN(b, flip, bit_size), M_PI_2, nir_atan(b, tan));

   /* Rather convoluted calculation of the sign of the result.  When x < 0 we
    * cannot use fsign because we need to be able to distinguish between
    * negative and positive zero.  We don't use bitwise arithmetic tricks for
    * consistency with the GLSL front-end.  When x >= 0 rcp_scaled_t will
    * always be non-negative so this won't be able to distinguish between
    * negative and positive zero, but we don't care because atan2 is
    * continuous along the whole positive y = 0 half-line, so it won't affect
    * the result significantly.
    */
   return nir_bcsel(b, nir_flt(b, nir_fmin(b, y, rcp_scaled_t), zero),
                    nir_fneg(b, arc), arc);
}

nir_ssa_def *
nir_get_texture_size(nir_builder *b, nir_tex_instr *tex)
{
   b->cursor = nir_before_instr(&tex->instr);

   nir_tex_instr *txs;

   unsigned num_srcs = 1; /* One for the LOD */
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_sampler_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_sampler_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle ||
          tex->src[i].src_type == nir_tex_src_sampler_handle)
         num_srcs++;
   }

   txs = nir_tex_instr_create(b->shader, num_srcs);
   txs->op = nir_texop_txs;
   txs->sampler_dim = tex->sampler_dim;
   txs->is_array = tex->is_array;
   txs->is_shadow = tex->is_shadow;
   txs->is_new_style_shadow = tex->is_new_style_shadow;
   txs->texture_index = tex->texture_index;
   txs->sampler_index = tex->sampler_index;
   txs->dest_type = nir_type_int32;

   unsigned idx = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_sampler_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_sampler_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle ||
          tex->src[i].src_type == nir_tex_src_sampler_handle) {
         nir_src_copy(&txs->src[idx].src, &tex->src[i].src, &txs->instr);
         txs->src[idx].src_type = tex->src[i].src_type;
         idx++;
      }
   }
   /* Add in an LOD because some back-ends require it */
   txs->src[idx].src = nir_src_for_ssa(nir_imm_int(b, 0));
   txs->src[idx].src_type = nir_tex_src_lod;

   nir_ssa_dest_init(&txs->instr, &txs->dest,
                     nir_tex_instr_dest_size(txs), 32, NULL);
   nir_builder_instr_insert(b, &txs->instr);

   return &txs->dest.ssa;
}

nir_ssa_def *
nir_get_texture_lod(nir_builder *b, nir_tex_instr *tex)
{
   b->cursor = nir_before_instr(&tex->instr);

   nir_tex_instr *tql;

   unsigned num_srcs = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_coord ||
          tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_sampler_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_sampler_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle ||
          tex->src[i].src_type == nir_tex_src_sampler_handle)
         num_srcs++;
   }

   tql = nir_tex_instr_create(b->shader, num_srcs);
   tql->op = nir_texop_lod;
   tql->coord_components = tex->coord_components;
   tql->sampler_dim = tex->sampler_dim;
   tql->is_array = tex->is_array;
   tql->is_shadow = tex->is_shadow;
   tql->is_new_style_shadow = tex->is_new_style_shadow;
   tql->texture_index = tex->texture_index;
   tql->sampler_index = tex->sampler_index;
   tql->dest_type = nir_type_float32;

   unsigned idx = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_coord ||
          tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_sampler_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_sampler_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle ||
          tex->src[i].src_type == nir_tex_src_sampler_handle) {
         nir_src_copy(&tql->src[idx].src, &tex->src[i].src, &tql->instr);
         tql->src[idx].src_type = tex->src[i].src_type;
         idx++;
      }
   }

   nir_ssa_dest_init(&tql->instr, &tql->dest, 2, 32, NULL);
   nir_builder_instr_insert(b, &tql->instr);

   /* The LOD is the y component of the result */
   return nir_channel(b, &tql->dest.ssa, 1);
}
