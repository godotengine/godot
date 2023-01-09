/*
 * Copyright © 2018 Red Hat Inc.
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

#ifndef NIR_BUILTIN_BUILDER_H
#define NIR_BUILTIN_BUILDER_H

#include "util/u_math.h"
#include "nir_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Functions are sorted alphabetically with removed type and "fast" prefix.
 * Definitions for functions in the C file come first.
 */

nir_ssa_def* nir_cross3(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y);
nir_ssa_def* nir_cross4(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y);
nir_ssa_def* nir_fast_length(nir_builder *b, nir_ssa_def *vec);
nir_ssa_def* nir_nextafter(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y);
nir_ssa_def* nir_normalize(nir_builder *b, nir_ssa_def *vec);
nir_ssa_def* nir_smoothstep(nir_builder *b, nir_ssa_def *edge0,
                            nir_ssa_def *edge1, nir_ssa_def *x);
nir_ssa_def* nir_upsample(nir_builder *b, nir_ssa_def *hi, nir_ssa_def *lo);
nir_ssa_def* nir_atan(nir_builder *b, nir_ssa_def *y_over_x);
nir_ssa_def* nir_atan2(nir_builder *b, nir_ssa_def *y, nir_ssa_def *x);

nir_ssa_def *
nir_get_texture_lod(nir_builder *b, nir_tex_instr *tex);

nir_ssa_def *
nir_get_texture_size(nir_builder *b, nir_tex_instr *tex);

static inline nir_ssa_def *
nir_nan_check2(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *res)
{
   return nir_bcsel(b, nir_fneu(b, x, x), x, nir_bcsel(b, nir_fneu(b, y, y), y, res));
}

static inline nir_ssa_def *
nir_fmax_abs_vec_comp(nir_builder *b, nir_ssa_def *vec)
{
   nir_ssa_def *abs = nir_fabs(b, vec);
   nir_ssa_def *res = nir_channel(b, abs, 0);
   for (unsigned i = 1; i < vec->num_components; ++i)
      res = nir_fmax(b, res, nir_channel(b, abs, i));
   return res;
}

static inline nir_ssa_def *
nir_iabs_diff(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *cond = nir_ige(b, x, y);
   nir_ssa_def *res0 = nir_isub(b, x, y);
   nir_ssa_def *res1 = nir_isub(b, y, x);
   return nir_bcsel(b, cond, res0, res1);
}

static inline nir_ssa_def *
nir_uabs_diff(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *cond = nir_uge(b, x, y);
   nir_ssa_def *res0 = nir_isub(b, x, y);
   nir_ssa_def *res1 = nir_isub(b, y, x);
   return nir_bcsel(b, cond, res0, res1);
}

static inline nir_ssa_def *
nir_fexp(nir_builder *b, nir_ssa_def *x)
{
   return nir_fexp2(b, nir_fmul_imm(b, x, M_LOG2E));
}

static inline nir_ssa_def *
nir_flog(nir_builder *b, nir_ssa_def *x)
{
   return nir_fmul_imm(b, nir_flog2(b, x), 1.0 / M_LOG2E);
}

static inline nir_ssa_def *
nir_imad24(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *z)
{
   nir_ssa_def *temp = nir_imul24(b, x, y);
   return nir_iadd(b, temp, z);
}

static inline nir_ssa_def *
nir_imad_hi(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *z)
{
   nir_ssa_def *temp = nir_imul_high(b, x, y);
   return nir_iadd(b, temp, z);
}

static inline nir_ssa_def *
nir_umad_hi(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *z)
{
   nir_ssa_def *temp = nir_umul_high(b, x, y);
   return nir_iadd(b, temp, z);
}

static inline nir_ssa_def *
nir_bitselect(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *s)
{
   return nir_ior(b, nir_iand(b, nir_inot(b, s), x), nir_iand(b, s, y));
}

static inline nir_ssa_def *
nir_copysign(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   uint64_t masks = 1ull << (x->bit_size - 1);
   uint64_t maskv = ~masks;

   nir_ssa_def *s = nir_imm_intN_t(b, masks, x->bit_size);
   nir_ssa_def *v = nir_imm_intN_t(b, maskv, x->bit_size);

   return nir_ior(b, nir_iand(b, x, v), nir_iand(b, y, s));
}

static inline nir_ssa_def *
nir_degrees(nir_builder *b, nir_ssa_def *val)
{
   return nir_fmul_imm(b, val, 180.0 / M_PI);
}

static inline nir_ssa_def *
nir_fdim(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *cond = nir_flt(b, y, x);
   nir_ssa_def *res = nir_fsub(b, x, y);
   nir_ssa_def *zero = nir_imm_floatN_t(b, 0.0, x->bit_size);

   // return NaN if either x or y are NaN, else x-y if x>y, else +0.0
   return nir_nan_check2(b, x, y, nir_bcsel(b, cond, res, zero));
}

static inline nir_ssa_def *
nir_fast_distance(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   return nir_fast_length(b, nir_fsub(b, x, y));
}

static inline nir_ssa_def*
nir_fast_normalize(nir_builder *b, nir_ssa_def *vec)
{
   return nir_fdiv(b, vec, nir_fast_length(b, vec));
}

static inline nir_ssa_def*
nir_fmad(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *z)
{
   return nir_fadd(b, nir_fmul(b, x, y), z);
}

static inline nir_ssa_def*
nir_maxmag(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *xabs = nir_fabs(b, x);
   nir_ssa_def *yabs = nir_fabs(b, y);

   nir_ssa_def *condy = nir_flt(b, xabs, yabs);
   nir_ssa_def *condx = nir_flt(b, yabs, xabs);

   return nir_bcsel(b, condy, y, nir_bcsel(b, condx, x, nir_fmax(b, x, y)));
}

static inline nir_ssa_def*
nir_minmag(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y)
{
   nir_ssa_def *xabs = nir_fabs(b, x);
   nir_ssa_def *yabs = nir_fabs(b, y);

   nir_ssa_def *condx = nir_flt(b, xabs, yabs);
   nir_ssa_def *condy = nir_flt(b, yabs, xabs);

   return nir_bcsel(b, condy, y, nir_bcsel(b, condx, x, nir_fmin(b, x, y)));
}

static inline nir_ssa_def*
nir_nan(nir_builder *b, nir_ssa_def *x)
{
   nir_ssa_def *nan = nir_imm_floatN_t(b, NAN, x->bit_size);
   if (x->num_components == 1)
      return nan;

   nir_ssa_def *nans[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < x->num_components; ++i)
      nans[i] = nan;

   return nir_vec(b, nans, x->num_components);
}

static inline nir_ssa_def *
nir_radians(nir_builder *b, nir_ssa_def *val)
{
   return nir_fmul_imm(b, val, M_PI / 180.0);
}

static inline nir_ssa_def *
nir_select(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *s)
{
   if (s->num_components != 1) {
      uint64_t mask = 1ull << (s->bit_size - 1);
      s = nir_iand(b, s, nir_imm_intN_t(b, mask, s->bit_size));
   }
   return nir_bcsel(b, nir_ieq_imm(b, s, 0), x, y);
}

static inline nir_ssa_def *
nir_ftan(nir_builder *b, nir_ssa_def *x)
{
   return nir_fdiv(b, nir_fsin(b, x), nir_fcos(b, x));
}

static inline nir_ssa_def *
nir_clz_u(nir_builder *b, nir_ssa_def *a)
{
   nir_ssa_def *val;
   val = nir_isub(b, nir_imm_intN_t(b, a->bit_size - 1, 32), nir_ufind_msb(b, a));
   return nir_u2uN(b, val, a->bit_size);
}

static inline nir_ssa_def *
nir_ctz_u(nir_builder *b, nir_ssa_def *a)
{
   nir_ssa_def *cond = nir_ieq(b, a, nir_imm_intN_t(b, 0, a->bit_size));

   return nir_bcsel(b, cond,
                    nir_imm_intN_t(b, a->bit_size, a->bit_size),
                    nir_u2uN(b, nir_find_lsb(b, a), a->bit_size));
}

#ifdef __cplusplus
}
#endif

#endif /* NIR_BUILTIN_BUILDER_H */
