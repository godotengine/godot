/*
 * Copyright © 2017 Intel Corporation
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

#include "nir_builder.h"

#include "util/format_rgb9e5.h"

static inline nir_ssa_def *
nir_shift_imm(nir_builder *b, nir_ssa_def *value, int left_shift)
{
   if (left_shift > 0)
      return nir_ishl(b, value, nir_imm_int(b, left_shift));
   else if (left_shift < 0)
      return nir_ushr(b, value, nir_imm_int(b, -left_shift));
   else
      return value;
}

static inline nir_ssa_def *
nir_shift(nir_builder *b, nir_ssa_def *value, nir_ssa_def *left_shift)
{
   return nir_bcsel(b,
                    nir_ige(b, left_shift, nir_imm_int(b, 0)),
                    nir_ishl(b, value, left_shift),
                    nir_ushr(b, value, nir_ineg(b, left_shift)));
}

static inline nir_ssa_def *
nir_mask_shift(struct nir_builder *b, nir_ssa_def *src,
               uint32_t mask, int left_shift)
{
   return nir_shift_imm(b, nir_iand(b, src, nir_imm_int(b, mask)), left_shift);
}

static inline nir_ssa_def *
nir_mask_shift_or(struct nir_builder *b, nir_ssa_def *dst, nir_ssa_def *src,
                  uint32_t src_mask, int src_left_shift)
{
   return nir_ior(b, nir_mask_shift(b, src, src_mask, src_left_shift), dst);
}

static inline nir_ssa_def *
nir_format_mask_uvec(nir_builder *b, nir_ssa_def *src, const unsigned *bits)
{
   nir_const_value mask[NIR_MAX_VEC_COMPONENTS];
   memset(mask, 0, sizeof(mask));
   for (unsigned i = 0; i < src->num_components; i++) {
      assert(bits[i] < 32);
      mask[i].u32 = (1u << bits[i]) - 1;
   }
   return nir_iand(b, src, nir_build_imm(b, src->num_components, 32, mask));
}

static inline nir_ssa_def *
nir_format_sign_extend_ivec(nir_builder *b, nir_ssa_def *src,
                            const unsigned *bits)
{
   assert(src->num_components <= 4);
   nir_ssa_def *comps[4];
   for (unsigned i = 0; i < src->num_components; i++) {
      nir_ssa_def *shift = nir_imm_int(b, src->bit_size - bits[i]);
      comps[i] = nir_ishr(b, nir_ishl(b, nir_channel(b, src, i), shift), shift);
   }
   return nir_vec(b, comps, src->num_components);
}


static inline nir_ssa_def *
nir_format_unpack_int(nir_builder *b, nir_ssa_def *packed,
                      const unsigned *bits, unsigned num_components,
                      bool sign_extend)
{
   assert(num_components >= 1 && num_components <= 4);
   const unsigned bit_size = packed->bit_size;
   nir_ssa_def *comps[4];

   if (bits[0] >= bit_size) {
      assert(bits[0] == bit_size);
      assert(num_components == 1);
      return packed;
   }

   unsigned next_chan = 0;
   unsigned offset = 0;
   for (unsigned i = 0; i < num_components; i++) {
      assert(bits[i] < bit_size);
      assert(offset + bits[i] <= bit_size);
      nir_ssa_def *chan = nir_channel(b, packed, next_chan);
      nir_ssa_def *lshift = nir_imm_int(b, bit_size - (offset + bits[i]));
      nir_ssa_def *rshift = nir_imm_int(b, bit_size - bits[i]);
      if (sign_extend)
         comps[i] = nir_ishr(b, nir_ishl(b, chan, lshift), rshift);
      else
         comps[i] = nir_ushr(b, nir_ishl(b, chan, lshift), rshift);
      offset += bits[i];
      if (offset >= bit_size) {
         next_chan++;
         offset -= bit_size;
      }
   }

   return nir_vec(b, comps, num_components);
}

static inline nir_ssa_def *
nir_format_unpack_uint(nir_builder *b, nir_ssa_def *packed,
                       const unsigned *bits, unsigned num_components)
{
   return nir_format_unpack_int(b, packed, bits, num_components, false);
}

static inline nir_ssa_def *
nir_format_unpack_sint(nir_builder *b, nir_ssa_def *packed,
                       const unsigned *bits, unsigned num_components)
{
   return nir_format_unpack_int(b, packed, bits, num_components, true);
}

static inline nir_ssa_def *
nir_format_pack_uint_unmasked(nir_builder *b, nir_ssa_def *color,
                              const unsigned *bits, unsigned num_components)
{
   assert(num_components >= 1 && num_components <= 4);
   nir_ssa_def *packed = nir_imm_int(b, 0);
   unsigned offset = 0;
   for (unsigned i = 0; i < num_components; i++) {
      packed = nir_ior(b, packed, nir_shift_imm(b, nir_channel(b, color, i),
                                               offset));
      offset += bits[i];
   }
   assert(offset <= packed->bit_size);

   return packed;
}

static inline nir_ssa_def *
nir_format_pack_uint_unmasked_ssa(nir_builder *b, nir_ssa_def *color,
                                  nir_ssa_def *bits)
{
   nir_ssa_def *packed = nir_imm_int(b, 0);
   nir_ssa_def *offset = nir_imm_int(b, 0);
   for (unsigned i = 0; i < bits->num_components; i++) {
      packed = nir_ior(b, packed, nir_ishl(b, nir_channel(b, color, i), offset));
      offset = nir_iadd(b, offset, nir_channel(b, bits, i));
   }
   return packed;
}

static inline nir_ssa_def *
nir_format_pack_uint(nir_builder *b, nir_ssa_def *color,
                     const unsigned *bits, unsigned num_components)
{
   return nir_format_pack_uint_unmasked(b, nir_format_mask_uvec(b, color, bits),
                                        bits, num_components);
}

static inline nir_ssa_def *
nir_format_bitcast_uvec_unmasked(nir_builder *b, nir_ssa_def *src,
                                 unsigned src_bits, unsigned dst_bits)
{
   assert(src->bit_size >= src_bits && src->bit_size >= dst_bits);
   assert(src_bits == 8 || src_bits == 16 || src_bits == 32);
   assert(dst_bits == 8 || dst_bits == 16 || dst_bits == 32);

   if (src_bits == dst_bits)
      return src;

   const unsigned dst_components =
      DIV_ROUND_UP(src->num_components * src_bits, dst_bits);
   assert(dst_components <= 4);

   nir_ssa_def *dst_chan[4] = {0};
   if (dst_bits > src_bits) {
      unsigned shift = 0;
      unsigned dst_idx = 0;
      for (unsigned i = 0; i < src->num_components; i++) {
         nir_ssa_def *shifted = nir_ishl(b, nir_channel(b, src, i),
                                            nir_imm_int(b, shift));
         if (shift == 0) {
            dst_chan[dst_idx] = shifted;
         } else {
            dst_chan[dst_idx] = nir_ior(b, dst_chan[dst_idx], shifted);
         }

         shift += src_bits;
         if (shift >= dst_bits) {
            dst_idx++;
            shift = 0;
         }
      }
   } else {
      nir_ssa_def *mask = nir_imm_int(b, ~0u >> (32 - dst_bits));

      unsigned src_idx = 0;
      unsigned shift = 0;
      for (unsigned i = 0; i < dst_components; i++) {
         dst_chan[i] = nir_iand(b, nir_ushr_imm(b, nir_channel(b, src, src_idx),
                                                shift),
                                   mask);
         shift += dst_bits;
         if (shift >= src_bits) {
            src_idx++;
            shift = 0;
         }
      }
   }

   return nir_vec(b, dst_chan, dst_components);
}

static inline nir_ssa_def *
_nir_format_norm_factor(nir_builder *b, const unsigned *bits,
                        unsigned num_components,
                        bool is_signed)
{
   nir_const_value factor[NIR_MAX_VEC_COMPONENTS];
   memset(factor, 0, sizeof(factor));
   for (unsigned i = 0; i < num_components; i++) {
      assert(bits[i] <= 32);
      factor[i].f32 = (1ull << (bits[i] - is_signed)) - 1;
   }
   return nir_build_imm(b, num_components, 32, factor);
}

static inline nir_ssa_def *
nir_format_unorm_to_float(nir_builder *b, nir_ssa_def *u, const unsigned *bits)
{
   nir_ssa_def *factor =
      _nir_format_norm_factor(b, bits, u->num_components, false);

   return nir_fdiv(b, nir_u2f32(b, u), factor);
}

static inline nir_ssa_def *
nir_format_snorm_to_float(nir_builder *b, nir_ssa_def *s, const unsigned *bits)
{
   nir_ssa_def *factor =
      _nir_format_norm_factor(b, bits, s->num_components, true);

   return nir_fmax(b, nir_fdiv(b, nir_i2f32(b, s), factor),
                      nir_imm_float(b, -1.0f));
}

static inline nir_ssa_def *
nir_format_float_to_unorm(nir_builder *b, nir_ssa_def *f, const unsigned *bits)
{
   nir_ssa_def *factor =
      _nir_format_norm_factor(b, bits, f->num_components, false);

   /* Clamp to the range [0, 1] */
   f = nir_fsat(b, f);

   return nir_f2u32(b, nir_fround_even(b, nir_fmul(b, f, factor)));
}

static inline nir_ssa_def *
nir_format_float_to_snorm(nir_builder *b, nir_ssa_def *f, const unsigned *bits)
{
   nir_ssa_def *factor =
      _nir_format_norm_factor(b, bits, f->num_components, true);

   /* Clamp to the range [-1, 1] */
   f = nir_fmin(b, nir_fmax(b, f, nir_imm_float(b, -1)), nir_imm_float(b, 1));

   return nir_f2i32(b, nir_fround_even(b, nir_fmul(b, f, factor)));
}

/* Converts a vector of floats to a vector of half-floats packed in the low 16
 * bits.
 */
static inline nir_ssa_def *
nir_format_float_to_half(nir_builder *b, nir_ssa_def *f)
{
   nir_ssa_def *zero = nir_imm_float(b, 0);
   nir_ssa_def *f16comps[4];
   for (unsigned i = 0; i < f->num_components; i++)
      f16comps[i] = nir_pack_half_2x16_split(b, nir_channel(b, f, i), zero);
   return nir_vec(b, f16comps, f->num_components);
}

static inline nir_ssa_def *
nir_format_linear_to_srgb(nir_builder *b, nir_ssa_def *c)
{
   nir_ssa_def *linear = nir_fmul(b, c, nir_imm_float(b, 12.92f));
   nir_ssa_def *curved =
      nir_fsub(b, nir_fmul(b, nir_imm_float(b, 1.055f),
                              nir_fpow(b, c, nir_imm_float(b, 1.0 / 2.4))),
                  nir_imm_float(b, 0.055f));

   return nir_fsat(b, nir_bcsel(b, nir_flt(b, c, nir_imm_float(b, 0.0031308f)),
                                   linear, curved));
}

static inline nir_ssa_def *
nir_format_srgb_to_linear(nir_builder *b, nir_ssa_def *c)
{
   nir_ssa_def *linear = nir_fdiv(b, c, nir_imm_float(b, 12.92f));
   nir_ssa_def *curved =
      nir_fpow(b, nir_fdiv(b, nir_fadd(b, c, nir_imm_float(b, 0.055f)),
                              nir_imm_float(b, 1.055f)),
                  nir_imm_float(b, 2.4f));

   return nir_fsat(b, nir_bcsel(b, nir_fge(b, nir_imm_float(b, 0.04045f), c),
                                   linear, curved));
}

/* Clamps a vector of uints so they don't extend beyond the given number of
 * bits per channel.
 */
static inline nir_ssa_def *
nir_format_clamp_uint(nir_builder *b, nir_ssa_def *f, const unsigned *bits)
{
   if (bits[0] == 32)
      return f;

   nir_const_value max[NIR_MAX_VEC_COMPONENTS];
   memset(max, 0, sizeof(max));
   for (unsigned i = 0; i < f->num_components; i++) {
      assert(bits[i] < 32);
      max[i].u32 = (1 << bits[i]) - 1;
   }
   return nir_umin(b, f, nir_build_imm(b, f->num_components, 32, max));
}

/* Clamps a vector of sints so they don't extend beyond the given number of
 * bits per channel.
 */
static inline nir_ssa_def *
nir_format_clamp_sint(nir_builder *b, nir_ssa_def *f, const unsigned *bits)
{
   if (bits[0] == 32)
      return f;

   nir_const_value min[NIR_MAX_VEC_COMPONENTS], max[NIR_MAX_VEC_COMPONENTS];
   memset(min, 0, sizeof(min));
   memset(max, 0, sizeof(max));
   for (unsigned i = 0; i < f->num_components; i++) {
      assert(bits[i] < 32);
      max[i].i32 = (1 << (bits[i] - 1)) - 1;
      min[i].i32 = -(1 << (bits[i] - 1));
   }
   f = nir_imin(b, f, nir_build_imm(b, f->num_components, 32, max));
   f = nir_imax(b, f, nir_build_imm(b, f->num_components, 32, min));

   return f;
}

static inline nir_ssa_def *
nir_format_unpack_11f11f10f(nir_builder *b, nir_ssa_def *packed)
{
   nir_ssa_def *chans[3];
   chans[0] = nir_mask_shift(b, packed, 0x000007ff, 4);
   chans[1] = nir_mask_shift(b, packed, 0x003ff800, -7);
   chans[2] = nir_mask_shift(b, packed, 0xffc00000, -17);

   for (unsigned i = 0; i < 3; i++)
      chans[i] = nir_unpack_half_2x16_split_x(b, chans[i]);

   return nir_vec(b, chans, 3);
}

static inline nir_ssa_def *
nir_format_pack_11f11f10f(nir_builder *b, nir_ssa_def *color)
{
   /* 10 and 11-bit floats are unsigned.  Clamp to non-negative */
   nir_ssa_def *clamped = nir_fmax(b, color, nir_imm_float(b, 0));

   nir_ssa_def *undef = nir_ssa_undef(b, 1, color->bit_size);
   nir_ssa_def *p1 = nir_pack_half_2x16_split(b, nir_channel(b, clamped, 0),
                                                 nir_channel(b, clamped, 1));
   nir_ssa_def *p2 = nir_pack_half_2x16_split(b, nir_channel(b, clamped, 2),
                                                 undef);

   /* A 10 or 11-bit float has the same exponent as a 16-bit float but with
    * fewer mantissa bits and no sign bit.  All we have to do is throw away
    * the sign bit and the bottom mantissa bits and shift it into place.
    */
   nir_ssa_def *packed = nir_imm_int(b, 0);
   packed = nir_mask_shift_or(b, packed, p1, 0x00007ff0, -4);
   packed = nir_mask_shift_or(b, packed, p1, 0x7ff00000, -9);
   packed = nir_mask_shift_or(b, packed, p2, 0x00007fe0, 17);

   return packed;
}

static inline nir_ssa_def *
nir_format_pack_r9g9b9e5(nir_builder *b, nir_ssa_def *color)
{
   /* See also float3_to_rgb9e5 */

   /* First, we need to clamp it to range. */
   nir_ssa_def *clamped = nir_fmin(b, color, nir_imm_float(b, MAX_RGB9E5));

   /* Get rid of negatives and NaN */
   clamped = nir_bcsel(b, nir_ult(b, nir_imm_int(b, 0x7f800000), color),
                          nir_imm_float(b, 0), clamped);

   /* maxrgb.u = MAX3(rc.u, gc.u, bc.u); */
   nir_ssa_def *maxu = nir_umax(b, nir_channel(b, clamped, 0),
                       nir_umax(b, nir_channel(b, clamped, 1),
                                   nir_channel(b, clamped, 2)));

   /* maxrgb.u += maxrgb.u & (1 << (23-9)); */
   maxu = nir_iadd(b, maxu, nir_iand(b, maxu, nir_imm_int(b, 1 << 14)));

   /* exp_shared = MAX2((maxrgb.u >> 23), -RGB9E5_EXP_BIAS - 1 + 127) +
    *              1 + RGB9E5_EXP_BIAS - 127;
    */
   nir_ssa_def *exp_shared =
      nir_iadd(b, nir_umax(b, nir_ushr_imm(b, maxu, 23),
                              nir_imm_int(b, -RGB9E5_EXP_BIAS - 1 + 127)),
                  nir_imm_int(b, 1 + RGB9E5_EXP_BIAS - 127));

   /* revdenom_biasedexp = 127 - (exp_shared - RGB9E5_EXP_BIAS -
    *                             RGB9E5_MANTISSA_BITS) + 1;
    */
   nir_ssa_def *revdenom_biasedexp =
      nir_isub(b, nir_imm_int(b, 127 + RGB9E5_EXP_BIAS +
                                 RGB9E5_MANTISSA_BITS + 1),
                  exp_shared);

   /* revdenom.u = revdenom_biasedexp << 23; */
   nir_ssa_def *revdenom =
      nir_ishl(b, revdenom_biasedexp, nir_imm_int(b, 23));

   /* rm = (int) (rc.f * revdenom.f);
    * gm = (int) (gc.f * revdenom.f);
    * bm = (int) (bc.f * revdenom.f);
    */
   nir_ssa_def *mantissa =
      nir_f2i32(b, nir_fmul(b, clamped, revdenom));

   /* rm = (rm & 1) + (rm >> 1);
    * gm = (gm & 1) + (gm >> 1);
    * bm = (bm & 1) + (bm >> 1);
    */
   mantissa = nir_iadd(b, nir_iand_imm(b, mantissa, 1),
                          nir_ushr_imm(b, mantissa, 1));

   nir_ssa_def *packed = nir_channel(b, mantissa, 0);
   packed = nir_mask_shift_or(b, packed, nir_channel(b, mantissa, 1), ~0, 9);
   packed = nir_mask_shift_or(b, packed, nir_channel(b, mantissa, 2), ~0, 18);
   packed = nir_mask_shift_or(b, packed, exp_shared, ~0, 27);

   return packed;
}
