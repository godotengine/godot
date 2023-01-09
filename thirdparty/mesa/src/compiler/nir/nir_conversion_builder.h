/*
 * Copyright © 2020 Collabora Ltd.
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

#ifndef NIR_CONVERSION_BUILDER_H
#define NIR_CONVERSION_BUILDER_H

#include "util/u_math.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline nir_ssa_def *
nir_round_float_to_int(nir_builder *b, nir_ssa_def *src,
                       nir_rounding_mode round)
{
   switch (round) {
   case nir_rounding_mode_ru:
      return nir_fceil(b, src);

   case nir_rounding_mode_rd:
      return nir_ffloor(b, src);

   case nir_rounding_mode_rtne:
      return nir_fround_even(b, src);

   case nir_rounding_mode_undef:
   case nir_rounding_mode_rtz:
      break;
   }
   unreachable("unexpected rounding mode");
}

static inline nir_ssa_def *
nir_round_float_to_float(nir_builder *b, nir_ssa_def *src,
                         unsigned dest_bit_size,
                         nir_rounding_mode round)
{
   unsigned src_bit_size = src->bit_size;
   if (dest_bit_size > src_bit_size)
      return src; /* No rounding is needed for an up-convert */

   nir_op low_conv = nir_type_conversion_op(nir_type_float | src_bit_size,
                                            nir_type_float | dest_bit_size,
                                            nir_rounding_mode_undef);
   nir_op high_conv = nir_type_conversion_op(nir_type_float | dest_bit_size,
                                             nir_type_float | src_bit_size,
                                             nir_rounding_mode_undef);

   switch (round) {
   case nir_rounding_mode_ru: {
      /* If lower-precision conversion results in a lower value, push it
      * up one ULP. */
      nir_ssa_def *lower_prec =
         nir_build_alu(b, low_conv, src, NULL, NULL, NULL);
      nir_ssa_def *roundtrip =
         nir_build_alu(b, high_conv, lower_prec, NULL, NULL, NULL);
      nir_ssa_def *cmp = nir_flt(b, roundtrip, src);
      nir_ssa_def *inf = nir_imm_floatN_t(b, INFINITY, dest_bit_size);
      return nir_bcsel(b, cmp, nir_nextafter(b, lower_prec, inf), lower_prec);
   }
   case nir_rounding_mode_rd: {
      /* If lower-precision conversion results in a higher value, push it
      * down one ULP. */
      nir_ssa_def *lower_prec =
         nir_build_alu(b, low_conv, src, NULL, NULL, NULL);
      nir_ssa_def *roundtrip =
         nir_build_alu(b, high_conv, lower_prec, NULL, NULL, NULL);
      nir_ssa_def *cmp = nir_flt(b, src, roundtrip);
      nir_ssa_def *neg_inf = nir_imm_floatN_t(b, -INFINITY, dest_bit_size);
      return nir_bcsel(b, cmp, nir_nextafter(b, lower_prec, neg_inf), lower_prec);
   }
   case nir_rounding_mode_rtz:
      return nir_bcsel(b, nir_flt(b, src, nir_imm_zero(b, 1, src->bit_size)),
                          nir_round_float_to_float(b, src, dest_bit_size,
                                                   nir_rounding_mode_ru),
                          nir_round_float_to_float(b, src, dest_bit_size,
                                                   nir_rounding_mode_rd));
   case nir_rounding_mode_rtne:
   case nir_rounding_mode_undef:
      break;
   }
   unreachable("unexpected rounding mode");
}

static inline nir_ssa_def *
nir_round_int_to_float(nir_builder *b, nir_ssa_def *src,
                       nir_alu_type src_type,
                       unsigned dest_bit_size,
                       nir_rounding_mode round)
{
   /* We only care whether or not its signed */
   src_type = nir_alu_type_get_base_type(src_type);

   unsigned mantissa_bits;
   switch (dest_bit_size) {
   case 16:
      mantissa_bits = 10;
      break;
   case 32:
      mantissa_bits = 23;
      break;
   case 64:
      mantissa_bits = 52;
      break;
   default: unreachable("Unsupported bit size");
   }

   if (src->bit_size < mantissa_bits)
      return src;

   if (src_type == nir_type_int) {
      nir_ssa_def *sign =
         nir_i2b(b, nir_ishr(b, src, nir_imm_int(b, src->bit_size - 1)));
      nir_ssa_def *abs = nir_iabs(b, src);
      nir_ssa_def *positive_rounded =
         nir_round_int_to_float(b, abs, nir_type_uint, dest_bit_size, round);
      nir_ssa_def *max_positive =
         nir_imm_intN_t(b, (1ull << (src->bit_size - 1)) - 1, src->bit_size);
      switch (round) {
      case nir_rounding_mode_rtz:
         return nir_bcsel(b, sign, nir_ineg(b, positive_rounded),
                                   positive_rounded);
         break;
      case nir_rounding_mode_ru:
         return nir_bcsel(b, sign,
                          nir_ineg(b, nir_round_int_to_float(b, abs, nir_type_uint, dest_bit_size, nir_rounding_mode_rd)),
                          nir_umin(b, positive_rounded, max_positive));
         break;
      case nir_rounding_mode_rd:
         return nir_bcsel(b, sign,
                          nir_ineg(b,
                                   nir_umin(b, max_positive,
                                            nir_round_int_to_float(b, abs, nir_type_uint, dest_bit_size, nir_rounding_mode_ru))),
                          positive_rounded);
      case nir_rounding_mode_rtne:
      case nir_rounding_mode_undef:
         break;
      }
      unreachable("unexpected rounding mode");
   } else {
      nir_ssa_def *mantissa_bit_size = nir_imm_int(b, mantissa_bits);
      nir_ssa_def *msb = nir_imax(b, nir_ufind_msb(b, src), mantissa_bit_size);
      nir_ssa_def *bits_to_lose = nir_isub(b, msb, mantissa_bit_size);
      nir_ssa_def *one = nir_imm_intN_t(b, 1, src->bit_size);
      nir_ssa_def *adjust = nir_ishl(b, one, bits_to_lose);
      nir_ssa_def *mask = nir_inot(b, nir_isub(b, adjust, one));
      nir_ssa_def *truncated = nir_iand(b, src, mask);
      switch (round) {
      case nir_rounding_mode_rtz:
      case nir_rounding_mode_rd:
         return truncated;
         break;
      case nir_rounding_mode_ru:
         return nir_bcsel(b, nir_ieq(b, src, truncated),
                             src, nir_uadd_sat(b, truncated, adjust));
      case nir_rounding_mode_rtne:
      case nir_rounding_mode_undef:
         break;
      }
      unreachable("unexpected rounding mode");
   }
}

/** Returns true if the representable range of a contains the representable
 * range of b.
 */
static inline bool
nir_alu_type_range_contains_type_range(nir_alu_type a, nir_alu_type b)
{
   /* Split types from bit sizes */
   nir_alu_type a_base_type = nir_alu_type_get_base_type(a);
   nir_alu_type b_base_type = nir_alu_type_get_base_type(b);
   unsigned a_bit_size = nir_alu_type_get_type_size(a);
   unsigned b_bit_size = nir_alu_type_get_type_size(b);

   /* This requires sized types */
   assert(a_bit_size > 0 && b_bit_size > 0);

   if (a_base_type == b_base_type && a_bit_size >= b_bit_size)
      return true;

   if (a_base_type == nir_type_int && b_base_type == nir_type_uint &&
       a_bit_size > b_bit_size)
      return true;

   /* 16-bit floats fit in 32-bit integers */
   if (a_base_type == nir_type_int && a_bit_size >= 32 &&
       b == nir_type_float16)
      return true;

   /* All signed or unsigned ints can fit in float or above. A uint8 can fit
    * in a float16.
    */
   if (a_base_type == nir_type_float && b_base_type != nir_type_float &&
       (a_bit_size >= 32 || b_bit_size == 8))
      return true;

   return false;
}

/**
 * Retrieves limits used for clamping a value of the src type into
 * the widest representable range of the dst type via cmp + bcsel
 */
static inline void
nir_get_clamp_limits(nir_builder *b,
                     nir_alu_type src_type,
                     nir_alu_type dest_type,
                     nir_ssa_def **low, nir_ssa_def **high)
{
   /* Split types from bit sizes */
   nir_alu_type src_base_type = nir_alu_type_get_base_type(src_type);
   nir_alu_type dest_base_type = nir_alu_type_get_base_type(dest_type);
   unsigned src_bit_size = nir_alu_type_get_type_size(src_type);
   unsigned dest_bit_size = nir_alu_type_get_type_size(dest_type);
   assert(dest_bit_size != 0 && src_bit_size != 0);

   *low = NULL;
   *high = NULL;

   /* limits of the destination type, expressed in the source type */
   switch (dest_base_type) {
   case nir_type_int: {
      int64_t ilow, ihigh;
      if (dest_bit_size == 64) {
         ilow = INT64_MIN;
         ihigh = INT64_MAX;
      } else {
         ilow = -(1ll << (dest_bit_size - 1));
         ihigh = (1ll << (dest_bit_size - 1)) - 1;
      }

      if (src_base_type == nir_type_int) {
         *low = nir_imm_intN_t(b, ilow, src_bit_size);
         *high = nir_imm_intN_t(b, ihigh, src_bit_size);
      } else if (src_base_type == nir_type_uint) {
         assert(src_bit_size >= dest_bit_size);
         *high = nir_imm_intN_t(b, ihigh, src_bit_size);
      } else {
         *low = nir_imm_floatN_t(b, ilow, src_bit_size);
         *high = nir_imm_floatN_t(b, ihigh, src_bit_size);
      }
      break;
   }
   case nir_type_uint: {
      uint64_t uhigh = dest_bit_size == 64 ?
         ~0ull : (1ull << dest_bit_size) - 1;
      if (src_base_type != nir_type_float) {
         *low = nir_imm_intN_t(b, 0, src_bit_size);
         if (src_base_type == nir_type_uint || src_bit_size > dest_bit_size)
            *high = nir_imm_intN_t(b, uhigh, src_bit_size);
      } else {
         *low = nir_imm_floatN_t(b, 0.0f, src_bit_size);
         *high = nir_imm_floatN_t(b, uhigh, src_bit_size);
      }
      break;
   }
   case nir_type_float: {
      double flow, fhigh;
      switch (dest_bit_size) {
      case 16:
         flow = -65504.0f;
         fhigh = 65504.0f;
         break;
      case 32:
         flow = -FLT_MAX;
         fhigh = FLT_MAX;
         break;
      case 64:
         flow = -DBL_MAX;
         fhigh = DBL_MAX;
         break;
      default:
         unreachable("Unhandled bit size");
      }

      switch (src_base_type) {
      case nir_type_int: {
         int64_t src_ilow, src_ihigh;
         if (src_bit_size == 64) {
            src_ilow = INT64_MIN;
            src_ihigh = INT64_MAX;
         } else {
            src_ilow = -(1ll << (src_bit_size - 1));
            src_ihigh = (1ll << (src_bit_size - 1)) - 1;
         }
         if (src_ilow < flow)
            *low = nir_imm_intN_t(b, flow, src_bit_size);
         if (src_ihigh > fhigh)
            *high = nir_imm_intN_t(b, fhigh, src_bit_size);
         break;
      }
      case nir_type_uint: {
         uint64_t src_uhigh = src_bit_size == 64 ?
            ~0ull : (1ull << src_bit_size) - 1;
         if (src_uhigh > fhigh)
            *high = nir_imm_intN_t(b, fhigh, src_bit_size);
         break;
      }
      case nir_type_float:
         *low = nir_imm_floatN_t(b, flow, src_bit_size);
         *high = nir_imm_floatN_t(b, fhigh, src_bit_size);
         break;
      default:
         unreachable("Clamping from unknown type");
      }
      break;
   }
   default:
      unreachable("clamping to unknown type");
      break;
   }
}

/**
 * Clamp the value into the widest representatble range of the
 * destination type with cmp + bcsel.
 * 
 * val/val_type: The variables used for bcsel
 * src/src_type: The variables used for comparison
 * dest_type: The type which determines the range used for comparison
 */
static inline nir_ssa_def *
nir_clamp_to_type_range(nir_builder *b,
                        nir_ssa_def *val, nir_alu_type val_type,
                        nir_ssa_def *src, nir_alu_type src_type,
                        nir_alu_type dest_type)
{
   assert(nir_alu_type_get_type_size(src_type) == 0 ||
          nir_alu_type_get_type_size(src_type) == src->bit_size);
   src_type |= src->bit_size;
   if (nir_alu_type_range_contains_type_range(dest_type, src_type))
      return val;

   /* limits of the destination type, expressed in the source type */
   nir_ssa_def *low = NULL, *high = NULL;
   nir_get_clamp_limits(b, src_type, dest_type, &low, &high);

   nir_ssa_def *low_cond = NULL, *high_cond = NULL;
   switch (nir_alu_type_get_base_type(src_type)) {
   case nir_type_int:
      low_cond = low ? nir_ilt(b, src, low) : NULL;
      high_cond = high ? nir_ilt(b, high, src) : NULL;
      break;
   case nir_type_uint:
      low_cond = low ? nir_ult(b, src, low) : NULL;
      high_cond = high ? nir_ult(b, high, src) : NULL;
      break;
   case nir_type_float:
      low_cond = low ? nir_fge(b, low, src) : NULL;
      high_cond = high ? nir_fge(b, src, high) : NULL;
      break;
   default:
      unreachable("clamping from unknown type");
   }

   nir_ssa_def *val_low = low, *val_high = high;
   if (val_type != src_type) {
      nir_get_clamp_limits(b, val_type, dest_type, &val_low, &val_high);
   }

   nir_ssa_def *res = val;
   if (low_cond && val_low)
      res = nir_bcsel(b, low_cond, val_low, res);
   if (high_cond && val_high)
      res = nir_bcsel(b, high_cond, val_high, res);

   return res;
}

static inline nir_rounding_mode
nir_simplify_conversion_rounding(nir_alu_type src_type,
                                 nir_alu_type dest_type,
                                 nir_rounding_mode rounding)
{
   nir_alu_type src_base_type = nir_alu_type_get_base_type(src_type);
   nir_alu_type dest_base_type = nir_alu_type_get_base_type(dest_type);
   unsigned src_bit_size = nir_alu_type_get_type_size(src_type);
   unsigned dest_bit_size = nir_alu_type_get_type_size(dest_type);
   assert(src_bit_size > 0 && dest_bit_size > 0);

   if (rounding == nir_rounding_mode_undef)
      return rounding;

   /* Pure integer conversion doesn't have any rounding */
   if (src_base_type != nir_type_float &&
       dest_base_type != nir_type_float)
      return nir_rounding_mode_undef;

   /* Float down-casts don't round */
   if (src_base_type == nir_type_float &&
       dest_base_type == nir_type_float &&
       dest_bit_size >= src_bit_size)
      return nir_rounding_mode_undef;

   /* Regular float to int conversions are RTZ */
   if (src_base_type == nir_type_float &&
       dest_base_type != nir_type_float &&
       rounding == nir_rounding_mode_rtz)
      return nir_rounding_mode_undef;

   /* The CL spec requires regular conversions to float to be RTNE */
   if (dest_base_type == nir_type_float &&
       rounding == nir_rounding_mode_rtne)
      return nir_rounding_mode_undef;

   /* Couldn't simplify */
   return rounding;
}

static inline nir_ssa_def *
nir_convert_with_rounding(nir_builder *b,
                          nir_ssa_def *src, nir_alu_type src_type,
                          nir_alu_type dest_type,
                          nir_rounding_mode round,
                          bool clamp)
{
   /* Some stuff wants sized types */
   assert(nir_alu_type_get_type_size(src_type) == 0 ||
          nir_alu_type_get_type_size(src_type) == src->bit_size);
   src_type |= src->bit_size;

   /* Split types from bit sizes */
   nir_alu_type src_base_type = nir_alu_type_get_base_type(src_type);
   nir_alu_type dest_base_type = nir_alu_type_get_base_type(dest_type);
   unsigned dest_bit_size = nir_alu_type_get_type_size(dest_type);

   /* Try to simplify the conversion if we can */
   clamp = clamp &&
      !nir_alu_type_range_contains_type_range(dest_type, src_type);
   round = nir_simplify_conversion_rounding(src_type, dest_type, round);

   /* For float -> int/uint conversions, we might not be able to represent
    * the destination range in the source float accurately. For these cases,
    * do the comparison in float range, but the bcsel in the destination range.
    */
   bool clamp_after_conversion = clamp &&
      src_base_type == nir_type_float &&
      dest_base_type != nir_type_float;

   /*
    * If we don't care about rounding and clamping, we can just use NIR's
    * built-in ops. There is also a special case for SPIR-V in shaders, where
    * f32/f64 -> f16 conversions can have one of two rounding modes applied,
    * which NIR has built-in opcodes for.
    *
    * For the rest, we have our own implementation of rounding and clamping.
    */
   bool trivial_convert;
   if (!clamp && round == nir_rounding_mode_undef) {
      trivial_convert = true;
   } else if (!clamp && src_type == nir_type_float32 &&
                        dest_type == nir_type_float16 &&
                        (round == nir_rounding_mode_rtne ||
                         round == nir_rounding_mode_rtz)) {
      trivial_convert = true;
   } else {
      trivial_convert = false;
   }

   if (trivial_convert)
      return nir_type_convert(b, src, src_type, dest_type, round);

   nir_ssa_def *dest = src;

   /* clamp the result into range */
   if (clamp && !clamp_after_conversion)
      dest = nir_clamp_to_type_range(b, src, src_type, src, src_type, dest_type);

   /* round with selected rounding mode */
   if (!trivial_convert && round != nir_rounding_mode_undef) {
      if (src_base_type == nir_type_float) {
         if (dest_base_type == nir_type_float) {
            dest = nir_round_float_to_float(b, dest, dest_bit_size, round);
         } else {
            dest = nir_round_float_to_int(b, dest, round);
         }
      } else {
         dest = nir_round_int_to_float(b, dest, src_type, dest_bit_size, round);
      }

      round = nir_rounding_mode_undef;
   }

   /* now we can convert the value */
   nir_op op = nir_type_conversion_op(src_type, dest_type, round);
   dest = nir_build_alu(b, op, dest, NULL, NULL, NULL);

   if (clamp_after_conversion)
      dest = nir_clamp_to_type_range(b, dest, dest_type, src, src_type, dest_type);

   return dest;
}

#ifdef __cplusplus
}
#endif

#endif /* NIR_CONVERSION_BUILDER_H */
