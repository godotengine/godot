/*
 * Copyright (C) 2011 Marek Olšák <maraeo@gmail.com>
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/* Copied from EXT_texture_shared_exponent and edited, getting rid of
 * expensive float math bits too. */

#ifndef RGB9E5_H
#define RGB9E5_H

#include <assert.h>
#include <stdint.h>
#include <math.h>

#define RGB9E5_EXPONENT_BITS          5
#define RGB9E5_MANTISSA_BITS          9
#define RGB9E5_EXP_BIAS               15
#define RGB9E5_MAX_VALID_BIASED_EXP   31

#define MAX_RGB9E5_EXP               (RGB9E5_MAX_VALID_BIASED_EXP - RGB9E5_EXP_BIAS)
#define RGB9E5_MANTISSA_VALUES       (1<<RGB9E5_MANTISSA_BITS)
#define MAX_RGB9E5_MANTISSA          (RGB9E5_MANTISSA_VALUES-1)
#define MAX_RGB9E5                   (((float)MAX_RGB9E5_MANTISSA)/RGB9E5_MANTISSA_VALUES * (1<<MAX_RGB9E5_EXP))

static inline int rgb9e5_ClampRange(float x)
{
   union { float f; uint32_t u; } f, max;
   f.f = x;
   max.f = MAX_RGB9E5;

   if (f.u > 0x7f800000)
  /* catches neg, NaNs */
      return 0;
   else if (f.u >= max.u)
      return max.u;
   else
      return f.u;
}

static inline uint32_t float3_to_rgb9e5(const float rgb[3])
{
   int rm, gm, bm, exp_shared;
   uint32_t revdenom_biasedexp;
   union { float f; uint32_t u; } rc, bc, gc, maxrgb, revdenom;

   rc.u = rgb9e5_ClampRange(rgb[0]);
   gc.u = rgb9e5_ClampRange(rgb[1]);
   bc.u = rgb9e5_ClampRange(rgb[2]);
   maxrgb.u = MAX3(rc.u, gc.u, bc.u);

   /*
    * Compared to what the spec suggests, instead of conditionally adjusting
    * the exponent after the fact do it here by doing the equivalent of +0.5 -
    * the int add will spill over into the exponent in this case.
    */
   maxrgb.u += maxrgb.u & (1 << (23-9));
   exp_shared = MAX2((maxrgb.u >> 23), -RGB9E5_EXP_BIAS - 1 + 127) +
                1 + RGB9E5_EXP_BIAS - 127;
   revdenom_biasedexp = 127 - (exp_shared - RGB9E5_EXP_BIAS -
                               RGB9E5_MANTISSA_BITS) + 1;
   revdenom.u = revdenom_biasedexp << 23;
   assert(exp_shared <= RGB9E5_MAX_VALID_BIASED_EXP);

   /*
    * The spec uses strict round-up behavior (d3d10 disagrees, but in any case
    * must match what is done above for figuring out exponent).
    * We avoid the doubles ((int) rc * revdenom + 0.5) by doing the rounding
    * ourselves (revdenom was adjusted by +1, above).
    */
   rm = (int) (rc.f * revdenom.f);
   gm = (int) (gc.f * revdenom.f);
   bm = (int) (bc.f * revdenom.f);
   rm = (rm & 1) + (rm >> 1);
   gm = (gm & 1) + (gm >> 1);
   bm = (bm & 1) + (bm >> 1);

   assert(rm <= MAX_RGB9E5_MANTISSA);
   assert(gm <= MAX_RGB9E5_MANTISSA);
   assert(bm <= MAX_RGB9E5_MANTISSA);
   assert(rm >= 0);
   assert(gm >= 0);
   assert(bm >= 0);

   return (exp_shared << 27) | (bm << 18) | (gm << 9) | rm;
}

static inline void rgb9e5_to_float3(uint32_t rgb, float retval[3])
{
   int exponent;
   union { float f; uint32_t u; } scale;

   exponent = (rgb >> 27) - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS;
   scale.u = (exponent + 127) << 23;

   retval[0] = ( rgb        & 0x1ff) * scale.f;
   retval[1] = ((rgb >> 9)  & 0x1ff) * scale.f;
   retval[2] = ((rgb >> 18) & 0x1ff) * scale.f;
}

#endif
