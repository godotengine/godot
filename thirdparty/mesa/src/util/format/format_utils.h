/**
 * \file format_utils.h
 * A collection of format conversion utility functions.
 */

/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2006  Brian Paul  All Rights Reserved.
 * Copyright (C) 2014  Intel Corporation  All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef UTIL_FORMAT_UTILS_H
#define UTIL_FORMAT_UTILS_H

#include "util/half_float.h"
#include "util/rounding.h"

/* Extends an integer of size SRC_BITS to one of size DST_BITS linearly */
#define EXTEND_NORMALIZED_INT(X, SRC_BITS, DST_BITS) \
      (((X) * (int)(u_uintN_max(DST_BITS) / u_uintN_max(SRC_BITS))) + \
       (((DST_BITS) % (SRC_BITS)) ? ((X) >> ((SRC_BITS) - (DST_BITS) % (SRC_BITS))) : 0))

static inline float
_mesa_unorm_to_float(unsigned x, unsigned src_bits)
{
   return x * (1.0f / (float)u_uintN_max(src_bits));
}

static inline float
_mesa_snorm_to_float(int x, unsigned src_bits)
{
   if (x <= -u_intN_max(src_bits))
      return -1.0f;
   else
      return x * (1.0f / (float)u_intN_max(src_bits));
}

static inline uint16_t
_mesa_unorm_to_half(unsigned x, unsigned src_bits)
{
   return _mesa_float_to_half(_mesa_unorm_to_float(x, src_bits));
}

static inline uint16_t
_mesa_snorm_to_half(int x, unsigned src_bits)
{
   return _mesa_float_to_half(_mesa_snorm_to_float(x, src_bits));
}

static inline unsigned
_mesa_float_to_unorm(float x, unsigned dst_bits)
{
   if (x < 0.0f)
      return 0;
   else if (x > 1.0f)
      return u_uintN_max(dst_bits);
   else
      return _mesa_i64roundevenf(x * u_uintN_max(dst_bits));
}

static inline unsigned
_mesa_half_to_unorm(uint16_t x, unsigned dst_bits)
{
   return _mesa_float_to_unorm(_mesa_half_to_float(x), dst_bits);
}

static inline unsigned
_mesa_unorm_to_unorm(unsigned x, unsigned src_bits, unsigned dst_bits)
{
   if (src_bits < dst_bits) {
      return EXTEND_NORMALIZED_INT(x, src_bits, dst_bits);
   } else if (src_bits > dst_bits) {
      unsigned src_half = (1u << (src_bits - 1)) - 1;

      if (src_bits + dst_bits > sizeof(x) * 8) {
         assert(src_bits + dst_bits <= sizeof(uint64_t) * 8);
         return (((uint64_t) x * u_uintN_max(dst_bits) + src_half) /
                 u_uintN_max(src_bits));
      } else {
         return (x * u_uintN_max(dst_bits) + src_half) / u_uintN_max(src_bits);
      }
   } else {
      return x;
   }
}

static inline unsigned
_mesa_snorm_to_unorm(int x, unsigned src_bits, unsigned dst_bits)
{
   if (x < 0)
      return 0;
   else
      return _mesa_unorm_to_unorm(x, src_bits - 1, dst_bits);
}

static inline int
_mesa_float_to_snorm(float x, unsigned dst_bits)
{
   if (x < -1.0f)
      return -u_intN_max(dst_bits);
   else if (x > 1.0f)
      return u_intN_max(dst_bits);
   else
      return _mesa_lroundevenf(x * u_intN_max(dst_bits));
}

static inline int
_mesa_half_to_snorm(uint16_t x, unsigned dst_bits)
{
   return _mesa_float_to_snorm(_mesa_half_to_float(x), dst_bits);
}

static inline int
_mesa_unorm_to_snorm(unsigned x, unsigned src_bits, unsigned dst_bits)
{
   return _mesa_unorm_to_unorm(x, src_bits, dst_bits - 1);
}

static inline int
_mesa_snorm_to_snorm(int x, unsigned src_bits, unsigned dst_bits)
{
   if (x < -u_intN_max(src_bits))
      return -u_intN_max(dst_bits);
   else if (src_bits < dst_bits)
      return EXTEND_NORMALIZED_INT(x, src_bits - 1, dst_bits - 1);
   else
      return x >> (src_bits - dst_bits);
}

static inline unsigned
_mesa_unsigned_to_unsigned(unsigned src, unsigned dst_size)
{
   return MIN2(src, u_uintN_max(dst_size));
}

static inline int
_mesa_unsigned_to_signed(unsigned src, unsigned dst_size)
{
   return MIN2(src, (unsigned)u_intN_max(dst_size));
}

static inline int
_mesa_signed_to_signed(int src, unsigned dst_size)
{
   return CLAMP(src, u_intN_min(dst_size), u_intN_max(dst_size));
}

static inline unsigned
_mesa_signed_to_unsigned(int src, unsigned dst_size)
{
   return CLAMP(src, 0, u_uintN_max(dst_size));
}

static inline unsigned
_mesa_float_to_unsigned(float src, unsigned dst_bits)
{
   if (src < 0.0f)
      return 0;
   if (src > (float)u_uintN_max(dst_bits))
       return u_uintN_max(dst_bits);
   return _mesa_signed_to_unsigned(src, dst_bits);
}

static inline unsigned
_mesa_float_to_signed(float src, unsigned dst_bits)
{
   if (src < (float)(-u_intN_max(dst_bits)))
      return -u_intN_max(dst_bits);
   if (src > (float)u_intN_max(dst_bits))
       return u_intN_max(dst_bits);
   return _mesa_signed_to_signed(src, dst_bits);
}

static inline unsigned
_mesa_half_to_unsigned(uint16_t src, unsigned dst_bits)
{
   if (_mesa_half_is_negative(src))
      return 0;
   return _mesa_unsigned_to_unsigned(_mesa_float_to_half(src), dst_bits);
}

static inline unsigned
_mesa_half_to_signed(uint16_t src, unsigned dst_bits)
{
   return _mesa_float_to_signed(_mesa_half_to_float(src), dst_bits);
}

#endif /* UTIL_FORMAT_UTILS_H */
