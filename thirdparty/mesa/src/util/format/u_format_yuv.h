/**************************************************************************
 *
 * Copyright 2010 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE COPYRIGHT HOLDERS, AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 **************************************************************************/


/**
 * @file
 * YUV colorspace conversion.
 *
 * @author Brian Paul <brianp@vmware.com>
 * @author Michal Krol <michal@vmware.com>
 * @author Jose Fonseca <jfonseca@vmware.com>
 *
 * See also:
 * - http://www.fourcc.org/fccyvrgb.php
 * - http://msdn.microsoft.com/en-us/library/ms893078
 * - http://en.wikipedia.org/wiki/YUV
 */


#ifndef U_FORMAT_YUV_H_
#define U_FORMAT_YUV_H_


#include "pipe/p_compiler.h"
#include "util/u_math.h"

#include "c99_compat.h"

/*
 * TODO: Ensure we use consistent and right floating formulas, with enough
 * precision in the coefficients.
 */

static inline void
util_format_rgb_float_to_yuv(float r, float g, float b,
                             uint8_t *y, uint8_t *u, uint8_t *v)
{
   const float _r = SATURATE(r);
   const float _g = SATURATE(g);
   const float _b = SATURATE(b);

   const float scale = 255.0f;

   const int _y = scale * ( (0.257f * _r) + (0.504f * _g) + (0.098f * _b));
   const int _u = scale * (-(0.148f * _r) - (0.291f * _g) + (0.439f * _b));
   const int _v = scale * ( (0.439f * _r) - (0.368f * _g) - (0.071f * _b));

   *y = _y + 16;
   *u = _u + 128;
   *v = _v + 128;
}


static inline void
util_format_yuv_to_rgb_float(uint8_t y, uint8_t u, uint8_t v,
                             float *r, float *g, float *b)
{
   const int _y = y - 16;
   const int _u = u - 128;
   const int _v = v - 128;

   const float y_factor = 255.0f / 219.0f;

   const float scale = 1.0f / 255.0f;

   *r = scale * (y_factor * _y               + 1.596f * _v);
   *g = scale * (y_factor * _y - 0.391f * _u - 0.813f * _v);
   *b = scale * (y_factor * _y + 2.018f * _u              );
}


static inline void
util_format_rgb_8unorm_to_yuv(uint8_t r, uint8_t g, uint8_t b,
                	      uint8_t *y, uint8_t *u, uint8_t *v)
{
   *y = ((  66 * r + 129 * g +  25 * b + 128) >> 8) +  16;
   *u = (( -38 * r -  74 * g + 112 * b + 128) >> 8) + 128;
   *v = (( 112 * r -  94 * g -  18 * b + 128) >> 8) + 128;
}


static inline void
util_format_yuv_to_rgb_8unorm(uint8_t y, uint8_t u, uint8_t v,
                              uint8_t *r, uint8_t *g, uint8_t *b)
{
   const int _y = y - 16;
   const int _u = u - 128;
   const int _v = v - 128;

   const int _r = (298 * _y            + 409 * _v + 128) >> 8;
   const int _g = (298 * _y - 100 * _u - 208 * _v + 128) >> 8;
   const int _b = (298 * _y + 516 * _u            + 128) >> 8;

   *r = CLAMP(_r, 0, 255);
   *g = CLAMP(_g, 0, 255);
   *b = CLAMP(_b, 0, 255);
}



void
util_format_uyvy_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                              const uint8_t *restrict src_row, unsigned src_stride,
                              unsigned width, unsigned height);

void
util_format_uyvy_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                               const uint8_t *restrict src_row, unsigned src_stride,
                               unsigned width, unsigned height);

void
util_format_uyvy_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                            const float *restrict src_row, unsigned src_stride,
                            unsigned width, unsigned height);

void
util_format_uyvy_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                             const uint8_t *restrict src_row, unsigned src_stride,
                             unsigned width, unsigned height);

void
util_format_uyvy_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                             unsigned i, unsigned j);

void
util_format_yuyv_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                              const uint8_t *restrict src_row, unsigned src_stride,
                              unsigned width, unsigned height);

void
util_format_yuyv_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                               const uint8_t *restrict src_row, unsigned src_stride,
                               unsigned width, unsigned height);

void
util_format_yuyv_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                            const float *restrict src_row, unsigned src_stride,
                            unsigned width, unsigned height);

void
util_format_yuyv_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                             const uint8_t *restrict src_row, unsigned src_stride,
                             unsigned width, unsigned height);

void
util_format_yuyv_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                             unsigned i, unsigned j);

void
util_format_r8g8_b8g8_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height);

void
util_format_r8g8_b8g8_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                          const uint8_t *restrict src_row, unsigned src_stride,
                                          unsigned width, unsigned height);

void
util_format_r8g8_b8g8_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const float *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height);

void
util_format_r8g8_b8g8_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height);

void
util_format_r8g8_b8g8_unorm_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                        unsigned i, unsigned j);

void
util_format_g8r8_g8b8_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height);

void
util_format_g8r8_g8b8_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                          const uint8_t *restrict src_row, unsigned src_stride,
                                          unsigned width, unsigned height);

void
util_format_g8r8_g8b8_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const float *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height);

void
util_format_g8r8_g8b8_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height);

void
util_format_g8r8_g8b8_unorm_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                        unsigned i, unsigned j);

#endif /* U_FORMAT_YUV_H_ */
