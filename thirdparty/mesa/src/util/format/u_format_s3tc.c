/**************************************************************************
 *
 * Copyright (C) 1999-2007  Brian Paul   All Rights Reserved.
 * Copyright (c) 2008 VMware, Inc.
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
 *
 **************************************************************************/

#include "util/format/u_format.h"
#include "util/format/u_format_s3tc.h"
#include "util/format_srgb.h"
#include "util/u_math.h"

#include "util/format/texcompress_s3tc_tmp.h"

util_format_dxtn_fetch_t util_format_dxt1_rgb_fetch = (util_format_dxtn_fetch_t)fetch_2d_texel_rgb_dxt1;
util_format_dxtn_fetch_t util_format_dxt1_rgba_fetch = (util_format_dxtn_fetch_t)fetch_2d_texel_rgba_dxt1;
util_format_dxtn_fetch_t util_format_dxt3_rgba_fetch = (util_format_dxtn_fetch_t)fetch_2d_texel_rgba_dxt3;
util_format_dxtn_fetch_t util_format_dxt5_rgba_fetch = (util_format_dxtn_fetch_t)fetch_2d_texel_rgba_dxt5;

util_format_dxtn_pack_t util_format_dxtn_pack = (util_format_dxtn_pack_t)tx_compress_dxtn;


/*
 * Pixel fetch.
 */

void
util_format_dxt1_rgb_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   util_format_dxt1_rgb_fetch(0, src, i, j, dst);
}

void
util_format_dxt1_rgba_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   util_format_dxt1_rgba_fetch(0, src, i, j, dst);
}

void
util_format_dxt3_rgba_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   util_format_dxt3_rgba_fetch(0, src, i, j, dst);
}

void
util_format_dxt5_rgba_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   util_format_dxt5_rgba_fetch(0, src, i, j, dst);
}

void
util_format_dxt1_rgb_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt1_rgb_fetch(0, src, i, j, tmp);
   dst[0] = ubyte_to_float(tmp[0]);
   dst[1] = ubyte_to_float(tmp[1]);
   dst[2] = ubyte_to_float(tmp[2]);
   dst[3] = 1.0;
}

void
util_format_dxt1_rgba_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt1_rgba_fetch(0, src, i, j, tmp);
   dst[0] = ubyte_to_float(tmp[0]);
   dst[1] = ubyte_to_float(tmp[1]);
   dst[2] = ubyte_to_float(tmp[2]);
   dst[3] = ubyte_to_float(tmp[3]);
}

void
util_format_dxt3_rgba_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt3_rgba_fetch(0, src, i, j, tmp);
   dst[0] = ubyte_to_float(tmp[0]);
   dst[1] = ubyte_to_float(tmp[1]);
   dst[2] = ubyte_to_float(tmp[2]);
   dst[3] = ubyte_to_float(tmp[3]);
}

void
util_format_dxt5_rgba_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt5_rgba_fetch(0, src, i, j, tmp);
   dst[0] = ubyte_to_float(tmp[0]);
   dst[1] = ubyte_to_float(tmp[1]);
   dst[2] = ubyte_to_float(tmp[2]);
   dst[3] = ubyte_to_float(tmp[3]);
}


/*
 * Block decompression.
 */

static inline void
util_format_dxtn_rgb_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height,
                                        util_format_dxtn_fetch_t fetch,
                                        unsigned block_size, bool srgb)
{
   const unsigned bw = 4, bh = 4, comps = 4;
   unsigned x, y, i, j;
   for(y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, bh);
      for(x = 0; x < width; x += bw) {
         const unsigned w = MIN2(width - x, bw);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               uint8_t *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*comps;
               fetch(0, src, i, j, dst);
               if (srgb) {
                  dst[0] = util_format_srgb_to_linear_8unorm(dst[0]);
                  dst[1] = util_format_srgb_to_linear_8unorm(dst[1]);
                  dst[2] = util_format_srgb_to_linear_8unorm(dst[2]);
               }
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_dxt1_rgb_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt1_rgb_fetch,
                                           8, false);
}

void
util_format_dxt1_rgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt1_rgba_fetch,
                                           8, false);
}

void
util_format_dxt3_rgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt3_rgba_fetch,
                                           16, false);
}

void
util_format_dxt5_rgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt5_rgba_fetch,
                                           16, false);
}

static inline void
util_format_dxtn_rgb_unpack_rgba_float(float *restrict dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height,
                                       util_format_dxtn_fetch_t fetch,
                                       unsigned block_size, bool srgb)
{
   unsigned x, y, i, j;
   for(y = 0; y < height; y += 4) {
      const uint8_t *src = src_row;
      for(x = 0; x < width; x += 4) {
         for(j = 0; j < 4; ++j) {
            for(i = 0; i < 4; ++i) {
               float *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*4;
               uint8_t tmp[4];
               fetch(0, src, i, j, tmp);
               if (srgb) {
                  dst[0] = util_format_srgb_8unorm_to_linear_float(tmp[0]);
                  dst[1] = util_format_srgb_8unorm_to_linear_float(tmp[1]);
                  dst[2] = util_format_srgb_8unorm_to_linear_float(tmp[2]);
               }
               else {
                  dst[0] = ubyte_to_float(tmp[0]);
                  dst[1] = ubyte_to_float(tmp[1]);
                  dst[2] = ubyte_to_float(tmp[2]);
               }
               dst[3] = ubyte_to_float(tmp[3]);
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_dxt1_rgb_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt1_rgb_fetch,
                                          8, false);
}

void
util_format_dxt1_rgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt1_rgba_fetch,
                                          8, false);
}

void
util_format_dxt3_rgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt3_rgba_fetch,
                                          16, false);
}

void
util_format_dxt5_rgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt5_rgba_fetch,
                                          16, false);
}


/*
 * Block compression.
 */

static inline void
util_format_dxtn_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                  const uint8_t *restrict src, unsigned src_stride,
                                  unsigned width, unsigned height,
                                  enum util_format_dxtn format,
                                  unsigned block_size, bool srgb)
{
   const unsigned bw = 4, bh = 4, comps = 4;
   unsigned x, y, i, j, k;
   for(y = 0; y < height; y += bh) {
      uint8_t *dst = dst_row;
      for(x = 0; x < width; x += bw) {
         uint8_t tmp[4][4][4];  /* [bh][bw][comps] */
         for(j = 0; j < bh; ++j) {
            for(i = 0; i < bw; ++i) {
               uint8_t src_tmp;
               for(k = 0; k < 3; ++k) {
                  src_tmp = src[(y + j)*src_stride/sizeof(*src) + (x+i)*comps + k];
                  if (srgb) {
                     tmp[j][i][k] = util_format_linear_to_srgb_8unorm(src_tmp);
                  }
                  else {
                     tmp[j][i][k] = src_tmp;
                  }
               }
               /* for sake of simplicity there's an unneeded 4th component for dxt1_rgb */
               tmp[j][i][3] = src[(y + j)*src_stride/sizeof(*src) + (x+i)*comps + 3];
            }
         }
         /* even for dxt1_rgb have 4 src comps */
         util_format_dxtn_pack(4, 4, 4, &tmp[0][0][0], format, dst, 0);
         dst += block_size;
      }
      dst_row += dst_stride / sizeof(*dst_row);
   }

}

void
util_format_dxt1_rgb_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                      const uint8_t *restrict src, unsigned src_stride,
                                      unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src, src_stride,
                                     width, height, UTIL_FORMAT_DXT1_RGB,
                                     8, false);
}

void
util_format_dxt1_rgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src, src_stride,
                                     width, height, UTIL_FORMAT_DXT1_RGBA,
                                     8, false);
}

void
util_format_dxt3_rgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src, src_stride,
                                     width, height, UTIL_FORMAT_DXT3_RGBA,
                                     16, false);
}

void
util_format_dxt5_rgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src, src_stride,
                                     width, height, UTIL_FORMAT_DXT5_RGBA,
                                     16, false);
}

static inline void
util_format_dxtn_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                 const float *restrict src, unsigned src_stride,
                                 unsigned width, unsigned height,
                                 enum util_format_dxtn format,
                                 unsigned block_size, bool srgb)
{
   unsigned x, y, i, j, k;
   for(y = 0; y < height; y += 4) {
      uint8_t *dst = dst_row;
      for(x = 0; x < width; x += 4) {
         uint8_t tmp[4][4][4];
         for(j = 0; j < 4; ++j) {
            for(i = 0; i < 4; ++i) {
               float src_tmp;
               for(k = 0; k < 3; ++k) {
                  src_tmp = src[(y + j)*src_stride/sizeof(*src) + (x+i)*4 + k];
                  if (srgb) {
                     tmp[j][i][k] = util_format_linear_float_to_srgb_8unorm(src_tmp);
                  }
                  else {
                     tmp[j][i][k] = float_to_ubyte(src_tmp);
                  }
               }
               /* for sake of simplicity there's an unneeded 4th component for dxt1_rgb */
               src_tmp = src[(y + j)*src_stride/sizeof(*src) + (x+i)*4 + 3];
               tmp[j][i][3] = float_to_ubyte(src_tmp);
            }
         }
         util_format_dxtn_pack(4, 4, 4, &tmp[0][0][0], format, dst, 0);
         dst += block_size;
      }
      dst_row += 4*dst_stride/sizeof(*dst_row);
   }
}

void
util_format_dxt1_rgb_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                     const float *src, unsigned src_stride,
                                     unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src, src_stride,
                                    width, height, UTIL_FORMAT_DXT1_RGB,
                                    8, false);
}

void
util_format_dxt1_rgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                      const float *src, unsigned src_stride,
                                      unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src, src_stride,
                                    width, height, UTIL_FORMAT_DXT1_RGBA,
                                    8, false);
}

void
util_format_dxt3_rgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                      const float *src, unsigned src_stride,
                                      unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src, src_stride,
                                    width, height, UTIL_FORMAT_DXT3_RGBA,
                                    16, false);
}

void
util_format_dxt5_rgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                      const float *src, unsigned src_stride,
                                      unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src, src_stride,
                                    width, height, UTIL_FORMAT_DXT5_RGBA,
                                    16, false);
}


/*
 * SRGB variants.
 */

void
util_format_dxt1_srgb_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   uint8_t tmp[4];
   util_format_dxt1_rgb_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_to_linear_8unorm(tmp[0]);
   dst[1] = util_format_srgb_to_linear_8unorm(tmp[1]);
   dst[2] = util_format_srgb_to_linear_8unorm(tmp[2]);
   dst[3] = 255;
}

void
util_format_dxt1_srgba_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   uint8_t tmp[4];
   util_format_dxt1_rgba_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_to_linear_8unorm(tmp[0]);
   dst[1] = util_format_srgb_to_linear_8unorm(tmp[1]);
   dst[2] = util_format_srgb_to_linear_8unorm(tmp[2]);
   dst[3] = tmp[3];
}

void
util_format_dxt3_srgba_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   uint8_t tmp[4];
   util_format_dxt3_rgba_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_to_linear_8unorm(tmp[0]);
   dst[1] = util_format_srgb_to_linear_8unorm(tmp[1]);
   dst[2] = util_format_srgb_to_linear_8unorm(tmp[2]);
   dst[3] = tmp[3];
}

void
util_format_dxt5_srgba_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   uint8_t tmp[4];
   util_format_dxt5_rgba_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_to_linear_8unorm(tmp[0]);
   dst[1] = util_format_srgb_to_linear_8unorm(tmp[1]);
   dst[2] = util_format_srgb_to_linear_8unorm(tmp[2]);
   dst[3] = tmp[3];
}

void
util_format_dxt1_srgb_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt1_rgb_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_8unorm_to_linear_float(tmp[0]);
   dst[1] = util_format_srgb_8unorm_to_linear_float(tmp[1]);
   dst[2] = util_format_srgb_8unorm_to_linear_float(tmp[2]);
   dst[3] = 1.0f;
}

void
util_format_dxt1_srgba_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt1_rgba_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_8unorm_to_linear_float(tmp[0]);
   dst[1] = util_format_srgb_8unorm_to_linear_float(tmp[1]);
   dst[2] = util_format_srgb_8unorm_to_linear_float(tmp[2]);
   dst[3] = ubyte_to_float(tmp[3]);
}

void
util_format_dxt3_srgba_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt3_rgba_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_8unorm_to_linear_float(tmp[0]);
   dst[1] = util_format_srgb_8unorm_to_linear_float(tmp[1]);
   dst[2] = util_format_srgb_8unorm_to_linear_float(tmp[2]);
   dst[3] = ubyte_to_float(tmp[3]);
}

void
util_format_dxt5_srgba_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   util_format_dxt5_rgba_fetch(0, src, i, j, tmp);
   dst[0] = util_format_srgb_8unorm_to_linear_float(tmp[0]);
   dst[1] = util_format_srgb_8unorm_to_linear_float(tmp[1]);
   dst[2] = util_format_srgb_8unorm_to_linear_float(tmp[2]);
   dst[3] = ubyte_to_float(tmp[3]);
}

void
util_format_dxt1_srgb_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt1_rgb_fetch,
                                           8, true);
}

void
util_format_dxt1_srgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt1_rgba_fetch,
                                           8, true);
}

void
util_format_dxt3_srgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt3_rgba_fetch,
                                           16, true);
}

void
util_format_dxt5_srgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           util_format_dxt5_rgba_fetch,
                                           16, true);
}

void
util_format_dxt1_srgb_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt1_rgb_fetch,
                                          8, true);
}

void
util_format_dxt1_srgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt1_rgba_fetch,
                                          8, true);
}

void
util_format_dxt3_srgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt3_rgba_fetch,
                                          16, true);
}

void
util_format_dxt5_srgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          util_format_dxt5_rgba_fetch,
                                          16, true);
}

void
util_format_dxt1_srgb_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src_row, src_stride,
                                     width, height, UTIL_FORMAT_DXT1_RGB,
                                     8, true);
}

void
util_format_dxt1_srgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src_row, src_stride,
                                     width, height, UTIL_FORMAT_DXT1_RGBA,
                                     8, true);
}

void
util_format_dxt3_srgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src_row, src_stride,
                                     width, height, UTIL_FORMAT_DXT3_RGBA,
                                     16, true);
}

void
util_format_dxt5_srgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_8unorm(dst_row, dst_stride, src_row, src_stride,
                                     width, height, UTIL_FORMAT_DXT5_RGBA,
                                     16, true);
}

void
util_format_dxt1_srgb_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src_row, src_stride,
                                    width, height, UTIL_FORMAT_DXT1_RGB,
                                    8, true);
}

void
util_format_dxt1_srgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src_row, src_stride,
                                    width, height, UTIL_FORMAT_DXT1_RGBA,
                                    8, true);
}

void
util_format_dxt3_srgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src_row, src_stride,
                                    width, height, UTIL_FORMAT_DXT3_RGBA,
                                    16, true);
}

void
util_format_dxt5_srgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_dxtn_pack_rgba_float(dst_row, dst_stride, src_row, src_stride,
                                    width, height, UTIL_FORMAT_DXT5_RGBA,
                                    16, true);
}

