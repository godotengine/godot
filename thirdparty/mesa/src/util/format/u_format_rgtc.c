/**************************************************************************
 *
 * Copyright (C) 2011 Red Hat Inc.
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

#include <stdio.h>
#include "util/format/u_format.h"
#include "util/format/u_format_rgtc.h"
#include "util/u_math.h"
#include "util/rgtc.h"

void
util_format_rgtc1_unorm_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   util_format_unsigned_fetch_texel_rgtc(0, src, i, j, dst, 1);
   dst[1] = 0;
   dst[2] = 0;
   dst[3] = 255;
}

void
util_format_rgtc1_unorm_unpack_r_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, comps = 1;
   unsigned x, y, i, j;
   unsigned block_size = 8;

   for(y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, bh);
      for(x = 0; x < width; x += bw) {
         const unsigned w = MIN2(width - x, bw);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               uint8_t *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*comps;
               util_format_unsigned_fetch_texel_rgtc(0, src, i, j, dst, 1);
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc1_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, comps = 4;
   unsigned x, y, i, j;
   unsigned block_size = 8;

   for(y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, bh);
      for(x = 0; x < width; x += bw) {
         const unsigned w = MIN2(width - x, bw);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               uint8_t *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*comps;
               util_format_unsigned_fetch_texel_rgtc(0, src, i, j, dst, 1);
               dst[1] = 0;
               dst[2] = 0;
               dst[3] = 255;
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc1_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row,
					 unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, bytes_per_block = 8;
   unsigned x, y, i, j;

   for(y = 0; y < height; y += bh) {
      uint8_t *dst = dst_row;
      for(x = 0; x < width; x += bw) {
         uint8_t tmp[4][4];  /* [bh][bw][comps] */
         for(j = 0; j < bh; ++j) {
            for(i = 0; i < bw; ++i) {
               tmp[j][i] = src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4];
            }
         }
         util_format_unsigned_encode_rgtc_ubyte(dst, tmp, 4, 4);
         dst += bytes_per_block;
      }
      dst_row += dst_stride / sizeof(*dst_row);
   }
}

void
util_format_rgtc1_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   unsigned x, y, i, j;
   int block_size = 8;
   for(y = 0; y < height; y += 4) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, 4);
      for(x = 0; x < width; x += 4) {
         const unsigned w = MIN2(width - x, 4);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               float *dst = (float *)((uint8_t *)dst_row + (y + j)*dst_stride + (x + i)*16);
               uint8_t tmp_r;
               util_format_unsigned_fetch_texel_rgtc(0, src, i, j, &tmp_r, 1);
               dst[0] = ubyte_to_float(tmp_r);
               dst[1] = 0.0;
               dst[2] = 0.0;
               dst[3] = 1.0;
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc1_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, bytes_per_block = 8;
   unsigned x, y, i, j;

   for(y = 0; y < height; y += bh) {
      uint8_t *dst = dst_row;
      for(x = 0; x < width; x += bw) {
         uint8_t tmp[4][4];  /* [bh][bw][comps] */
         for(j = 0; j < bh; ++j) {
            for(i = 0; i < bw; ++i) {
               tmp[j][i] = float_to_ubyte(src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4]);
            }
         }
         util_format_unsigned_encode_rgtc_ubyte(dst, tmp, 4, 4);
         dst += bytes_per_block;
      }
      dst_row += dst_stride / sizeof(*dst_row);
   }
}

void
util_format_rgtc1_unorm_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp_r;
   util_format_unsigned_fetch_texel_rgtc(0, src, i, j, &tmp_r, 1);
   dst[0] = ubyte_to_float(tmp_r);
   dst[1] = 0.0;
   dst[2] = 0.0;
   dst[3] = 1.0;
}

void
util_format_rgtc1_snorm_fetch_rgba_8unorm(UNUSED uint8_t *restrict dst, UNUSED const uint8_t *restrict src,
                                          UNUSED unsigned i, UNUSED unsigned j)
{
   fprintf(stderr,"%s\n", __func__);
}

void
util_format_rgtc1_snorm_unpack_rgba_8unorm(UNUSED uint8_t *restrict dst_row, UNUSED unsigned dst_stride,
                                           UNUSED const uint8_t *restrict src_row, UNUSED unsigned src_stride,
                                           UNUSED unsigned width, UNUSED unsigned height)
{
   fprintf(stderr,"%s\n", __func__);
}

void
util_format_rgtc1_snorm_unpack_r_8snorm(int8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, comps = 1;
   unsigned x, y, i, j;
   unsigned block_size = 8;

   for(y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, bh);
      for(x = 0; x < width; x += bw) {
         const unsigned w = MIN2(width - x, bw);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               int8_t *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*comps;
               util_format_signed_fetch_texel_rgtc(0, (const int8_t *)src, i, j, dst, 1);
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc1_snorm_pack_rgba_8unorm(UNUSED uint8_t *restrict dst_row, UNUSED unsigned dst_stride,
                                         UNUSED const uint8_t *restrict src_row, UNUSED unsigned src_stride,
                                         UNUSED unsigned width, UNUSED unsigned height)
{
   fprintf(stderr,"%s\n", __func__);
}

void
util_format_rgtc1_snorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, bytes_per_block = 8;
   unsigned x, y, i, j;

   for(y = 0; y < height; y += bh) {
      int8_t *dst = (int8_t *)dst_row;
      for(x = 0; x < width; x += bw) {
         int8_t tmp[4][4];  /* [bh][bw][comps] */
         for(j = 0; j < bh; ++j) {
            for(i = 0; i < bw; ++i) {
               tmp[j][i] = float_to_byte_tex(src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4]);
            }
         }
         util_format_signed_encode_rgtc_ubyte(dst, tmp, 4, 4);
         dst += bytes_per_block;
      }
      dst_row += dst_stride / sizeof(*dst_row);
   }
}

void
util_format_rgtc1_snorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   unsigned x, y, i, j;
   int block_size = 8;
   for(y = 0; y < height; y += 4) {
      const int8_t *src = (int8_t *)src_row;
      const unsigned h = MIN2(height - y, 4);
      for(x = 0; x < width; x += 4) {
         const unsigned w = MIN2(width - x, 4);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               float *dst = (float *)((uint8_t *)dst_row + (y + j)*dst_stride + (x + i)*16);
               int8_t tmp_r;
               util_format_signed_fetch_texel_rgtc(0, src, i, j, &tmp_r, 1);
               dst[0] = byte_to_float_tex(tmp_r);
               dst[1] = 0.0;
               dst[2] = 0.0;
               dst[3] = 1.0;
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc1_snorm_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   int8_t tmp_r;
   util_format_signed_fetch_texel_rgtc(0, (int8_t *)src, i, j, &tmp_r, 1);
   dst[0] = byte_to_float_tex(tmp_r);
   dst[1] = 0.0;
   dst[2] = 0.0;
   dst[3] = 1.0;
}


void
util_format_rgtc2_unorm_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   util_format_unsigned_fetch_texel_rgtc(0, src, i, j, dst, 2);
   util_format_unsigned_fetch_texel_rgtc(0, src + 8, i, j, dst + 1, 2);
   dst[2] = 0;
   dst[3] = 255;
}

void
util_format_rgtc2_unorm_unpack_rg_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, comps = 2;
   unsigned x, y, i, j;
   unsigned block_size = 16;

   for(y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, bh);
      for(x = 0; x < width; x += bw) {
         const unsigned w = MIN2(width - x, bw);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               uint8_t *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*comps;
               util_format_unsigned_fetch_texel_rgtc(0, src, i, j, dst, 2);
               util_format_unsigned_fetch_texel_rgtc(0, src + 8, i, j, dst + 1, 2);
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc2_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, comps = 4;
   unsigned x, y, i, j;
   unsigned block_size = 16;

   for(y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, bh);
      for(x = 0; x < width; x += bw) {
         const unsigned w = MIN2(width - x, bw);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               uint8_t *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*comps;
               util_format_unsigned_fetch_texel_rgtc(0, src, i, j, dst, 2);
               util_format_unsigned_fetch_texel_rgtc(0, src + 8, i, j, dst + 1, 2);
               dst[2] = 0;
               dst[3] = 255;
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc2_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, bytes_per_block = 16;
   unsigned x, y, i, j;

   for(y = 0; y < height; y += bh) {
      uint8_t *dst = dst_row;
      for(x = 0; x < width; x += bw) {
         uint8_t tmp_r[4][4];  /* [bh][bw] */
         uint8_t tmp_g[4][4];  /* [bh][bw] */
         for(j = 0; j < bh; ++j) {
            for(i = 0; i < bw; ++i) {
               tmp_r[j][i] = src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4];
               tmp_g[j][i] = src_row[((y + j)*src_stride/sizeof(*src_row) + (x + i)*4) + 1];
            }
         }
         util_format_unsigned_encode_rgtc_ubyte(dst, tmp_r, 4, 4);
         util_format_unsigned_encode_rgtc_ubyte(dst + 8, tmp_g, 4, 4);
         dst += bytes_per_block;
      }
      dst_row += dst_stride / sizeof(*dst_row);
   }
}

void
util_format_rxtc2_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height, unsigned chan2off)
{
   const unsigned bw = 4, bh = 4, bytes_per_block = 16;
   unsigned x, y, i, j;

   for(y = 0; y < height; y += bh) {
      uint8_t *dst = dst_row;
      for(x = 0; x < width; x += bw) {
         uint8_t tmp_r[4][4];  /* [bh][bw][comps] */
         uint8_t tmp_g[4][4];  /* [bh][bw][comps] */
         for(j = 0; j < bh; ++j) {
            for(i = 0; i < bw; ++i) {
               tmp_r[j][i] = float_to_ubyte(src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4]);
               tmp_g[j][i] = float_to_ubyte(src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4 + chan2off]);
            }
         }
         util_format_unsigned_encode_rgtc_ubyte(dst, tmp_r, 4, 4);
         util_format_unsigned_encode_rgtc_ubyte(dst + 8, tmp_g, 4, 4);
         dst += bytes_per_block;
      }
      dst_row += dst_stride / sizeof(*dst_row);
   }
}

void
util_format_rgtc2_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_rxtc2_unorm_pack_rgba_float(dst_row, dst_stride, src_row, src_stride, width, height, 1);
}

void
util_format_rgtc2_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   unsigned x, y, i, j;
   int block_size = 16;
   for(y = 0; y < height; y += 4) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, 4);
      for(x = 0; x < width; x += 4) {
         const unsigned w = MIN2(width - x, 4);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               float *dst = (float *)((uint8_t *)dst_row + (y + j)*dst_stride + (x + i)*16);
               uint8_t tmp_r, tmp_g;
               util_format_unsigned_fetch_texel_rgtc(0, src, i, j, &tmp_r, 2);
               util_format_unsigned_fetch_texel_rgtc(0, src + 8, i, j, &tmp_g, 2);
               dst[0] = ubyte_to_float(tmp_r);
               dst[1] = ubyte_to_float(tmp_g);
               dst[2] = 0.0;
               dst[3] = 1.0;
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc2_unorm_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp_r, tmp_g;
   util_format_unsigned_fetch_texel_rgtc(0, src, i, j, &tmp_r, 2);
   util_format_unsigned_fetch_texel_rgtc(0, src + 8, i, j, &tmp_g, 2);
   dst[0] = ubyte_to_float(tmp_r);
   dst[1] = ubyte_to_float(tmp_g);
   dst[2] = 0.0;
   dst[3] = 1.0;
}


void
util_format_rgtc2_snorm_fetch_rgba_8unorm(UNUSED uint8_t *restrict dst, UNUSED const uint8_t *restrict src,
                                          UNUSED unsigned i, UNUSED unsigned j)
{
   fprintf(stderr,"%s\n", __func__);
}

void
util_format_rgtc2_snorm_unpack_rgba_8unorm(UNUSED uint8_t *restrict dst_row, UNUSED unsigned dst_stride,
                                           UNUSED const uint8_t *restrict src_row, UNUSED unsigned src_stride,
                                           UNUSED unsigned width, UNUSED unsigned height)
{
   fprintf(stderr,"%s\n", __func__);
}

void
util_format_rgtc2_snorm_unpack_rg_8snorm(int8_t *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, comps = 2;
   unsigned x, y, i, j;
   unsigned block_size = 16;

   for(y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      const unsigned h = MIN2(height - y, bh);
      for(x = 0; x < width; x += bw) {
         const unsigned w = MIN2(width - x, bw);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               int8_t *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i)*comps;
               util_format_signed_fetch_texel_rgtc(0, (const int8_t *)src, i, j, (int8_t *)dst, 2);
               util_format_signed_fetch_texel_rgtc(0, (const int8_t *)src + 8, i, j, (int8_t *)dst + 1, 2);
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rgtc2_snorm_pack_rgba_8unorm(UNUSED uint8_t *restrict dst_row, UNUSED unsigned dst_stride,
                                         UNUSED const uint8_t *restrict src_row, UNUSED unsigned src_stride,
                                         UNUSED unsigned width, UNUSED unsigned height)
{
   fprintf(stderr,"%s\n", __func__);
}

void
util_format_rgtc2_snorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   unsigned x, y, i, j;
   int block_size = 16;
   for(y = 0; y < height; y += 4) {
      const int8_t *src = (int8_t *)src_row;
      const unsigned h = MIN2(height - y, 4);
      for(x = 0; x < width; x += 4) {
         const unsigned w = MIN2(width - x, 4);
         for(j = 0; j < h; ++j) {
            for(i = 0; i < w; ++i) {
               float *dst = (float *)((uint8_t *)dst_row + (y + j)*dst_stride + (x + i)*16);
               int8_t tmp_r, tmp_g;
               util_format_signed_fetch_texel_rgtc(0, src, i, j, &tmp_r, 2);
               util_format_signed_fetch_texel_rgtc(0, src + 8, i, j, &tmp_g, 2);
               dst[0] = byte_to_float_tex(tmp_r);
               dst[1] = byte_to_float_tex(tmp_g);
               dst[2] = 0.0;
               dst[3] = 1.0;
            }
         }
         src += block_size;
      }
      src_row += src_stride;
   }
}

void
util_format_rxtc2_snorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height, unsigned chan2off)
{
   const unsigned bw = 4, bh = 4, bytes_per_block = 16;
   unsigned x, y, i, j;

   for(y = 0; y < height; y += bh) {
      int8_t *dst = (int8_t *)dst_row;
      for(x = 0; x < width; x += bw) {
         int8_t tmp_r[4][4];  /* [bh][bw][comps] */
         int8_t tmp_g[4][4];  /* [bh][bw][comps] */
         for(j = 0; j < bh; ++j) {
            for(i = 0; i < bw; ++i) {
               tmp_r[j][i] = float_to_byte_tex(src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4]);
               tmp_g[j][i] = float_to_byte_tex(src_row[(y + j)*src_stride/sizeof(*src_row) + (x + i)*4 + chan2off]);
            }
         }
         util_format_signed_encode_rgtc_ubyte(dst, tmp_r, 4, 4);
         util_format_signed_encode_rgtc_ubyte(dst + 8, tmp_g, 4, 4);
         dst += bytes_per_block;
      }
      dst_row += dst_stride / sizeof(*dst_row);
   }
}

void
util_format_rgtc2_snorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride, const float *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   util_format_rxtc2_snorm_pack_rgba_float(dst_row, dst_stride, src_row, src_stride, width, height, 1);
}

void
util_format_rgtc2_snorm_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   int8_t tmp_r, tmp_g;
   util_format_signed_fetch_texel_rgtc(0, (int8_t *)src, i, j, &tmp_r, 2);
   util_format_signed_fetch_texel_rgtc(0, (int8_t *)src + 8, i, j, &tmp_g, 2);
   dst[0] = byte_to_float_tex(tmp_r);
   dst[1] = byte_to_float_tex(tmp_g);
   dst[2] = 0.0;
   dst[3] = 1.0;
}

