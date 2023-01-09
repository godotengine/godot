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
#include "util/format/u_format_bptc.h"
#include "u_format_pack.h"
#include "util/format_srgb.h"
#include "util/u_math.h"

#include "util/format/texcompress_bptc_tmp.h"

void
util_format_bptc_rgba_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                               const uint8_t *restrict src_row, unsigned src_stride,
                                               unsigned width, unsigned height)
{
  decompress_rgba_unorm(width, height,
                        src_row, src_stride,
                        dst_row, dst_stride);
}

void
util_format_bptc_rgba_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                             const uint8_t *restrict src_row, unsigned src_stride,
                                             unsigned width, unsigned height)
{
   compress_rgba_unorm(width, height,
                       src_row, src_stride,
                       dst_row, dst_stride);
}

void
util_format_bptc_rgba_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                              const uint8_t *restrict src_row, unsigned src_stride,
                                              unsigned width, unsigned height)
{
   uint8_t *temp_block;
   temp_block = malloc(width * height * 4 * sizeof(uint8_t));
   decompress_rgba_unorm(width, height,
                         src_row, src_stride,
                         temp_block, width * 4 * sizeof(uint8_t));
   /* Direct call to row unpack instead of util_format_rgba_unpack_rect()
    * to avoid table lookup that would pull in all unpack symbols.
    */
   for (int y = 0; y < height; y++) {
      util_format_r8g8b8a8_unorm_unpack_rgba_float((char *)dst_row + dst_stride * y,
                                                    temp_block + 4 * width * y,
                                                    width);
   }
   free((void *) temp_block);
}

void
util_format_bptc_rgba_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                            const float *restrict src_row, unsigned src_stride,
                                            unsigned width, unsigned height)
{
   uint8_t *temp_block;
   temp_block = malloc(width * height * 4 * sizeof(uint8_t));
   /* Direct call to row unpack instead of util_format_rgba_unpack_rect()
    * to avoid table lookup that would pull in all unpack symbols.
    */
   for (int y = 0; y < height; y++) {
      util_format_r32g32b32a32_float_unpack_rgba_8unorm(
                        temp_block + 4 * width * y,
                        (uint8_t *)src_row + src_stride * y,
                        width);
   }
   compress_rgba_unorm(width, height,
                       temp_block, width * 4 * sizeof(uint8_t),
                       dst_row, dst_stride);
   free((void *) temp_block);
}

void
util_format_bptc_rgba_unorm_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                             unsigned width, unsigned height)
{
   uint8_t temp_block[4];

   fetch_rgba_unorm_from_block(src + ((width * sizeof(uint8_t)) * (height / 4) + (width / 4)) * 16,
                               temp_block, (width % 4) + (height % 4) * 4);

   util_format_read_4(PIPE_FORMAT_R8G8B8A8_UNORM,
                      dst, 4 * sizeof(float),
                      temp_block, 4 * sizeof(uint8_t),
                      0, 0, 1, 1);
}

void
util_format_bptc_srgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                          const uint8_t *restrict src_row, unsigned src_stride,
                                          unsigned width, unsigned height)
{
   decompress_rgba_unorm(width, height,
                         src_row, src_stride,
                         dst_row, dst_stride);
}

void
util_format_bptc_srgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   compress_rgba_unorm(width, height,
                       src_row, src_stride,
                       dst_row, dst_stride);
}

void
util_format_bptc_srgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   uint8_t *temp_block;
   temp_block = malloc(width * height * 4 * sizeof(uint8_t));
   decompress_rgba_unorm(width, height,
                         src_row, src_stride,
                         temp_block, width * 4 * sizeof(uint8_t));

   /* Direct call to row unpack instead of util_format_rgba_unpack_rect()
    * to avoid table lookup that would pull in all unpack symbols.
    */
   for (int y = 0; y < height; y++) {
      util_format_r8g8b8a8_srgb_unpack_rgba_float((char *)dst_row + dst_stride * y,
                                                  temp_block + width * 4 * y,
                                                  width);
   }

   free((void *) temp_block);
}

void
util_format_bptc_srgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const float *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   compress_rgb_float(width, height,
                      src_row, src_stride,
                      dst_row, dst_stride,
                      true);
}

void
util_format_bptc_srgba_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                        unsigned width, unsigned height)
{
   uint8_t temp_block[4];

   fetch_rgba_unorm_from_block(src + ((width * sizeof(uint8_t)) * (height / 4) + (width / 4)) * 16,
                               temp_block, (width % 4) + (height % 4) * 4);
   util_format_r8g8b8a8_srgb_fetch_rgba(dst, temp_block, 0, 0);
}

void
util_format_bptc_rgb_float_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                              const uint8_t *restrict src_row, unsigned src_stride,
                                              unsigned width, unsigned height)
{
   float *temp_block;
   temp_block = malloc(width * height * 4 * sizeof(float));
   decompress_rgb_float(width, height,
                        src_row, src_stride,
                        temp_block, width * 4 * sizeof(float),
                        true);
   /* Direct call to row unpack instead of util_format_rgba_unpack_rect()
    * to avoid table lookup that would pull in all unpack symbols.
    */
   for (int y = 0; y < height; y++) {
      util_format_r32g32b32a32_float_unpack_rgba_8unorm(
          dst_row + dst_stride * y,
          (const uint8_t *)temp_block + width * 4 * sizeof(float) * y,
          width);
   }
   free((void *) temp_block);
}

void
util_format_bptc_rgb_float_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                            const uint8_t *restrict src_row, unsigned src_stride,
                                            unsigned width, unsigned height)
{
   compress_rgba_unorm(width, height,
                       src_row, src_stride,
                       dst_row, dst_stride);
}

void
util_format_bptc_rgb_float_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                             const uint8_t *restrict src_row, unsigned src_stride,
                                             unsigned width, unsigned height)
{
   decompress_rgb_float(width, height,
                        src_row, src_stride,
                        dst_row, dst_stride,
                        true);
}

void
util_format_bptc_rgb_float_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                           const float *restrict src_row, unsigned src_stride,
                                           unsigned width, unsigned height)
{
   compress_rgb_float(width, height,
                      src_row, src_stride,
                      dst_row, dst_stride,
                      true);
}

void
util_format_bptc_rgb_float_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                            unsigned width, unsigned height)
{
   fetch_rgb_float_from_block(src + ((width * sizeof(uint8_t)) * (height / 4) + (width / 4)) * 16,
                              dst, (width % 4) + (height % 4) * 4, true);
}

void
util_format_bptc_rgb_ufloat_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                               const uint8_t *restrict src_row, unsigned src_stride,
                                               unsigned width, unsigned height)
{
   float *temp_block;
   temp_block = malloc(width * height * 4 * sizeof(float));
   decompress_rgb_float(width, height,
                        src_row, src_stride,
                        temp_block, width * 4 * sizeof(float),
                        false);
   /* Direct call to row unpack instead of util_format_rgba_unpack_8unorm()
    * to avoid table lookup that would pull in all unpack symbols.
    */
   for (int y = 0; y < height; y++) {
      util_format_r32g32b32a32_float_unpack_rgba_8unorm(dst_row + dst_stride * y,
                                                        (void *)(temp_block + 4 * width * y),
                                                        width);
   }
   free((void *) temp_block);
}

void
util_format_bptc_rgb_ufloat_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                             const uint8_t *restrict src_row, unsigned src_stride,
                                             unsigned width, unsigned height)
{
   compress_rgba_unorm(width, height,
                       src_row, src_stride,
                       dst_row, dst_stride);
}

void
util_format_bptc_rgb_ufloat_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                              const uint8_t *restrict src_row, unsigned src_stride,
                                              unsigned width, unsigned height)
{
   decompress_rgb_float(width, height,
                        src_row, src_stride,
                        dst_row, dst_stride,
                        false);
}

void
util_format_bptc_rgb_ufloat_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                            const float *restrict src_row, unsigned src_stride,
                                            unsigned width, unsigned height)
{
   compress_rgb_float(width, height,
                      src_row, src_stride,
                      dst_row, dst_stride,
                      false);
}

void
util_format_bptc_rgb_ufloat_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                             unsigned width, unsigned height)
{
   fetch_rgb_float_from_block(src + ((width * sizeof(uint8_t)) * (height / 4) + (width / 4)) * 16,
                              dst, (width % 4) + (height % 4) * 4, false);
}
