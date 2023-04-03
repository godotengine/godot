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
 * YUV and RGB subsampled formats conversion.
 *
 * @author Jose Fonseca <jfonseca@vmware.com>
 */


#include "util/u_debug.h"
#include "util/format/u_format_yuv.h"


void
util_format_r8g8_b8g8_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      float *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      float r, g0, g1, b;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         r  = ubyte_to_float((value >>  0) & 0xff);
         g0 = ubyte_to_float((value >>  8) & 0xff);
         b  = ubyte_to_float((value >> 16) & 0xff);
         g1 = ubyte_to_float((value >> 24) & 0xff);

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 1.0f; /* a */
         dst += 4;

         dst[0] = r;    /* r */
         dst[1] = g1;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 1.0f; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         r  = ubyte_to_float((value >>  0) & 0xff);
         g0 = ubyte_to_float((value >>  8) & 0xff);
         b  = ubyte_to_float((value >> 16) & 0xff);
         g1 = ubyte_to_float((value >> 24) & 0xff);

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 1.0f; /* a */
      }

      src_row = (uint8_t *)src_row + src_stride;
      dst_row = (uint8_t *)dst_row + dst_stride;
   }
}


void
util_format_r8g8_b8g8_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                          const uint8_t *restrict src_row, unsigned src_stride,
                                          unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      uint8_t *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      uint8_t r, g0, g1, b;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         r  = (value >>  0) & 0xff;
         g0 = (value >>  8) & 0xff;
         b  = (value >> 16) & 0xff;
         g1 = (value >> 24) & 0xff;

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 0xff; /* a */
         dst += 4;

         dst[0] = r;    /* r */
         dst[1] = g1;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 0xff; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         r  = (value >>  0) & 0xff;
         g0 = (value >>  8) & 0xff;
         b  = (value >> 16) & 0xff;
         g1 = (value >> 24) & 0xff;

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 0xff; /* a */
      }

      src_row += src_stride/sizeof(*src_row);
      dst_row += dst_stride/sizeof(*dst_row);
   }
}


void
util_format_r8g8_b8g8_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const float *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const float *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      float r, g0, g1, b;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         r  = 0.5f*(src[0] + src[4]);
         g0 = src[1];
         g1 = src[5];
         b  = 0.5f*(src[2] + src[6]);

         value  = (uint32_t)float_to_ubyte(r);
         value |= (uint32_t)float_to_ubyte(g0) <<  8;
         value |= (uint32_t)float_to_ubyte(b)  << 16;
         value |= (uint32_t)float_to_ubyte(g1) << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         r  = src[0];
         g0 = src[1];
         g1 = 0;
         b  = src[2];

         value  = (uint32_t)float_to_ubyte(r);
         value |= (uint32_t)float_to_ubyte(g0) <<  8;
         value |= (uint32_t)float_to_ubyte(b)  << 16;
         value |= (uint32_t)float_to_ubyte(g1) << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_r8g8_b8g8_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const uint8_t *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      uint32_t r, g0, g1, b;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         r  = (src[0] + src[4] + 1) >> 1;
         g0 = src[1];
         g1 = src[5];
         b  = (src[2] + src[6] + 1) >> 1;

         value  = r;
         value |= (uint32_t)g0 <<  8;
         value |= (uint32_t)b  << 16;
         value |= (uint32_t)g1 << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         r  = src[0];
         g0 = src[1];
         g1 = 0;
         b  = src[2];

         value  = r;
         value |= (uint32_t)g0 <<  8;
         value |= (uint32_t)b  << 16;
         value |= (uint32_t)g1 << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_r8g8_b8g8_unorm_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src,
                                        unsigned i, ASSERTED unsigned j)
{
   float *dst = in_dst;

   assert(i < 2);
   assert(j < 1);

   dst[0] = ubyte_to_float(src[0]);             /* r */
   dst[1] = ubyte_to_float(src[1 + 2*i]);       /* g */
   dst[2] = ubyte_to_float(src[2]);             /* b */
   dst[3] = 1.0f;                               /* a */
}


void
util_format_g8r8_g8b8_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      float *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      float r, g0, g1, b;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         g0 = ubyte_to_float((value >>  0) & 0xff);
         r  = ubyte_to_float((value >>  8) & 0xff);
         g1 = ubyte_to_float((value >> 16) & 0xff);
         b  = ubyte_to_float((value >> 24) & 0xff);

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 1.0f; /* a */
         dst += 4;

         dst[0] = r;    /* r */
         dst[1] = g1;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 1.0f; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         g0 = ubyte_to_float((value >>  0) & 0xff);
         r  = ubyte_to_float((value >>  8) & 0xff);
         g1 = ubyte_to_float((value >> 16) & 0xff);
         b  = ubyte_to_float((value >> 24) & 0xff);

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 1.0f; /* a */
      }

      src_row = (uint8_t *)src_row + src_stride;
      dst_row = (uint8_t *)dst_row + dst_stride;
   }
}


void
util_format_g8r8_g8b8_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                          const uint8_t *restrict src_row, unsigned src_stride,
                                          unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      uint8_t *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      uint8_t r, g0, g1, b;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         g0 = (value >>  0) & 0xff;
         r  = (value >>  8) & 0xff;
         g1 = (value >> 16) & 0xff;
         b  = (value >> 24) & 0xff;

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 0xff; /* a */
         dst += 4;

         dst[0] = r;    /* r */
         dst[1] = g1;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 0xff; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         g0 = (value >>  0) & 0xff;
         r  = (value >>  8) & 0xff;
         g1 = (value >> 16) & 0xff;
         b  = (value >> 24) & 0xff;

         dst[0] = r;    /* r */
         dst[1] = g0;   /* g */
         dst[2] = b;    /* b */
         dst[3] = 0xff; /* a */
      }

      src_row += src_stride/sizeof(*src_row);
      dst_row += dst_stride/sizeof(*dst_row);
   }
}


void
util_format_g8r8_g8b8_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const float *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const float *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      float r, g0, g1, b;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         r  = 0.5f*(src[0] + src[4]);
         g0 = src[1];
         g1 = src[5];
         b  = 0.5f*(src[2] + src[6]);

         value  = (uint32_t)float_to_ubyte(g0);
         value |= (uint32_t)float_to_ubyte(r)  <<  8;
         value |= (uint32_t)float_to_ubyte(g1) << 16;
         value |= (uint32_t)float_to_ubyte(b)  << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         r  = src[0];
         g0 = src[1];
         g1 = 0;
         b  = src[2];

         value  = (uint32_t)float_to_ubyte(g0);
         value |= (uint32_t)float_to_ubyte(r)  <<  8;
         value |= (uint32_t)float_to_ubyte(g1) << 16;
         value |= (uint32_t)float_to_ubyte(b)  << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_g8r8_g8b8_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const uint8_t *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      uint32_t r, g0, g1, b;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         r  = (src[0] + src[4] + 1) >> 1;
         g0 = src[1];
         g1 = src[5];
         b  = (src[2] + src[6] + 1) >> 1;

         value  = g0;
         value |= (uint32_t)r  <<  8;
         value |= (uint32_t)g1 << 16;
         value |= (uint32_t)b  << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         r  = src[0];
         g0 = src[1];
         g1 = 0;
         b  = src[2];

         value  = g0;
         value |= (uint32_t)r  <<  8;
         value |= (uint32_t)g1 << 16;
         value |= (uint32_t)b  << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_g8r8_g8b8_unorm_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src,
                                        unsigned i, ASSERTED unsigned j)
{
   float *dst = in_dst;

   assert(i < 2);
   assert(j < 1);

   dst[0] = ubyte_to_float(src[1]);             /* r */
   dst[1] = ubyte_to_float(src[0 + 2*i]);       /* g */
   dst[2] = ubyte_to_float(src[3]);             /* b */
   dst[3] = 1.0f;                               /* a */
}


void
util_format_uyvy_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                              const uint8_t *restrict src_row, unsigned src_stride,
                              unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      float *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      uint8_t y0, y1, u, v;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         u  = (value >>  0) & 0xff;
         y0 = (value >>  8) & 0xff;
         v  = (value >> 16) & 0xff;
         y1 = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_float(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 1.0f; /* a */
         dst += 4;

         util_format_yuv_to_rgb_float(y1, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 1.0f; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         u  = (value >>  0) & 0xff;
         y0 = (value >>  8) & 0xff;
         v  = (value >> 16) & 0xff;
         y1 = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_float(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 1.0f; /* a */
      }

      src_row = (uint8_t *)src_row + src_stride;
      dst_row = (uint8_t *)dst_row + dst_stride;
   }
}


void
util_format_uyvy_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                               const uint8_t *restrict src_row, unsigned src_stride,
                               unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      uint8_t *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      uint8_t y0, y1, u, v;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         u  = (value >>  0) & 0xff;
         y0 = (value >>  8) & 0xff;
         v  = (value >> 16) & 0xff;
         y1 = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_8unorm(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 0xff; /* a */
         dst += 4;

         util_format_yuv_to_rgb_8unorm(y1, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 0xff; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         u  = (value >>  0) & 0xff;
         y0 = (value >>  8) & 0xff;
         v  = (value >> 16) & 0xff;
         y1 = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_8unorm(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 0xff; /* a */
      }

      src_row += src_stride/sizeof(*src_row);
      dst_row += dst_stride/sizeof(*dst_row);
   }
}


void
util_format_uyvy_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                            const float *restrict src_row, unsigned src_stride,
                            unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const float *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      uint8_t y0, y1, u, v;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         uint8_t y0, y1, u0, u1, v0, v1, u, v;

         util_format_rgb_float_to_yuv(src[0], src[1], src[2],
                                      &y0, &u0, &v0);
         util_format_rgb_float_to_yuv(src[4], src[5], src[6],
                                      &y1, &u1, &v1);

         u = (u0 + u1 + 1) >> 1;
         v = (v0 + v1 + 1) >> 1;

         value  = u;
         value |= (uint32_t)y0 <<  8;
         value |= (uint32_t)v  << 16;
         value |= (uint32_t)y1 << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         util_format_rgb_float_to_yuv(src[0], src[1], src[2],
                                      &y0, &u, &v);
         y1 = 0;

         value  = u;
         value |= (uint32_t)y0 <<  8;
         value |= (uint32_t)v  << 16;
         value |= (uint32_t)y1 << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_uyvy_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                             const uint8_t *restrict src_row, unsigned src_stride,
                             unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const uint8_t *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      uint8_t y0, y1, u, v;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         uint8_t y0, y1, u0, u1, v0, v1, u, v;

         util_format_rgb_8unorm_to_yuv(src[0], src[1], src[2],
                                       &y0, &u0, &v0);
         util_format_rgb_8unorm_to_yuv(src[4], src[5], src[6],
                                       &y1, &u1, &v1);

         u = (u0 + u1 + 1) >> 1;
         v = (v0 + v1 + 1) >> 1;

         value  = u;
         value |= (uint32_t)y0 <<  8;
         value |= (uint32_t)v  << 16;
         value |= (uint32_t)y1 << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         util_format_rgb_8unorm_to_yuv(src[0], src[1], src[2],
                                       &y0, &u, &v);
         y1 = 0;

         value  = u;
         value |= (uint32_t)y0 <<  8;
         value |= (uint32_t)v  << 16;
         value |= (uint32_t)y1 << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_uyvy_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src,
                             unsigned i, ASSERTED unsigned j)
{
   float *dst = in_dst;
   uint8_t y, u, v;

   assert(i < 2);
   assert(j < 1);

   y = src[1 + i*2];
   u = src[0];
   v = src[2];

   util_format_yuv_to_rgb_float(y, u, v, &dst[0], &dst[1], &dst[2]);

   dst[3] = 1.0f;
}


void
util_format_yuyv_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                              const uint8_t *restrict src_row, unsigned src_stride,
                              unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      float *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      uint8_t y0, y1, u, v;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         y0 = (value >>  0) & 0xff;
         u  = (value >>  8) & 0xff;
         y1 = (value >> 16) & 0xff;
         v  = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_float(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 1.0f; /* a */
         dst += 4;

         util_format_yuv_to_rgb_float(y1, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 1.0f; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         y0 = (value >>  0) & 0xff;
         u  = (value >>  8) & 0xff;
         y1 = (value >> 16) & 0xff;
         v  = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_float(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 1.0f; /* a */
      }

      src_row = (uint8_t *)src_row + src_stride;
      dst_row = (uint8_t *)dst_row + dst_stride;
   }
}


void
util_format_yuyv_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                               const uint8_t *restrict src_row, unsigned src_stride,
                               unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      uint8_t *dst = dst_row;
      const uint32_t *src = (const uint32_t *)src_row;
      uint32_t value;
      uint8_t y0, y1, u, v;

      for (x = 0; x + 1 < width; x += 2) {
         value = util_cpu_to_le32(*src++);

         y0 = (value >>  0) & 0xff;
         u  = (value >>  8) & 0xff;
         y1 = (value >> 16) & 0xff;
         v  = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_8unorm(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 0xff; /* a */
         dst += 4;

         util_format_yuv_to_rgb_8unorm(y1, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 0xff; /* a */
         dst += 4;
      }

      if (x < width) {
         value = util_cpu_to_le32(*src);

         y0 = (value >>  0) & 0xff;
         u  = (value >>  8) & 0xff;
         y1 = (value >> 16) & 0xff;
         v  = (value >> 24) & 0xff;

         util_format_yuv_to_rgb_8unorm(y0, u, v, &dst[0], &dst[1], &dst[2]);
         dst[3] = 0xff; /* a */
      }

      src_row += src_stride/sizeof(*src_row);
      dst_row += dst_stride/sizeof(*dst_row);
   }
}


void
util_format_yuyv_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                            const float *restrict src_row, unsigned src_stride,
                            unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const float *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      uint8_t y0, y1, u, v;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         uint8_t y0, y1, u0, u1, v0, v1, u, v;

         util_format_rgb_float_to_yuv(src[0], src[1], src[2],
                                      &y0, &u0, &v0);
         util_format_rgb_float_to_yuv(src[4], src[5], src[6],
                                      &y1, &u1, &v1);

         u = (u0 + u1 + 1) >> 1;
         v = (v0 + v1 + 1) >> 1;

         value  = y0;
         value |= (uint32_t)u  <<  8;
         value |= (uint32_t)y1 << 16;
         value |= (uint32_t)v  << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         util_format_rgb_float_to_yuv(src[0], src[1], src[2],
                                      &y0, &u, &v);
         y1 = 0;

         value  = y0;
         value |= (uint32_t)u  <<  8;
         value |= (uint32_t)y1 << 16;
         value |= (uint32_t)v  << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_yuyv_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                             const uint8_t *restrict src_row, unsigned src_stride,
                             unsigned width, unsigned height)
{
   unsigned x, y;

   for (y = 0; y < height; y += 1) {
      const uint8_t *src = src_row;
      uint32_t *dst = (uint32_t *)dst_row;
      uint8_t y0, y1, u, v;
      uint32_t value;

      for (x = 0; x + 1 < width; x += 2) {
         uint8_t y0, y1, u0, u1, v0, v1, u, v;

         util_format_rgb_8unorm_to_yuv(src[0], src[1], src[2],
                                       &y0, &u0, &v0);
         util_format_rgb_8unorm_to_yuv(src[4], src[5], src[6],
                                       &y1, &u1, &v1);

         u = (u0 + u1 + 1) >> 1;
         v = (v0 + v1 + 1) >> 1;

         value  = y0;
         value |= (uint32_t)u  <<  8;
         value |= (uint32_t)y1 << 16;
         value |= (uint32_t)v  << 24;

         *dst++ = util_le32_to_cpu(value);

         src += 8;
      }

      if (x < width) {
         util_format_rgb_8unorm_to_yuv(src[0], src[1], src[2],
                                       &y0, &u, &v);
         y1 = 0;

         value  = y0;
         value |= (uint32_t)u  <<  8;
         value |= (uint32_t)y1 << 16;
         value |= (uint32_t)v  << 24;

         *dst = util_le32_to_cpu(value);
      }

      dst_row += dst_stride/sizeof(*dst_row);
      src_row += src_stride/sizeof(*src_row);
   }
}


void
util_format_yuyv_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src,
                             unsigned i, ASSERTED unsigned j)
{
   float *dst = in_dst;
   uint8_t y, u, v;

   assert(i < 2);
   assert(j < 1);

   y = src[0 + i*2];
   u = src[1];
   v = src[3];

   util_format_yuv_to_rgb_float(y, u, v, &dst[0], &dst[1], &dst[2]);

   dst[3] = 1.0f;
}
