#include "pipe/p_compiler.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "util/format/u_format_etc.h"

/* define etc1_parse_block and etc. */
#define UINT8_TYPE uint8_t
#define TAG(x) x
#include "util/format/texcompress_etc_tmp.h"
#undef TAG
#undef UINT8_TYPE

void
util_format_etc1_rgb8_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   etc1_unpack_rgba8888(dst_row, dst_stride, src_row, src_stride, width, height);
}

void
util_format_etc1_rgb8_pack_rgba_8unorm(UNUSED uint8_t *restrict dst_row, UNUSED unsigned dst_stride,
                                       UNUSED const uint8_t *restrict src_row, UNUSED unsigned src_stride,
                                       UNUSED unsigned width, UNUSED unsigned height)
{
   assert(0);
}

void
util_format_etc1_rgb8_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride, const uint8_t *restrict src_row, unsigned src_stride, unsigned width, unsigned height)
{
   const unsigned bw = 4, bh = 4, bs = 8, comps = 4;
   struct etc1_block block;
   unsigned x, y, i, j;

   for (y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;

      for (x = 0; x < width; x+= bw) {
         etc1_parse_block(&block, src);

         for (j = 0; j < bh; j++) {
            float *dst = (float *)((uint8_t *)dst_row + (y + j) * dst_stride + x * comps * 4);
            uint8_t tmp[3];

            for (i = 0; i < bw; i++) {
               etc1_fetch_texel(&block, i, j, tmp);
               dst[0] = ubyte_to_float(tmp[0]);
               dst[1] = ubyte_to_float(tmp[1]);
               dst[2] = ubyte_to_float(tmp[2]);
               dst[3] = 1.0f;
               dst += comps;
            }
         }

         src += bs;
      }

      src_row += src_stride;
   }
}

void
util_format_etc1_rgb8_pack_rgba_float(UNUSED uint8_t *restrict dst_row, UNUSED unsigned dst_stride,
                                      UNUSED const float *restrict src_row, UNUSED unsigned src_stride,
                                      UNUSED unsigned width, UNUSED unsigned height)
{
   assert(0);
}

void
util_format_etc1_rgb8_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   struct etc1_block block;
   uint8_t tmp[3];

   assert(i < 4 && j < 4); /* check i, j against 4x4 block size */

   etc1_parse_block(&block, src);
   etc1_fetch_texel(&block, i, j, tmp);

   dst[0] = ubyte_to_float(tmp[0]);
   dst[1] = ubyte_to_float(tmp[1]);
   dst[2] = ubyte_to_float(tmp[2]);
   dst[3] = 1.0f;
}
