/*
 * Copyright (C) 2011 LunarG, Inc.
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

/*
 * Included by texcompress_etc1 and gallium to define ETC1 decoding routines.
 */

struct TAG(etc1_block) {
   uint32_t pixel_indices;
   int flipped;
   const int *modifier_tables[2];
   UINT8_TYPE base_colors[2][3];
};

static UINT8_TYPE
TAG(etc1_base_color_diff_hi)(UINT8_TYPE in)
{
   return (in & 0xf8) | (in >> 5);
}

static UINT8_TYPE
TAG(etc1_base_color_diff_lo)(UINT8_TYPE in)
{
   static const int lookup[8] = { 0, 1, 2, 3, -4, -3, -2, -1 };

   in = (in >> 3) + lookup[in & 0x7];

   return (in << 3) | (in >> 2);
}

static UINT8_TYPE
TAG(etc1_base_color_ind_hi)(UINT8_TYPE in)
{
   return (in & 0xf0) | ((in & 0xf0) >> 4);
}

static UINT8_TYPE
TAG(etc1_base_color_ind_lo)(UINT8_TYPE in)
{
   return ((in & 0xf) << 4) | (in & 0xf);
}

static UINT8_TYPE
TAG(etc1_clamp)(UINT8_TYPE base, int modifier)
{
   int tmp = (int) base + modifier;

   /* CLAMP(tmp, 0, 255) */
   return (UINT8_TYPE) ((tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp));
}

static const int TAG(etc1_modifier_tables)[8][4] = {
   {  2,   8,  -2,   -8},
   {  5,  17,  -5,  -17},
   {  9,  29,  -9,  -29},
   { 13,  42, -13,  -42},
   { 18,  60, -18,  -60},
   { 24,  80, -24,  -80},
   { 33, 106, -33, -106},
   { 47, 183, -47, -183}
};

static void
TAG(etc1_parse_block)(struct TAG(etc1_block) *block, const UINT8_TYPE *src)
{
   if (src[3] & 0x2) {
      /* differential mode */
      block->base_colors[0][0] = (int) TAG(etc1_base_color_diff_hi)(src[0]);
      block->base_colors[1][0] = (int) TAG(etc1_base_color_diff_lo)(src[0]);
      block->base_colors[0][1] = (int) TAG(etc1_base_color_diff_hi)(src[1]);
      block->base_colors[1][1] = (int) TAG(etc1_base_color_diff_lo)(src[1]);
      block->base_colors[0][2] = (int) TAG(etc1_base_color_diff_hi)(src[2]);
      block->base_colors[1][2] = (int) TAG(etc1_base_color_diff_lo)(src[2]);
   }
   else {
      /* individual mode */
      block->base_colors[0][0] = (int) TAG(etc1_base_color_ind_hi)(src[0]);
      block->base_colors[1][0] = (int) TAG(etc1_base_color_ind_lo)(src[0]);
      block->base_colors[0][1] = (int) TAG(etc1_base_color_ind_hi)(src[1]);
      block->base_colors[1][1] = (int) TAG(etc1_base_color_ind_lo)(src[1]);
      block->base_colors[0][2] = (int) TAG(etc1_base_color_ind_hi)(src[2]);
      block->base_colors[1][2] = (int) TAG(etc1_base_color_ind_lo)(src[2]);
   }

   /* pick modifier tables */
   block->modifier_tables[0] = TAG(etc1_modifier_tables)[(src[3] >> 5) & 0x7];
   block->modifier_tables[1] = TAG(etc1_modifier_tables)[(src[3] >> 2) & 0x7];

   block->flipped = (src[3] & 0x1);

   block->pixel_indices =
      (src[4] << 24) | (src[5] << 16) | (src[6] << 8) | src[7];
}

static void
TAG(etc1_fetch_texel)(const struct TAG(etc1_block) *block,
      int x, int y, UINT8_TYPE *dst)
{
   const UINT8_TYPE *base_color;
   int modifier, bit, idx, blk;

   /* get pixel index */
   bit = y + x * 4;
   idx = ((block->pixel_indices >> (15 + bit)) & 0x2) |
         ((block->pixel_indices >>      (bit)) & 0x1);

   /* get subblock */
   blk = (block->flipped) ? (y >= 2) : (x >= 2);

   base_color = block->base_colors[blk];
   modifier = block->modifier_tables[blk][idx];

   dst[0] = TAG(etc1_clamp)(base_color[0], modifier);
   dst[1] = TAG(etc1_clamp)(base_color[1], modifier);
   dst[2] = TAG(etc1_clamp)(base_color[2], modifier);
}

static void
etc1_unpack_rgba8888(uint8_t *dst_row,
                     unsigned dst_stride,
                     const uint8_t *src_row,
                     unsigned src_stride,
                     unsigned width,
                     unsigned height)
{
   const unsigned bw = 4, bh = 4, bs = 8, comps = 4;
   struct etc1_block block;
   unsigned x, y, i, j;

   for (y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;

      for (x = 0; x < width; x+= bw) {
         etc1_parse_block(&block, src);

         for (j = 0; j < MIN2(bh, height - y); j++) {
            uint8_t *dst = dst_row + (y + j) * dst_stride + x * comps;
            for (i = 0; i < MIN2(bw, width - x); i++) {
               etc1_fetch_texel(&block, i, j, dst);
               dst[3] = 255;
               dst += comps;
            }
         }

         src += bs;
      }

      src_row += src_stride;
   }
}
