/**************************************************************************
 *
 * Copyright 2013 Advanced Micro Devices, Inc.
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
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/*
 * Authors:
 *      Christian König <christian.koenig@amd.com>
 *
 */

/*
 * Functions for reading the raw byte sequence payload of H.264
 */

#ifndef vl_rbsp_h
#define vl_rbsp_h

#include "util/vl_vlc.h"

struct vl_rbsp {
   struct vl_vlc nal;
   unsigned escaped;
};

/**
 * Initialize the RBSP object
 */
static inline void vl_rbsp_init(struct vl_rbsp *rbsp, struct vl_vlc *nal, unsigned num_bits)
{
   unsigned valid, bits_left = vl_vlc_bits_left(nal);
   int i;

   /* copy the position */
   rbsp->nal = *nal;

   /* search for the end of the NAL unit */
   while (vl_vlc_search_byte(nal, num_bits, 0x00)) {
      if (vl_vlc_peekbits(nal, 24) == 0x000001 ||
          vl_vlc_peekbits(nal, 32) == 0x00000001) {
         vl_vlc_limit(&rbsp->nal, bits_left - vl_vlc_bits_left(nal));
         break;
      }
      vl_vlc_eatbits(nal, 8);
   }

   valid = vl_vlc_valid_bits(&rbsp->nal);
   /* search for the emulation prevention three byte */
   for (i = 24; i <= valid; i += 8) {
      if ((vl_vlc_peekbits(&rbsp->nal, i) & 0xffffff) == 0x3) {
         vl_vlc_removebits(&rbsp->nal, i - 8, 8);
         i += 8;
      }
   }

   valid = vl_vlc_valid_bits(&rbsp->nal);

   rbsp->escaped = (valid >= 16) ? 16 : ((valid >= 8) ? 8 : 0);
}

/**
 * Make at least 16 more bits available
 */
static inline void vl_rbsp_fillbits(struct vl_rbsp *rbsp)
{
   unsigned valid = vl_vlc_valid_bits(&rbsp->nal);
   unsigned i, bits;

   /* abort if we still have enough bits */
   if (valid >= 32)
      return;

   vl_vlc_fillbits(&rbsp->nal);

   /* abort if we have less than 24 bits left in this nal */
   if (vl_vlc_bits_left(&rbsp->nal) < 24)
      return;

   /* check that we have enough bits left from the last fillbits */
   assert(valid >= rbsp->escaped);

   /* handle the already escaped bits */
   valid -= rbsp->escaped;

   /* search for the emulation prevention three byte */
   rbsp->escaped = 16;
   bits = vl_vlc_valid_bits(&rbsp->nal);
   for (i = valid + 24; i <= bits; i += 8) {
      if ((vl_vlc_peekbits(&rbsp->nal, i) & 0xffffff) == 0x3) {
         vl_vlc_removebits(&rbsp->nal, i - 8, 8);
         rbsp->escaped = bits - i;
         bits -= 8;
         i += 8;
      }
   }
}

/**
 * Return an unsigned integer from the first n bits
 */
static inline unsigned vl_rbsp_u(struct vl_rbsp *rbsp, unsigned n)
{
   if (n == 0)
      return 0;

   vl_rbsp_fillbits(rbsp);
   return vl_vlc_get_uimsbf(&rbsp->nal, n);
}

/**
 * Return an unsigned exponential Golomb encoded integer
 */
static inline unsigned vl_rbsp_ue(struct vl_rbsp *rbsp)
{
   unsigned bits = 0;

   vl_rbsp_fillbits(rbsp);
   while (!vl_vlc_get_uimsbf(&rbsp->nal, 1))
      ++bits;

   return (1 << bits) - 1 + vl_rbsp_u(rbsp, bits);
}

/**
 * Return an signed exponential Golomb encoded integer
 */
static inline signed vl_rbsp_se(struct vl_rbsp *rbsp)
{
   signed codeNum = vl_rbsp_ue(rbsp);
   if (codeNum & 1)
      return (codeNum + 1) >> 1;
   else
      return -(codeNum >> 1);
}

/**
 * Are more data available in the RBSP ?
 */
static inline bool vl_rbsp_more_data(struct vl_rbsp *rbsp)
{
   unsigned bits, value;

   if (vl_vlc_bits_left(&rbsp->nal) > 8)
      return true;

   bits = vl_vlc_valid_bits(&rbsp->nal);
   value = vl_vlc_peekbits(&rbsp->nal, bits);
   if (value == 0 || value == (1 << (bits - 1)))
      return false;

   return true;
}

#endif /* vl_rbsp_h */
