/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014-2015, 2024, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2014, Jay Foad.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* This file is included by jcgray-altivec.c */


void jsimd_rgb_gray_convert_altivec(JDIMENSION img_width, JSAMPARRAY input_buf,
                                    JSAMPIMAGE output_buf,
                                    JDIMENSION output_row, int num_rows)
{
  JSAMPROW inptr, outptr;
  int pitch = img_width * RGB_PIXELSIZE, num_cols;
#ifdef __BIG_ENDIAN__
  int offset;
  unsigned char __attribute__((aligned(16))) tmpbuf[RGB_PIXELSIZE * 16];
#endif

  __vector unsigned char rgb0, rgb1 = { 0 }, rgb2 = { 0 },
    rgbg0, rgbg1, rgbg2, rgbg3, y;
#if defined(__BIG_ENDIAN__) || RGB_PIXELSIZE == 4
  __vector unsigned char rgb3 = { 0 };
#endif
#if defined(__BIG_ENDIAN__) && RGB_PIXELSIZE == 4
  __vector unsigned char rgb4 = { 0 };
#endif
  __vector short rg0, rg1, rg2, rg3, bg0, bg1, bg2, bg3;
  __vector unsigned short yl, yh;
  __vector int y0, y1, y2, y3;

  /* Constants */
  __vector short pw_f0299_f0337 = { __4X2(F_0_299, F_0_337) },
    pw_f0114_f0250 = { __4X2(F_0_114, F_0_250) };
  __vector int pd_onehalf = { __4X(ONE_HALF) };
  __vector unsigned char pb_zero = { __16X(0) },
#ifdef __BIG_ENDIAN__
    shift_pack_index =
      { 0, 1, 4, 5,  8,  9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29 };
#else
    shift_pack_index =
      { 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31 };
#endif

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr = output_buf[0][output_row];
    output_row++;

    for (num_cols = pitch; num_cols > 0;
         num_cols -= RGB_PIXELSIZE * 16, inptr += RGB_PIXELSIZE * 16,
         outptr += 16) {

#ifdef __BIG_ENDIAN__
      /* Load 16 pixels == 48 or 64 bytes */
      offset = (size_t)inptr & 15;
      if (offset) {
        __vector unsigned char unaligned_shift_index;
        int bytes = num_cols + offset;

        if (bytes < (RGB_PIXELSIZE + 1) * 16 && (bytes & 15)) {
          /* Slow path to prevent buffer overread.  Since there is no way to
           * read a partial AltiVec register, overread would occur on the last
           * chunk of the last image row if the right edge is not on a 16-byte
           * boundary.  It could also occur on other rows if the bytes per row
           * is low enough.  Since we can't determine whether we're on the last
           * image row, we have to assume every row is the last.
           */
          memcpy(tmpbuf, inptr, min(num_cols, RGB_PIXELSIZE * 16));
          rgb0 = vec_ld(0, tmpbuf);
          rgb1 = vec_ld(16, tmpbuf);
          rgb2 = vec_ld(32, tmpbuf);
#if RGB_PIXELSIZE == 4
          rgb3 = vec_ld(48, tmpbuf);
#endif
        } else {
          /* Fast path */
          rgb0 = vec_ld(0, inptr);
          if (bytes > 16)
            rgb1 = vec_ld(16, inptr);
          if (bytes > 32)
            rgb2 = vec_ld(32, inptr);
          if (bytes > 48)
            rgb3 = vec_ld(48, inptr);
#if RGB_PIXELSIZE == 4
          if (bytes > 64)
            rgb4 = vec_ld(64, inptr);
#endif
          unaligned_shift_index = vec_lvsl(0, inptr);
          rgb0 = vec_perm(rgb0, rgb1, unaligned_shift_index);
          rgb1 = vec_perm(rgb1, rgb2, unaligned_shift_index);
          rgb2 = vec_perm(rgb2, rgb3, unaligned_shift_index);
#if RGB_PIXELSIZE == 4
          rgb3 = vec_perm(rgb3, rgb4, unaligned_shift_index);
#endif
        }
      } else {
        if (num_cols < RGB_PIXELSIZE * 16 && (num_cols & 15)) {
          /* Slow path */
          memcpy(tmpbuf, inptr, min(num_cols, RGB_PIXELSIZE * 16));
          rgb0 = vec_ld(0, tmpbuf);
          rgb1 = vec_ld(16, tmpbuf);
          rgb2 = vec_ld(32, tmpbuf);
#if RGB_PIXELSIZE == 4
          rgb3 = vec_ld(48, tmpbuf);
#endif
        } else {
          /* Fast path */
          rgb0 = vec_ld(0, inptr);
          if (num_cols > 16)
            rgb1 = vec_ld(16, inptr);
          if (num_cols > 32)
            rgb2 = vec_ld(32, inptr);
#if RGB_PIXELSIZE == 4
          if (num_cols > 48)
            rgb3 = vec_ld(48, inptr);
#endif
        }
      }
#else
      /* Little endian */
      rgb0 = vec_vsx_ld(0, inptr);
      if (num_cols > 16)
        rgb1 = vec_vsx_ld(16, inptr);
      if (num_cols > 32)
        rgb2 = vec_vsx_ld(32, inptr);
#if RGB_PIXELSIZE == 4
      if (num_cols > 48)
        rgb3 = vec_vsx_ld(48, inptr);
#endif
#endif

#if RGB_PIXELSIZE == 3
      /* rgb0 = R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
       * rgb1 = G5 B5 R6 G6 B6 R7 G7 B7 R8 G8 B8 R9 G9 B9 Ra Ga
       * rgb2 = Ba Rb Gb Bb Rc Gc Bc Rd Gd Bd Re Ge Be Rf Gf Bf
       *
       * rgbg0 = R0 G0 R1 G1 R2 G2 R3 G3 B0 G0 B1 G1 B2 G2 B3 G3
       * rgbg1 = R4 G4 R5 G5 R6 G6 R7 G7 B4 G4 B5 G5 B6 G6 B7 G7
       * rgbg2 = R8 G8 R9 G9 Ra Ga Rb Gb B8 G8 B9 G9 Ba Ga Bb Gb
       * rgbg3 = Rc Gc Rd Gd Re Ge Rf Gf Bc Gc Bd Gd Be Ge Bf Gf
       */
      rgbg0 = vec_perm(rgb0, rgb0, (__vector unsigned char)RGBG_INDEX0);
      rgbg1 = vec_perm(rgb0, rgb1, (__vector unsigned char)RGBG_INDEX1);
      rgbg2 = vec_perm(rgb1, rgb2, (__vector unsigned char)RGBG_INDEX2);
      rgbg3 = vec_perm(rgb2, rgb2, (__vector unsigned char)RGBG_INDEX3);
#else
      /* rgb0 = R0 G0 B0 X0 R1 G1 B1 X1 R2 G2 B2 X2 R3 G3 B3 X3
       * rgb1 = R4 G4 B4 X4 R5 G5 B5 X5 R6 G6 B6 X6 R7 G7 B7 X7
       * rgb2 = R8 G8 B8 X8 R9 G9 B9 X9 Ra Ga Ba Xa Rb Gb Bb Xb
       * rgb3 = Rc Gc Bc Xc Rd Gd Bd Xd Re Ge Be Xe Rf Gf Bf Xf
       *
       * rgbg0 = R0 G0 R1 G1 R2 G2 R3 G3 B0 G0 B1 G1 B2 G2 B3 G3
       * rgbg1 = R4 G4 R5 G5 R6 G6 R7 G7 B4 G4 B5 G5 B6 G6 B7 G7
       * rgbg2 = R8 G8 R9 G9 Ra Ga Rb Gb B8 G8 B9 G9 Ba Ga Bb Gb
       * rgbg3 = Rc Gc Rd Gd Re Ge Rf Gf Bc Gc Bd Gd Be Ge Bf Gf
       */
      rgbg0 = vec_perm(rgb0, rgb0, (__vector unsigned char)RGBG_INDEX);
      rgbg1 = vec_perm(rgb1, rgb1, (__vector unsigned char)RGBG_INDEX);
      rgbg2 = vec_perm(rgb2, rgb2, (__vector unsigned char)RGBG_INDEX);
      rgbg3 = vec_perm(rgb3, rgb3, (__vector unsigned char)RGBG_INDEX);
#endif

      /* rg0 = R0 G0 R1 G1 R2 G2 R3 G3
       * bg0 = B0 G0 B1 G1 B2 G2 B3 G3
       * ...
       *
       * NOTE: We have to use vec_merge*() here because vec_unpack*() doesn't
       * support unsigned vectors.
       */
      rg0 = (__vector signed short)VEC_UNPACKHU(rgbg0);
      bg0 = (__vector signed short)VEC_UNPACKLU(rgbg0);
      rg1 = (__vector signed short)VEC_UNPACKHU(rgbg1);
      bg1 = (__vector signed short)VEC_UNPACKLU(rgbg1);
      rg2 = (__vector signed short)VEC_UNPACKHU(rgbg2);
      bg2 = (__vector signed short)VEC_UNPACKLU(rgbg2);
      rg3 = (__vector signed short)VEC_UNPACKHU(rgbg3);
      bg3 = (__vector signed short)VEC_UNPACKLU(rgbg3);

      /* (Original)
       * Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
       *
       * (This implementation)
       * Y  =  0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G
       */

      /* Calculate Y values */

      y0 = vec_msums(rg0, pw_f0299_f0337, pd_onehalf);
      y1 = vec_msums(rg1, pw_f0299_f0337, pd_onehalf);
      y2 = vec_msums(rg2, pw_f0299_f0337, pd_onehalf);
      y3 = vec_msums(rg3, pw_f0299_f0337, pd_onehalf);
      y0 = vec_msums(bg0, pw_f0114_f0250, y0);
      y1 = vec_msums(bg1, pw_f0114_f0250, y1);
      y2 = vec_msums(bg2, pw_f0114_f0250, y2);
      y3 = vec_msums(bg3, pw_f0114_f0250, y3);
      /* Clever way to avoid 4 shifts + 2 packs.  This packs the high word from
       * each dword into a new 16-bit vector, which is the equivalent of
       * descaling the 32-bit results (right-shifting by 16 bits) and then
       * packing them.
       */
      yl = vec_perm((__vector unsigned short)y0, (__vector unsigned short)y1,
                    shift_pack_index);
      yh = vec_perm((__vector unsigned short)y2, (__vector unsigned short)y3,
                    shift_pack_index);
      y = vec_pack(yl, yh);
      vec_st(y, 0, outptr);
    }
  }
}
