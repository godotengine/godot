/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2015, 2024, D. R. Commander.  All Rights Reserved.
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

/* This file is included by jdmerge-altivec.c */


void jsimd_h2v1_merged_upsample_altivec(JDIMENSION output_width,
                                        JSAMPIMAGE input_buf,
                                        JDIMENSION in_row_group_ctr,
                                        JSAMPARRAY output_buf)
{
  JSAMPROW outptr, inptr0, inptr1, inptr2;
  int pitch = output_width * RGB_PIXELSIZE, num_cols, yloop;
#ifdef __BIG_ENDIAN__
  int offset;
#endif
  unsigned char __attribute__((aligned(16))) tmpbuf[RGB_PIXELSIZE * 16];

  __vector unsigned char rgb0, rgb1, rgb2, rgbx0, rgbx1, rgbx2, rgbx3,
    y, cb, cr;
#ifdef __BIG_ENDIAN__
  __vector unsigned char edgel, edgeh, edges, out0, out1, out2, out3;
#if RGB_PIXELSIZE == 4
  __vector unsigned char out4;
#endif
#endif
#if RGB_PIXELSIZE == 4
  __vector unsigned char rgb3;
#endif
  __vector short rg0, rg1, rg2, rg3, bx0, bx1, bx2, bx3, ye, yo, cbl, cbh,
    crl, crh, r_yl, r_yh, g_yl, g_yh, b_yl, b_yh, g_y0w, g_y1w, g_y2w, g_y3w,
    rl, rh, gl, gh, bl, bh, re, ro, ge, go, be, bo;
  __vector int g_y0, g_y1, g_y2, g_y3;

  /* Constants
   * NOTE: The >> 1 is to compensate for the fact that vec_madds() returns 17
   * high-order bits, not 16.
   */
  __vector short pw_f0402 = { __8X(F_0_402 >> 1) },
    pw_mf0228 = { __8X(-F_0_228 >> 1) },
    pw_mf0344_f0285 = { __4X2(-F_0_344, F_0_285) },
    pw_one = { __8X(1) }, pw_255 = { __8X(255) },
    pw_cj = { __8X(CENTERJSAMPLE) };
  __vector int pd_onehalf = { __4X(ONE_HALF) };
  __vector unsigned char pb_zero = { __16X(0) },
#ifdef __BIG_ENDIAN__
    shift_pack_index =
      {  0,  1,  4,  5,  8,  9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29 },
    even_index =
      {  0, 16,  0, 18,  0, 20,  0, 22,  0, 24,  0, 26,  0, 28,  0, 30 },
    odd_index =
      {  0, 17,  0, 19,  0, 21,  0, 23,  0, 25,  0, 27,  0, 29,  0, 31 };
#else
    shift_pack_index =
      {  2,  3,  6,  7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31 },
    even_index =
      { 16,  0, 18,  0, 20,  0, 22,  0, 24,  0, 26,  0, 28,  0, 30,  0 },
    odd_index =
      { 17,  0, 19,  0, 21,  0, 23,  0, 25,  0, 27,  0, 29,  0, 31,  0 };
#endif

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  for (num_cols = pitch; num_cols > 0; inptr1 += 16, inptr2 += 16) {

    cb = vec_ld(0, inptr1);
    /* NOTE: We have to use vec_merge*() here because vec_unpack*() doesn't
     * support unsigned vectors.
     */
    cbl = (__vector signed short)VEC_UNPACKHU(cb);
    cbh = (__vector signed short)VEC_UNPACKLU(cb);
    cbl = vec_sub(cbl, pw_cj);
    cbh = vec_sub(cbh, pw_cj);

    cr = vec_ld(0, inptr2);
    crl = (__vector signed short)VEC_UNPACKHU(cr);
    crh = (__vector signed short)VEC_UNPACKLU(cr);
    crl = vec_sub(crl, pw_cj);
    crh = vec_sub(crh, pw_cj);

    /* (Original)
     * R = Y                + 1.40200 * Cr
     * G = Y - 0.34414 * Cb - 0.71414 * Cr
     * B = Y + 1.77200 * Cb
     *
     * (This implementation)
     * R = Y                + 0.40200 * Cr + Cr
     * G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
     * B = Y - 0.22800 * Cb + Cb + Cb
     */
    b_yl = vec_add(cbl, cbl);
    b_yh = vec_add(cbh, cbh);
    b_yl = vec_madds(b_yl, pw_mf0228, pw_one);
    b_yh = vec_madds(b_yh, pw_mf0228, pw_one);
    b_yl = vec_sra(b_yl, (__vector unsigned short)pw_one);
    b_yh = vec_sra(b_yh, (__vector unsigned short)pw_one);
    b_yl = vec_add(b_yl, cbl);
    b_yh = vec_add(b_yh, cbh);
    b_yl = vec_add(b_yl, cbl);
    b_yh = vec_add(b_yh, cbh);

    r_yl = vec_add(crl, crl);
    r_yh = vec_add(crh, crh);
    r_yl = vec_madds(r_yl, pw_f0402, pw_one);
    r_yh = vec_madds(r_yh, pw_f0402, pw_one);
    r_yl = vec_sra(r_yl, (__vector unsigned short)pw_one);
    r_yh = vec_sra(r_yh, (__vector unsigned short)pw_one);
    r_yl = vec_add(r_yl, crl);
    r_yh = vec_add(r_yh, crh);

    g_y0w = vec_mergeh(cbl, crl);
    g_y1w = vec_mergel(cbl, crl);
    g_y0 = vec_msums(g_y0w, pw_mf0344_f0285, pd_onehalf);
    g_y1 = vec_msums(g_y1w, pw_mf0344_f0285, pd_onehalf);
    g_y2w = vec_mergeh(cbh, crh);
    g_y3w = vec_mergel(cbh, crh);
    g_y2 = vec_msums(g_y2w, pw_mf0344_f0285, pd_onehalf);
    g_y3 = vec_msums(g_y3w, pw_mf0344_f0285, pd_onehalf);
    /* Clever way to avoid 4 shifts + 2 packs.  This packs the high word from
     * each dword into a new 16-bit vector, which is the equivalent of
     * descaling the 32-bit results (right-shifting by 16 bits) and then
     * packing them.
     */
    g_yl = vec_perm((__vector short)g_y0, (__vector short)g_y1,
                    shift_pack_index);
    g_yh = vec_perm((__vector short)g_y2, (__vector short)g_y3,
                    shift_pack_index);
    g_yl = vec_sub(g_yl, crl);
    g_yh = vec_sub(g_yh, crh);

    for (yloop = 0; yloop < 2 && num_cols > 0; yloop++,
         num_cols -= RGB_PIXELSIZE * 16,
         outptr += RGB_PIXELSIZE * 16, inptr0 += 16) {

      y = vec_ld(0, inptr0);
      ye = (__vector signed short)vec_perm(pb_zero, y, even_index);
      yo = (__vector signed short)vec_perm(pb_zero, y, odd_index);

      if (yloop == 0) {
        be = vec_add(b_yl, ye);
        bo = vec_add(b_yl, yo);
        re = vec_add(r_yl, ye);
        ro = vec_add(r_yl, yo);
        ge = vec_add(g_yl, ye);
        go = vec_add(g_yl, yo);
      } else {
        be = vec_add(b_yh, ye);
        bo = vec_add(b_yh, yo);
        re = vec_add(r_yh, ye);
        ro = vec_add(r_yh, yo);
        ge = vec_add(g_yh, ye);
        go = vec_add(g_yh, yo);
      }

      rl = vec_mergeh(re, ro);
      rh = vec_mergel(re, ro);
      gl = vec_mergeh(ge, go);
      gh = vec_mergel(ge, go);
      bl = vec_mergeh(be, bo);
      bh = vec_mergel(be, bo);

      rg0 = vec_mergeh(rl, gl);
      bx0 = vec_mergeh(bl, pw_255);
      rg1 = vec_mergel(rl, gl);
      bx1 = vec_mergel(bl, pw_255);
      rg2 = vec_mergeh(rh, gh);
      bx2 = vec_mergeh(bh, pw_255);
      rg3 = vec_mergel(rh, gh);
      bx3 = vec_mergel(bh, pw_255);

      rgbx0 = vec_packsu(rg0, bx0);
      rgbx1 = vec_packsu(rg1, bx1);
      rgbx2 = vec_packsu(rg2, bx2);
      rgbx3 = vec_packsu(rg3, bx3);

#if RGB_PIXELSIZE == 3
      /* rgbx0 = R0 G0 R1 G1 R2 G2 R3 G3 B0 X0 B1 X1 B2 X2 B3 X3
       * rgbx1 = R4 G4 R5 G5 R6 G6 R7 G7 B4 X4 B5 X5 B6 X6 B7 X7
       * rgbx2 = R8 G8 R9 G9 Ra Ga Rb Gb B8 X8 B9 X9 Ba Xa Bb Xb
       * rgbx3 = Rc Gc Rd Gd Re Ge Rf Gf Bc Xc Bd Xd Be Xe Bf Xf
       *
       * rgb0 = R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
       * rgb1 = G5 B5 R6 G6 B6 R7 G7 B7 R8 G8 B8 R9 G9 B9 Ra Ga
       * rgb2 = Ba Rb Gb Bb Rc Gc Bc Rd Gd Bd Re Ge Be Rf Gf Bf
       */
      rgb0 = vec_perm(rgbx0, rgbx1, (__vector unsigned char)RGB_INDEX0);
      rgb1 = vec_perm(rgbx1, rgbx2, (__vector unsigned char)RGB_INDEX1);
      rgb2 = vec_perm(rgbx2, rgbx3, (__vector unsigned char)RGB_INDEX2);
#else
      /* rgbx0 = R0 G0 R1 G1 R2 G2 R3 G3 B0 X0 B1 X1 B2 X2 B3 X3
       * rgbx1 = R4 G4 R5 G5 R6 G6 R7 G7 B4 X4 B5 X5 B6 X6 B7 X7
       * rgbx2 = R8 G8 R9 G9 Ra Ga Rb Gb B8 X8 B9 X9 Ba Xa Bb Xb
       * rgbx3 = Rc Gc Rd Gd Re Ge Rf Gf Bc Xc Bd Xd Be Xe Bf Xf
       *
       * rgb0 = R0 G0 B0 X0 R1 G1 B1 X1 R2 G2 B2 X2 R3 G3 B3 X3
       * rgb1 = R4 G4 B4 X4 R5 G5 B5 X5 R6 G6 B6 X6 R7 G7 B7 X7
       * rgb2 = R8 G8 B8 X8 R9 G9 B9 X9 Ra Ga Ba Xa Rb Gb Bb Xb
       * rgb3 = Rc Gc Bc Xc Rd Gd Bd Xd Re Ge Be Xe Rf Gf Bf Xf
       */
      rgb0 = vec_perm(rgbx0, rgbx0, (__vector unsigned char)RGB_INDEX);
      rgb1 = vec_perm(rgbx1, rgbx1, (__vector unsigned char)RGB_INDEX);
      rgb2 = vec_perm(rgbx2, rgbx2, (__vector unsigned char)RGB_INDEX);
      rgb3 = vec_perm(rgbx3, rgbx3, (__vector unsigned char)RGB_INDEX);
#endif

#ifdef __BIG_ENDIAN__
      offset = (size_t)outptr & 15;
      if (offset) {
        __vector unsigned char unaligned_shift_index;
        int bytes = num_cols + offset;

        if (bytes < (RGB_PIXELSIZE + 1) * 16 && (bytes & 15)) {
          /* Slow path to prevent buffer overwrite.  Since there is no way to
           * write a partial AltiVec register, overwrite would occur on the
           * last chunk of the last image row if the right edge is not on a
           * 16-byte boundary.  It could also occur on other rows if the bytes
           * per row is low enough.  Since we can't determine whether we're on
           * the last image row, we have to assume every row is the last.
           */
          vec_st(rgb0, 0, tmpbuf);
          vec_st(rgb1, 16, tmpbuf);
          vec_st(rgb2, 32, tmpbuf);
#if RGB_PIXELSIZE == 4
          vec_st(rgb3, 48, tmpbuf);
#endif
          memcpy(outptr, tmpbuf, min(num_cols, RGB_PIXELSIZE * 16));
        } else {
          /* Fast path */
          unaligned_shift_index = vec_lvsl(0, outptr);
          edgel = vec_ld(0, outptr);
          edgeh = vec_ld(min(num_cols - 1, RGB_PIXELSIZE * 16), outptr);
          edges = vec_perm(edgeh, edgel, unaligned_shift_index);
          unaligned_shift_index = vec_lvsr(0, outptr);
          out0 = vec_perm(edges, rgb0, unaligned_shift_index);
          out1 = vec_perm(rgb0, rgb1, unaligned_shift_index);
          out2 = vec_perm(rgb1, rgb2, unaligned_shift_index);
#if RGB_PIXELSIZE == 4
          out3 = vec_perm(rgb2, rgb3, unaligned_shift_index);
          out4 = vec_perm(rgb3, edges, unaligned_shift_index);
#else
          out3 = vec_perm(rgb2, edges, unaligned_shift_index);
#endif
          vec_st(out0, 0, outptr);
          if (bytes > 16)
            vec_st(out1, 16, outptr);
          if (bytes > 32)
            vec_st(out2, 32, outptr);
          if (bytes > 48)
            vec_st(out3, 48, outptr);
#if RGB_PIXELSIZE == 4
          if (bytes > 64)
            vec_st(out4, 64, outptr);
#endif
        }
      } else {
#endif /* __BIG_ENDIAN__ */
        if (num_cols < RGB_PIXELSIZE * 16 && (num_cols & 15)) {
          /* Slow path */
          VEC_ST(rgb0, 0, tmpbuf);
          VEC_ST(rgb1, 16, tmpbuf);
          VEC_ST(rgb2, 32, tmpbuf);
#if RGB_PIXELSIZE == 4
          VEC_ST(rgb3, 48, tmpbuf);
#endif
          memcpy(outptr, tmpbuf, min(num_cols, RGB_PIXELSIZE * 16));
        } else {
          /* Fast path */
          VEC_ST(rgb0, 0, outptr);
          if (num_cols > 16)
            VEC_ST(rgb1, 16, outptr);
          if (num_cols > 32)
            VEC_ST(rgb2, 32, outptr);
#if RGB_PIXELSIZE == 4
          if (num_cols > 48)
            VEC_ST(rgb3, 48, outptr);
#endif
        }
#ifdef __BIG_ENDIAN__
      }
#endif
    }
  }
}


void jsimd_h2v2_merged_upsample_altivec(JDIMENSION output_width,
                                        JSAMPIMAGE input_buf,
                                        JDIMENSION in_row_group_ctr,
                                        JSAMPARRAY output_buf)
{
  JSAMPROW inptr, outptr;

  inptr = input_buf[0][in_row_group_ctr];
  outptr = output_buf[0];

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2];
  jsimd_h2v1_merged_upsample_altivec(output_width, input_buf, in_row_group_ctr,
                                     output_buf);

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2 + 1];
  output_buf[0] = output_buf[1];
  jsimd_h2v1_merged_upsample_altivec(output_width, input_buf, in_row_group_ctr,
                                     output_buf);

  input_buf[0][in_row_group_ctr] = inptr;
  output_buf[0] = outptr;
}
