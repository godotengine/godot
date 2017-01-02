/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx_ports/mem.h"
#include "filter_x86.h"

extern const short vp8_six_tap_mmx[8][6*8];

extern void vp8_filter_block1d_h6_mmx
(
    unsigned char   *src_ptr,
    unsigned short  *output_ptr,
    unsigned int    src_pixels_per_line,
    unsigned int    pixel_step,
    unsigned int    output_height,
    unsigned int    output_width,
    const short      *vp8_filter
);
extern void vp8_filter_block1dc_v6_mmx
(
    unsigned short *src_ptr,
    unsigned char  *output_ptr,
    int             output_pitch,
    unsigned int    pixels_per_line,
    unsigned int    pixel_step,
    unsigned int    output_height,
    unsigned int    output_width,
    const short    *vp8_filter
);
extern void vp8_filter_block1d8_h6_sse2
(
    unsigned char  *src_ptr,
    unsigned short *output_ptr,
    unsigned int    src_pixels_per_line,
    unsigned int    pixel_step,
    unsigned int    output_height,
    unsigned int    output_width,
    const short    *vp8_filter
);
extern void vp8_filter_block1d16_h6_sse2
(
    unsigned char  *src_ptr,
    unsigned short *output_ptr,
    unsigned int    src_pixels_per_line,
    unsigned int    pixel_step,
    unsigned int    output_height,
    unsigned int    output_width,
    const short    *vp8_filter
);
extern void vp8_filter_block1d8_v6_sse2
(
    unsigned short *src_ptr,
    unsigned char *output_ptr,
    int dst_ptich,
    unsigned int pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    const short    *vp8_filter
);
extern void vp8_filter_block1d16_v6_sse2
(
    unsigned short *src_ptr,
    unsigned char *output_ptr,
    int dst_ptich,
    unsigned int pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    const short    *vp8_filter
);
extern void vp8_unpack_block1d16_h6_sse2
(
    unsigned char  *src_ptr,
    unsigned short *output_ptr,
    unsigned int    src_pixels_per_line,
    unsigned int    output_height,
    unsigned int    output_width
);
extern void vp8_filter_block1d8_h6_only_sse2
(
    unsigned char  *src_ptr,
    unsigned int    src_pixels_per_line,
    unsigned char  *output_ptr,
    int dst_ptich,
    unsigned int    output_height,
    const short    *vp8_filter
);
extern void vp8_filter_block1d16_h6_only_sse2
(
    unsigned char  *src_ptr,
    unsigned int    src_pixels_per_line,
    unsigned char  *output_ptr,
    int dst_ptich,
    unsigned int    output_height,
    const short    *vp8_filter
);
extern void vp8_filter_block1d8_v6_only_sse2
(
    unsigned char *src_ptr,
    unsigned int   src_pixels_per_line,
    unsigned char *output_ptr,
    int dst_ptich,
    unsigned int   output_height,
    const short   *vp8_filter
);


#if HAVE_MMX
void vp8_sixtap_predict4x4_mmx
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{
    DECLARE_ALIGNED(16, unsigned short, FData2[16*16]);  /* Temp data bufffer used in filtering */
    const short *HFilter, *VFilter;
    HFilter = vp8_six_tap_mmx[xoffset];
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line), FData2, src_pixels_per_line, 1, 9, 8, HFilter);
    VFilter = vp8_six_tap_mmx[yoffset];
    vp8_filter_block1dc_v6_mmx(FData2 + 8, dst_ptr, dst_pitch, 8, 4 , 4, 4, VFilter);

}


void vp8_sixtap_predict16x16_mmx
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{

    DECLARE_ALIGNED(16, unsigned short, FData2[24*24]);  /* Temp data bufffer used in filtering */

    const short *HFilter, *VFilter;


    HFilter = vp8_six_tap_mmx[xoffset];

    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line),    FData2,   src_pixels_per_line, 1, 21, 32, HFilter);
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line) + 4,  FData2 + 4, src_pixels_per_line, 1, 21, 32, HFilter);
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line) + 8,  FData2 + 8, src_pixels_per_line, 1, 21, 32, HFilter);
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line) + 12, FData2 + 12, src_pixels_per_line, 1, 21, 32, HFilter);

    VFilter = vp8_six_tap_mmx[yoffset];
    vp8_filter_block1dc_v6_mmx(FData2 + 32, dst_ptr,   dst_pitch, 32, 16 , 16, 16, VFilter);
    vp8_filter_block1dc_v6_mmx(FData2 + 36, dst_ptr + 4, dst_pitch, 32, 16 , 16, 16, VFilter);
    vp8_filter_block1dc_v6_mmx(FData2 + 40, dst_ptr + 8, dst_pitch, 32, 16 , 16, 16, VFilter);
    vp8_filter_block1dc_v6_mmx(FData2 + 44, dst_ptr + 12, dst_pitch, 32, 16 , 16, 16, VFilter);

}


void vp8_sixtap_predict8x8_mmx
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{

    DECLARE_ALIGNED(16, unsigned short, FData2[256]);    /* Temp data bufffer used in filtering */

    const short *HFilter, *VFilter;

    HFilter = vp8_six_tap_mmx[xoffset];
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line),    FData2,   src_pixels_per_line, 1, 13, 16, HFilter);
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line) + 4,  FData2 + 4, src_pixels_per_line, 1, 13, 16, HFilter);

    VFilter = vp8_six_tap_mmx[yoffset];
    vp8_filter_block1dc_v6_mmx(FData2 + 16, dst_ptr,   dst_pitch, 16, 8 , 8, 8, VFilter);
    vp8_filter_block1dc_v6_mmx(FData2 + 20, dst_ptr + 4, dst_pitch, 16, 8 , 8, 8, VFilter);

}


void vp8_sixtap_predict8x4_mmx
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{

    DECLARE_ALIGNED(16, unsigned short, FData2[256]);    /* Temp data bufffer used in filtering */

    const short *HFilter, *VFilter;

    HFilter = vp8_six_tap_mmx[xoffset];
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line),    FData2,   src_pixels_per_line, 1, 9, 16, HFilter);
    vp8_filter_block1d_h6_mmx(src_ptr - (2 * src_pixels_per_line) + 4,  FData2 + 4, src_pixels_per_line, 1, 9, 16, HFilter);

    VFilter = vp8_six_tap_mmx[yoffset];
    vp8_filter_block1dc_v6_mmx(FData2 + 16, dst_ptr,   dst_pitch, 16, 8 , 4, 8, VFilter);
    vp8_filter_block1dc_v6_mmx(FData2 + 20, dst_ptr + 4, dst_pitch, 16, 8 , 4, 8, VFilter);

}



void vp8_bilinear_predict16x16_mmx
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{
    vp8_bilinear_predict8x8_mmx(src_ptr,   src_pixels_per_line, xoffset, yoffset, dst_ptr,   dst_pitch);
    vp8_bilinear_predict8x8_mmx(src_ptr + 8, src_pixels_per_line, xoffset, yoffset, dst_ptr + 8, dst_pitch);
    vp8_bilinear_predict8x8_mmx(src_ptr + 8 * src_pixels_per_line,   src_pixels_per_line, xoffset, yoffset, dst_ptr + dst_pitch * 8,   dst_pitch);
    vp8_bilinear_predict8x8_mmx(src_ptr + 8 * src_pixels_per_line + 8, src_pixels_per_line, xoffset, yoffset, dst_ptr + dst_pitch * 8 + 8, dst_pitch);
}
#endif


#if HAVE_SSE2
void vp8_sixtap_predict16x16_sse2
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch

)
{
    DECLARE_ALIGNED(16, unsigned short, FData2[24*24]);    /* Temp data bufffer used in filtering */

    const short *HFilter, *VFilter;

    if (xoffset)
    {
        if (yoffset)
        {
            HFilter = vp8_six_tap_mmx[xoffset];
            vp8_filter_block1d16_h6_sse2(src_ptr - (2 * src_pixels_per_line), FData2,   src_pixels_per_line, 1, 21, 32, HFilter);
            VFilter = vp8_six_tap_mmx[yoffset];
            vp8_filter_block1d16_v6_sse2(FData2 + 32, dst_ptr,   dst_pitch, 32, 16 , 16, dst_pitch, VFilter);
        }
        else
        {
            /* First-pass only */
            HFilter = vp8_six_tap_mmx[xoffset];
            vp8_filter_block1d16_h6_only_sse2(src_ptr, src_pixels_per_line, dst_ptr, dst_pitch, 16, HFilter);
        }
    }
    else
    {
        /* Second-pass only */
        VFilter = vp8_six_tap_mmx[yoffset];
        vp8_unpack_block1d16_h6_sse2(src_ptr - (2 * src_pixels_per_line), FData2,   src_pixels_per_line, 21, 32);
        vp8_filter_block1d16_v6_sse2(FData2 + 32, dst_ptr,   dst_pitch, 32, 16 , 16, dst_pitch, VFilter);
    }
}


void vp8_sixtap_predict8x8_sse2
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{
    DECLARE_ALIGNED(16, unsigned short, FData2[256]);  /* Temp data bufffer used in filtering */
    const short *HFilter, *VFilter;

    if (xoffset)
    {
        if (yoffset)
        {
            HFilter = vp8_six_tap_mmx[xoffset];
            vp8_filter_block1d8_h6_sse2(src_ptr - (2 * src_pixels_per_line), FData2,   src_pixels_per_line, 1, 13, 16, HFilter);
            VFilter = vp8_six_tap_mmx[yoffset];
            vp8_filter_block1d8_v6_sse2(FData2 + 16, dst_ptr,   dst_pitch, 16, 8 , 8, dst_pitch, VFilter);
        }
        else
        {
            /* First-pass only */
            HFilter = vp8_six_tap_mmx[xoffset];
            vp8_filter_block1d8_h6_only_sse2(src_ptr, src_pixels_per_line, dst_ptr, dst_pitch, 8, HFilter);
        }
    }
    else
    {
        /* Second-pass only */
        VFilter = vp8_six_tap_mmx[yoffset];
        vp8_filter_block1d8_v6_only_sse2(src_ptr - (2 * src_pixels_per_line), src_pixels_per_line, dst_ptr, dst_pitch, 8, VFilter);
    }
}


void vp8_sixtap_predict8x4_sse2
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{
    DECLARE_ALIGNED(16, unsigned short, FData2[256]);  /* Temp data bufffer used in filtering */
    const short *HFilter, *VFilter;

    if (xoffset)
    {
        if (yoffset)
        {
            HFilter = vp8_six_tap_mmx[xoffset];
            vp8_filter_block1d8_h6_sse2(src_ptr - (2 * src_pixels_per_line), FData2,   src_pixels_per_line, 1, 9, 16, HFilter);
            VFilter = vp8_six_tap_mmx[yoffset];
            vp8_filter_block1d8_v6_sse2(FData2 + 16, dst_ptr,   dst_pitch, 16, 8 , 4, dst_pitch, VFilter);
        }
        else
        {
            /* First-pass only */
            HFilter = vp8_six_tap_mmx[xoffset];
            vp8_filter_block1d8_h6_only_sse2(src_ptr, src_pixels_per_line, dst_ptr, dst_pitch, 4, HFilter);
        }
    }
    else
    {
        /* Second-pass only */
        VFilter = vp8_six_tap_mmx[yoffset];
        vp8_filter_block1d8_v6_only_sse2(src_ptr - (2 * src_pixels_per_line), src_pixels_per_line, dst_ptr, dst_pitch, 4, VFilter);
    }
}

#endif

#if HAVE_SSSE3

extern void vp8_filter_block1d8_h6_ssse3
(
    unsigned char  *src_ptr,
    unsigned int    src_pixels_per_line,
    unsigned char  *output_ptr,
    unsigned int    output_pitch,
    unsigned int    output_height,
    unsigned int    vp8_filter_index
);

extern void vp8_filter_block1d16_h6_ssse3
(
    unsigned char  *src_ptr,
    unsigned int    src_pixels_per_line,
    unsigned char  *output_ptr,
    unsigned int    output_pitch,
    unsigned int    output_height,
    unsigned int    vp8_filter_index
);

extern void vp8_filter_block1d16_v6_ssse3
(
    unsigned char *src_ptr,
    unsigned int   src_pitch,
    unsigned char *output_ptr,
    unsigned int   out_pitch,
    unsigned int   output_height,
    unsigned int   vp8_filter_index
);

extern void vp8_filter_block1d8_v6_ssse3
(
    unsigned char *src_ptr,
    unsigned int   src_pitch,
    unsigned char *output_ptr,
    unsigned int   out_pitch,
    unsigned int   output_height,
    unsigned int   vp8_filter_index
);

extern void vp8_filter_block1d4_h6_ssse3
(
    unsigned char  *src_ptr,
    unsigned int    src_pixels_per_line,
    unsigned char  *output_ptr,
    unsigned int    output_pitch,
    unsigned int    output_height,
    unsigned int    vp8_filter_index
);

extern void vp8_filter_block1d4_v6_ssse3
(
    unsigned char *src_ptr,
    unsigned int   src_pitch,
    unsigned char *output_ptr,
    unsigned int   out_pitch,
    unsigned int   output_height,
    unsigned int   vp8_filter_index
);

void vp8_sixtap_predict16x16_ssse3
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch

)
{
    DECLARE_ALIGNED(16, unsigned char, FData2[24*24]);

    if (xoffset)
    {
        if (yoffset)
        {
            vp8_filter_block1d16_h6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                          src_pixels_per_line, FData2,
                                          16, 21, xoffset);
            vp8_filter_block1d16_v6_ssse3(FData2 , 16, dst_ptr, dst_pitch,
                                          16, yoffset);
        }
        else
        {
            /* First-pass only */
            vp8_filter_block1d16_h6_ssse3(src_ptr, src_pixels_per_line,
                                          dst_ptr, dst_pitch, 16, xoffset);
        }
    }
    else
    {
        if (yoffset)
        {
            /* Second-pass only */
            vp8_filter_block1d16_v6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                          src_pixels_per_line,
                                          dst_ptr, dst_pitch, 16, yoffset);
        }
        else
        {
            /* ssse3 second-pass only function couldn't handle (xoffset==0 &&
             * yoffset==0) case correctly. Add copy function here to guarantee
             * six-tap function handles all possible offsets. */
            vp8_copy_mem16x16(src_ptr, src_pixels_per_line, dst_ptr, dst_pitch);
        }
    }
}

void vp8_sixtap_predict8x8_ssse3
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{
    DECLARE_ALIGNED(16, unsigned char, FData2[256]);

    if (xoffset)
    {
        if (yoffset)
        {
            vp8_filter_block1d8_h6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                         src_pixels_per_line, FData2,
                                         8, 13, xoffset);
            vp8_filter_block1d8_v6_ssse3(FData2, 8, dst_ptr, dst_pitch,
                                         8, yoffset);
        }
        else
        {
            vp8_filter_block1d8_h6_ssse3(src_ptr, src_pixels_per_line,
                                         dst_ptr, dst_pitch, 8, xoffset);
        }
    }
    else
    {
        if (yoffset)
        {
            /* Second-pass only */
            vp8_filter_block1d8_v6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                         src_pixels_per_line,
                                         dst_ptr, dst_pitch, 8, yoffset);
        }
        else
        {
            /* ssse3 second-pass only function couldn't handle (xoffset==0 &&
             * yoffset==0) case correctly. Add copy function here to guarantee
             * six-tap function handles all possible offsets. */
            vp8_copy_mem8x8(src_ptr, src_pixels_per_line, dst_ptr, dst_pitch);
        }
    }
}


void vp8_sixtap_predict8x4_ssse3
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{
    DECLARE_ALIGNED(16, unsigned char, FData2[256]);

    if (xoffset)
    {
        if (yoffset)
        {
            vp8_filter_block1d8_h6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                         src_pixels_per_line, FData2,
                                         8, 9, xoffset);
            vp8_filter_block1d8_v6_ssse3(FData2, 8, dst_ptr, dst_pitch,
                                         4, yoffset);
        }
        else
        {
            /* First-pass only */
            vp8_filter_block1d8_h6_ssse3(src_ptr, src_pixels_per_line,
                                         dst_ptr, dst_pitch, 4, xoffset);
        }
    }
    else
    {
        if (yoffset)
        {
            /* Second-pass only */
            vp8_filter_block1d8_v6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                         src_pixels_per_line,
                                         dst_ptr, dst_pitch, 4, yoffset);
        }
        else
        {
            /* ssse3 second-pass only function couldn't handle (xoffset==0 &&
             * yoffset==0) case correctly. Add copy function here to guarantee
             * six-tap function handles all possible offsets. */
            vp8_copy_mem8x4(src_ptr, src_pixels_per_line, dst_ptr, dst_pitch);
        }
    }
}

void vp8_sixtap_predict4x4_ssse3
(
    unsigned char  *src_ptr,
    int   src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{
  DECLARE_ALIGNED(16, unsigned char, FData2[4*9]);

  if (xoffset)
  {
      if (yoffset)
      {
          vp8_filter_block1d4_h6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                       src_pixels_per_line,
                                       FData2, 4, 9, xoffset);
          vp8_filter_block1d4_v6_ssse3(FData2, 4, dst_ptr, dst_pitch,
                                       4, yoffset);
      }
      else
      {
          vp8_filter_block1d4_h6_ssse3(src_ptr, src_pixels_per_line,
                                       dst_ptr, dst_pitch, 4, xoffset);
      }
  }
  else
  {
      if (yoffset)
      {
          vp8_filter_block1d4_v6_ssse3(src_ptr - (2 * src_pixels_per_line),
                                       src_pixels_per_line,
                                       dst_ptr, dst_pitch, 4, yoffset);
      }
      else
      {
        /* ssse3 second-pass only function couldn't handle (xoffset==0 &&
          * yoffset==0) case correctly. Add copy function here to guarantee
          * six-tap function handles all possible offsets. */
          int r;

          for (r = 0; r < 4; r++)
          {
            dst_ptr[0]  = src_ptr[0];
            dst_ptr[1]  = src_ptr[1];
            dst_ptr[2]  = src_ptr[2];
            dst_ptr[3]  = src_ptr[3];
            dst_ptr     += dst_pitch;
            src_ptr     += src_pixels_per_line;
          }
      }
  }
}

#endif
