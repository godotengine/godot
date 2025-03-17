/*
 * jsimd_i386.c
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2013-2014, 2016, 2018, 2022-2024, D. R. Commander.
 * Copyright (C) 2015-2016, 2018, 2022, Matthieu Darbois.
 *
 * Based on the x86 SIMD extension for IJG JPEG library,
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 * For conditions of distribution and use, see copyright notice in jsimdext.inc
 *
 * This file contains the interface between the "normal" portions
 * of the library and the SIMD implementations when running on a
 * 32-bit x86 architecture.
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jsimd.h"
#include "../../src/jdct.h"
#include "../../src/jsimddct.h"
#include "../jsimd.h"

/*
 * In the PIC cases, we have no guarantee that constants will keep
 * their alignment. This macro allows us to verify it at runtime.
 */
#define IS_ALIGNED(ptr, order)  (((unsigned)ptr & ((1 << order) - 1)) == 0)

#define IS_ALIGNED_SSE(ptr)  (IS_ALIGNED(ptr, 4)) /* 16 byte alignment */
#define IS_ALIGNED_AVX(ptr)  (IS_ALIGNED(ptr, 5)) /* 32 byte alignment */

static THREAD_LOCAL unsigned int simd_support = (unsigned int)(~0);
static THREAD_LOCAL unsigned int simd_huffman = 1;

/*
 * Check what SIMD accelerations are supported.
 */
LOCAL(void)
init_simd(void)
{
#ifndef NO_GETENV
  char env[2] = { 0 };
#endif

  if (simd_support != ~0U)
    return;

  simd_support = jpeg_simd_cpu_support();

#ifndef NO_GETENV
  /* Force different settings through environment variables */
  if (!GETENV_S(env, 2, "JSIMD_FORCEMMX") && !strcmp(env, "1"))
    simd_support &= JSIMD_MMX;
  if (!GETENV_S(env, 2, "JSIMD_FORCE3DNOW") && !strcmp(env, "1"))
    simd_support &= JSIMD_3DNOW | JSIMD_MMX;
  if (!GETENV_S(env, 2, "JSIMD_FORCESSE") && !strcmp(env, "1"))
    simd_support &= JSIMD_SSE | JSIMD_MMX;
  if (!GETENV_S(env, 2, "JSIMD_FORCESSE2") && !strcmp(env, "1"))
    simd_support &= JSIMD_SSE2;
  if (!GETENV_S(env, 2, "JSIMD_FORCEAVX2") && !strcmp(env, "1"))
    simd_support &= JSIMD_AVX2;
  if (!GETENV_S(env, 2, "JSIMD_FORCENONE") && !strcmp(env, "1"))
    simd_support = 0;
  if (!GETENV_S(env, 2, "JSIMD_NOHUFFENC") && !strcmp(env, "1"))
    simd_huffman = 0;
#endif
}

GLOBAL(int)
jsimd_can_rgb_ycc(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  if ((simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_rgb_ycc_convert_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_rgb_ycc_convert_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_rgb_gray(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  if ((simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_rgb_gray_convert_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_rgb_gray_convert_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_ycc_rgb(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  if ((simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_ycc_rgb_convert_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_ycc_rgb_convert_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_ycc_rgb565(void)
{
  return 0;
}

GLOBAL(void)
jsimd_rgb_ycc_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                      JSAMPIMAGE output_buf, JDIMENSION output_row,
                      int num_rows)
{
  void (*avx2fct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);
  void (*sse2fct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);
  void (*mmxfct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);

  if (simd_support == ~0U)
    init_simd();

  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    avx2fct = jsimd_extrgb_ycc_convert_avx2;
    sse2fct = jsimd_extrgb_ycc_convert_sse2;
    mmxfct = jsimd_extrgb_ycc_convert_mmx;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    avx2fct = jsimd_extrgbx_ycc_convert_avx2;
    sse2fct = jsimd_extrgbx_ycc_convert_sse2;
    mmxfct = jsimd_extrgbx_ycc_convert_mmx;
    break;
  case JCS_EXT_BGR:
    avx2fct = jsimd_extbgr_ycc_convert_avx2;
    sse2fct = jsimd_extbgr_ycc_convert_sse2;
    mmxfct = jsimd_extbgr_ycc_convert_mmx;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    avx2fct = jsimd_extbgrx_ycc_convert_avx2;
    sse2fct = jsimd_extbgrx_ycc_convert_sse2;
    mmxfct = jsimd_extbgrx_ycc_convert_mmx;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    avx2fct = jsimd_extxbgr_ycc_convert_avx2;
    sse2fct = jsimd_extxbgr_ycc_convert_sse2;
    mmxfct = jsimd_extxbgr_ycc_convert_mmx;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    avx2fct = jsimd_extxrgb_ycc_convert_avx2;
    sse2fct = jsimd_extxrgb_ycc_convert_sse2;
    mmxfct = jsimd_extxrgb_ycc_convert_mmx;
    break;
  default:
    avx2fct = jsimd_rgb_ycc_convert_avx2;
    sse2fct = jsimd_rgb_ycc_convert_sse2;
    mmxfct = jsimd_rgb_ycc_convert_mmx;
    break;
  }

  if (simd_support & JSIMD_AVX2)
    avx2fct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
  else if (simd_support & JSIMD_SSE2)
    sse2fct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
  else
    mmxfct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
}

GLOBAL(void)
jsimd_rgb_gray_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                       JSAMPIMAGE output_buf, JDIMENSION output_row,
                       int num_rows)
{
  void (*avx2fct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);
  void (*sse2fct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);
  void (*mmxfct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);

  if (simd_support == ~0U)
    init_simd();

  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    avx2fct = jsimd_extrgb_gray_convert_avx2;
    sse2fct = jsimd_extrgb_gray_convert_sse2;
    mmxfct = jsimd_extrgb_gray_convert_mmx;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    avx2fct = jsimd_extrgbx_gray_convert_avx2;
    sse2fct = jsimd_extrgbx_gray_convert_sse2;
    mmxfct = jsimd_extrgbx_gray_convert_mmx;
    break;
  case JCS_EXT_BGR:
    avx2fct = jsimd_extbgr_gray_convert_avx2;
    sse2fct = jsimd_extbgr_gray_convert_sse2;
    mmxfct = jsimd_extbgr_gray_convert_mmx;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    avx2fct = jsimd_extbgrx_gray_convert_avx2;
    sse2fct = jsimd_extbgrx_gray_convert_sse2;
    mmxfct = jsimd_extbgrx_gray_convert_mmx;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    avx2fct = jsimd_extxbgr_gray_convert_avx2;
    sse2fct = jsimd_extxbgr_gray_convert_sse2;
    mmxfct = jsimd_extxbgr_gray_convert_mmx;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    avx2fct = jsimd_extxrgb_gray_convert_avx2;
    sse2fct = jsimd_extxrgb_gray_convert_sse2;
    mmxfct = jsimd_extxrgb_gray_convert_mmx;
    break;
  default:
    avx2fct = jsimd_rgb_gray_convert_avx2;
    sse2fct = jsimd_rgb_gray_convert_sse2;
    mmxfct = jsimd_rgb_gray_convert_mmx;
    break;
  }

  if (simd_support & JSIMD_AVX2)
    avx2fct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
  else if (simd_support & JSIMD_SSE2)
    sse2fct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
  else
    mmxfct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
}

GLOBAL(void)
jsimd_ycc_rgb_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                      JDIMENSION input_row, JSAMPARRAY output_buf,
                      int num_rows)
{
  void (*avx2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
  void (*sse2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);
  void (*mmxfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);

  if (simd_support == ~0U)
    init_simd();

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    avx2fct = jsimd_ycc_extrgb_convert_avx2;
    sse2fct = jsimd_ycc_extrgb_convert_sse2;
    mmxfct = jsimd_ycc_extrgb_convert_mmx;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    avx2fct = jsimd_ycc_extrgbx_convert_avx2;
    sse2fct = jsimd_ycc_extrgbx_convert_sse2;
    mmxfct = jsimd_ycc_extrgbx_convert_mmx;
    break;
  case JCS_EXT_BGR:
    avx2fct = jsimd_ycc_extbgr_convert_avx2;
    sse2fct = jsimd_ycc_extbgr_convert_sse2;
    mmxfct = jsimd_ycc_extbgr_convert_mmx;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    avx2fct = jsimd_ycc_extbgrx_convert_avx2;
    sse2fct = jsimd_ycc_extbgrx_convert_sse2;
    mmxfct = jsimd_ycc_extbgrx_convert_mmx;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    avx2fct = jsimd_ycc_extxbgr_convert_avx2;
    sse2fct = jsimd_ycc_extxbgr_convert_sse2;
    mmxfct = jsimd_ycc_extxbgr_convert_mmx;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    avx2fct = jsimd_ycc_extxrgb_convert_avx2;
    sse2fct = jsimd_ycc_extxrgb_convert_sse2;
    mmxfct = jsimd_ycc_extxrgb_convert_mmx;
    break;
  default:
    avx2fct = jsimd_ycc_rgb_convert_avx2;
    sse2fct = jsimd_ycc_rgb_convert_sse2;
    mmxfct = jsimd_ycc_rgb_convert_mmx;
    break;
  }

  if (simd_support & JSIMD_AVX2)
    avx2fct(cinfo->output_width, input_buf, input_row, output_buf, num_rows);
  else if (simd_support & JSIMD_SSE2)
    sse2fct(cinfo->output_width, input_buf, input_row, output_buf, num_rows);
  else
    mmxfct(cinfo->output_width, input_buf, input_row, output_buf, num_rows);
}

GLOBAL(void)
jsimd_ycc_rgb565_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                         JDIMENSION input_row, JSAMPARRAY output_buf,
                         int num_rows)
{
}

GLOBAL(int)
jsimd_can_h2v2_downsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if (simd_support & JSIMD_AVX2)
    return 1;
  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_downsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if (simd_support & JSIMD_AVX2)
    return 1;
  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_h2v2_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_h2v2_downsample_avx2(cinfo->image_width, cinfo->max_v_samp_factor,
                               compptr->v_samp_factor,
                               compptr->width_in_blocks, input_data,
                               output_data);
  else if (simd_support & JSIMD_SSE2)
    jsimd_h2v2_downsample_sse2(cinfo->image_width, cinfo->max_v_samp_factor,
                               compptr->v_samp_factor,
                               compptr->width_in_blocks, input_data,
                               output_data);
  else
    jsimd_h2v2_downsample_mmx(cinfo->image_width, cinfo->max_v_samp_factor,
                              compptr->v_samp_factor, compptr->width_in_blocks,
                              input_data, output_data);
}

GLOBAL(void)
jsimd_h2v1_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_h2v1_downsample_avx2(cinfo->image_width, cinfo->max_v_samp_factor,
                               compptr->v_samp_factor,
                               compptr->width_in_blocks, input_data,
                               output_data);
  else if (simd_support & JSIMD_SSE2)
    jsimd_h2v1_downsample_sse2(cinfo->image_width, cinfo->max_v_samp_factor,
                               compptr->v_samp_factor,
                               compptr->width_in_blocks, input_data,
                               output_data);
  else
    jsimd_h2v1_downsample_mmx(cinfo->image_width, cinfo->max_v_samp_factor,
                              compptr->v_samp_factor, compptr->width_in_blocks,
                              input_data, output_data);
}

GLOBAL(int)
jsimd_can_h2v2_upsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if (simd_support & JSIMD_AVX2)
    return 1;
  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_upsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if (simd_support & JSIMD_AVX2)
    return 1;
  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_h2v2_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_h2v2_upsample_avx2(cinfo->max_v_samp_factor, cinfo->output_width,
                             input_data, output_data_ptr);
  else if (simd_support & JSIMD_SSE2)
    jsimd_h2v2_upsample_sse2(cinfo->max_v_samp_factor, cinfo->output_width,
                             input_data, output_data_ptr);
  else
    jsimd_h2v2_upsample_mmx(cinfo->max_v_samp_factor, cinfo->output_width,
                            input_data, output_data_ptr);
}

GLOBAL(void)
jsimd_h2v1_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_h2v1_upsample_avx2(cinfo->max_v_samp_factor, cinfo->output_width,
                             input_data, output_data_ptr);
  else if (simd_support & JSIMD_SSE2)
    jsimd_h2v1_upsample_sse2(cinfo->max_v_samp_factor, cinfo->output_width,
                             input_data, output_data_ptr);
  else
    jsimd_h2v1_upsample_mmx(cinfo->max_v_samp_factor, cinfo->output_width,
                            input_data, output_data_ptr);
}

GLOBAL(int)
jsimd_can_h2v2_fancy_upsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if ((simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_fancy_upsample_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_fancy_upsample_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_fancy_upsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if ((simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_fancy_upsample_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_fancy_upsample_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_h2v2_fancy_upsample_avx2(cinfo->max_v_samp_factor,
                                   compptr->downsampled_width, input_data,
                                   output_data_ptr);
  else if (simd_support & JSIMD_SSE2)
    jsimd_h2v2_fancy_upsample_sse2(cinfo->max_v_samp_factor,
                                   compptr->downsampled_width, input_data,
                                   output_data_ptr);
  else
    jsimd_h2v2_fancy_upsample_mmx(cinfo->max_v_samp_factor,
                                  compptr->downsampled_width, input_data,
                                  output_data_ptr);
}

GLOBAL(void)
jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_h2v1_fancy_upsample_avx2(cinfo->max_v_samp_factor,
                                   compptr->downsampled_width, input_data,
                                   output_data_ptr);
  else if (simd_support & JSIMD_SSE2)
    jsimd_h2v1_fancy_upsample_sse2(cinfo->max_v_samp_factor,
                                   compptr->downsampled_width, input_data,
                                   output_data_ptr);
  else
    jsimd_h2v1_fancy_upsample_mmx(cinfo->max_v_samp_factor,
                                  compptr->downsampled_width, input_data,
                                  output_data_ptr);
}

GLOBAL(int)
jsimd_can_h2v2_merged_upsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if ((simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_merged_upsample_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_merged_upsample_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_merged_upsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if ((simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_merged_upsample_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_merged_upsample_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  void (*avx2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
  void (*sse2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
  void (*mmxfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);

  if (simd_support == ~0U)
    init_simd();

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    avx2fct = jsimd_h2v2_extrgb_merged_upsample_avx2;
    sse2fct = jsimd_h2v2_extrgb_merged_upsample_sse2;
    mmxfct = jsimd_h2v2_extrgb_merged_upsample_mmx;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    avx2fct = jsimd_h2v2_extrgbx_merged_upsample_avx2;
    sse2fct = jsimd_h2v2_extrgbx_merged_upsample_sse2;
    mmxfct = jsimd_h2v2_extrgbx_merged_upsample_mmx;
    break;
  case JCS_EXT_BGR:
    avx2fct = jsimd_h2v2_extbgr_merged_upsample_avx2;
    sse2fct = jsimd_h2v2_extbgr_merged_upsample_sse2;
    mmxfct = jsimd_h2v2_extbgr_merged_upsample_mmx;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    avx2fct = jsimd_h2v2_extbgrx_merged_upsample_avx2;
    sse2fct = jsimd_h2v2_extbgrx_merged_upsample_sse2;
    mmxfct = jsimd_h2v2_extbgrx_merged_upsample_mmx;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    avx2fct = jsimd_h2v2_extxbgr_merged_upsample_avx2;
    sse2fct = jsimd_h2v2_extxbgr_merged_upsample_sse2;
    mmxfct = jsimd_h2v2_extxbgr_merged_upsample_mmx;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    avx2fct = jsimd_h2v2_extxrgb_merged_upsample_avx2;
    sse2fct = jsimd_h2v2_extxrgb_merged_upsample_sse2;
    mmxfct = jsimd_h2v2_extxrgb_merged_upsample_mmx;
    break;
  default:
    avx2fct = jsimd_h2v2_merged_upsample_avx2;
    sse2fct = jsimd_h2v2_merged_upsample_sse2;
    mmxfct = jsimd_h2v2_merged_upsample_mmx;
    break;
  }

  if (simd_support & JSIMD_AVX2)
    avx2fct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
  else if (simd_support & JSIMD_SSE2)
    sse2fct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
  else
    mmxfct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
}

GLOBAL(void)
jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  void (*avx2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
  void (*sse2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);
  void (*mmxfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);

  if (simd_support == ~0U)
    init_simd();

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    avx2fct = jsimd_h2v1_extrgb_merged_upsample_avx2;
    sse2fct = jsimd_h2v1_extrgb_merged_upsample_sse2;
    mmxfct = jsimd_h2v1_extrgb_merged_upsample_mmx;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    avx2fct = jsimd_h2v1_extrgbx_merged_upsample_avx2;
    sse2fct = jsimd_h2v1_extrgbx_merged_upsample_sse2;
    mmxfct = jsimd_h2v1_extrgbx_merged_upsample_mmx;
    break;
  case JCS_EXT_BGR:
    avx2fct = jsimd_h2v1_extbgr_merged_upsample_avx2;
    sse2fct = jsimd_h2v1_extbgr_merged_upsample_sse2;
    mmxfct = jsimd_h2v1_extbgr_merged_upsample_mmx;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    avx2fct = jsimd_h2v1_extbgrx_merged_upsample_avx2;
    sse2fct = jsimd_h2v1_extbgrx_merged_upsample_sse2;
    mmxfct = jsimd_h2v1_extbgrx_merged_upsample_mmx;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    avx2fct = jsimd_h2v1_extxbgr_merged_upsample_avx2;
    sse2fct = jsimd_h2v1_extxbgr_merged_upsample_sse2;
    mmxfct = jsimd_h2v1_extxbgr_merged_upsample_mmx;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    avx2fct = jsimd_h2v1_extxrgb_merged_upsample_avx2;
    sse2fct = jsimd_h2v1_extxrgb_merged_upsample_sse2;
    mmxfct = jsimd_h2v1_extxrgb_merged_upsample_mmx;
    break;
  default:
    avx2fct = jsimd_h2v1_merged_upsample_avx2;
    sse2fct = jsimd_h2v1_merged_upsample_sse2;
    mmxfct = jsimd_h2v1_merged_upsample_mmx;
    break;
  }

  if (simd_support & JSIMD_AVX2)
    avx2fct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
  else if (simd_support & JSIMD_SSE2)
    sse2fct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
  else
    mmxfct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
}

GLOBAL(int)
jsimd_can_convsamp(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  if (simd_support & JSIMD_AVX2)
    return 1;
  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_convsamp_float(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;

  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_SSE)
    return 1;
  if (simd_support & JSIMD_3DNOW)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_convsamp(JSAMPARRAY sample_data, JDIMENSION start_col,
               DCTELEM *workspace)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_convsamp_avx2(sample_data, start_col, workspace);
  else if (simd_support & JSIMD_SSE2)
    jsimd_convsamp_sse2(sample_data, start_col, workspace);
  else
    jsimd_convsamp_mmx(sample_data, start_col, workspace);
}

GLOBAL(void)
jsimd_convsamp_float(JSAMPARRAY sample_data, JDIMENSION start_col,
                     FAST_FLOAT *workspace)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_SSE2)
    jsimd_convsamp_float_sse2(sample_data, start_col, workspace);
  else if (simd_support & JSIMD_SSE)
    jsimd_convsamp_float_sse(sample_data, start_col, workspace);
  else
    jsimd_convsamp_float_3dnow(sample_data, start_col, workspace);
}

GLOBAL(int)
jsimd_can_fdct_islow(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  if ((simd_support & JSIMD_AVX2) && IS_ALIGNED_AVX(jconst_fdct_islow_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_fdct_islow_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_fdct_ifast(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_fdct_ifast_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_fdct_float(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;

  if ((simd_support & JSIMD_SSE) && IS_ALIGNED_SSE(jconst_fdct_float_sse))
    return 1;
  if (simd_support & JSIMD_3DNOW)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_fdct_islow(DCTELEM *data)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_fdct_islow_avx2(data);
  else if (simd_support & JSIMD_SSE2)
    jsimd_fdct_islow_sse2(data);
  else
    jsimd_fdct_islow_mmx(data);
}

GLOBAL(void)
jsimd_fdct_ifast(DCTELEM *data)
{
  if (simd_support == ~0U)
    init_simd();

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_fdct_islow_sse2))
    jsimd_fdct_ifast_sse2(data);
  else
    jsimd_fdct_ifast_mmx(data);
}

GLOBAL(void)
jsimd_fdct_float(FAST_FLOAT *data)
{
  if (simd_support == ~0U)
    init_simd();

  if ((simd_support & JSIMD_SSE) && IS_ALIGNED_SSE(jconst_fdct_float_sse))
    jsimd_fdct_float_sse(data);
  else if (simd_support & JSIMD_3DNOW)
    jsimd_fdct_float_3dnow(data);
}

GLOBAL(int)
jsimd_can_quantize(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  if (simd_support & JSIMD_AVX2)
    return 1;
  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_quantize_float(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;

  if (simd_support & JSIMD_SSE2)
    return 1;
  if (simd_support & JSIMD_SSE)
    return 1;
  if (simd_support & JSIMD_3DNOW)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_quantize_avx2(coef_block, divisors, workspace);
  else if (simd_support & JSIMD_SSE2)
    jsimd_quantize_sse2(coef_block, divisors, workspace);
  else
    jsimd_quantize_mmx(coef_block, divisors, workspace);
}

GLOBAL(void)
jsimd_quantize_float(JCOEFPTR coef_block, FAST_FLOAT *divisors,
                     FAST_FLOAT *workspace)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_SSE2)
    jsimd_quantize_float_sse2(coef_block, divisors, workspace);
  else if (simd_support & JSIMD_SSE)
    jsimd_quantize_float_sse(coef_block, divisors, workspace);
  else
    jsimd_quantize_float_3dnow(coef_block, divisors, workspace);
}

GLOBAL(int)
jsimd_can_idct_2x2(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_red_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_idct_4x4(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_red_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_idct_2x2(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  if (simd_support == ~0U)
    init_simd();

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_red_sse2))
    jsimd_idct_2x2_sse2(compptr->dct_table, coef_block, output_buf,
                        output_col);
  else
    jsimd_idct_2x2_mmx(compptr->dct_table, coef_block, output_buf, output_col);
}

GLOBAL(void)
jsimd_idct_4x4(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  if (simd_support == ~0U)
    init_simd();

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_red_sse2))
    jsimd_idct_4x4_sse2(compptr->dct_table, coef_block, output_buf,
                        output_col);
  else
    jsimd_idct_4x4_mmx(compptr->dct_table, coef_block, output_buf, output_col);
}

GLOBAL(int)
jsimd_can_idct_islow(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;

  if ((simd_support & JSIMD_AVX2) && IS_ALIGNED_AVX(jconst_idct_islow_avx2))
    return 1;
  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_islow_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_idct_ifast(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(IFAST_MULT_TYPE) != 2)
    return 0;
  if (IFAST_SCALE_BITS != 2)
    return 0;

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_ifast_sse2))
    return 1;
  if (simd_support & JSIMD_MMX)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_idct_float(void)
{
  init_simd();

  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;
  if (sizeof(FLOAT_MULT_TYPE) != 4)
    return 0;

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_float_sse2))
    return 1;
  if ((simd_support & JSIMD_SSE) && IS_ALIGNED_SSE(jconst_idct_float_sse))
    return 1;
  if (simd_support & JSIMD_3DNOW)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_idct_islow(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  if (simd_support == ~0U)
    init_simd();

  if (simd_support & JSIMD_AVX2)
    jsimd_idct_islow_avx2(compptr->dct_table, coef_block, output_buf,
                          output_col);
  else if (simd_support & JSIMD_SSE2)
    jsimd_idct_islow_sse2(compptr->dct_table, coef_block, output_buf,
                          output_col);
  else
    jsimd_idct_islow_mmx(compptr->dct_table, coef_block, output_buf,
                         output_col);
}

GLOBAL(void)
jsimd_idct_ifast(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  if (simd_support == ~0U)
    init_simd();

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_ifast_sse2))
    jsimd_idct_ifast_sse2(compptr->dct_table, coef_block, output_buf,
                          output_col);
  else
    jsimd_idct_ifast_mmx(compptr->dct_table, coef_block, output_buf,
                         output_col);
}

GLOBAL(void)
jsimd_idct_float(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  if (simd_support == ~0U)
    init_simd();

  if ((simd_support & JSIMD_SSE2) && IS_ALIGNED_SSE(jconst_idct_float_sse2))
    jsimd_idct_float_sse2(compptr->dct_table, coef_block, output_buf,
                          output_col);
  else if ((simd_support & JSIMD_SSE) && IS_ALIGNED_SSE(jconst_idct_float_sse))
    jsimd_idct_float_sse(compptr->dct_table, coef_block, output_buf,
                         output_col);
  else
    jsimd_idct_float_3dnow(compptr->dct_table, coef_block, output_buf,
                           output_col);
}

GLOBAL(int)
jsimd_can_huff_encode_one_block(void)
{
  init_simd();

  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;

  if ((simd_support & JSIMD_SSE2) && simd_huffman &&
      IS_ALIGNED_SSE(jconst_huff_encode_one_block))
    return 1;

  return 0;
}

GLOBAL(JOCTET *)
jsimd_huff_encode_one_block(void *state, JOCTET *buffer, JCOEFPTR block,
                            int last_dc_val, c_derived_tbl *dctbl,
                            c_derived_tbl *actbl)
{
  return jsimd_huff_encode_one_block_sse2(state, buffer, block, last_dc_val,
                                          dctbl, actbl);
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_first_prepare(void)
{
  init_simd();

  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (SIZEOF_SIZE_T != 4)
    return 0;
  if (simd_support & JSIMD_SSE2)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_encode_mcu_AC_first_prepare(const JCOEF *block,
                                  const int *jpeg_natural_order_start, int Sl,
                                  int Al, UJCOEF *values, size_t *zerobits)
{
  jsimd_encode_mcu_AC_first_prepare_sse2(block, jpeg_natural_order_start,
                                         Sl, Al, values, zerobits);
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_refine_prepare(void)
{
  init_simd();

  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (SIZEOF_SIZE_T != 4)
    return 0;
  if (simd_support & JSIMD_SSE2)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_encode_mcu_AC_refine_prepare(const JCOEF *block,
                                   const int *jpeg_natural_order_start, int Sl,
                                   int Al, UJCOEF *absvalues, size_t *bits)
{
  return jsimd_encode_mcu_AC_refine_prepare_sse2(block,
                                                 jpeg_natural_order_start,
                                                 Sl, Al, absvalues, bits);
}
