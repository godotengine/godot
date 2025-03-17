/*
 * jsimd_mips.c
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2014, 2016, 2018, 2020, 2022, 2024,
 *           D. R. Commander.
 * Copyright (C) 2013-2014, MIPS Technologies, Inc., California.
 * Copyright (C) 2015-2016, 2018, 2022, Matthieu Darbois.
 *
 * Based on the x86 SIMD extension for IJG JPEG library,
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 * For conditions of distribution and use, see copyright notice in jsimdext.inc
 *
 * This file contains the interface between the "normal" portions
 * of the library and the SIMD implementations when running on a
 * MIPS architecture.
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jsimd.h"
#include "../../src/jdct.h"
#include "../../src/jsimddct.h"
#include "../jsimd.h"

#include <ctype.h>

static THREAD_LOCAL unsigned int simd_support = ~0;

#if !(defined(__mips_dsp) && (__mips_dsp_rev >= 2)) && defined(__linux__)

LOCAL(void)
parse_proc_cpuinfo(const char *search_string)
{
  const char *file_name = "/proc/cpuinfo";
  char cpuinfo_line[256];
  FILE *f = NULL;

  simd_support = 0;

  if ((f = fopen(file_name, "r")) != NULL) {
    while (fgets(cpuinfo_line, sizeof(cpuinfo_line), f) != NULL) {
      if (strstr(cpuinfo_line, search_string) != NULL) {
        fclose(f);
        simd_support |= JSIMD_DSPR2;
        return;
      }
    }
    fclose(f);
  }
  /* Did not find string in the proc file, or not Linux ELF. */
}

#endif

/*
 * Check what SIMD accelerations are supported.
 */
LOCAL(void)
init_simd(void)
{
#ifndef NO_GETENV
  char *env = NULL;
#endif

  if (simd_support != ~0U)
    return;

  simd_support = 0;

#if defined(__mips_dsp) && (__mips_dsp_rev >= 2)
  simd_support |= JSIMD_DSPR2;
#elif defined(__linux__)
  /* We still have a chance to use MIPS DSPR2 regardless of globally used
   * -mdspr2 options passed to gcc by performing runtime detection via
   * /proc/cpuinfo parsing on linux */
  parse_proc_cpuinfo("MIPS 74K");
#endif

#ifndef NO_GETENV
  /* Force different settings through environment variables */
  env = getenv("JSIMD_FORCEDSPR2");
  if ((env != NULL) && (strcmp(env, "1") == 0))
    simd_support = JSIMD_DSPR2;
  env = getenv("JSIMD_FORCENONE");
  if ((env != NULL) && (strcmp(env, "1") == 0))
    simd_support = 0;
#endif
}

static const int mips_idct_ifast_coefs[4] = {
  0x45404540,           /* FIX( 1.082392200 / 2) =  17734 = 0x4546 */
  0x5A805A80,           /* FIX( 1.414213562 / 2) =  23170 = 0x5A82 */
  0x76407640,           /* FIX( 1.847759065 / 2) =  30274 = 0x7642 */
  0xAC60AC60            /* FIX(-2.613125930 / 4) = -21407 = 0xAC61 */
};

/* The following struct is borrowed from jdsample.c */
typedef void (*upsample1_ptr) (j_decompress_ptr cinfo,
                               jpeg_component_info *compptr,
                               JSAMPARRAY input_data,
                               JSAMPARRAY *output_data_ptr);
typedef struct {
  struct jpeg_upsampler pub;
  JSAMPARRAY color_buf[MAX_COMPONENTS];
  upsample1_ptr methods[MAX_COMPONENTS];
  int next_row_out;
  JDIMENSION rows_to_go;
  int rowgroup_height[MAX_COMPONENTS];
  UINT8 h_expand[MAX_COMPONENTS];
  UINT8 v_expand[MAX_COMPONENTS];
} my_upsampler;

typedef my_upsampler *my_upsample_ptr;

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

  if (simd_support & JSIMD_DSPR2)
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

  if (simd_support & JSIMD_DSPR2)
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

  if (simd_support & JSIMD_DSPR2)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_ycc_rgb565(void)
{
  return 0;
}

GLOBAL(int)
jsimd_c_can_null_convert(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if (simd_support & JSIMD_DSPR2)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_rgb_ycc_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                      JSAMPIMAGE output_buf, JDIMENSION output_row,
                      int num_rows)
{
  void (*dspr2fct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);

  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    dspr2fct = jsimd_extrgb_ycc_convert_dspr2;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    dspr2fct = jsimd_extrgbx_ycc_convert_dspr2;
    break;
  case JCS_EXT_BGR:
    dspr2fct = jsimd_extbgr_ycc_convert_dspr2;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    dspr2fct = jsimd_extbgrx_ycc_convert_dspr2;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    dspr2fct = jsimd_extxbgr_ycc_convert_dspr2;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    dspr2fct = jsimd_extxrgb_ycc_convert_dspr2;
    break;
  default:
    dspr2fct = jsimd_extrgb_ycc_convert_dspr2;
    break;
  }

  dspr2fct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
}

GLOBAL(void)
jsimd_rgb_gray_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                       JSAMPIMAGE output_buf, JDIMENSION output_row,
                       int num_rows)
{
  void (*dspr2fct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);

  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    dspr2fct = jsimd_extrgb_gray_convert_dspr2;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    dspr2fct = jsimd_extrgbx_gray_convert_dspr2;
    break;
  case JCS_EXT_BGR:
    dspr2fct = jsimd_extbgr_gray_convert_dspr2;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    dspr2fct = jsimd_extbgrx_gray_convert_dspr2;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    dspr2fct = jsimd_extxbgr_gray_convert_dspr2;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    dspr2fct = jsimd_extxrgb_gray_convert_dspr2;
    break;
  default:
    dspr2fct = jsimd_extrgb_gray_convert_dspr2;
    break;
  }

  dspr2fct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
}

GLOBAL(void)
jsimd_ycc_rgb_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                      JDIMENSION input_row, JSAMPARRAY output_buf,
                      int num_rows)
{
  void (*dspr2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    dspr2fct = jsimd_ycc_extrgb_convert_dspr2;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    dspr2fct = jsimd_ycc_extrgbx_convert_dspr2;
    break;
  case JCS_EXT_BGR:
    dspr2fct = jsimd_ycc_extbgr_convert_dspr2;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    dspr2fct = jsimd_ycc_extbgrx_convert_dspr2;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    dspr2fct = jsimd_ycc_extxbgr_convert_dspr2;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    dspr2fct = jsimd_ycc_extxrgb_convert_dspr2;
    break;
  default:
    dspr2fct = jsimd_ycc_extrgb_convert_dspr2;
    break;
  }

  dspr2fct(cinfo->output_width, input_buf, input_row, output_buf, num_rows);
}

GLOBAL(void)
jsimd_ycc_rgb565_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                         JDIMENSION input_row, JSAMPARRAY output_buf,
                         int num_rows)
{
}

GLOBAL(void)
jsimd_c_null_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                     JSAMPIMAGE output_buf, JDIMENSION output_row,
                     int num_rows)
{
  jsimd_c_null_convert_dspr2(cinfo->image_width, input_buf, output_buf,
                             output_row, num_rows, cinfo->num_components);
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

  /* FIXME: jsimd_h2v2_downsample_dspr2() fails some of the TJBench tiling
   * regression tests, probably because the DSPr2 SIMD implementation predates
   * those tests. */
#if 0
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(int)
jsimd_can_h2v2_smooth_downsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (DCTSIZE != 8)
    return 0;

  if (simd_support & JSIMD_DSPR2)
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

  /* FIXME: jsimd_h2v1_downsample_dspr2() fails some of the TJBench tiling
   * regression tests, probably because the DSPr2 SIMD implementation predates
   * those tests. */
#if 0
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(void)
jsimd_h2v2_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  jsimd_h2v2_downsample_dspr2(cinfo->image_width, cinfo->max_v_samp_factor,
                              compptr->v_samp_factor, compptr->width_in_blocks,
                              input_data, output_data);
}

GLOBAL(void)
jsimd_h2v2_smooth_downsample(j_compress_ptr cinfo,
                             jpeg_component_info *compptr,
                             JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  jsimd_h2v2_smooth_downsample_dspr2(input_data, output_data,
                                     compptr->v_samp_factor,
                                     cinfo->max_v_samp_factor,
                                     cinfo->smoothing_factor,
                                     compptr->width_in_blocks,
                                     cinfo->image_width);
}

GLOBAL(void)
jsimd_h2v1_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  jsimd_h2v1_downsample_dspr2(cinfo->image_width, cinfo->max_v_samp_factor,
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

  if (simd_support & JSIMD_DSPR2)
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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(int)
jsimd_can_int_upsample(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  if (simd_support & JSIMD_DSPR2)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_h2v2_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v2_upsample_dspr2(cinfo->max_v_samp_factor, cinfo->output_width,
                            input_data, output_data_ptr);
}

GLOBAL(void)
jsimd_h2v1_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v1_upsample_dspr2(cinfo->max_v_samp_factor, cinfo->output_width,
                            input_data, output_data_ptr);
}

GLOBAL(void)
jsimd_int_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                   JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  my_upsample_ptr upsample = (my_upsample_ptr)cinfo->upsample;

  jsimd_int_upsample_dspr2(upsample->h_expand[compptr->component_index],
                           upsample->v_expand[compptr->component_index],
                           input_data, output_data_ptr, cinfo->output_width,
                           cinfo->max_v_samp_factor);
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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(void)
jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v2_fancy_upsample_dspr2(cinfo->max_v_samp_factor,
                                  compptr->downsampled_width, input_data,
                                  output_data_ptr);
}

GLOBAL(void)
jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v1_fancy_upsample_dspr2(cinfo->max_v_samp_factor,
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

  if (simd_support & JSIMD_DSPR2)
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

  if (simd_support & JSIMD_DSPR2)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  void (*dspr2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, JSAMPLE *);

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    dspr2fct = jsimd_h2v2_extrgb_merged_upsample_dspr2;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    dspr2fct = jsimd_h2v2_extrgbx_merged_upsample_dspr2;
    break;
  case JCS_EXT_BGR:
    dspr2fct = jsimd_h2v2_extbgr_merged_upsample_dspr2;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    dspr2fct = jsimd_h2v2_extbgrx_merged_upsample_dspr2;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    dspr2fct = jsimd_h2v2_extxbgr_merged_upsample_dspr2;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    dspr2fct = jsimd_h2v2_extxrgb_merged_upsample_dspr2;
    break;
  default:
    dspr2fct = jsimd_h2v2_extrgb_merged_upsample_dspr2;
    break;
  }

  dspr2fct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf,
           cinfo->sample_range_limit);
}

GLOBAL(void)
jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  void (*dspr2fct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, JSAMPLE *);

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    dspr2fct = jsimd_h2v1_extrgb_merged_upsample_dspr2;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    dspr2fct = jsimd_h2v1_extrgbx_merged_upsample_dspr2;
    break;
  case JCS_EXT_BGR:
    dspr2fct = jsimd_h2v1_extbgr_merged_upsample_dspr2;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    dspr2fct = jsimd_h2v1_extbgrx_merged_upsample_dspr2;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    dspr2fct = jsimd_h2v1_extxbgr_merged_upsample_dspr2;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    dspr2fct = jsimd_h2v1_extxrgb_merged_upsample_dspr2;
    break;
  default:
    dspr2fct = jsimd_h2v1_extrgb_merged_upsample_dspr2;
    break;
  }

  dspr2fct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf,
           cinfo->sample_range_limit);
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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(int)
jsimd_can_convsamp_float(void)
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

#ifndef __mips_soft_float
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(void)
jsimd_convsamp(JSAMPARRAY sample_data, JDIMENSION start_col,
               DCTELEM *workspace)
{
  jsimd_convsamp_dspr2(sample_data, start_col, workspace);
}

GLOBAL(void)
jsimd_convsamp_float(JSAMPARRAY sample_data, JDIMENSION start_col,
                     FAST_FLOAT *workspace)
{
#ifndef __mips_soft_float
  jsimd_convsamp_float_dspr2(sample_data, start_col, workspace);
#endif
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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(int)
jsimd_can_fdct_float(void)
{
  return 0;
}

GLOBAL(void)
jsimd_fdct_islow(DCTELEM *data)
{
  jsimd_fdct_islow_dspr2(data);
}

GLOBAL(void)
jsimd_fdct_ifast(DCTELEM *data)
{
  jsimd_fdct_ifast_dspr2(data);
}

GLOBAL(void)
jsimd_fdct_float(FAST_FLOAT *data)
{
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

  if (simd_support & JSIMD_DSPR2)
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
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;

#ifndef __mips_soft_float
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(void)
jsimd_quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  jsimd_quantize_dspr2(coef_block, divisors, workspace);
}

GLOBAL(void)
jsimd_quantize_float(JCOEFPTR coef_block, FAST_FLOAT *divisors,
                     FAST_FLOAT *workspace)
{
#ifndef __mips_soft_float
  jsimd_quantize_float_dspr2(coef_block, divisors, workspace);
#endif
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

  if (simd_support & JSIMD_DSPR2)
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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(int)
jsimd_can_idct_6x6(void)
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

  if (simd_support & JSIMD_DSPR2)
    return 1;

  return 0;
}

GLOBAL(int)
jsimd_can_idct_12x12(void)
{
  init_simd();

  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;

  if (simd_support & JSIMD_DSPR2)
    return 1;

  return 0;
}

GLOBAL(void)
jsimd_idct_2x2(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  jsimd_idct_2x2_dspr2(compptr->dct_table, coef_block, output_buf, output_col);
}

GLOBAL(void)
jsimd_idct_4x4(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  int workspace[DCTSIZE * 4]; /* buffers data between passes */

  jsimd_idct_4x4_dspr2(compptr->dct_table, coef_block, output_buf, output_col,
                       workspace);
}

GLOBAL(void)
jsimd_idct_6x6(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  jsimd_idct_6x6_dspr2(compptr->dct_table, coef_block, output_buf, output_col);
}

GLOBAL(void)
jsimd_idct_12x12(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  int workspace[96];
  int output[12] = {
    (int)(output_buf[0] + output_col),
    (int)(output_buf[1] + output_col),
    (int)(output_buf[2] + output_col),
    (int)(output_buf[3] + output_col),
    (int)(output_buf[4] + output_col),
    (int)(output_buf[5] + output_col),
    (int)(output_buf[6] + output_col),
    (int)(output_buf[7] + output_col),
    (int)(output_buf[8] + output_col),
    (int)(output_buf[9] + output_col),
    (int)(output_buf[10] + output_col),
    (int)(output_buf[11] + output_col)
  };

  jsimd_idct_12x12_pass1_dspr2(coef_block, compptr->dct_table, workspace);
  jsimd_idct_12x12_pass2_dspr2(workspace, output);
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

  if (simd_support & JSIMD_DSPR2)
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

#if defined(__MIPSEL__)
  if (simd_support & JSIMD_DSPR2)
    return 1;
#endif

  return 0;
}

GLOBAL(int)
jsimd_can_idct_float(void)
{
  return 0;
}

GLOBAL(void)
jsimd_idct_islow(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  int output[8] = {
    (int)(output_buf[0] + output_col),
    (int)(output_buf[1] + output_col),
    (int)(output_buf[2] + output_col),
    (int)(output_buf[3] + output_col),
    (int)(output_buf[4] + output_col),
    (int)(output_buf[5] + output_col),
    (int)(output_buf[6] + output_col),
    (int)(output_buf[7] + output_col)
  };

  jsimd_idct_islow_dspr2(coef_block, compptr->dct_table, output,
                         IDCT_range_limit(cinfo));
}

GLOBAL(void)
jsimd_idct_ifast(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  JCOEFPTR inptr;
  IFAST_MULT_TYPE *quantptr;
  DCTELEM workspace[DCTSIZE2];  /* buffers data between passes */

  /* Pass 1: process columns from input, store into work array. */

  inptr = coef_block;
  quantptr = (IFAST_MULT_TYPE *)compptr->dct_table;

  jsimd_idct_ifast_cols_dspr2(inptr, quantptr, workspace,
                              mips_idct_ifast_coefs);

  /* Pass 2: process rows from work array, store into output array. */
  /* Note that we must descale the results by a factor of 8 == 2**3, */
  /* and also undo the PASS1_BITS scaling. */

  jsimd_idct_ifast_rows_dspr2(workspace, output_buf, output_col,
                              mips_idct_ifast_coefs);
}

GLOBAL(void)
jsimd_idct_float(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
}

GLOBAL(int)
jsimd_can_huff_encode_one_block(void)
{
  return 0;
}

GLOBAL(JOCTET *)
jsimd_huff_encode_one_block(void *state, JOCTET *buffer, JCOEFPTR block,
                            int last_dc_val, c_derived_tbl *dctbl,
                            c_derived_tbl *actbl)
{
  return NULL;
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_first_prepare(void)
{
  return 0;
}

GLOBAL(void)
jsimd_encode_mcu_AC_first_prepare(const JCOEF *block,
                                  const int *jpeg_natural_order_start, int Sl,
                                  int Al, UJCOEF *values, size_t *zerobits)
{
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_refine_prepare(void)
{
  return 0;
}

GLOBAL(int)
jsimd_encode_mcu_AC_refine_prepare(const JCOEF *block,
                                   const int *jpeg_natural_order_start, int Sl,
                                   int Al, UJCOEF *absvalues, size_t *bits)
{
  return 0;
}
