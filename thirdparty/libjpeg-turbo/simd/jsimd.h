/*
 * simd/jsimd.h
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2011, 2014-2016, 2018, 2020, 2022, D. R. Commander.
 * Copyright (C) 2013-2014, MIPS Technologies, Inc., California.
 * Copyright (C) 2014, Linaro Limited.
 * Copyright (C) 2015-2016, 2018, 2022, Matthieu Darbois.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 * Copyright (C) 2020, Arm Limited.
 *
 * Based on the x86 SIMD extension for IJG JPEG library,
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 * For conditions of distribution and use, see copyright notice in jsimdext.inc
 *
 */

/* Bitmask for supported acceleration methods */

#define JSIMD_NONE     0x00
#define JSIMD_MMX      0x01
#define JSIMD_3DNOW    0x02
#define JSIMD_SSE      0x04
#define JSIMD_SSE2     0x08
#define JSIMD_NEON     0x10
#define JSIMD_DSPR2    0x20
#define JSIMD_ALTIVEC  0x40
#define JSIMD_AVX2     0x80
#define JSIMD_MMI      0x100

/* SIMD Ext: retrieve SIMD/CPU information */
EXTERN(unsigned int) jpeg_simd_cpu_support(void);

/* RGB & extended RGB --> YCC Colorspace Conversion */
EXTERN(void) jsimd_rgb_ycc_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_ycc_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_ycc_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_ycc_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_ycc_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_ycc_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

extern const int jconst_rgb_ycc_convert_sse2[];
EXTERN(void) jsimd_rgb_ycc_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_ycc_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_ycc_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_ycc_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_ycc_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_ycc_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

extern const int jconst_rgb_ycc_convert_avx2[];
EXTERN(void) jsimd_rgb_ycc_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_ycc_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_ycc_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_ycc_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_ycc_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_ycc_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

EXTERN(void) jsimd_rgb_ycc_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_ycc_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_ycc_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_ycc_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_ycc_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_ycc_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

#ifndef NEON_INTRINSICS

EXTERN(void) jsimd_extrgb_ycc_convert_neon_slowld3
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_neon_slowld3
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

#endif

EXTERN(void) jsimd_rgb_ycc_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_ycc_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_ycc_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_ycc_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_ycc_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_ycc_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

EXTERN(void) jsimd_rgb_ycc_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_ycc_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_ycc_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_ycc_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_ycc_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_ycc_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

EXTERN(void) jsimd_rgb_ycc_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_ycc_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_ycc_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_ycc_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_ycc_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_ycc_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_ycc_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

/* RGB & extended RGB --> Grayscale Colorspace Conversion */
EXTERN(void) jsimd_rgb_gray_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_gray_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_gray_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_gray_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_gray_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_gray_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_gray_convert_mmx
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

extern const int jconst_rgb_gray_convert_sse2[];
EXTERN(void) jsimd_rgb_gray_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_gray_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_gray_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_gray_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_gray_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_gray_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_gray_convert_sse2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

extern const int jconst_rgb_gray_convert_avx2[];
EXTERN(void) jsimd_rgb_gray_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_gray_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_gray_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_gray_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_gray_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_gray_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_gray_convert_avx2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

EXTERN(void) jsimd_rgb_gray_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_gray_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_gray_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_gray_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_gray_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_gray_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_gray_convert_neon
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

EXTERN(void) jsimd_rgb_gray_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_gray_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_gray_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_gray_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_gray_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_gray_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_gray_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

EXTERN(void) jsimd_rgb_gray_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_gray_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_gray_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_gray_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_gray_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_gray_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_gray_convert_mmi
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

EXTERN(void) jsimd_rgb_gray_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgb_gray_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extrgbx_gray_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgr_gray_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extbgrx_gray_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxbgr_gray_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);
EXTERN(void) jsimd_extxrgb_gray_convert_altivec
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows);

/* YCC --> RGB & extended RGB Colorspace Conversion */
EXTERN(void) jsimd_ycc_rgb_convert_mmx
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgb_convert_mmx
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgbx_convert_mmx
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_mmx
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgrx_convert_mmx
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxbgr_convert_mmx
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxrgb_convert_mmx
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

extern const int jconst_ycc_rgb_convert_sse2[];
EXTERN(void) jsimd_ycc_rgb_convert_sse2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgb_convert_sse2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgbx_convert_sse2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_sse2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgrx_convert_sse2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxbgr_convert_sse2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxrgb_convert_sse2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

extern const int jconst_ycc_rgb_convert_avx2[];
EXTERN(void) jsimd_ycc_rgb_convert_avx2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgb_convert_avx2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgbx_convert_avx2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_avx2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgrx_convert_avx2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxbgr_convert_avx2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxrgb_convert_avx2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

EXTERN(void) jsimd_ycc_rgb_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgb_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgbx_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgrx_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxbgr_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxrgb_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_rgb565_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

#ifndef NEON_INTRINSICS

EXTERN(void) jsimd_ycc_extrgb_convert_neon_slowst3
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_neon_slowst3
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

#endif

EXTERN(void) jsimd_ycc_rgb_convert_dspr2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgb_convert_dspr2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgbx_convert_dspr2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_dspr2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgrx_convert_dspr2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxbgr_convert_dspr2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxrgb_convert_dspr2
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

EXTERN(void) jsimd_ycc_rgb_convert_mmi
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgb_convert_mmi
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgbx_convert_mmi
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_mmi
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgrx_convert_mmi
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxbgr_convert_mmi
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxrgb_convert_mmi
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

EXTERN(void) jsimd_ycc_rgb_convert_altivec
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgb_convert_altivec
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extrgbx_convert_altivec
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgr_convert_altivec
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extbgrx_convert_altivec
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxbgr_convert_altivec
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);
EXTERN(void) jsimd_ycc_extxrgb_convert_altivec
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);

/* NULL Colorspace Conversion */
EXTERN(void) jsimd_c_null_convert_dspr2
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf,
   JDIMENSION output_row, int num_rows, int num_components);

/* h2v1 Downsampling */
EXTERN(void) jsimd_h2v1_downsample_mmx
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_sse2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_avx2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_neon
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_dspr2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_altivec
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

/* h2v2 Downsampling */
EXTERN(void) jsimd_h2v2_downsample_mmx
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v2_downsample_sse2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v2_downsample_avx2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v2_downsample_neon
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v2_downsample_dspr2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v2_downsample_mmi
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v2_downsample_altivec
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

/* h2v2 Smooth Downsampling */
EXTERN(void) jsimd_h2v2_smooth_downsample_dspr2
  (JSAMPARRAY input_data, JSAMPARRAY output_data, JDIMENSION v_samp_factor,
   int max_v_samp_factor, int smoothing_factor, JDIMENSION width_in_blocks,
   JDIMENSION image_width);


/* Upsampling */
EXTERN(void) jsimd_h2v1_upsample_mmx
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_mmx
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_sse2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_sse2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_avx2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_avx2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_neon
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_neon
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_dspr2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_dspr2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_int_upsample_dspr2
  (UINT8 h_expand, UINT8 v_expand, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr, JDIMENSION output_width,
   int max_v_samp_factor);

EXTERN(void) jsimd_h2v1_upsample_altivec
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_altivec
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

/* Fancy Upsampling */
EXTERN(void) jsimd_h2v1_fancy_upsample_mmx
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_mmx
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

extern const int jconst_fancy_upsample_sse2[];
EXTERN(void) jsimd_h2v1_fancy_upsample_sse2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_sse2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

extern const int jconst_fancy_upsample_avx2[];
EXTERN(void) jsimd_h2v1_fancy_upsample_avx2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_avx2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_neon
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_neon
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h1v2_fancy_upsample_neon
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_dspr2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_dspr2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_mmi
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_mmi
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_altivec
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_altivec
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

/* Merged Upsampling */
EXTERN(void) jsimd_h2v1_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgb_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgbx_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgr_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgrx_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxbgr_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxrgb_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v2_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgb_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgbx_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgr_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgrx_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxbgr_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxrgb_merged_upsample_mmx
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

extern const int jconst_merged_upsample_sse2[];
EXTERN(void) jsimd_h2v1_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgb_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgbx_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgr_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgrx_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxbgr_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxrgb_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v2_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgb_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgbx_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgr_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgrx_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxbgr_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxrgb_merged_upsample_sse2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

extern const int jconst_merged_upsample_avx2[];
EXTERN(void) jsimd_h2v1_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgb_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgbx_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgr_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgrx_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxbgr_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxrgb_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v2_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgb_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgbx_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgr_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgrx_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxbgr_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxrgb_merged_upsample_avx2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v1_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgb_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgbx_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgr_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgrx_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxbgr_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxrgb_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v2_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgb_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgbx_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgr_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgrx_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxbgr_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxrgb_merged_upsample_neon
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v1_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v1_extrgb_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v1_extrgbx_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v1_extbgr_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v1_extbgrx_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v1_extxbgr_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v1_extxrgb_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);

EXTERN(void) jsimd_h2v2_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v2_extrgb_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v2_extrgbx_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v2_extbgr_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v2_extbgrx_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v2_extxbgr_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);
EXTERN(void) jsimd_h2v2_extxrgb_merged_upsample_dspr2
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf, JSAMPLE *range);

EXTERN(void) jsimd_h2v1_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgb_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgbx_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgr_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgrx_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxbgr_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxrgb_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v2_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgb_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgbx_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgr_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgrx_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxbgr_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxrgb_merged_upsample_mmi
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v1_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgb_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extrgbx_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgr_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extbgrx_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxbgr_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v1_extxrgb_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

EXTERN(void) jsimd_h2v2_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgb_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extrgbx_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgr_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extbgrx_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxbgr_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);
EXTERN(void) jsimd_h2v2_extxrgb_merged_upsample_altivec
  (JDIMENSION output_width, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr,
   JSAMPARRAY output_buf);

/* Sample Conversion */
EXTERN(void) jsimd_convsamp_mmx
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_sse2
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_avx2
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_neon
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_dspr2
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_altivec
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

/* Floating Point Sample Conversion */
EXTERN(void) jsimd_convsamp_float_3dnow
  (JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace);

EXTERN(void) jsimd_convsamp_float_sse
  (JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace);

EXTERN(void) jsimd_convsamp_float_sse2
  (JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace);

EXTERN(void) jsimd_convsamp_float_dspr2
  (JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace);

/* Accurate Integer Forward DCT */
EXTERN(void) jsimd_fdct_islow_mmx(DCTELEM *data);

extern const int jconst_fdct_islow_sse2[];
EXTERN(void) jsimd_fdct_islow_sse2(DCTELEM *data);

extern const int jconst_fdct_islow_avx2[];
EXTERN(void) jsimd_fdct_islow_avx2(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_neon(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_dspr2(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_mmi(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_altivec(DCTELEM *data);

/* Fast Integer Forward DCT */
EXTERN(void) jsimd_fdct_ifast_mmx(DCTELEM *data);

extern const int jconst_fdct_ifast_sse2[];
EXTERN(void) jsimd_fdct_ifast_sse2(DCTELEM *data);

EXTERN(void) jsimd_fdct_ifast_neon(DCTELEM *data);

EXTERN(void) jsimd_fdct_ifast_dspr2(DCTELEM *data);

EXTERN(void) jsimd_fdct_ifast_mmi(DCTELEM *data);

EXTERN(void) jsimd_fdct_ifast_altivec(DCTELEM *data);

/* Floating Point Forward DCT */
EXTERN(void) jsimd_fdct_float_3dnow(FAST_FLOAT *data);

extern const int jconst_fdct_float_sse[];
EXTERN(void) jsimd_fdct_float_sse(FAST_FLOAT *data);

/* Quantization */
EXTERN(void) jsimd_quantize_mmx
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_sse2
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_avx2
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_neon
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_dspr2
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_mmi
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_altivec
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

/* Floating Point Quantization */
EXTERN(void) jsimd_quantize_float_3dnow
  (JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace);

EXTERN(void) jsimd_quantize_float_sse
  (JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace);

EXTERN(void) jsimd_quantize_float_sse2
  (JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace);

EXTERN(void) jsimd_quantize_float_dspr2
  (JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace);

/* Scaled Inverse DCT */
EXTERN(void) jsimd_idct_2x2_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_4x4_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_red_sse2[];
EXTERN(void) jsimd_idct_2x2_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_4x4_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_2x2_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_4x4_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_2x2_dspr2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_4x4_dspr2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col, int *workspace);
EXTERN(void) jsimd_idct_6x6_dspr2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_12x12_pass1_dspr2
  (JCOEFPTR coef_block, void *dct_table, int *workspace);
EXTERN(void) jsimd_idct_12x12_pass2_dspr2
  (int *workspace, int *output);

/* Accurate Integer Inverse DCT */
EXTERN(void) jsimd_idct_islow_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_islow_sse2[];
EXTERN(void) jsimd_idct_islow_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_islow_avx2[];
EXTERN(void) jsimd_idct_islow_avx2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_dspr2
  (void *dct_table, JCOEFPTR coef_block, int *output_buf, JSAMPLE *output_col);

EXTERN(void) jsimd_idct_islow_mmi
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_altivec
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

/* Fast Integer Inverse DCT */
EXTERN(void) jsimd_idct_ifast_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_ifast_sse2[];
EXTERN(void) jsimd_idct_ifast_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_ifast_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_ifast_cols_dspr2
  (JCOEF *inptr, IFAST_MULT_TYPE *quantptr, DCTELEM *wsptr,
   const int *idct_coefs);
EXTERN(void) jsimd_idct_ifast_rows_dspr2
  (DCTELEM *wsptr, JSAMPARRAY output_buf, JDIMENSION output_col,
   const int *idct_coefs);

EXTERN(void) jsimd_idct_ifast_mmi
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_ifast_altivec
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

/* Floating Point Inverse DCT */
EXTERN(void) jsimd_idct_float_3dnow
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_float_sse[];
EXTERN(void) jsimd_idct_float_sse
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_float_sse2[];
EXTERN(void) jsimd_idct_float_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

/* Huffman coding */
extern const int jconst_huff_encode_one_block[];
EXTERN(JOCTET *) jsimd_huff_encode_one_block_sse2
  (void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val,
   c_derived_tbl *dctbl, c_derived_tbl *actbl);

EXTERN(JOCTET *) jsimd_huff_encode_one_block_neon
  (void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val,
   c_derived_tbl *dctbl, c_derived_tbl *actbl);

#ifndef NEON_INTRINSICS

EXTERN(JOCTET *) jsimd_huff_encode_one_block_neon_slowtbl
  (void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val,
   c_derived_tbl *dctbl, c_derived_tbl *actbl);

#endif

/* Progressive Huffman encoding */
EXTERN(void) jsimd_encode_mcu_AC_first_prepare_sse2
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *values, size_t *zerobits);

EXTERN(void) jsimd_encode_mcu_AC_first_prepare_neon
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *values, size_t *zerobits);

EXTERN(int) jsimd_encode_mcu_AC_refine_prepare_sse2
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *absvalues, size_t *bits);

EXTERN(int) jsimd_encode_mcu_AC_refine_prepare_neon
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *absvalues, size_t *bits);
