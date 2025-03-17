;
; jdmerge.asm - merged upsampling/color conversion (AVX2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2024, D. R. Commander.
; Copyright (C) 2015, Intel Corporation.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jsimdext.inc"

; --------------------------------------------------------------------------

%define SCALEBITS  16

F_0_344 equ  22554              ; FIX(0.34414)
F_0_714 equ  46802              ; FIX(0.71414)
F_1_402 equ  91881              ; FIX(1.40200)
F_1_772 equ 116130              ; FIX(1.77200)
F_0_402 equ (F_1_402 - 65536)   ; FIX(1.40200) - FIX(1)
F_0_285 equ ( 65536 - F_0_714)  ; FIX(1) - FIX(0.71414)
F_0_228 equ (131072 - F_1_772)  ; FIX(2) - FIX(1.77200)

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_merged_upsample_avx2)

EXTN(jconst_merged_upsample_avx2):

PW_F0402        times 16 dw  F_0_402
PW_MF0228       times 16 dw -F_0_228
PW_MF0344_F0285 times 8  dw -F_0_344, F_0_285
PW_ONE          times 16 dw  1
PD_ONEHALF      times 8  dd  1 << (SCALEBITS - 1)

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32

%include "jdmrgext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_RGB_RED
%define RGB_GREEN  EXT_RGB_GREEN
%define RGB_BLUE  EXT_RGB_BLUE
%define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
%define jsimd_h2v1_merged_upsample_avx2 \
  jsimd_h2v1_extrgb_merged_upsample_avx2
%define jsimd_h2v2_merged_upsample_avx2 \
  jsimd_h2v2_extrgb_merged_upsample_avx2
%include "jdmrgext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_RGBX_RED
%define RGB_GREEN  EXT_RGBX_GREEN
%define RGB_BLUE  EXT_RGBX_BLUE
%define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
%define jsimd_h2v1_merged_upsample_avx2 \
  jsimd_h2v1_extrgbx_merged_upsample_avx2
%define jsimd_h2v2_merged_upsample_avx2 \
  jsimd_h2v2_extrgbx_merged_upsample_avx2
%include "jdmrgext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_BGR_RED
%define RGB_GREEN  EXT_BGR_GREEN
%define RGB_BLUE  EXT_BGR_BLUE
%define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
%define jsimd_h2v1_merged_upsample_avx2 \
  jsimd_h2v1_extbgr_merged_upsample_avx2
%define jsimd_h2v2_merged_upsample_avx2 \
  jsimd_h2v2_extbgr_merged_upsample_avx2
%include "jdmrgext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_BGRX_RED
%define RGB_GREEN  EXT_BGRX_GREEN
%define RGB_BLUE  EXT_BGRX_BLUE
%define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
%define jsimd_h2v1_merged_upsample_avx2 \
  jsimd_h2v1_extbgrx_merged_upsample_avx2
%define jsimd_h2v2_merged_upsample_avx2 \
  jsimd_h2v2_extbgrx_merged_upsample_avx2
%include "jdmrgext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_XBGR_RED
%define RGB_GREEN  EXT_XBGR_GREEN
%define RGB_BLUE  EXT_XBGR_BLUE
%define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
%define jsimd_h2v1_merged_upsample_avx2 \
  jsimd_h2v1_extxbgr_merged_upsample_avx2
%define jsimd_h2v2_merged_upsample_avx2 \
  jsimd_h2v2_extxbgr_merged_upsample_avx2
%include "jdmrgext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_XRGB_RED
%define RGB_GREEN  EXT_XRGB_GREEN
%define RGB_BLUE  EXT_XRGB_BLUE
%define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
%define jsimd_h2v1_merged_upsample_avx2 \
  jsimd_h2v1_extxrgb_merged_upsample_avx2
%define jsimd_h2v2_merged_upsample_avx2 \
  jsimd_h2v2_extxrgb_merged_upsample_avx2
%include "jdmrgext-avx2.asm"
